"""Utilities for multi-piece segmentation.

This module intentionally avoids Gradio/Plotly dependencies so it can be reused
by the UI layer (app.py) and by benchmarks/scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from wcmbot.matcher import compute_piece_mask


@dataclass
class MultipieceRegion:
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    area: float


def _compute_piece_mask_keep_all(
    compute_piece_mask_fn: Callable,
    image_bgr: np.ndarray,
    matcher_config,
    *,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Call a compute_piece_mask-like function with best-effort compatibility."""
    try:
        return compute_piece_mask_fn(
            image_bgr,
            matcher_config,
            keep_largest_component=False,
            template_bgr=template_bgr,
            template_mask=template_mask,
        )
    except TypeError:
        if template_bgr is not None:
            try:
                return compute_piece_mask_fn(
                    image_bgr,
                    matcher_config,
                    keep_largest_component=False,
                    template_bgr=template_bgr,
                )
            except TypeError:
                pass
        try:
            return compute_piece_mask_fn(
                image_bgr, matcher_config, keep_largest_component=False
            )
        except TypeError:
            # Older signatures may not accept keep_largest_component.
            return compute_piece_mask_fn(image_bgr, matcher_config)


def _get_multipiece_config(matcher_config):
    """Get config for multipiece mask computation.

    If multipiece_mask_mode is set, returns a modified config that uses that
    mode for the initial splitting. Otherwise returns the original config.
    """
    if (
        hasattr(matcher_config, "multipiece_mask_mode")
        and matcher_config.multipiece_mask_mode
    ):
        # Create a new config with the multipiece mask mode using replace()
        return replace(matcher_config, mask_mode=matcher_config.multipiece_mask_mode)
    return matcher_config


def compute_multipiece_mask(
    image_bgr: np.ndarray,
    matcher_config,
    *,
    compute_piece_mask_fn: Callable = compute_piece_mask,
    invert_background: bool = True,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a binary mask for separating multiple pieces.

    If the mask selects mostly background, it is optionally inverted. Template
    imagery can be supplied to support template-aware segmentation.

    Uses multipiece_mask_mode if set in config, otherwise uses mask_mode.
    """
    # Use multipiece-specific mask mode if configured
    multipiece_config = _get_multipiece_config(matcher_config)

    mask01 = _compute_piece_mask_keep_all(
        compute_piece_mask_fn,
        image_bgr,
        multipiece_config,
        template_bgr=template_bgr,
        template_mask=template_mask,
    )

    if invert_background and mask01.sum() > 0.5 * mask01.size:
        mask01 = (mask01 == 0).astype(np.uint8)

    return mask01


def find_multipiece_regions(
    image_bgr: np.ndarray,
    matcher_config,
    *,
    compute_piece_mask_fn: Callable = compute_piece_mask,
    min_area_frac: float = 0.002,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
) -> tuple[list[MultipieceRegion], np.ndarray]:
    """Find candidate piece regions via contour detection.

    Returns:
      (regions, mask01)

    The region ordering is stable and intended to match the UI expectations:
    rows are inferred by grouping contour centers by vertical proximity, then
    sorting left-to-right within each row.
    """

    mask01 = compute_multipiece_mask(
        image_bgr,
        matcher_config,
        compute_piece_mask_fn=compute_piece_mask_fn,
        template_bgr=template_bgr,
        template_mask=template_mask,
    )
    mask255 = (mask01 > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask01

    image_area = int(image_bgr.shape[0] * image_bgr.shape[1])
    min_area = max(400, int(image_area * float(min_area_frac)))

    regions: list[MultipieceRegion] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append(MultipieceRegion(bbox=(x, y, w, h), contour=cnt, area=area))

    if not regions:
        return [], mask01

    heights = np.array([region.bbox[3] for region in regions], dtype=np.float32)
    row_thresh = float(np.median(heights) * 0.6)

    centers = [
        (
            region.bbox[0] + region.bbox[2] / 2.0,
            region.bbox[1] + region.bbox[3] / 2.0,
            idx,
        )
        for idx, region in enumerate(regions)
    ]
    centers.sort(key=lambda t: t[1])

    rows: list[dict] = []
    for cx, cy, idx in centers:
        if not rows:
            rows.append({"cy": cy, "items": [(cx, cy, idx)]})
            continue
        if abs(cy - rows[-1]["cy"]) <= row_thresh:
            rows[-1]["items"].append((cx, cy, idx))
            rows[-1]["cy"] = float(np.mean([item[1] for item in rows[-1]["items"]]))
        else:
            rows.append({"cy": cy, "items": [(cx, cy, idx)]})

    ordered: list[MultipieceRegion] = []
    for row in rows:
        row["items"].sort(key=lambda t: t[0])
        ordered.extend([regions[idx] for _, _, idx in row["items"]])

    return ordered, mask01


def find_multipiece_region_dicts(
    image_bgr: np.ndarray,
    matcher_config,
    *,
    compute_piece_mask_fn: Callable = compute_piece_mask,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
    min_area_frac: float = 0.002,
) -> tuple[list[dict], np.ndarray]:
    """Compatibility wrapper returning historical region dicts.

    The original UI/test code represented regions as dictionaries:
      {"bbox": (x, y, w, h), "contour": contour, "area": area}

    Prefer using find_multipiece_regions() for typed access.
    """
    regions, mask01 = find_multipiece_regions(
        image_bgr,
        matcher_config,
        compute_piece_mask_fn=compute_piece_mask_fn,
        template_bgr=template_bgr,
        template_mask=template_mask,
        min_area_frac=min_area_frac,
    )

    region_dicts = [
        {
            "bbox": r.bbox,
            "contour": r.contour,
            "area": r.area,
        }
        for r in regions
    ]
    return region_dicts, mask01
