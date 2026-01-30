"""Utilities for multi-piece segmentation.

This module intentionally avoids Gradio/Plotly dependencies so it can be reused
by the UI layer (app.py) and by benchmarks/scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from wcmbot.matcher import compute_piece_mask, remove_background_ai


@dataclass
class MultipieceRegion:
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    area: float
    # If set, the piece image with white background (BGR) for standard masking
    piece_bgr: Optional[np.ndarray] = None


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

    When mask_mode="ai", uses AI background removal once for the entire image
    and returns regions with piece_bgra pre-removed. This is more efficient
    than running AI on each individual piece.

    Prefer using find_multipiece_regions() for typed access.
    """
    # Check if AI mode should be used
    # Use multipiece_mask_mode if set, otherwise use mask_mode
    effective_mask_mode = None
    if (
        hasattr(matcher_config, "multipiece_mask_mode")
        and matcher_config.multipiece_mask_mode
    ):
        effective_mask_mode = matcher_config.multipiece_mask_mode
    elif hasattr(matcher_config, "mask_mode"):
        effective_mask_mode = matcher_config.mask_mode

    if effective_mask_mode == "ai":
        # Use AI background removal for efficient one-time processing
        regions, mask01 = find_multipiece_regions_ai(
            image_bgr,
            matcher_config,
            min_area_frac=min_area_frac,
        )
    else:
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
            "piece_bgr": r.piece_bgr,  # Include white-background piece if available
        }
        for r in regions
    ]
    return region_dicts, mask01


def find_multipiece_regions_ai(
    image_bgr: np.ndarray,
    matcher_config,
    *,
    min_area_frac: float = 0.002,
) -> tuple[list[MultipieceRegion], np.ndarray]:
    """Find piece regions using AI background removal (rembg).

    This function removes the background once using the AI model, replaces
    the background with white, then extracts individual piece regions.
    Each region includes a piece_bgr attribute with white background.

    The white background works well with template default HSV masking since
    white doesn't match the green/dark HSV ranges used for piece detection.

    Args:
        image_bgr: BGR image containing multiple pieces.
        matcher_config: Configuration (multipiece_mask_mode not used here).
        min_area_frac: Minimum contour area as fraction of image area.

    Returns:
        (regions, mask01) where regions have piece_bgr with white background.
    """
    # Remove background from entire image using AI
    image_bgra = remove_background_ai(image_bgr)

    # Extract alpha channel as mask
    if image_bgra.shape[2] == 4:
        alpha = image_bgra[:, :, 3]
    else:
        # Fallback if no alpha
        alpha = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255

    # Threshold alpha to binary mask
    _, mask255 = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    mask01 = (mask255 > 0).astype(np.uint8)

    # Replace background with white: where alpha < 128, set to white
    image_white_bg = image_bgra[:, :, :3].copy()  # BGR channels
    background_mask = alpha < 128
    image_white_bg[background_mask] = [255, 255, 255]  # White background

    # Find contours
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

        # Extract piece with white background (BGR for standard masking)
        piece_bgr = image_white_bg[y : y + h, x : x + w].copy()

        regions.append(
            MultipieceRegion(
                bbox=(x, y, w, h), contour=cnt, area=area, piece_bgr=piece_bgr
            )
        )

    if not regions:
        return [], mask01

    # Sort regions by row, then left-to-right within each row
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
