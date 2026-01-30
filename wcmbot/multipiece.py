"""Utilities for multi-piece segmentation.

This module intentionally avoids Gradio/Plotly dependencies so it can be reused
by the UI layer (app.py) and by benchmarks/scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from wcmbot.matcher import (
    compute_chrominance_mask,
    compute_gradient_mask,
    compute_piece_mask,
)


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


def compute_multipiece_mask(
    image_bgr: np.ndarray,
    matcher_config,
    *,
    compute_piece_mask_fn: Callable = compute_piece_mask,
    invert_background: bool = True,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
    use_chrominance_fallback: bool = True,
    use_gradient_enhancement: bool = True,
) -> np.ndarray:
    """Compute a binary mask for separating multiple pieces.

    If the mask selects mostly background, it is optionally inverted. Template
    imagery can be supplied to support template-aware segmentation.

    When use_chrominance_fallback is True (default), the function will use
    chrominance-based masking to fill internal holes in pieces that may be
    caused by piece colors matching the background HSV ranges.

    When use_gradient_enhancement is True (default), the function will use
    gradient-based edge detection to improve piece boundaries. This is only
    applied when significant hue overlap (>5%) is detected between piece
    foreground and background colors, which indicates that color-based
    segmentation is unreliable for the image. When piece/background colors
    are distinct (like blue background with green pieces), gradient
    enhancement is skipped to preserve the accurate HSV-based segmentation.
    """
    mask01 = _compute_piece_mask_keep_all(
        compute_piece_mask_fn,
        image_bgr,
        matcher_config,
        template_bgr=template_bgr,
        template_mask=template_mask,
    )

    if invert_background and mask01.sum() > 0.5 * mask01.size:
        mask01 = (mask01 == 0).astype(np.uint8)

    # Enhance each detected piece region using gradient-based edge detection
    # This is applied per-piece because gradient detection works best on
    # individual piece crops, not the full multi-piece image
    # Only apply if pieces have significant color overlap with background
    if use_gradient_enhancement and mask01.sum() > 0:
        mask255 = mask01 * 255
        contours, _ = cv2.findContours(
            mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            image_area = image_bgr.shape[0] * image_bgr.shape[1]
            min_contour_area = max(100, int(image_area * 0.0005))

            # Check if pieces have significant color overlap with background
            # by sampling a few pieces
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            sample_contours = [
                c for c in contours if cv2.contourArea(c) >= min_contour_area
            ][:5]
            total_overlap = 0.0

            for cnt in sample_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                pad = max(10, int(min(w, h) * 0.15))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = (
                    min(image_bgr.shape[1], x + w + pad),
                    min(image_bgr.shape[0], y + h + pad),
                )

                piece_hsv = hsv[y0:y1, x0:x1]
                piece_mask = mask01[y0:y1, x0:x1]

                bg_mask = (piece_mask == 0).astype(np.uint8)
                fg_mask = (piece_mask > 0).astype(np.uint8)

                if bg_mask.sum() >= 100 and fg_mask.sum() >= 100:
                    bg_hue = piece_hsv[bg_mask == 1, 0]
                    fg_hue = piece_hsv[fg_mask == 1, 0]
                    bg_h_low, bg_h_high = np.percentile(bg_hue, [10, 90])

                    # Check for hue overlap, accounting for circular hue wraparound
                    # Hue wraps at 180 in OpenCV (0-179), so red can be near 0 or 179
                    if bg_h_high - bg_h_low > 90:
                        # Background hue spans more than half the range, likely wraparound
                        # In this case, treat as non-overlapping (background is mixed)
                        fg_in_bg = 0
                    else:
                        fg_in_bg = ((fg_hue >= bg_h_low) & (fg_hue <= bg_h_high)).sum()

                    # fg_hue is guaranteed non-empty by fg_mask.sum() >= 100 check
                    overlap = fg_in_bg / len(fg_hue)
                    total_overlap += overlap

            avg_overlap = total_overlap / len(sample_contours) if sample_contours else 0

            # Apply gradient enhancement if average hue overlap exceeds threshold.
            # The 5% threshold was determined empirically:
            # - many_pieces.jpg (blue bg): ~0.1% overlap - no enhancement needed
            # - difficult_multipiece.jpg (brown bg): ~12% overlap - enhancement helps
            # Values above 5% indicate significant color overlap where HSV fails.
            hue_overlap_threshold = 0.05
            if avg_overlap > hue_overlap_threshold:
                for cnt in contours:
                    if cv2.contourArea(cnt) >= min_contour_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        pad = max(5, int(min(w, h) * 0.1))
                        x0 = max(0, x - pad)
                        y0 = max(0, y - pad)
                        x1 = min(image_bgr.shape[1], x + w + pad)
                        y1 = min(image_bgr.shape[0], y + h + pad)

                        # Extract piece region and compute gradient mask
                        piece_region = image_bgr[y0:y1, x0:x1]
                        piece_gradient = compute_gradient_mask(piece_region)

                        # Create a convex hull mask to constrain gradient updates
                        # This prevents the gradient mask from merging adjacent pieces
                        hull = cv2.convexHull(cnt)
                        hull_local = hull - [x0, y0]
                        hull_constraint = np.zeros_like(piece_gradient)
                        cv2.drawContours(hull_constraint, [hull_local], -1, 1, -1)

                        # Dilate hull slightly to allow for edge recovery
                        dilate_kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (5, 5)
                        )
                        hull_constraint = cv2.dilate(
                            hull_constraint, dilate_kernel, iterations=2
                        )

                        # Apply gradient only within hull constraint
                        piece_gradient = piece_gradient & hull_constraint

                        # Update the mask in this region
                        current_region = mask01[y0:y1, x0:x1]
                        mask01[y0:y1, x0:x1] = np.maximum(
                            current_region, piece_gradient
                        )

    # Use chrominance-based segmentation to fill remaining holes
    if use_chrominance_fallback and mask01.sum() > 0:
        kernel_size = getattr(matcher_config, "mask_kernel_size", 7)
        open_iters = getattr(matcher_config, "mask_open_iters", 2)
        close_iters = getattr(matcher_config, "mask_close_iters", 2)
        chroma_mask = compute_chrominance_mask(
            image_bgr,
            kernel_size=kernel_size,
            open_iters=open_iters,
            close_iters=close_iters,
        )

        # Find the convex hull of each connected component to define piece boundaries
        # Then fill holes within those boundaries using chrominance mask
        mask255 = mask01 * 255
        contours, _ = cv2.findContours(
            mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Create a mask of convex hulls for each piece
            # Use min_contour_area proportional to image size to filter noise
            image_area = image_bgr.shape[0] * image_bgr.shape[1]
            min_contour_area = max(100, int(image_area * 0.00002))
            hull_mask = np.zeros_like(mask01)
            for cnt in contours:
                if cv2.contourArea(cnt) >= min_contour_area:
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(hull_mask, [hull], -1, 1, -1)

            # Fill holes: within the convex hull, add chrominance foreground
            enhanced = np.where(hull_mask > 0, (mask01 | chroma_mask), mask01)
            mask01 = enhanced.astype(np.uint8)

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
        {"bbox": r.bbox, "contour": r.contour, "area": r.area} for r in regions
    ]
    return region_dicts, mask01
