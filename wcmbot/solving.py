"""Non-UI puzzle-solving helpers.

The goal of this module is to keep the core "solve" pipeline reusable from:
- the Gradio UI (app.py)
- scripts / benchmarks

It deliberately avoids Gradio/Plotly imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from wcmbot.matcher import (
    MatcherConfig,
    MatchPayload,
    build_matcher_config,
    find_piece_in_template_bgr,
)
from wcmbot.multipiece import find_multipiece_regions
from wcmbot.template_settings import TemplateSpec


def build_matcher_config_for_template(
    template_spec: TemplateSpec,
    extra_overrides: Optional[dict[str, object]] = None,
) -> MatcherConfig:
    """Create a MatcherConfig from a TemplateSpec.

    Centralizes how template-specific overrides are applied.

    Args:
        template_spec: Template configuration.
        extra_overrides: Additional overrides to apply (e.g., mask_mode from UI).
    """

    overrides: dict[str, object] = {
        "rows": template_spec.rows,
        "cols": template_spec.cols,
        "crop_x": template_spec.crop_x,
        "crop_y": template_spec.crop_y,
        **(template_spec.matcher_overrides or {}),
        **(extra_overrides or {}),
    }
    return build_matcher_config(overrides)


def solve_piece_payload_from_bgr(
    piece_bgr: np.ndarray,
    template_spec: TemplateSpec,
    *,
    auto_align: bool,
    template_rotation: Optional[int] = None,
    matcher_config: Optional[MatcherConfig] = None,
) -> MatchPayload:
    """Solve a single piece from an in-memory BGR image."""

    config = matcher_config or build_matcher_config_for_template(template_spec)
    return find_piece_in_template_bgr(
        piece_bgr,
        str(template_spec.template_path),
        knobs_x=None,
        knobs_y=None,
        auto_align=bool(auto_align),
        infer_knobs=True,
        template_rotation=template_rotation,
        matcher_config=config,
    )


@dataclass(frozen=True)
class MultipieceSolveItem:
    index: int
    region: dict
    piece_bgr: np.ndarray
    payload: Optional[MatchPayload]


def iter_multipiece_payloads_from_bgr(
    grid_bgr: np.ndarray,
    template_spec: TemplateSpec,
    *,
    auto_align: bool,
    template_rotation: Optional[int] = None,
    matcher_config: Optional[MatcherConfig] = None,
    regions: Optional[list[dict]] = None,
    min_area_frac: float = 0.002,
    pad_frac: float = 0.06,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
) -> Iterator[MultipieceSolveItem]:
    """Yield per-piece payloads from a multi-piece image.

    This is a low-level helper intended for the UI and benchmarks.

    Notes:
    - Returns crops as BGR numpy arrays.
    - Uses the same segmentation ordering as the UI.
    - If regions are provided, segmentation is skipped and those region dicts are used.
    """

    config = matcher_config or build_matcher_config_for_template(template_spec)

    if regions is None:
        typed_regions, _mask01 = find_multipiece_regions(
            grid_bgr,
            config,
            min_area_frac=min_area_frac,
            template_bgr=template_bgr,
            template_mask=template_mask,
        )
        region_dicts = [
            {"bbox": r.bbox, "contour": r.contour, "area": r.area}
            for r in typed_regions
        ]
    else:
        region_dicts = list(regions)

    for idx, region in enumerate(region_dicts):
        x, y, w, h = region["bbox"]
        pad = max(4, int(min(w, h) * float(pad_frac)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(grid_bgr.shape[1], x + w + pad)
        y1 = min(grid_bgr.shape[0], y + h + pad)

        # Use pre-processed piece image if available (white background from AI mode)
        piece_bgr_preprocessed = region.get("piece_bgr")
        if piece_bgr_preprocessed is not None:
            # Use BGR with white background - standard template masking will work
            crop_img = piece_bgr_preprocessed
        else:
            crop_img = grid_bgr[y0:y1, x0:x1].copy()

        payload = None
        try:
            payload = solve_piece_payload_from_bgr(
                crop_img,
                template_spec,
                auto_align=auto_align,
                template_rotation=template_rotation,
                matcher_config=config,
            )
        except Exception:  # pylint: disable=broad-except
            payload = None
        yield MultipieceSolveItem(
            index=idx,
            region=region,
            piece_bgr=crop_img,
            payload=payload,
        )
