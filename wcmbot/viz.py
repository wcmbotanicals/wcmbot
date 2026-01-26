"""Visualization helpers (NumPy/OpenCV only).

This module is UI-agnostic: no Gradio/Plotly imports.
It exists so the UI layer (app.py) and scripts/benchmarks can share image
composition/annotation logic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def rotate_template_preview(
    image_rgb: Optional[np.ndarray], rotation: int
) -> Optional[np.ndarray]:
    """Rotate an RGB image by multiples of 90 degrees."""
    if image_rgb is None:
        return None
    if rotation == 0:
        return image_rgb
    k = -(int(rotation) // 90)
    return np.rot90(image_rgb, k=k)


def stack_images_vertical(
    images: list[np.ndarray],
    *,
    gap: int = 8,
    background: int = 255,
    max_width: int = 0,
) -> tuple[Optional[np.ndarray], list[int]]:
    """Stack images vertically and return (stacked_image, list_of_heights)."""
    if not images:
        return None, []

    prepared: list[np.ndarray] = []
    for img in images:
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        prepared.append(img)

    if not prepared:
        return None, []

    widths = [img.shape[1] for img in prepared]
    max_w = max(widths)
    if max_width and max_w > max_width:
        scale = max_width / max_w
        resized = []
        for img in prepared:
            h, w = img.shape[:2]
            resized.append(cv2.resize(img, (int(w * scale), int(h * scale))))
        prepared = resized
        widths = [img.shape[1] for img in prepared]
        max_w = max(widths)

    heights = [img.shape[0] for img in prepared]
    total_h = sum(heights) + gap * (len(prepared) - 1)
    out = np.full((total_h, max_w, 3), background, dtype=np.uint8)
    y = 0
    for img in prepared:
        h, w = img.shape[:2]
        out[y : y + h, :w] = img
        y += h + gap
    return out, heights


def annotate_pair_image(image: np.ndarray, label: str) -> np.ndarray:
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    pad = 6
    x = pad
    y = pad + text_h
    cv2.rectangle(
        annotated,
        (x - pad, y - text_h - pad),
        (x + text_w + pad, y + pad),
        (255, 255, 255),
        -1,
    )
    cv2.putText(annotated, label, (x, y), font, font_scale, (0, 0, 0), thickness)
    return annotated


def rotate_image(
    img: np.ndarray,
    angle: float,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    """Rotate an image around its center with bounds expansion."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -float(angle), 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def overlay_piece_on_template(
    template_bgr: np.ndarray,
    piece_bgr: np.ndarray,
    piece_mask: np.ndarray,
    match: dict,
    *,
    alpha: float = 0.85,
) -> None:
    """Overlay a piece onto a template image in-place.

    Expects match to contain 'tl', 'br', and optionally 'rot'.
    """

    tlx, tly = match.get("tl", (0, 0))
    brx, bry = match.get("br", (0, 0))
    target_w = max(1, brx - tlx)
    target_h = max(1, bry - tly)
    if target_w <= 0 or target_h <= 0:
        return

    mask01 = (piece_mask > 0).astype(np.uint8) * 255
    rot = match.get("rot", 0)
    rot_piece = rotate_image(
        piece_bgr, rot, interpolation=cv2.INTER_LINEAR, border_value=(0, 0, 0)
    )
    rot_mask = rotate_image(
        mask01, rot, interpolation=cv2.INTER_NEAREST, border_value=0
    )
    piece_rs = cv2.resize(
        rot_piece, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    mask_rs = cv2.resize(
        rot_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )

    tmpl_h, tmpl_w = template_bgr.shape[:2]
    src_x0 = max(0, -tlx)
    src_y0 = max(0, -tly)
    src_x1 = target_w - max(0, brx - tmpl_w)
    src_y1 = target_h - max(0, bry - tmpl_h)
    dst_x0 = max(0, tlx)
    dst_y0 = max(0, tly)
    dst_x1 = min(tmpl_w, brx)
    dst_y1 = min(tmpl_h, bry)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return

    piece_patch = piece_rs[src_y0:src_y1, src_x0:src_x1]
    mask_patch = mask_rs[src_y0:src_y1, src_x0:src_x1]
    template_patch = template_bgr[dst_y0:dst_y1, dst_x0:dst_x1]
    if piece_patch.shape[:2] != template_patch.shape[:2]:
        return

    mask_norm = (mask_patch > 127).astype(np.float32)
    if mask_norm.ndim == 2:
        mask_norm = mask_norm[:, :, np.newaxis]
    mask_norm = np.clip(mask_norm * float(alpha), 0.0, 1.0)
    blended_patch = (template_patch * (1 - mask_norm) + piece_patch * mask_norm).astype(
        np.uint8
    )
    template_bgr[dst_y0:dst_y1, dst_x0:dst_x1] = blended_patch


def build_multipiece_overview(
    grid_bgr: np.ndarray,
    regions: list[dict],
    colors_bgr: list[Tuple[int, int, int]],
    *,
    max_dim: int = 1400,
) -> np.ndarray:
    """Render a diagnostic overview showing detected piece contours + indices."""

    h, w = grid_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        preview = cv2.resize(
            grid_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        preview = grid_bgr.copy()

    for idx, region in enumerate(regions):
        color = colors_bgr[idx % len(colors_bgr)]
        cnt = region["contour"]
        if scale < 1.0:
            cnt = (cnt * scale).astype(np.int32)
        cv2.drawContours(preview, [cnt], -1, color, 2)

        x, y, bw, bh = region["bbox"]
        if scale < 1.0:
            x = int(x * scale)
            y = int(y * scale)
            bw = max(1, int(bw * scale))
            bh = max(1, int(bh * scale))
        cx = x + bw // 2
        cy = y + bh // 2

        label = str(idx + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(1.4, min(bw, bh) / 60)
        thickness = max(2, int(font_scale * 1.5))
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        label_x = int(cx - text_w / 2)
        label_y = int(cy + text_h / 2)
        label_x = max(x + 4, min(label_x, x + bw - text_w - 4))
        label_y = max(y + text_h + 4, min(label_y, y + bh - 4))

        pad = 6
        box_tl = (max(0, label_x - pad), max(0, label_y - text_h - pad))
        box_br = (
            min(preview.shape[1] - 1, label_x + text_w + pad),
            min(preview.shape[0] - 1, label_y + pad),
        )
        cv2.rectangle(preview, box_tl, box_br, (0, 0, 0), -1)
        cv2.putText(
            preview,
            label,
            (label_x, label_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
