"""Gradio interface for the jigsaw puzzle solver"""

import base64
import os
import random
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import cv2
import gradio as gr
import numpy as np
import plotly.express as px
from PIL import Image

from wcmbot import __version__
from wcmbot.matcher import (
    build_matcher_config,
    compute_piece_mask,
    find_piece_in_template,
    format_match_summary,
    preload_template_cache,
    render_primary_views,
)
from wcmbot.template_settings import load_template_registry

BASE_DIR = Path(__file__).resolve().parent
MUSPAN_LOGO_PATH = BASE_DIR / "media" / "muspan_logo.png"

VIEW_KEYS = [
    "template_color",
    "template_bin",
    "piece_crop",
    "piece_mask",
    "piece_bin",
    "resized_piece",
    "grid_overview",
    "zoom_focus",
    "zoom_template",
    "zoom_piece",
    "zoom_pair",
]

VIEW_LABELS = {
    "template_color": "Template (color)",
    "template_bin": "Template bin",
    "piece_crop": "Piece (cropped)",
    "piece_mask": "Piece mask",
    "piece_bin": "Piece binary pattern",
    "resized_piece": "Resized piece preview",
    "grid_overview": "Multipiece overview",
    "zoom_focus": "Best match (zoomed)",
    "zoom_template": "Best match (template view)",
    "zoom_piece": "Piece (masked + rotated)",
    "zoom_pair": "Best match (side-by-side)",
}

TEMPLATE_ROTATION_OPTIONS = [0, 90, 180, 270]

GRID_COLORS_BGR = [
    (0, 0, 255),  # Red
    (255, 0, 0),  # Blue
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Pink
    (255, 128, 0),  # Light blue
    (128, 255, 0),  # Green-cyan
]

MULTIPIECE_DEFAULT = True
MAX_DYNAMIC_BUTTONS = 50


def make_zoomable_plot(image: Optional[np.ndarray]):
    """Create a Plotly figure with zoom/pan for a numpy RGB image."""
    if image is None:
        base = np.zeros((10, 10, 3), dtype=np.uint8)
    else:
        base = image
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)
    fig = px.imshow(base)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def _stack_images_vertical(
    images: list[np.ndarray],
    gap: int = 8,
    background: int = 255,
    max_width: int = 0,
) -> tuple[Optional[np.ndarray], list[int]]:
    """Stack images vertically and return (stacked_image, list_of_individual_heights)."""
    if not images:
        return None, []

    prepared = []
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


def _annotate_pair_image(image: np.ndarray, label: str) -> np.ndarray:
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


def _annotate_grid_image(
    grid_img: Image.Image, rows: int, cols: int, colors_bgr
) -> np.ndarray:
    """Overlay colored index numbers on a grid image."""
    grid_rgb = np.array(grid_img.convert("RGB"))
    grid_bgr = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR)
    height, width = grid_bgr.shape[:2]
    cell_w = width // cols
    cell_h = height // rows
    scale = max(1.2, min(cell_w, cell_h) / 90)
    thickness = max(3, int(scale * 3))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        label = str(idx + 1)
        text_size, _ = cv2.getTextSize(label, font, scale, thickness)
        text_w, text_h = text_size
        center_x = c * cell_w + cell_w // 2
        center_y = r * cell_h + cell_h // 2
        x = center_x - text_w // 2
        y = center_y + text_h // 2
        x = max(c * cell_w + 4, min(x, (c + 1) * cell_w - text_w - 4))
        y = max(r * cell_h + text_h + 4, min(y, (r + 1) * cell_h - 4))
        color = colors_bgr[idx % len(colors_bgr)]
        cv2.putText(
            grid_bgr,
            label,
            (x, y),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            grid_bgr,
            label,
            (x, y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)


def _rotate_image(
    img: np.ndarray,
    angle: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
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


def _overlay_piece_on_template(
    template_bgr: np.ndarray,
    piece_bgr: np.ndarray,
    piece_mask: np.ndarray,
    match: dict,
    alpha: float = 0.85,
) -> None:
    tlx, tly = match.get("tl", (0, 0))
    brx, bry = match.get("br", (0, 0))
    target_w = max(1, brx - tlx)
    target_h = max(1, bry - tly)
    if target_w <= 0 or target_h <= 0:
        return

    mask01 = (piece_mask > 0).astype(np.uint8) * 255
    rot = match.get("rot", 0)
    rot_piece = _rotate_image(
        piece_bgr, rot, interpolation=cv2.INTER_LINEAR, border_value=(0, 0, 0)
    )
    rot_mask = _rotate_image(
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
    mask_norm = np.clip(mask_norm * alpha, 0.0, 1.0)
    blended_patch = (template_patch * (1 - mask_norm) + piece_patch * mask_norm).astype(
        np.uint8
    )
    template_bgr[dst_y0:dst_y1, dst_x0:dst_x1] = blended_patch


def _rotate_template_preview(
    image: Optional[np.ndarray], rotation: int
) -> Optional[np.ndarray]:
    if image is None:
        return None
    if rotation == 0:
        return image
    k = -(rotation // 90)
    return np.rot90(image, k=k)


def _compute_multipiece_mask(piece_bgr: np.ndarray, matcher_config) -> np.ndarray:
    """Compute binary mask for multipiece segmentation with optional background inversion."""
    mask01 = compute_piece_mask(piece_bgr, matcher_config, keep_largest_component=False)

    # Invert if segmentation captured mostly background.
    if mask01.sum() > 0.5 * mask01.size:
        mask01 = (mask01 == 0).astype(np.uint8)

    return mask01


def _find_multipiece_regions(
    image_bgr: np.ndarray, matcher_config, min_area_frac: float = 0.002
):
    mask01 = _compute_multipiece_mask(image_bgr, matcher_config)
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask01

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = max(400, int(image_area * min_area_frac))
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append({"bbox": (x, y, w, h), "contour": cnt, "area": area})

    if not regions:
        return [], mask01

    heights = np.array([item["bbox"][3] for item in regions], dtype=np.float32)
    row_thresh = float(np.median(heights) * 0.6)
    centers = [
        (
            item["bbox"][0] + item["bbox"][2] / 2.0,
            item["bbox"][1] + item["bbox"][3] / 2.0,
            idx,
        )
        for idx, item in enumerate(regions)
    ]
    centers.sort(key=lambda t: t[1])
    rows = []
    for cx, cy, idx in centers:
        if not rows:
            rows.append({"cy": cy, "items": [(cx, cy, idx)]})
            continue
        if abs(cy - rows[-1]["cy"]) <= row_thresh:
            rows[-1]["items"].append((cx, cy, idx))
            rows[-1]["cy"] = float(np.mean([item[1] for item in rows[-1]["items"]]))
        else:
            rows.append({"cy": cy, "items": [(cx, cy, idx)]})

    ordered = []
    for row in rows:
        row["items"].sort(key=lambda t: t[0])
        ordered.extend([regions[idx] for _, _, idx in row["items"]])
    regions = ordered
    return regions, mask01


def _build_multipiece_overview(
    grid_bgr: np.ndarray,
    regions,
    colors,
    max_dim: int = 1400,
) -> np.ndarray:
    h, w = grid_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        preview = cv2.resize(
            grid_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        preview = grid_bgr.copy()

    for idx, region in enumerate(regions):
        color = colors[idx % len(colors)]
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


def get_random_ad():
    """Get a random advertisement banner HTML"""
    logo_html = ""
    if MUSPAN_LOGO_DATA_URI:
        logo_html = (
            f'<img src="{MUSPAN_LOGO_DATA_URI}" alt="Muspan" '
            'style="height: 80px; width: auto; max-width: 200px; object-fit: contain;">'
        )
    ads = [
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px 20px; 
                    border-radius: 10px; 
                    border: 3px solid #5a67d8; 
                    margin: 15px 0; 
                    text-align: center; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    position: relative;">
            <span style="position: absolute; top: 5px; right: 10px; color: rgba(255, 255, 255, 0.7); font-size: 10px; font-weight: bold;">Ad</span>
            <div style="display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap;">
                {logo_html}
                <p style="color: white; font-size: 16px; margin: 0; font-weight: 500; flex: 1; min-width: 300px;">
                    🔧 Solve YOUR mathematical problems with <strong>Muspan</strong> - the ultimate toolbox for spatial analysis! 
                    Visit <a href="https://www.muspan.co.uk/" target="_blank" style="color: #ffd700; text-decoration: underline;">www.muspan.co.uk</a>
                </p>
            </div>
        </div>
        """,
        """
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 15px 20px; 
                    border-radius: 10px; 
                    border: 3px dashed #e91e63; 
                    margin: 15px 0; 
                    text-align: center; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    position: relative;">
            <span style="position: absolute; top: 5px; right: 10px; color: rgba(255, 255, 255, 0.7); font-size: 10px; font-weight: bold;">Ad</span>
            <p style="color: white; font-size: 16px; margin: 0; font-weight: 500;">
                🧬 Mathematical biologists HATE him! One simple trick to invoke Schnakenberg kinetics. 
                <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank" style="color: #ffeb3b; text-decoration: underline;">Click here to learn more...</a>
            </p>
        </div>
        """,
        """
        <div style="background: linear-gradient(135deg, #ffefba 0%, #ffffff 100%); 
                    padding: 15px 20px; 
                    border-radius: 10px; 
                    border: 3px solid #f4b41a; 
                    margin: 15px 0; 
                    text-align: center; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    position: relative;">
            <span style="position: absolute; top: 5px; right: 10px; color: rgba(0, 0, 0, 0.5); font-size: 10px; font-weight: bold;">Ad</span>
            <p style="color: #2d2d2d; font-size: 16px; margin: 0; font-weight: 600;">
                Did you use the programming language Julia between 2018 and 2023? 
                Then you could be entitled to thousands of pounds of compensation. 
                <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank" style="color: #d35400; text-decoration: underline;">Click here to find out more</a>
            </p>
        </div>
        """,
    ]
    return random.choice(ads)


def _build_muspan_logo_data_uri() -> str:
    if not MUSPAN_LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(MUSPAN_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


MUSPAN_LOGO_DATA_URI = _build_muspan_logo_data_uri()


def check_template_exists(template_path: Path):
    """Check that the required template file exists."""
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file not found at {template_path}. "
            "Please ensure the puzzle template image exists before running the app."
        )


def get_template_image(
    template_path: Path, *, crop_x: int, crop_y: int
) -> Optional[np.ndarray]:
    """Get the template image."""
    if template_path.exists():
        img = np.array(Image.open(template_path))
        crop_x = int(crop_x)
        crop_y = int(crop_y)
        if crop_x < 0 or crop_y < 0:
            raise ValueError("crop_x and crop_y must be non-negative.")
        if crop_x or crop_y:
            h, w = img.shape[:2]
            if crop_x * 2 >= w or crop_y * 2 >= h:
                raise ValueError(
                    f"Template crop too large for {w}x{h}: "
                    f"crop_x={crop_x} crop_y={crop_y}"
                )
            img = img[crop_y : h - crop_y, crop_x : w - crop_x]
        return img
    return None


def _views_to_outputs(
    views: Dict[str, Optional[np.ndarray]],
    location: str,
    summary: str,
    state,
    idx: int,
    batch_state,
):
    ordered = [views.get(key) for key in VIEW_KEYS]
    # Build visibility updates for button containers
    button_visibility = _build_button_visibility(batch_state)
    return (*ordered, location, summary, state, idx, batch_state, *button_visibility)


def _build_button_visibility(batch_state):
    """Return visibility updates for button containers; spacers stay hidden."""
    if not batch_state or "total" not in batch_state:
        button_updates = [gr.update(visible=False)] * MAX_DYNAMIC_BUTTONS
        spacer_updates = [gr.update(visible=False, value="")] * (
            MAX_DYNAMIC_BUTTONS + 1
        )
        return button_updates + spacer_updates

    total = int(batch_state.get("total", 0))
    button_updates = [
        gr.update(visible=(i < total)) for i in range(MAX_DYNAMIC_BUTTONS)
    ]
    spacer_updates = [gr.update(visible=False, value="")] * (MAX_DYNAMIC_BUTTONS + 1)
    return button_updates + spacer_updates


def _blank_outputs(message: str, template_id: str, template_rotation: int):
    blank_views = {key: None for key in VIEW_KEYS}
    rotated_template = _rotate_template_preview(
        TEMPLATE_IMAGES.get(template_id), template_rotation
    )
    blank_views["template_color"] = rotated_template
    blank_views["zoom_template"] = make_zoomable_plot(rotated_template)
    blank_views["zoom_pair"] = None
    location = "Run the matcher to infer a row and column."
    return _views_to_outputs(blank_views, location, message, None, 0, None)


def _format_match_location(payload, idx: int) -> str:
    if payload is None or not getattr(payload, "matches", None):
        return "Run the matcher to infer a row and column."
    match = payload.matches[idx]
    return f"**Row:** {match['row']}  **Col:** {match['col']}"


def _render_match_payload(payload, idx: int, batch_state=None):
    views = render_primary_views(payload, idx)
    views["zoom_template"] = make_zoomable_plot(views.get("zoom_template"))
    location = _format_match_location(payload, idx)
    summary = format_match_summary(payload, idx)
    return _views_to_outputs(views, location, summary, payload, idx, batch_state)


def _clamp_match_index(payload, idx: int) -> int:
    if payload is None or not getattr(payload, "matches", None):
        return 0
    return max(0, min(idx, len(payload.matches) - 1))


def _format_multipiece_table(piece_states, total: int) -> str:
    coord_lines = [
        "| Piece | Row | Col |",
        "|-------|-----|-----|",
    ]
    colors_rgb = [(b, g, r) for (b, g, r) in GRID_COLORS_BGR]
    for idx in range(total):
        piece_color = colors_rgb[idx % len(colors_rgb)]
        piece_num_html = (
            f'<span style="color: rgb({piece_color[0]}, '
            f'{piece_color[1]}, {piece_color[2]})">**{idx + 1}**</span>'
        )
        state = piece_states[idx] if idx < len(piece_states) else None
        if not state:
            coord_lines.append(f"| {piece_num_html} | - | - |")
            continue
        if state.get("error"):
            coord_lines.append(f"| {piece_num_html} | Error | Error |")
            continue
        payload = state.get("payload")
        match_index = _clamp_match_index(payload, state.get("match_index", 0))
        if not payload or not getattr(payload, "matches", None):
            coord_lines.append(f"| {piece_num_html} | - | - |")
            continue
        match = payload.matches[match_index]
        coord_lines.append(
            f"| {piece_num_html} | {match.get('row', '-')} | {match.get('col', '-')} |"
        )
    return "\n".join(coord_lines)


def _build_multipiece_views_from_state(batch_state, last_result=None):
    if not batch_state:
        return {key: None for key in VIEW_KEYS}

    piece_states = batch_state.get("piece_states") or []
    grid_overview = batch_state.get("grid_overview")
    template_rgb = batch_state.get("template_rgb")
    template_id = batch_state.get("template_id")
    rotation = batch_state.get("rotation", 0)

    latest_zoom = None
    latest_pair = None
    latest_piece = None
    if last_result is not None:
        outputs = list(last_result)
        zoom_focus_idx = VIEW_KEYS.index("zoom_focus")
        latest_zoom = outputs[zoom_focus_idx] if zoom_focus_idx < len(outputs) else None
        zoom_pair_idx = VIEW_KEYS.index("zoom_pair")
        latest_pair = outputs[zoom_pair_idx] if zoom_pair_idx < len(outputs) else None
        zoom_piece_idx = VIEW_KEYS.index("zoom_piece")
        latest_piece = (
            outputs[zoom_piece_idx] if zoom_piece_idx < len(outputs) else None
        )

    template_view = None
    if template_rgb is not None:
        template_marked = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2BGR).copy()
        for pidx, state in enumerate(piece_states):
            payload = state.get("payload") if state else None
            if not payload or not getattr(payload, "matches", None):
                continue
            match_index = _clamp_match_index(payload, state.get("match_index", 0))
            match = payload.matches[match_index]
            color = GRID_COLORS_BGR[pidx % len(GRID_COLORS_BGR)]

            piece_rgb = getattr(payload, "piece_rgb", None)
            piece_bin = getattr(payload, "piece_bin", None)
            if piece_rgb is not None and piece_bin is not None:
                piece_bgr = cv2.cvtColor(piece_rgb, cv2.COLOR_RGB2BGR)
                _overlay_piece_on_template(template_marked, piece_bgr, piece_bin, match)

            contours = match.get("contours", [])
            if contours:
                for cnt in contours:
                    cnt = np.asarray(cnt).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(template_marked, [cnt], True, color, 2)
            else:
                tlx, tly = match.get("tl", (0, 0))
                brx, bry = match.get("br", (0, 0))
                cv2.rectangle(template_marked, (tlx, tly), (brx, bry), color, 2)

            tlx, tly = match.get("tl", (0, 0))
            brx, bry = match.get("br", (tlx, tly))
            label = f"{pidx + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_w, text_h = text_size
            center = match.get("center")
            if center and len(center) == 2:
                center_x, center_y = center
            else:
                center_x = tlx + (brx - tlx) // 2
                center_y = tly + (bry - tly) // 2
            x = int(center_x - text_w / 2)
            y = int(center_y + text_h / 2)
            cv2.putText(
                template_marked,
                label,
                (x, y),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                template_marked,
                label,
                (x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        template_marked_rgb = cv2.cvtColor(template_marked, cv2.COLOR_BGR2RGB)
        template_view = make_zoomable_plot(template_marked_rgb)

    views = {key: None for key in VIEW_KEYS}
    views["grid_overview"] = grid_overview
    views["zoom_focus"] = (
        latest_zoom
        if isinstance(latest_zoom, np.ndarray)
        else np.zeros((200, 200, 3), dtype=np.uint8)
    )
    views["zoom_pair"] = (
        latest_pair
        if isinstance(latest_pair, np.ndarray)
        else np.zeros((200, 200, 3), dtype=np.uint8)
    )
    views["zoom_piece"] = (
        latest_piece
        if isinstance(latest_piece, np.ndarray)
        else np.zeros((200, 200, 3), dtype=np.uint8)
    )
    multipiece_pairs = []
    piece_indices_with_pairs = []
    for pidx, state in enumerate(piece_states):
        payload = state.get("payload") if state else None
        if not payload or not getattr(payload, "matches", None):
            continue
        match_index = _clamp_match_index(payload, state.get("match_index", 0))
        try:
            pair_view = render_primary_views(payload, match_index).get("zoom_pair")
        except Exception:
            pair_view = None
        if isinstance(pair_view, np.ndarray):
            labeled = _annotate_pair_image(pair_view, f"Piece {pidx + 1}")
            multipiece_pairs.append(labeled)
            piece_indices_with_pairs.append(pidx)

    piece_heights = []
    if multipiece_pairs:
        stacked, piece_heights = _stack_images_vertical(multipiece_pairs)
        views["zoom_pair"] = stacked
        # Store heights in batch_state for button alignment
        if batch_state is not None:
            batch_state["piece_image_heights"] = piece_heights
            batch_state["piece_indices_displayed"] = piece_indices_with_pairs
    views["zoom_template"] = (
        template_view if template_view is not None else make_zoomable_plot(None)
    )
    rotated_template = _rotate_template_preview(
        TEMPLATE_IMAGES.get(template_id), rotation
    )
    views["template_color"] = rotated_template
    return views


def _advance_multipiece_candidate(piece_index: int, batch_state):
    if not batch_state or "piece_states" not in batch_state:
        blank_views = {key: None for key in VIEW_KEYS}
        return _views_to_outputs(
            blank_views,
            "Run the matcher once a piece is uploaded.",
            "",
            None,
            0,
            batch_state,
        )
    piece_states = batch_state.get("piece_states") or []
    total = int(batch_state.get("total") or len(piece_states))
    if piece_index < 0 or piece_index >= len(piece_states):
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Piece {piece_index + 1} is not available yet.",
            None,
            0,
            batch_state,
        )
    state = piece_states[piece_index]
    if not state or state.get("payload") is None:
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Piece {piece_index + 1} is not available yet.",
            None,
            0,
            batch_state,
        )
    payload = state.get("payload")
    if not payload or not getattr(payload, "matches", None):
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Piece {piece_index + 1} has no matches.",
            None,
            0,
            batch_state,
        )
    match_index = _clamp_match_index(payload, state.get("match_index", 0))
    next_index = (match_index + 1) % len(payload.matches)
    state["match_index"] = next_index

    views = _build_multipiece_views_from_state(batch_state)
    coord_markdown = _format_multipiece_table(piece_states, total)
    summary = (
        f"Piece {piece_index + 1}: candidate {next_index + 1}/{len(payload.matches)}."
    )
    return _views_to_outputs(
        views,
        coord_markdown,
        summary,
        None,
        0,
        batch_state,
    )


def _rotate_multipiece_candidate(piece_index: int, rotation_deg: float, batch_state):
    """Apply manual rotation to a piece and rematch without auto-alignment."""
    if not batch_state or "piece_states" not in batch_state:
        blank_views = {key: None for key in VIEW_KEYS}
        return _views_to_outputs(
            blank_views,
            "Run the matcher once a piece is uploaded.",
            "",
            None,
            0,
            batch_state,
        )

    piece_states = batch_state.get("piece_states") or []
    total = int(batch_state.get("total") or len(piece_states))
    template_id = batch_state.get("template_id")
    template_rotation = batch_state.get("rotation", 0)

    if piece_index < 0 or piece_index >= len(piece_states):
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Piece {piece_index + 1} is not available yet.",
            None,
            0,
            batch_state,
        )

    state = piece_states[piece_index]
    if not state or state.get("payload") is None:
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Piece {piece_index + 1} has no data to rotate.",
            None,
            0,
            batch_state,
        )

    # Get the original piece path from cached data
    cached_piece_path = state.get("piece_path")
    if not cached_piece_path or not os.path.exists(cached_piece_path):
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Piece {piece_index + 1} data not available for rotation.",
            None,
            0,
            batch_state,
        )

    # Get current auto-alignment and manual rotation
    payload = state.get("payload")
    auto_align_deg = getattr(payload, "auto_align_deg", 0.0) if payload else 0.0
    current_manual_rotation = state.get("manual_rotation", 0.0)
    new_manual_rotation = current_manual_rotation + rotation_deg

    # Total rotation = original auto-alignment + cumulative manual adjustments
    total_rotation = auto_align_deg + new_manual_rotation

    # Get template spec and run matcher with forced rotation (no auto-align)
    template_spec = TEMPLATE_REGISTRY.get(template_id)
    try:
        matcher_config = build_matcher_config(
            {
                "rows": template_spec.rows,
                "cols": template_spec.cols,
                "crop_x": template_spec.crop_x,
                "crop_y": template_spec.crop_y,
                **template_spec.matcher_overrides,
            }
        )

        # Load the original piece and apply total rotation
        piece_bgr = cv2.imread(cached_piece_path)
        if piece_bgr is None:
            raise ValueError(f"Could not load piece image: {cached_piece_path}")

        # Apply total rotation to the piece
        if abs(total_rotation) > 0.01:
            h, w = piece_bgr.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, total_rotation, 1.0)
            piece_bgr = cv2.warpAffine(piece_bgr, rot_matrix, (w, h))

        # Save rotated piece to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_rotated_path = tmp.name
            cv2.imwrite(tmp_rotated_path, piece_bgr)

        try:
            # Run matcher without auto-align
            payload = find_piece_in_template(
                tmp_rotated_path,
                str(template_spec.template_path),
                knobs_x=None,
                knobs_y=None,
                auto_align=False,  # Don't auto-align, use manual rotation only
                infer_knobs=True,
                template_rotation=template_rotation,
                matcher_config=matcher_config,
            )

            # Update state with new payload and rotation
            state["payload"] = payload
            state["match_index"] = 0
            state["manual_rotation"] = new_manual_rotation

        finally:
            try:
                os.remove(tmp_rotated_path)
            except OSError:
                # Best-effort cleanup: ignoring failures to delete the temporary file
                pass

        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        summary = (
            f"Piece {piece_index + 1}: rotated {new_manual_rotation:+.1f}° "
            f"(total: {total_rotation:+.1f}°, {len(payload.matches)} candidates)."
        )
        return _views_to_outputs(
            views,
            coord_markdown,
            summary,
            None,
            0,
            batch_state,
        )
    except Exception as exc:
        views = _build_multipiece_views_from_state(batch_state)
        coord_markdown = _format_multipiece_table(piece_states, total)
        return _views_to_outputs(
            views,
            coord_markdown,
            f"Error rotating piece {piece_index + 1}: {exc}",
            None,
            0,
            batch_state,
        )


def _change_match(
    step: int,
    payload,
    current_index: int,
    template_id: str,
    template_rotation: int,
    batch_state=None,
):
    if payload is None or not getattr(payload, "matches", None):
        return _blank_outputs(
            "Run the matcher once a piece is uploaded.",
            template_id,
            template_rotation,
        )
    total = len(payload.matches)
    if total == 0:
        return _blank_outputs("No matches available.", template_id, template_rotation)
    idx = (current_index or 0) + step
    idx %= total
    return _render_match_payload(payload, idx, batch_state)


def solve_puzzle(piece_path, template_id, auto_align, template_rotation):
    """Run the high-performance matcher and return visualization slices"""
    template_spec = TEMPLATE_REGISTRY.get(template_id)
    rotation = (
        int(template_rotation)
        if template_rotation is not None
        else template_spec.default_rotation
    )
    if not piece_path or not os.path.exists(piece_path):
        return _blank_outputs(
            "Please upload a puzzle piece image.", template_id, rotation
        )

    try:
        matcher_config = build_matcher_config(
            {
                "rows": template_spec.rows,
                "cols": template_spec.cols,
                "crop_x": template_spec.crop_x,
                "crop_y": template_spec.crop_y,
                **template_spec.matcher_overrides,
            }
        )
        payload = find_piece_in_template(
            piece_path,
            str(template_spec.template_path),
            knobs_x=None,
            knobs_y=None,
            auto_align=bool(auto_align),
            infer_knobs=True,
            template_rotation=rotation,
            matcher_config=matcher_config,
        )
        return _render_match_payload(payload, 0)
    except Exception as exc:  # pylint: disable=broad-except
        return _blank_outputs(f"Error: {exc}", template_id, rotation)


def solve_puzzle_multipiece(piece_path, template_id, auto_align, template_rotation):
    """Detect multiple pieces in an image and stream match results."""
    template_spec = TEMPLATE_REGISTRY.get(template_id)
    rotation = (
        int(template_rotation)
        if template_rotation is not None
        else template_spec.default_rotation
    )

    if not piece_path or not os.path.exists(piece_path):
        yield _blank_outputs(
            "Please upload a puzzle piece image.", template_id, rotation
        )
        return

    try:
        grid_img = Image.open(piece_path).convert("RGB")
    except Exception as exc:
        yield _blank_outputs(
            f"Could not read grid image: {exc}",
            template_id,
            rotation,
        )
        return

    grid_bgr = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2BGR)
    matcher_config = build_matcher_config(
        {
            "rows": template_spec.rows,
            "cols": template_spec.cols,
            "crop_x": template_spec.crop_x,
            "crop_y": template_spec.crop_y,
            **template_spec.matcher_overrides,
        }
    )
    regions, _ = _find_multipiece_regions(grid_bgr, matcher_config)
    if not regions:
        yield _blank_outputs(
            "No pieces detected in the image.",
            template_id,
            rotation,
        )
        return

    total = len(regions)
    all_results = []
    piece_states = [None] * total
    colors = GRID_COLORS_BGR

    grid_overview = _build_multipiece_overview(grid_bgr, regions, colors)
    # Get template RGB for overlay
    template_rgb = None
    for spec in TEMPLATE_REGISTRY.templates.values():
        if spec.template_id == template_id:
            template_rgb = get_template_image(
                spec.template_path, crop_x=spec.crop_x, crop_y=spec.crop_y
            )
            break

    batch_state = {
        "template_id": template_id,
        "rotation": rotation,
        "grid_overview": grid_overview,
        "template_rgb": template_rgb,
        "piece_states": piece_states,
        "total": total,
    }

    for idx, region in enumerate(regions):
        x, y, w, h = region["bbox"]
        pad = max(4, int(min(w, h) * 0.06))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(grid_bgr.shape[1], x + w + pad)
        y1 = min(grid_bgr.shape[0], y + h + pad)
        crop = grid_img.crop((x0, y0, x1, y1))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            crop.save(tmp_path)

        try:
            result = solve_puzzle(tmp_path, template_id, auto_align, rotation)
            all_results.append((idx, region, result))
            payload = None
            if result is not None:
                outputs = list(result)
                state_idx = len(VIEW_KEYS) + 2
                if state_idx < len(outputs):
                    payload = outputs[state_idx]
            piece_states[idx] = {
                "payload": payload,
                "match_index": 0,
                "piece_path": tmp_path,  # Cache the piece path for rotation
                "manual_rotation": 0.0,
            }
        except Exception:
            all_results.append((idx, region, None))
            piece_states[idx] = {
                "payload": None,
                "match_index": 0,
                "error": True,
                "piece_path": tmp_path,
                "manual_rotation": 0.0,
            }

        coord_markdown = _format_multipiece_table(piece_states, total)

        last_result = all_results[-1][2] if all_results else None
        streamed_views = _build_multipiece_views_from_state(
            batch_state, last_result=last_result
        )
        last_summary = ""
        if last_result is not None:
            outputs = list(last_result)
            summary_idx = len(VIEW_KEYS) + 1
            if summary_idx < len(outputs):
                last_summary = str(outputs[summary_idx])
        yield _views_to_outputs(
            streamed_views,
            coord_markdown,
            (
                f"Processed {len(all_results)}/{total} pieces."
                + (f"\n\n{last_summary}" if last_summary else "")
            ),
            None,
            0,
            batch_state,
        )

    combined_location = _format_multipiece_table(piece_states, total)

    final_views = _build_multipiece_views_from_state(
        batch_state, last_result=all_results[-1][2] if all_results else None
    )

    summary = f"Processed {len(all_results)}/{total} pieces."
    yield _views_to_outputs(
        final_views, combined_location, summary, None, 0, batch_state
    )


def solve_single_or_batch(
    piece_path, template_id, auto_align, template_rotation, batch_mode
):
    """Dispatch to single-piece solve or streamed multipiece mode."""
    if batch_mode:
        yield from solve_puzzle_multipiece(
            piece_path, template_id, auto_align, template_rotation
        )
    else:
        result = solve_puzzle(piece_path, template_id, auto_align, template_rotation)
        yield result


def goto_previous_match(
    state, current_index, template_id, template_rotation, batch_state=None
):
    return _change_match(
        -1, state, current_index, template_id, template_rotation, batch_state
    )


def goto_next_match(
    state, current_index, template_id, template_rotation, batch_state=None
):
    return _change_match(
        1, state, current_index, template_id, template_rotation, batch_state
    )


TEMPLATE_REGISTRY = load_template_registry()
DEFAULT_TEMPLATE_ID = TEMPLATE_REGISTRY.default_template_id
DEFAULT_TEMPLATE_SPEC = TEMPLATE_REGISTRY.get(DEFAULT_TEMPLATE_ID)

for spec in TEMPLATE_REGISTRY.templates.values():
    check_template_exists(spec.template_path)
    try:
        overrides = spec.matcher_overrides
        if "binarize_blur_ksz" in overrides or "match_blur_ksz" in overrides:
            preload_template_cache(
                str(spec.template_path),
                blur_ksz=overrides.get("match_blur_ksz"),
                binarize_blur_ksz=overrides.get("binarize_blur_ksz"),
                crop_x=spec.crop_x,
                crop_y=spec.crop_y,
            )
        else:
            preload_template_cache(
                str(spec.template_path),
                crop_x=spec.crop_x,
                crop_y=spec.crop_y,
            )
    except Exception:
        # Preloading is an optimization; if it fails, templates will load on-demand
        pass

TEMPLATE_IMAGES = {
    spec.template_id: get_template_image(
        spec.template_path, crop_x=spec.crop_x, crop_y=spec.crop_y
    )
    for spec in TEMPLATE_REGISTRY.templates.values()
}
DEFAULT_TEMPLATE_PREVIEW = _rotate_template_preview(
    TEMPLATE_IMAGES.get(DEFAULT_TEMPLATE_ID), DEFAULT_TEMPLATE_SPEC.default_rotation
)
DEFAULT_TEMPLATE_PLOT = make_zoomable_plot(DEFAULT_TEMPLATE_PREVIEW)

# Create Gradio interface
app_theme = gr.themes.Soft()
with gr.Blocks(title=f"🧩 WCMBot v{__version__}") as demo:
    gr.Markdown(
        f"""
    # 🧩 WCMBot v{__version__}

    Upload a picture of a jigsaw puzzle piece and let WCMBot infer its tab
    counts and location in the full puzzle template!

    Notes:
    - Pictures must show a single puzzle piece on a plain (not blue) background.
    - The piece should be aligned roughly upright in the picture for best results.
      Optional auto-align (experimental) can correct small tilts (rotations of multiples of 90° are evaluated).
    - Template rotation can be adjusted in 90° increments.
    
    This app is almost entirely vibe-coded. If you and/or your AI agents would like to
    contribute to its development, proposals and PRs are very welcome at
    https://github.com/wcmbotanicals/wcmbot.
    """
    )

    # Display random advertisement banner per session
    ad_banner = gr.HTML()
    demo.load(fn=get_random_ad, outputs=ad_banner)

    gr.HTML(
        """
    <style>
    #primary-template-view img {
        cursor: zoom-in;
    }
    .batch-button-group {
        width: 100%;
        padding: 12px 8px;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background: #f9f9f9;
        margin-bottom: 0px;
        box-sizing: border-box;
    }
    .batch-button-group button {
        width: 100%;
        margin: 3px 0;
    }
    /* Ensure title alignment */
    h3 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Puzzle Piece")
            piece_input = gr.Image(
                label="Puzzle Piece",
                type="filepath",
                sources=["upload", "clipboard"],
                height=300,
                elem_id="piece-upload",
            )
            grid_overview_header = gr.Markdown(
                "### Multipiece overview (numbered)", visible=MULTIPIECE_DEFAULT
            )
            image_components = {}
            image_components["grid_overview"] = gr.Image(
                label="Numbered multipiece overview",
                type="numpy",
                interactive=False,
                height=300,
                visible=MULTIPIECE_DEFAULT,
            )
            with gr.Accordion("Settings", open=False):
                template_selector = gr.Dropdown(
                    label="Template",
                    choices=TEMPLATE_REGISTRY.choices(),
                    value=DEFAULT_TEMPLATE_ID,
                )
                auto_align_checkbox = gr.Checkbox(
                    label="Auto-align (experimental)",
                    value=DEFAULT_TEMPLATE_SPEC.auto_align_default,
                )
                batch_grid_checkbox = gr.Checkbox(
                    label="Multipiece mode (batch)",
                    value=MULTIPIECE_DEFAULT,
                    info=(
                        "If checked, detect multiple pieces in the upload and solve "
                        "each in sequence."
                    ),
                )
                show_batch_buttons_checkbox = gr.Checkbox(
                    label="Show per-piece candidate buttons",
                    value=True,
                    info=(
                        "Show a Next-candidate button for each detected piece when "
                        "multipiece mode is enabled."
                    ),
                )
                diagnostic_mode_checkbox = gr.Checkbox(
                    label="Show diagnostic visualizations",
                    value=False,
                    info=(
                        "Show detailed mask, binary, and processing steps for single pieces. "
                        "Has no effect when 'Multipiece mode (batch)' is enabled."
                    ),
                )
                template_rotation = gr.Dropdown(
                    label="Template rotation (degrees)",
                    choices=TEMPLATE_ROTATION_OPTIONS,
                    value=DEFAULT_TEMPLATE_SPEC.default_rotation,
                )
            solve_button = gr.Button(
                "🔍 Find Piece Location", variant="primary", size="lg"
            )
        with gr.Column(scale=1):
            gr.Markdown("### Inferred row/column")
            match_location = gr.Markdown(
                "Run the matcher to infer a row and column.",
                elem_id="match-location",
            )
            match_summary = gr.Markdown(
                "Run the matcher to view detailed plots.",
                elem_id="match-summary",
            )
            gr.Markdown("### Best match (template view)")
            image_components["zoom_template"] = gr.Plot(
                value=DEFAULT_TEMPLATE_PLOT,
                elem_id="primary-template-view",
            )
            gr.Markdown("Use the controls to zoom and pan the image.")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Best match (side-by-side)")
            image_components["zoom_pair"] = gr.Image(
                label="Piece + template match (outline)",
                type="numpy",
                interactive=False,
            )
        batch_buttons_column = gr.Column(scale=1, visible=MULTIPIECE_DEFAULT)
        with batch_buttons_column:
            gr.Markdown("### Per-piece controls")
            # Add top spacer (half height)
            top_spacer = gr.HTML("", visible=False)

            batch_button_groups = []
            batch_button_containers = []
            batch_spacers = [top_spacer]  # Start with top spacer
            batch_rotation_inputs = []
            for i in range(MAX_DYNAMIC_BUTTONS):
                container = gr.Group(elem_classes=["batch-button-group"], visible=False)
                batch_button_containers.append(container)
                with container:
                    gr.Markdown(f"**Piece {i + 1}**")
                    next_btn = gr.Button("Next candidate", size="sm")
                    gr.HTML(
                        '<div style="font-size: 13px; margin-top: 8px; margin-bottom: 4px;">Rotate:</div>'
                    )
                    with gr.Row():
                        rotation_input = gr.Number(
                            value=2.5,
                            minimum=0,
                            maximum=10,
                            step=0.1,
                            show_label=False,
                            container=False,
                            scale=2,
                            min_width=60,
                        )
                        gr.HTML(
                            '<div style="font-size: 14px; padding: 8px 4px;">°</div>',
                            scale=0,
                            min_width=20,
                        )
                        rotate_ccw_btn = gr.Button(
                            "↺ CCW", size="sm", scale=3, min_width=70
                        )
                        rotate_cw_btn = gr.Button(
                            "↻ CW", size="sm", scale=3, min_width=70
                        )
                    batch_button_groups.append(
                        {
                            "next": next_btn,
                            "rotate_cw": rotate_cw_btn,
                            "rotate_ccw": rotate_ccw_btn,
                            "rotation_input": rotation_input,
                        }
                    )
                    batch_rotation_inputs.append(rotation_input)
                # Add spacer after each group (except last)
                if i < MAX_DYNAMIC_BUTTONS - 1:
                    spacer = gr.HTML("", visible=False)
                    batch_spacers.append(spacer)

            # Add bottom spacer (half height)
            bottom_spacer = gr.HTML("", visible=False)
            batch_spacers.append(bottom_spacer)
    with gr.Row(visible=False):
        image_components["zoom_piece"] = gr.Image(
            label="Piece (masked + rotated)",
            type="numpy",
            interactive=False,
        )
        image_components["zoom_focus"] = gr.Image(
            label="Template match (outline)",
            type="numpy",
            interactive=False,
        )
    # Diagnostic outputs - hidden by default, shown when diagnostic mode enabled
    diagnostic_header = gr.Markdown(
        "### Match visualizations/diagnostics", visible=False
    )

    other_keys = [
        key
        for key in VIEW_KEYS
        if key
        not in (
            "zoom_template",
            "zoom_focus",
            "zoom_piece",
            "zoom_pair",
            "grid_overview",
        )
    ]

    diagnostic_row1 = gr.Row(visible=False)
    with diagnostic_row1:
        for key in other_keys[:4]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                value=DEFAULT_TEMPLATE_PREVIEW if key == "template_color" else None,
                interactive=False,
                height=260,
            )
            image_components[key] = comp

    diagnostic_row2 = gr.Row(visible=False)
    with diagnostic_row2:
        for key in other_keys[4:]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                interactive=False,
                height=260,
            )
            image_components[key] = comp

    diagnostic_controls = gr.Row(visible=False)
    with diagnostic_controls:
        prev_button = gr.Button("⬅️ Previous match")
        next_button = gr.Button("Next match ➡️")

    match_state = gr.State()
    match_index = gr.State(0)
    batch_state = gr.State()

    ordered_components = [image_components[key] for key in VIEW_KEYS]

    def _toggle_diagnostics(show_diag):
        """Toggle visibility of diagnostic outputs."""
        return {
            diagnostic_header: gr.update(visible=show_diag),
            diagnostic_row1: gr.update(visible=show_diag),
            diagnostic_row2: gr.update(visible=show_diag),
            diagnostic_controls: gr.update(visible=show_diag),
        }

    diagnostic_mode_checkbox.change(
        fn=_toggle_diagnostics,
        inputs=[diagnostic_mode_checkbox],
        outputs=[
            diagnostic_header,
            diagnostic_row1,
            diagnostic_row2,
            diagnostic_controls,
        ],
    )

    def _toggle_grid_overview(show_grid, show_buttons):
        show_buttons = bool(show_grid) and bool(show_buttons)
        return {
            grid_overview_header: gr.update(visible=show_grid),
            image_components["grid_overview"]: gr.update(visible=show_grid),
            batch_buttons_column: gr.update(visible=show_buttons),
        }

    batch_grid_checkbox.change(
        fn=_toggle_grid_overview,
        inputs=[batch_grid_checkbox, show_batch_buttons_checkbox],
        outputs=[
            grid_overview_header,
            image_components["grid_overview"],
            batch_buttons_column,
        ],
    )
    show_batch_buttons_checkbox.change(
        fn=_toggle_grid_overview,
        inputs=[batch_grid_checkbox, show_batch_buttons_checkbox],
        outputs=[
            grid_overview_header,
            image_components["grid_overview"],
            batch_buttons_column,
        ],
    )

    def _on_template_change(selected_template):
        spec = TEMPLATE_REGISTRY.get(selected_template)
        defaults = _blank_outputs(
            "Run the matcher once a piece is uploaded.",
            selected_template,
            spec.default_rotation,
        )
        return (
            spec.default_rotation,
            spec.auto_align_default,
            *defaults,
        )

    template_selector.change(
        fn=_on_template_change,
        inputs=[template_selector],
        outputs=[
            template_rotation,
            auto_align_checkbox,
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
            batch_state,
            *batch_button_containers,
            *batch_spacers,
        ],
    )

    def _no_update_outputs(state, idx, batch_state):
        num_spacers = (
            MAX_DYNAMIC_BUTTONS + 1
        )  # One top spacer, MAX_DYNAMIC_BUTTONS-1 between button groups, and one bottom spacer
        return (
            *([gr.update()] * len(VIEW_KEYS)),
            gr.update(),
            gr.update(),
            state,
            idx,
            batch_state,
            *([gr.update()] * MAX_DYNAMIC_BUTTONS),  # Button visibility updates
            *([gr.update()] * num_spacers),  # Spacer updates (including top and bottom)
        )

    def _on_piece_change(
        piece_path,
        template_id,
        auto_align,
        template_rotation,
        batch_mode,
        state,
        idx,
        batch_state,
    ):
        if not piece_path:
            yield _no_update_outputs(state, idx, batch_state)
            return
        result = solve_single_or_batch(
            piece_path, template_id, auto_align, template_rotation, batch_mode
        )
        yield from result

    # Auto-run matching whenever a new piece is uploaded or pasted
    piece_input.upload(
        fn=_on_piece_change,
        inputs=[
            piece_input,
            template_selector,
            auto_align_checkbox,
            template_rotation,
            batch_grid_checkbox,
            match_state,
            match_index,
            batch_state,
        ],
        outputs=[
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
            batch_state,
            *batch_button_containers,
            *batch_spacers,
        ],
    )

    solve_button.click(
        fn=solve_single_or_batch,
        inputs=[
            piece_input,
            template_selector,
            auto_align_checkbox,
            template_rotation,
            batch_grid_checkbox,
        ],
        outputs=[
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
            batch_state,
            *batch_button_containers,
            *batch_spacers,
        ],
    )
    prev_button.click(
        fn=goto_previous_match,
        inputs=[
            match_state,
            match_index,
            template_selector,
            template_rotation,
            batch_state,
        ],
        outputs=[
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
            batch_state,
            *batch_button_containers,
            *batch_spacers,
        ],
    )
    next_button.click(
        fn=goto_next_match,
        inputs=[
            match_state,
            match_index,
            template_selector,
            template_rotation,
            batch_state,
        ],
        outputs=[
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
            batch_state,
            *batch_button_containers,
            *batch_spacers,
        ],
    )

    # Hook up batch button handlers
    for i, button_group in enumerate(batch_button_groups):
        rotation_input = button_group["rotation_input"]

        # Next candidate button
        button_group["next"].click(
            fn=partial(_advance_multipiece_candidate, i),
            inputs=[batch_state],
            outputs=[
                *ordered_components,
                match_location,
                match_summary,
                match_state,
                match_index,
                batch_state,
                *batch_button_containers,
                *batch_spacers,
            ],
        )

        # Rotate clockwise button (negative angle for CW)
        def make_rotate_cw_handler(piece_idx):
            def handler(batch_state, rotation_deg):
                deg = 1 if rotation_deg is None else rotation_deg
                return _rotate_multipiece_candidate(piece_idx, -abs(deg), batch_state)

            return handler

        button_group["rotate_cw"].click(
            fn=make_rotate_cw_handler(i),
            inputs=[batch_state, rotation_input],
            outputs=[
                *ordered_components,
                match_location,
                match_summary,
                match_state,
                match_index,
                batch_state,
                *batch_button_containers,
                *batch_spacers,
            ],
        )

        # Rotate counter-clockwise button (positive angle for CCW)
        def make_rotate_ccw_handler(piece_idx):
            def handler(batch_state, rotation_deg):
                deg = 1 if rotation_deg is None else rotation_deg
                return _rotate_multipiece_candidate(piece_idx, abs(deg), batch_state)

            return handler

        button_group["rotate_ccw"].click(
            fn=make_rotate_ccw_handler(i),
            inputs=[batch_state, rotation_input],
            outputs=[
                *ordered_components,
                match_location,
                match_summary,
                match_state,
                match_index,
                batch_state,
                *batch_button_containers,
                *batch_spacers,
            ],
        )

    gr.Markdown(
        """
    ---
    ### About
    Use the navigation buttons to inspect alternative placements when multiple
    candidates score highly.
    """
    )

if __name__ == "__main__":
    demo.launch(theme=app_theme)
