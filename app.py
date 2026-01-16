"""Gradio interface for the jigsaw puzzle solver"""

import base64
import os
import random
import tempfile
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
    "zoom_focus": "Best match (zoomed)",
    "zoom_template": "Best match (template view)",
    "zoom_piece": "Piece (masked + rotated)",
    "zoom_pair": "Best match (side-by-side)",
}

TEMPLATE_ROTATION_OPTIONS = [0, 90, 180, 270]


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


def _rotate_template_preview(
    image: Optional[np.ndarray], rotation: int
) -> Optional[np.ndarray]:
    if image is None:
        return None
    if rotation == 0:
        return image
    k = -(rotation // 90)
    return np.rot90(image, k=k)


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


def get_template_image(template_path: Path) -> Optional[np.ndarray]:
    """Get the template image."""
    if template_path.exists():
        return np.array(Image.open(template_path))
    return None


def _views_to_outputs(
    views: Dict[str, Optional[np.ndarray]],
    location: str,
    summary: str,
    state,
    idx: int,
):
    ordered = [views.get(key) for key in VIEW_KEYS]
    return (*ordered, location, summary, state, idx)


def _blank_outputs(message: str, template_id: str, template_rotation: int):
    blank_views = {key: None for key in VIEW_KEYS}
    rotated_template = _rotate_template_preview(
        TEMPLATE_IMAGES.get(template_id), template_rotation
    )
    blank_views["template_color"] = rotated_template
    blank_views["zoom_template"] = make_zoomable_plot(rotated_template)
    location = "Run the matcher to infer a row and column."
    return _views_to_outputs(blank_views, location, message, None, 0)


def _format_match_location(payload, idx: int) -> str:
    if payload is None or not getattr(payload, "matches", None):
        return "Run the matcher to infer a row and column."
    match = payload.matches[idx]
    return f"**Row:** {match['row']}  **Col:** {match['col']}"


def _render_match_payload(payload, idx: int):
    views = render_primary_views(payload, idx)
    views["zoom_template"] = make_zoomable_plot(views.get("zoom_template"))
    location = _format_match_location(payload, idx)
    summary = format_match_summary(payload, idx)
    return _views_to_outputs(views, location, summary, payload, idx)


def _change_match(
    step: int,
    payload,
    current_index: int,
    template_id: str,
    template_rotation: int,
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
    return _render_match_payload(payload, idx)


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


def solve_puzzle_grid(piece_path, template_id, auto_align, template_rotation):
    """Split a 3x3 grid upload into nine cells and stream results as each completes."""
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

    width, height = grid_img.size
    rows = cols = 3
    cell_w = width // cols
    cell_h = height // rows
    if cell_w <= 0 or cell_h <= 0:
        yield _blank_outputs(
            "Grid image is too small to split.",
            template_id,
            rotation,
        )
        return

    total = rows * cols
    all_results = []

    # Define distinct colors (BGR) - no green
    colors = [
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

    # Convert BGR to RGB for HTML display
    colors_rgb = [(b, g, r) for (b, g, r) in colors]

    # Get template RGB for overlay
    template_rgb = None
    for spec in TEMPLATE_REGISTRY.templates.values():
        if spec.template_id == template_id:
            template_rgb = get_template_image(spec.template_path)
            break

    for idx in range(total):
        r = idx // cols
        c = idx % cols
        box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
        crop = grid_img.crop(box)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            crop.save(tmp_path)

        try:
            result = solve_puzzle(tmp_path, template_id, auto_align, rotation)
            all_results.append((idx, r, c, result))
        except Exception:
            all_results.append((idx, r, c, None))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                # Ignore errors if temporary file already deleted or inaccessible
                pass

        # Build and yield incremental update after each piece
        coord_lines = [
            "| Piece | Grid Position | Row | Col |",
            "|-------|---------------|-----|-----|",
        ]
        for i, (pidx, pr, pc, presult) in enumerate(all_results):
            piece_color = colors_rgb[pidx % len(colors_rgb)]
            piece_num_html = f'<span style="color: rgb({piece_color[0]}, {piece_color[1]}, {piece_color[2]})">**{pidx + 1}**</span>'
            if presult is None:
                coord_lines.append(
                    f"| {piece_num_html} | r{pr + 1}, c{pc + 1} | Error | Error |"
                )
                continue
            outputs = list(presult)
            location_idx = len(VIEW_KEYS)
            location_text = (
                str(outputs[location_idx]) if location_idx < len(outputs) else ""
            )
            try:
                if "**Row:**" in location_text and "**Col:**" in location_text:
                    row_match = (
                        location_text.split("**Row:**")[1].split("**")[0].strip()
                    )
                    col_match = (
                        location_text.split("**Col:**")[1].split("\n")[0].strip()
                    )
                    coord_lines.append(
                        f"| {piece_num_html} | r{pr + 1}, c{pc + 1} | {row_match} | {col_match} |"
                    )
                else:
                    coord_lines.append(
                        f"| {piece_num_html} | r{pr + 1}, c{pc + 1} | - | - |"
                    )
            except (IndexError, AttributeError):
                coord_lines.append(
                    f"| {piece_num_html} | r{pr + 1}, c{pc + 1} | - | - |"
                )

        # Add placeholder rows for unprocessed pieces
        remaining = total - len(all_results)
        for _ in range(remaining):
            coord_lines.append("| - | - | - | - |")

        coord_markdown = "\n".join(coord_lines)

        yield (
            None,  # image plot (unchanged here)
            coord_markdown,
            f"Processed {len(all_results)}/{total} pieces",
        )

    for i in range(idx + 1, total):
        upr = i // cols
        upc = i % cols
        piece_color = colors_rgb[i % len(colors_rgb)]
        piece_num_html = f'<span style="color: rgb({piece_color[0]}, {piece_color[1]}, {piece_color[2]})">**{i + 1}**</span>'
        coord_lines.append(f"| {piece_num_html} | r{upr + 1}, c{upc + 1} | ... | ... |")

    combined_location = "\n".join(coord_lines)

    # Build 3x3 grid with processed results and blank placeholders
    zoom_images = []
    for i in range(total):
        if i < len(all_results):
            _, _, _, presult = all_results[i]
            if presult is None:
                zoom_images.append(np.zeros((200, 200, 3), dtype=np.uint8))
            else:
                outputs = list(presult)
                zoom_focus_idx = VIEW_KEYS.index("zoom_focus")
                zoom_img = (
                    outputs[zoom_focus_idx] if zoom_focus_idx < len(outputs) else None
                )
                if zoom_img is not None and isinstance(zoom_img, np.ndarray):
                    zoom_images.append(zoom_img)
                else:
                    zoom_images.append(np.zeros((200, 200, 3), dtype=np.uint8))
        else:
            # Placeholder for unprocessed piece
            zoom_images.append(np.full((200, 200, 3), 200, dtype=np.uint8))

    # Arrange in 3x3 grid
    if zoom_images:
        heights = [img.shape[0] for img in zoom_images if img.shape[0] > 0]
        widths = [img.shape[1] for img in zoom_images if img.shape[1] > 0]
        target_h = int(np.median(heights)) if heights else 200
        target_w = int(np.median(widths)) if widths else 200

        uniform_images = []
        for img in zoom_images:
            if img.shape[0] > 0 and img.shape[1] > 0:
                h, w = img.shape[:2]
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(
                    img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left

                padded = cv2.copyMakeBorder(
                    resized,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            else:
                padded = np.full((target_h, target_w, 3), 200, dtype=np.uint8)
            uniform_images.append(padded)

        grid_rows = []
        for gr in range(3):
            row_imgs = uniform_images[gr * 3 : (gr + 1) * 3]
            grid_rows.append(np.hstack(row_imgs))

        combined_zoom = np.vstack(grid_rows)
    else:
        combined_zoom = np.full((600, 600, 3), 200, dtype=np.uint8)

    # Build template overlay with processed pieces
    template_view = None
    if template_rgb is not None:
        template_marked = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2BGR).copy()

        for pidx, pr, pc, presult in all_results:
            if presult is None:
                continue
            try:
                outputs = list(presult)
                state_idx = len(VIEW_KEYS) + 2
                if state_idx < len(outputs):
                    payload = outputs[state_idx]
                    if payload and hasattr(payload, "matches") and payload.matches:
                        match = payload.matches[0]
                        color = colors[pidx % len(colors)]

                        contours = match.get("contours", [])
                        if contours:
                            for cnt in contours:
                                cnt = np.asarray(cnt).reshape(-1, 2).astype(np.int32)
                                cv2.polylines(template_marked, [cnt], True, color, 2)
                        else:
                            tlx, tly = match.get("tl", (0, 0))
                            brx, bry = match.get("br", (0, 0))
                            cv2.rectangle(
                                template_marked, (tlx, tly), (brx, bry), color, 2
                            )

                        tlx, tly = match.get("tl", (0, 0))
                        cv2.putText(
                            template_marked,
                            f"{pidx + 1}",
                            (tlx + 5, tly + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
            except Exception:
                # Skip drawing overlay if match result is malformed or drawing fails
                pass

        template_marked_rgb = cv2.cvtColor(template_marked, cv2.COLOR_BGR2RGB)
        template_view = make_zoomable_plot(template_marked_rgb)

    final_views = {key: None for key in VIEW_KEYS}
    final_views["zoom_focus"] = combined_zoom
    final_views["zoom_pair"] = combined_zoom
    final_views["zoom_template"] = (
        template_view if template_view is not None else make_zoomable_plot(None)
    )
    rotated_template = _rotate_template_preview(
        TEMPLATE_IMAGES.get(template_id), rotation
    )
    final_views["template_color"] = rotated_template

    summary = f"Processed {len(all_results)}/{total} pieces."
    yield _views_to_outputs(final_views, combined_location, summary, None, 0)


def solve_single_or_batch(
    piece_path, template_id, auto_align, template_rotation, batch_mode
):
    """Dispatch to single-piece solve or streamed 3x3 batch mode."""
    if batch_mode:
        yield from solve_puzzle_grid(
            piece_path, template_id, auto_align, template_rotation
        )
    else:
        yield solve_puzzle(piece_path, template_id, auto_align, template_rotation)


def goto_previous_match(state, current_index, template_id, template_rotation):
    return _change_match(-1, state, current_index, template_id, template_rotation)


def goto_next_match(state, current_index, template_id, template_rotation):
    return _change_match(1, state, current_index, template_id, template_rotation)


TEMPLATE_REGISTRY = load_template_registry()
DEFAULT_TEMPLATE_ID = TEMPLATE_REGISTRY.default_template_id
DEFAULT_TEMPLATE_SPEC = TEMPLATE_REGISTRY.get(DEFAULT_TEMPLATE_ID)

for spec in TEMPLATE_REGISTRY.templates.values():
    check_template_exists(spec.template_path)
    try:
        if "binarize_blur_ksz" in spec.matcher_overrides:
            preload_template_cache(
                str(spec.template_path),
                blur_ksz=spec.matcher_overrides.get("match_blur_ksz"),
                binarize_blur_ksz=spec.matcher_overrides["binarize_blur_ksz"],
            )
        elif "match_blur_ksz" in spec.matcher_overrides:
            preload_template_cache(
                str(spec.template_path),
                blur_ksz=spec.matcher_overrides["match_blur_ksz"],
            )
        else:
            preload_template_cache(str(spec.template_path))
    except Exception:
        # Preloading is an optimization; if it fails, templates will load on-demand
        pass

TEMPLATE_IMAGES = {
    spec.template_id: get_template_image(spec.template_path)
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
                    label="Treat upload as 3x3 grid (batch)",
                    value=False,
                    info="If checked, split the uploaded image into 9 equal cells and solve each in sequence.",
                )
                diagnostic_mode_checkbox = gr.Checkbox(
                    label="Show diagnostic visualizations",
                    value=False,
                    info=(
                        "Show detailed mask, binary, and processing steps for single pieces. "
                        "Has no effect when 'Treat upload as 3x3 grid (batch)' is enabled."
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
            image_components = {}
            image_components["zoom_template"] = gr.Plot(
                value=DEFAULT_TEMPLATE_PLOT,
                elem_id="primary-template-view",
            )
            gr.Markdown("Use the controls to zoom and pan the image.")

    gr.Markdown("### Best match (side-by-side)")
    image_components["zoom_pair"] = gr.Image(
        label="Piece + template match (outline)",
        type="numpy",
        interactive=False,
        height=300,
    )
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
        if key not in ("zoom_template", "zoom_focus", "zoom_piece", "zoom_pair")
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
        ],
    )

    # Auto-run matching whenever a new piece is uploaded or pasted
    piece_input.change(
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
        ],
    )
    prev_button.click(
        fn=goto_previous_match,
        inputs=[match_state, match_index, template_selector, template_rotation],
        outputs=[
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
        ],
    )
    next_button.click(
        fn=goto_next_match,
        inputs=[match_state, match_index, template_selector, template_rotation],
        outputs=[
            *ordered_components,
            match_location,
            match_summary,
            match_state,
            match_index,
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
