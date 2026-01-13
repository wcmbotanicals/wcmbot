"""Gradio interface for the jigsaw puzzle solver"""

import base64
import os
import random
from pathlib import Path
from typing import Dict, Optional

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


def goto_previous_match(state, current_index, template_id, template_rotation):
    return _change_match(-1, state, current_index, template_id, template_rotation)


def goto_next_match(state, current_index, template_id, template_rotation):
    return _change_match(1, state, current_index, template_id, template_rotation)


TEMPLATE_REGISTRY = load_template_registry()
DEFAULT_TEMPLATE_ID = TEMPLATE_REGISTRY.default_template_id
DEFAULT_TEMPLATE_SPEC = TEMPLATE_REGISTRY.get(DEFAULT_TEMPLATE_ID)

for spec in TEMPLATE_REGISTRY.templates.values():
    check_template_exists(spec.template_path)
    preload_template_cache(str(spec.template_path))

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
                "Run the matcher to infer a row and column."
            )
            gr.Markdown("### Best match (template view)")
            image_components = {}
            image_components["zoom_template"] = gr.Plot(
                value=DEFAULT_TEMPLATE_PLOT,
                elem_id="primary-template-view",
            )
            gr.Markdown("Use the controls to zoom and pan the image.")

    gr.Markdown("### Best match (zoomed)")
    image_components["zoom_focus"] = gr.Image(
        label=VIEW_LABELS["zoom_focus"],
        type="numpy",
        interactive=False,
        height=320,
    )

    gr.Markdown("### Match visualizations/diagnostics")

    other_keys = [
        key for key in VIEW_KEYS if key not in ("zoom_template", "zoom_focus")
    ]
    with gr.Row():
        for key in other_keys[:4]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                value=DEFAULT_TEMPLATE_PREVIEW if key == "template_color" else None,
                interactive=False,
                height=260,
            )
            image_components[key] = comp

    with gr.Row():
        for key in other_keys[4:]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                interactive=False,
                height=260,
            )
            image_components[key] = comp

    with gr.Row():
        prev_button = gr.Button("⬅️ Previous match")
        next_button = gr.Button("Next match ➡️")
        match_summary = gr.Markdown("Run the matcher to view detailed plots.")

    match_state = gr.State()
    match_index = gr.State(0)

    ordered_components = [image_components[key] for key in VIEW_KEYS]

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

    solve_button.click(
        fn=solve_puzzle,
        inputs=[piece_input, template_selector, auto_align_checkbox, template_rotation],
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
