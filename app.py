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

from matcher import (
    find_piece_in_template,
    format_match_summary,
    preload_template_cache,
    render_primary_views,
)
from version import __version__

# Hardcoded paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "media" / "templates" / "sample_puzzle.png"
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


def check_template_exists():
    """Check that the required template file exists"""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Template file not found at {TEMPLATE_PATH}. "
            "Please ensure the puzzle template image exists before running the app."
        )


def get_template_image() -> Optional[np.ndarray]:
    """Get the template image"""
    if TEMPLATE_PATH.exists():
        return np.array(Image.open(TEMPLATE_PATH))
    return None


def _views_to_outputs(
    views: Dict[str, Optional[np.ndarray]],
    summary: str,
    state,
    idx: int,
):
    ordered = [views.get(key) for key in VIEW_KEYS]
    return (*ordered, summary, state, idx)


def _blank_outputs(message: str):
    blank_views = {key: None for key in VIEW_KEYS}
    blank_views["template_color"] = DEFAULT_TEMPLATE_IMAGE
    blank_views["zoom_template"] = DEFAULT_TEMPLATE_PLOT
    return _views_to_outputs(blank_views, message, None, 0)


def _render_match_payload(payload, idx: int):
    views = render_primary_views(payload, idx)
    views["zoom_template"] = make_zoomable_plot(views.get("zoom_template"))
    summary = format_match_summary(payload, idx)
    return _views_to_outputs(views, summary, payload, idx)


def _change_match(step: int, payload, current_index: int):
    if payload is None or not getattr(payload, "matches", None):
        return _blank_outputs("Run the matcher once a piece is uploaded.")
    total = len(payload.matches)
    if total == 0:
        return _blank_outputs("No matches available.")
    idx = (current_index or 0) + step
    idx %= total
    return _render_match_payload(payload, idx)


def solve_puzzle(piece_path):
    """Run the high-performance matcher and return visualization slices"""
    if not piece_path or not os.path.exists(piece_path):
        return _blank_outputs("Please upload a puzzle piece image.")

    try:
        payload = find_piece_in_template(
            piece_path,
            str(TEMPLATE_PATH),
            knobs_x=None,
            knobs_y=None,
            auto_align=True,
            infer_knobs=True,
        )
        return _render_match_payload(payload, 0)
    except Exception as exc:  # pylint: disable=broad-except
        return _blank_outputs(f"Error: {exc}")


def goto_previous_match(state, current_index):
    return _change_match(-1, state, current_index)


def goto_next_match(state, current_index):
    return _change_match(1, state, current_index)


# Check template exists on startup and pre-load the static preview
check_template_exists()
preload_template_cache(str(TEMPLATE_PATH))
DEFAULT_TEMPLATE_IMAGE = get_template_image()
DEFAULT_TEMPLATE_PLOT = make_zoomable_plot(DEFAULT_TEMPLATE_IMAGE)

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
      Small tilts are corrected using the piece mask (rotations of multiples of 90° are evaluated).
    
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
            solve_button = gr.Button(
                "🔍 Find Piece Location", variant="primary", size="lg"
            )
        with gr.Column(scale=1):
            gr.Markdown("### Best match (template view)")
            image_components = {}
            image_components["zoom_template"] = gr.Plot(
                value=DEFAULT_TEMPLATE_PLOT,
                elem_id="primary-template-view",
            )
            gr.Markdown("Use the controls to zoom and pan the image.")

    gr.Markdown("### Match visualizations/diagnostics")

    other_keys = [key for key in VIEW_KEYS if key != "zoom_template"]
    with gr.Row():
        for key in other_keys[:4]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                value=DEFAULT_TEMPLATE_IMAGE if key == "template_color" else None,
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

    solve_button.click(
        fn=solve_puzzle,
        inputs=[piece_input],
        outputs=[*ordered_components, match_summary, match_state, match_index],
    )
    prev_button.click(
        fn=goto_previous_match,
        inputs=[match_state, match_index],
        outputs=[*ordered_components, match_summary, match_state, match_index],
    )
    next_button.click(
        fn=goto_next_match,
        inputs=[match_state, match_index],
        outputs=[*ordered_components, match_summary, match_state, match_index],
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
