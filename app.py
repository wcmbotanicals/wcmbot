"""Gradio interface for the jigsaw puzzle solver"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import plotly.express as px
from PIL import Image

from matcher import find_piece_in_template, format_match_summary, render_primary_views

# Hardcoded paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "media" / "templates" / "sample_puzzle.png"

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


def _change_match(step: int, payload, current_index: int):
    if payload is None or not getattr(payload, "matches", None):
        return _blank_outputs("Run the matcher once a piece is uploaded.")
    total = len(payload.matches)
    if total == 0:
        return _blank_outputs("No matches available.")
    idx = (current_index or 0) + step
    idx %= total
    views = render_primary_views(payload, idx)
    views["zoom_template"] = make_zoomable_plot(views.get("zoom_template"))
    summary = format_match_summary(payload, idx)
    return _views_to_outputs(views, summary, payload, idx)


def solve_puzzle(piece_path, knobs_x, knobs_y):
    """Run the high-performance matcher and return visualization slices"""
    if not piece_path or not os.path.exists(piece_path):
        return _blank_outputs("Please upload a puzzle piece image.")
    if knobs_x is None or knobs_y is None:
        return _blank_outputs(
            "Please select the tab counts before running the matcher."
        )

    try:
        knobs_x = int(knobs_x)
        knobs_y = int(knobs_y)
    except (TypeError, ValueError):
        return _blank_outputs("Tab counts must be whole numbers.")

    if knobs_x < 0 or knobs_y < 0:
        return _blank_outputs("Tab counts cannot be negative.")

    try:
        payload = find_piece_in_template(
            piece_path, str(TEMPLATE_PATH), knobs_x, knobs_y
        )
        views = render_primary_views(payload, 0)
        views["zoom_template"] = make_zoomable_plot(views.get("zoom_template"))
        summary = format_match_summary(payload, 0)
        return _views_to_outputs(views, summary, payload, 0)
    except Exception as exc:  # pylint: disable=broad-except
        return _blank_outputs(f"Error: {exc}")


def goto_previous_match(state, current_index):
    return _change_match(-1, state, current_index)


def goto_next_match(state, current_index):
    return _change_match(1, state, current_index)


# Check template exists on startup and pre-load the static preview
check_template_exists()
DEFAULT_TEMPLATE_IMAGE = get_template_image()
DEFAULT_TEMPLATE_PLOT = make_zoomable_plot(DEFAULT_TEMPLATE_IMAGE)

# Create Gradio interface
app_theme = gr.themes.Soft()
with gr.Blocks(title="🧩 Jigsaw Puzzle Solver") as demo:
    gr.Markdown(
        """
    # 🧩 Jigsaw Puzzle Solver

    Upload a puzzle piece and view the same diagnostic plots that the offline
    pipeline renders. Navigate across the top matches to inspect alternatives.
    """
    )

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
            with gr.Row():
                knobs_x_input = gr.Number(
                    label="Horizontal tabs",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=2,
                )
                knobs_y_input = gr.Number(
                    label="Vertical tabs",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=2,
                )
            gr.Markdown(
                "Tabs are the protruding connectors on each side of the piece. "
                "Set how many tabs this piece has horizontally and vertically."
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
            gr.Markdown("Click the image to open the zoom/pan viewer.")

    gr.Markdown("### Match visualizations")

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
        inputs=[piece_input, knobs_x_input, knobs_y_input],
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
