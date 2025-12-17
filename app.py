"""Gradio interface for the jigsaw puzzle solver"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
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
    summary = format_match_summary(payload, idx)
    return _views_to_outputs(views, summary, payload, idx)


def solve_puzzle(piece_image):
    """Run the high-performance matcher and return visualization slices"""
    if piece_image is None:
        return _blank_outputs("Please upload a puzzle piece image.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            piece_img = Image.fromarray(piece_image.astype(np.uint8))
            piece_img.save(tmp_path)

        payload = find_piece_in_template(tmp_path, str(TEMPLATE_PATH))
        views = render_primary_views(payload, 0)
        summary = format_match_summary(payload, 0)
        return _views_to_outputs(views, summary, payload, 0)
    except Exception as exc:  # pylint: disable=broad-except
        return _blank_outputs(f"Error: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def goto_previous_match(state, current_index):
    return _change_match(-1, state, current_index)


def goto_next_match(state, current_index):
    return _change_match(1, state, current_index)


# Check template exists on startup and pre-load the static preview
check_template_exists()
DEFAULT_TEMPLATE_IMAGE = get_template_image()

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

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Puzzle Piece")
            piece_input = gr.Image(
                label="Puzzle Piece",
                type="numpy",
                sources=["upload", "clipboard"],
                height=300,
            )
            solve_button = gr.Button(
                "🔍 Find Piece Location", variant="primary", size="lg"
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "### Visualizations\n"
                "Each pane below mirrors the first debug figure from the CLI tool."
            )

    gr.Markdown("### Match visualizations")

    image_components: List[gr.Image] = []
    with gr.Row():
        for key in VIEW_KEYS[:4]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                value=DEFAULT_TEMPLATE_IMAGE if key == "template_color" else None,
                interactive=False,
                height=260,
            )
            image_components.append(comp)

    with gr.Row():
        for key in VIEW_KEYS[4:]:
            comp = gr.Image(
                label=VIEW_LABELS[key],
                type="numpy",
                interactive=False,
                height=260,
            )
            image_components.append(comp)

    with gr.Row():
        prev_button = gr.Button("⬅️ Previous match")
        next_button = gr.Button("Next match ➡️")
        match_summary = gr.Markdown("Run the matcher to view detailed plots.")

    match_state = gr.State()
    match_index = gr.State(0)

    solve_button.click(
        fn=solve_puzzle,
        inputs=[piece_input],
        outputs=[*image_components, match_summary, match_state, match_index],
    )
    prev_button.click(
        fn=goto_previous_match,
        inputs=[match_state, match_index],
        outputs=[*image_components, match_summary, match_state, match_index],
    )
    next_button.click(
        fn=goto_next_match,
        inputs=[match_state, match_index],
        outputs=[*image_components, match_summary, match_state, match_index],
    )

    gr.Markdown(
        """
    ---
    ### About
    This app now uses the multi-scale binary matcher from `1.py` and exposes all
    diagnostic plots directly in the UI. Use the navigation buttons to inspect
    alternative placements when multiple candidates score highly.
    """
    )

if __name__ == "__main__":
    demo.launch(theme=app_theme)
