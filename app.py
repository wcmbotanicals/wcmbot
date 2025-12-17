"""Gradio interface for the jigsaw puzzle solver"""
import os
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr

from matcher import find_piece_in_template, highlight_position

# Hardcoded paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "media" / "templates" / "sample_puzzle.png"


def check_template_exists():
    """Check that the required template file exists"""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Template file not found at {TEMPLATE_PATH}. "
            "Please ensure the puzzle template image exists before running the app."
        )


def solve_puzzle(piece_image):
    """Find where the puzzle piece fits in the template"""
    if piece_image is None:
        return None, "Please upload a puzzle piece image"
    
    if not TEMPLATE_PATH.exists():
        return None, "No puzzle template available"
    
    try:
        # Save the piece image temporarily
        piece_path = "/tmp/temp_piece.png"
        piece_img = Image.fromarray(piece_image)
        piece_img.save(piece_path)
        
        # Find the piece location
        x, y, confidence = find_piece_in_template(
            piece_path,
            str(TEMPLATE_PATH)
        )
        
        # Generate highlighted template
        highlighted_img = highlight_position(str(TEMPLATE_PATH), x, y)
        
        result_text = f"""
**Match Found!** ✓

- **Position**: ({x}, {y})
- **Confidence**: {confidence * 100:.1f}%

The green circle shows where your piece fits in the puzzle.
"""
        
        return np.array(highlighted_img), result_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_template_image():
    """Get the template image"""
    if TEMPLATE_PATH.exists():
        return np.array(Image.open(TEMPLATE_PATH))
    return None


# Check template exists on startup
check_template_exists()

# Create Gradio interface
app_theme = gr.themes.Soft()
with gr.Blocks(title="🧩 Jigsaw Puzzle Solver") as demo:
    gr.Markdown("""
    # 🧩 Jigsaw Puzzle Solver
    
    Upload a puzzle piece and discover where it fits in the template!
    
    ## How to use:
    1. See the puzzle template on the right
    2. Upload an image of a puzzle piece
    3. Click "Find Piece Location" to see where it fits
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Puzzle Piece")
            piece_input = gr.Image(
                label="Puzzle Piece",
                type="numpy",
                sources=["upload", "clipboard"],
                height=300
            )
            solve_button = gr.Button("🔍 Find Piece Location", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Puzzle Template")
            template_display = gr.Image(
                label="Template",
                type="numpy",
                value=get_template_image(),
                interactive=False,
                height=300
            )
    
    with gr.Row():
        result_display = gr.Image(label="Match Result", type="numpy", height=400)
        result_text = gr.Markdown()
    
    solve_button.click(
        fn=solve_puzzle,
        inputs=[piece_input],
        outputs=[result_display, result_text]
    )
    
    gr.Markdown("""
    ---
    ### About
    This app uses computer vision techniques to match puzzle pieces with their correct positions in a template.
    It employs template matching and feature detection algorithms to find the best match.
    """)

if __name__ == "__main__":
    demo.launch(theme=app_theme)
