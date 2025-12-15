"""HuggingFace Gradio interface for the jigsaw puzzle solver"""
import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Set up Django
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jigsaw_project.settings')

import django
django.setup()

from puzzle.models import PuzzleTemplate
from puzzle.matcher import find_piece_in_template, highlight_position
from django.core.files.uploadedfile import SimpleUploadedFile


def initialize_demo_data():
    """Initialize demo puzzle template if none exists"""
    if PuzzleTemplate.objects.count() == 0:
        # Create sample template
        from create_sample_images import create_puzzle_template
        template_img = create_puzzle_template()
        
        # Save to temp file
        temp_path = "/tmp/sample_puzzle.png"
        template_img.save(temp_path)
        
        # Create database entry
        with open(temp_path, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Demo Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        return template
    return PuzzleTemplate.objects.first()


def solve_puzzle(piece_image):
    """Find where the puzzle piece fits in the template"""
    if piece_image is None:
        return None, "Please upload a puzzle piece image"
    
    # Get the template
    template = PuzzleTemplate.objects.first()
    if template is None:
        return None, "No puzzle template available"
    
    try:
        # Save the piece image temporarily
        piece_path = "/tmp/temp_piece.png"
        piece_img = Image.fromarray(piece_image)
        piece_img.save(piece_path)
        
        # Find the piece location
        x, y, confidence = find_piece_in_template(
            piece_path,
            template.template_image.path
        )
        
        # Generate highlighted template
        highlighted_img = highlight_position(template.template_image.path, x, y)
        
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
    """Get the current template image"""
    template = PuzzleTemplate.objects.first()
    if template:
        img = Image.open(template.template_image.path)
        return np.array(img)
    return None


# Initialize demo data
initialize_demo_data()

# Create Gradio interface
with gr.Blocks(title="🧩 Jigsaw Puzzle Solver", theme=gr.themes.Soft()) as demo:
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
    demo.launch()
