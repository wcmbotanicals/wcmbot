"""Create sample puzzle template and piece images for testing"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add project to path
sys.path.insert(0, '/home/runner/work/wcmbot/wcmbot')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jigsaw_project.settings')

import django
django.setup()

from puzzle.models import PuzzleTemplate


def create_puzzle_template(width=800, height=600):
    """Create a sample jigsaw puzzle template with pieces outlined"""
    # Create image with white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a colorful grid pattern
    piece_width = 100
    piece_height = 100
    
    colors = [
        (255, 200, 200),  # Light red
        (200, 255, 200),  # Light green
        (200, 200, 255),  # Light blue
        (255, 255, 200),  # Light yellow
        (255, 200, 255),  # Light magenta
        (200, 255, 255),  # Light cyan
    ]
    
    # Fill pieces with colors
    for row in range(0, height, piece_height):
        for col in range(0, width, piece_width):
            color_idx = ((row // piece_height) + (col // piece_width)) % len(colors)
            draw.rectangle(
                [col, row, col + piece_width, row + piece_height],
                fill=colors[color_idx],
                outline='black',
                width=3
            )
            
            # Add piece number
            piece_num = (row // piece_height) * (width // piece_width) + (col // piece_width) + 1
            text = str(piece_num)
            # Use default font
            bbox = draw.textbbox((0, 0), text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = col + (piece_width - text_width) // 2
            text_y = row + (piece_height - text_height) // 2
            draw.text((text_x, text_y), text, fill='black')
    
    return img


def create_puzzle_piece(template_img, x, y, width=100, height=100):
    """Extract a piece from the template"""
    # Add some padding
    piece = template_img.crop((x, y, x + width, y + height))
    return piece


def main():
    print("Creating sample puzzle template...")
    template_img = create_puzzle_template()
    template_path = '/home/runner/work/wcmbot/wcmbot/media/templates/sample_puzzle.png'
    template_img.save(template_path)
    print(f"Saved template to {template_path}")
    
    # Create database entry
    template, created = PuzzleTemplate.objects.get_or_create(
        name="Sample Puzzle",
        defaults={'template_image': 'templates/sample_puzzle.png'}
    )
    
    if not created:
        template.template_image = 'templates/sample_puzzle.png'
        template.save()
    
    print(f"Created template in database: {template.name} (ID: {template.id})")
    
    # Create a few sample pieces
    print("\nCreating sample pieces...")
    pieces_info = [
        (200, 100, "piece_1.png"),  # Piece in row 1, col 2
        (400, 300, "piece_2.png"),  # Piece in row 3, col 4
        (0, 0, "piece_3.png"),      # Top-left corner piece
    ]
    
    for x, y, filename in pieces_info:
        piece_img = create_puzzle_piece(template_img, x, y)
        piece_path = f'/home/runner/work/wcmbot/wcmbot/media/pieces/{filename}'
        piece_img.save(piece_path)
        print(f"Saved piece to {piece_path}")
    
    print("\nDone! Template and sample pieces created.")
    print(f"Template ID: {template.id}")


if __name__ == "__main__":
    main()
