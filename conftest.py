"""Pytest configuration and fixtures"""
import pytest
from PIL import Image
import os


@pytest.fixture
def sample_template_image(tmp_path):
    """Create a sample template image for testing"""
    img = Image.new('RGB', (400, 300), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a simple grid
    for i in range(0, 400, 100):
        for j in range(0, 300, 100):
            color = (100 + i // 2, 100 + j // 2, 200)
            draw.rectangle([i, j, i + 100, j + 100], fill=color, outline='black', width=2)
    
    img_path = tmp_path / "template.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_piece_image(tmp_path):
    """Create a sample piece image for testing"""
    img = Image.new('RGB', (100, 100), (150, 150, 200))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 100], outline='black', width=2)
    
    img_path = tmp_path / "piece.png"
    img.save(img_path)
    return img_path
