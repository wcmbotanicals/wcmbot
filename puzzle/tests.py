"""Unit and integration tests for the puzzle app"""
import pytest
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from PIL import Image
import io
import os

from .models import PuzzleTemplate, PuzzlePiece
from .matcher import find_piece_in_template, highlight_position


@pytest.mark.django_db
class TestPuzzleModels:
    """Test puzzle models"""
    
    def test_create_puzzle_template(self, sample_template_image):
        """Test creating a puzzle template"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Test Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        assert template.name == "Test Puzzle"
        assert template.template_image is not None
        assert PuzzleTemplate.objects.count() == 1
    
    def test_create_puzzle_piece(self, sample_template_image, sample_piece_image):
        """Test creating a puzzle piece"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Test Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        with open(sample_piece_image, 'rb') as f:
            piece = PuzzlePiece.objects.create(
                template=template,
                piece_image=SimpleUploadedFile("piece.png", f.read(), content_type="image/png"),
                matched_x=100,
                matched_y=150,
                confidence_score=0.95
            )
        
        assert piece.template == template
        assert piece.matched_x == 100
        assert piece.matched_y == 150
        assert piece.confidence_score == 0.95
        assert template.pieces.count() == 1
    
    def test_puzzle_template_str(self, sample_template_image):
        """Test string representation of PuzzleTemplate"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="My Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        assert str(template) == "My Puzzle"


@pytest.mark.django_db
class TestPuzzleViews:
    """Test puzzle views"""
    
    def test_index_view(self):
        """Test the index view"""
        client = Client()
        response = client.get(reverse('puzzle:index'))
        
        assert response.status_code == 200
        assert b'Jigsaw Puzzle Solver' in response.content
    
    def test_index_with_templates(self, sample_template_image):
        """Test index view with templates"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Test Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        client = Client()
        response = client.get(reverse('puzzle:index'))
        
        assert response.status_code == 200
        assert b'Test Puzzle' in response.content
    
    def test_puzzle_detail_view(self, sample_template_image):
        """Test the puzzle detail view"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Test Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        client = Client()
        response = client.get(reverse('puzzle:detail', args=[template.id]))
        
        assert response.status_code == 200
        assert b'Test Puzzle' in response.content
        assert b'Upload Puzzle Piece' in response.content
    
    def test_upload_piece_view(self, sample_template_image, sample_piece_image):
        """Test uploading a puzzle piece"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Test Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        client = Client()
        
        with open(sample_piece_image, 'rb') as f:
            response = client.post(
                reverse('puzzle:upload', args=[template.id]),
                {'piece_image': f},
                format='multipart'
            )
        
        assert response.status_code == 200
        data = response.json()
        assert 'success' in data or 'error' in data
        
        if data.get('success'):
            assert 'x' in data
            assert 'y' in data
            assert 'confidence' in data
            assert PuzzlePiece.objects.count() == 1
    
    def test_upload_piece_missing_image(self, sample_template_image):
        """Test uploading without an image"""
        with open(sample_template_image, 'rb') as f:
            template = PuzzleTemplate.objects.create(
                name="Test Puzzle",
                template_image=SimpleUploadedFile("template.png", f.read(), content_type="image/png")
            )
        
        client = Client()
        response = client.post(reverse('puzzle:upload', args=[template.id]))
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data


@pytest.mark.unit
class TestMatcher:
    """Test image matching functionality"""
    
    def test_find_piece_in_template(self, sample_template_image, sample_piece_image):
        """Test finding a piece in the template"""
        x, y, confidence = find_piece_in_template(sample_piece_image, sample_template_image)
        
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert x >= 0
        assert y >= 0
    
    def test_highlight_position(self, sample_template_image):
        """Test highlighting a position on the template"""
        highlighted = highlight_position(sample_template_image, 150, 150)
        
        assert highlighted is not None
        assert isinstance(highlighted, Image.Image)
        # Check that the image is not empty
        assert highlighted.size[0] > 0
        assert highlighted.size[1] > 0
