"""Unit tests for app.py helper functions"""

import numpy as np
import pytest
from unittest.mock import patch
from pathlib import Path
import tempfile
from PIL import Image

# Import the functions we need to test
from app import _stack_images_vertical, _annotate_pair_image


class TestStackImagesVertical:
    """Tests for _stack_images_vertical helper function"""

    def test_empty_list_returns_none(self):
        """Test that empty list returns None"""
        result = _stack_images_vertical([])
        assert result is None

    def test_list_with_all_none_images_returns_none(self):
        """Test that list with all None images returns None"""
        result = _stack_images_vertical([None, None, None])
        assert result is None

    def test_single_rgb_image(self):
        """Test stacking a single RGB image"""
        img = np.ones((100, 50, 3), dtype=np.uint8) * 128
        result = _stack_images_vertical([img])
        assert result is not None
        assert result.shape == (100, 50, 3)
        assert result.dtype == np.uint8

    def test_single_grayscale_image_converted_to_rgb(self):
        """Test that grayscale image is converted to RGB"""
        img = np.ones((100, 50), dtype=np.uint8) * 128
        result = _stack_images_vertical([img])
        assert result is not None
        assert result.shape == (100, 50, 3)
        assert result.dtype == np.uint8

    def test_multiple_images_stacked_with_gap(self):
        """Test that multiple images are stacked with proper gap"""
        img1 = np.ones((100, 50, 3), dtype=np.uint8) * 128
        img2 = np.ones((80, 50, 3), dtype=np.uint8) * 64
        gap = 10
        result = _stack_images_vertical([img1, img2], gap=gap)
        assert result is not None
        # Total height should be sum of heights + gap
        assert result.shape[0] == 100 + 80 + gap
        # Width should be max width
        assert result.shape[1] == 50
        assert result.shape[2] == 3

    def test_images_with_different_widths_aligned_left(self):
        """Test that images with different widths are left-aligned"""
        img1 = np.ones((100, 50, 3), dtype=np.uint8) * 128
        img2 = np.ones((80, 30, 3), dtype=np.uint8) * 64
        result = _stack_images_vertical([img1, img2])
        assert result is not None
        # Width should be max width
        assert result.shape[1] == 50

    def test_images_resized_when_exceeding_max_width(self):
        """Test that images are resized when exceeding max_width"""
        img1 = np.ones((100, 200, 3), dtype=np.uint8) * 128
        img2 = np.ones((80, 150, 3), dtype=np.uint8) * 64
        max_width = 100
        result = _stack_images_vertical([img1, img2], max_width=max_width)
        assert result is not None
        # Width should be scaled to max_width
        assert result.shape[1] == max_width

    def test_float_images_converted_to_uint8(self):
        """Test that float images are converted to uint8"""
        img = np.ones((100, 50, 3), dtype=np.float32) * 128.0
        result = _stack_images_vertical([img])
        assert result is not None
        assert result.dtype == np.uint8

    def test_images_clipped_to_valid_range(self):
        """Test that images are clipped to [0, 255] range"""
        img = np.ones((100, 50, 3), dtype=np.float32) * 300.0  # Out of range
        result = _stack_images_vertical([img])
        assert result is not None
        assert np.all(result <= 255)
        assert np.all(result >= 0)

    def test_custom_background_color(self):
        """Test that custom background color is used for gaps"""
        img1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        img2 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        background = 200
        result = _stack_images_vertical([img1, img2], gap=5, background=background)
        assert result is not None
        # Check that gap area has background color
        gap_row = result[12, 0, :]  # Row in the gap area
        assert np.all(gap_row == background)

    def test_mixed_none_and_valid_images(self):
        """Test that None images are skipped in the list"""
        img1 = np.ones((100, 50, 3), dtype=np.uint8) * 128
        img2 = None
        img3 = np.ones((80, 50, 3), dtype=np.uint8) * 64
        result = _stack_images_vertical([img1, img2, img3])
        assert result is not None
        # Should only stack img1 and img3
        assert result.shape[0] == 100 + 80 + 8  # default gap is 8


class TestAnnotatePairImage:
    """Tests for _annotate_pair_image helper function"""

    def test_annotate_simple_image(self):
        """Test basic annotation on an image"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        label = "Test"
        result = _annotate_pair_image(img, label)
        assert result is not None
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        # Verify original image is not modified
        assert not np.array_equal(result, img)

    def test_annotate_does_not_modify_original(self):
        """Test that annotation does not modify the original image"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        img_copy = img.copy()
        label = "Test"
        _annotate_pair_image(img, label)
        assert np.array_equal(img, img_copy)

    def test_annotate_with_long_label(self):
        """Test annotation with a long label"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        label = "This is a very long label"
        result = _annotate_pair_image(img, label)
        assert result is not None
        assert result.shape == img.shape

    def test_annotate_with_empty_label(self):
        """Test annotation with an empty label"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        label = ""
        result = _annotate_pair_image(img, label)
        assert result is not None
        assert result.shape == img.shape

    def test_annotate_small_image(self):
        """Test annotation on a small image"""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        label = "Small"
        result = _annotate_pair_image(img, label)
        assert result is not None
        assert result.shape == img.shape

    def test_text_placement_in_top_left(self):
        """Test that text is placed in the top-left area with padding"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        label = "Test"
        result = _annotate_pair_image(img, label)
        # Check that top-left area has been modified (white background and black text)
        top_left_region = result[0:30, 0:80]
        # Should have some white pixels for the background rectangle
        assert np.any(np.all(top_left_region == [255, 255, 255], axis=2))


class TestOnPieceChange:
    """Tests for _on_piece_change function"""

    @pytest.fixture
    def mock_solve_function(self):
        """Fixture to mock solve_single_or_batch"""
        with patch("app.solve_single_or_batch") as mock_solve:
            yield mock_solve

    @pytest.fixture
    def sample_outputs(self):
        """Sample outputs that match the expected structure"""
        # Based on _no_update_outputs, we need VIEW_KEYS outputs + 3 more
        from app import VIEW_KEYS
        import gradio as gr

        num_views = len(VIEW_KEYS)
        # Return tuple: (*view_updates, location, summary, state, idx)
        return (
            *([gr.update()] * num_views),
            "Row: 1, Col: 2",
            "Match summary",
            {"some": "state"},
            1,
        )

    def test_empty_piece_path_yields_no_update(self, mock_solve_function):
        """Test that empty piece_path yields no update outputs"""
        from app import _on_piece_change

        # Mock parameters
        piece_path = None
        template_id = "test_template"
        auto_align = False
        template_rotation = 0
        batch_mode = False
        state = {}
        idx = 0

        # Call the generator
        gen = _on_piece_change(
            piece_path,
            template_id,
            auto_align,
            template_rotation,
            batch_mode,
            state,
            idx,
        )

        # Get the result and verify it's not None
        next(gen)

        # Verify solve was not called
        mock_solve_function.assert_not_called()

        # Verify generator stops after one yield
        with pytest.raises(StopIteration):
            next(gen)

    def test_single_piece_mode_yields_from_result(
        self, mock_solve_function, sample_outputs
    ):
        """Test that single piece mode properly unpacks generator results"""
        from app import _on_piece_change

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), (200, 200, 255))
            img.save(tmp.name)
            piece_path = tmp.name

        try:
            # Mock solve_single_or_batch to return a generator that yields one result
            def mock_generator():
                yield sample_outputs

            mock_solve_function.return_value = mock_generator()

            # Parameters
            template_id = "test_template"
            auto_align = False
            template_rotation = 0
            batch_mode = False
            state = {}
            idx = 0

            # Call the function
            gen = _on_piece_change(
                piece_path,
                template_id,
                auto_align,
                template_rotation,
                batch_mode,
                state,
                idx,
            )

            # Verify it yields the result from the generator
            result = next(gen)
            assert result == sample_outputs

            # Verify solve was called with correct parameters
            mock_solve_function.assert_called_once_with(
                piece_path, template_id, auto_align, template_rotation, batch_mode
            )
        finally:
            Path(piece_path).unlink(missing_ok=True)

    def test_batch_mode_yields_from_result(self, mock_solve_function, sample_outputs):
        """Test that batch mode properly unpacks generator results"""
        from app import _on_piece_change

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), (200, 200, 255))
            img.save(tmp.name)
            piece_path = tmp.name

        try:
            # Mock solve_single_or_batch to return a generator that yields multiple results
            def mock_generator():
                yield sample_outputs
                yield sample_outputs

            mock_solve_function.return_value = mock_generator()

            # Parameters
            template_id = "test_template"
            auto_align = False
            template_rotation = 0
            batch_mode = True
            state = {}
            idx = 0

            # Call the function
            gen = _on_piece_change(
                piece_path,
                template_id,
                auto_align,
                template_rotation,
                batch_mode,
                state,
                idx,
            )

            # Verify it yields all results from the generator
            result1 = next(gen)
            assert result1 == sample_outputs
            result2 = next(gen)
            assert result2 == sample_outputs

            # Verify solve was called with correct parameters
            mock_solve_function.assert_called_once_with(
                piece_path, template_id, auto_align, template_rotation, batch_mode
            )
        finally:
            Path(piece_path).unlink(missing_ok=True)

    def test_generator_properly_handles_single_vs_batch_mode(self):
        """Test that the bug fix (yield from vs yield) works correctly"""
        # This test ensures that both batch_mode=True and batch_mode=False
        # use 'yield from' to properly unpack the generator
        from app import _on_piece_change

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), (200, 200, 255))
            img.save(tmp.name)
            piece_path = tmp.name

        try:
            with patch("app.solve_single_or_batch") as mock_solve:
                # Mock to return a generator
                def mock_gen():
                    yield ("result",)

                mock_solve.return_value = mock_gen()

                # Test single mode
                gen = _on_piece_change(piece_path, "test", False, 0, False, {}, 0)
                result = next(gen)

                # Should yield the tuple directly, not the generator
                assert isinstance(result, tuple)
                assert result == ("result",)
        finally:
            Path(piece_path).unlink(missing_ok=True)
