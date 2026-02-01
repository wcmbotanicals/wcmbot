"""Tests for GPU-aware multipiece segmentation workflow"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def sample_multipiece_image_path():
    """Create a simple test image file"""
    try:
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (400, 200), (255, 255, 255))
            img.save(tmp.name)
            yield tmp.name
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def sample_regions():
    """Sample regions for multipiece detection"""
    return [
        {"bbox": (10, 10, 80, 80), "contour": np.array([[10, 10]]), "area": 6400},
        {"bbox": (110, 10, 80, 80), "contour": np.array([[110, 10]]), "area": 6400},
    ]


@pytest.fixture
def mock_template_spec():
    """Mock template spec"""
    spec = MagicMock()
    spec.default_rotation = 0
    spec.rows = 28
    spec.cols = 36
    spec.crop_x = 0
    spec.crop_y = 0
    spec.matcher_overrides = {}
    return spec


class TestMultipieceGPUWorkflow:
    """Tests for GPU-aware multipiece segmentation workflow without actual GPU/rembg"""

    def test_cpu_workflow_logic(self, sample_regions, mock_template_spec):
        """Test CPU workflow uses AI for matching when segmentation_mode is ai"""
        # Import here to delay gradio import until after mocking
        import sys

        # Mock gradio before importing app
        sys.modules["gradio"] = MagicMock()

        # Now we can safely test the logic
        from wcmbot.solving import build_matcher_config_for_template

        # Simulate CPU workflow logic
        # CPU workflow: template default for split, AI for matching if requested
        split_config = build_matcher_config_for_template(mock_template_spec)
        match_overrides = {"mask_mode": "ai"}
        match_config = build_matcher_config_for_template(
            mock_template_spec, match_overrides
        )

        # Verify split uses default (no AI override)
        assert split_config.mask_mode != "ai" or split_config.mask_mode is None

        # Verify match uses AI
        assert match_config.mask_mode == "ai"

    def test_gpu_workflow_logic(self, sample_regions, mock_template_spec):
        """Test GPU workflow uses AI for split, white-bg masking for matching"""
        from wcmbot.solving import build_matcher_config_for_template

        # Simulate GPU workflow logic
        # GPU workflow: AI for split, template default for matching
        split_config = build_matcher_config_for_template(
            mock_template_spec, {"mask_mode": "ai"}
        )
        match_config = build_matcher_config_for_template(
            mock_template_spec, {"mask_mode": "white_bg"}
        )

        # Verify split uses AI
        assert split_config.mask_mode == "ai"

        # Verify match is non-AI and uses white-bg masking
        assert match_config.mask_mode == "white_bg"

    def test_can_rembg_use_gpu_returns_bool(self):
        """Test that can_rembg_use_gpu returns a boolean"""
        from wcmbot.matcher import can_rembg_use_gpu

        # Should return a boolean without errors
        result = can_rembg_use_gpu()
        assert isinstance(result, bool)

    def test_can_rembg_use_gpu_detects_cpu_when_no_gpu(self):
        """Test that can_rembg_use_gpu returns False when no GPU providers available"""
        with patch("wcmbot.matcher._detect_rembg_device_and_providers") as mock_detect:
            mock_detect.return_value = ("cpu", ["CPUExecutionProvider"])

            from wcmbot.matcher import can_rembg_use_gpu

            result = can_rembg_use_gpu()
            assert result is False

    def test_can_rembg_use_gpu_detects_gpu_when_available(self):
        """Test that can_rembg_use_gpu returns True when GPU providers available"""
        with patch("wcmbot.matcher._detect_rembg_device_and_providers") as mock_detect:
            mock_detect.return_value = (
                "gpu",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            from wcmbot.matcher import can_rembg_use_gpu

            result = can_rembg_use_gpu()
            assert result is True

    def test_background_removal_composite_logic(self):
        """Test the split-mask-based white background composite logic"""

        grid_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        split_mask01 = np.zeros((100, 100), dtype=np.uint8)
        split_mask01[10:90, 20:80] = 1

        mask01 = (split_mask01 > 0).astype(np.uint8)
        alpha = mask01.astype(np.float32)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        white_bg = np.full_like(grid_bgr, 255, dtype=np.uint8)
        result = (
            grid_bgr.astype(np.float32) * alpha_3ch
            + white_bg.astype(np.float32) * (1 - alpha_3ch)
        ).astype(np.uint8)

        assert result.shape == grid_bgr.shape
        # Foreground region should be preserved.
        assert np.all(result[10:90, 20:80] == 128)
        # Background region should be white.
        assert np.all(result[0:10, :, :] == 255)

    def test_default_mode_uses_default_configs(self, mock_template_spec):
        """Test that default segmentation mode doesn't add AI overrides"""
        from wcmbot.solving import build_matcher_config_for_template

        # Neither workflow should add AI override for default mode
        split_config = build_matcher_config_for_template(mock_template_spec)
        match_config = build_matcher_config_for_template(mock_template_spec)

        # Both should use template default (not AI)
        assert split_config.mask_mode != "ai" or split_config.mask_mode is None
        assert match_config.mask_mode != "ai" or match_config.mask_mode is None
