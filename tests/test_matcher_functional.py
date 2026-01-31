import os
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

from wcmbot.matcher import (
    COLS,
    ROWS,
    _apply_background_cluster_mask,
    _apply_template_cluster_mask,
    _background_bgr,
    _background_color_clusters,
    _background_distance_from_border,
    _fill_mask_holes,
    _mask_by_gradient,
    _recover_piece_edges,
    _smooth_mask_edges,
    _smooth_piece_contour,
    _template_color_clusters,
    build_matcher_config,
    find_piece_in_template,
)
from wcmbot.multipiece import find_multipiece_region_dicts
from wcmbot.template_settings import load_template_registry

HERE = os.path.dirname(__file__)
TEMPLATE_REGISTRY = load_template_registry()
SAMPLE_SPEC = TEMPLATE_REGISTRY.get("sample_puzzle")
SAMPLE_TEMPLATE_PATH = os.fspath(SAMPLE_SPEC.template_path)
SAMPLE_PIECES_DIR = (
    os.fspath(SAMPLE_SPEC.piece_dirs[0])
    if SAMPLE_SPEC.piece_dirs
    else os.path.join(HERE, "..", "media", "pieces", "sample_puzzle")
)
GRASS_SPEC = TEMPLATE_REGISTRY.get("grass_puzzle")
GRASS_TEMPLATE_PATH = os.fspath(GRASS_SPEC.template_path)
GRASS_PIECES_DIR = (
    os.fspath(GRASS_SPEC.piece_dirs[0])
    if GRASS_SPEC.piece_dirs
    else os.path.join(HERE, "..", "media", "pieces", "grass_puzzle")
)

BASE_CASES = [
    ("piece_1.jpg", 0, 0, 90, 7, 25),
    ("piece_2.jpg", 2, 2, 0, 11, 20),
    ("piece_3.jpg", 2, 2, 90, 15, 12),
    ("piece_4.jpg", 1, 1, 0, 13, 27),
    ("piece_5.jpg", 0, 2, 180, 11, 6),
    ("piece_6.jpg", 1, 2, 270, 18, 20),
    ("piece_7.jpg", 2, 0, 270, 7, 13),
    ("piece_8.jpg", 0, 2, 270, 18, 24),
    ("piece_9.jpg", 2, 0, 90, 18, 25),
    ("piece_10.jpg", 0, 2, 90, 2, 5),
    ("piece_11.jpg", 1, 1, 180, 27, 8),
    ("piece_12.jpg", 1, 1, 0, 18, 21),
    ("piece_13.jpg", 2, 0, 90, 2, 4),
    ("piece_14.jpg", 0, 2, 180, 25, 10),
]
GRASS_CASES = [
    ("grass_piece_1.jpg", 0, 2, 270, 21, 25),
    ("grass_piece_2.jpg", 2, 2, 0, 6, 3),
    ("grass_piece_3.jpg", 1, 2, 0, 11, 24),
    ("grass_piece_4.jpg", 0, 0, 0, 28, 13),
    ("grass_piece_5.jpg", 1, 1, 0, 27, 13),
    ("grass_piece_6.jpg", 0, 2, 0, 19, 12),
]
EXPECTED_LOCATION_CASES = [("sample_puzzle", *case) for case in BASE_CASES] + [
    ("grass_puzzle", *case) for case in GRASS_CASES
]

MANY_PIECES_EXPECTED = {
    1: (11, 10),
    2: (16, 22),
    3: (15, 13),
    4: (23, 33),
    5: (12, 10),
    6: (25, 13),
    7: (17, 32),
    8: (16, 17),
    9: (24, 33),
    10: (14, 23),
    11: (10, 31),
    12: (16, 33),
    13: (12, 4),
    14: (22, 11),
    15: (16, 19),
    16: (27, 27),
    17: (25, 22),
    18: (10, 19),
    19: (16, 30),
    20: (15, 19),
    21: (12, 7),
    22: (13, 27),
    23: (13, 14),
    24: (14, 25),
    25: (12, 3),
}


DIFFICULT_MULTIPIECE_EXPECTED = {
    1: (17, 21),
    2: (4, 17),
    3: (12, 21),
    4: (3, 8),
    6: (17, 2),
    7: (22, 7),
    8: (2, 8),
    9: (3, 32),
    10: (8, 22),
    11: (8, 5),
    12: (19, 18),
    13: (26, 8),
    14: (21, 5),
    15: (26, 4),
    16: (25, 34),
    17: (25, 21),
    18: (6, 5),
    19: (9, 5),
    20: (9, 21),
    21: (26, 20),
    22: (21, 2),
    24: (9, 35),
    25: (18, 8),
    26: (10, 5),
    27: (5, 5),
    28: (26, 9),
}


MANY_PIECES_EXPECTED_KNOBS = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 1),
    4: (1, 1),
    5: (1, 1),
    6: (1, 1),
    7: (1, 2),
    8: (1, 1),
    9: (1, 1),
    10: (1, 1),
    11: (1, 1),
    12: (1, 1),
    13: (1, 2),
    14: (1, 1),
    15: (1, 1),
    16: (1, 1),
    17: (1, 1),
    18: (1, 1),
    19: (1, 1),
    20: (1, 1),
    21: (1, 1),
    22: (1, 1),
    23: (1, 1),
    24: (1, 1),
    25: (1, 1),
}


ROTATION_SWEEP_DEGREES = [-15, -10, -5, -2.5, 0, 2.5, 5, 10, 15]
TEMPLATE_ROTATION_CASE = ("piece_5.jpg", 0, 2, 180, 11, 6)


def _rotate_piece_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    bg = _background_bgr(img)
    return cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )


def _write_rotated_piece(
    tmp_path: os.PathLike, piece_filename: str, angle_deg: float
) -> str:
    piece_path = os.path.join(SAMPLE_PIECES_DIR, piece_filename)
    img = cv2.imread(piece_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load piece image: {piece_path}")
    rotated = _rotate_piece_image(img, angle_deg)
    stem, _ = os.path.splitext(piece_filename)
    safe_angle = str(angle_deg).replace(".", "p")
    out_path = os.path.join(tmp_path, f"{stem}_rot_{safe_angle}deg.png")
    cv2.imwrite(out_path, rotated)
    return out_path


@pytest.mark.e2e
@pytest.mark.parametrize(
    "template_id,piece_filename,knobs_x,knobs_y,exp_rot,exp_row,exp_col",
    EXPECTED_LOCATION_CASES,
)
def test_find_piece_expected_location(
    template_id, piece_filename, knobs_x, knobs_y, exp_rot, exp_row, exp_col
):
    spec = TEMPLATE_REGISTRY.get(template_id)
    template_path = os.fspath(spec.template_path)
    if template_id == "grass_puzzle":
        pieces_dir = GRASS_PIECES_DIR
    else:
        pieces_dir = SAMPLE_PIECES_DIR
    piece_path = os.path.join(pieces_dir, piece_filename)
    matcher_config = build_matcher_config(
        {
            "rows": spec.rows,
            "cols": spec.cols,
            "crop_x": spec.crop_x,
            "crop_y": spec.crop_y,
            **spec.matcher_overrides,
        }
    )

    payload = find_piece_in_template(
        piece_image_path=piece_path,
        template_image_path=template_path,
        knobs_x=None,
        knobs_y=None,
        infer_knobs=True,
        auto_align=True,
        matcher_config=matcher_config,
    )

    assert payload.matches, "No matches returned by matcher"
    top = payload.matches[0]

    assert payload.knobs_inferred, f"knob inference off for {piece_filename}"
    assert payload.knobs_x == knobs_x, f"knobs_x mismatch for {piece_filename}"
    assert payload.knobs_y == knobs_y, f"knobs_y mismatch for {piece_filename}"

    assert top["rot"] == exp_rot, f"rotation mismatch for {piece_filename}"
    assert top["row"] == exp_row, f"row mismatch for {piece_filename}"
    assert top["col"] == exp_col, f"col mismatch for {piece_filename}"


@pytest.mark.e2e
@pytest.mark.skip(reason="slow test")
@pytest.mark.parametrize("extra_deg", ROTATION_SWEEP_DEGREES)
@pytest.mark.parametrize(
    "piece_filename,knobs_x,knobs_y,exp_rot,exp_row,exp_col",
    BASE_CASES,
)
def test_find_piece_expected_location_with_rotation(
    tmp_path,
    piece_filename,
    knobs_x,
    knobs_y,
    exp_rot,
    exp_row,
    exp_col,
    extra_deg,
):
    template_path = SAMPLE_TEMPLATE_PATH
    if extra_deg == 0:
        piece_path = os.path.join(SAMPLE_PIECES_DIR, piece_filename)
    else:
        piece_path = _write_rotated_piece(tmp_path, piece_filename, extra_deg)

    payload = find_piece_in_template(
        piece_image_path=piece_path,
        template_image_path=template_path,
        knobs_x=knobs_x,
        knobs_y=knobs_y,
        auto_align=True,
    )

    assert payload.matches, (
        f"No matches returned for {piece_filename} at {extra_deg}deg"
    )
    top = payload.matches[0]

    if top["row"] != exp_row or top["col"] != exp_col:
        pytest.fail(
            "placement mismatch for "
            f"{piece_filename} at {extra_deg}deg: "
            f"got row={top['row']} col={top['col']} "
            f"(rot={top['rot']} score={top['score']:.3f}), "
            f"expected row={exp_row} col={exp_col}"
        )


@pytest.mark.e2e
@pytest.mark.parametrize("template_rotation", [0, 180])
def test_find_piece_with_template_rotation(template_rotation):
    piece_filename, knobs_x, knobs_y, exp_rot, exp_row, exp_col = TEMPLATE_ROTATION_CASE
    piece_path = os.path.join(SAMPLE_PIECES_DIR, piece_filename)

    payload = find_piece_in_template(
        piece_image_path=piece_path,
        template_image_path=SAMPLE_TEMPLATE_PATH,
        knobs_x=knobs_x,
        knobs_y=knobs_y,
        auto_align=True,
        template_rotation=template_rotation,
    )

    assert payload.matches, "No matches returned by matcher"
    top = payload.matches[0]

    if template_rotation == 180:
        exp_row = ROWS - exp_row + 1
        exp_col = COLS - exp_col + 1
    exp_rot = (exp_rot + template_rotation) % 360

    assert top["rot"] == exp_rot, (
        f"rotation mismatch for {piece_filename} at template rotation "
        f"{template_rotation}deg"
    )
    assert top["row"] == exp_row, (
        f"row mismatch for {piece_filename} at template rotation {template_rotation}deg"
    )
    assert top["col"] == exp_col, (
        f"col mismatch for {piece_filename} at template rotation {template_rotation}deg"
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "grid_filename,expected_count,mask_mode,test_type,minimum_correct,check_knobs",
    [
        ("many_pieces.jpg", 25, None, "many_pieces", 23, True),
        # ("many_pieces.jpg", 25, "ai", "many_pieces", 23, True),
        ("difficult_multipiece.jpg", 28, None, "difficult_multipiece", 22, False),
        # ("difficult_multipiece.jpg", 28, "ai", "difficult_multipiece", 22, False),
    ],
)
def test_multipiece_batch_parameterised(
    grid_filename, expected_count, mask_mode, test_type, minimum_correct, check_knobs
):
    """Test multipiece matching with different configurations and mask modes."""
    spec = TEMPLATE_REGISTRY.get("grass_puzzle")

    if test_type == "many_pieces":
        expected = MANY_PIECES_EXPECTED
    elif test_type == "difficult_multipiece":
        expected = DIFFICULT_MULTIPIECE_EXPECTED
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Build configurations
    base_config = {
        "rows": spec.rows,
        "cols": spec.cols,
        "crop_x": spec.crop_x,
        "crop_y": spec.crop_y,
        **spec.matcher_overrides,
    }

    if mask_mode == "ai":
        split_config = build_matcher_config(base_config)
        match_config = build_matcher_config({**base_config, "mask_mode": "ai"})
    else:
        split_config = build_matcher_config(base_config)
        match_config = split_config

    # Load grid image
    grid_path = os.path.join(GRASS_PIECES_DIR, grid_filename)
    grid_img = Image.open(grid_path).convert("RGB")
    grid_bgr = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2BGR)

    # Find regions
    regions, _ = find_multipiece_region_dicts(grid_bgr, split_config)
    assert len(regions) == expected_count, f"Expected {expected_count} pieces detected"

    # Track results
    placements = {}
    inferred_knobs = {}
    knob_mismatches = []
    mismatches = []
    correct = 0
    template_path = os.fspath(spec.template_path)

    for idx, region in enumerate(regions, start=1):
        x, y, w, h = region["bbox"]
        pad = max(4, int(min(w, h) * 0.06))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(grid_bgr.shape[1], x + w + pad)
        y1 = min(grid_bgr.shape[0], y + h + pad)
        crop = grid_img.crop((x0, y0, x1, y1))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            crop.save(tmp.name)
            crop_path = tmp.name
        try:
            payload = find_piece_in_template(
                piece_image_path=crop_path,
                template_image_path=template_path,
                knobs_x=None,
                knobs_y=None,
                infer_knobs=True,
                auto_align=True,
                template_rotation=spec.default_rotation,
                matcher_config=match_config,
            )
        finally:
            os.unlink(crop_path)

        assert payload.matches, f"No match returned for piece {idx}"

        # Track knob inference if this test type requires it
        if check_knobs:
            assert payload.knobs_inferred, f"knob inference off for piece {idx}"
            inferred_knobs[idx] = (payload.knobs_x, payload.knobs_y)
            exp_knobs_x, exp_knobs_y = MANY_PIECES_EXPECTED_KNOBS[idx]
            if payload.knobs_x != exp_knobs_x or payload.knobs_y != exp_knobs_y:
                knob_mismatches.append(
                    (
                        idx,
                        (payload.knobs_x, payload.knobs_y),
                        (exp_knobs_x, exp_knobs_y),
                    )
                )

        top = payload.matches[0]
        placements[idx] = (top["row"], top["col"])

        # Check correctness based on test type
        expected_idx = expected.get(idx)
        if expected_idx is not None and expected_idx == placements[idx]:
            correct += 1

    assert correct >= minimum_correct, (
        f"Expected at least {minimum_correct} correctly placed pieces, got {correct}: {placements}"
    )

    if check_knobs:
        correct_knobs = len(regions) - len(knob_mismatches)
        assert correct_knobs >= 12, (
            "Knob inference mismatches for some pieces: "
            f"{knob_mismatches}. All inferred: {inferred_knobs}"
        )


# Tests for new mask processing helper functions
class TestMaskHelpers:
    """Test suite for mask processing helper functions."""

    def test_fill_mask_holes_empty_mask(self):
        """Test _fill_mask_holes with an empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _fill_mask_holes(mask)
        assert result.shape == mask.shape
        assert result.sum() == 0

    def test_fill_mask_holes_full_mask(self):
        """Test _fill_mask_holes with a full mask."""
        mask = np.ones((100, 100), dtype=np.uint8)
        result = _fill_mask_holes(mask)
        assert result.shape == mask.shape
        assert result.sum() > 0

    def test_fill_mask_holes_with_holes(self):
        """Test _fill_mask_holes correctly fills interior holes."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Draw a hollow rectangle
        mask[20:80, 20:80] = 1
        mask[40:60, 40:60] = 0
        result = _fill_mask_holes(mask)
        # Result should have the hole filled
        assert result[50, 50] > 0

    def test_smooth_mask_edges_preserves_shape(self):
        """Test _smooth_mask_edges preserves mask shape."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        result = _smooth_mask_edges(mask, kernel_size=7)
        assert result.shape == mask.shape

    def test_smooth_mask_edges_empty_mask(self):
        """Test _smooth_mask_edges with empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _smooth_mask_edges(mask, kernel_size=7)
        assert result.shape == mask.shape
        assert result.sum() == 0

    def test_background_distance_from_border_valid_image(self):
        """Test _background_distance_from_border with a valid image."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add a distinct border
        img[:10, :] = [100, 100, 100]
        img[-10:, :] = [100, 100, 100]
        img[:, :10] = [100, 100, 100]
        img[:, -10:] = [100, 100, 100]
        dist_u8, otsu = _background_distance_from_border(img)
        assert dist_u8 is not None
        assert otsu is not None
        assert dist_u8.shape == (100, 100)

    def test_background_distance_from_border_uniform_image(self):
        """Test _background_distance_from_border with uniform image."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        dist_u8, otsu = _background_distance_from_border(img)
        # May return None or valid values depending on implementation
        if dist_u8 is not None:
            assert dist_u8.shape == (100, 100)

    def test_recover_piece_edges_preserves_shape(self):
        """Test _recover_piece_edges preserves mask shape."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        result = _recover_piece_edges(img, mask, kernel_size=7)
        assert result.shape == mask.shape

    def test_recover_piece_edges_recovers_edges(self):
        """Test _recover_piece_edges correctly expands mask at edges."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        result = _recover_piece_edges(img, mask, kernel_size=7)
        # Result should expand the mask, so sum should be >= original
        assert result.sum() >= mask.sum()

    def test_smooth_piece_contour_preserves_shape(self):
        """Test _smooth_piece_contour preserves mask shape."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        result = _smooth_piece_contour(mask)
        assert result.shape == mask.shape

    def test_smooth_piece_contour_empty_mask(self):
        """Test _smooth_piece_contour with empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _smooth_piece_contour(mask)
        assert result.shape == mask.shape
        assert result.sum() == 0

    def test_template_color_clusters_returns_valid_shapes(self):
        """Test _template_color_clusters returns properly shaped outputs."""
        template = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add color variation
        template[0:50, 0:50] = [100, 150, 100]
        template[50:100, 50:100] = [150, 100, 150]
        centers, thresholds = _template_color_clusters(template, k=4)
        assert centers.shape[1] == 2  # AB channels in LAB
        assert len(thresholds) == 4

    def test_template_color_clusters_with_mask(self):
        """Test _template_color_clusters with a provided mask."""
        template = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        centers, thresholds = _template_color_clusters(
            template, template_mask=mask, k=3
        )
        assert centers is not None
        assert thresholds is not None

    def test_apply_template_cluster_mask_returns_binary(self):
        """Test _apply_template_cluster_mask returns binary mask."""
        piece = np.ones((100, 100, 3), dtype=np.uint8) * 150
        centers = np.array([[0, 0], [10, 10]], dtype=np.float32)
        thresholds = np.array([20, 20], dtype=np.float32)
        result = _apply_template_cluster_mask(piece, centers, thresholds)
        assert result.shape == (100, 100)
        assert np.all((result == 0) | (result == 1))

    def test_background_color_clusters_returns_valid_shapes(self):
        """Test _background_color_clusters returns properly shaped outputs."""
        piece = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add border variation
        piece[:10, :] = [100, 100, 100]
        piece[-10:, :] = [100, 100, 100]
        piece[:, :10] = [100, 100, 100]
        piece[:, -10:] = [100, 100, 100]
        centers, thresholds = _background_color_clusters(piece, k=3)
        assert centers.shape[1] == 2  # AB channels
        assert len(thresholds) == 3

    def test_apply_background_cluster_mask_returns_binary(self):
        """Test _apply_background_cluster_mask returns binary mask."""
        piece = np.ones((100, 100, 3), dtype=np.uint8) * 150
        centers = np.array([[0, 0]], dtype=np.float32)
        thresholds = np.array([20], dtype=np.float32)
        result = _apply_background_cluster_mask(piece, centers, thresholds)
        assert result.shape == (100, 100)
        assert np.all((result == 0) | (result == 1))

    def test_apply_background_cluster_mask_empty_thresholds(self):
        """Test _apply_background_cluster_mask handles zero thresholds."""
        piece = np.ones((100, 100, 3), dtype=np.uint8) * 150
        centers = np.array([[0, 0]], dtype=np.float32)
        thresholds = np.array([0], dtype=np.float32)  # Zero threshold
        result = _apply_background_cluster_mask(piece, centers, thresholds)
        assert result.shape == (100, 100)
        # With zero threshold, result should be all zeros or mostly zeros
        assert result.sum() == 0

    def test_mask_by_gradient_creates_valid_mask(self):
        """Test _mask_by_gradient produces a valid binary mask."""
        # Create a synthetic piece image with distinct foreground
        piece = np.ones((200, 200, 3), dtype=np.uint8) * 180  # Light background
        # Draw a dark circle as the "piece"
        cv2.circle(piece, (100, 100), 60, (50, 80, 50), -1)
        result = _mask_by_gradient(piece, kernel_size=7, open_iters=2, close_iters=2)
        assert result.shape == (200, 200)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})
        # Should detect the circle region
        assert result.sum() > 0

    def test_mask_by_gradient_detects_piece_boundary(self):
        """Test _mask_by_gradient correctly detects a piece-like boundary."""
        # Create image with a rectangle representing a piece
        piece = np.ones((150, 150, 3), dtype=np.uint8) * 200  # Light gray background
        # Draw a dark rectangle as the "piece"
        piece[30:120, 30:120] = [40, 60, 40]
        result = _mask_by_gradient(piece, kernel_size=5, open_iters=1, close_iters=2)
        # Center of the piece should be detected
        assert result[75, 75] == 1
        # Corners (background) should not be detected
        assert result[5, 5] == 0

    def test_mask_by_ai_creates_valid_mask(self):
        """Test _mask_by_ai produces a valid binary mask."""
        pytest.importorskip("rembg")  # Skip if rembg not installed
        from unittest.mock import MagicMock, patch

        from wcmbot.matcher import _mask_by_ai

        # Create a piece image with distinct foreground/background
        piece = np.zeros((100, 100, 3), dtype=np.uint8)
        piece[20:80, 20:80] = [128, 128, 128]  # Gray center "piece"

        # Mock rembg to return a plausible RGBA result
        mock_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        mock_rgba[20:80, 20:80, :3] = [128, 128, 128]
        mock_rgba[20:80, 20:80, 3] = 255  # Alpha = 255 for foreground

        with patch("rembg.remove") as mock_remove:
            with patch("wcmbot.matcher._get_rembg_session") as mock_session:
                mock_remove.return_value = mock_rgba
                mock_session.return_value = MagicMock()

                result = _mask_by_ai(piece, kernel_size=7, open_iters=2, close_iters=2)

        # Should be valid binary mask
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)
        assert set(np.unique(result)).issubset({0, 1})

    def test_mask_by_ai_detects_piece_boundary(self):
        """Test _mask_by_ai correctly identifies piece boundaries."""
        pytest.importorskip("rembg")  # Skip if rembg not installed
        from unittest.mock import MagicMock, patch

        from wcmbot.matcher import _mask_by_ai

        piece = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Mock rembg to return alpha with circular mask
        mock_rgba = np.ones((100, 100, 4), dtype=np.uint8) * 128
        y, x = np.ogrid[:100, :100]
        dist = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        mock_rgba[:, :, 3] = np.where(dist < 40, 255, 0).astype(np.uint8)

        with patch("rembg.remove") as mock_remove:
            with patch("wcmbot.matcher._get_rembg_session") as mock_session:
                mock_remove.return_value = mock_rgba
                mock_session.return_value = MagicMock()

                result = _mask_by_ai(piece, kernel_size=7, open_iters=2, close_iters=2)

        # Center should be foreground
        assert result[50, 50] == 1
        # Corners should be background
        assert result[0, 0] == 0
