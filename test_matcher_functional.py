import os

import cv2
import numpy as np
import pytest

from matcher import find_piece_in_template

HERE = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(HERE, "media", "templates", "sample_puzzle.png")
PIECES_DIR = os.path.join(HERE, "media", "pieces")

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

ROTATION_SWEEP_DEGREES = [-15, -10, -5, -2.5, 0, 2.5, 5, 10, 15]


def _background_bgr(img: np.ndarray) -> tuple[int, int, int]:
    h, w = img.shape[:2]
    samples = np.array(
        [
            img[0, 0],
            img[0, w - 1],
            img[h - 1, 0],
            img[h - 1, w - 1],
            img[0, w // 2],
            img[h - 1, w // 2],
            img[h // 2, 0],
            img[h // 2, w - 1],
        ],
        dtype=np.float32,
    )
    median = np.median(samples, axis=0).round().astype(np.uint8)
    return int(median[0]), int(median[1]), int(median[2])


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
    piece_path = os.path.join(PIECES_DIR, piece_filename)
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
    "piece_filename,knobs_x,knobs_y,exp_rot,exp_row,exp_col",
    BASE_CASES,
)
def test_find_piece_expected_location(
    piece_filename, knobs_x, knobs_y, exp_rot, exp_row, exp_col
):
    template_path = TEMPLATE_PATH
    piece_path = os.path.join(PIECES_DIR, piece_filename)

    payload = find_piece_in_template(
        piece_image_path=piece_path,
        template_image_path=template_path,
        knobs_x=None,
        knobs_y=None,
        infer_knobs=True,
        auto_align=True,
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
@pytest.mark.skip(reason="auto-alignment needs more work")
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
    template_path = TEMPLATE_PATH
    if extra_deg == 0:
        piece_path = os.path.join(PIECES_DIR, piece_filename)
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
