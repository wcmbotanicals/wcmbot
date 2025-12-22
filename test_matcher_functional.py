import os

import pytest

from matcher import find_piece_in_template


@pytest.mark.e2e
@pytest.mark.parametrize(
    "piece_filename,knobs_x,knobs_y,exp_rot,exp_row,exp_col",
    [
        ("piece_1.jpg", 0, 0, 90, 7, 25),
        ("piece_2.jpg", 2, 2, 0, 11, 20),
        ("piece_3.jpg", 2, 2, 90, 15, 12),
        ("piece_4.jpg", 1, 1, 0, 13, 27),
        ("piece_5.jpg", 0, 2, 180, 11, 6),
        ("piece_6.jpg", 1, 2, 270, 18, 20),
        ("piece_7.jpg", 2, 0, 270, 7, 13),
        ("piece_8.jpg", 0, 2, 270, 18, 24),
        ("piece_10.jpg", 0, 2, 90, 2, 5),
        ("piece_11.jpg", 1, 1, 180, 27, 8),
        ("piece_12.jpg", 1, 1, 0, 18, 21),
        ("piece_13.jpg", 2, 0, 90, 2, 4),
        ("piece_14.jpg", 0, 2, 180, 25, 10),
    ],
)
def test_find_piece_expected_location(
    piece_filename, knobs_x, knobs_y, exp_rot, exp_row, exp_col
):
    here = os.path.dirname(__file__)
    template_path = os.path.join(here, "media", "templates", "sample_puzzle.png")
    piece_path = os.path.join(here, "media", "pieces", piece_filename)

    payload = find_piece_in_template(
        piece_image_path=piece_path,
        template_image_path=template_path,
        knobs_x=knobs_x,
        knobs_y=knobs_y,
    )

    assert payload.matches, "No matches returned by matcher"
    top = payload.matches[0]

    assert top["rot"] == exp_rot, f"rotation mismatch for {piece_filename}"
    assert top["row"] == exp_row, f"row mismatch for {piece_filename}"
    assert top["col"] == exp_col, f"col mismatch for {piece_filename}"
