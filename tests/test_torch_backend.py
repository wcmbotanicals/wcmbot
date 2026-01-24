import numpy as np
import pytest

import cv2


def test_torch_ccorr_normed_matches_opencv_cpu() -> None:
    pytest.importorskip("torch")

    from wcmbot.matcher import _match_template_torch_ccorr_normed

    rng = np.random.default_rng(0)
    template = rng.normal(size=(64, 48)).astype(np.float32)
    patch = rng.normal(size=(11, 9)).astype(np.float32)

    cv = cv2.matchTemplate(template, patch, cv2.TM_CCORR_NORMED)
    th = _match_template_torch_ccorr_normed(template, patch, device="cpu")

    assert cv.shape == th.shape
    np.testing.assert_allclose(th, cv, rtol=1e-4, atol=1e-5)


def test_torch_ccorr_normed_empty_when_patch_larger() -> None:
    pytest.importorskip("torch")

    from wcmbot.matcher import _match_template_torch_ccorr_normed

    template = np.zeros((8, 8), dtype=np.float32)
    patch = np.ones((9, 9), dtype=np.float32)

    out = _match_template_torch_ccorr_normed(template, patch, device="cpu")
    assert out.shape == (0, 0)
    assert out.dtype == np.float32
