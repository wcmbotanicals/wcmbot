"""
Modern puzzle matcher that mirrors the high-performance pipeline from 1.py.
Exposes helper utilities so the UI can render the debug-style plots without
needing to reproduce image-processing logic.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------- configuration ----------
COLS = 36
ROWS = 28
PIECE_CELLS_APPROX = (1, 1)
EST_SCALE_WINDOW = np.linspace(0.8, 1.2, num=11).tolist()
ROTATIONS = [0, 90, 180, 270]
TOP_MATCH_COUNT = 5
TOP_MATCH_SCAN_MULTIPLIER = 50
TOP_MATCH_SCAN_MULT = TOP_MATCH_SCAN_MULTIPLIER  # Backward-compatible alias; prefer TOP_MATCH_SCAN_MULTIPLIER
PROFILE_ENV = "WCMBOT_PROFILE"
COARSE_FACTOR = 0.4
COARSE_TOP_K = 3
COARSE_PADDING_PIXELS = 24
COARSE_PAD_PX = (
    COARSE_PADDING_PIXELS  # Backward-compatible alias; prefer COARSE_PADDING_PIXELS
)
COARSE_MIN_SIDE = 240

KNOB_WIDTH_FRAC = 1.0 / 3.0
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MATCH_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

LOWER_BLUE1 = np.array([90, 60, 40], dtype=np.uint8)
UPPER_BLUE1 = np.array([140, 255, 255], dtype=np.uint8)
LOWER_BLUE2 = np.array([85, 30, 60], dtype=np.uint8)
UPPER_BLUE2 = np.array([160, 255, 220], dtype=np.uint8)

OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_MASK_AREA_FRAC = 0.0005
AUTO_ALIGN_MIN_DEG = 0.0
AUTO_ALIGN_MAX_DEG = 20.0
AUTO_ALIGN_MIN_LINES = 4
AUTO_ALIGN_MIN_AREA_FRAC = -0.1  # only reject if bbox gets substantially worse
AUTO_ALIGN_HOUGH_THRESHOLD = 50
AUTO_ALIGN_HOUGH_MIN_LINE = 40
AUTO_ALIGN_HOUGH_MAX_GAP = 20
INFER_KNOBS_TIE_EPS = 0.01
INFER_KNOBS_LOW_FILL = 0.50
INFER_KNOBS_HIGH_FILL = 0.65


# ---------- helper dataclasses ----------
@dataclass
class MatchPayload:
    template_rgb: np.ndarray
    template_bin: np.ndarray
    piece_rgb: np.ndarray
    piece_mask: np.ndarray
    piece_bin: np.ndarray
    matches: List[Dict]
    template_shape: Tuple[int, int]
    auto_align_deg: float = 0.0
    knobs_x: Optional[int] = None
    knobs_y: Optional[int] = None
    knobs_inferred: bool = False


@dataclass
class TemplateCacheEntry:
    mtime: float
    template_rgb: np.ndarray
    template_bin: np.ndarray
    blur_cache: Dict[Optional[Tuple[int, int]], np.ndarray]


_TEMPLATE_CACHE: Dict[str, TemplateCacheEntry] = {}


# ---------- helpers ----------
def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _load_template_cached(path: str) -> TemplateCacheEntry:
    """
    Load a template image with caching and mtime-based invalidation.

    Caches the template image (both RGB and binarized versions) to avoid
    redundant disk I/O and preprocessing. The cache is invalidated automatically
    when the file's modification time changes.

    Args:
        path: Filesystem path to the template image.

    Returns:
        A TemplateCacheEntry containing the cached template data.

    Raises:
        RuntimeError: If the image file does not exist or cannot be loaded.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Failed to load image: {path}")
    mtime = os.path.getmtime(path)
    entry = _TEMPLATE_CACHE.get(path)
    if entry and entry.mtime == mtime:
        return entry
    template = _load_image(path)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template_bin = _binarize_two_color(template)
    entry = TemplateCacheEntry(
        mtime=mtime,
        template_rgb=template_rgb,
        template_bin=template_bin,
        blur_cache={},
    )
    _TEMPLATE_CACHE[path] = entry
    return entry


def _get_template_blur_f32(
    template_bin: np.ndarray,
    blur_ksz: Optional[Tuple[int, int]],
    blur_cache: Dict[Optional[Tuple[int, int]], np.ndarray],
) -> np.ndarray:
    """
    Get a blurred float32 version of the template with caching.

    Converts the binarized template to float32 and optionally applies Gaussian
    blur. Results are cached to avoid redundant preprocessing when the same
    blur kernel is used multiple times.

    Args:
        template_bin: Binary template image (values 0 or 1, or 0-255).
        blur_ksz: Optional Gaussian blur kernel size tuple (width, height).
            If None, no blurring is applied.
        blur_cache: Dictionary to cache blurred results by kernel size.

    Returns:
        Float32 array of the (optionally blurred) template.
    """
    cached = blur_cache.get(blur_ksz)
    if cached is not None:
        return cached
    T = (
        (template_bin * 255).astype(np.uint8)
        if template_bin.max() <= 1
        else template_bin.astype(np.uint8)
    )
    if blur_ksz is not None:
        T_blur = cv2.GaussianBlur(T, blur_ksz, 0)
    else:
        T_blur = T.copy()
    T_blur_f32 = T_blur.astype(np.float32)
    blur_cache[blur_ksz] = T_blur_f32
    return T_blur_f32


def preload_template_cache(
    template_image_path: str, blur_ksz: Optional[Tuple[int, int]] = (3, 3)
) -> None:
    """
    Preload a template image and its blurred binary representation into the cache.

    This is a convenience API for callers (e.g. the web app) that want to
    amortize the cost of loading, binarizing and optionally blurring the
    template image before handling the first real request. Calling this
    function during application startup reduces the latency of the first
    request that needs to match against the given template.

    Args:
        template_image_path: Filesystem path to the template image that will
            be used for matching.
        blur_ksz: Optional Gaussian blur kernel size to precompute on the
            binarized template. If None, the unblurred template is cached.
    """
    entry = _load_template_cached(template_image_path)
    _get_template_blur_f32(entry.template_bin, blur_ksz, entry.blur_cache)


def _binarize_two_color(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw // 255).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box of non-zero pixels in a mask.

    Args:
        mask: Binary mask array where non-zero values indicate regions of interest.

    Returns:
        Tuple of (y_min, y_max, x_min, x_max) coordinates defining the
        bounding box of all non-zero pixels in the mask.

    Raises:
        RuntimeError: If the mask is empty (contains no non-zero pixels).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Crop failed: empty mask")
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _keep_largest_component(
    mask01: np.ndarray, min_frac: float = MIN_MASK_AREA_FRAC
) -> np.ndarray:
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask01)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    h, w = mask01.shape[:2]
    if area < min_frac * (h * w):
        return np.zeros_like(mask01)
    out = np.zeros_like(mask255)
    cv2.drawContours(out, [largest], -1, 255, thickness=-1)
    return (out // 255).astype(np.uint8)


def _mask_by_blue(piece_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, LOWER_BLUE1, UPPER_BLUE1)
    m2 = cv2.inRange(hsv, LOWER_BLUE2, UPPER_BLUE2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=CLOSE_ITERS)
    mask01 = (mask > 0).astype(np.uint8)
    mask01 = _keep_largest_component(mask01)
    if mask01.sum() == 0:
        raise RuntimeError(
            "Blue segmentation produced empty mask - tune HSV ranges or check image"
        )
    return mask01


def _background_bgr(img_bgr: np.ndarray) -> Tuple[int, int, int]:
    h, w = img_bgr.shape[:2]
    samples = np.array(
        [
            img_bgr[0, 0],
            img_bgr[0, w - 1],
            img_bgr[h - 1, 0],
            img_bgr[h - 1, w - 1],
            img_bgr[0, w // 2],
            img_bgr[h - 1, w // 2],
            img_bgr[h // 2, 0],
            img_bgr[h // 2, w - 1],
        ],
        dtype=np.float32,
    )
    median = np.median(samples, axis=0).round().astype(np.uint8)
    return int(median[0]), int(median[1]), int(median[2])


def _rotate_img(
    img: np.ndarray,
    angle: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int | Tuple[int, int, int] = 0,
) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _mask_bbox_area(mask01: np.ndarray) -> int:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return 0
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def _estimate_mask_tilt(mask01: np.ndarray) -> Tuple[Optional[float], int]:
    """
    Estimate tilt angle using Hough line detection on mask edges.

    Uses a weighted mean of detected line angles, where weights are the
    line lengths. Longer lines are more reliable edge detections than short
    segments, so this approach is more robust than using a median or unweighted mean.
    """
    edges = cv2.Canny(mask01.astype(np.uint8) * 255, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=AUTO_ALIGN_HOUGH_THRESHOLD,
        minLineLength=AUTO_ALIGN_HOUGH_MIN_LINE,
        maxLineGap=AUTO_ALIGN_HOUGH_MAX_GAP,
    )
    if lines is None:
        return None, 0
    angles: List[float] = []
    lengths: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < AUTO_ALIGN_HOUGH_MIN_LINE * 0.5:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        angle = ((angle + 45) % 90) - 45
        angles.append(float(angle))
        lengths.append(length)
    if not angles:
        return None, 0

    # Use mean with outlier rejection

    angles_array = np.array(angles, dtype=np.float32)
    iqr_angle = np.percentile(angles_array, 75) - np.percentile(angles_array, 25)
    inliers = angles_array[
        abs(angles_array - np.median(angles_array)) <= (1.5 * iqr_angle)
    ]
    if len(inliers) == 0:
        return None, 0
    mean_angle = float(np.mean(inliers))
    return mean_angle, len(inliers)


def _estimate_alignment_from_mask(mask01: np.ndarray) -> float:
    """
    Estimate alignment correction using Hough line detection.

    Detects straight edges in the mask and computes a weighted mean angle.
    Only applies correction if:
    1. Sufficient straight lines are detected
    2. The correction tightens the bounding box (validation)
    3. The angle is within reasonable bounds
    """

    # Crop mask to bounding box before estimating tilt
    angle, line_count = _estimate_mask_tilt(mask01)

    # Need at least a few lines to be confident
    if angle is None or line_count < AUTO_ALIGN_MIN_LINES:
        return 0.0

    correction = -angle

    # Only consider corrections within a reasonable range
    if abs(correction) > AUTO_ALIGN_MAX_DEG:
        return 0.0

    # Validate that the correction actually helps by checking bbox tightness
    area0 = _mask_bbox_area(mask01)
    if area0 <= 0:
        return 0.0

    rotated_mask = _rotate_img(
        (mask01 > 0).astype(np.uint8) * 255,
        correction,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )
    area1 = _mask_bbox_area(rotated_mask)
    if area1 <= 0:
        return 0.0

    # Require that bbox gets tighter with a reasonable threshold (0.8% = 0.008)
    # Use a modest threshold to catch real alignment issues while avoiding over-correction
    area_delta = (area0 - area1) / float(area0)
    if area_delta < AUTO_ALIGN_MIN_AREA_FRAC:
        return 0.0

    return float(correction)


def _estimate_scales(
    template_shape: Tuple[int, int],
    piece_mask: np.ndarray,
    knobs_x: int,
    knobs_y: int,
    scale_window: List[float] = EST_SCALE_WINDOW,
) -> Tuple[float, List[float]]:
    th, tw = template_shape
    cell_w = tw / COLS
    cell_h = th / ROWS
    desired_core_w = cell_w * PIECE_CELLS_APPROX[0]
    desired_core_h = cell_h * PIECE_CELLS_APPROX[1]
    desired_full_w = desired_core_w * (1.0 + knobs_x * KNOB_WIDTH_FRAC)
    desired_full_h = desired_core_h * (1.0 + knobs_y * KNOB_WIDTH_FRAC)

    mh, mw = piece_mask.shape
    if mw == 0 or mh == 0:
        raise RuntimeError("Piece mask has zero size")

    est_scale_w = desired_full_w / mw
    est_scale_h = desired_full_h / mh
    piece_area_px = piece_mask.sum()
    desired_area_px = desired_core_w * desired_core_h
    if piece_area_px > 0 and desired_area_px > 0:
        est_scale_area = np.sqrt(desired_area_px / piece_area_px)
    else:
        est_scale_area = (est_scale_w + est_scale_h) / 2.0
    est_scale = (est_scale_w * 0.45) + (est_scale_h * 0.45) + (est_scale_area * 0.10)
    if not (0.02 < est_scale < 20.0):
        raise RuntimeError(
            f"Estimated scale {est_scale:.3f} is implausible - check PIECE_CELLS_APPROX and image sizes"
        )
    scales = [est_scale * f for f in scale_window]
    return est_scale, scales


def _infer_knob_counts(
    piece_mask: np.ndarray,
    template_shape: Tuple[int, int],
) -> Tuple[int, int]:
    mh, mw = piece_mask.shape
    if mw == 0 or mh == 0:
        return 0, 0

    piece_area_px = float(piece_mask.sum())
    if piece_area_px <= 0:
        return 0, 0
    fill_ratio = piece_area_px / float(mw * mh)

    th, tw = template_shape
    cell_w = tw / COLS
    cell_h = th / ROWS
    desired_core_w = cell_w * PIECE_CELLS_APPROX[0]
    desired_core_h = cell_h * PIECE_CELLS_APPROX[1]
    desired_area_px = desired_core_w * desired_core_h

    scored: List[Tuple[float, int, int]] = []
    for kx in range(3):
        for ky in range(3):
            desired_full_w = desired_core_w * (1.0 + kx * KNOB_WIDTH_FRAC)
            desired_full_h = desired_core_h * (1.0 + ky * KNOB_WIDTH_FRAC)
            est_scale_w = desired_full_w / mw
            est_scale_h = desired_full_h / mh
            est_scale_area = np.sqrt(desired_area_px / piece_area_px)
            diff = (
                abs(est_scale_w - est_scale_h)
                + 0.5 * abs(est_scale_w - est_scale_area)
                + 0.5 * abs(est_scale_h - est_scale_area)
            )
            scored.append((float(diff), kx, ky))

    scored.sort(key=lambda item: item[0])
    best_diff = scored[0][0]
    candidates = [item for item in scored if item[0] <= best_diff + INFER_KNOBS_TIE_EPS]
    if fill_ratio <= INFER_KNOBS_LOW_FILL:
        candidates.sort(key=lambda item: (-(item[1] + item[2]), item[0]))
    elif fill_ratio >= INFER_KNOBS_HIGH_FILL:
        candidates.sort(key=lambda item: ((item[1] + item[2]), item[0]))
    chosen = candidates[0]
    return chosen[1], chosen[2]


def _candidate_is_close(candidate: Dict, existing: Dict) -> bool:
    cand_center = candidate["center"]
    cand_w = candidate["br"][0] - candidate["tl"][0]
    cand_h = candidate["br"][1] - candidate["tl"][1]
    ex_center = existing["center"]
    ex_w = existing["br"][0] - existing["tl"][0]
    ex_h = existing["br"][1] - existing["tl"][1]
    dx = cand_center[0] - ex_center[0]
    dy = cand_center[1] - ex_center[1]
    proximity_thresh = max(12.0, min(cand_w, ex_w) * 0.25, min(cand_h, ex_h) * 0.25)
    return (dx * dx + dy * dy) <= (proximity_thresh * proximity_thresh)


def _update_top_matches(
    top_matches: List[Dict], candidate: Dict, max_len: int = TOP_MATCH_COUNT
) -> None:
    if max_len <= 0:
        return
    for idx, existing in enumerate(top_matches):
        if _candidate_is_close(candidate, existing):
            if candidate["score"] > existing["score"]:
                top_matches[idx] = candidate
            return
    top_matches.append(candidate)
    top_matches.sort(key=lambda d: d["score"], reverse=True)
    if len(top_matches) > max_len:
        del top_matches[max_len:]


def _attach_contours_to_matches(
    matches: List[Dict], base_mask: np.ndarray, dilate_kernel: np.ndarray
) -> List[Dict]:
    if not matches:
        return []
    base_mask255 = (base_mask > 0).astype(np.uint8) * 255
    enriched = []
    rot_cache: Dict[int, np.ndarray] = {}
    for match in matches:
        nm = dict(match)
        rot = match["rot"]
        rot_mask = rot_cache.get(rot)
        if rot_mask is None:
            rot_mask = _rotate_img(base_mask255, rot)
            rot_mask = (rot_mask > 127).astype(np.uint8) * 255
            rot_mask = cv2.morphologyEx(
                rot_mask, cv2.MORPH_DILATE, dilate_kernel, iterations=1
            )
            rot_cache[rot] = rot_mask
        ws = int(round(rot_mask.shape[1] * match["scale"]))
        hs = int(round(rot_mask.shape[0] * match["scale"]))
        if ws <= 0 or hs <= 0:
            nm["contours"] = []
            enriched.append(nm)
            continue
        mask_s = cv2.resize(rot_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        tlx, tly = nm["tl"]
        offset = np.array([[[tlx, tly]]], dtype=np.int32)
        nm["contours"] = [
            (cnt.astype(np.int32) + offset).astype(np.int32)
            for cnt in contours
            if cnt.size
        ]
        enriched.append(nm)
    return enriched


def _match_template_multiscale_binary(
    template_bin_img: np.ndarray,
    piece_bin_pattern: np.ndarray,
    piece_mask: np.ndarray,
    cols: int,
    rows: int,
    scales: List[float],
    rotations: List[int],
    blur_ksz: Optional[Tuple[int, int]] = (3, 3),
    corr_method: int = cv2.TM_CCORR_NORMED,
    template_blur_f32: Optional[np.ndarray] = None,
) -> Tuple[Dict, List[Dict]]:
    if template_blur_f32 is None:
        T = (
            (template_bin_img * 255).astype(np.uint8)
            if template_bin_img.max() <= 1
            else template_bin_img.astype(np.uint8)
        )
        if blur_ksz is not None:
            T_blur = cv2.GaussianBlur(T, blur_ksz, 0)
        else:
            T_blur = T.copy()
        T_blur_f32 = T_blur.astype(np.float32)
    else:
        T_blur_f32 = template_blur_f32
    P = (
        (piece_bin_pattern * 255).astype(np.uint8)
        if piece_bin_pattern.max() <= 1
        else piece_bin_pattern.astype(np.uint8)
    )
    if piece_mask.max() <= 1:
        M = (piece_mask > 0).astype(np.uint8) * 255
    else:
        M = (piece_mask > 127).astype(np.uint8) * 255

    th, tw = T_blur_f32.shape[:2]
    cell_w = tw / cols
    cell_h = th / rows

    combo_candidates: List[Dict] = []
    dilate_ker = MATCH_DILATE_KERNEL

    use_coarse = 0.0 < COARSE_FACTOR < 1.0 and min(tw, th) >= COARSE_MIN_SIDE
    if use_coarse:
        tw_c = max(1, int(round(tw * COARSE_FACTOR)))
        th_c = max(1, int(round(th * COARSE_FACTOR)))
        if tw_c < 2 or th_c < 2:
            use_coarse = False
            T_coarse_blur = None
        else:
            T_coarse_blur = cv2.resize(
                T_blur_f32, (tw_c, th_c), interpolation=cv2.INTER_AREA
            )
    else:
        T_coarse_blur = None

    def _candidate_order(flat: np.ndarray, max_len: int) -> np.ndarray:
        if flat.size <= max_len:
            return np.argsort(flat)[::-1]
        scan_count = min(flat.size, max(max_len * TOP_MATCH_SCAN_MULTIPLIER, max_len))
        order = np.argpartition(flat, -scan_count)[-scan_count:]
        return order[np.argsort(flat[order])[::-1]]

    def _scan_candidates(
        res: np.ndarray,
        order: np.ndarray,
        res_w: int,
        ws: int,
        hs: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Dict]:
        combo_best_local: List[Dict] = []
        for idx in order:
            if len(combo_best_local) >= TOP_MATCH_COUNT:
                break
            y, x = divmod(int(idx), res_w)
            x0 = x + offset_x
            y0 = y + offset_y
            cx = x0 + ws / 2
            cy = y0 + hs / 2
            tl = (int(x0), int(y0))
            br = (int(x0 + ws), int(y0 + hs))
            candidate = {
                "score": float(res[y, x]),
                "rot": rot,
                "scale": scale,
                "col": int(cx / cell_w) + 1,
                "row": int(cy / cell_h) + 1,
                "tl": tl,
                "br": br,
                "center": (float(cx), float(cy)),
            }
            if any(
                _candidate_is_close(candidate, existing)
                for existing in combo_best_local
            ):
                continue
            combo_best_local.append(candidate)
        return combo_best_local

    def _collect_matches(
        res: np.ndarray,
        ws: int,
        hs: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Dict]:
        flat = res.ravel()
        order = _candidate_order(flat, TOP_MATCH_COUNT)
        res_w = res.shape[1]
        combo_best = _scan_candidates(
            res, order, res_w, ws, hs, offset_x=offset_x, offset_y=offset_y
        )
        if len(combo_best) < TOP_MATCH_COUNT and order.size < flat.size:
            order = np.argsort(flat)[::-1]
            combo_best = _scan_candidates(
                res, order, res_w, ws, hs, offset_x=offset_x, offset_y=offset_y
            )
        return combo_best

    def _collect_coarse_positions(
        res: np.ndarray, ws: int, hs: int, top_k: int
    ) -> List[Dict]:
        flat = res.ravel()
        order = _candidate_order(flat, top_k)
        res_w = res.shape[1]
        positions: List[Dict] = []
        for idx in order:
            if len(positions) >= top_k:
                break
            y, x = divmod(int(idx), res_w)
            candidate = {
                "score": float(res[y, x]),
                "tl": (int(x), int(y)),
                "br": (int(x + ws), int(y + hs)),
                "center": (float(x + ws / 2), float(y + hs / 2)),
            }
            if any(_candidate_is_close(candidate, existing) for existing in positions):
                continue
            positions.append(candidate)
        return positions

    for rot in rotations:
        P_r = _rotate_img(P, rot)
        M_r = _rotate_img(M, rot)
        M_r = (M_r > 127).astype(np.uint8) * 255
        M_r = cv2.morphologyEx(M_r, cv2.MORPH_DILATE, dilate_ker, iterations=1)
        M_r01 = (M_r > 127).astype(np.float32)

        for scale in scales:
            ws = int(round(P_r.shape[1] * scale))
            hs = int(round(P_r.shape[0] * scale))
            if ws <= 0 or hs <= 0 or ws >= tw or hs >= th:
                continue

            patt_s = cv2.resize(P_r, (ws, hs), interpolation=cv2.INTER_NEAREST)
            mask_s = cv2.resize(M_r01, (ws, hs), interpolation=cv2.INTER_NEAREST)

            if blur_ksz is not None:
                patt_s_blur = cv2.GaussianBlur(patt_s, blur_ksz, 0).astype(np.float32)
            else:
                patt_s_blur = patt_s.astype(np.float32)

            patt_masked = patt_s_blur * mask_s
            combo_added = False

            if use_coarse and T_coarse_blur is not None:
                ws_c = max(1, int(round(ws * COARSE_FACTOR)))
                hs_c = max(1, int(round(hs * COARSE_FACTOR)))
                if (
                    1 < ws_c < T_coarse_blur.shape[1]
                    and 1 < hs_c < T_coarse_blur.shape[0]
                ):
                    patt_c = cv2.resize(
                        patt_s_blur, (ws_c, hs_c), interpolation=cv2.INTER_AREA
                    )
                    mask_c = cv2.resize(
                        mask_s, (ws_c, hs_c), interpolation=cv2.INTER_NEAREST
                    )
                    patt_masked_c = patt_c * mask_c
                    res_c = cv2.matchTemplate(T_coarse_blur, patt_masked_c, corr_method)
                    if res_c.size:
                        coarse_positions = _collect_coarse_positions(
                            res_c, ws_c, hs_c, COARSE_TOP_K
                        )
                        seen_rois = set()
                        for coarse in coarse_positions:
                            x_c, y_c = coarse["tl"]
                            x_full = int(round(x_c / COARSE_FACTOR))
                            y_full = int(round(y_c / COARSE_FACTOR))
                            x_full = max(0, min(x_full, tw - ws))
                            y_full = max(0, min(y_full, th - hs))
                            x0 = max(0, x_full - COARSE_PADDING_PIXELS)
                            y0 = max(0, y_full - COARSE_PADDING_PIXELS)
                            x1 = min(tw, x_full + ws + COARSE_PADDING_PIXELS)
                            y1 = min(th, y_full + hs + COARSE_PADDING_PIXELS)
                            roi_key = (x0, y0, x1, y1)
                            if roi_key in seen_rois:
                                continue
                            seen_rois.add(roi_key)
                            roi = T_blur_f32[y0:y1, x0:x1]
                            if roi.shape[0] < hs or roi.shape[1] < ws:
                                continue
                            res = cv2.matchTemplate(roi, patt_masked, corr_method)
                            if res.size == 0:
                                continue
                            combo_best = _collect_matches(
                                res, ws, hs, offset_x=x0, offset_y=y0
                            )
                            if combo_best:
                                combo_candidates.extend(combo_best)
                                combo_added = True

            if not combo_added:
                res = cv2.matchTemplate(T_blur_f32, patt_masked, corr_method)

                if res.size == 0:
                    continue

                combo_best = _collect_matches(res, ws, hs)
                combo_candidates.extend(combo_best)

    if not combo_candidates:
        raise RuntimeError("No match found (binary matcher)")

    combo_candidates.sort(key=lambda c: c["score"], reverse=True)
    top_matches: List[Dict] = []
    for candidate in combo_candidates:
        _update_top_matches(top_matches, candidate, TOP_MATCH_COUNT)
    best = top_matches[0]
    return best, top_matches


def _ensure_three_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img


def _binary_to_uint8(img01: np.ndarray) -> np.ndarray:
    return img01.astype(np.uint8) * 255


def _create_resized_preview(
    piece_bin: np.ndarray, piece_mask: np.ndarray, match: Dict
) -> np.ndarray:
    rot = match["rot"]
    scale = match["scale"]
    rot_bin = _rotate_img(_binary_to_uint8(piece_bin), rot)
    rot_mask = _rotate_img(_binary_to_uint8(piece_mask), rot)
    ws = max(1, match["br"][0] - match["tl"][0])
    hs = max(1, match["br"][1] - match["tl"][1])
    rv = cv2.resize(rot_bin, (ws, hs), interpolation=cv2.INTER_NEAREST)
    rv_mask = cv2.resize(rot_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)
    rv = (rv * (rv_mask > 127)).astype(np.uint8)
    return rv


def _render_zoom_image(
    template_rgb: np.ndarray,
    template_shape: Tuple[int, int],
    piece_bin: np.ndarray,
    piece_mask: np.ndarray,
    match: Dict,
    zoom: int = 98,
) -> np.ndarray:
    tlx, tly = match["tl"]
    brx, bry = match["br"]
    th, tw = template_shape

    if zoom <= 0:
        zx0, zy0 = 0, 0
        zx1, zy1 = tw, th
    elif zoom >= 100:
        zx0, zy0 = max(0, tlx), max(0, tly)
        zx1, zy1 = min(tw, brx), min(th, bry)
    else:
        max_pad = max(8, int(min(template_shape) * 0.02))
        t = zoom / 100.0
        pad_factor = (100 - zoom) / 100.0
        pad = int(max_pad * (1 + 9 * pad_factor))
        bbox_x0 = max(0, tlx - pad)
        bbox_y0 = max(0, tly - pad)
        bbox_x1 = min(tw, brx + pad)
        bbox_y1 = min(th, bry + pad)
        zx0 = int(0 * (1 - t) + bbox_x0 * t)
        zy0 = int(0 * (1 - t) + bbox_y0 * t)
        zx1 = int(tw * (1 - t) + bbox_x1 * t)
        zy1 = int(th * (1 - t) + bbox_y1 * t)

    zx0 = max(0, min(zx0, tw - 1))
    zy0 = max(0, min(zy0, th - 1))
    zx1 = max(zx0 + 1, min(zx1, tw))
    zy1 = max(zy0 + 1, min(zy1, th))

    region_rgb = template_rgb[zy0:zy1, zx0:zx1].copy()
    piece_x0 = tlx - zx0
    piece_y0 = tly - zy0
    piece_x1 = piece_x0 + (brx - tlx)
    piece_y1 = piece_y0 + (bry - tly)

    rot = match["rot"]
    rot_bin = _rotate_img(_binary_to_uint8(piece_bin), rot)
    rot_mask = _rotate_img(_binary_to_uint8(piece_mask), rot)
    target_h = max(1, piece_y1 - piece_y0)
    target_w = max(1, piece_x1 - piece_x0)
    pv = cv2.resize(rot_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(rot_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    pv3 = _ensure_three_channel(pv)

    if (
        piece_x0 < region_rgb.shape[1]
        and piece_y0 < region_rgb.shape[0]
        and piece_x1 > 0
        and piece_y1 > 0
    ):
        src_x0 = max(0, -piece_x0)
        src_y0 = max(0, -piece_y0)
        src_x1 = pv3.shape[1] - max(0, piece_x1 - region_rgb.shape[1])
        src_y1 = pv3.shape[0] - max(0, piece_y1 - region_rgb.shape[0])
        dst_x0 = max(0, piece_x0)
        dst_y0 = max(0, piece_y0)
        dst_x1 = min(region_rgb.shape[1], piece_x1)
        dst_y1 = min(region_rgb.shape[0], piece_y1)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0 and src_x1 > src_x0 and src_y1 > src_y0:
            piece_patch = pv3[src_y0:src_y1, src_x0:src_x1]
            mask_patch = mask[src_y0:src_y1, src_x0:src_x1]
            template_patch = region_rgb[dst_y0:dst_y1, dst_x0:dst_x1]
            if piece_patch.shape[:2] == template_patch.shape[:2]:
                mask_norm = (mask_patch > 127).astype(np.float32)
                if mask_norm.ndim == 2:
                    mask_norm = mask_norm[:, :, np.newaxis]
                blended_patch = (
                    template_patch * (1 - mask_norm * 0.4)
                    + piece_patch * (mask_norm * 0.4)
                ).astype(np.uint8)
                region_rgb[dst_y0:dst_y1, dst_x0:dst_x1] = blended_patch

    contours = match.get("contours", [])
    region_bgr = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)
    if contours:
        for cnt in contours:
            cnt = np.asarray(cnt).reshape(-1, 2)
            if cnt.shape[0] < 2:
                continue
            cnt_offset = cnt - np.array([zx0, zy0])
            cnt_offset = cnt_offset.astype(np.int32)
            cv2.polylines(region_bgr, [cnt_offset], True, (0, 0, 255), 2)
    else:
        rect_x = tlx - zx0
        rect_y = tly - zy0
        cv2.rectangle(
            region_bgr,
            (rect_x, rect_y),
            (rect_x + (brx - tlx), rect_y + (bry - tly)),
            (0, 255, 0),
            2,
        )
    return cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)


def find_piece_in_template(
    piece_image_path: str,
    template_image_path: str,
    knobs_x: Optional[int],
    knobs_y: Optional[int],
    auto_align: bool = False,
    infer_knobs: Optional[bool] = None,
) -> MatchPayload:
    profile_value = os.getenv(PROFILE_ENV, "").strip().lower()
    profile = profile_value not in ("", "0", "false", "no")
    if profile:
        t0 = time.perf_counter()
        marks: List[Tuple[str, float]] = []

    template_entry = _load_template_cached(template_image_path)
    if profile:
        marks.append(("template", time.perf_counter()))

    piece = _load_image(piece_image_path)
    if profile:
        marks.append(("piece", time.perf_counter()))

    template_rgb = template_entry.template_rgb
    template_bin = template_entry.template_bin

    piece_mask = _mask_by_blue(piece)
    if profile:
        marks.append(("mask", time.perf_counter()))

    y0, y1, x0, x1 = _mask_bbox(piece_mask)
    piece_crop = piece[y0:y1, x0:x1].copy()
    piece_mask_crop = piece_mask[y0:y1, x0:x1].copy()
    if profile:
        marks.append(("crop", time.perf_counter()))
    piece_bin = _binarize_two_color(piece_crop) * piece_mask_crop
    piece_rgb = cv2.cvtColor(piece_crop, cv2.COLOR_BGR2RGB)
    if profile:
        marks.append(("binarize", time.perf_counter()))

    infer_knobs_enabled = bool(infer_knobs)
    if knobs_x is None or knobs_y is None:
        infer_knobs_enabled = True
    if isinstance(knobs_x, (int, float)) and knobs_x < 0:
        infer_knobs_enabled = True
    if isinstance(knobs_y, (int, float)) and knobs_y < 0:
        infer_knobs_enabled = True

    auto_align_enabled = auto_align
    auto_align_deg = 0.0
    if auto_align_enabled:
        correction = _estimate_alignment_from_mask(piece_mask_crop)
        if abs(correction) >= AUTO_ALIGN_MIN_DEG:
            bg = _background_bgr(piece)
            piece = _rotate_img(
                piece,
                correction,
                interpolation=cv2.INTER_LINEAR,
                border_value=bg,
            )
            auto_align_deg = correction
            piece_mask = _mask_by_blue(piece)
            y0, y1, x0, x1 = _mask_bbox(piece_mask)
            piece_crop = piece[y0:y1, x0:x1].copy()
            piece_mask_crop = piece_mask[y0:y1, x0:x1].copy()
            piece_bin = _binarize_two_color(piece_crop) * piece_mask_crop
            piece_rgb = cv2.cvtColor(piece_crop, cv2.COLOR_BGR2RGB)
        if profile:
            marks.append(("auto_align", time.perf_counter()))

    knobs_inferred = False
    if infer_knobs_enabled:
        knobs_x, knobs_y = _infer_knob_counts(
            piece_mask_crop,
            template_bin.shape,
        )
        knobs_inferred = True
        if profile:
            marks.append(("knob_infer", time.perf_counter()))
    else:
        knobs_x = int(knobs_x)
        knobs_y = int(knobs_y)

    _, scales = _estimate_scales(template_bin.shape, piece_mask_crop, knobs_x, knobs_y)
    if profile:
        marks.append(("scale", time.perf_counter()))

    template_blur_f32 = _get_template_blur_f32(
        template_bin, (3, 3), template_entry.blur_cache
    )
    _, top_matches = _match_template_multiscale_binary(
        template_bin,
        piece_bin,
        piece_mask_crop,
        COLS,
        ROWS,
        scales,
        ROTATIONS,
        blur_ksz=(3, 3),
        corr_method=cv2.TM_CCORR_NORMED,
        template_blur_f32=template_blur_f32,
    )
    if profile:
        marks.append(("match", time.perf_counter()))

    top_matches = _attach_contours_to_matches(
        top_matches, piece_mask_crop, MATCH_DILATE_KERNEL
    )
    if profile:
        marks.append(("contours", time.perf_counter()))
    for idx, match in enumerate(top_matches):
        match["index"] = idx
        match["cx"] = int(round(match["center"][0]))
        match["cy"] = int(round(match["center"][1]))
        match["width"] = match["br"][0] - match["tl"][0]
        match["height"] = match["br"][1] - match["tl"][1]

    if profile:
        t_end = time.perf_counter()
        prev = t0
        parts = []
        for label, ts in marks:
            parts.append(f"{label}={((ts - prev) * 1000.0):.2f}ms")
            prev = ts
        parts.append(f"total={((t_end - t0) * 1000.0):.2f}ms")
        print("matcher profile:", " ".join(parts))

    return MatchPayload(
        template_rgb=template_rgb,
        template_bin=template_bin,
        piece_rgb=piece_rgb,
        piece_mask=piece_mask_crop,
        piece_bin=piece_bin,
        matches=top_matches,
        template_shape=template_bin.shape,
        auto_align_deg=auto_align_deg,
        knobs_x=knobs_x,
        knobs_y=knobs_y,
        knobs_inferred=knobs_inferred,
    )


def _static_views(payload: MatchPayload) -> Dict[str, np.ndarray]:
    template_bin_viz = _binary_to_uint8(payload.template_bin)
    piece_mask_viz = _binary_to_uint8(payload.piece_mask)
    piece_bin_viz = _binary_to_uint8(payload.piece_bin)
    return {
        "template_color": payload.template_rgb,
        "template_bin": _ensure_three_channel(template_bin_viz),
        "piece_crop": payload.piece_rgb,
        "piece_mask": _ensure_three_channel(piece_mask_viz),
        "piece_bin": _ensure_three_channel(piece_bin_viz),
    }


def render_primary_views(
    payload: MatchPayload, match_index: int
) -> Dict[str, np.ndarray]:
    if not payload.matches:
        raise RuntimeError("No matches available to render")
    idx = max(0, min(match_index, len(payload.matches) - 1))
    match = payload.matches[idx]
    static = _static_views(payload)
    preview = _ensure_three_channel(
        _create_resized_preview(payload.piece_bin, payload.piece_mask, match)
    )
    zoom = _render_zoom_image(
        payload.template_rgb,
        payload.template_shape,
        payload.piece_bin,
        payload.piece_mask,
        match,
        zoom=98,
    )
    zoom_full = _render_zoom_image(
        payload.template_rgb,
        payload.template_shape,
        payload.piece_bin,
        payload.piece_mask,
        match,
        zoom=0,
    )
    static.update(
        {
            "resized_piece": preview,
            "zoom_focus": zoom,
            "zoom_template": zoom_full,
        }
    )
    return static


def format_match_summary(payload: MatchPayload, match_index: int) -> str:
    if not payload.matches:
        return "No matches available."
    idx = max(0, min(match_index, len(payload.matches) - 1))
    match = payload.matches[idx]
    lines = [
        f"Match #{idx + 1} / {len(payload.matches)}",
        f"Score: {match['score']:.3f} | Rotation: {match['rot']}° | "
        f"Scale: {match['scale']:.4f}",
        f"Grid position: row {match['row']}, col {match['col']}",
    ]
    if (
        payload.knobs_inferred
        and payload.knobs_x is not None
        and payload.knobs_y is not None
    ):
        lines.append(f"Tabs inferred: {payload.knobs_x} x {payload.knobs_y}")
    if abs(payload.auto_align_deg) >= 0.1:
        lines.append(f"Auto-align (cw): {payload.auto_align_deg:+.1f}°")
    return "  \n".join(lines)


def highlight_position(
    template_image_path: str, x: int, y: int, radius: int = 30
) -> np.ndarray:
    tpl = cv2.imread(template_image_path)
    if tpl is None:
        raise ValueError("Could not load template")
    cv2.circle(tpl, (int(x), int(y)), radius, (0, 255, 0), 3)
    cv2.circle(tpl, (int(x), int(y)), radius + 5, (255, 255, 0), 2)
    tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)
    return tpl_rgb
