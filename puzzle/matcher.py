# puzzle/matcher.py
"""
Robust chamfer-only puzzle matcher (drop-in).
Main improvements:
 - Robust silhouette extraction with multiple fallbacks when mask-based edges are empty.
 - Defensive checks to avoid "no candidate" when piece silhouette can be derived.
 - Debug image saving same style as before (PUZZLE_MATCHER_DEBUG=1).
Public API:
 - find_piece_in_template(piece_image_path, template_image_path)
 - highlight_position(template_image_path, x, y, radius=30)
"""

import os
import time
from itertools import count
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Optional Django MEDIA_ROOT
try:
    from django.conf import settings
except Exception:
    settings = None

DEBUG_MATCHER = os.environ.get("PUZZLE_MATCHER_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DEBUG_COUNTER = count()
_DEBUG_STATE = {}


def _debug_enabled():
    return DEBUG_MATCHER


def _debug_reset(context):
    if not _debug_enabled():
        return
    _DEBUG_STATE.clear()
    _DEBUG_STATE["context"] = context
    _DEBUG_STATE["start"] = time.time()


def _debug_log(msg):
    if not _debug_enabled():
        return
    ctx = _DEBUG_STATE.get("context", "matcher")
    print(f"[matcher:{ctx}] {msg}")


def _debug_base_dir():
    if settings and getattr(settings, "MEDIA_ROOT", None):
        return Path(settings.MEDIA_ROOT)
    return Path(__file__).resolve().parents[1] / "media"


def _debug_dir():
    if not _debug_enabled():
        return None
    d = _debug_base_dir() / "debug" / _DEBUG_STATE.get("context", "matcher")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _debug_save_image(tag, image):
    if not _debug_enabled() or image is None:
        return
    d = _debug_dir()
    if d is None:
        return
    arr = image
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    filename = f"{int(time.time() * 1000)}_{next(_DEBUG_COUNTER)}_{tag}.png"
    cv2.imwrite(str(d / filename), arr)
    _debug_log(f"Saved debug image: {filename}")


# --- helpers ---
def _limit_image_size(image, max_dim=900):
    if image is None:
        return None
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return image
    scale = max_dim / float(max_side)
    return cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _rotate_image(image, angle):
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    return cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


# --- preprocessing ---
def _preprocess_piece_image(image):
    """
    Return a grayscale cropped piece where background is zeroed (uint8).
    """
    if image is None:
        raise ValueError("Piece image is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    # adaptive threshold fallback
    try:
        mask = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    except Exception:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) < 30:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    _debug_save_image("piece_mask_before_crop", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _debug_log("No contours found in piece; returning downscaled gray")
        out = _limit_image_size(gray)
        _debug_save_image("piece_processed_fallback", out)
        return out
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        _debug_log("Largest contour too small; returning downscaled gray")
        out = _limit_image_size(gray)
        _debug_save_image("piece_processed_small", out)
        return out
    x, y, w, h = cv2.boundingRect(largest)
    pad = 6
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    crop = gray[y0:y1, x0:x1]
    mask_crop = np.zeros_like(crop)
    shifted = largest - np.array([[x0, y0]])
    cv2.drawContours(mask_crop, [shifted], -1, 255, thickness=-1)
    piece = cv2.bitwise_and(crop, crop, mask=mask_crop)
    if cv2.countNonZero(mask_crop) < 25:
        out = _limit_image_size(crop)
        _debug_save_image("piece_processed_smallmask", out)
        return out
    out = _limit_image_size(piece)
    _debug_save_image("piece_processed", out)
    return out


# --- robust silhouette extraction with multiple fallbacks ---
def _extract_silhouette_from_mask(mask, gray_crop=None):
    """
    Attempt to return a single-channel silhouette (non-zero = edge pixels) using several fallbacks:
    1) Canny on blurred mask (preferred)
    2) If empty, draw the largest contour to a mask and re-run Canny
    3) If still empty, run Canny directly on gray_crop with multiple thresholds
    4) If still empty, dilate and return contour outline
    Returns silhouette_edges (uint8)
    """
    if mask is None and gray_crop is None:
        return None
    # Ensure a binary mask if provided
    if mask is not None:
        bin_mask = (mask > 0).astype(np.uint8) * 255
    else:
        bin_mask = None

    def canny_on(img, low=50, high=150):
        b = cv2.GaussianBlur(img, (3, 3), 0)
        e = cv2.Canny(b, low, high)
        e = cv2.morphologyEx(e, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return e

    # 1) try mask => canny
    if bin_mask is not None:
        edges = canny_on(bin_mask)
        if edges is not None and edges.sum() > 0:
            _debug_save_image("silhouette_edges_mask_canny", edges)
            return edges
        # fallback: draw largest contour and re-run
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            tmp = np.zeros_like(bin_mask)
            cv2.drawContours(
                tmp, [largest], -1, 255, thickness=2
            )  # draw outline (thickness 2)
            edges2 = canny_on(tmp)
            if edges2 is not None and edges2.sum() > 0:
                _debug_save_image("silhouette_edges_contourdraw", edges2)
                return edges2

    # 2) try Canny on grayscale crop with varied thresholds (if available)
    if gray_crop is not None:
        for low, high in [(30, 120), (40, 160), (20, 100), (10, 80)]:
            e = canny_on(gray_crop, low=low, high=high)
            if e is not None and e.sum() > 0:
                _debug_save_image(f"silhouette_edges_gray_canny_{low}_{high}", e)
                return e

    # 3) If still nothing, as last resort produce contour outline from bin_mask (even if small)
    if bin_mask is not None:
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            tmp = np.zeros_like(bin_mask)
            cv2.drawContours(tmp, contours, -1, 255, thickness=1)
            tmp = cv2.dilate(tmp, np.ones((3, 3), np.uint8), iterations=1)
            _debug_save_image("silhouette_edges_last_contour", tmp)
            return tmp

    # nothing worked
    _debug_log("Silhouette extraction failed: no edges derived")
    return None


# --- template distance transform ---
def _compute_template_distance_transform(template_gray):
    blur = cv2.GaussianBlur(template_gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    _debug_save_image("template_edges", edges)
    bin_inv = (edges == 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(bin_inv, cv2.DIST_L2, 5)
    if _debug_enabled():
        vis = np.uint8(np.clip(dt / (dt.max() + 1e-9) * 255.0, 0, 255))
        _debug_save_image("template_dt_norm", vis)
    return edges, dt


# --- chamfer scoring ---
def _chamfer_score_at_location(dt, piece_edge_mask, top_left):
    x, y = int(top_left[0]), int(top_left[1])
    ph, pw = piece_edge_mask.shape[:2]
    H, W = dt.shape[:2]
    if x < 0 or y < 0 or x + pw > W or y + ph > H:
        return None
    sub = dt[y : y + ph, x : x + pw]
    mask = piece_edge_mask > 0
    if not np.any(mask):
        return None
    vals = sub[mask]
    return float(vals.mean()), int(mask.sum())


def _chamfer_match_template(dt, piece_edge_mask, search_step=4, search_region=None):
    th, tw = piece_edge_mask.shape[:2]
    H, W = dt.shape[:2]
    if search_region:
        rx0, ry0, rx1, ry1 = search_region
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(W, rx1), min(H, ry1)
    else:
        rx0, ry0, rx1, ry1 = 0, 0, W, H
    best = None
    stride = max(1, int(search_step))
    for y in range(ry0, ry1 - th + 1, stride):
        for x in range(rx0, rx1 - tw + 1, stride):
            res = _chamfer_score_at_location(dt, piece_edge_mask, (x, y))
            if res is None:
                continue
            mean_dist, count = res
            if best is None or mean_dist < best["mean_dist"]:
                best = {"mean_dist": mean_dist, "count": count, "top_left": (x, y)}
    if best is None:
        raise ValueError("No valid placement found for chamfer")
    x, y = best["top_left"]
    return x + tw // 2, y + th // 2, best["mean_dist"], best["count"], best["top_left"]


# --- scaling candidates ---
def _scale_candidates(piece_shape, template_shape, num_scales=6):
    piece_h, piece_w = piece_shape[:2]
    template_h, template_w = template_shape[:2]
    if piece_h == 0 or piece_w == 0:
        raise ValueError("Piece image is empty after preprocessing")
    max_ratio = min(template_w / piece_w, template_h / piece_h)
    if max_ratio <= 0:
        raise ValueError("Template is smaller than the piece")
    max_scale = max_ratio * 0.98 if max_ratio < 1.0 else min(max_ratio, 1.5)
    max_scale = max(max_scale, 0.05)
    if max_scale < 0.25:
        min_scale = max(0.05, max_scale * 0.5)
    else:
        min_scale = max(0.2, max_scale / 3.0)
    if min_scale > max_scale:
        min_scale = max_scale * 0.8
    if num_scales < 2 or min_scale == max_scale:
        scales = [max_scale]
    else:
        scales = np.linspace(min_scale, max_scale, num_scales).tolist()
    if min_scale <= 1.0 <= max_scale:
        scales.append(1.0)
    return sorted(set(round(s, 3) for s in scales if s > 0))


# --- multi-scale chamfer match with silhouette fallbacks ---
def _multi_scale_chamfer_match(
    template_gray, piece_gray, template_dt=None, angles=None, scales=None
):
    if template_dt is None:
        _, dt = _compute_template_distance_transform(template_gray)
    else:
        dt = template_dt
    if angles is None:
        angles = [0, -15, 15, 30, -30, 45, -45, 90]
    if scales is None:
        scales = _scale_candidates(piece_gray.shape, template_gray.shape)
    best = None

    # crop piece to mask bbox for speed
    init_mask = (piece_gray > 5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        init_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)
        piece_crop = piece_gray[by : by + bh, bx : bx + bw]
    else:
        piece_crop = piece_gray.copy()

    for angle in angles:
        rotated = _rotate_image(piece_crop, angle)
        rgray = (
            rotated if rotated.ndim == 2 else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        )
        _, rmask = cv2.threshold(rgray, 5, 255, cv2.THRESH_BINARY)
        # robust silhouette extraction (this is the key fix)
        silhouette = _extract_silhouette_from_mask(rmask, gray_crop=rgray)
        if (
            silhouette is None
            or silhouette.size == 0
            or cv2.countNonZero(silhouette) < 6
        ):
            # skip this rotation if silhouette couldn't be derived
            _debug_log(f"Rotation {angle}: silhouette empty -> skipping")
            continue
        for scale in scales:
            pw = max(3, int(silhouette.shape[1] * scale))
            ph = max(3, int(silhouette.shape[0] * scale))
            if pw > template_gray.shape[1] or ph > template_gray.shape[0]:
                continue
            candidate_edges = cv2.resize(
                silhouette, (pw, ph), interpolation=cv2.INTER_NEAREST
            )
            if candidate_edges.shape[0] < 4 or candidate_edges.shape[1] < 4:
                continue
            try:
                cx, cy, mean_dist, count, top_left = _chamfer_match_template(
                    dt, candidate_edges, search_step=8
                )
            except ValueError:
                continue
            score = 1.0 / (1.0 + mean_dist)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "cx": cx,
                    "cy": cy,
                    "angle": angle,
                    "scale": scale,
                    "mean_dist": mean_dist,
                    "count": count,
                    "top_left": top_left,
                    "size": (pw, ph),
                }
                _debug_log(
                    f"New best: angle={angle} scale={scale:.3f} score={score:.4f} mean_dist={mean_dist:.3f}"
                )
                if _debug_enabled():
                    tlx, tly = top_left
                    overlay = cv2.cvtColor(template_gray, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(
                        overlay, (tlx, tly), (tlx + pw, tly + ph), (0, 255, 0), 2
                    )
                    _debug_save_image("chamfer_candidate_overlay", overlay)
    if best is None:
        raise ValueError("Chamfer matching found no candidate")

    # refine local search
    tlx, tly = best["top_left"]
    pw, ph = best["size"]
    rotated_best = _rotate_image(piece_crop, best["angle"])
    rgray_best = (
        rotated_best
        if rotated_best.ndim == 2
        else cv2.cvtColor(rotated_best, cv2.COLOR_BGR2GRAY)
    )
    _, rmask_best = cv2.threshold(rgray_best, 5, 255, cv2.THRESH_BINARY)
    silhouette_best = _extract_silhouette_from_mask(rmask_best, gray_crop=rgray_best)
    candidate_edges = cv2.resize(
        silhouette_best, (pw, ph), interpolation=cv2.INTER_NEAREST
    )
    best_ref = None
    # iterate in small neighborhood +/-8 px
    for dy in range(max(0, tly - 8), min(dt.shape[0] - ph, tly + 8) + 1):
        for dx in range(max(0, tlx - 8), min(dt.shape[1] - pw, tlx + 8) + 1):
            res = _chamfer_score_at_location(dt, candidate_edges, (dx, dy))
            if res is None:
                continue
            mean_dist, count = res
            score = 1.0 / (1.0 + mean_dist)
            if best_ref is None or score > best_ref["score"]:
                best_ref = {
                    "score": score,
                    "cx": dx + pw // 2,
                    "cy": dy + ph // 2,
                    "mean_dist": mean_dist,
                    "top_left": (dx, dy),
                }
    if best_ref:
        _debug_log(
            f"Chamfer refined: score={best_ref['score']:.4f} mean_dist={best_ref['mean_dist']:.3f}"
        )
        return int(best_ref["cx"]), int(best_ref["cy"]), float(best_ref["score"])
    _debug_log(
        f"Chamfer final (coarse) score={best['score']:.4f} mean_dist={best['mean_dist']:.3f}"
    )
    return int(best["cx"]), int(best["cy"]), float(best["score"])


# --- public API ---
def find_piece_in_template(piece_image_path, template_image_path):
    _debug_reset(Path(str(piece_image_path)).stem)
    piece = cv2.imread(str(piece_image_path))
    template = cv2.imread(str(template_image_path))
    if piece is None or template is None:
        raise ValueError("Could not load images")
    _debug_save_image("piece_raw", piece)
    _debug_save_image("template_raw", template)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.equalizeHist(template_gray)
    _, template_dt = _compute_template_distance_transform(template_gray)
    processed_piece = _preprocess_piece_image(piece)
    piece_gray = (
        processed_piece
        if processed_piece.ndim == 2
        else cv2.cvtColor(processed_piece, cv2.COLOR_BGR2GRAY)
    )
    cx, cy, score = _multi_scale_chamfer_match(
        template_gray, piece_gray, template_dt=template_dt
    )
    _debug_log(f"Chamfer result: ({cx},{cy}) score={score:.4f}")
    if _debug_enabled():
        highlight = highlight_position(template_image_path, cx, cy)
        _debug_save_image(
            "final_highlight", cv2.cvtColor(np.array(highlight), cv2.COLOR_RGB2BGR)
        )
    return int(cx), int(cy), float(score)


def highlight_position(template_image_path, x, y, radius=30):
    tpl = cv2.imread(str(template_image_path))
    if tpl is None:
        raise ValueError("Could not load template")
    cv2.circle(tpl, (int(x), int(y)), radius, (0, 255, 0), 3)
    cv2.circle(tpl, (int(x), int(y)), radius + 5, (255, 255, 0), 2)
    tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)
    return Image.fromarray(tpl_rgb)
