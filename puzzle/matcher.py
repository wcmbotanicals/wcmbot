# puzzle/matcher.py
"""
Template matching-based puzzle matcher using edge detection.
Main features:
 - Edge-based template matching with cv2.matchTemplate
 - Saturation-based background removal for pieces
 - Multi-scale and multi-rotation matching
 - Debug image saving (PUZZLE_MATCHER_DEBUG=1)
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

# Configuration
COLS = 36
ROWS = 28
PIECE_CELLS_APPROX = (1, 1)
EST_SCALE_WINDOW = [0.9, 0.95, 1.0, 1.05, 1.1]
ROTATIONS = [0, 90, 180, 270]
CANNY_LOW = 50
CANNY_HIGH = 150
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

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


# --- Core helper functions ---
def _load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _binarize_two_color(img_bgr):
    """Convert image to binary using Otsu's threshold."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw // 255).astype(np.uint8)


def _remove_background_by_saturation(piece_bgr, sat_thresh=40):
    """Remove background using HSV saturation channel."""
    hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    mask = (s > sat_thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Piece segmentation failed")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    final = np.zeros_like(mask)
    cv2.drawContours(final, [contours[0]], -1, 255, thickness=-1)
    return (final // 255).astype(np.uint8)


def _crop_to_mask(img, mask):
    """Crop image to bounding box of mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Crop failed: empty mask")
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()


def _rotate_img(img, angle):
    """Rotate image by given angle (degrees)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


# --- Template matching (strict) ---
def _match_template_multiscale_strict(
    template_img, piece_pattern, piece_mask, cols, rows, scales, rotations
):
    """
    Perform multi-scale, multi-rotation template matching.
    Returns best match with grid coordinates, score, rotation, and scale.
    """
    th, tw = template_img.shape[:2]
    cell_w = tw / cols
    cell_h = th / rows

    best = None
    grid_heat = np.full((rows, cols), np.nan, dtype=float)
    rot_map = np.full((rows, cols), -1, dtype=int)

    for rot in rotations:
        patt_r = _rotate_img(piece_pattern, rot)
        mask_r = _rotate_img(piece_mask, rot)
        mask_r = (mask_r > 127).astype(np.uint8) * 255

        for scale in scales:
            ws = int(round(patt_r.shape[1] * scale))
            hs = int(round(patt_r.shape[0] * scale))

            if ws <= 0 or hs <= 0:
                _debug_log(f"Invalid scale {scale} produced non-positive patch size")
                continue

            if ws >= template_img.shape[1] or hs >= template_img.shape[0]:
                _debug_log(f"Scaled piece larger than template at scale {scale}")
                continue

            patt_s = cv2.resize(patt_r, (ws, hs), interpolation=cv2.INTER_LINEAR)
            mask_s = cv2.resize(mask_r, (ws, hs), interpolation=cv2.INTER_NEAREST)

            res = cv2.matchTemplate(
                template_img, patt_s, cv2.TM_CCORR_NORMED, mask=mask_s
            )
            _, maxv, _, maxloc = cv2.minMaxLoc(res)

            if best is None or maxv > best["score"]:
                cx = maxloc[0] + ws / 2
                cy = maxloc[1] + hs / 2
                best = {
                    "score": float(maxv),
                    "rot": rot,
                    "scale": scale,
                    "col": int(cx / cell_w),
                    "row": int(cy / cell_h),
                    "tl": (int(maxloc[0]), int(maxloc[1])),
                    "br": (int(maxloc[0] + ws), int(maxloc[1] + hs)),
                    "cx": int(cx),
                    "cy": int(cy),
                }
                _debug_log(
                    f"New best: rot={rot} scale={scale:.3f} score={maxv:.4f} at ({int(cx)},{int(cy)})"
                )

            # Coarse sampling into grid_heat
            res_h, res_w = res.shape
            step_y = max(1, res_h // 50)
            step_x = max(1, res_w // 50)
            for y in range(0, res_h, step_y):
                for x in range(0, res_w, step_x):
                    cx_g = x + ws / 2
                    cy_g = y + hs / 2
                    col = int(cx_g / cell_w)
                    row = int(cy_g / cell_h)
                    if 0 <= col < cols and 0 <= row < rows:
                        val = float(res[y, x])
                        if np.isnan(grid_heat[row, col]) or val > grid_heat[row, col]:
                            grid_heat[row, col] = val
                            rot_map[row, col] = rot

    if best is None:
        raise RuntimeError("No match found")

    _debug_save_image("grid_heat", (grid_heat * 255).astype(np.uint8))
    return best, grid_heat, rot_map


def highlight_position(template_image_path, x, y, radius=30):
    tpl = cv2.imread(str(template_image_path))
    if tpl is None:
        raise ValueError("Could not load template")
    cv2.circle(tpl, (int(x), int(y)), radius, (0, 255, 0), 3)
    cv2.circle(tpl, (int(x), int(y)), radius + 5, (255, 255, 0), 2)
    tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)
    return Image.fromarray(tpl_rgb)


# --- Public API ---
def find_piece_in_template(piece_image_path, template_image_path):
    """
    Find piece position in template using edge-based template matching.
    Returns (cx, cy, score) - center coordinates and match score.
    """
    _debug_reset(Path(str(piece_image_path)).stem)

    # Load images
    template = _load_image(template_image_path)
    piece = _load_image(piece_image_path)

    _debug_save_image("piece_raw", piece)
    _debug_save_image("template_raw", template)

    # Template processing
    template_bin = _binarize_two_color(template)
    template_edges = cv2.Canny(
        (template_bin * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH
    )
    _debug_save_image("template_bin", template_bin * 255)
    _debug_save_image("template_edges", template_edges)

    # Piece segmentation (saturation-based)
    piece_mask = _remove_background_by_saturation(piece, sat_thresh=40)
    piece_crop = _crop_to_mask(piece, piece_mask)
    piece_mask_crop = _crop_to_mask((piece_mask * 255).astype(np.uint8), piece_mask)
    piece_mask_crop = (piece_mask_crop > 0).astype(np.uint8)

    _debug_save_image("piece_crop", piece_crop)
    _debug_save_image("piece_mask", piece_mask_crop * 255)

    # Piece binarization and edges
    piece_bin = _binarize_two_color(piece_crop) * piece_mask_crop
    piece_edges = cv2.Canny((piece_bin * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)

    _debug_save_image("piece_bin", piece_bin * 255)
    _debug_save_image("piece_edges", piece_edges)

    # Scale estimation
    th, tw = template.shape[:2]
    cell_w = tw / COLS
    cell_h = th / ROWS
    desired_w = cell_w * PIECE_CELLS_APPROX[0]
    desired_h = cell_h * PIECE_CELLS_APPROX[1]

    ph, pw = piece_edges.shape
    if pw == 0 or ph == 0:
        raise RuntimeError("Piece crop has zero size")

    est_scale = ((desired_w / pw) + (desired_h / ph)) / 2.0
    if not (0.05 < est_scale < 10.0):
        raise RuntimeError(
            f"Estimated scale {est_scale:.3f} is implausible - check configuration"
        )

    scales = [est_scale * f for f in EST_SCALE_WINDOW]
    _debug_log(f"Estimated scale={est_scale:.3f}, trying scales={scales}")

    # Multi-scale template matching
    best, grid_heat, rot_map = _match_template_multiscale_strict(
        template_edges.astype(np.uint8),
        piece_edges.astype(np.uint8),
        (piece_mask_crop * 255).astype(np.uint8),
        COLS,
        ROWS,
        scales,
        ROTATIONS,
    )

    _debug_log(
        f"Best match: col={best['col']} row={best['row']} rot={best['rot']} "
        f"scale={best['scale']:.3f} score={best['score']:.4f} pos=({best['cx']},{best['cy']})"
    )

    # Save debug visualization
    if _debug_enabled():
        highlight = highlight_position(template_image_path, best["cx"], best["cy"])
        _debug_save_image(
            "final_highlight", cv2.cvtColor(np.array(highlight), cv2.COLOR_RGB2BGR)
        )

    return int(best["cx"]), int(best["cy"]), float(best["score"])
