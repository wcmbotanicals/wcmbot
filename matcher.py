"""
Modern puzzle matcher that mirrors the high-performance pipeline from 1.py.
Exposes helper utilities so the UI can render the debug-style plots without
needing to reproduce image-processing logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ---------- configuration ----------
COLS = 36
ROWS = 28
PIECE_CELLS_APPROX = (1, 1)
EST_SCALE_WINDOW = np.linspace(0.8, 1.2, num=11).tolist()
ROTATIONS = [0, 90, 180, 270]
TOP_MATCH_COUNT = 5
TOP_MATCH_SCAN_MULT = 50

KNOB_WIDTH_FRAC = 1.0 / 3.0
CANNY_LOW = 50
CANNY_HIGH = 150
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MATCH_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

LOWER_BLUE1 = np.array([90, 60, 40], dtype=np.uint8)
UPPER_BLUE1 = np.array([140, 255, 255], dtype=np.uint8)
LOWER_BLUE2 = np.array([85, 30, 60], dtype=np.uint8)
UPPER_BLUE2 = np.array([160, 255, 220], dtype=np.uint8)

OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_MASK_AREA_FRAC = 0.0005


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


@dataclass
class TemplateCacheEntry:
    mtime: float
    template_bgr: np.ndarray
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
        template_bgr=template,
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


def _binarize_two_color(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw // 255).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Crop failed: empty mask")
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _keep_largest_component(mask01: np.ndarray, min_frac: float = MIN_MASK_AREA_FRAC) -> np.ndarray:
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask01)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area = cv2.contourArea(contours[0])
    h, w = mask01.shape[:2]
    if area < min_frac * (h * w):
        return np.zeros_like(mask01)
    out = np.zeros_like(mask255)
    cv2.drawContours(out, [contours[0]], -1, 255, thickness=-1)
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


def _crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = _mask_bbox(mask)
    return img[y0:y1, x0:x1].copy()


def _rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


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


def _update_top_matches(top_matches: List[Dict], candidate: Dict, max_len: int = TOP_MATCH_COUNT) -> None:
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

    def _scan_candidates(order: np.ndarray, res_w: int, ws: int, hs: int) -> List[Dict]:
        combo_best_local: List[Dict] = []
        for idx in order:
            if len(combo_best_local) >= TOP_MATCH_COUNT:
                break
            y, x = divmod(int(idx), res_w)
            cx = x + ws / 2
            cy = y + hs / 2
            tl = (int(x), int(y))
            br = (int(x + ws), int(y + hs))
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

            res = cv2.matchTemplate(
                T_blur_f32, patt_masked, corr_method
            )

            if res.size == 0:
                continue

            res_h, res_w = res.shape
            flat = res.ravel()
            if flat.size <= TOP_MATCH_COUNT:
                order = np.argsort(flat)[::-1]
            else:
                top_k = min(
                    flat.size,
                    max(TOP_MATCH_COUNT * TOP_MATCH_SCAN_MULT, TOP_MATCH_COUNT),
                )
                order = np.argpartition(flat, -top_k)[-top_k:]
                order = order[np.argsort(flat[order])[::-1]]

            combo_best = _scan_candidates(order, res_w, ws, hs)
            if len(combo_best) < TOP_MATCH_COUNT and order.size < flat.size:
                # Fallback to full ordering only if partial scan could not yield
                # enough distinct matches.
                order = np.argsort(flat)[::-1]
                combo_best = _scan_candidates(order, res_w, ws, hs)
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
    return (img01.astype(np.uint8) * 255).astype(np.uint8)


def _create_resized_preview(piece_bin: np.ndarray, piece_mask: np.ndarray, match: Dict) -> np.ndarray:
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
    knobs_x: int,
    knobs_y: int,
) -> MatchPayload:
    template_entry = _load_template_cached(template_image_path)
    piece = _load_image(piece_image_path)

    template_rgb = template_entry.template_rgb
    template_bin = template_entry.template_bin

    piece_mask = _mask_by_blue(piece)
    y0, y1, x0, x1 = _mask_bbox(piece_mask)
    piece_crop = piece[y0:y1, x0:x1].copy()
    piece_mask_crop = piece_mask[y0:y1, x0:x1].copy()
    piece_bin = _binarize_two_color(piece_crop) * piece_mask_crop
    piece_rgb = cv2.cvtColor(piece_crop, cv2.COLOR_BGR2RGB)

    th, tw = template_bin.shape
    cell_w = tw / COLS
    cell_h = th / ROWS
    desired_core_w = cell_w * PIECE_CELLS_APPROX[0]
    desired_core_h = cell_h * PIECE_CELLS_APPROX[1]
    desired_full_w = desired_core_w * (1.0 + knobs_x * KNOB_WIDTH_FRAC)
    desired_full_h = desired_core_h * (1.0 + knobs_y * KNOB_WIDTH_FRAC)

    mh, mw = piece_mask_crop.shape
    if mw == 0 or mh == 0:
        raise RuntimeError("Piece mask has zero size")

    est_scale_w = desired_full_w / mw
    est_scale_h = desired_full_h / mh
    piece_area_px = piece_mask_crop.sum()
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
    scales = [est_scale * f for f in EST_SCALE_WINDOW]

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

    top_matches = _attach_contours_to_matches(
        top_matches, piece_mask_crop, MATCH_DILATE_KERNEL
    )
    for idx, match in enumerate(top_matches):
        match["index"] = idx
        match["cx"] = int(round(match["center"][0]))
        match["cy"] = int(round(match["center"][1]))
        match["width"] = match["br"][0] - match["tl"][0]
        match["height"] = match["br"][1] - match["tl"][1]

    return MatchPayload(
        template_rgb=template_rgb,
        template_bin=template_bin,
        piece_rgb=piece_rgb,
        piece_mask=piece_mask_crop,
        piece_bin=piece_bin,
        matches=top_matches,
        template_shape=template_bin.shape,
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


def render_primary_views(payload: MatchPayload, match_index: int) -> Dict[str, np.ndarray]:
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
    return (
        f"Match #{idx + 1} / {len(payload.matches)}  \n"
        f"Score: {match['score']:.3f} | Rotation: {match['rot']}° | "
        f"Scale: {match['scale']:.4f}  \n"
        f"Grid position: row {match['row']}, col {match['col']}"
    )


def highlight_position(template_image_path: str, x: int, y: int, radius: int = 30) -> np.ndarray:
    tpl = cv2.imread(template_image_path)
    if tpl is None:
        raise ValueError("Could not load template")
    cv2.circle(tpl, (int(x), int(y)), radius, (0, 255, 0), 3)
    cv2.circle(tpl, (int(x), int(y)), radius + 5, (255, 255, 0), 2)
    tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)
    return tpl_rgb
