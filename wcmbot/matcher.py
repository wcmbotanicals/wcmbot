"""
Modern puzzle matcher that mirrors the high-performance pipeline from 1.py.
Exposes helper utilities so the UI can render the debug-style plots without
needing to reproduce image-processing logic.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)

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
GRID_CENTER_WEIGHT = 0.03
KNOB_WIDTH_FRAC = 1.0 / 3.0
CROP_X_PX = 0
CROP_Y_PX = 0
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MATCH_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

LOWER_BLUE1 = np.array([90, 60, 40], dtype=np.uint8)
UPPER_BLUE1 = np.array([140, 255, 255], dtype=np.uint8)
LOWER_BLUE2 = np.array([85, 30, 60], dtype=np.uint8)
UPPER_BLUE2 = np.array([160, 255, 220], dtype=np.uint8)
LOWER_GREEN1 = np.array([35, 40, 40], dtype=np.uint8)
UPPER_GREEN1 = np.array([85, 255, 255], dtype=np.uint8)

OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_MASK_AREA_FRAC = 0.0005
AUTO_ALIGN_MIN_DEG = 0.0
AUTO_ALIGN_MAX_DEG = 45.0
AUTO_ALIGN_MIN_LINES = 4
AUTO_ALIGN_MIN_AREA_FRAC = -0.1  # only reject if bbox gets substantially worse
AUTO_ALIGN_REFERENCE_SIZE = 1024  # Reference size for normalizing images before masking
AUTO_ALIGN_HOUGH_THRESHOLD_FRAC = 0.035
AUTO_ALIGN_HOUGH_MIN_LINE_FRAC = 0.028
AUTO_ALIGN_HOUGH_MAX_GAP_FRAC = 0.014
INFER_KNOBS_TIE_EPS = 0.01
INFER_KNOBS_LOW_FILL = 0.50
INFER_KNOBS_HIGH_FILL = 0.65
BINARIZE_BLUR_KSZ = (5, 5)
MATCH_BLUR_KSZ = (3, 3)
RESIZE_RETHRESHOLD = False
PIECE_BG_PAD_FRAC = 0.03


# ---------- helper dataclasses ----------
@dataclass(frozen=True)
class MatcherConfig:
    cols: int = COLS
    rows: int = ROWS
    piece_cells_approx: Tuple[float, float] = PIECE_CELLS_APPROX
    est_scale_window: List[float] = field(
        default_factory=lambda: list(EST_SCALE_WINDOW)
    )
    rotations: List[int] = field(default_factory=lambda: list(ROTATIONS))
    top_match_count: int = TOP_MATCH_COUNT
    top_match_scan_multiplier: int = TOP_MATCH_SCAN_MULTIPLIER
    coarse_factor: float = COARSE_FACTOR
    coarse_top_k: int = COARSE_TOP_K
    coarse_padding_pixels: int = COARSE_PADDING_PIXELS
    coarse_min_side: int = COARSE_MIN_SIDE
    preserve_edges_coarse: bool = False
    parallel_matching: bool = True
    grid_center_weight: float = GRID_CENTER_WEIGHT
    knob_width_frac: float = KNOB_WIDTH_FRAC
    crop_x: int = CROP_X_PX
    crop_y: int = CROP_Y_PX
    binarize_blur_ksz: Optional[Tuple[int, int]] = BINARIZE_BLUR_KSZ
    resize_rethreshold: bool = RESIZE_RETHRESHOLD
    match_blur_ksz: Optional[Tuple[int, int]] = MATCH_BLUR_KSZ
    mask_mode: str = "blue"
    mask_hsv_ranges: Optional[List[Tuple[List[int], List[int]]]] = None
    mask_kernel_size: int = 7
    mask_open_iters: int = OPEN_ITERS
    mask_close_iters: int = CLOSE_ITERS


def build_matcher_config(
    overrides: Optional[Dict[str, object]] = None,
) -> MatcherConfig:
    payload = {
        "cols": COLS,
        "rows": ROWS,
        "piece_cells_approx": PIECE_CELLS_APPROX,
        "est_scale_window": list(EST_SCALE_WINDOW),
        "rotations": list(ROTATIONS),
        "top_match_count": TOP_MATCH_COUNT,
        "top_match_scan_multiplier": TOP_MATCH_SCAN_MULTIPLIER,
        "coarse_factor": COARSE_FACTOR,
        "coarse_top_k": COARSE_TOP_K,
        "coarse_padding_pixels": COARSE_PADDING_PIXELS,
        "coarse_min_side": COARSE_MIN_SIDE,
        "preserve_edges_coarse": False,
        "parallel_matching": True,
        "grid_center_weight": GRID_CENTER_WEIGHT,
        "knob_width_frac": KNOB_WIDTH_FRAC,
        "crop_x": CROP_X_PX,
        "crop_y": CROP_Y_PX,
        "binarize_blur_ksz": BINARIZE_BLUR_KSZ,
        "resize_rethreshold": RESIZE_RETHRESHOLD,
        "match_blur_ksz": MATCH_BLUR_KSZ,
        "mask_mode": "blue",
        "mask_hsv_ranges": None,
        "mask_kernel_size": 7,
        "mask_open_iters": OPEN_ITERS,
        "mask_close_iters": CLOSE_ITERS,
    }
    if not overrides:
        return MatcherConfig(**payload)
    for key, value in overrides.items():
        if key not in payload:
            raise ValueError(f"Unknown matcher override: {key}")
        if key in ("binarize_blur_ksz", "match_blur_ksz"):
            payload[key] = _normalize_kernel(value)
            continue
        payload[key] = value
    return MatcherConfig(**payload)


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
    resize_rethreshold: bool = False


@dataclass
class TemplateCacheEntry:
    mtime: float
    template_rgb: np.ndarray
    template_bin: np.ndarray
    blur_cache: Dict[Optional[Tuple[int, int]], np.ndarray]
    binarize_blur_ksz: Optional[Tuple[int, int]]
    crop_x: int = 0
    crop_y: int = 0


_TEMPLATE_CACHE: Dict[
    Tuple[str, Optional[Tuple[int, int]], int, int], TemplateCacheEntry
] = {}


# ---------- helpers ----------
def _normalize_kernel(
    kernel: Tuple[int, int] | List[int] | None,
) -> Optional[Tuple[int, int]]:
    if kernel is None:
        return None
    if isinstance(kernel, (list, tuple)) and len(kernel) == 2:
        return int(kernel[0]), int(kernel[1])
    raise ValueError("binarize_blur_ksz must be a 2-item list/tuple or null.")


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _load_template_cached(
    path: str,
    binarize_blur_ksz: Optional[Tuple[int, int]] = BINARIZE_BLUR_KSZ,
    crop_x: int = CROP_X_PX,
    crop_y: int = CROP_Y_PX,
) -> TemplateCacheEntry:
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
    blur_ksz = _normalize_kernel(binarize_blur_ksz)
    crop_x = int(crop_x)
    crop_y = int(crop_y)
    cache_key = (path, blur_ksz, crop_x, crop_y)
    entry = _TEMPLATE_CACHE.get(cache_key)
    if entry and entry.mtime == mtime:
        return entry
    template = _load_image(path)
    if crop_x < 0 or crop_y < 0:
        raise ValueError("crop_x and crop_y must be non-negative.")
    if crop_x or crop_y:
        h, w = template.shape[:2]
        if crop_x * 2 >= w or crop_y * 2 >= h:
            raise ValueError(
                f"Template crop too large for {w}x{h}: crop_x={crop_x} crop_y={crop_y}"
            )
        template = template[crop_y : h - crop_y, crop_x : w - crop_x]
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template_bin = _binarize_two_color(template, blur_ksz)
    entry = TemplateCacheEntry(
        mtime=mtime,
        template_rgb=template_rgb,
        template_bin=template_bin,
        blur_cache={},
        binarize_blur_ksz=blur_ksz,
        crop_x=crop_x,
        crop_y=crop_y,
    )
    _TEMPLATE_CACHE[cache_key] = entry
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
    template_image_path: str,
    blur_ksz: Optional[Tuple[int, int]] = MATCH_BLUR_KSZ,
    binarize_blur_ksz: Optional[Tuple[int, int]] = BINARIZE_BLUR_KSZ,
    crop_x: int = CROP_X_PX,
    crop_y: int = CROP_Y_PX,
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
    entry = _load_template_cached(
        template_image_path,
        binarize_blur_ksz=binarize_blur_ksz,
        crop_x=crop_x,
        crop_y=crop_y,
    )
    _get_template_blur_f32(entry.template_bin, blur_ksz, entry.blur_cache)


def _enhance_contrast_gray(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to stabilize thresholding across lighting variations."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_coarse_resize(
    img: np.ndarray, factor: float, preserve_edges: bool = False
) -> np.ndarray:
    """
    Resize image for coarse matching, optionally preserving high-frequency detail.

    For grass puzzles and other fine-detail templates, enable preserve_edges=True
    to use a milder edge-preservation strategy during downsampling.

    Args:
        img: Input image to resize.
        factor: Scaling factor (< 1.0 for downsampling).
        preserve_edges: If True, use mild unsharp mask before downsampling to preserve detail.

    Returns:
        Resized image.
    """
    if factor >= 1.0:
        return img

    if not preserve_edges:
        return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    # Mild unsharp mask: enhances edges without over-sharpening or creating artifacts
    # This is gentler than a sharpening kernel and preserves grass blades without distortion
    blurred = cv2.GaussianBlur(img, (3, 3), 1.0)
    # Unsharp mask formula: sharpened = original * (1 + weight) + blurred * (-weight)
    # Weight of 0.3 provides mild enhancement without overflow/underflow.
    # cv2.addWeighted clamps values to [0, 255] for uint8 or valid range for float32.
    sharpened = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)

    # Use INTER_AREA for high-quality downsampling after mild sharpening
    return cv2.resize(
        sharpened, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
    )


def _binarize_two_color(
    img_bgr: np.ndarray, blur_ksz: Optional[Tuple[int, int]] = BINARIZE_BLUR_KSZ
) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _enhance_contrast_gray(gray)
    if blur_ksz is not None:
        gray = cv2.GaussianBlur(gray, blur_ksz, 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw // 255).astype(np.uint8)


def _binarize_median_threshold(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    blur_ksz: Optional[Tuple[int, int]] = BINARIZE_BLUR_KSZ,
) -> np.ndarray:
    """Binarize using median intensity of masked region as threshold.

    This ensures roughly 50% of the piece is black and 50% white, preserving
    internal pattern contrast regardless of absolute brightness.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _enhance_contrast_gray(gray)
    if blur_ksz is not None:
        gray = cv2.GaussianBlur(gray, blur_ksz, 0)
    gray_processed = gray

    if mask is not None:
        # Find median of masked pixels only
        masked_pixels = gray_processed[mask > 0]
        if len(masked_pixels) > 0:
            threshold = int(np.median(masked_pixels))
        else:
            threshold = 127
    else:
        # Use overall median
        threshold = int(np.median(gray_processed))

    _, bw = cv2.threshold(gray_processed, threshold, 255, cv2.THRESH_BINARY)
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


def _cleanup_mask(
    mask: np.ndarray, kernel_size: int, open_iters: int, close_iters: int
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    h, w = mask.shape[:2]
    pad = max(1, kernel_size)
    y0 = max(0, ys.min() - pad)
    y1 = min(h, ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(w, xs.max() + 1 + pad)
    cropped = mask[y0:y1, x0:x1]
    if open_iters > 0:
        cropped = cv2.morphologyEx(
            cropped, cv2.MORPH_OPEN, kernel, iterations=open_iters
        )
    if close_iters > 0:
        cropped = cv2.morphologyEx(
            cropped, cv2.MORPH_CLOSE, kernel, iterations=close_iters
        )
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = cropped
    return out


def _mask_by_blue(
    piece_bgr: np.ndarray,
    kernel_size: int,
    open_iters: int,
    close_iters: int,
    keep_largest_component: bool = True,
) -> np.ndarray:
    hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, LOWER_BLUE1, UPPER_BLUE1)
    m2 = cv2.inRange(hsv, LOWER_BLUE2, UPPER_BLUE2)
    mask = cv2.bitwise_or(m1, m2)
    mask = _cleanup_mask(mask, kernel_size, open_iters, close_iters)
    mask01 = (mask > 0).astype(np.uint8)
    if keep_largest_component:
        mask01 = _keep_largest_component(mask01)
    if mask01.sum() == 0:
        raise RuntimeError(
            "Blue segmentation produced empty mask - tune HSV ranges or check image"
        )
    return mask01


def _mask_by_green(
    piece_bgr: np.ndarray,
    kernel_size: int,
    open_iters: int,
    close_iters: int,
    keep_largest_component: bool = True,
) -> np.ndarray:
    hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN1, UPPER_GREEN1)
    mask = _cleanup_mask(mask, kernel_size, open_iters, close_iters)
    mask01 = (mask > 0).astype(np.uint8)
    if keep_largest_component:
        mask01 = _keep_largest_component(mask01)
    if mask01.sum() == 0:
        raise RuntimeError(
            "Green segmentation produced empty mask - tune HSV ranges or check image"
        )
    return mask01


def _mask_by_hsv_ranges(
    piece_bgr: np.ndarray,
    ranges: List[Tuple[List[int], List[int]]],
    kernel_size: int,
    open_iters: int,
    close_iters: int,
    keep_largest_component: bool = True,
) -> np.ndarray:
    hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in ranges:
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        curr = cv2.inRange(hsv, lower_arr, upper_arr)
        mask = curr if mask is None else cv2.bitwise_or(mask, curr)
    if mask is None:
        raise RuntimeError("HSV segmentation requires at least one range.")
    mask = _cleanup_mask(mask, kernel_size, open_iters, close_iters)
    mask01 = (mask > 0).astype(np.uint8)
    if keep_largest_component:
        mask01 = _keep_largest_component(mask01)
    if mask01.sum() == 0:
        raise RuntimeError(
            "HSV segmentation produced empty mask - tune HSV ranges or check image"
        )
    return mask01


def compute_piece_mask(
    piece_bgr: np.ndarray, config: MatcherConfig, keep_largest_component: bool = True
) -> np.ndarray:
    """Compute a binary mask for a puzzle piece based on color mode.

    Supports "blue", "green", or "hsv"/"hsv_ranges" modes. Returns a binary
    mask (0 or 1) with the piece foreground isolated at the input image size.

    Args:
        piece_bgr: BGR image of the piece.
        config: MatcherConfig with mask settings.
        keep_largest_component: If True, keep only the largest connected component.
                               If False, keep all detected foreground. Defaults to True.

    Returns:
        np.ndarray: Binary mask of the same height and width as ``piece_bgr``,
            with ``dtype`` ``uint8`` and values 0 or 1, where 1 indicates the
            piece foreground and 0 indicates background.
    """
    mask_mode = (config.mask_mode or "blue").lower()
    kernel_size = int(config.mask_kernel_size)
    open_iters = int(config.mask_open_iters)
    close_iters = int(config.mask_close_iters)
    if mask_mode == "blue":
        return _mask_by_blue(
            piece_bgr, kernel_size, open_iters, close_iters, keep_largest_component
        )
    if mask_mode == "green":
        return _mask_by_green(
            piece_bgr, kernel_size, open_iters, close_iters, keep_largest_component
        )
    if mask_mode in ("hsv", "hsv_ranges"):
        if not config.mask_hsv_ranges:
            raise RuntimeError("mask_hsv_ranges must be set for hsv mask mode.")
        return _mask_by_hsv_ranges(
            piece_bgr,
            config.mask_hsv_ranges,
            kernel_size,
            open_iters,
            close_iters,
            keep_largest_component,
        )
    raise RuntimeError(f"Unknown mask_mode: {config.mask_mode}")


def crop_image_to_mask(
    img_bgr: np.ndarray, mask01: np.ndarray, pad_frac: float = 0.05, min_pad: int = 0
) -> np.ndarray:
    """Crop image tightly to the mask content with optional padding.

    Args:
        img_bgr: Input image.
        mask01: Binary mask (0 or 1) indicating foreground.
        pad_frac: Fraction of bounding box to add as padding.
        min_pad: Minimum padding in pixels.

    Returns:
        np.ndarray: Cropped image containing only the masked region with padding.

    Raises:
        RuntimeError: If the provided mask contains no foreground pixels.
    """
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        raise RuntimeError(
            "Empty mask passed to crop_image_to_mask; no foreground pixels found."
        )

    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1

    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    pad = max(min_pad, int(round(min(bbox_h, bbox_w) * pad_frac)))

    y_start = max(0, y_min - pad)
    y_end = min(img_bgr.shape[0], y_max + pad)
    x_start = max(0, x_min - pad)
    x_end = min(img_bgr.shape[1], x_max + pad)

    return img_bgr[y_start:y_end, x_start:x_end]


def _compute_piece_mask_for_alignment(
    piece_bgr: np.ndarray, config: MatcherConfig, keep_largest_component: bool = True
) -> np.ndarray:
    """Compute a normalized mask for auto-alignment.

    Resizes the input so the shortest side equals AUTO_ALIGN_REFERENCE_SIZE before
    masking. The returned mask is at the normalized size to keep Hough parameters
    consistent regardless of the original image dimensions.
    """
    h, w = piece_bgr.shape[:2]
    min_dim = max(1, min(h, w))
    if min_dim != AUTO_ALIGN_REFERENCE_SIZE:
        scale = AUTO_ALIGN_REFERENCE_SIZE / min_dim
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        piece_bgr = cv2.resize(
            piece_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    return compute_piece_mask(
        piece_bgr, config, keep_largest_component=keep_largest_component
    )


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


def _pad_piece_image(
    piece_bgr: np.ndarray,
    pad_frac: float = PIECE_BG_PAD_FRAC,
    min_pad: int = 4,
) -> np.ndarray:
    h, w = piece_bgr.shape[:2]
    if h == 0 or w == 0:
        return piece_bgr
    pad_px = max(min_pad, int(round(min(h, w) * pad_frac)))
    bg = _background_bgr(piece_bgr)
    return cv2.copyMakeBorder(
        piece_bgr,
        pad_px,
        pad_px,
        pad_px,
        pad_px,
        cv2.BORDER_CONSTANT,
        value=bg,
    )


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


def _normalize_template_rotation(rotation: Optional[int]) -> int:
    if rotation is None:
        return 0
    if rotation % 90 != 0:
        raise ValueError("Template rotation must be a multiple of 90 degrees.")
    rotation = rotation % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Template rotation must be 0, 90, 180, or 270 degrees.")
    return rotation


def _rotate_template_quadrant(img: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return img
    k = -(rotation // 90)
    return np.rot90(img, k=k)


def _mask_bbox_area(mask01: np.ndarray) -> int:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return 0
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def _estimate_mask_tilt(mask01: np.ndarray) -> Tuple[Optional[float], int]:
    """
    Estimate tilt angle using Hough line detection on mask edges.

    The mask is expected to be normalized to a reference size by the caller
    (for example via _compute_piece_mask_for_alignment()), so proportional
    Hough parameters produce consistent results regardless of original image size.
    Detected line angles are aggregated using a mean with IQR-based outlier
    rejection to obtain a robust tilt estimate.
    """
    edges = cv2.Canny(mask01.astype(np.uint8) * 255, 50, 150)
    h, w = mask01.shape[:2]
    min_dim = max(1, min(h, w))
    min_line = max(1, int(round(min_dim * AUTO_ALIGN_HOUGH_MIN_LINE_FRAC)))
    max_gap = max(1, int(round(min_dim * AUTO_ALIGN_HOUGH_MAX_GAP_FRAC)))
    threshold = max(1, int(round(min_dim * AUTO_ALIGN_HOUGH_THRESHOLD_FRAC)))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_line,
        maxLineGap=max_gap,
    )
    if lines is None:
        return None, 0
    angles: List[float] = []
    lengths: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_line * 0.5:
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

    # Crop mask to bounding box before estimating tilt to reduce empty-border bias
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return 0.0
    mask_crop = mask01[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    # Normalize cropped mask to reference size so Hough parameters are consistent.
    crop_h, crop_w = mask_crop.shape[:2]
    crop_min = max(1, min(crop_h, crop_w))
    if crop_min != AUTO_ALIGN_REFERENCE_SIZE:
        scale = AUTO_ALIGN_REFERENCE_SIZE / crop_min
        norm_h = max(1, int(round(crop_h * scale)))
        norm_w = max(1, int(round(crop_w * scale)))
        mask_for_tilt = cv2.resize(
            mask_crop, (norm_w, norm_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask_for_tilt = mask_crop
    angle, line_count = _estimate_mask_tilt(mask_for_tilt)

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
    config: MatcherConfig,
) -> Tuple[float, List[float]]:
    th, tw = template_shape
    cell_w = tw / config.cols
    cell_h = th / config.rows
    desired_core_w = cell_w * config.piece_cells_approx[0]
    desired_core_h = cell_h * config.piece_cells_approx[1]
    desired_full_w = desired_core_w * (1.0 + knobs_x * config.knob_width_frac)
    desired_full_h = desired_core_h * (1.0 + knobs_y * config.knob_width_frac)

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
    scales = [est_scale * f for f in config.est_scale_window]
    valid_scales: List[float] = []
    for scale in scales:
        ws = int(round(mw * scale))
        hs = int(round(mh * scale))
        if ws <= 0 or hs <= 0 or ws >= tw or hs >= th:
            continue
        valid_scales.append(scale)
    if not valid_scales:
        raise RuntimeError(
            "Estimated scale produced no valid candidates; "
            f"est_scale={est_scale:.4f} piece_size=({mw}x{mh}) "
            f"template_size=({tw}x{th}) knobs=({knobs_x},{knobs_y})"
        )
    return est_scale, valid_scales


def _infer_knob_counts(
    piece_mask: np.ndarray,
    template_shape: Tuple[int, int],
    config: MatcherConfig,
) -> Tuple[int, int]:
    mh, mw = piece_mask.shape
    if mw == 0 or mh == 0:
        return 0, 0

    piece_area_px = float(piece_mask.sum())
    if piece_area_px <= 0:
        return 0, 0
    fill_ratio = piece_area_px / float(mw * mh)

    th, tw = template_shape
    cell_w = tw / config.cols
    cell_h = th / config.rows
    desired_core_w = cell_w * config.piece_cells_approx[0]
    desired_core_h = cell_h * config.piece_cells_approx[1]
    desired_area_px = desired_core_w * desired_core_h

    scored: List[Tuple[float, int, int]] = []
    for kx in range(3):
        for ky in range(3):
            desired_full_w = desired_core_w * (1.0 + kx * config.knob_width_frac)
            desired_full_h = desired_core_h * (1.0 + ky * config.knob_width_frac)
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


def _rotate_knob_counts(
    knobs_x: Optional[int], knobs_y: Optional[int], rotation: int
) -> Tuple[Optional[int], Optional[int]]:
    if knobs_x is None or knobs_y is None:
        return knobs_x, knobs_y
    if rotation % 180 == 90:
        return knobs_y, knobs_x
    return knobs_x, knobs_y


def _core_center_from_mask(
    mask01: np.ndarray,
    knobs_x: Optional[int],
    knobs_y: Optional[int],
) -> Tuple[float, float]:
    if mask01.size == 0:
        return 0.0, 0.0
    mask = (mask01 > 0).astype(np.uint8)
    mh, mw = mask.shape
    if mh == 0 or mw == 0:
        return float(mw) / 2.0, float(mh) / 2.0

    kx = max(0, int(knobs_x)) if knobs_x is not None else 0
    ky = max(0, int(knobs_y)) if knobs_y is not None else 0
    max_knobs = min(2, max(kx, ky))
    ratio = max(0.45, 0.55 - (0.05 * max_knobs))

    col_sum = mask.sum(axis=0)
    row_sum = mask.sum(axis=1)
    if col_sum.max() <= 0 or row_sum.max() <= 0:
        return float(mw) / 2.0, float(mh) / 2.0
    # Use the widest high-coverage span to drop protruding tabs.
    col_thresh = ratio * float(col_sum.max())
    row_thresh = ratio * float(row_sum.max())
    left = int(np.argmax(col_sum >= col_thresh))
    right = int(mw - 1 - np.argmax(col_sum[::-1] >= col_thresh))
    top = int(np.argmax(row_sum >= row_thresh))
    bottom = int(mh - 1 - np.argmax(row_sum[::-1] >= row_thresh))

    cx = (left + right + 1) / 2.0
    cy = (top + bottom + 1) / 2.0
    return cx, cy


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


def _grid_center_proximity(
    cx: float, cy: float, cell_w: float, cell_h: float, cols: int, rows: int
) -> float:
    if cell_w <= 0 or cell_h <= 0 or cols <= 0 or rows <= 0:
        return 0.0
    col_idx = int(round(cx / cell_w - 0.5))
    row_idx = int(round(cy / cell_h - 0.5))
    col_idx = max(0, min(col_idx, cols - 1))
    row_idx = max(0, min(row_idx, rows - 1))
    center_x = (col_idx + 0.5) * cell_w
    center_y = (row_idx + 0.5) * cell_h
    dx = cx - center_x
    dy = cy - center_y
    max_dist = 0.5 * float(np.hypot(cell_w, cell_h))
    if max_dist <= 0:
        return 0.0
    dist = float(np.hypot(dx, dy))
    return max(0.0, 1.0 - min(dist / max_dist, 1.0))


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


def _match_scale_rotation_combo(
    scale: float,
    rot: int,
    piece_bin_pattern: np.ndarray,
    piece_mask: np.ndarray,
    template_blur_f32: np.ndarray,
    template_coarse_f32: Optional[np.ndarray],
    config: MatcherConfig,
    knobs_x: Optional[int],
    knobs_y: Optional[int],
    blur_ksz: Optional[Tuple[int, int]],
    corr_method: int,
    cols: int,
    rows: int,
) -> List[Dict]:
    """
    Match a single scale/rotation combination with optional coarse-then-fine.

    This function is extracted to enable parallel execution via ThreadPoolExecutor.
    Supports coarse-then-fine optimization when template_coarse_f32 is provided.
    Returns a list of match candidates for this specific configuration.
    """
    P = (
        (piece_bin_pattern * 255).astype(np.uint8)
        if piece_bin_pattern.max() <= 1
        else piece_bin_pattern.astype(np.uint8)
    )
    if piece_mask.max() <= 1:
        M = (piece_mask > 0).astype(np.uint8) * 255
    else:
        M = (piece_mask > 127).astype(np.uint8) * 255

    th, tw = template_blur_f32.shape[:2]
    cell_w = tw / cols
    cell_h = th / rows

    dilate_ker = MATCH_DILATE_KERNEL
    top_match_count = config.top_match_count
    top_match_scan_multiplier = config.top_match_scan_multiplier
    grid_center_weight = config.grid_center_weight
    coarse_factor = config.coarse_factor
    coarse_top_k = config.coarse_top_k
    coarse_padding_pixels = config.coarse_padding_pixels

    P_r = _rotate_img(P, rot)
    M_r = _rotate_img(M, rot)
    M_r_raw01 = (M_r > 127).astype(np.uint8)
    rot_knobs_x, rot_knobs_y = _rotate_knob_counts(knobs_x, knobs_y, rot)
    core_center = _core_center_from_mask(M_r_raw01, rot_knobs_x, rot_knobs_y)
    M_r = (M_r_raw01 > 0).astype(np.uint8) * 255
    M_r = cv2.morphologyEx(M_r, cv2.MORPH_DILATE, dilate_ker, iterations=1)
    M_r01 = (M_r > 127).astype(np.float32)

    ws = int(round(P_r.shape[1] * scale))
    hs = int(round(P_r.shape[0] * scale))
    if ws <= 0 or hs <= 0 or ws >= tw or hs >= th:
        return []

    scale_x = ws / float(P_r.shape[1])
    scale_y = hs / float(P_r.shape[0])
    core_offset_x = core_center[0] * scale_x
    core_offset_y = core_center[1] * scale_y

    patt_s = _resize_for_match(P_r, ws, hs, rethreshold=config.resize_rethreshold)
    mask_s = _resize_for_match(M_r01, ws, hs)

    if blur_ksz is not None:
        patt_s_blur = cv2.GaussianBlur(patt_s, blur_ksz, 0).astype(np.float32)
    else:
        patt_s_blur = patt_s.astype(np.float32)

    patt_masked = patt_s_blur * mask_s

    def _candidate_order(flat: np.ndarray, max_len: int) -> np.ndarray:
        if flat.size <= max_len:
            return np.argsort(flat)[::-1]
        scan_count = min(flat.size, max(max_len * top_match_scan_multiplier, max_len))
        order = np.argpartition(flat, -scan_count)[-scan_count:]
        return order[np.argsort(flat[order])[::-1]]

    def _scan_candidates(
        res: np.ndarray,
        order: np.ndarray,
        res_w: int,
        ws: int,
        hs: int,
        offset_x: int,
        offset_y: int,
        core_offset_x: float,
        core_offset_y: float,
    ) -> List[Dict]:
        combo_best_local: List[Dict] = []
        for idx in order:
            if len(combo_best_local) >= top_match_count:
                break
            y, x = divmod(int(idx), res_w)
            x0 = x + offset_x
            y0 = y + offset_y
            cx = x0 + core_offset_x
            cy = y0 + core_offset_y
            base_score = float(res[y, x])
            proximity = _grid_center_proximity(cx, cy, cell_w, cell_h, cols, rows)
            score = base_score + (grid_center_weight * proximity)
            tl = (int(x0), int(y0))
            br = (int(x0 + ws), int(y0 + hs))
            candidate = {
                "score": score,
                "score_raw": base_score,
                "grid_score": proximity,
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

    # Decide whether to use coarse-then-fine
    # ws < tw and hs < th are already guaranteed by the check at line 1133
    use_coarse = template_coarse_f32 is not None and 0.0 < coarse_factor < 1.0

    if use_coarse:
        # Coarse pass: find approximate locations
        th_c, tw_c = template_coarse_f32.shape[:2]
        ws_c = max(1, int(round(ws * coarse_factor)))
        hs_c = max(1, int(round(hs * coarse_factor)))

        if 1 < ws_c < tw_c and 1 < hs_c < th_c:
            patt_c = _adaptive_coarse_resize(
                patt_s_blur, coarse_factor, config.preserve_edges_coarse
            )
            # Ensure exact dimensions
            if patt_c.shape[:2] != (hs_c, ws_c):
                patt_c = cv2.resize(patt_c, (ws_c, hs_c), interpolation=cv2.INTER_AREA)
            mask_c = cv2.resize(mask_s, (ws_c, hs_c), interpolation=cv2.INTER_NEAREST)
            patt_masked_c = patt_c * mask_c

            res_c = cv2.matchTemplate(template_coarse_f32, patt_masked_c, corr_method)

            if res_c.size > 0:
                # Find top-K coarse candidates
                flat_c = res_c.ravel()
                order_c = _candidate_order(flat_c, coarse_top_k)
                res_w_c = res_c.shape[1]

                # Fine pass on each coarse candidate
                all_fine_candidates = []
                for idx in order_c[:coarse_top_k]:
                    y_c, x_c = divmod(int(idx), res_w_c)
                    # Map coarse position back to full resolution
                    x_full = int(round(x_c / coarse_factor))
                    y_full = int(round(y_c / coarse_factor))
                    x_full = max(0, min(x_full, tw - ws))
                    y_full = max(0, min(y_full, th - hs))

                    # Define padded ROI around coarse match
                    x0 = max(0, x_full - coarse_padding_pixels)
                    y0 = max(0, y_full - coarse_padding_pixels)
                    x1 = min(tw, x_full + ws + coarse_padding_pixels)
                    y1 = min(th, y_full + hs + coarse_padding_pixels)

                    roi = template_blur_f32[y0:y1, x0:x1]
                    if roi.shape[0] < hs or roi.shape[1] < ws:
                        continue

                    # Fine match in ROI
                    res_fine = cv2.matchTemplate(roi, patt_masked, corr_method)
                    if res_fine.size == 0:
                        continue

                    flat_fine = res_fine.ravel()
                    order_fine = _candidate_order(flat_fine, top_match_count)
                    res_w_fine = res_fine.shape[1]

                    fine_candidates = _scan_candidates(
                        res_fine,
                        order_fine,
                        res_w_fine,
                        ws,
                        hs,
                        x0,
                        y0,
                        core_offset_x,
                        core_offset_y,
                    )
                    all_fine_candidates.extend(fine_candidates)

                # Only short-circuit if the coarse refinement produced candidates.
                # If none were found, fall back to full-resolution matching below.
                if all_fine_candidates:
                    return all_fine_candidates

    # Fallback: full-resolution matching without coarse pass
    res = cv2.matchTemplate(template_blur_f32, patt_masked, corr_method)

    if res.size == 0:
        return []

    flat = res.ravel()
    order = _candidate_order(flat, top_match_count)
    res_w = res.shape[1]
    return _scan_candidates(
        res, order, res_w, ws, hs, 0, 0, core_offset_x, core_offset_y
    )


def _match_template_multiscale_binary(
    template_bin_img: np.ndarray,
    piece_bin_pattern: np.ndarray,
    piece_mask: np.ndarray,
    cols: int,
    rows: int,
    scales: List[float],
    rotations: List[int],
    config: MatcherConfig,
    knobs_x: Optional[int] = None,
    knobs_y: Optional[int] = None,
    blur_ksz: Optional[Tuple[int, int]] = (3, 3),
    corr_method: int = cv2.TM_CCORR_NORMED,
    template_blur_f32: Optional[np.ndarray] = None,
) -> Tuple[Dict, List[Dict]]:
    top_match_count = config.top_match_count
    top_match_scan_multiplier = config.top_match_scan_multiplier
    coarse_factor = config.coarse_factor
    coarse_top_k = config.coarse_top_k
    coarse_padding_pixels = config.coarse_padding_pixels
    coarse_min_side = config.coarse_min_side
    grid_center_weight = config.grid_center_weight

    # Check if we should use parallel execution
    # Threshold of 8 configs balances parallelization benefits vs. thread creation overhead.
    # Below 8, the overhead of thread management exceeds the performance gains.
    num_configs = len(scales) * len(rotations)
    use_parallel = config.parallel_matching and num_configs >= 8

    if use_parallel:
        # Parallel path: use ThreadPoolExecutor for scale/rotation combinations

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

        # Prepare coarse template if needed
        th, tw = T_blur_f32.shape[:2]
        use_coarse = 0.0 < coarse_factor < 1.0 and min(tw, th) >= coarse_min_side
        if use_coarse:
            T_coarse_blur = _adaptive_coarse_resize(
                T_blur_f32, coarse_factor, config.preserve_edges_coarse
            )
        else:
            T_coarse_blur = None

        all_candidates = []
        # Determine an appropriate number of worker threads based on CPU count
        total_tasks = len(scales) * len(rotations)
        cpu_count = os.cpu_count() or 1
        max_workers = min(total_tasks, cpu_count)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _match_scale_rotation_combo,
                    scale,
                    rot,
                    piece_bin_pattern,
                    piece_mask,
                    T_blur_f32,
                    T_coarse_blur,
                    config,
                    knobs_x,
                    knobs_y,
                    blur_ksz,
                    corr_method,
                    cols,
                    rows,
                )
                for scale in scales
                for rot in rotations
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_candidates.extend(result)

                    # Early termination if we find an excellent match
                    if result:
                        max_score = max(c["score"] for c in result)
                        if max_score > 0.85:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            # Ensure all futures have finished or been cancelled
                            for f in futures:
                                try:
                                    # Block until the future completes or raises,
                                    # ignoring any exceptions here since failures
                                    # are handled elsewhere.
                                    f.result()
                                except Exception:
                                    pass
                            break
                except Exception as e:
                    # Log but don't crash on individual match failures
                    logger.warning("Parallel match failed: %s", e)
                    continue

        if not all_candidates:
            raise RuntimeError("No match found (binary matcher - parallel)")

        all_candidates.sort(key=lambda c: c["score"], reverse=True)
        top_matches: List[Dict] = []
        for candidate in all_candidates:
            _update_top_matches(top_matches, candidate, top_match_count)
        best = top_matches[0]
        return best, top_matches

    # Sequential path: original implementation with coarse-then-fine
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

    use_coarse = 0.0 < coarse_factor < 1.0 and min(tw, th) >= coarse_min_side
    if use_coarse:
        tw_c = max(1, int(round(tw * coarse_factor)))
        th_c = max(1, int(round(th * coarse_factor)))
        if tw_c < 2 or th_c < 2:
            use_coarse = False
            T_coarse_blur = None
        else:
            T_coarse_blur = _adaptive_coarse_resize(
                T_blur_f32, coarse_factor, config.preserve_edges_coarse
            )
    else:
        T_coarse_blur = None

    def _candidate_order(flat: np.ndarray, max_len: int) -> np.ndarray:
        if flat.size <= max_len:
            return np.argsort(flat)[::-1]
        scan_count = min(flat.size, max(max_len * top_match_scan_multiplier, max_len))
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
        core_offset_x: Optional[float] = None,
        core_offset_y: Optional[float] = None,
    ) -> List[Dict]:
        combo_best_local: List[Dict] = []
        if core_offset_x is None:
            core_offset_x = ws / 2
        if core_offset_y is None:
            core_offset_y = hs / 2
        for idx in order:
            if len(combo_best_local) >= top_match_count:
                break
            y, x = divmod(int(idx), res_w)
            x0 = x + offset_x
            y0 = y + offset_y
            cx = x0 + core_offset_x
            cy = y0 + core_offset_y
            base_score = float(res[y, x])
            proximity = _grid_center_proximity(cx, cy, cell_w, cell_h, cols, rows)
            score = base_score + (grid_center_weight * proximity)
            tl = (int(x0), int(y0))
            br = (int(x0 + ws), int(y0 + hs))
            candidate = {
                "score": score,
                "score_raw": base_score,
                "grid_score": proximity,
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
        core_offset_x: Optional[float] = None,
        core_offset_y: Optional[float] = None,
    ) -> List[Dict]:
        flat = res.ravel()
        order = _candidate_order(flat, top_match_count)
        res_w = res.shape[1]
        combo_best = _scan_candidates(
            res,
            order,
            res_w,
            ws,
            hs,
            offset_x=offset_x,
            offset_y=offset_y,
            core_offset_x=core_offset_x,
            core_offset_y=core_offset_y,
        )
        if len(combo_best) < top_match_count and order.size < flat.size:
            order = np.argsort(flat)[::-1]
            combo_best = _scan_candidates(
                res,
                order,
                res_w,
                ws,
                hs,
                offset_x=offset_x,
                offset_y=offset_y,
                core_offset_x=core_offset_x,
                core_offset_y=core_offset_y,
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
        M_r_raw01 = (M_r > 127).astype(np.uint8)
        rot_knobs_x, rot_knobs_y = _rotate_knob_counts(knobs_x, knobs_y, rot)
        core_center = _core_center_from_mask(M_r_raw01, rot_knobs_x, rot_knobs_y)
        M_r = (M_r_raw01 > 0).astype(np.uint8) * 255
        M_r = cv2.morphologyEx(M_r, cv2.MORPH_DILATE, dilate_ker, iterations=1)
        M_r01 = (M_r > 127).astype(np.float32)

        for scale in scales:
            ws = int(round(P_r.shape[1] * scale))
            hs = int(round(P_r.shape[0] * scale))
            if ws <= 0 or hs <= 0 or ws >= tw or hs >= th:
                continue
            scale_x = ws / float(P_r.shape[1])
            scale_y = hs / float(P_r.shape[0])
            core_offset_x = core_center[0] * scale_x
            core_offset_y = core_center[1] * scale_y

            patt_s = _resize_for_match(
                P_r, ws, hs, rethreshold=config.resize_rethreshold
            )
            mask_s = _resize_for_match(M_r01, ws, hs)

            if blur_ksz is not None:
                patt_s_blur = cv2.GaussianBlur(patt_s, blur_ksz, 0).astype(np.float32)
            else:
                patt_s_blur = patt_s.astype(np.float32)

            patt_masked = patt_s_blur * mask_s
            combo_added = False

            if use_coarse and T_coarse_blur is not None:
                ws_c = max(1, int(round(ws * coarse_factor)))
                hs_c = max(1, int(round(hs * coarse_factor)))
                if (
                    1 < ws_c < T_coarse_blur.shape[1]
                    and 1 < hs_c < T_coarse_blur.shape[0]
                ):
                    patt_c = _adaptive_coarse_resize(
                        patt_s_blur, coarse_factor, config.preserve_edges_coarse
                    )
                    # Resize to exact dimensions after sharpening
                    if patt_c.shape[:2] != (hs_c, ws_c):
                        patt_c = cv2.resize(
                            patt_c, (ws_c, hs_c), interpolation=cv2.INTER_AREA
                        )
                    mask_c = cv2.resize(
                        mask_s, (ws_c, hs_c), interpolation=cv2.INTER_NEAREST
                    )
                    patt_masked_c = patt_c * mask_c
                    res_c = cv2.matchTemplate(T_coarse_blur, patt_masked_c, corr_method)
                    if res_c.size:
                        coarse_positions = _collect_coarse_positions(
                            res_c, ws_c, hs_c, coarse_top_k
                        )
                        seen_rois = set()
                        for coarse in coarse_positions:
                            x_c, y_c = coarse["tl"]
                            x_full = int(round(x_c / coarse_factor))
                            y_full = int(round(y_c / coarse_factor))
                            x_full = max(0, min(x_full, tw - ws))
                            y_full = max(0, min(y_full, th - hs))
                            x0 = max(0, x_full - coarse_padding_pixels)
                            y0 = max(0, y_full - coarse_padding_pixels)
                            x1 = min(tw, x_full + ws + coarse_padding_pixels)
                            y1 = min(th, y_full + hs + coarse_padding_pixels)
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
                                res,
                                ws,
                                hs,
                                offset_x=x0,
                                offset_y=y0,
                                core_offset_x=core_offset_x,
                                core_offset_y=core_offset_y,
                            )
                            if combo_best:
                                combo_candidates.extend(combo_best)
                                combo_added = True

            if not combo_added:
                res = cv2.matchTemplate(T_blur_f32, patt_masked, corr_method)

                if res.size == 0:
                    continue

                combo_best = _collect_matches(
                    res,
                    ws,
                    hs,
                    core_offset_x=core_offset_x,
                    core_offset_y=core_offset_y,
                )
                combo_candidates.extend(combo_best)

    if not combo_candidates:
        raise RuntimeError("No match found (binary matcher)")

    combo_candidates.sort(key=lambda c: c["score"], reverse=True)
    top_matches: List[Dict] = []
    for candidate in combo_candidates:
        _update_top_matches(top_matches, candidate, top_match_count)
    best = top_matches[0]
    return best, top_matches


def _ensure_three_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img


def _binary_to_uint8(img01: np.ndarray) -> np.ndarray:
    return img01.astype(np.uint8) * 255


def _resize_for_match(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    rethreshold: bool = False,
) -> np.ndarray:
    h, w = img.shape[:2]
    if target_w == w and target_h == h:
        return img
    if target_w < w or target_h < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    resized = cv2.resize(img, (target_w, target_h), interpolation=interp)
    if rethreshold:
        if resized.dtype != np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        resized = (resized > 127).astype(np.uint8) * 255
    return resized


def _create_resized_preview(
    piece_bin: np.ndarray,
    piece_mask: np.ndarray,
    match: Dict,
    rethreshold: bool = False,
) -> np.ndarray:
    rot = match["rot"]
    rot_bin = _rotate_img(_binary_to_uint8(piece_bin), rot)
    rot_mask = _rotate_img(_binary_to_uint8(piece_mask), rot)
    ws = max(1, match["br"][0] - match["tl"][0])
    hs = max(1, match["br"][1] - match["tl"][1])
    rv = _resize_for_match(rot_bin, ws, hs, rethreshold=rethreshold)
    rv_mask = _resize_for_match(rot_mask, ws, hs)
    rv = (rv * (rv_mask > 127)).astype(np.uint8)
    return rv


def _render_masked_piece_view(
    piece_rgb: np.ndarray, piece_mask: np.ndarray, match: Dict
) -> np.ndarray:
    rot = match["rot"]
    mask01 = (piece_mask > 0).astype(np.uint8)
    masked_rgb = piece_rgb.copy()
    masked_rgb[mask01 == 0] = 255
    rot_rgb = _rotate_img(
        masked_rgb,
        rot,
        interpolation=cv2.INTER_LINEAR,
        border_value=(255, 255, 255),
    )
    rot_mask = _rotate_img(
        mask01 * 255,
        rot,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )

    target_h = max(1, match["br"][1] - match["tl"][1])
    target_w = max(1, match["br"][0] - match["tl"][0])
    if rot_rgb.shape[0] != target_h or rot_rgb.shape[1] != target_w:
        interp = (
            cv2.INTER_AREA
            if rot_rgb.shape[0] > target_h or rot_rgb.shape[1] > target_w
            else cv2.INTER_LINEAR
        )
        rot_rgb = cv2.resize(rot_rgb, (target_w, target_h), interpolation=interp)
        rot_mask = cv2.resize(
            rot_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    piece_view = rot_rgb.copy()
    piece_view[rot_mask <= 127] = 255
    return piece_view


def _combine_side_by_side(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    gap: int = 4,
    background: int = 255,
    min_height: int = 0,
    max_width: int = 0,
) -> Optional[np.ndarray]:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left

    left = _ensure_three_channel(left)
    right = _ensure_three_channel(right)
    if left.dtype != np.uint8:
        left = np.clip(left, 0, 255).astype(np.uint8)
    if right.dtype != np.uint8:
        right = np.clip(right, 0, 255).astype(np.uint8)

    h1 = left.shape[0]
    h2 = right.shape[0]
    target_h = max(h1, h2)
    if min_height > 0:
        target_h = max(target_h, min_height)

    def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == height:
            return img
        new_w = max(1, int(round(w * height / h)))
        interp = cv2.INTER_AREA if height < h else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, height), interpolation=interp)

    left = _resize_to_height(left, target_h)
    right = _resize_to_height(right, target_h)

    total_w = left.shape[1] + gap + right.shape[1]
    canvas = np.full((target_h, total_w, 3), background, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    x_off = left.shape[1] + gap
    canvas[: right.shape[0], x_off : x_off + right.shape[1]] = right
    if max_width > 0 and canvas.shape[1] > max_width:
        scale = max_width / canvas.shape[1]
        new_h = max(1, int(round(canvas.shape[0] * scale)))
        canvas = cv2.resize(canvas, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return canvas


def _render_zoom_image(
    template_rgb: np.ndarray,
    template_shape: Tuple[int, int],
    piece_bin: np.ndarray,
    piece_mask: np.ndarray,
    match: Dict,
    zoom: int = 98,
    overlay_piece: bool = False,
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

    if overlay_piece:
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
        mask = cv2.resize(
            rot_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
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

            if (
                dst_x1 > dst_x0
                and dst_y1 > dst_y0
                and src_x1 > src_x0
                and src_y1 > src_y0
            ):
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
    template_rotation: Optional[int] = None,
    matcher_config: Optional[MatcherConfig] = None,
) -> MatchPayload:
    config = matcher_config or build_matcher_config()
    profile_value = os.getenv(PROFILE_ENV, "").strip().lower()
    profile = profile_value not in ("", "0", "false", "no")
    if profile:
        t0 = time.perf_counter()
        marks: List[Tuple[str, float]] = []

    template_entry = _load_template_cached(
        template_image_path,
        config.binarize_blur_ksz,
        crop_x=config.crop_x,
        crop_y=config.crop_y,
    )
    if profile:
        marks.append(("template", time.perf_counter()))

    piece = _load_image(piece_image_path)
    piece = _pad_piece_image(piece)
    if profile:
        marks.append(("piece", time.perf_counter()))

    template_rgb = template_entry.template_rgb
    template_bin = template_entry.template_bin
    rotation = _normalize_template_rotation(template_rotation)
    template_blur_cache = template_entry.blur_cache
    if rotation:
        template_rgb = _rotate_template_quadrant(template_rgb, rotation)
        template_bin = _rotate_template_quadrant(template_bin, rotation)
        template_blur_cache = {}

    piece_mask = compute_piece_mask(piece, config)
    if profile:
        marks.append(("mask", time.perf_counter()))

    y0, y1, x0, x1 = _mask_bbox(piece_mask)
    piece_crop = piece[y0:y1, x0:x1].copy()
    piece_mask_crop = piece_mask[y0:y1, x0:x1].copy()
    if profile:
        marks.append(("crop", time.perf_counter()))

    piece_bin = (
        _binarize_median_threshold(
            piece_crop, piece_mask_crop, config.binarize_blur_ksz
        )
        * piece_mask_crop
    )
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
        align_mask = _compute_piece_mask_for_alignment(piece, config)
        correction = _estimate_alignment_from_mask(align_mask)
        if abs(correction) >= AUTO_ALIGN_MIN_DEG:
            bg = _background_bgr(piece)
            piece = _rotate_img(
                piece,
                correction,
                interpolation=cv2.INTER_LINEAR,
                border_value=bg,
            )
            auto_align_deg = correction
            piece_mask = compute_piece_mask(piece, config)
            y0, y1, x0, x1 = _mask_bbox(piece_mask)
            piece_crop = piece[y0:y1, x0:x1].copy()
            piece_mask_crop = piece_mask[y0:y1, x0:x1].copy()
            piece_bin = (
                _binarize_median_threshold(
                    piece_crop, piece_mask_crop, config.binarize_blur_ksz
                )
                * piece_mask_crop
            )
            piece_rgb = cv2.cvtColor(piece_crop, cv2.COLOR_BGR2RGB)
        if profile:
            marks.append(("auto_align", time.perf_counter()))

    knobs_inferred = False
    if infer_knobs_enabled:
        knobs_x, knobs_y = _infer_knob_counts(
            piece_mask_crop,
            template_bin.shape,
            config,
        )
        knobs_inferred = True
        if profile:
            marks.append(("knob_infer", time.perf_counter()))
    else:
        knobs_x = int(knobs_x)
        knobs_y = int(knobs_y)

    _, scales = _estimate_scales(
        template_bin.shape, piece_mask_crop, knobs_x, knobs_y, config
    )
    if profile:
        marks.append(("scale", time.perf_counter()))

    template_blur_f32 = _get_template_blur_f32(
        template_bin, config.match_blur_ksz, template_blur_cache
    )
    _, top_matches = _match_template_multiscale_binary(
        template_bin,
        piece_bin,
        piece_mask_crop,
        config.cols,
        config.rows,
        scales,
        config.rotations,
        config,
        knobs_x=knobs_x,
        knobs_y=knobs_y,
        blur_ksz=config.match_blur_ksz,
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
            parts.append(f"{label}={(ts - prev) * 1000.0:.2f}ms")
            prev = ts
        parts.append(f"total={(t_end - t0) * 1000.0:.2f}ms")
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
        resize_rethreshold=config.resize_rethreshold,
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
        _create_resized_preview(
            payload.piece_bin,
            payload.piece_mask,
            match,
            rethreshold=payload.resize_rethreshold,
        )
    )
    zoom = _render_zoom_image(
        payload.template_rgb,
        payload.template_shape,
        payload.piece_bin,
        payload.piece_mask,
        match,
        zoom=98,
    )
    piece_view = _render_masked_piece_view(payload.piece_rgb, payload.piece_mask, match)
    zoom_pair_view = _render_zoom_image(
        payload.template_rgb,
        payload.template_shape,
        payload.piece_bin,
        payload.piece_mask,
        match,
        zoom=100,
    )
    zoom_pair = _combine_side_by_side(
        piece_view,
        zoom_pair_view,
        min_height=300,
        max_width=900,
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
            "zoom_piece": piece_view,
            "zoom_focus": zoom,
            "zoom_pair": zoom_pair,
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
