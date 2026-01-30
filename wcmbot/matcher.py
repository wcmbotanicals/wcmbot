"""
Modern puzzle matcher that mirrors the high-performance pipeline from 1.py.
Exposes helper utilities so the UI can render the debug-style plots without
needing to reproduce image-processing logic.
"""

from __future__ import annotations

import logging
import math
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
PROFILE_ENV = "WCMBOT_PROFILE"
USE_TORCH_ENV = "WCMBOT_USE_TORCH"
TORCH_DEVICE_ENV = "WCMBOT_TORCH_DEVICE"
TORCH_JIT_ENV = "WCMBOT_TORCH_JIT"
COARSE_FACTOR_ENV = "WCMBOT_COARSE_FACTOR"
PARALLEL_MATCHING_ENV = "WCMBOT_PARALLEL_MATCHING"
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

# Magic number thresholds for mask refinement
# These thresholds control the behavior of mask shape refinement with template clustering
MASK_FILL_RATIO_THRESHOLD = (
    0.84  # Minimum fill ratio before applying aggressive clustering
)
MASK_FILL_RATIO_MULTIPLIER = (
    0.35  # Multiplier for scale adjustment based on fill deficit
)
MASK_FILL_SCALE_FACTOR = 0.8  # Scaling factor for fill ratio adjustment
MASK_ALLOW_GROWTH_THRESHOLD = 0.86  # Fill ratio threshold for allowing mask growth
MASK_AGGRESSIVE_GROWTH_THRESHOLD = 0.9  # Fill ratio threshold for aggressive growth
BG_DISTANCE_THRESHOLD = 2  # Distance threshold offset for background detection
BORDER_DISTANCE_PERCENTILE = 8  # Border distance percentile for edge recovery


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
    multipiece_mask_mode: Optional[str] = None  # If set, use this mode for multipiece
    mask_hsv_ranges: Optional[List[Tuple[List[int], List[int]]]] = None
    mask_kernel_size: int = 7
    mask_open_iters: int = OPEN_ITERS
    mask_close_iters: int = CLOSE_ITERS
    mask_shape_refine: bool = False
    mask_skip: bool = False  # If True, assume already background-removed (all fg)
    template_clustering: bool = False
    template_cluster_k: int = 4
    template_cluster_percentile: float = 98.0
    template_cluster_scale: float = 1.15
    render_full_res: bool = True
    use_torch: bool = False
    torch_device: Optional[str] = None


def build_matcher_config(
    overrides: Optional[Dict[str, object]] = None,
) -> MatcherConfig:
    use_torch_env = os.getenv(USE_TORCH_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    torch_device_env = os.getenv(TORCH_DEVICE_ENV, "").strip() or None

    parallel_matching = True
    parallel_env = os.getenv(PARALLEL_MATCHING_ENV, "").strip().lower()
    if parallel_env:
        parallel_matching = parallel_env in {"1", "true", "yes", "on"}

    coarse_factor = COARSE_FACTOR
    coarse_factor_env = os.getenv(COARSE_FACTOR_ENV, "").strip()
    if coarse_factor_env:
        try:
            coarse_factor = float(coarse_factor_env)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {COARSE_FACTOR_ENV}={coarse_factor_env!r}; expected a float"
            ) from exc

    forced_keys: set[str] = set()
    if parallel_env:
        forced_keys.add("parallel_matching")
    if coarse_factor_env:
        forced_keys.add("coarse_factor")

    payload = {
        "cols": COLS,
        "rows": ROWS,
        "piece_cells_approx": PIECE_CELLS_APPROX,
        "est_scale_window": list(EST_SCALE_WINDOW),
        "rotations": list(ROTATIONS),
        "top_match_count": TOP_MATCH_COUNT,
        "top_match_scan_multiplier": TOP_MATCH_SCAN_MULTIPLIER,
        "coarse_factor": coarse_factor,
        "coarse_top_k": COARSE_TOP_K,
        "coarse_padding_pixels": COARSE_PADDING_PIXELS,
        "coarse_min_side": COARSE_MIN_SIDE,
        "preserve_edges_coarse": False,
        "parallel_matching": parallel_matching,
        "use_torch": use_torch_env,
        "torch_device": torch_device_env,
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
        "mask_shape_refine": False,
        "template_clustering": False,
        "template_cluster_k": 4,
        "template_cluster_percentile": 98.0,
        "template_cluster_scale": 1.15,
        "render_full_res": True,
    }
    if not overrides:
        return MatcherConfig(**payload)
    for key, value in overrides.items():
        if key in forced_keys:
            continue
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
    render_full_res: bool = True


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
    t_u8 = (
        (template_bin * 255).astype(np.uint8)
        if template_bin.max() <= 1
        else template_bin.astype(np.uint8)
    )
    if blur_ksz is not None:
        t_blur = cv2.GaussianBlur(t_u8, blur_ksz, 0)
    else:
        t_blur = t_u8.copy()
    t_blur_f32 = t_blur.astype(np.float32)
    blur_cache[blur_ksz] = t_blur_f32
    return t_blur_f32


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

    blurred = cv2.GaussianBlur(img, (3, 3), 1.0)
    sharpened = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)
    return cv2.resize(
        sharpened, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
    )


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _default_torch_device() -> str:
    import torch

    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def assert_torch_accel_available() -> str:
    """Return the best torch accelerator device or raise if none exist."""
    if not _torch_available():
        raise RuntimeError("--gpu requested but PyTorch is not available.")
    device = _default_torch_device()
    if device == "cpu":
        raise RuntimeError(
            "--gpu requested but no PyTorch MPS/CUDA device is available; the --gpu flag requires a GPU/MPS accelerator."
        )
    return device


def _match_template_torch_ccorr_normed(
    template_f32: np.ndarray,
    patch_f32: np.ndarray,
    device: Optional[str] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Torch implementation of cv2.TM_CCORR_NORMED for float32 arrays."""

    import torch
    import torch.nn.functional as F

    if template_f32.size == 0 or patch_f32.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    th, tw = int(template_f32.shape[0]), int(template_f32.shape[1])
    ph, pw = int(patch_f32.shape[0]), int(patch_f32.shape[1])
    if ph > th or pw > tw:
        return np.empty((0, 0), dtype=np.float32)

    dev = device or _default_torch_device()
    t = torch.from_numpy(np.ascontiguousarray(template_f32)).to(torch.float32)[
        None, None
    ]
    p = torch.from_numpy(np.ascontiguousarray(patch_f32)).to(torch.float32)[None, None]
    t = t.to(dev)
    p = p.to(dev)

    ones = torch.ones((1, 1, ph, pw), device=dev, dtype=torch.float32)
    with torch.no_grad():
        num = F.conv2d(t, p)
        sum_t2 = torch.sum(p * p)
        sum_i2 = F.conv2d(t * t, ones)
        den = torch.sqrt(sum_i2 * sum_t2).clamp_min(eps)
        out = (num / den).to(dtype=torch.float32)
        return (
            out.squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def _torch_gaussian_sigma_for_ksize(ksize: int) -> float:
    # OpenCV's default sigma when sigma=0 (approx):
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    if ksize <= 1:
        return 0.0
    return 0.3 * (((ksize - 1) * 0.5) - 1.0) + 0.8


def _torch_gaussian_kernel1d(ksize: int, sigma: float):
    import torch

    ksize = int(ksize)
    if ksize <= 1:
        return torch.tensor([1.0], dtype=torch.float32)
    if sigma <= 0:
        sigma = _torch_gaussian_sigma_for_ksize(ksize)
    half = (ksize - 1) / 2.0
    x = torch.arange(ksize, dtype=torch.float32) - half
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def _torch_gaussian_blur2d(img_t, ksz: Tuple[int, int]):
    """Gaussian blur for NCHW float32 using separable 1D kernels."""
    import torch.nn.functional as F

    if img_t.ndim != 4:
        raise ValueError("Expected NCHW tensor")

    ky, kx = int(ksz[1]), int(ksz[0])
    if ky <= 1 and kx <= 1:
        return img_t

    # OpenCV uses BORDER_DEFAULT (reflect101-ish). torch's 'reflect' is close enough.
    out = img_t
    if kx > 1:
        k1 = _torch_gaussian_kernel1d(kx, sigma=0.0).to(device=out.device)
        w = k1.view(1, 1, 1, kx)
        pad_x = kx // 2
        out = F.pad(out, (pad_x, pad_x, 0, 0), mode="reflect")
        out = F.conv2d(out, w)
    if ky > 1:
        k1 = _torch_gaussian_kernel1d(ky, sigma=0.0).to(device=out.device)
        w = k1.view(1, 1, ky, 1)
        pad_y = ky // 2
        out = F.pad(out, (0, 0, pad_y, pad_y), mode="reflect")
        out = F.conv2d(out, w)
    return out


class _TorchMatchContext:
    """Keep templates resident on a torch device to avoid per-call upload overhead."""

    _jit_ccorr_topk = None

    def __init__(
        self,
        template_full: np.ndarray,
        template_coarse: Optional[np.ndarray],
        device: Optional[str],
    ) -> None:
        import threading

        import torch

        self.enabled = True
        self.device = device or _default_torch_device()
        self._ones_cache: Dict[Tuple[int, int], "torch.Tensor"] = {}
        self._ones_lock = threading.Lock()

        self._use_jit_topk = False
        try:
            if os.getenv(TORCH_JIT_ENV) not in {"1", "true", "yes", "on"}:
                raise RuntimeError("Torch JIT disabled")
            if _TorchMatchContext._jit_ccorr_topk is None:
                import torch.nn.functional as F

                @torch.jit.script
                def _ccorr_normed_topk(
                    img_t: torch.Tensor,
                    img_sq_t: torch.Tensor,
                    templ_t: torch.Tensor,
                    ones_t: torch.Tensor,
                    k: int,
                    eps: float,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    num = F.conv2d(img_t, templ_t)
                    sum_t2 = torch.sum(templ_t * templ_t)
                    sum_i2 = F.conv2d(img_sq_t, ones_t)
                    den = torch.sqrt(sum_i2 * sum_t2).clamp_min(eps)
                    out = (num / den).to(dtype=torch.float32).squeeze(0).squeeze(0)
                    flat = out.reshape(-1)
                    if flat.numel() == 0:
                        return (
                            torch.empty((0,), dtype=torch.float32, device=flat.device),
                            torch.empty((0,), dtype=torch.int64, device=flat.device),
                        )
                    kk = k
                    if kk < 1:
                        kk = 1
                    if kk > flat.numel():
                        kk = int(flat.numel())
                    vals, idxs = torch.topk(flat, kk)
                    return vals, idxs

                _TorchMatchContext._jit_ccorr_topk = _ccorr_normed_topk

            self._use_jit_topk = True
        except Exception:
            self._use_jit_topk = False

        template_full_f32 = np.ascontiguousarray(
            template_full.astype(np.float32, copy=False)
        )
        self.template_full_t = torch.from_numpy(template_full_f32)[None, None].to(
            self.device
        )
        self.template_full_sq_t = self.template_full_t * self.template_full_t

        self.template_coarse_t = None
        self.template_coarse_sq_t = None
        if template_coarse is not None:
            template_coarse_f32 = np.ascontiguousarray(
                template_coarse.astype(np.float32, copy=False)
            )
            self.template_coarse_t = torch.from_numpy(template_coarse_f32)[
                None, None
            ].to(self.device)
            self.template_coarse_sq_t = self.template_coarse_t * self.template_coarse_t

    @property
    def has_coarse(self) -> bool:
        return (
            self.template_coarse_t is not None and self.template_coarse_sq_t is not None
        )

    def disable(self) -> None:
        self.enabled = False

    def _ones(self, h: int, w: int):
        import torch

        key = (int(h), int(w))
        cached = self._ones_cache.get(key)
        if cached is not None:
            return cached
        with self._ones_lock:
            cached = self._ones_cache.get(key)
            if cached is not None:
                return cached
            ones = torch.ones(
                (1, 1, key[0], key[1]),
                device=self.device,
                dtype=torch.float32,
            )
            self._ones_cache[key] = ones
            return ones

    def _match_ccorr_normed(
        self,
        img_t,
        img_sq_t,
        patch: object,
        eps: float = 1e-8,
    ) -> np.ndarray:
        if not self.enabled:
            raise RuntimeError("Torch matcher disabled")

        import torch
        import torch.nn.functional as F

        def _on_expected_device(t: "torch.Tensor") -> bool:
            dev = str(self.device or "")
            if dev.startswith("cuda"):
                return t.device.type == "cuda"
            return t.device.type == dev

        if isinstance(patch, np.ndarray):
            patch_f32 = np.ascontiguousarray(patch.astype(np.float32, copy=False))
            h, w = patch_f32.shape[:2]
            templ_t = torch.from_numpy(patch_f32)[None, None].to(self.device)
        else:
            templ_t = patch
            if templ_t.ndim == 2:
                templ_t = templ_t[None, None]
            if templ_t.ndim != 4:
                raise ValueError("Torch patch tensor must be 2D or NCHW")
            if not _on_expected_device(templ_t):
                templ_t = templ_t.to(self.device)
            templ_t = templ_t.to(dtype=torch.float32)
            h, w = int(templ_t.shape[-2]), int(templ_t.shape[-1])

        with torch.no_grad():
            num = F.conv2d(img_t, templ_t)
            sum_t2 = torch.sum(templ_t * templ_t)
            sum_i2 = F.conv2d(img_sq_t, self._ones(h, w))
            den = torch.sqrt(sum_i2 * sum_t2).clamp_min(eps)
            out = (num / den).to(dtype=torch.float32)
            return out.squeeze(0).squeeze(0).detach().cpu().numpy()

    def _topk_ccorr_normed(
        self,
        img_t,
        img_sq_t,
        patch: object,
        k: int,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return top-k (values, flat_indices, res_w) without downloading the full score map."""

        if not self.enabled:
            raise RuntimeError("Torch matcher disabled")

        import torch
        import torch.nn.functional as F

        def _on_expected_device(t: "torch.Tensor") -> bool:
            dev = str(self.device or "")
            if dev.startswith("cuda"):
                return t.device.type == "cuda"
            return t.device.type == dev

        if isinstance(patch, np.ndarray):
            patch_f32 = np.ascontiguousarray(patch.astype(np.float32, copy=False))
            h, w = patch_f32.shape[:2]
            if h <= 0 or w <= 0:
                return (
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    0,
                )
            templ_t = torch.from_numpy(patch_f32)[None, None].to(self.device)
        else:
            templ_t = patch
            if templ_t.ndim == 2:
                templ_t = templ_t[None, None]
            if templ_t.ndim != 4:
                raise ValueError("Torch patch tensor must be 2D or NCHW")
            if not _on_expected_device(templ_t):
                templ_t = templ_t.to(self.device)
            templ_t = templ_t.to(dtype=torch.float32)
            h, w = int(templ_t.shape[-2]), int(templ_t.shape[-1])
        if h <= 0 or w <= 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                0,
            )

        with torch.no_grad():
            num = F.conv2d(img_t, templ_t)
            sum_t2 = torch.sum(templ_t * templ_t)
            sum_i2 = F.conv2d(img_sq_t, self._ones(h, w))
            den = torch.sqrt(sum_i2 * sum_t2).clamp_min(eps)

            res_w = int(img_t.shape[-1]) - int(w) + 1

            k = int(k)
            if k <= 0:
                return (
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    res_w,
                )

            if self._use_jit_topk and _TorchMatchContext._jit_ccorr_topk is not None:
                try:
                    vals_t, idxs_t = _TorchMatchContext._jit_ccorr_topk(
                        img_t,
                        img_sq_t,
                        templ_t,
                        self._ones(h, w),
                        k,
                        float(eps),
                    )
                    return (
                        vals_t.detach().cpu().numpy().astype(np.float32, copy=False),
                        idxs_t.detach().cpu().numpy().astype(np.int64, copy=False),
                        res_w,
                    )
                except Exception:
                    pass

            out = (num / den).to(dtype=torch.float32).squeeze(0).squeeze(0)
            flat = out.reshape(-1)
            if flat.numel() == 0:
                return (
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    res_w,
                )
            k = min(k, int(flat.numel()))
            vals, idxs = torch.topk(flat, k)
            return (
                vals.detach().cpu().numpy().astype(np.float32, copy=False),
                idxs.detach().cpu().numpy().astype(np.int64, copy=False),
                res_w,
            )

    def _topk_ccorr_normed_batch(
        self,
        img_t,
        img_sq_t,
        patches_t,
        k: int,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Batched top-k for many templates of the same shape.

        patches_t: NCHW tensor where N is the number of patches.
        Returns (vals, idxs, res_w) where vals/idxs are (N, k) numpy arrays.
        """

        if not self.enabled:
            raise RuntimeError("Torch matcher disabled")

        import torch
        import torch.nn.functional as F

        if patches_t.ndim != 4:
            raise ValueError("Torch batch patches must be NCHW")
        if patches_t.shape[1] != 1:
            raise ValueError("Torch batch patches must have C=1")

        patches_t = patches_t.to(device=self.device, dtype=torch.float32)
        n = int(patches_t.shape[0])
        h = int(patches_t.shape[-2])
        w = int(patches_t.shape[-1])
        res_w = int(img_t.shape[-1]) - int(w) + 1

        if n <= 0 or h <= 0 or w <= 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.int64),
                res_w,
            )

        k = int(k)
        if k <= 0:
            return (
                np.empty((n, 0), dtype=np.float32),
                np.empty((n, 0), dtype=np.int64),
                res_w,
            )

        with torch.no_grad():
            # Treat each patch as an output channel.
            num = F.conv2d(img_t, patches_t)
            # sum_t2 per patch (channel)
            sum_t2 = torch.sum(patches_t * patches_t, dim=(1, 2, 3))  # (N,)
            sum_i2 = F.conv2d(img_sq_t, self._ones(h, w))  # (1,1,Ho,Wo)
            den = torch.sqrt(sum_i2 * sum_t2.view(1, n, 1, 1)).clamp_min(eps)
            out = (num / den).to(dtype=torch.float32).squeeze(0)  # (N,Ho,Wo)
            flat = out.reshape(n, -1)
            if flat.numel() == 0:
                return (
                    np.empty((n, 0), dtype=np.float32),
                    np.empty((n, 0), dtype=np.int64),
                    res_w,
                )
            kk = min(k, int(flat.shape[1]))
            vals, idxs = torch.topk(flat, kk, dim=1)
            return (
                vals.detach().cpu().numpy().astype(np.float32, copy=False),
                idxs.detach().cpu().numpy().astype(np.int64, copy=False),
                res_w,
            )

    def match_full(self, patch: np.ndarray) -> np.ndarray:
        return self._match_ccorr_normed(
            self.template_full_t,
            self.template_full_sq_t,
            patch,
        )

    def topk_full(
        self, patch: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        return self._topk_ccorr_normed(
            self.template_full_t,
            self.template_full_sq_t,
            patch,
            k,
        )

    def topk_full_batch(self, patches_t, k: int) -> Tuple[np.ndarray, np.ndarray, int]:
        return self._topk_ccorr_normed_batch(
            self.template_full_t,
            self.template_full_sq_t,
            patches_t,
            k,
        )

    def match_coarse(self, patch: np.ndarray) -> np.ndarray:
        if not self.has_coarse:
            raise RuntimeError("Torch coarse matcher not initialized")
        return self._match_ccorr_normed(
            self.template_coarse_t,
            self.template_coarse_sq_t,
            patch,
        )

    def topk_coarse(
        self, patch: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if not self.has_coarse:
            raise RuntimeError("Torch coarse matcher not initialized")
        return self._topk_ccorr_normed(
            self.template_coarse_t,
            self.template_coarse_sq_t,
            patch,
            k,
        )

    def topk_coarse_batch(
        self, patches_t, k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if not self.has_coarse:
            raise RuntimeError("Torch coarse matcher not initialized")
        return self._topk_ccorr_normed_batch(
            self.template_coarse_t,
            self.template_coarse_sq_t,
            patches_t,
            k,
        )


def _match_template(
    template_img: np.ndarray,
    patch: np.ndarray,
    corr_method: int,
    config: MatcherConfig,
    torch_ctx: Optional[_TorchMatchContext],
    template_kind: str,
) -> np.ndarray:
    """Match template with optional Torch acceleration and graceful fallback."""

    if (
        config.use_torch
        and template_kind in ("full", "coarse")
        and corr_method == cv2.TM_CCORR_NORMED
        and _torch_available()
    ):
        torch_device = (config.torch_device or _default_torch_device()).strip().lower()
        if torch_device != "cpu":
            try:
                if torch_ctx is not None and torch_ctx.enabled:
                    if template_kind == "full":
                        return torch_ctx.match_full(patch)
                    if template_kind == "coarse" and torch_ctx.has_coarse:
                        return torch_ctx.match_coarse(patch)

                return _match_template_torch_ccorr_normed(
                    template_img,
                    patch,
                    device=torch_device,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Torch match failed (%s): %s", template_kind, exc)

    return cv2.matchTemplate(template_img, patch, corr_method)


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


def _fill_mask_holes(mask01: np.ndarray) -> np.ndarray:
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(mask255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood = padded.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, flood_inv)
    filled = filled[1:-1, 1:-1]
    return (filled > 0).astype(np.uint8)


def _smooth_mask_edges(mask01: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(3, min(5, int(kernel_size)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    mask255 = cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask255 = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, kernel, iterations=1)
    return (mask255 > 0).astype(np.uint8)


def _background_distance_from_border(
    img_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    h, w = img_bgr.shape[:2]
    border_px = int(max(6, min(24, round(min(h, w) * 0.04))))
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border_px, :] = 1
    border_mask[-border_px:, :] = 1
    border_mask[:, :border_px] = 1
    border_mask[:, -border_px:] = 1
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_pixels = lab[border_mask == 1]
    if border_pixels.size == 0:
        return None, None
    bg_color = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(lab - bg_color, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_u8 = cv2.GaussianBlur(dist_u8, (3, 3), 0)
    otsu, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dist_u8, int(otsu)


def _recover_piece_edges(
    piece_bgr: np.ndarray, mask01: np.ndarray, kernel_size: int
) -> np.ndarray:
    dist_u8, otsu = _background_distance_from_border(piece_bgr)
    if dist_u8 is None or otsu is None:
        return mask01
    k = max(3, min(7, int(kernel_size)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.morphologyEx(
        mask01.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=1
    )
    eroded = cv2.morphologyEx(
        mask01.astype(np.uint8), cv2.MORPH_ERODE, kernel, iterations=1
    )
    edge_band = (dilated > 0) & (eroded == 0)
    add = edge_band & (dist_u8 >= max(0, otsu - 8))
    out = (mask01 > 0) | add
    return out.astype(np.uint8)


def _smooth_piece_contour(mask01: np.ndarray) -> np.ndarray:
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask01
    cnt = max(contours, key=cv2.contourArea)
    arc = cv2.arcLength(cnt, True)
    epsilon = max(2.0, 0.01 * arc)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    smoothed = np.zeros_like(mask255)
    cv2.drawContours(smoothed, [approx], -1, 255, -1)
    merged = cv2.bitwise_or(mask255, smoothed)
    return (merged > 0).astype(np.uint8)


def _template_color_clusters(
    template_bgr: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    k: int = 4,
    percentile: float = 98.0,
    scale: float = 1.15,
) -> Tuple[np.ndarray, np.ndarray]:
    template_lab = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if template_mask is None or template_mask.sum() == 0:
        fg = template_lab.reshape(-1, 3)
    else:
        fg = template_lab[template_mask > 0]
    fg_ab = fg[:, 1:3]
    if fg_ab.shape[0] > 200_000:
        idx = np.linspace(0, fg_ab.shape[0] - 1, num=200_000).astype(int)
        fg_ab = fg_ab[idx]
    if fg_ab.shape[0] < k:
        mean_ab = np.median(fg_ab, axis=0, keepdims=True)
        dist_fg = np.linalg.norm(fg_ab - mean_ab[0], axis=1)
        thresh = float(np.percentile(dist_fg, percentile)) * scale
        return mean_ab, np.array([thresh], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(
        fg_ab.astype(np.float32),
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.float32)
    thresholds = []
    for idx in range(k):
        cluster = fg_ab[labels.ravel() == idx]
        if cluster.size == 0:
            thresholds.append(0.0)
            continue
        dist = np.linalg.norm(cluster - centers[idx], axis=1)
        thresholds.append(float(np.percentile(dist, percentile)) * scale)
    return centers, np.array(thresholds, dtype=np.float32)


def _apply_template_cluster_mask(
    piece_bgr: np.ndarray, centers: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    piece_lab = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    piece_ab = piece_lab[:, :, 1:3]
    mask = np.zeros(piece_ab.shape[:2], dtype=np.uint8)
    for center, thresh in zip(centers, thresholds):
        if thresh <= 0:
            continue
        dist_piece = np.linalg.norm(piece_ab - center, axis=2)
        mask |= (dist_piece <= thresh).astype(np.uint8)
    return mask


def _background_color_clusters(
    piece_bgr: np.ndarray,
    k: int = 3,
    percentile: float = 98.0,
    scale: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = piece_bgr.shape[:2]
    border_px = int(max(8, min(40, round(min(h, w) * 0.06))))
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border_px, :] = 1
    border_mask[-border_px:, :] = 1
    border_mask[:, :border_px] = 1
    border_mask[:, -border_px:] = 1
    lab = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = lab[border_mask == 1][:, 1:3]
    if bg.shape[0] > 200_000:
        idx = np.linspace(0, bg.shape[0] - 1, num=200_000).astype(int)
        bg = bg[idx]
    if bg.shape[0] < k:
        mean_ab = np.median(bg, axis=0, keepdims=True)
        dist_bg = np.linalg.norm(bg - mean_ab[0], axis=1)
        thresh = float(np.percentile(dist_bg, percentile)) * scale
        return mean_ab, np.array([thresh], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(
        bg.astype(np.float32),
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.float32)
    thresholds = []
    for idx in range(k):
        cluster = bg[labels.ravel() == idx]
        if cluster.size == 0:
            thresholds.append(0.0)
            continue
        dist = np.linalg.norm(cluster - centers[idx], axis=1)
        thresholds.append(float(np.percentile(dist, percentile)) * scale)
    return centers, np.array(thresholds, dtype=np.float32)


def _apply_background_cluster_mask(
    piece_bgr: np.ndarray, centers: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    piece_lab = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    piece_ab = piece_lab[:, :, 1:3]
    mask = np.zeros(piece_ab.shape[:2], dtype=np.uint8)
    for center, thresh in zip(centers, thresholds):
        if thresh <= 0:
            continue
        dist_piece = np.linalg.norm(piece_ab - center, axis=2)
        mask |= (dist_piece <= thresh).astype(np.uint8)
    return mask


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


def _mask_by_gradient(
    piece_bgr: np.ndarray,
    kernel_size: int,
    open_iters: int,
    close_iters: int,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Segment piece using gradient-based edge detection.

    This approach works by:
    1. Converting to grayscale and applying heavy blur to smooth internal texture
    2. Computing morphological gradient to detect edges
    3. Thresholding and closing gaps in the edge contour
    4. Finding the largest contour and filling it to create the mask

    This works best when there is clear contrast between the piece edge and
    background. For images with low edge contrast, consider using mask_mode="ai"
    instead.

    Args:
        piece_bgr: BGR image of the piece.
        kernel_size: Morphological kernel size for cleanup.
        open_iters: Opening iterations for cleanup.
        close_iters: Closing iterations for cleanup.
        keep_largest_component: If True, keep only the largest connected component.

    Returns:
        Binary mask (0 or 1) with the piece foreground.
    """
    gray = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)

    # Heavy blur to smooth internal texture while preserving piece edges
    blur_size = max(11, kernel_size * 2 - 1)
    if blur_size % 2 == 0:
        blur_size += 1
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Morphological gradient detects edges
    grad_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, grad_kernel)

    # Threshold to find strong edges using Otsu's method
    _, edge_thresh = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Close gaps in edge contour
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    edges_closed = cv2.morphologyEx(
        edge_thresh, cv2.MORPH_CLOSE, close_kernel, iterations=close_iters
    )

    # Find largest contour (should be the piece boundary)
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "Gradient segmentation produced no contours - check image contrast"
        )

    largest = max(contours, key=cv2.contourArea)

    # Create mask by filling the contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    # Apply standard cleanup
    mask = _cleanup_mask(mask, kernel_size, open_iters, close_iters)
    mask01 = (mask > 0).astype(np.uint8)

    if keep_largest_component:
        mask01 = _keep_largest_component(mask01)

    if mask01.sum() == 0:
        raise RuntimeError(
            "Gradient segmentation produced empty mask - check image contrast"
        )

    return mask01


# Global rembg session (lazy initialized)
_REMBG_SESSION = None


def _get_rembg_session():
    """Get or create the rembg session (lazy initialization)."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session

            _REMBG_SESSION = new_session("isnet-general-use")
        except ImportError as e:
            raise RuntimeError(
                "rembg package not installed. Install with: pip install rembg"
            ) from e
    return _REMBG_SESSION


def _mask_by_ai(
    piece_bgr: np.ndarray,
    kernel_size: int,
    open_iters: int,
    close_iters: int,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Segment piece using AI-based background removal (rembg).

    Uses the ISNet-general-use model via rembg for high-quality background
    removal. This produces excellent masks but is slower (~1s per piece) than
    color-based methods.

    Args:
        piece_bgr: BGR image of the piece.
        kernel_size: Morphological kernel size for cleanup.
        open_iters: Opening iterations for cleanup.
        close_iters: Closing iterations for cleanup.
        keep_largest_component: If True, keep only the largest connected component.

    Returns:
        Binary mask (0 or 1) with the piece foreground.
    """
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError(
            "rembg package not installed. Install with: pip install rembg"
        ) from e

    # Convert BGR to RGB for rembg
    piece_rgb = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2RGB)

    # Get rembg session
    session = _get_rembg_session()

    # Remove background - returns RGBA with alpha channel as mask
    result_rgba = remove(piece_rgb, session=session)

    # Extract alpha channel as mask
    if result_rgba.shape[2] == 4:
        alpha = result_rgba[:, :, 3]
    else:
        # Fallback: compare to original to find changed pixels
        alpha = np.any(result_rgba != piece_rgb, axis=2).astype(np.uint8) * 255

    # Threshold to binary
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # Apply standard cleanup
    mask = _cleanup_mask(mask, kernel_size, open_iters, close_iters)
    mask01 = (mask > 0).astype(np.uint8)

    if keep_largest_component:
        mask01 = _keep_largest_component(mask01)

    if mask01.sum() == 0:
        raise RuntimeError("AI segmentation produced empty mask")

    return mask01


def remove_background_ai(image_bgr: np.ndarray) -> np.ndarray:
    """Remove background from image using AI (rembg).

    Returns the image with transparent background (BGRA format).
    This is useful for preprocessing multipiece images once before
    splitting into individual pieces.

    Args:
        image_bgr: BGR image.

    Returns:
        BGRA image with transparent background.
    """
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError(
            "rembg package not installed. Install with: pip install rembg"
        ) from e

    # Convert BGR to RGB for rembg
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Get rembg session
    session = _get_rembg_session()

    # Remove background - returns RGBA
    result_rgba = remove(image_rgb, session=session)

    # Convert RGB channels back to BGR, keep alpha
    result_bgr = cv2.cvtColor(result_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    if result_rgba.shape[2] == 4:
        alpha = result_rgba[:, :, 3]
    else:
        # rembg should always return RGBA, but fallback to full opacity if not
        alpha = np.full(result_bgr.shape[:2], 255, dtype=np.uint8)
    result_bgra = np.dstack([result_bgr, alpha])

    return result_bgra


def compute_piece_mask(
    piece_bgr: np.ndarray,
    config: MatcherConfig,
    keep_largest_component: bool = True,
    template_bgr: Optional[np.ndarray] = None,
    template_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a binary mask for a puzzle piece based on color mode.

    Supports "blue", "green", "hsv"/"hsv_ranges", "gradient", or "ai" modes.
    Returns a binary mask (0 or 1) with the piece foreground isolated at the
    input image size.

    If config.mask_skip is True, returns an all-foreground mask (useful when
    background has already been removed, e.g., by AI preprocessing).

    If the input image has 4 channels (BGRA), the alpha channel is used directly
    as the mask, skipping color-based segmentation. This enables efficient
    multipiece matching when background has been pre-removed.

    Args:
        piece_bgr: BGR or BGRA image of the piece. If BGRA, alpha is used as mask.
        config: MatcherConfig with mask settings.
        keep_largest_component: If True, keep only the largest connected component.
                               If False, keep all detected foreground. Defaults to True.

    Returns:
        np.ndarray: Binary mask of the same height and width as ``piece_bgr``,
            with ``dtype`` ``uint8`` and values 0 or 1, where 1 indicates the
            piece foreground and 0 indicates background.
    """
    # If mask_skip is set, return all-foreground mask
    if config.mask_skip:
        return np.ones(piece_bgr.shape[:2], dtype=np.uint8)

    # If input is BGRA (4 channels), use alpha channel directly as mask
    if piece_bgr.ndim == 3 and piece_bgr.shape[2] == 4:
        alpha = piece_bgr[:, :, 3]
        # Threshold alpha to binary (alpha > 127 means foreground)
        mask01 = (alpha > 127).astype(np.uint8)
        if keep_largest_component and mask01.sum() > 0:
            mask01 = _keep_largest_component(mask01)
        return mask01

    mask_mode = (config.mask_mode or "blue").lower()
    kernel_size = int(config.mask_kernel_size)
    open_iters = int(config.mask_open_iters)
    close_iters = int(config.mask_close_iters)
    if mask_mode == "blue":
        mask01 = _mask_by_blue(
            piece_bgr, kernel_size, open_iters, close_iters, keep_largest_component
        )
    elif mask_mode == "green":
        mask01 = _mask_by_green(
            piece_bgr, kernel_size, open_iters, close_iters, keep_largest_component
        )
    elif mask_mode in ("hsv", "hsv_ranges"):
        if not config.mask_hsv_ranges:
            raise RuntimeError("mask_hsv_ranges must be set for hsv mask mode.")
        mask01 = _mask_by_hsv_ranges(
            piece_bgr,
            config.mask_hsv_ranges,
            kernel_size,
            open_iters,
            close_iters,
            keep_largest_component,
        )
    elif mask_mode == "gradient":
        mask01 = _mask_by_gradient(
            piece_bgr, kernel_size, open_iters, close_iters, keep_largest_component
        )
    elif mask_mode == "ai":
        mask01 = _mask_by_ai(
            piece_bgr, kernel_size, open_iters, close_iters, keep_largest_component
        )
    else:
        raise RuntimeError(f"Unknown mask_mode: {config.mask_mode}")

    template_clustering_applied = False
    if (
        config.template_clustering
        and template_bgr is not None
        and mask_mode in ("hsv", "hsv_ranges")
    ):
        try:
            y0, y1, x0, x1 = _mask_bbox(mask01)
            bbox_area = max(1, (y1 - y0) * (x1 - x0))
            fill_ratio = float(mask01.sum()) / float(bbox_area)
        except RuntimeError:
            fill_ratio = 0.0
        if fill_ratio < MASK_FILL_RATIO_THRESHOLD:
            cluster_scale = float(config.template_cluster_scale)
            cluster_scale *= 1.0 + min(
                MASK_FILL_RATIO_MULTIPLIER,
                (MASK_FILL_RATIO_THRESHOLD - fill_ratio) * MASK_FILL_SCALE_FACTOR,
            )
            centers, thresholds = _template_color_clusters(
                template_bgr,
                template_mask=template_mask,
                k=int(config.template_cluster_k),
                percentile=float(config.template_cluster_percentile),
                scale=cluster_scale,
            )
            template_mask01 = _apply_template_cluster_mask(
                piece_bgr, centers, thresholds
            )
            bg_centers, bg_thresholds = _background_color_clusters(piece_bgr)
            bg_mask01 = _apply_background_cluster_mask(
                piece_bgr, bg_centers, bg_thresholds
            )
            seed_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, kernel_size), max(3, kernel_size))
            )
            seed = cv2.morphologyEx(
                (mask01 > 0).astype(np.uint8) * 255,
                cv2.MORPH_DILATE,
                seed_kernel,
                iterations=2,
            )
            template_mask01 = (template_mask01 > 0).astype(np.uint8)
            template_mask01 = np.where(bg_mask01 > 0, 0, template_mask01).astype(
                np.uint8
            )
            mask01 = np.where(seed > 0, (mask01 | template_mask01), mask01).astype(
                np.uint8
            )
            allow_growth = fill_ratio < MASK_ALLOW_GROWTH_THRESHOLD
            if allow_growth:
                dist_u8, otsu = _background_distance_from_border(piece_bgr)
                if dist_u8 is None or otsu is None:
                    bg_dist_mask = np.zeros_like(mask01)
                else:
                    bg_dist_mask = (
                        dist_u8 >= max(0, otsu + BG_DISTANCE_THRESHOLD)
                    ).astype(np.uint8)
                candidate = (template_mask01 | bg_dist_mask).astype(np.uint8)
                candidate = np.where(bg_mask01 > 0, 0, candidate).astype(np.uint8)
                mask01 = np.where(seed > 0, (mask01 | candidate), mask01).astype(
                    np.uint8
                )
                if fill_ratio < MASK_AGGRESSIVE_GROWTH_THRESHOLD:
                    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    dilated = cv2.morphologyEx(
                        (mask01 > 0).astype(np.uint8) * 255,
                        cv2.MORPH_DILATE,
                        dil_kernel,
                        iterations=1,
                    )
                    dilated01 = (dilated > 0).astype(np.uint8)
                    constrained = (dilated01 > 0) & (candidate > 0)
                    mask01 = np.where(constrained, 1, mask01).astype(np.uint8)
            # Apply shape refinement as part of template clustering pipeline
            if config.mask_shape_refine:
                mask01 = _recover_piece_edges(piece_bgr, mask01, kernel_size)
                mask01 = _smooth_piece_contour(mask01)
                mask01 = _fill_mask_holes(mask01)
                mask01 = _smooth_mask_edges(mask01, kernel_size)
            else:
                if fill_ratio < MASK_ALLOW_GROWTH_THRESHOLD:
                    mask01 = _recover_piece_edges(piece_bgr, mask01, kernel_size)
                mask01 = _fill_mask_holes(mask01)
                mask01 = _smooth_mask_edges(mask01, kernel_size)
            mask01 = _cleanup_mask(mask01 * 255, kernel_size, open_iters, close_iters)
            mask01 = (mask01 > 0).astype(np.uint8)
            if keep_largest_component:
                mask01 = _keep_largest_component(mask01)
            template_clustering_applied = True

    # Apply shape refinement for non-template clustering cases or when explicitly enabled
    if not template_clustering_applied and config.mask_shape_refine:
        mask01 = _recover_piece_edges(piece_bgr, mask01, kernel_size)
        mask01 = _smooth_piece_contour(mask01)
        mask01 = _smooth_mask_edges(mask01, kernel_size)
        mask01 = _fill_mask_holes(mask01)

    return mask01


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
    mask01 = (piece_mask > 0).astype(np.uint8)
    mh, mw = mask01.shape
    if mw == 0 or mh == 0:
        return 0, 0

    piece_area_px = float(mask01.sum())
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


def _candidate_order(
    flat: np.ndarray,
    max_len: int,
    scan_multiplier: int,
) -> np.ndarray:
    if flat.size <= max_len:
        return np.argsort(flat)[::-1]
    scan_count = min(flat.size, max(max_len * scan_multiplier, max_len))
    order = np.argpartition(flat, -scan_count)[-scan_count:]
    return order[np.argsort(flat[order])[::-1]]


def _scan_candidates_topk(
    values: np.ndarray,
    order: np.ndarray,
    res_w: int,
    ws: int,
    hs: int,
    rot_value: int,
    scale_value: float,
    cell_w: float,
    cell_h: float,
    cols: int,
    rows: int,
    grid_center_weight: float,
    top_match_count: int,
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

    for i, idx in enumerate(order):
        if len(combo_best_local) >= top_match_count:
            break
        y, x = divmod(int(idx), res_w)
        x0 = x + offset_x
        y0 = y + offset_y
        cx = x0 + core_offset_x
        cy = y0 + core_offset_y
        base_score = float(values[i])
        proximity = _grid_center_proximity(cx, cy, cell_w, cell_h, cols, rows)
        score = base_score + (grid_center_weight * proximity)
        tl = (int(x0), int(y0))
        br = (int(x0 + ws), int(y0 + hs))
        candidate = {
            "score": score,
            "score_raw": base_score,
            "grid_score": proximity,
            "rot": int(rot_value),
            "scale": float(scale_value),
            "col": int(cx / cell_w) + 1,
            "row": int(cy / cell_h) + 1,
            "tl": tl,
            "br": br,
            "center": (float(cx), float(cy)),
        }
        if any(
            _candidate_is_close(candidate, existing) for existing in combo_best_local
        ):
            continue
        combo_best_local.append(candidate)
    return combo_best_local


def _scan_candidates(
    res: np.ndarray,
    order: np.ndarray,
    res_w: int,
    ws: int,
    hs: int,
    rot_value: int,
    scale_value: float,
    cell_w: float,
    cell_h: float,
    cols: int,
    rows: int,
    grid_center_weight: float,
    top_match_count: int,
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
            "rot": int(rot_value),
            "scale": float(scale_value),
            "col": int(cx / cell_w) + 1,
            "row": int(cy / cell_h) + 1,
            "tl": tl,
            "br": br,
            "center": (float(cx), float(cy)),
        }
        if any(
            _candidate_is_close(candidate, existing) for existing in combo_best_local
        ):
            continue
        combo_best_local.append(candidate)
    return combo_best_local


def _collect_matches(
    res: np.ndarray,
    ws: int,
    hs: int,
    rot_value: int,
    scale_value: float,
    cell_w: float,
    cell_h: float,
    cols: int,
    rows: int,
    grid_center_weight: float,
    top_match_count: int,
    top_match_scan_multiplier: int,
    offset_x: int = 0,
    offset_y: int = 0,
    core_offset_x: Optional[float] = None,
    core_offset_y: Optional[float] = None,
) -> List[Dict]:
    flat = res.ravel()
    order = _candidate_order(flat, top_match_count, top_match_scan_multiplier)
    res_w = res.shape[1]
    combo_best = _scan_candidates(
        res,
        order,
        res_w,
        ws,
        hs,
        rot_value,
        scale_value,
        cell_w,
        cell_h,
        cols,
        rows,
        grid_center_weight,
        top_match_count,
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
            rot_value,
            scale_value,
            cell_w,
            cell_h,
            cols,
            rows,
            grid_center_weight,
            top_match_count,
            offset_x=offset_x,
            offset_y=offset_y,
            core_offset_x=core_offset_x,
            core_offset_y=core_offset_y,
        )
    return combo_best


def _collect_coarse_positions_topk(
    values: np.ndarray,
    order: np.ndarray,
    res_w: int,
    ws: int,
    hs: int,
    top_k: int,
) -> List[Dict]:
    positions: List[Dict] = []
    for i, idx in enumerate(order):
        if len(positions) >= top_k:
            break
        y, x = divmod(int(idx), res_w)
        candidate = {
            "score": float(values[i]),
            "tl": (int(x), int(y)),
            "br": (int(x + ws), int(y + hs)),
            "center": (float(x + ws / 2), float(y + hs / 2)),
        }
        if any(_candidate_is_close(candidate, existing) for existing in positions):
            continue
        positions.append(candidate)
    return positions


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
    torch_ctx: Optional[_TorchMatchContext],
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
            res_c = _match_template(
                template_coarse_f32,
                patt_masked_c,
                corr_method,
                config,
                torch_ctx,
                template_kind="coarse",
            )

            if res_c.size > 0:
                # Find top-K coarse candidates
                flat_c = res_c.ravel()
                order_c = _candidate_order(
                    flat_c, coarse_top_k, top_match_scan_multiplier
                )
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
                    res_fine = _match_template(
                        roi,
                        patt_masked,
                        corr_method,
                        config,
                        torch_ctx,
                        template_kind="roi",
                    )
                    if res_fine.size == 0:
                        continue

                    flat_fine = res_fine.ravel()
                    order_fine = _candidate_order(
                        flat_fine, top_match_count, top_match_scan_multiplier
                    )
                    res_w_fine = res_fine.shape[1]

                    fine_candidates = _scan_candidates(
                        res_fine,
                        order_fine,
                        res_w_fine,
                        ws,
                        hs,
                        rot,
                        scale,
                        cell_w,
                        cell_h,
                        cols,
                        rows,
                        grid_center_weight,
                        top_match_count,
                        offset_x=x0,
                        offset_y=y0,
                        core_offset_x=core_offset_x,
                        core_offset_y=core_offset_y,
                    )
                    all_fine_candidates.extend(fine_candidates)

                # Only short-circuit if the coarse refinement produced candidates.
                # If none were found, fall back to full-resolution matching below.
                if all_fine_candidates:
                    return all_fine_candidates

    # Fallback: full-resolution matching without coarse pass
    res = _match_template(
        template_blur_f32,
        patt_masked,
        corr_method,
        config,
        torch_ctx,
        template_kind="full",
    )

    if res.size == 0:
        return []

    flat = res.ravel()
    order = _candidate_order(flat, top_match_count, top_match_scan_multiplier)
    res_w = res.shape[1]
    return _scan_candidates(
        res,
        order,
        res_w,
        ws,
        hs,
        rot,
        scale,
        cell_w,
        cell_h,
        cols,
        rows,
        grid_center_weight,
        top_match_count,
        offset_x=0,
        offset_y=0,
        core_offset_x=core_offset_x,
        core_offset_y=core_offset_y,
    )


def _match_rotation_scale_sweep(
    rot: int,
    scales: List[float],
    piece_bin_pattern: np.ndarray,
    piece_mask: np.ndarray,
    template_blur_f32: np.ndarray,
    template_coarse_f32: Optional[np.ndarray],
    torch_ctx: Optional[_TorchMatchContext],
    config: MatcherConfig,
    knobs_x: Optional[int],
    knobs_y: Optional[int],
    blur_ksz: Optional[Tuple[int, int]],
    corr_method: int,
    cols: int,
    rows: int,
) -> List[Dict]:
    """Match all scales for a single rotation.

    This is used by the CPU-parallel path to avoid re-rotating the piece/mask
    for every (scale, rotation) combination.
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

    use_coarse = template_coarse_f32 is not None and 0.0 < coarse_factor < 1.0

    used_torch = (
        config.use_torch
        and corr_method == cv2.TM_CCORR_NORMED
        and torch_ctx is not None
        and torch_ctx.enabled
    )

    # Build per-scale payloads once, then run batched torch top-k by patch size.
    scale_jobs: List[Dict] = []
    for scale in scales:
        ws = int(round(P_r.shape[1] * scale))
        hs = int(round(P_r.shape[0] * scale))
        if ws <= 0 or hs <= 0 or ws >= tw or hs >= th:
            continue

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
        job = {
            "scale": float(scale),
            "ws": int(ws),
            "hs": int(hs),
            "core_offset_x": float(core_offset_x),
            "core_offset_y": float(core_offset_y),
            "patt_masked": patt_masked,
            "mask_s": mask_s,
            "patt_s_blur": patt_s_blur,
        }

        if use_coarse:
            th_c, tw_c = template_coarse_f32.shape[:2]
            ws_c = max(1, int(round(ws * coarse_factor)))
            hs_c = max(1, int(round(hs * coarse_factor)))
            if 1 < ws_c < tw_c and 1 < hs_c < th_c:
                patt_c = _adaptive_coarse_resize(
                    patt_s_blur, coarse_factor, config.preserve_edges_coarse
                )
                if patt_c.shape[:2] != (hs_c, ws_c):
                    patt_c = cv2.resize(
                        patt_c, (ws_c, hs_c), interpolation=cv2.INTER_AREA
                    )
                mask_c = cv2.resize(
                    mask_s, (ws_c, hs_c), interpolation=cv2.INTER_NEAREST
                )
                patt_masked_c = patt_c * mask_c
                job.update(
                    {
                        "ws_c": int(ws_c),
                        "hs_c": int(hs_c),
                        "patt_masked_c": patt_masked_c,
                    }
                )

        scale_jobs.append(job)

    all_candidates: List[Dict] = []
    if not scale_jobs:
        return all_candidates

    # --- Coarse pass (batched on torch when enabled) ---
    coarse_succeeded: Dict[float, bool] = {}
    if use_coarse and template_coarse_f32 is not None:
        # Group by coarse patch shape.
        coarse_groups: Dict[Tuple[int, int], List[Dict]] = {}
        for job in scale_jobs:
            if "patt_masked_c" not in job:
                continue
            key = (int(job["hs_c"]), int(job["ws_c"]))
            coarse_groups.setdefault(key, []).append(job)

        for (hs_c, ws_c), jobs in coarse_groups.items():
            coarse_positions_list: List[List[Dict]] = []
            res_w_c = 0
            if used_torch and torch_ctx.has_coarse:
                try:
                    import torch

                    patches_np = [
                        np.ascontiguousarray(
                            j["patt_masked_c"].astype(np.float32, copy=False)
                        )
                        for j in jobs
                    ]
                    patches_t = torch.from_numpy(np.stack(patches_np, axis=0))[
                        :, None
                    ].to(torch_ctx.device)
                    flat_size = (template_coarse_f32.shape[1] - ws_c + 1) * (
                        template_coarse_f32.shape[0] - hs_c + 1
                    )
                    if flat_size > 0:
                        k = min(
                            flat_size,
                            max(coarse_top_k * top_match_scan_multiplier, coarse_top_k),
                        )
                        vals_b, idxs_b, res_w_c = torch_ctx.topk_coarse_batch(
                            patches_t, k
                        )
                        for i in range(vals_b.shape[0]):
                            coarse_positions_list.append(
                                _collect_coarse_positions_topk(
                                    vals_b[i],
                                    idxs_b[i],
                                    res_w_c,
                                    ws_c,
                                    hs_c,
                                    coarse_top_k,
                                )
                            )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Torch coarse batch topk failed: %s", exc)
                    coarse_positions_list = []

            if not coarse_positions_list:
                # Fallback: OpenCV match per scale in this group.
                for j in jobs:
                    res_c = _match_template(
                        template_coarse_f32,
                        j["patt_masked_c"],
                        corr_method,
                        config,
                        torch_ctx,
                        template_kind="coarse",
                    )
                    if res_c.size:
                        flat_c = res_c.ravel()
                        order_c = _candidate_order(
                            flat_c, coarse_top_k, top_match_scan_multiplier
                        )
                        coarse_positions_list.append(
                            [
                                {
                                    "score": float(res_c.ravel()[int(idx)]),
                                    "tl": (
                                        int(divmod(int(idx), res_c.shape[1])[1]),
                                        int(divmod(int(idx), res_c.shape[1])[0]),
                                    ),
                                }
                                for idx in order_c[:coarse_top_k]
                            ]
                        )
                    else:
                        coarse_positions_list.append([])

            # ROI refinement per scale job.
            for j, coarse_positions in zip(jobs, coarse_positions_list):
                ws = int(j["ws"])
                hs = int(j["hs"])
                scale = float(j["scale"])
                patt_masked = j["patt_masked"]
                core_offset_x = float(j["core_offset_x"])
                core_offset_y = float(j["core_offset_y"])

                candidates_for_scale: List[Dict] = []
                for coarse in coarse_positions[:coarse_top_k]:
                    x_c, y_c = coarse.get("tl", (0, 0))
                    x_full = int(round(x_c / coarse_factor))
                    y_full = int(round(y_c / coarse_factor))
                    x_full = max(0, min(x_full, tw - ws))
                    y_full = max(0, min(y_full, th - hs))

                    x0 = max(0, x_full - coarse_padding_pixels)
                    y0 = max(0, y_full - coarse_padding_pixels)
                    x1 = min(tw, x_full + ws + coarse_padding_pixels)
                    y1 = min(th, y_full + hs + coarse_padding_pixels)

                    roi = template_blur_f32[y0:y1, x0:x1]
                    if roi.shape[0] < hs or roi.shape[1] < ws:
                        continue
                    res_fine = _match_template(
                        roi,
                        patt_masked,
                        corr_method,
                        config,
                        torch_ctx,
                        template_kind="roi",
                    )
                    if res_fine.size == 0:
                        continue
                    flat_fine = res_fine.ravel()
                    order_fine = _candidate_order(
                        flat_fine, top_match_count, top_match_scan_multiplier
                    )
                    res_w_fine = res_fine.shape[1]
                    candidates_for_scale.extend(
                        _scan_candidates(
                            res_fine,
                            order_fine,
                            res_w_fine,
                            ws,
                            hs,
                            rot,
                            scale,
                            cell_w,
                            cell_h,
                            cols,
                            rows,
                            grid_center_weight,
                            top_match_count,
                            offset_x=x0,
                            offset_y=y0,
                            core_offset_x=core_offset_x,
                            core_offset_y=core_offset_y,
                        )
                    )

                if candidates_for_scale:
                    coarse_succeeded[scale] = True
                    all_candidates.extend(candidates_for_scale)
                    # Mirror the outer parallel early-exit threshold.
                    # Returning early here makes the ThreadPool cancellation
                    # far more effective (otherwise we'd still sweep the rest
                    # of the scales in this rotation task).
                    if max(c["score"] for c in candidates_for_scale) > 0.85:
                        return all_candidates
                else:
                    coarse_succeeded[scale] = False

    # --- Full pass for scales that didn't yield coarse candidates ---
    pending = [
        j for j in scale_jobs if not coarse_succeeded.get(float(j["scale"]), False)
    ]
    if not pending:
        return all_candidates

    if used_torch:
        try:
            import torch

            full_groups: Dict[Tuple[int, int], List[Dict]] = {}
            for j in pending:
                key = (int(j["hs"]), int(j["ws"]))
                full_groups.setdefault(key, []).append(j)

            for (hs, ws), jobs in full_groups.items():
                flat_size = (tw - ws + 1) * (th - hs + 1)
                if flat_size <= 0:
                    continue
                k = min(
                    flat_size,
                    max(top_match_count * top_match_scan_multiplier, top_match_count),
                )
                patches_np = [
                    np.ascontiguousarray(
                        j["patt_masked"].astype(np.float32, copy=False)
                    )
                    for j in jobs
                ]
                patches_t = torch.from_numpy(np.stack(patches_np, axis=0))[:, None].to(
                    torch_ctx.device
                )
                vals_b, idxs_b, res_w = torch_ctx.topk_full_batch(patches_t, k)
                for j, vals, idxs in zip(jobs, vals_b, idxs_b):
                    scale = float(j["scale"])
                    core_offset_x = float(j["core_offset_x"])
                    core_offset_y = float(j["core_offset_y"])
                    order = idxs
                    all_candidates.extend(
                        _scan_candidates_topk(
                            vals,
                            order,
                            res_w,
                            int(j["ws"]),
                            int(j["hs"]),
                            rot,
                            scale,
                            cell_w,
                            cell_h,
                            cols,
                            rows,
                            grid_center_weight,
                            top_match_count,
                            offset_x=0,
                            offset_y=0,
                            core_offset_x=core_offset_x,
                            core_offset_y=core_offset_y,
                        )
                    )
                if all_candidates and max(c["score"] for c in all_candidates) > 0.85:
                    return all_candidates
            return all_candidates
        except Exception as exc:  # pragma: no cover
            logger.warning("Torch full batch topk failed; falling back: %s", exc)

    # Fallback: original per-scale full match.
    for j in pending:
        ws = int(j["ws"])
        hs = int(j["hs"])
        scale = float(j["scale"])
        res = _match_template(
            template_blur_f32,
            j["patt_masked"],
            corr_method,
            config,
            torch_ctx,
            template_kind="full",
        )
        if res.size:
            flat = res.ravel()
            order = _candidate_order(flat, top_match_count, top_match_scan_multiplier)
            res_w = res.shape[1]
            all_candidates.extend(
                _scan_candidates(
                    res,
                    order,
                    res_w,
                    ws,
                    hs,
                    rot,
                    scale,
                    cell_w,
                    cell_h,
                    cols,
                    rows,
                    grid_center_weight,
                    top_match_count,
                    offset_x=0,
                    offset_y=0,
                    core_offset_x=float(j["core_offset_x"]),
                    core_offset_y=float(j["core_offset_y"]),
                )
            )

    return all_candidates


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

    th, tw = T_blur_f32.shape[:2]
    cell_w = tw / cols
    cell_h = th / rows

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

    torch_ctx: Optional[_TorchMatchContext] = None
    if config.use_torch and corr_method == cv2.TM_CCORR_NORMED and _torch_available():
        try:
            chosen_device = (config.torch_device or _default_torch_device()).strip()
            if chosen_device.lower() == "cpu":
                # Torch CPU path is extremely slow and can be memory-hungry for this workload.
                # Fall back to OpenCV to avoid surprising stalls.
                logger.info("Torch device is CPU; falling back to OpenCV")
                raise RuntimeError("Torch CPU device not supported in matcher")
            torch_ctx = _TorchMatchContext(
                T_blur_f32,
                T_coarse_blur if use_coarse else None,
                device=chosen_device,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Torch init failed; using OpenCV: %s", exc)
            torch_ctx = None

    # Check if we should use parallel execution
    # Threshold of 8 configs balances parallelization benefits vs. thread creation overhead.
    # Below 8, the overhead of thread management exceeds the performance gains.
    num_configs = len(scales) * len(rotations)
    use_parallel = config.parallel_matching and num_configs >= 8

    # Torch fast-path: batch across *all* rotations+scales by patch size.
    # This reduces the number of conv2d launches compared to per-rotation sweeps.
    if (
        config.use_torch
        and corr_method == cv2.TM_CCORR_NORMED
        and torch_ctx is not None
        and torch_ctx.enabled
    ):
        P = (
            (piece_bin_pattern * 255).astype(np.uint8)
            if piece_bin_pattern.max() <= 1
            else piece_bin_pattern.astype(np.uint8)
        )
        if piece_mask.max() <= 1:
            M = (piece_mask > 0).astype(np.uint8) * 255
        else:
            M = (piece_mask > 127).astype(np.uint8) * 255

        dilate_ker = MATCH_DILATE_KERNEL

        jobs: List[Dict] = []
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
                    patt_s_blur = cv2.GaussianBlur(patt_s, blur_ksz, 0).astype(
                        np.float32
                    )
                else:
                    patt_s_blur = patt_s.astype(np.float32)
                patt_masked = patt_s_blur * mask_s

                job = {
                    "rot": int(rot),
                    "scale": float(scale),
                    "ws": int(ws),
                    "hs": int(hs),
                    "core_offset_x": float(core_offset_x),
                    "core_offset_y": float(core_offset_y),
                    "patt_masked": patt_masked,
                }

                if use_coarse and T_coarse_blur is not None and torch_ctx.has_coarse:
                    ws_c = max(1, int(round(ws * coarse_factor)))
                    hs_c = max(1, int(round(hs * coarse_factor)))
                    if (
                        1 < ws_c < T_coarse_blur.shape[1]
                        and 1 < hs_c < T_coarse_blur.shape[0]
                    ):
                        patt_c = _adaptive_coarse_resize(
                            patt_s_blur,
                            coarse_factor,
                            config.preserve_edges_coarse,
                        )
                        if patt_c.shape[:2] != (hs_c, ws_c):
                            patt_c = cv2.resize(
                                patt_c,
                                (ws_c, hs_c),
                                interpolation=cv2.INTER_AREA,
                            )
                        mask_c = cv2.resize(
                            mask_s, (ws_c, hs_c), interpolation=cv2.INTER_NEAREST
                        )
                        patt_masked_c = patt_c * mask_c
                        job.update(
                            {
                                "ws_c": int(ws_c),
                                "hs_c": int(hs_c),
                                "patt_masked_c": patt_masked_c,
                            }
                        )
                jobs.append(job)

        if not jobs:
            raise RuntimeError("No match found (binary matcher)")

        # --- Coarse batched pass + ROI refinement (ROI stays OpenCV) ---
        candidates: List[Dict] = []
        coarse_hit: Dict[Tuple[int, float, int], bool] = {}
        if use_coarse and T_coarse_blur is not None and torch_ctx.has_coarse:
            coarse_groups: Dict[Tuple[int, int], List[int]] = {}
            for idx, j in enumerate(jobs):
                if "patt_masked_c" not in j:
                    continue
                key = (int(j["hs_c"]), int(j["ws_c"]))
                coarse_groups.setdefault(key, []).append(idx)

            for (hs_c, ws_c), idxs in coarse_groups.items():
                try:
                    import torch

                    patches_np = [
                        np.ascontiguousarray(
                            jobs[i]["patt_masked_c"].astype(np.float32, copy=False)
                        )
                        for i in idxs
                    ]
                    patches_t = (
                        torch.from_numpy(np.stack(patches_np, axis=0))[:, None]
                        .to(torch_ctx.device)
                        .to(dtype=torch.float32)
                    )
                    flat_size = (T_coarse_blur.shape[1] - ws_c + 1) * (
                        T_coarse_blur.shape[0] - hs_c + 1
                    )
                    if flat_size <= 0:
                        continue
                    k = min(
                        flat_size,
                        max(coarse_top_k * top_match_scan_multiplier, coarse_top_k),
                    )
                    vals_b, idxs_b, res_w_c = torch_ctx.topk_coarse_batch(patches_t, k)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Torch coarse batch failed: %s", exc)
                    continue

                for local_row, job_idx in enumerate(idxs):
                    j = jobs[job_idx]
                    rot = int(j["rot"])
                    scale = float(j["scale"])
                    ws = int(j["ws"])
                    hs = int(j["hs"])
                    core_offset_x = float(j["core_offset_x"])
                    core_offset_y = float(j["core_offset_y"])
                    patt_masked = j["patt_masked"]

                    coarse_positions = _collect_coarse_positions_topk(
                        vals_b[local_row],
                        idxs_b[local_row],
                        res_w_c,
                        ws_c,
                        hs_c,
                        coarse_top_k,
                    )
                    if not coarse_positions:
                        coarse_hit[(rot, scale, ws)] = False
                        continue

                    found_any = False
                    for coarse in coarse_positions[:coarse_top_k]:
                        x_c, y_c = coarse["tl"]
                        x_full = int(round(x_c / coarse_factor))
                        y_full = int(round(y_c / coarse_factor))
                        x_full = max(0, min(x_full, tw - ws))
                        y_full = max(0, min(y_full, th - hs))
                        x0 = max(0, x_full - coarse_padding_pixels)
                        y0 = max(0, y_full - coarse_padding_pixels)
                        x1 = min(tw, x_full + ws + coarse_padding_pixels)
                        y1 = min(th, y_full + hs + coarse_padding_pixels)
                        roi = T_blur_f32[y0:y1, x0:x1]
                        if roi.shape[0] < hs or roi.shape[1] < ws:
                            continue
                        res_fine = _match_template(
                            roi,
                            patt_masked,
                            corr_method,
                            config,
                            torch_ctx,
                            template_kind="roi",
                        )
                        if res_fine.size == 0:
                            continue
                        flat = res_fine.ravel()
                        order = _candidate_order(
                            flat, top_match_count, top_match_scan_multiplier
                        )
                        res_w = res_fine.shape[1]
                        for idx_flat in order:
                            if len(candidates) >= top_match_count * len(rotations):
                                break
                            y, x = divmod(int(idx_flat), res_w)
                            x0f = x + x0
                            y0f = y + y0
                            cx = x0f + core_offset_x
                            cy = y0f + core_offset_y
                            base_score = float(res_fine[y, x])
                            proximity = _grid_center_proximity(
                                cx, cy, cell_w, cell_h, cols, rows
                            )
                            score = base_score + (grid_center_weight * proximity)
                            cand = {
                                "score": score,
                                "score_raw": base_score,
                                "grid_score": proximity,
                                "rot": rot,
                                "scale": scale,
                                "col": int(cx / cell_w) + 1,
                                "row": int(cy / cell_h) + 1,
                                "tl": (int(x0f), int(y0f)),
                                "br": (int(x0f + ws), int(y0f + hs)),
                                "center": (float(cx), float(cy)),
                            }
                            if any(_candidate_is_close(cand, ex) for ex in candidates):
                                continue
                            candidates.append(cand)
                            found_any = True
                            break
                    coarse_hit[(rot, scale, ws)] = found_any

        # --- Full batched pass for remaining jobs ---
        pending_idxs: List[int] = []
        for idx, j in enumerate(jobs):
            rot = int(j["rot"])
            scale = float(j["scale"])
            ws = int(j["ws"])
            if coarse_hit.get((rot, scale, ws), False):
                continue
            pending_idxs.append(idx)

        full_groups: Dict[Tuple[int, int], List[int]] = {}
        for idx in pending_idxs:
            j = jobs[idx]
            key = (int(j["hs"]), int(j["ws"]))
            full_groups.setdefault(key, []).append(idx)

        for (hs, ws), idxs in full_groups.items():
            flat_size = (tw - ws + 1) * (th - hs + 1)
            if flat_size <= 0:
                continue
            k = min(
                flat_size,
                max(top_match_count * top_match_scan_multiplier, top_match_count),
            )
            try:
                import torch

                patches_np = [
                    np.ascontiguousarray(
                        jobs[i]["patt_masked"].astype(np.float32, copy=False)
                    )
                    for i in idxs
                ]
                patches_t = (
                    torch.from_numpy(np.stack(patches_np, axis=0))[:, None]
                    .to(torch_ctx.device)
                    .to(dtype=torch.float32)
                )
                vals_b, idxs_b, res_w = torch_ctx.topk_full_batch(patches_t, k)
            except Exception as exc:  # pragma: no cover
                logger.warning("Torch full batch failed; falling back: %s", exc)
                vals_b = None
                idxs_b = None
                res_w = 0

            if vals_b is None or idxs_b is None:
                for job_idx in idxs:
                    j = jobs[job_idx]
                    res = _match_template(
                        T_blur_f32,
                        j["patt_masked"],
                        corr_method,
                        config,
                        torch_ctx,
                        template_kind="full",
                    )
                    if res.size == 0:
                        continue
                    flat = res.ravel()
                    order = _candidate_order(
                        flat, top_match_count, top_match_scan_multiplier
                    )
                    res_w2 = res.shape[1]
                    candidates.extend(
                        _scan_candidates_topk(
                            flat[order],
                            order,
                            res_w2,
                            int(j["ws"]),
                            int(j["hs"]),
                            int(j["rot"]),
                            float(j["scale"]),
                            cell_w,
                            cell_h,
                            cols,
                            rows,
                            grid_center_weight,
                            top_match_count,
                            core_offset_x=float(j["core_offset_x"]),
                            core_offset_y=float(j["core_offset_y"]),
                        )
                    )
                continue

            for local_row, job_idx in enumerate(idxs):
                j = jobs[job_idx]
                candidates.extend(
                    _scan_candidates_topk(
                        vals_b[local_row],
                        idxs_b[local_row],
                        res_w,
                        int(j["ws"]),
                        int(j["hs"]),
                        int(j["rot"]),
                        float(j["scale"]),
                        cell_w,
                        cell_h,
                        cols,
                        rows,
                        grid_center_weight,
                        top_match_count,
                        core_offset_x=float(j["core_offset_x"]),
                        core_offset_y=float(j["core_offset_y"]),
                    )
                )

        if not candidates:
            raise RuntimeError("No match found (binary matcher)")

        candidates.sort(key=lambda c: c["score"], reverse=True)
        top_matches: List[Dict] = []
        for candidate in candidates:
            _update_top_matches(top_matches, candidate, top_match_count)
        best = top_matches[0]
        return best, top_matches

    if use_parallel:
        # Parallel path: use ThreadPoolExecutor across rotations, sweeping scales
        # within each worker. This avoids re-rotating the piece/mask per scale.

        all_candidates = []
        cpu_count = os.cpu_count() or 1
        # When using torch (esp. on GPU/MPS), avoid duplicating per-rotation
        # preprocessing across scale chunks. One task per rotation is enough.
        if config.use_torch:
            tasks_per_rotation = 1
        else:
            # Use more than one task per rotation when we have spare cores.
            # This keeps the rotation reuse benefit while still saturating the CPU.
            desired_tasks = min(cpu_count, len(rotations) * len(scales))
            tasks_per_rotation = max(
                1,
                min(
                    len(scales),
                    int(math.ceil(desired_tasks / max(1, len(rotations)))),
                ),
            )
        chunk_size = max(1, int(math.ceil(len(scales) / tasks_per_rotation)))
        scale_chunks = [
            scales[i : i + chunk_size] for i in range(0, len(scales), chunk_size)
        ]

        total_tasks = len(rotations) * len(scale_chunks)
        max_workers = min(total_tasks, cpu_count)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _match_rotation_scale_sweep,
                    rot,
                    scale_chunk,
                    piece_bin_pattern,
                    piece_mask,
                    T_blur_f32,
                    T_coarse_blur,
                    torch_ctx,
                    config,
                    knobs_x,
                    knobs_y,
                    blur_ksz,
                    corr_method,
                    cols,
                    rows,
                )
                for rot in rotations
                for scale_chunk in scale_chunks
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_candidates.extend(result)

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
    P = (
        (piece_bin_pattern * 255).astype(np.uint8)
        if piece_bin_pattern.max() <= 1
        else piece_bin_pattern.astype(np.uint8)
    )
    if piece_mask.max() <= 1:
        M = (piece_mask > 0).astype(np.uint8) * 255
    else:
        M = (piece_mask > 127).astype(np.uint8) * 255

    combo_candidates: List[Dict] = []
    dilate_ker = MATCH_DILATE_KERNEL

    for rot in rotations:
        P_r = _rotate_img(P, rot)
        M_r = _rotate_img(M, rot)
        M_r_raw01 = (M_r > 127).astype(np.uint8)
        rot_knobs_x, rot_knobs_y = _rotate_knob_counts(knobs_x, knobs_y, rot)
        core_center = _core_center_from_mask(M_r_raw01, rot_knobs_x, rot_knobs_y)
        M_r = (M_r_raw01 > 0).astype(np.uint8) * 255
        M_r = cv2.morphologyEx(M_r, cv2.MORPH_DILATE, dilate_ker, iterations=1)
        M_r01 = (M_r > 127).astype(np.float32)

        # Optional torch-side preprocessing to avoid per-scale CPU work and uploads.
        torch_p_rot = None
        torch_m_rot = None
        torch_can_preprocess = (
            config.use_torch
            and corr_method == cv2.TM_CCORR_NORMED
            and torch_ctx is not None
            and torch_ctx.enabled
            and not config.resize_rethreshold
            and not use_coarse
        )
        if torch_can_preprocess:
            try:
                import torch

                torch_p_rot = torch.from_numpy(
                    np.ascontiguousarray(P_r.astype(np.float32, copy=False))
                )[None, None].to(torch_ctx.device)
                torch_m_rot = torch.from_numpy(
                    np.ascontiguousarray(M_r01.astype(np.float32, copy=False))
                )[None, None].to(torch_ctx.device)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Torch preprocess init failed: %s", exc)
                torch_p_rot = None
                torch_m_rot = None
        rot_candidates: List[Dict] = []

        def _run_scales(scale_list: List[float]) -> None:
            for scale in scale_list:
                ws = int(round(P_r.shape[1] * scale))
                hs = int(round(P_r.shape[0] * scale))
                if ws <= 0 or hs <= 0 or ws >= tw or hs >= th:
                    continue
                scale_x = ws / float(P_r.shape[1])
                scale_y = hs / float(P_r.shape[0])
                core_offset_x = core_center[0] * scale_x
                core_offset_y = core_center[1] * scale_y

                patt_s = None
                mask_s = None
                patt_s_blur = None
                patt_masked = None

                # Coarse path needs CPU-side patt/mask for ROI refinement.
                if use_coarse:
                    patt_s = _resize_for_match(
                        P_r, ws, hs, rethreshold=config.resize_rethreshold
                    )
                    mask_s = _resize_for_match(M_r01, ws, hs)

                    if blur_ksz is not None:
                        patt_s_blur = cv2.GaussianBlur(patt_s, blur_ksz, 0).astype(
                            np.float32
                        )
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
                        if patt_s_blur is None or mask_s is None:
                            continue
                        patt_c = _adaptive_coarse_resize(
                            patt_s_blur, coarse_factor, config.preserve_edges_coarse
                        )
                        if patt_c.shape[:2] != (hs_c, ws_c):
                            patt_c = cv2.resize(
                                patt_c, (ws_c, hs_c), interpolation=cv2.INTER_AREA
                            )
                        mask_c = cv2.resize(
                            mask_s, (ws_c, hs_c), interpolation=cv2.INTER_NEAREST
                        )
                        patt_masked_c = patt_c * mask_c
                        coarse_positions: List[Dict] = []
                        used_torch_coarse = (
                            config.use_torch
                            and corr_method == cv2.TM_CCORR_NORMED
                            and torch_ctx is not None
                            and torch_ctx.enabled
                            and torch_ctx.has_coarse
                        )
                        if used_torch_coarse:
                            try:
                                flat_size = (T_coarse_blur.shape[1] - ws_c + 1) * (
                                    T_coarse_blur.shape[0] - hs_c + 1
                                )
                                if flat_size > 0:
                                    k = min(
                                        flat_size,
                                        max(
                                            coarse_top_k * top_match_scan_multiplier,
                                            coarse_top_k,
                                        ),
                                    )
                                    for _ in range(3):
                                        vals_c, idxs_c, res_w_c = torch_ctx.topk_coarse(
                                            patt_masked_c, k
                                        )
                                        coarse_positions = (
                                            _collect_coarse_positions_topk(
                                                vals_c,
                                                idxs_c,
                                                res_w_c,
                                                ws_c,
                                                hs_c,
                                                coarse_top_k,
                                            )
                                        )
                                        if (
                                            len(coarse_positions) >= coarse_top_k
                                            or k >= flat_size
                                        ):
                                            break
                                        k = min(flat_size, k * 4)
                            except Exception as exc:  # pragma: no cover
                                logger.warning("Torch coarse topk failed: %s", exc)

                        if not coarse_positions:
                            res_c = _match_template(
                                T_coarse_blur,
                                patt_masked_c,
                                corr_method,
                                config,
                                torch_ctx,
                                template_kind="coarse",
                            )
                            if res_c.size:
                                flat = res_c.ravel()
                                order = _candidate_order(
                                    flat, coarse_top_k, top_match_scan_multiplier
                                )
                                values = flat[order]
                                coarse_positions = _collect_coarse_positions_topk(
                                    values,
                                    order,
                                    res_c.shape[1],
                                    ws_c,
                                    hs_c,
                                    coarse_top_k,
                                )

                        if coarse_positions:
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
                                if patt_masked is None:
                                    continue
                                res = _match_template(
                                    roi,
                                    patt_masked,
                                    corr_method,
                                    config,
                                    torch_ctx,
                                    template_kind="roi",
                                )
                                if res.size == 0:
                                    continue
                                combo_best = _collect_matches(
                                    res,
                                    ws,
                                    hs,
                                    rot,
                                    scale,
                                    cell_w,
                                    cell_h,
                                    cols,
                                    rows,
                                    grid_center_weight,
                                    top_match_count,
                                    top_match_scan_multiplier,
                                    offset_x=x0,
                                    offset_y=y0,
                                    core_offset_x=core_offset_x,
                                    core_offset_y=core_offset_y,
                                )
                                if combo_best:
                                    rot_candidates.extend(combo_best)
                                    combo_added = True

                if not combo_added:
                    used_torch_full = (
                        config.use_torch
                        and corr_method == cv2.TM_CCORR_NORMED
                        and torch_ctx is not None
                        and torch_ctx.enabled
                    )

                    # If coarse matching is disabled and we can preprocess on-device,
                    # build the patch tensor directly on torch_ctx.device and avoid uploads.
                    if (
                        used_torch_full
                        and not use_coarse
                        and torch_p_rot is not None
                        and torch_m_rot is not None
                    ):
                        try:
                            import torch.nn.functional as F

                            in_h = int(torch_p_rot.shape[-2])
                            in_w = int(torch_p_rot.shape[-1])
                            if ws < in_w or hs < in_h:
                                mode = "area"
                                patt_s_t = F.interpolate(
                                    torch_p_rot,
                                    size=(hs, ws),
                                    mode=mode,
                                )
                            else:
                                patt_s_t = F.interpolate(
                                    torch_p_rot,
                                    size=(hs, ws),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            mask_s_t = F.interpolate(
                                torch_m_rot,
                                size=(hs, ws),
                                mode="nearest",
                            )
                            if blur_ksz is not None:
                                patt_s_blur_t = _torch_gaussian_blur2d(
                                    patt_s_t, blur_ksz
                                )
                            else:
                                patt_s_blur_t = patt_s_t
                            patt_masked_t = patt_s_blur_t * mask_s_t

                            flat_size = (tw - ws + 1) * (th - hs + 1)
                            if flat_size <= 0:
                                return
                            k = min(
                                flat_size,
                                max(
                                    top_match_count * top_match_scan_multiplier,
                                    top_match_count,
                                ),
                            )
                            combo_best_topk: List[Dict] = []
                            for _ in range(3):
                                vals, idxs, res_w = torch_ctx.topk_full(
                                    patt_masked_t, k
                                )
                                combo_best_topk = _scan_candidates_topk(
                                    vals,
                                    idxs,
                                    res_w,
                                    ws,
                                    hs,
                                    rot,
                                    scale,
                                    cell_w,
                                    cell_h,
                                    cols,
                                    rows,
                                    grid_center_weight,
                                    top_match_count,
                                    core_offset_x=core_offset_x,
                                    core_offset_y=core_offset_y,
                                )
                                if (
                                    len(combo_best_topk) >= top_match_count
                                    or k >= flat_size
                                ):
                                    break
                                k = min(flat_size, k * 4)
                            if combo_best_topk:
                                rot_candidates.extend(combo_best_topk)
                                return
                        except Exception as exc:  # pragma: no cover
                            logger.warning(
                                "Torch preprocess full match failed: %s", exc
                            )

                    if used_torch_full:
                        try:
                            if patt_masked is None:
                                patt_s = _resize_for_match(
                                    P_r,
                                    ws,
                                    hs,
                                    rethreshold=config.resize_rethreshold,
                                )
                                mask_s = _resize_for_match(M_r01, ws, hs)

                                if blur_ksz is not None:
                                    patt_s_blur = cv2.GaussianBlur(
                                        patt_s, blur_ksz, 0
                                    ).astype(np.float32)
                                else:
                                    patt_s_blur = patt_s.astype(np.float32)

                                patt_masked = patt_s_blur * mask_s

                            flat_size = (tw - ws + 1) * (th - hs + 1)
                            if flat_size <= 0:
                                return
                            k = min(
                                flat_size,
                                max(
                                    top_match_count * top_match_scan_multiplier,
                                    top_match_count,
                                ),
                            )
                            combo_best_topk: List[Dict] = []
                            for _ in range(3):
                                vals, idxs, res_w = torch_ctx.topk_full(patt_masked, k)
                                combo_best_topk = _scan_candidates_topk(
                                    vals,
                                    idxs,
                                    res_w,
                                    ws,
                                    hs,
                                    rot,
                                    scale,
                                    cell_w,
                                    cell_h,
                                    cols,
                                    rows,
                                    grid_center_weight,
                                    top_match_count,
                                    core_offset_x=core_offset_x,
                                    core_offset_y=core_offset_y,
                                )
                                if (
                                    len(combo_best_topk) >= top_match_count
                                    or k >= flat_size
                                ):
                                    break
                                k = min(flat_size, k * 4)
                            if combo_best_topk:
                                rot_candidates.extend(combo_best_topk)
                                return
                        except Exception as exc:  # pragma: no cover
                            logger.warning("Torch full topk failed: %s", exc)

                    if patt_masked is None:
                        patt_s = _resize_for_match(
                            P_r, ws, hs, rethreshold=config.resize_rethreshold
                        )
                        mask_s = _resize_for_match(M_r01, ws, hs)
                        if blur_ksz is not None:
                            patt_s_blur = cv2.GaussianBlur(patt_s, blur_ksz, 0).astype(
                                np.float32
                            )
                        else:
                            patt_s_blur = patt_s.astype(np.float32)
                        patt_masked = patt_s_blur * mask_s

                    res = _match_template(
                        T_blur_f32,
                        patt_masked,
                        corr_method,
                        config,
                        torch_ctx,
                        template_kind="full",
                    )

                    if res.size == 0:
                        return

                    combo_best = _collect_matches(
                        res,
                        ws,
                        hs,
                        rot,
                        scale,
                        cell_w,
                        cell_h,
                        cols,
                        rows,
                        grid_center_weight,
                        top_match_count,
                        top_match_scan_multiplier,
                        core_offset_x=core_offset_x,
                        core_offset_y=core_offset_y,
                    )
                    rot_candidates.extend(combo_best)

        _run_scales(list(scales))

        combo_candidates.extend(rot_candidates)

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
    render_full_res: bool = True,
) -> np.ndarray:
    rot = match["rot"]
    rot_bin = _rotate_img(_binary_to_uint8(piece_bin), rot)
    rot_mask = _rotate_img(_binary_to_uint8(piece_mask), rot)
    if render_full_res:
        rv = rot_bin
        rv_mask = rot_mask
    else:
        ws = max(1, match["br"][0] - match["tl"][0])
        hs = max(1, match["br"][1] - match["tl"][1])
        rv = _resize_for_match(rot_bin, ws, hs, rethreshold=rethreshold)
        rv_mask = _resize_for_match(rot_mask, ws, hs)
    rv = (rv * (rv_mask > 127)).astype(np.uint8)
    return rv


def _render_masked_piece_view(
    piece_rgb: np.ndarray,
    piece_mask: np.ndarray,
    match: Dict,
    render_full_res: bool = True,
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
    if not render_full_res and (
        rot_rgb.shape[0] != target_h or rot_rgb.shape[1] != target_w
    ):
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
    piece_bgr = _load_image(piece_image_path)
    return find_piece_in_template_bgr(
        piece_bgr,
        template_image_path,
        knobs_x=knobs_x,
        knobs_y=knobs_y,
        auto_align=auto_align,
        infer_knobs=infer_knobs,
        template_rotation=template_rotation,
        matcher_config=matcher_config,
    )


def find_piece_in_template_bgr(
    piece_bgr: np.ndarray,
    template_image_path: str,
    knobs_x: Optional[int],
    knobs_y: Optional[int],
    auto_align: bool = False,
    infer_knobs: Optional[bool] = None,
    template_rotation: Optional[int] = None,
    matcher_config: Optional[MatcherConfig] = None,
) -> MatchPayload:
    """Like find_piece_in_template, but accepts a BGR numpy array for the piece.

    This avoids repeated disk I/O when solving multi-piece images.
    """

    if piece_bgr is None:
        raise ValueError("piece_bgr must be a numpy array")
    if piece_bgr.ndim == 2:
        piece = cv2.cvtColor(piece_bgr, cv2.COLOR_GRAY2BGR)
    elif piece_bgr.ndim == 3 and piece_bgr.shape[2] == 4:
        piece = cv2.cvtColor(piece_bgr, cv2.COLOR_BGRA2BGR)
    else:
        piece = piece_bgr

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

    template_bgr = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2BGR)
    template_mask01 = (template_bin > 0).astype(np.uint8)
    piece_mask = compute_piece_mask(
        piece,
        config,
        template_bgr=template_bgr,
        template_mask=template_mask01,
    )
    knob_mask: Optional[np.ndarray] = None
    if (config.mask_mode or "blue").lower() in ("hsv", "hsv_ranges"):
        knob_mask = compute_piece_mask(piece, config)
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
            piece_mask = compute_piece_mask(
                piece,
                config,
                template_bgr=template_bgr,
                template_mask=template_mask01,
            )
            if knob_mask is not None:
                knob_mask = compute_piece_mask(piece, config)
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
        knob_mask_crop = piece_mask_crop
        if knob_mask is not None:
            ky0, ky1, kx0, kx1 = _mask_bbox(knob_mask)
            knob_mask_crop = knob_mask[ky0:ky1, kx0:kx1].copy()
        knobs_x, knobs_y = _infer_knob_counts(
            knob_mask_crop,
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
        render_full_res=config.render_full_res,
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
            render_full_res=payload.render_full_res,
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
    piece_view = _render_masked_piece_view(
        payload.piece_rgb,
        payload.piece_mask,
        match,
        render_full_res=payload.render_full_res,
    )
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
        min_height=0 if payload.render_full_res else 300,
        max_width=0 if payload.render_full_res else 900,
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
