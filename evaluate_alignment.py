#!/usr/bin/env python3
"""
Standalone script for evaluating the auto-align procedure.

Usage:
    python evaluate_alignment.py path/to/piece.jpg --rotation 5.0
    python evaluate_alignment.py path/to/piece.jpg -r 0
"""

import argparse
import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------- configuration (from matcher.py) ----------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
LOWER_BLUE1 = np.array([90, 60, 40], dtype=np.uint8)
UPPER_BLUE1 = np.array([140, 255, 255], dtype=np.uint8)
LOWER_BLUE2 = np.array([85, 30, 60], dtype=np.uint8)
UPPER_BLUE2 = np.array([160, 255, 220], dtype=np.uint8)
OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_MASK_AREA_FRAC = 0.0005
AUTO_ALIGN_MIN_DEG = 2.0
AUTO_ALIGN_MAX_DEG = 20.0
AUTO_ALIGN_MIN_LINES = 5
AUTO_ALIGN_MIN_AREA_FRAC = 0.008
AUTO_ALIGN_HOUGH_THRESHOLD = 50
AUTO_ALIGN_HOUGH_MIN_LINE = 40
AUTO_ALIGN_HOUGH_MAX_GAP = 20


# ---------- helper functions ----------
def _load_image(path: str) -> np.ndarray:
    """Load an image from disk."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _background_bgr(img_bgr: np.ndarray) -> Tuple[int, int, int]:
    """Estimate background color from image corners."""
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
    """Rotate an image by the given angle (in degrees)."""
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


def _keep_largest_component(
    mask01: np.ndarray, min_frac: float = MIN_MASK_AREA_FRAC
) -> np.ndarray:
    """Keep only the largest connected component in the mask."""
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
    """Create a binary mask by detecting blue regions."""
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


def _mask_bbox_area(mask01: np.ndarray) -> int:
    """Compute the area of the bounding box around non-zero mask pixels."""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return 0
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def _estimate_mask_tilt(
    mask01: np.ndarray,
) -> Tuple[Optional[float], int, Optional[np.ndarray]]:
    """
    Estimate tilt angle using Hough line detection on mask edges.

    Returns:
        Tuple of (weighted_mean_angle, line_count, lines_array)
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
        return None, 0, None
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
        return None, 0, None

    angles_array = np.array(angles, dtype=np.float32)
    lengths_array = np.array(lengths, dtype=np.float32)
    weighted_mean = np.average(angles_array, weights=lengths_array)

    return float(weighted_mean), len(angles), lines


def _estimate_alignment_from_mask(mask01: np.ndarray) -> Tuple[float, dict]:
    """
    Estimate alignment correction using Hough line detection.

    Returns:
        Tuple of (correction_angle, debug_info_dict)
    """
    debug_info = {}

    angle, line_count, lines = _estimate_mask_tilt(mask01)
    debug_info["detected_angle"] = angle
    debug_info["line_count"] = line_count
    debug_info["hough_lines"] = lines

    if angle is None or line_count < AUTO_ALIGN_MIN_LINES:
        debug_info["reason"] = (
            f"Insufficient lines detected ({line_count} < {AUTO_ALIGN_MIN_LINES})"
        )
        return 0.0, debug_info

    correction = -angle
    debug_info["correction"] = correction

    if abs(correction) > AUTO_ALIGN_MAX_DEG:
        debug_info["reason"] = (
            f"Correction too large ({abs(correction):.2f}° > {AUTO_ALIGN_MAX_DEG}°)"
        )
        return 0.0, debug_info

    area0 = _mask_bbox_area(mask01)
    debug_info["bbox_area_before"] = area0

    if area0 <= 0:
        debug_info["reason"] = "Empty mask bounding box"
        return 0.0, debug_info

    rotated_mask = _rotate_img(
        (mask01 > 0).astype(np.uint8) * 255,
        correction,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )
    area1 = _mask_bbox_area(rotated_mask)
    debug_info["bbox_area_after"] = area1

    if area1 <= 0:
        debug_info["reason"] = "Empty rotated mask bounding box"
        return 0.0, debug_info

    area_delta = (area0 - area1) / float(area0)
    debug_info["area_delta_frac"] = area_delta

    if area_delta < AUTO_ALIGN_MIN_AREA_FRAC:
        debug_info["reason"] = (
            f"Bbox improvement too small ({area_delta:.4f} < {AUTO_ALIGN_MIN_AREA_FRAC})"
        )
        return 0.0, debug_info

    debug_info["reason"] = "Alignment applied"
    return float(correction), debug_info


def visualize_alignment(
    piece_path: str,
    pre_rotation: float = 0.0,
    save_path: Optional[str] = None,
) -> None:
    """
    Evaluate and visualize the auto-alignment procedure.

    Args:
        piece_path: Path to the piece image
        pre_rotation: Rotation angle to apply before auto-alignment (degrees)
        save_path: Optional path to save the visualization
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating alignment for: {os.path.basename(piece_path)}")
    print(f"Pre-rotation: {pre_rotation}°")
    print(f"{'=' * 60}\n")

    # Load and optionally rotate the piece
    piece_original = _load_image(piece_path)

    if abs(pre_rotation) > 0.01:
        bg = _background_bgr(piece_original)
        piece = _rotate_img(
            piece_original,
            pre_rotation,
            interpolation=cv2.INTER_LINEAR,
            border_value=bg,
        )
        print(f"Applied pre-rotation of {pre_rotation}°")
    else:
        piece = piece_original

    # Create mask from blue regions
    print("Creating mask from blue regions...")
    mask_before = _mask_by_blue(piece)

    # Estimate alignment
    print("Estimating alignment correction...")
    correction, debug_info = _estimate_alignment_from_mask(mask_before)

    print(f"\nAlignment Analysis:")
    print(f"  Detected angle: {debug_info.get('detected_angle', 'N/A')}")
    print(f"  Lines detected: {debug_info.get('line_count', 0)}")
    print(f"  Proposed correction: {correction:.2f}°")
    if "bbox_area_before" in debug_info:
        print(f"  BBox area before: {debug_info['bbox_area_before']}")
    if "bbox_area_after" in debug_info:
        print(f"  BBox area after: {debug_info['bbox_area_after']}")
    if "area_delta_frac" in debug_info:
        print(f"  Area improvement: {debug_info['area_delta_frac'] * 100:.2f}%")
    print(f"  Result: {debug_info.get('reason', 'Unknown')}")

    # Apply alignment if significant
    if abs(correction) >= AUTO_ALIGN_MIN_DEG:
        print(f"\nApplying alignment correction of {correction:.2f}°")
        bg = _background_bgr(piece)
        piece_aligned = _rotate_img(
            piece,
            correction,
            interpolation=cv2.INTER_LINEAR,
            border_value=bg,
        )
        mask_after = _mask_by_blue(piece_aligned)
    else:
        print(
            f"\nNo significant correction applied (|{correction:.2f}°| < {AUTO_ALIGN_MIN_DEG}°)"
        )
        piece_aligned = piece
        mask_after = mask_before

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Auto-Alignment Evaluation: {os.path.basename(piece_path)}", fontsize=16
    )

    # Row 1: Original state
    axes[0, 0].imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Before Alignment\n(pre-rot: {pre_rotation}°)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_before, cmap="gray")
    axes[0, 1].set_title(
        f"Mask Before\n(BBox area: {debug_info.get('bbox_area_before', 'N/A')})"
    )
    axes[0, 1].axis("off")

    # Draw Hough lines on mask
    if debug_info.get("hough_lines") is not None:
        mask_with_lines = cv2.cvtColor(
            (mask_before * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        for x1, y1, x2, y2 in debug_info["hough_lines"][:, 0]:
            cv2.line(mask_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[0, 2].imshow(cv2.cvtColor(mask_with_lines, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(
            f"Hough Lines Detected\n({debug_info.get('line_count', 0)} lines)"
        )
    else:
        axes[0, 2].text(
            0.5, 0.5, "No lines detected", ha="center", va="center", fontsize=12
        )
        axes[0, 2].set_title("Hough Lines Detected\n(0 lines)")
    axes[0, 2].axis("off")

    # Row 2: Aligned state
    axes[1, 0].imshow(cv2.cvtColor(piece_aligned, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"After Alignment\n(correction: {correction:.2f}°)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mask_after, cmap="gray")
    axes[1, 1].set_title(
        f"Mask After\n(BBox area: {debug_info.get('bbox_area_after', 'N/A')})"
    )
    axes[1, 1].axis("off")

    # Summary text
    summary_text = (
        f"Pre-rotation: {pre_rotation}°\n"
        f"Detected tilt: {debug_info.get('detected_angle', 'N/A')}\n"
        f"Lines found: {debug_info.get('line_count', 0)}\n"
        f"Correction: {correction:.2f}°\n"
        f"Area improvement: {debug_info.get('area_delta_frac', 0) * 100:.2f}%\n"
        f"\n{debug_info.get('reason', 'Unknown')}"
    )
    axes[1, 2].text(
        0.1,
        0.5,
        summary_text,
        ha="left",
        va="center",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 2].set_title("Summary")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nVisualization saved to: {save_path}")

    plt.show()
    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the auto-align procedure on a puzzle piece image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s media/pieces/piece_1.jpg --rotation 5.0
  %(prog)s media/pieces/piece_2.jpg -r 10 --save alignment_result.png
        """,
    )
    parser.add_argument("piece_path", help="Path to the puzzle piece image")
    parser.add_argument(
        "-r",
        "--rotation",
        type=float,
        default=0.0,
        help="Rotation angle to pre-apply (degrees, default: 0.0)",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Save visualization to this path (optional)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.piece_path):
        print(f"Error: File not found: {args.piece_path}")
        return 1

    try:
        visualize_alignment(
            piece_path=args.piece_path,
            pre_rotation=args.rotation,
            save_path=args.save,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
