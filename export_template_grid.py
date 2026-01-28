#!/usr/bin/env python3
"""Export a puzzle template with grid overlay to PNG.

Usage:
    python export_template_grid.py <puzzle_name> [--dpi DPI] [--output OUTPUT]

Example:
    python export_template_grid.py grass_puzzle
    python export_template_grid.py sample_puzzle --dpi 300 --output my_template.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from wcmbot.template_settings import load_template_registry
from wcmbot.viz import draw_grid_on_template, rotate_template_preview


def export_template_with_grid(
    template_id: str,
    dpi: int = 150,
    output_path: str | None = None,
    rotation: int | None = None,
) -> Path:
    """Export template with grid overlay to PNG.

    Args:
        template_id: ID of the template to export
        dpi: Dots per inch for the export (default 150)
        output_path: Output file path (default: output/{template_id}_grid.png)
        rotation: Template rotation in degrees (0, 90, 180, 270). If None, uses template default.

    Returns:
        Path to the exported PNG file
    """
    # Load template registry
    registry = load_template_registry()
    template_spec = registry.get(template_id)

    if template_spec is None:
        raise ValueError(f"Template '{template_id}' not found in registry")

    # Load template image
    if not template_spec.template_path.exists():
        raise FileNotFoundError(
            f"Template image not found: {template_spec.template_path}"
        )

    template_img = np.array(Image.open(template_spec.template_path))

    # Convert RGBA to RGB if necessary
    if template_img.shape[2] == 4:
        template_img = template_img[:, :, :3]

    # Apply cropping if specified
    if template_spec.crop_x > 0 or template_spec.crop_y > 0:
        h, w = template_img.shape[:2]
        template_img = template_img[
            template_spec.crop_y : h - template_spec.crop_y,
            template_spec.crop_x : w - template_spec.crop_x,
        ]

    # Apply grid overlay
    if rotation is None:
        rotation = template_spec.default_rotation
    print(
        f"Applying grid overlay ({template_spec.rows} rows × {template_spec.cols} cols, "
        f"rotation: {rotation}°)..."
    )

    # Rotate the image first
    rotated_img = rotate_template_preview(template_img, rotation)

    # Determine grid dimensions based on rotation
    # When rotating 90 or 270, rows and cols swap
    grid_rows = template_spec.rows
    grid_cols = template_spec.cols
    if rotation in (90, 270):
        grid_rows, grid_cols = grid_cols, grid_rows

    # Apply grid to rotated image with no additional rotation
    # Note: draw_grid_on_template adds 40px margins on each side for labels
    template_with_grid = draw_grid_on_template(
        rotated_img,
        grid_rows,
        grid_cols,
        rotation=0,  # Image is already rotated, so no rotation needed here
    )

    # Get target size in pixels
    # export_width_cm is the width of the template *before* margins are added
    export_width_cm = template_spec.export_width_cm
    cm_per_inch = 2.54
    pixels_per_inch = dpi
    pixels_per_cm = pixels_per_inch / cm_per_inch

    # Calculate target width in pixels (for the template portion)
    template_target_width = int(export_width_cm * pixels_per_cm)

    # The grid image has margins, so we need to scale based on template portion
    # Grid adds 80px total margin (40px on each side)
    template_h, template_w = rotated_img.shape[:2]
    grid_h, grid_w = template_with_grid.shape[:2]

    # Scale to get target template width, then add margins
    scale = template_target_width / template_w
    target_width = int(grid_w * scale)

    # Resize to target size while maintaining aspect ratio
    h, w = template_with_grid.shape[:2]
    target_height = int(h * scale)

    # Calculate physical dimensions
    template_height_cm = export_width_cm * (template_h / template_w)
    total_width_cm = export_width_cm * (grid_w / template_w)
    total_height_cm = template_height_cm * (grid_h / template_h)
    margin_cm = (total_width_cm - export_width_cm) / 2

    print(f"Resizing to {target_width}×{target_height} pixels ({dpi} DPI)...")
    resized = cv2.resize(
        template_with_grid,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # Convert RGB to BGR for OpenCV, then back to RGB for PIL
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    output_img = Image.fromarray(cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2RGB))

    # Determine output path
    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{template_id}_grid_{dpi}dpi.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    output_img.save(output_path, "PNG", dpi=(dpi, dpi))
    print(f"✓ Exported to: {output_path}")
    print(f"  Pixel size: {target_width}×{target_height} @ {dpi} DPI")
    print(f"  Template: {export_width_cm:.1f} cm × {template_height_cm:.1f} cm")
    print(
        f"  With margins: {total_width_cm:.1f} cm × {total_height_cm:.1f} cm (±{margin_cm:.1f} cm margins)"
    )

    return output_path


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Export a puzzle template with grid overlay to PNG"
    )
    parser.add_argument(
        "template", nargs="?", help="Template ID (e.g., 'grass_puzzle')"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Dots per inch for export (default: 150)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: output/{template}_grid_{dpi}dpi.png)",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        help="Template rotation in degrees (default: use template's default_rotation)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available templates",
    )

    args = parser.parse_args()

    try:
        if args.list:
            registry = load_template_registry()
            print("Available templates:")
            for spec in registry.templates.values():
                print(f"  - {spec.template_id}: {spec.label}")
                print(
                    f"    ({spec.rows}×{spec.cols} grid, export width: {spec.export_width_cm} cm)"
                )
            return

        if args.template is None:
            parser.error("template is required unless --list is used")

        export_template_with_grid(
            args.template, dpi=args.dpi, output_path=args.output, rotation=args.rotation
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
