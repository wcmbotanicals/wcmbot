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
from wcmbot.viz import draw_grid_on_template


def export_template_with_grid(
    template_id: str,
    dpi: int = 150,
    output_path: str | None = None,
) -> Path:
    """Export template with grid overlay to PNG.

    Args:
        template_id: ID of the template to export
        dpi: Dots per inch for the export (default 150)
        output_path: Output file path (default: output/{template_id}_grid.png)

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
    print(
        f"Applying grid overlay ({template_spec.rows} rows × {template_spec.cols} cols)..."
    )
    template_with_grid = draw_grid_on_template(
        template_img,
        template_spec.rows,
        template_spec.cols,
        rotation=0,  # Export uses original orientation
    )

    # Get target size in pixels
    export_width_cm = template_spec.export_width_cm
    cm_per_inch = 2.54
    pixels_per_inch = dpi
    pixels_per_cm = pixels_per_inch / cm_per_inch

    # Calculate target width in pixels
    target_width = int(export_width_cm * pixels_per_cm)

    # Resize to target size while maintaining aspect ratio
    h, w = template_with_grid.shape[:2]
    scale = target_width / w
    target_height = int(h * scale)
    export_height_cm = export_width_cm * (h / w)

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
    print(f"  Size: {target_width}×{target_height} pixels @ {dpi} DPI")
    print(
        f"  Physical dimensions: {export_width_cm:.1f} cm wide × {export_height_cm:.1f} cm tall"
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

        export_template_with_grid(args.template, dpi=args.dpi, output_path=args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
