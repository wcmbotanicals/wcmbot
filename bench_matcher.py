#!/usr/bin/env python3

import argparse
import math
import os
import statistics
import time
from typing import Dict, List, Tuple

import matcher

CASE_MATRIX: List[Tuple[str, int, int]] = [
    ("piece_1.jpg", 0, 0),
    ("piece_2.jpg", 2, 2),
    ("piece_3.jpg", 2, 2),
]


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _format_ms(value_s: float) -> str:
    return f"{value_s * 1000.0:.2f} ms"


def _resolve_cases(selected: List[str]) -> List[Tuple[str, int, int]]:
    if not selected:
        return CASE_MATRIX
    by_name = {name: (name, kx, ky) for name, kx, ky in CASE_MATRIX}
    resolved = []
    for name in selected:
        item = by_name.get(name)
        if not item:
            raise ValueError(f"Unknown case '{name}'. Available: {', '.join(by_name)}")
        resolved.append(item)
    return resolved


def _run_benchmark(
    template_path: str,
    pieces_dir: str,
    cases: List[Tuple[str, int, int]],
    iterations: int,
    repeats: int,
    warmup: int,
) -> Dict[str, List[float]]:
    timings: Dict[str, List[float]] = {name: [] for name, _, _ in cases}

    for _ in range(warmup):
        for name, knobs_x, knobs_y in cases:
            piece_path = os.path.join(pieces_dir, name)
            matcher.find_piece_in_template(
                piece_image_path=piece_path,
                template_image_path=template_path,
                knobs_x=knobs_x,
                knobs_y=knobs_y,
            )

    for _ in range(repeats):
        for _ in range(iterations):
            for name, knobs_x, knobs_y in cases:
                piece_path = os.path.join(pieces_dir, name)
                start = time.perf_counter()
                matcher.find_piece_in_template(
                    piece_image_path=piece_path,
                    template_image_path=template_path,
                    knobs_x=knobs_x,
                    knobs_y=knobs_y,
                )
                timings[name].append(time.perf_counter() - start)

    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark matcher runtime.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Iterations per repeat (per case).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeat count for the iteration loop.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup passes before timing.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case filename to benchmark (repeatable).",
    )
    parser.add_argument(
        "--coarse-factor",
        type=float,
        default=None,
        help="Override matcher.COARSE_FACTOR.",
    )
    parser.add_argument(
        "--coarse-top-k",
        type=int,
        default=None,
        help="Override matcher.COARSE_TOP_K.",
    )
    parser.add_argument(
        "--coarse-pad",
        type=int,
        default=None,
        help="Override matcher.COARSE_PAD_PX.",
    )
    parser.add_argument(
        "--coarse-min-side",
        type=int,
        default=None,
        help="Override matcher.COARSE_MIN_SIDE.",
    )
    parser.add_argument(
        "--coarse-off",
        action="store_true",
        help="Disable coarse matching pass.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "media", "templates", "sample_puzzle.png")
    pieces_dir = os.path.join(base_dir, "media", "pieces")

    if args.coarse_off:
        matcher.COARSE_FACTOR = 0.0
    if args.coarse_factor is not None:
        matcher.COARSE_FACTOR = args.coarse_factor
    if args.coarse_top_k is not None:
        matcher.COARSE_TOP_K = args.coarse_top_k
    if args.coarse_pad is not None:
        matcher.COARSE_PAD_PX = args.coarse_pad
    if args.coarse_min_side is not None:
        matcher.COARSE_MIN_SIDE = args.coarse_min_side

    cases = _resolve_cases(args.case)
    for name, _, _ in cases:
        piece_path = os.path.join(pieces_dir, name)
        if not os.path.exists(piece_path):
            raise FileNotFoundError(f"Missing piece image: {piece_path}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Missing template image: {template_path}")

    timings = _run_benchmark(
        template_path=template_path,
        pieces_dir=pieces_dir,
        cases=cases,
        iterations=args.iterations,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    total_runs = sum(len(v) for v in timings.values())
    print(
        f"Runs: {total_runs} | cases: {len(cases)} | "
        f"iterations: {args.iterations} | repeats: {args.repeats} | warmup: {args.warmup}"
    )
    print(
        "coarse:",
        f"factor={matcher.COARSE_FACTOR}",
        f"top_k={matcher.COARSE_TOP_K}",
        f"pad={matcher.COARSE_PAD_PX}",
        f"min_side={matcher.COARSE_MIN_SIDE}",
    )

    combined = []
    for name, values in timings.items():
        combined.extend(values)
        sorted_vals = sorted(values)
        print(
            f"{name}: median {_format_ms(statistics.median(sorted_vals))}, "
            f"mean {_format_ms(statistics.mean(sorted_vals))}, "
            f"p95 {_format_ms(_percentile(sorted_vals, 95))}, "
            f"min {_format_ms(sorted_vals[0])}, "
            f"max {_format_ms(sorted_vals[-1])}"
        )

    if combined:
        sorted_all = sorted(combined)
        print(
            f"overall: median {_format_ms(statistics.median(sorted_all))}, "
            f"mean {_format_ms(statistics.mean(sorted_all))}, "
            f"p95 {_format_ms(_percentile(sorted_all, 95))}, "
            f"min {_format_ms(sorted_all[0])}, "
            f"max {_format_ms(sorted_all[-1])}"
        )


if __name__ == "__main__":
    main()
