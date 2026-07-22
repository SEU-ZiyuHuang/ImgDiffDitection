#!/usr/bin/env python3
"""Command-line interface for single-component image comparison.

Usage::

    python cli.py compare \\
        --standard <path> --live <path> \\
        --roi "17 0.5 0.5 0.4 0.5" \\
        --output <dir> \\
        [--config <path>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when running CLI from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imagecmp.service import ImageComparisonService


def _cmd_compare(args: argparse.Namespace) -> int:
    service = ImageComparisonService()
    try:
        result = service.compare(
            standard_path=Path(args.standard),
            live_path=Path(args.live),
            roi=args.roi,
            output_dir=Path(args.output),
            config_path=Path(args.config) if args.config else None,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = {
        "state": result.state.value,
        "unavailable_reason": result.unavailable_reason.value
        if result.unavailable_reason else None,
        "unavailable_detail": result.unavailable_detail,
        "detection_regions": [
            {
                "x": r.x, "y": r.y,
                "width": r.width, "height": r.height,
                "confidence": r.confidence,
                "evidence_channels": r.evidence_channels,
            }
            for r in result.detection_regions
        ],
        "artifacts": {
            "alignment_image": str(result.artifacts.alignment_image)
            if result.artifacts else None,
            "valid_mask": str(result.artifacts.valid_mask)
            if result.artifacts else None,
            "difference_mask": str(result.artifacts.difference_mask)
            if result.artifacts else None,
            "difference_heatmap": str(result.artifacts.difference_heatmap)
            if result.artifacts else None,
            "annotated_image": str(result.artifacts.annotated_image)
            if result.artifacts else None,
        },
        "alignment_metrics": result.alignment_metrics,
        "config_version": result.config_version,
    }

    import json
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-component image comparison for anomaly detection"
    )
    sub = parser.add_subparsers(dest="command")

    compare_parser = sub.add_parser("compare", help="Compare two images for a single component")
    compare_parser.add_argument("--standard", required=True, help="Standard (reference) image path")
    compare_parser.add_argument("--live", required=True, help="Live (inspection) image path")
    compare_parser.add_argument(
        "--roi", required=True,
        help="YOLO ROI in standard image: 'class_id center_x center_y width height'"
    )
    compare_parser.add_argument("--output", required=True, help="Output directory for evidence artifacts")
    compare_parser.add_argument("--config", help="Optional calibration config JSON path")

    args = parser.parse_args()
    if args.command == "compare":
        return _cmd_compare(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
