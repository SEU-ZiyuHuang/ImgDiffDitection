#!/usr/bin/env python3
"""Command-line entry points for daily detection and calibration.

``compare`` handles one expected component.  ``compare-case`` and
``calibrate-case`` handle the repository's case layout, including every
additional reference image in the case directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imagecmp import ImageComparisonService, MultiComponentImageComparisonService


def _component_result_json(result) -> dict:
    artifacts = result.artifacts
    return {
        "state": result.state.value,
        "unavailable_reason": (
            result.unavailable_reason.value if result.unavailable_reason else None
        ),
        "unavailable_detail": result.unavailable_detail,
        "detection_regions": [_region_json(region) for region in result.detection_regions],
        "artifacts": _artifacts_json(artifacts),
        "alignment_metrics": result.alignment_metrics,
        "config_version": result.config_version,
    }


def _image_result_json(result) -> dict:
    return {
        "state": result.state.value,
        "selected_reference": {
            "id": result.selected_reference_id,
            "path": str(result.selected_reference_path),
        },
        "reference_attempts": [_reference_json(item) for item in result.reference_attempts],
        "components": [
            {
                "component_index": item.component_index,
                "category": item.category,
                "state": item.state.value,
                "unavailable_reason": (
                    item.unavailable_reason.value if item.unavailable_reason else None
                ),
                "unavailable_detail": item.unavailable_detail,
                "detection_regions": [_region_json(region) for region in item.detection_regions],
                "artifacts": _artifacts_json(item.artifacts),
                "alignment_metrics": item.alignment_metrics,
            }
            for item in result.component_conclusions
        ],
        "manifest": str(result.manifest_path),
        "config_version": result.config_version,
    }


def _calibration_json(observation) -> dict:
    return {
        "mode": "calibration",
        "selected_reference": {
            "id": observation.selected_reference_id,
            "path": str(observation.selected_reference_path),
        },
        "reference_attempts": [_reference_json(item) for item in observation.reference_attempts],
        "components": [
            {
                "component_index": item.component_index,
                "category": item.category,
                "alignment_metrics": item.alignment_metrics,
                "difference_candidate_count": item.difference_candidate_count,
                "difference_regions": [_region_json(region) for region in item.difference_regions],
                "observation_detail": item.observation_detail,
                "artifacts": _artifacts_json(item.artifacts),
            }
            for item in observation.component_observations
        ],
        "manifest": str(observation.manifest_path),
        "processing_profile_version": observation.processing_profile_version,
    }


def _region_json(region) -> dict:
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
        "confidence": region.confidence,
        "evidence_channels": region.evidence_channels,
        "decision_eligible": region.decision_eligible,
    }


def _artifacts_json(artifacts) -> dict | None:
    if artifacts is None:
        return None
    return {
        "alignment_image": str(artifacts.alignment_image),
        "valid_mask": str(artifacts.valid_mask),
        "difference_mask": str(artifacts.difference_mask),
        "difference_heatmap": str(artifacts.difference_heatmap),
        "annotated_image": str(artifacts.annotated_image),
    }


def _reference_json(item) -> dict:
    return {
        "reference_id": item.reference_id,
        "reference_path": str(item.reference_path),
        "alignment_diagnostic": item.alignment_diagnostic,
        "alignment_metrics": item.alignment_metrics,
        "selected": item.selected,
    }


def _run_single_component(args: argparse.Namespace) -> int:
    result = ImageComparisonService().compare(
        standard_path=Path(args.standard),
        live_path=Path(args.live),
        roi=args.roi,
        output_dir=Path(args.output),
        config_path=Path(args.config),
    )
    print(json.dumps(_component_result_json(result), ensure_ascii=False, indent=2))
    return 0


def _run_daily_case(args: argparse.Namespace) -> int:
    result = MultiComponentImageComparisonService().compare_case_daily(
        case_directory=Path(args.case_directory),
        output_dir=Path(args.output),
        config_path=Path(args.config),
    )
    print(json.dumps(_image_result_json(result), ensure_ascii=False, indent=2))
    return 0


def _run_calibration_case(args: argparse.Namespace) -> int:
    observation = MultiComponentImageComparisonService().calibrate_case(
        case_directory=Path(args.case_directory),
        output_dir=Path(args.output),
        processing_config_path=(Path(args.processing_config)
                                if args.processing_config else None),
    )
    print(json.dumps(_calibration_json(observation), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="供电设备图像异动检测")
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("compare", help="日常检测：比较单个部件")
    single.add_argument("--standard", required=True, help="标准图路径")
    single.add_argument("--live", required=True, help="实时图路径")
    single.add_argument(
        "--roi", required=True,
        help="标准图部件框：'类别 中心横坐标 中心纵坐标 宽 高'",
    )
    single.add_argument("--config", required=True, help="完整标定配置文件")
    single.add_argument("--output", required=True, help="本地证据输出目录")
    single.set_defaults(handler=_run_single_component)

    daily_case = commands.add_parser(
        "compare-case", help="日常检测：处理一个包含多张参考图的案例目录"
    )
    daily_case.add_argument("--case-directory", required=True, help="案例目录")
    daily_case.add_argument("--config", required=True, help="完整标定配置文件")
    daily_case.add_argument("--output", required=True, help="本地证据输出目录")
    daily_case.set_defaults(handler=_run_daily_case)

    calibration_case = commands.add_parser(
        "calibrate-case", help="标定模式：只输出观察指标和证据，不输出业务结论"
    )
    calibration_case.add_argument("--case-directory", required=True, help="案例目录")
    calibration_case.add_argument("--output", required=True, help="本地观察结果目录")
    calibration_case.add_argument(
        "--processing-config", help="可选的完整处理配置；省略时使用开发处理配置"
    )
    calibration_case.set_defaults(handler=_run_calibration_case)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
