#!/usr/bin/env python3
"""Run calibration observations for every direct case under a local dataset root.

The tool deliberately uses the public multi-component calibration service.
It never issues daily-detection business conclusions and never uploads or
copies source images.  Every completed case receives its own evidence
directory and calibration_observation.json under --output-root.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_MODULE_ROOT = REPOSITORY_ROOT / "imagecmp_py"
sys.path.insert(0, str(PYTHON_MODULE_ROOT))

from imagecmp import MultiComponentImageComparisonService  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write per-case calibration observations and evidence for a local "
            "image-comparison dataset."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--processing-config",
        type=Path,
        help="optional complete processing configuration for calibration observations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N case directories, in name order",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases that already contain calibration_observation.json",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based index of this independent batch shard (default: 0)",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of independent batch shards sharing one output root (default: 1)",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        help="maximum OpenCV worker threads used by this batch process",
    )
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be a positive integer")
    if arguments.shard_count <= 0:
        parser.error("--shard-count must be a positive integer")
    if not 0 <= arguments.shard_index < arguments.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if arguments.opencv_threads is not None and arguments.opencv_threads <= 0:
        parser.error("--opencv-threads must be a positive integer")
    return arguments


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_paths(
    input_root: Path, output_root: Path, resume: bool, shard_count: int
) -> tuple[Path, Path]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise ValueError(f"input root is not an existing directory: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("output root must not be inside the input dataset")
    if output_root.exists() and not resume and shard_count == 1:
        raise ValueError(
            f"output root already exists: {output_root}; choose a new path or use --resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return input_root, output_root


def _case_directories(
    input_root: Path, limit: int | None, shard_index: int, shard_count: int
) -> list[Path]:
    directories = sorted(path for path in input_root.iterdir() if path.is_dir())
    directories = directories[shard_index::shard_count]
    return directories if limit is None else directories[:limit]


def _summary_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "completed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "reference_images": 0,
        "component_observations": 0,
    }
    for record in records:
        status = record["status"]
        if status in counts:
            counts[status] += 1
        counts["reference_images"] += int(record.get("reference_count", 0))
        counts["component_observations"] += int(record.get("component_count", 0))
    return counts


def _write_progress_summary(path: Path, summary: dict[str, Any]) -> None:
    summary["counts"] = _summary_counts(summary["cases"])
    summary["updated_at_utc"] = _utc_now()
    _write_json(path, summary)


def _load_resume_summary(
    summary_path: Path, input_root: Path, output_root: Path
) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from invalid batch summary: {exc}") from exc
    if not isinstance(summary, dict) or not isinstance(summary.get("cases"), list):
        raise ValueError("cannot resume from batch summary without a cases list")
    if summary.get("input_root") != str(input_root):
        raise ValueError("batch summary input root does not match --input-root")
    if summary.get("output_root") != str(output_root):
        raise ValueError("batch summary output root does not match --output-root")
    summary["resumed_at_utc"] = _utc_now()
    return summary


def run(arguments: argparse.Namespace) -> int:
    if arguments.opencv_threads is not None:
        cv2.setNumThreads(arguments.opencv_threads)
    input_root, output_root = _validate_paths(
        arguments.input_root,
        arguments.output_root,
        arguments.resume,
        arguments.shard_count,
    )
    directories = _case_directories(
        input_root,
        arguments.limit,
        arguments.shard_index,
        arguments.shard_count,
    )
    summary_filename = (
        "batch_summary.json"
        if arguments.shard_count == 1
        else (
            f"batch_summary_shard_{arguments.shard_index:02d}_"
            f"of_{arguments.shard_count:02d}.json"
        )
    )
    summary_path = output_root / summary_filename
    if arguments.resume and summary_path.is_file():
        summary = _load_resume_summary(summary_path, input_root, output_root)
        summary["case_count_requested"] = len(directories)
    else:
        summary = {
            "mode": "calibration",
            "input_root": str(input_root),
            "output_root": str(output_root),
            "processing_config": (
                str(arguments.processing_config.resolve())
                if arguments.processing_config is not None
                else None
            ),
            "opencv_threads": cv2.getNumThreads(),
            "started_at_utc": _utc_now(),
            "case_count_requested": len(directories),
            "shard_index": arguments.shard_index,
            "shard_count": arguments.shard_count,
            "cases": [],
        }
    recorded_cases = {
        record.get("case")
        for record in summary["cases"]
        if isinstance(record, dict) and isinstance(record.get("case"), str)
    }
    service = MultiComponentImageComparisonService()

    for ordinal, case_directory in enumerate(directories, start=1):
        relative_case_path = case_directory.relative_to(input_root)
        case_output = output_root / relative_case_path
        observation_path = case_output / "calibration_observation.json"
        record: dict[str, Any] = {
            "case": str(relative_case_path),
            "output_directory": str(case_output),
        }
        if arguments.resume and observation_path.is_file():
            if str(relative_case_path) not in recorded_cases:
                record["status"] = "skipped_existing"
                summary["cases"].append(record)
                recorded_cases.add(str(relative_case_path))
            _write_progress_summary(summary_path, summary)
            print(f"[{ordinal}/{len(directories)}] existing {relative_case_path}", flush=True)
            continue

        try:
            observation = service.calibrate_case(
                case_directory=case_directory,
                output_dir=case_output,
                processing_config_path=arguments.processing_config,
            )
            record.update({
                "status": "completed",
                "reference_count": len(observation.reference_attempts),
                "component_count": len(observation.component_observations),
                "selected_reference_id": observation.selected_reference_id,
                "manifest": str(observation.manifest_path),
            })
            print(f"[{ordinal}/{len(directories)}] completed {relative_case_path}", flush=True)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            record.update({
                "status": "failed",
                "error": str(exc),
            })
            print(f"[{ordinal}/{len(directories)}] failed {relative_case_path}: {exc}", flush=True)

        summary["cases"].append(record)
        recorded_cases.add(str(relative_case_path))
        _write_progress_summary(summary_path, summary)

    summary["finished_at_utc"] = _utc_now()
    _write_progress_summary(summary_path, summary)
    print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)
    return 0 if summary["counts"]["failed"] == 0 else 1


def main() -> int:
    try:
        return run(parse_arguments())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
