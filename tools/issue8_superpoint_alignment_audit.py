#!/usr/bin/env python3
"""Audit the optional SuperPoint + LightGlue alignment fallback locally.

The tool deliberately evaluates image-level alignment only.  It does not read
ROI labels, run component difference detection, emit a daily business state,
or copy any source image.  For each case it reproduces the production
reference-selection rule twice: first with ORB only, then with the optional
fallback enabled.  A fallback result is called a rescue only when the selected
ORB-only reference is not usable and the selected fallback result is usable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imagecmp_py"))

from imagecmp.alignment import AlignmentDiagnostic, align, clear_alignment_cache  # noqa: E402
from imagecmp.config import CalibratedConfig, load_config  # noqa: E402
from imagecmp.multi_component import _reference_score  # noqa: E402
from imagecmp.references import discover_case_input, read_color_image  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--opencv-threads", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be positive")
    if arguments.shard_count <= 0 or not 0 <= arguments.shard_index < arguments.shard_count:
        parser.error("invalid shard index/count")
    if arguments.opencv_threads <= 0:
        parser.error("--opencv-threads must be positive")
    return arguments


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_case(case_directory: Path, config: CalibratedConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return exact image-level reference selection evidence for one config."""
    clear_alignment_cache()
    case = discover_case_input(case_directory)
    live = read_color_image(case.live_path)
    if live is None:
        raise RuntimeError(f"cannot decode live image: {case.live_path}")
    live_gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)

    attempts: list[dict[str, Any]] = []
    scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for order, reference in enumerate(case.references):
        standard = read_color_image(reference.path)
        if standard is None:
            row = {
                "reference_id": reference.reference_id,
                "reference_path": str(reference.path),
                "diagnostic": "reference_unreadable",
                "method": "none",
                "fallback_attempted": 0,
                "fallback_diagnostic": "",
                "fallback_match_count": 0,
                "fallback_elapsed_ms": None,
            }
            score = (-1.0, -float(order))
        else:
            result = align(cv2.cvtColor(standard, cv2.COLOR_BGR2GRAY), live_gray, config)
            metrics = result.as_metrics_dict()
            row = {
                "reference_id": reference.reference_id,
                "reference_path": str(reference.path),
                "diagnostic": result.diagnostic.value,
                "method": result.correspondence_method,
                "fallback_attempted": int(metrics.get("alignment_fallback_attempted", 0)),
                "fallback_diagnostic": metrics.get("alignment_fallback_diagnostic", ""),
                "fallback_match_count": metrics.get("alignment_fallback_feature_match_count", 0),
                "fallback_elapsed_ms": metrics.get("alignment_fallback_elapsed_ms"),
            }
            score = _reference_score(result, order)
        attempts.append(row)
        scored.append((score, row))

    selected_score, selected = max(scored, key=lambda item: item[0])
    del selected_score
    return selected, attempts


def _summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    selected = [row for row in rows if not row.get("error")]
    diagnostics = Counter(row[f"{prefix}_diagnostic"] for row in selected)
    methods = Counter(row[f"{prefix}_method"] for row in selected)
    return {
        "cases": len(selected),
        "selected_diagnostics": dict(diagnostics),
        "selected_methods": dict(methods),
        "usable_cases": diagnostics[AlignmentDiagnostic.USABLE.value],
        "usable_rate": diagnostics[AlignmentDiagnostic.USABLE.value] / len(selected) if selected else None,
    }


def run(arguments: argparse.Namespace) -> int:
    cv2.setNumThreads(arguments.opencv_threads)
    input_root = arguments.input_root.resolve()
    output = arguments.output_directory.resolve()
    if not input_root.is_dir():
        raise ValueError(f"input root does not exist: {input_root}")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    config = load_config(arguments.config.resolve())
    fallback_config = replace(
        config,
        alignment=replace(config.alignment, superpoint_lightglue_fallback_enabled=True),
    )
    directories = sorted(path for path in input_root.iterdir() if path.is_dir())
    directories = directories[arguments.shard_index::arguments.shard_count]
    if arguments.limit is not None:
        directories = directories[:arguments.limit]

    case_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for index, directory in enumerate(directories, start=1):
        row: dict[str, Any] = {"case": directory.name}
        try:
            orb_selected, orb_attempts = _audit_case(directory, config)
            fallback_selected, fallback_attempts = _audit_case(directory, fallback_config)
            row.update({
                "orb_reference_id": orb_selected["reference_id"],
                "orb_diagnostic": orb_selected["diagnostic"],
                "orb_method": orb_selected["method"],
                "fallback_reference_id": fallback_selected["reference_id"],
                "fallback_diagnostic": fallback_selected["diagnostic"],
                "fallback_method": fallback_selected["method"],
                "selected_reference_changed": int(
                    orb_selected["reference_id"] != fallback_selected["reference_id"]
                ),
                "selected_fallback_attempted": fallback_selected["fallback_attempted"],
                "rescued_to_usable": int(
                    orb_selected["diagnostic"] != AlignmentDiagnostic.USABLE.value
                    and fallback_selected["diagnostic"] == AlignmentDiagnostic.USABLE.value
                ),
                "regressed_from_usable": int(
                    orb_selected["diagnostic"] == AlignmentDiagnostic.USABLE.value
                    and fallback_selected["diagnostic"] != AlignmentDiagnostic.USABLE.value
                ),
            })
            for mode, attempts in (("orb", orb_attempts), ("fallback", fallback_attempts)):
                for attempt in attempts:
                    reference_rows.append({"case": directory.name, "mode": mode, **attempt})
            print(
                f"[{index}/{len(directories)}] {directory.name}: "
                f"{row['orb_diagnostic']} -> {row['fallback_diagnostic']}",
                flush=True,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            row["error"] = str(exc)
            print(f"[{index}/{len(directories)}] failed {directory.name}: {exc}", flush=True)
        case_rows.append(row)

    valid_rows = [row for row in case_rows if not row.get("error")]
    report = {
        "report_kind": "issue8_superpoint_lightglue_alignment_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Local alignment-only audit. It records whether the optional model "
            "can rescue the existing trusted-alignment gates; it is not an anomaly "
            "accuracy, recall, precision, or deployment-readiness result."
        ),
        "input_root": str(input_root),
        "config": str(arguments.config.resolve()),
        "shard_index": arguments.shard_index,
        "shard_count": arguments.shard_count,
        "requested_cases": len(directories),
        "failed_cases": sum(bool(row.get("error")) for row in case_rows),
        "orb_only": _summary(case_rows, "orb"),
        "with_superpoint_lightglue": _summary(case_rows, "fallback"),
        "comparison": {
            "rescued_to_usable_cases": sum(int(row.get("rescued_to_usable", 0)) for row in valid_rows),
            "regressed_from_usable_cases": sum(int(row.get("regressed_from_usable", 0)) for row in valid_rows),
            "selected_reference_changed_cases": sum(int(row.get("selected_reference_changed", 0)) for row in valid_rows),
            "selected_model_alignment_cases": sum(
                row.get("fallback_method") == "superpoint_lightglue" for row in valid_rows
            ),
            "fallback_attempted_reference_attempts": sum(
                int(row.get("fallback_attempted", 0)) for row in reference_rows
                if row.get("mode") == "fallback"
            ),
        },
    }
    _write_json(output / "alignment_audit_report.json", report)
    _write_csv(output / "case_alignment_audit.csv", case_rows)
    _write_csv(output / "reference_alignment_audit.csv", reference_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed_cases"] == 0 else 1


def main() -> int:
    try:
        raise SystemExit(run(_arguments()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
