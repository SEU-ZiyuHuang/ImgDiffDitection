#!/usr/bin/env python3
"""Merge sharded Issue 8 SuperPoint + LightGlue alignment-audit results.

The merged report preserves the case-level definition used by each shard:
one live image is counted as usable only when its selected reference passes
the existing trusted-alignment gates.  It never turns a fallback attempt into
an anomaly conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int(row: dict[str, str], name: str) -> int:
    return int(row.get(name) or 0)


def _summary(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    diagnostics = Counter(row.get(f"{prefix}_diagnostic", "") for row in valid)
    methods = Counter(row.get(f"{prefix}_method", "") for row in valid)
    usable = diagnostics["usable"]
    return {
        "cases": len(valid),
        "selected_diagnostics": dict(sorted(diagnostics.items())),
        "selected_methods": dict(sorted(methods.items())),
        "usable_cases": usable,
        "usable_rate": usable / len(valid) if valid else None,
    }


def run(arguments: argparse.Namespace) -> int:
    root = arguments.shard_root.resolve()
    output = arguments.output_directory.resolve()
    if not root.is_dir():
        raise ValueError(f"shard root does not exist: {root}")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")

    shards = sorted(path for path in root.glob("shard_*") if path.is_dir())
    if not shards:
        raise ValueError(f"no shard directories found below: {root}")
    case_rows: list[dict[str, str]] = []
    reference_rows: list[dict[str, str]] = []
    shard_reports: list[dict[str, Any]] = []
    for shard in shards:
        report_path = shard / "alignment_audit_report.json"
        case_path = shard / "case_alignment_audit.csv"
        reference_path = shard / "reference_alignment_audit.csv"
        missing = [str(path) for path in (report_path, case_path, reference_path) if not path.is_file()]
        if missing:
            raise ValueError("incomplete shard output: " + ", ".join(missing))
        shard_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        case_rows.extend(_read_csv(case_path))
        reference_rows.extend(_read_csv(reference_path))

    if len({row.get("case") for row in case_rows}) != len(case_rows):
        raise ValueError("duplicate case names found across shards")
    case_rows.sort(key=lambda row: row.get("case", ""))
    reference_rows.sort(key=lambda row: (row.get("case", ""), row.get("mode", ""), row.get("reference_id", "")))
    valid = [row for row in case_rows if not row.get("error")]
    transition = Counter(
        f"{row.get('orb_diagnostic', '')} -> {row.get('fallback_diagnostic', '')}"
        for row in valid
    )
    failed_cases = [
        {"case": row.get("case", ""), "error": row.get("error", "")}
        for row in case_rows if row.get("error")
    ]
    report = {
        "report_kind": "issue8_superpoint_lightglue_alignment_audit_merged",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Local alignment-only audit. A usable case means that the selected "
            "reference passed the unchanged trusted-alignment gates. This does "
            "not measure anomaly accuracy, recall, precision, or deployment readiness."
        ),
        "shard_root": str(root),
        "merged_shards": [str(path) for path in shards],
        "requested_cases": len(case_rows),
        "failed_cases": len(failed_cases),
        "failed_case_details": failed_cases,
        "orb_only": _summary(case_rows, "orb"),
        "with_superpoint_lightglue": _summary(case_rows, "fallback"),
        "comparison": {
            "diagnostic_transition_cases": dict(sorted(transition.items())),
            "rescued_to_usable_cases": sum(_int(row, "rescued_to_usable") for row in valid),
            "regressed_from_usable_cases": sum(_int(row, "regressed_from_usable") for row in valid),
            "selected_reference_changed_cases": sum(_int(row, "selected_reference_changed") for row in valid),
            "selected_model_alignment_cases": sum(
                row.get("fallback_method") == "superpoint_lightglue" for row in valid
            ),
            "fallback_attempted_reference_attempts": sum(
                _int(row, "fallback_attempted") for row in reference_rows
                if row.get("mode") == "fallback"
            ),
        },
    }
    output.mkdir(parents=True)
    (output / "alignment_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "case_alignment_audit.csv", case_rows)
    _write_csv(output / "reference_alignment_audit.csv", reference_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    try:
        raise SystemExit(run(_arguments()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
