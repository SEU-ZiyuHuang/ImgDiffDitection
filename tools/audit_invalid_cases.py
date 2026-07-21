#!/usr/bin/env python3
"""Audit P-1 invalid cases without changing any source image or ROI file.

The script compares the original strict YOLO boundary rule with the agreed
boundary-tolerance rule. A tolerated ROI is clipped to [0, 1] only in the
report; the source coordinate file is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as error:  # pragma: no cover - reported to the operator.
    raise SystemExit(
        "Pillow is required to decode image files. Install it with: "
        "python -m pip install Pillow"
    ) from error


@dataclass
class RoiFinding:
    line: int
    raw: str
    status: str
    reason: str
    left: float | None = None
    top: float | None = None
    right: float | None = None
    bottom: float | None = None
    normalized_left: float | None = None
    normalized_top: float | None = None
    normalized_right: float | None = None
    normalized_bottom: float | None = None


@dataclass
class CaseFinding:
    case: str
    old_status: str
    old_errors: str
    image_status: str
    roi_findings: list[RoiFinding]
    verdict: str
    rationale: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-report", type=Path, required=True,
                        help="P-1 p1_cases.csv report to audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Allowed normalized-coordinate overshoot (default: 0.01)")
    parser.add_argument("--standard-name", default="标准源图.jpg")
    parser.add_argument("--live-name", default="对比截图.jpg")
    parser.add_argument("--roi-name", default="标准源图坐标.txt")
    return parser.parse_args()


def decode_image(path: Path) -> str:
    if not path.is_file():
        return f"missing: {path.name}"
    try:
        # verify catches truncated/corrupt container data; load decodes pixels.
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return "readable"
    except (OSError, UnidentifiedImageError) as error:
        return f"unreadable: {error}"


def finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def audit_roi_line(line_number: int, raw: str, tolerance: float) -> RoiFinding | None:
    text = raw.strip()
    if not text:
        return None
    fields = text.split()
    if len(fields) != 5:
        return RoiFinding(line_number, raw, "invalid", "expected five YOLO fields")
    try:
        _, center_x, center_y, width, height = fields
        center_x, center_y, width, height = map(float, (center_x, center_y, width, height))
    except ValueError:
        return RoiFinding(line_number, raw, "invalid", "ROI contains a non-numeric value")
    if not finite(center_x, center_y, width, height):
        return RoiFinding(line_number, raw, "invalid", "ROI contains a non-finite value")
    if width <= 0.0 or height <= 0.0:
        return RoiFinding(line_number, raw, "invalid", "ROI width and height must be positive")

    left = center_x - width / 2.0
    right = center_x + width / 2.0
    top = center_y - height / 2.0
    bottom = center_y + height / 2.0
    bounds = (left, top, right, bottom)
    if all(0.0 <= value <= 1.0 for value in bounds):
        return RoiFinding(line_number, raw, "strictly_valid", "inside [0, 1]", left, top, right, bottom)
    if all(-tolerance <= value <= 1.0 + tolerance for value in bounds):
        return RoiFinding(
            line_number,
            raw,
            "tolerated_and_normalized",
            f"all boundaries are within [-{tolerance:g}, {1.0 + tolerance:g}]",
            left,
            top,
            right,
            bottom,
            max(0.0, left),
            max(0.0, top),
            min(1.0, right),
            min(1.0, bottom),
        )
    return RoiFinding(
        line_number,
        raw,
        "invalid",
        f"boundary exceeds [-{tolerance:g}, {1.0 + tolerance:g}]",
        left,
        top,
        right,
        bottom,
    )


def audit_rois(path: Path, tolerance: float) -> list[RoiFinding]:
    if not path.is_file():
        return [RoiFinding(0, "", "invalid", "missing ROI file")]
    findings: list[RoiFinding] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        finding = audit_roi_line(line_number, raw, tolerance)
        if finding is not None:
            findings.append(finding)
    if not findings:
        findings.append(RoiFinding(0, "", "invalid", "ROI file contains no ROI"))
    return findings


def invalid_report_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as report:
        yield from (row for row in csv.DictReader(report) if row["status"] == "invalid")


def audit_case(row: dict[str, str], arguments: argparse.Namespace) -> CaseFinding:
    case_path = arguments.dataset / row["case"]
    image_results = [
        decode_image(case_path / arguments.standard_name),
        decode_image(case_path / arguments.live_name),
    ]
    image_status = "; ".join(image_results)
    roi_findings = audit_rois(case_path / arguments.roi_name, arguments.tolerance)
    has_image_problem = any(result != "readable" for result in image_results)
    has_substantive_roi_problem = any(finding.status == "invalid" for finding in roi_findings)
    has_normalization = any(finding.status == "tolerated_and_normalized" for finding in roi_findings)

    if has_image_problem:
        return CaseFinding(row["case"], row["status"], row["errors"], image_status, roi_findings,
                           "remains_invalid", "at least one image cannot be fully decoded")
    if has_substantive_roi_problem:
        return CaseFinding(row["case"], row["status"], row["errors"], image_status, roi_findings,
                           "remains_invalid", "at least one ROI exceeds the agreed boundary tolerance")
    if has_normalization:
        return CaseFinding(row["case"], row["status"], row["errors"], image_status, roi_findings,
                           "recovered_by_tolerance", "all images decode and every ROI is strict or tolerated")
    return CaseFinding(row["case"], row["status"], row["errors"], image_status, roi_findings,
                       "unexpected_old_invalid", "strict ROI audit found no invalidity; inspect parser parity")


def write_csv(path: Path, findings: list[CaseFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=(
            "case", "old_errors", "verdict", "rationale", "image_status", "roi_statuses"))
        writer.writeheader()
        for finding in findings:
            writer.writerow({
                "case": finding.case,
                "old_errors": finding.old_errors,
                "verdict": finding.verdict,
                "rationale": finding.rationale,
                "image_status": finding.image_status,
                "roi_statuses": "; ".join(
                    f"line {roi.line}: {roi.status} ({roi.reason})" for roi in finding.roi_findings),
            })


def main() -> int:
    arguments = parse_arguments()
    if arguments.tolerance < 0.0:
        raise SystemExit("--tolerance must be non-negative")
    if not arguments.dataset.is_dir():
        raise SystemExit("--dataset must be an existing directory")
    if not arguments.case_report.is_file():
        raise SystemExit("--case-report must be an existing p1_cases.csv file")

    findings = [audit_case(row, arguments) for row in invalid_report_rows(arguments.case_report)]
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_invalid_cases": len(findings),
        "tolerance": arguments.tolerance,
        "verdict_counts": dict(Counter(finding.verdict for finding in findings)),
        "findings": [asdict(finding) for finding in findings],
    }
    (arguments.output / "invalid_case_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(arguments.output / "invalid_case_audit.csv", findings)
    print(json.dumps({key: summary[key] for key in ("source_invalid_cases", "tolerance", "verdict_counts")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
