#!/usr/bin/env python3
"""Compare two local-only calibration analyses without interpreting anomalies.

The command compares a pre-#6 batch with a trusted-alignment batch.  It only
uses aggregate analysis files and component-observation CSVs already written
on the local machine; source images and evidence pixels are never read or
copied.  Its output describes evidence-gate behaviour, not detection accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two local image-comparison calibration analyses."
    )
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--current-analysis", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise ValueError(f"cannot read CSV {path}: {exc}") from exc


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter((row.get(key) or "missing") for row in rows))


def _is_suspect_signature(row: dict[str, str]) -> bool:
    center = _number(row.get("center_displacement_relative_diagonal"))
    area_ratio = _number(row.get("projected_area_ratio"))
    coverage = _number(row.get("spatial_coverage"))
    ecc_converged = _number(row.get("ecc_converged"))
    return bool(
        center is not None and center <= 0.05
        and area_ratio is not None and abs(area_ratio - 1.0) <= 0.15
        and coverage is not None and coverage < 0.20
        and ecc_converged == 0.0
    )


def _signature_summary(rows: list[dict[str, str]]) -> dict[str, int | float | None]:
    usable = [row for row in rows if row.get("diagnostic") == "usable"]
    signatures = [row for row in usable if _is_suspect_signature(row)]
    all_signatures = [row for row in rows if _is_suspect_signature(row)]
    return {
        "usable_components": len(usable),
        "suspect_usable_components": len(signatures),
        "suspect_usable_rate": _rate(len(signatures), len(usable)),
        "all_signature_components": len(all_signatures),
    }


def _mapping_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    recorded = [row for row in rows if row.get("component_mapping_usable") not in (None, "")]
    usable = [row for row in recorded if _number(row.get("component_mapping_usable")) == 1.0]
    rejected = [row for row in recorded if _number(row.get("component_mapping_usable")) == 0.0]
    global_usable_rejected = [
        row for row in rejected if row.get("diagnostic") == "usable"
    ]
    return {
        "mapping_recorded_components": len(recorded),
        "mapping_usable_components": len(usable),
        "mapping_rejected_components": len(rejected),
        "mapping_usable_rate": _rate(len(usable), len(recorded)),
        "global_alignment_usable_but_mapping_rejected": len(global_usable_rejected),
        "rejection_reasons": _counts(rejected, "component_mapping_failure_reason"),
    }


def _distribution_value(analysis: dict[str, Any], metric: str) -> dict[str, Any]:
    alignment = analysis.get("alignment", {})
    component_metrics = alignment.get("component_metrics", {}) if isinstance(alignment, dict) else {}
    value = component_metrics.get(metric, {}) if isinstance(component_metrics, dict) else {}
    return value if isinstance(value, dict) else {}


def _batch_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    batch = analysis.get("batch", {})
    references = analysis.get("references", {})
    components = analysis.get("components", {})
    candidates = analysis.get("difference_candidates", {})
    return {
        "requested_cases": batch.get("requested_cases"),
        "completed_cases": batch.get("completed_cases"),
        "failed_cases": batch.get("failed_cases"),
        "observed_components": components.get("observed_components"),
        "selected_reference_diagnostics": references.get("selected_diagnostics", {}),
        "additional_reference_selected_cases": references.get("additional_selected_cases"),
        "components_with_candidates": components.get("components_with_candidates"),
        "candidate_regions": candidates.get("regions"),
    }


def compare(baseline_directory: Path, current_directory: Path) -> dict[str, Any]:
    baseline_analysis = _read_json(baseline_directory / "calibration_analysis.json")
    current_analysis = _read_json(current_directory / "calibration_analysis.json")
    baseline_rows = _read_csv(baseline_directory / "component_observations.csv")
    current_rows = _read_csv(current_directory / "component_observations.csv")

    metric_names = (
        "spatial_coverage",
        "ecc_correlation",
        "valid_overlap_ratio",
        "roi_valid_overlap_ratio",
        "appearance_ncc",
        "appearance_ssim",
        "effective_resolution_scale",
    )
    return {
        "report_kind": "issue6_trusted_alignment_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "suspect_false_usable_signature": (
                "diagnostic == usable, near-identity homography, spatial_coverage < 0.20, "
                "and ecc_converged == 0"
            ),
            "scope": (
                "Calibration observations only.  Neither candidate counts nor gate rates are "
                "anomaly accuracy, recall, precision, or production-readiness metrics."
            ),
        },
        "baseline": {
            "analysis_directory": str(baseline_directory),
            "batch": _batch_summary(baseline_analysis),
            "component_diagnostics": _counts(baseline_rows, "diagnostic"),
            "ecc_converged": _counts(baseline_rows, "ecc_converged"),
            "suspect_false_usable_signature": _signature_summary(baseline_rows),
        },
        "current": {
            "analysis_directory": str(current_directory),
            "batch": _batch_summary(current_analysis),
            "component_diagnostics": _counts(current_rows, "diagnostic"),
            "ecc_converged": _counts(current_rows, "ecc_converged"),
            "suspect_false_usable_signature": _signature_summary(current_rows),
            "component_mapping": _mapping_summary(current_rows),
        },
        "component_metric_distributions": {
            metric: {
                "baseline": _distribution_value(baseline_analysis, metric),
                "current": _distribution_value(current_analysis, metric),
            }
            for metric in metric_names
        },
    }


def _percent(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "无分母"


def _markdown_distribution(name: str, baseline: dict[str, Any], current: dict[str, Any]) -> str:
    return (
        f"| {name} | {baseline.get('count', 0)} | {baseline.get('median', '—')} | "
        f"{current.get('count', 0)} | {current.get('median', '—')} |"
    )


def write_report(output_directory: Path, report: dict[str, Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "issue6_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    baseline = report["baseline"]
    current = report["current"]
    mapping = current["component_mapping"]
    baseline_signature = baseline["suspect_false_usable_signature"]
    current_signature = current["suspect_false_usable_signature"]
    lines = [
        "# 阶段 5 与阶段 6 本地标定对比",
        "",
        "本报告仅比较本地标定观察和证据闸门行为；不输出正常/异动业务结论，"
        "也不构成准确率、召回率、漏检率或上线验收结论。",
        "",
        "## 批次覆盖",
        "",
        "| 指标 | 阶段 5 | 阶段 6 |",
        "| --- | ---: | ---: |",
        f"| 请求案例 | {baseline['batch']['requested_cases']} | {current['batch']['requested_cases']} |",
        f"| 完成案例 | {baseline['batch']['completed_cases']} | {current['batch']['completed_cases']} |",
        f"| 失败案例 | {baseline['batch']['failed_cases']} | {current['batch']['failed_cases']} |",
        f"| 观察部件 | {baseline['batch']['observed_components']} | {current['batch']['observed_components']} |",
        f"| 选中新增参考图案例 | {baseline['batch']['additional_reference_selected_cases']} | {current['batch']['additional_reference_selected_cases']} |",
        "",
        "## 全图对齐诊断与伪可用签名",
        "",
        f"- 阶段 5 组件对齐诊断：{json.dumps(baseline['component_diagnostics'], ensure_ascii=False)}。",
        f"- 阶段 6 组件对齐诊断：{json.dumps(current['component_diagnostics'], ensure_ascii=False)}。",
        f"- 伪可用签名（近似恒等 H、低覆盖、ECC 未收敛）：阶段 5 为 "
        f"{baseline_signature['suspect_usable_components']}/"
        f"{baseline_signature['usable_components']}（{_percent(baseline_signature['suspect_usable_rate'])}）；"
        f"阶段 6 为 {current_signature['suspect_usable_components']}/"
        f"{current_signature['usable_components']}（{_percent(current_signature['suspect_usable_rate'])}）。",
        f"- 阶段 6 仍观察到该签名 {current_signature['all_signature_components']} 个，"
        "但只有仍被标为全图 usable 的才计入伪可用率。",
        "",
        "## 阶段 6 新增部件映射闸门",
        "",
        f"- 记录映射证据的部件：{mapping['mapping_recorded_components']}；"
        f"通过：{mapping['mapping_usable_components']}；拒绝：{mapping['mapping_rejected_components']}；"
        f"通过率：{_percent(mapping['mapping_usable_rate'])}。",
        f"- 全图对齐仍为 usable、但被部件映射闸门阻断："
        f"{mapping['global_alignment_usable_but_mapping_rejected']}。",
        f"- 映射拒绝原因：{json.dumps(mapping['rejection_reasons'], ensure_ascii=False)}。",
        "",
        "## 指标分布（部件观察）",
        "",
        "| 指标 | 阶段 5 样本数 | 阶段 5 中位数 | 阶段 6 样本数 | 阶段 6 中位数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in report["component_metric_distributions"].items():
        lines.append(_markdown_distribution(name, item["baseline"], item["current"]))
    lines.extend([
        "",
        "## 差异候选观察",
        "",
        f"- 阶段 5：{baseline['batch']['components_with_candidates']} 个部件有候选，"
        f"共 {baseline['batch']['candidate_regions']} 个候选区域。",
        f"- 阶段 6：{current['batch']['components_with_candidates']} 个部件有候选，"
        f"共 {current['batch']['candidate_regions']} 个候选区域。",
        "- 该变化受新的证据闸门影响：映射不可信的部件不会进入差异计算，"
        "因此候选数量不是异动数量，也不能直接比较为检出能力变化。",
        "",
    ])
    (output_directory / "issue6_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        arguments = parse_arguments()
        baseline = arguments.baseline_analysis.resolve()
        current = arguments.current_analysis.resolve()
        if not baseline.is_dir() or not current.is_dir():
            raise ValueError("both analysis paths must be existing directories")
        report = compare(baseline, current)
        write_report(arguments.output_directory.resolve(), report)
        print(json.dumps({
            "output_directory": str(arguments.output_directory.resolve()),
            "baseline_components": report["baseline"]["batch"]["observed_components"],
            "current_components": report["current"]["batch"]["observed_components"],
        }, ensure_ascii=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
