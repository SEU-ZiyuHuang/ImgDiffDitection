#!/usr/bin/env python3
"""Create a local-only analytical report from calibration-batch outputs.

The command reads only batch summary JSON files and per-case calibration
observations.  It never reads, copies, displays, or uploads the source images.
Its reports describe observations for calibration; they do not emit normal,
change, or production-readiness conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ARTIFACT_KEYS = (
    "alignment_image",
    "valid_mask",
    "difference_mask",
    "difference_heatmap",
    "annotated_image",
)
ALIGNMENT_METRIC_KEYS = (
    "feature_match_count",
    "inlier_count",
    "inlier_rate",
    "reprojection_error_pixels",
    "spatial_coverage",
    "projected_area_ratio",
    "valid_overlap_ratio",
    "ecc_correlation",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local calibration observations without reading source images "
            "or producing business conclusions."
        )
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        required=True,
        help="completed calibration batch directory containing shard summaries",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="directory for generated JSON, Markdown, and CSV reports",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return content


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _distribution(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = sorted(
        number for value in values if (number := _finite_number(value)) is not None
    )
    if not numbers:
        return {"count": 0, "minimum": None, "p05": None, "median": None,
                "p95": None, "maximum": None, "mean": None}
    return {
        "count": len(numbers),
        "minimum": numbers[0],
        "p05": _quantile(numbers, 0.05),
        "median": _quantile(numbers, 0.50),
        "p95": _quantile(numbers, 0.95),
        "maximum": numbers[-1],
        "mean": fmean(numbers),
    }


def _alignment_distributions(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    records = list(records)
    return {
        key: _distribution(record.get(key) for record in records)
        for key in ALIGNMENT_METRIC_KEYS
    }


def _case_type(case_name: str) -> str:
    return case_name.rsplit("_", 1)[1] if "_" in case_name else "unspecified"


def _artifact_status(artifacts: Any) -> tuple[bool, int, list[str]]:
    if not isinstance(artifacts, dict):
        return False, 0, list(ARTIFACT_KEYS)
    missing = []
    present = 0
    for key in ARTIFACT_KEYS:
        value = artifacts.get(key)
        if isinstance(value, str) and Path(value).is_file():
            present += 1
        else:
            missing.append(key)
    return present == len(ARTIFACT_KEYS), present, missing


def _error_category(error: str) -> str:
    if "cannot decode live image" in error:
        return "cannot_decode_live_image"
    if "cannot decode any reference image" in error:
        return "cannot_decode_reference_image"
    if "invalid ROI" in error:
        return "invalid_roi"
    return "other_runtime_or_input_error"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _markdown_distribution(name: str, distribution: dict[str, Any]) -> str:
    if distribution["count"] == 0:
        return f"- {name}：没有有效数值。"
    return (
        f"- {name}：样本数 {distribution['count']}，最小值 {distribution['minimum']:.4g}，"
        f"5% 分位 {distribution['p05']:.4g}，中位数 {distribution['median']:.4g}，"
        f"95% 分位 {distribution['p95']:.4g}，最大值 {distribution['maximum']:.4g}。"
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    batch = report["batch"]
    references = report["references"]
    components = report["components"]
    candidates = report["difference_candidates"]
    evidence = report["evidence"]
    lines = [
        "# 标定批次详细分析",
        "",
        "本报告只汇总本地标定观察与证据完整性；不输出正常、异动或上线可用性结论。",
        "",
        "## 批次覆盖",
        "",
        f"- 请求案例：{batch['requested_cases']}；完成：{batch['completed_cases']}；失败：{batch['failed_cases']}。",
        f"- 找到观察结果：{batch['observation_files_found']}；缺少观察结果的已完成案例：{batch['completed_without_observation']}。",
        "",
        "## 参考图选择",
        "",
        f"- 多参考图案例：{references['multi_reference_cases']}；选中新增参考图的案例：{references['additional_selected_cases']}。",
        f"- 参考图配准诊断分布：{json.dumps(references['attempt_diagnostics'], ensure_ascii=False)}。",
        f"- 被选中参考图的配准诊断分布：{json.dumps(references['selected_diagnostics'], ensure_ascii=False)}。",
        "",
        "## 配准观察指标（被选中的参考图）",
        "",
    ]
    for name, distribution in report["alignment"]["selected_reference_metrics"].items():
        lines.append(_markdown_distribution(name, distribution))
    lines.extend([
        "",
        "## 部件与差异候选",
        "",
        f"- 已观察部件：{components['observed_components']}；存在差异候选的部件：{components['components_with_candidates']}。",
        f"- 差异候选区域总数：{candidates['regions']}；候选区域面积统计见 JSON 和 CSV 明细。",
        f"- 完整证据部件：{evidence['complete_components']}；证据不完整部件：{evidence['incomplete_components']}。",
        "",
        "## 失败案例",
        "",
    ])
    if report["failures"]["records"]:
        for record in report["failures"]["records"]:
            lines.append(f"- {record['case']}：{record['error']}")
    else:
        lines.append("- 无。")
    lines.extend([
        "",
        "## 使用边界",
        "",
        "- 分位数仅描述当前观察分布，不自动构成正式门槛。",
        "- 正式门槛仍需基于已确认正常样本、保留验证样本与受控压力测试确定。",
        "- 当前差异候选数量不是异动数量，也不能用于计算异动召回率或漏检率。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(batch_output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load batch observations and return report plus detail rows."""
    batch_output = batch_output.resolve()
    summary_paths = sorted(batch_output.glob("batch_summary_shard_*.json"))
    if not summary_paths:
        summary_paths = sorted(batch_output.glob("batch_summary.json"))
    if not summary_paths:
        raise ValueError(f"no batch summary found in {batch_output}")

    summaries = [_read_json(path) for path in summary_paths]
    requested_cases = sum(int(summary.get("case_count_requested", 0)) for summary in summaries)
    case_records = [
        record
        for summary in summaries
        for record in summary.get("cases", [])
        if isinstance(record, dict)
    ]

    completed_records = [record for record in case_records if record.get("status") == "completed"]
    failed_records = [record for record in case_records if record.get("status") == "failed"]
    failures = []
    for record in failed_records:
        error = str(record.get("error", "unspecified error"))
        failures.append({
            "case": str(record.get("case", "unknown")),
            "error": error,
            "error_category": _error_category(error),
        })

    case_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    selected_attempts: list[dict[str, Any]] = []
    category_components: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    type_components: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    attempt_diagnostics: Counter[str] = Counter()
    selected_diagnostics: Counter[str] = Counter()
    evidence_missing: Counter[str] = Counter()
    additional_selected_cases = 0
    multi_reference_cases = 0
    completed_without_observation = 0

    for record in completed_records:
        case_name = str(record.get("case", "unknown"))
        case_type = _case_type(case_name)
        manifest_value = record.get("manifest")
        manifest_path = Path(manifest_value) if isinstance(manifest_value, str) else None
        base_row = {
            "case": case_name,
            "case_type": case_type,
            "status": "completed",
            "reference_count": record.get("reference_count"),
            "component_count": record.get("component_count"),
            "selected_reference_id": record.get("selected_reference_id"),
            "manifest": str(manifest_path) if manifest_path else "",
        }
        if manifest_path is None or not manifest_path.is_file():
            completed_without_observation += 1
            base_row["observation_available"] = False
            case_rows.append(base_row)
            continue

        observation = _read_json(manifest_path)
        reference_attempts = observation.get("reference_attempts", [])
        components = observation.get("components", [])
        base_row.update({
            "observation_available": True,
            "reference_attempt_count": len(reference_attempts) if isinstance(reference_attempts, list) else 0,
            "observed_component_count": len(components) if isinstance(components, list) else 0,
        })
        if isinstance(reference_attempts, list) and len(reference_attempts) > 1:
            multi_reference_cases += 1
        selected_reference = observation.get("selected_reference", {})
        selected_reference_id = (
            selected_reference.get("id") if isinstance(selected_reference, dict) else None
        )
        if isinstance(selected_reference_id, str) and selected_reference_id != "primary":
            additional_selected_cases += 1

        if isinstance(reference_attempts, list):
            for attempt in reference_attempts:
                if not isinstance(attempt, dict):
                    continue
                metrics = attempt.get("alignment_metrics", {})
                diagnostic = str(attempt.get("alignment_diagnostic", "missing"))
                attempt_diagnostics[diagnostic] += 1
                attempt_row = {
                    "case": case_name,
                    "case_type": case_type,
                    "reference_id": attempt.get("reference_id"),
                    "selected": bool(attempt.get("selected")),
                    "diagnostic": diagnostic,
                    **(metrics if isinstance(metrics, dict) else {}),
                }
                attempts.append(attempt_row)
                if bool(attempt.get("selected")):
                    selected_attempts.append(attempt_row)
                    selected_diagnostics[diagnostic] += 1

        if not isinstance(components, list):
            case_rows.append(base_row)
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            artifacts_complete, artifacts_present, missing_artifacts = _artifact_status(
                component.get("artifacts")
            )
            for missing in missing_artifacts:
                evidence_missing[missing] += 1
            metrics = component.get("alignment_metrics", {})
            regions = component.get("difference_regions", [])
            if not isinstance(regions, list):
                regions = []
            category = str(component.get("category", "unknown"))
            component_row = {
                "case": case_name,
                "case_type": case_type,
                "component_index": component.get("component_index"),
                "category": category,
                "diagnostic": metrics.get("diagnostic") if isinstance(metrics, dict) else None,
                "difference_candidate_count": component.get("difference_candidate_count", 0),
                "candidate_region_records": len(regions),
                "evidence_complete": artifacts_complete,
                "evidence_files_present": artifacts_present,
                "missing_artifacts": "|".join(missing_artifacts),
                **(metrics if isinstance(metrics, dict) else {}),
            }
            component_rows.append(component_row)
            category_components[category].append(component_row)
            type_components[case_type].append(component_row)
            for region_index, region in enumerate(regions):
                if not isinstance(region, dict):
                    continue
                width = _finite_number(region.get("width"))
                height = _finite_number(region.get("height"))
                region_rows.append({
                    "case": case_name,
                    "case_type": case_type,
                    "component_index": component.get("component_index"),
                    "category": category,
                    "region_index": region_index,
                    "width": width,
                    "height": height,
                    "area_pixels": width * height if width is not None and height is not None else None,
                    "confidence": region.get("confidence"),
                    "evidence_channels": region.get("evidence_channels", []),
                })
        case_rows.append(base_row)

    for failure in failures:
        case_rows.append({
            "case": failure["case"],
            "case_type": _case_type(failure["case"]),
            "status": "failed",
            "error": failure["error"],
            "error_category": failure["error_category"],
        })

    def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "components": len(rows),
            "components_with_candidates": sum(
                _finite_number(row.get("difference_candidate_count")) not in (None, 0.0)
                for row in rows
            ),
            "candidate_count": _distribution(row.get("difference_candidate_count") for row in rows),
            "diagnostics": dict(Counter(str(row.get("diagnostic", "missing")) for row in rows)),
            "alignment_metrics": _alignment_distributions(rows),
        }

    candidate_channels: Counter[str] = Counter()
    for row in region_rows:
        channels = row["evidence_channels"]
        if isinstance(channels, list):
            candidate_channels.update(str(channel) for channel in channels)

    report = {
        "report_kind": "calibration_observation_analysis",
        "generated_at_utc": _utc_now(),
        "batch_output": str(batch_output),
        "batch": {
            "requested_cases": requested_cases,
            "completed_cases": len(completed_records),
            "failed_cases": len(failed_records),
            "observation_files_found": len(completed_records) - completed_without_observation,
            "completed_without_observation": completed_without_observation,
            "summary_files": [str(path) for path in summary_paths],
        },
        "failures": {
            "by_category": dict(Counter(item["error_category"] for item in failures)),
            "records": failures,
        },
        "references": {
            "reference_attempts": len(attempts),
            "multi_reference_cases": multi_reference_cases,
            "additional_selected_cases": additional_selected_cases,
            "attempt_diagnostics": dict(attempt_diagnostics),
            "selected_diagnostics": dict(selected_diagnostics),
        },
        "alignment": {
            "all_reference_metrics": _alignment_distributions(attempts),
            "selected_reference_metrics": _alignment_distributions(selected_attempts),
            "component_metrics": _alignment_distributions(component_rows),
        },
        "components": {
            "observed_components": len(component_rows),
            "components_with_candidates": sum(
                _finite_number(row.get("difference_candidate_count")) not in (None, 0.0)
                for row in component_rows
            ),
            "candidate_count": _distribution(
                row.get("difference_candidate_count") for row in component_rows
            ),
            "by_category": {
                name: group_summary(rows) for name, rows in sorted(category_components.items())
            },
            "by_case_type": {
                name: group_summary(rows) for name, rows in sorted(type_components.items())
            },
        },
        "difference_candidates": {
            "regions": len(region_rows),
            "region_area_pixels": _distribution(row.get("area_pixels") for row in region_rows),
            "region_confidence": _distribution(row.get("confidence") for row in region_rows),
            "evidence_channels": dict(candidate_channels),
        },
        "evidence": {
            "complete_components": sum(bool(row["evidence_complete"]) for row in component_rows),
            "incomplete_components": sum(not bool(row["evidence_complete"]) for row in component_rows),
            "missing_by_artifact": dict(evidence_missing),
        },
        "interpretation_limits": [
            "This report summarizes calibration observations, not business conclusions.",
            "Difference candidates are not labeled anomalies and must not be used as anomaly counts.",
            "Distribution percentiles are inputs to controlled calibration, not automatic production thresholds.",
        ],
    }
    return report, case_rows, component_rows, region_rows


def run(arguments: argparse.Namespace) -> int:
    batch_output = arguments.batch_output.resolve()
    output_directory = arguments.output_directory.resolve()
    if not batch_output.is_dir():
        raise ValueError(f"batch output directory does not exist: {batch_output}")
    output_directory.mkdir(parents=True, exist_ok=True)

    report, case_rows, component_rows, region_rows = analyze(batch_output)
    _write_json(output_directory / "calibration_analysis.json", report)
    _write_csv(output_directory / "case_analysis.csv", case_rows)
    _write_csv(output_directory / "component_observations.csv", component_rows)
    _write_csv(output_directory / "candidate_regions.csv", region_rows)
    _write_markdown(output_directory / "calibration_analysis.md", report)
    print(json.dumps({
        "output_directory": str(output_directory),
        "completed_cases": report["batch"]["completed_cases"],
        "failed_cases": report["batch"]["failed_cases"],
        "observed_components": report["components"]["observed_components"],
    }, ensure_ascii=False))
    return 0


def main() -> int:
    try:
        return run(parse_arguments())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
