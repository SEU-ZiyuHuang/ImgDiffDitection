#!/usr/bin/env python3
"""Diagnose P-1 rows classified as ``unreliable`` without reading source images.

The script consumes only the v2 ``p1_cases.csv`` report.  It writes a Markdown
summary, a machine-readable JSON summary, and a per-case CSV sorted by the
number of diagnostic reasons.  Reason counts overlap by design: a single case
can have weak ECC and poor spatial coverage at the same time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


METRIC_COLUMNS = (
    "feature_match_count",
    "inlier_count",
    "inlier_rate",
    "reprojection_error_pixels",
    "spatial_coverage",
    "center_displacement_relative_diagonal",
    "corner_displacement_median_pixels",
    "projected_area_ratio",
    "valid_overlap_ratio",
    "ecc_correlation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="v2 p1_cases.csv path")
    parser.add_argument("--output", required=True, type=Path, help="local output directory")
    parser.add_argument(
        "--report",
        type=Path,
        help="optional p1_characterization_report.json; the policy is copied into the JSON diagnosis",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("case report contains no rows")
    required = {
        "case",
        "case_type",
        "component_categories",
        "status",
        "alignment_diagnostic",
        "alignment_diagnostic_reasons",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"case report is not P-1 v2; missing columns: {', '.join(sorted(missing))}")
    return rows


def parse_reasons(row: dict[str, str]) -> list[str]:
    return [reason.strip() for reason in row["alignment_diagnostic_reasons"].split(";") if reason.strip()]


def reason_family(reason: str) -> str:
    if reason.startswith("ECC correlation"):
        return "ecc_quality"
    if "spatial coverage" in reason:
        return "spatial_support"
    if reason.startswith("feature matches") or reason.startswith("inliers below") or reason.startswith("inlier rate"):
        return "match_support"
    if "reprojection error" in reason:
        return "local_fit_error"
    if "projected area" in reason or "valid overlap" in reason:
        return "global_geometry"
    if "homography" in reason or "projected geometry" in reason:
        return "homography_availability"
    return "other"


def exclusive_pattern(reasons: list[str]) -> str:
    families = {reason_family(reason) for reason in reasons}
    if families == {"ecc_quality"}:
        return "only_low_ecc"
    if families == {"spatial_support"}:
        return "only_low_spatial_coverage"
    if families == {"ecc_quality", "spatial_support"}:
        return "low_ecc_and_low_spatial_coverage"
    if len(reasons) == 1:
        return "only_" + next(iter(families))
    return "multiple_or_other_evidence"


def finite_number(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    remainder = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * remainder


def distribution(rows: Iterable[dict[str, str]], column: str) -> dict[str, float | int | None]:
    values = sorted(value for row in rows if (value := finite_number(row, column)) is not None)
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "p05": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "p05": quantile(values, 0.05),
        "p95": quantile(values, 0.95),
        "max": values[-1],
    }


def diagnostic_counts(rows: Iterable[dict[str, str]]) -> Counter[str]:
    return Counter(row["alignment_diagnostic"] for row in rows)


def grouped_diagnostic_rates(
    rows: Iterable[dict[str, str]], group_column: str, split_categories: bool = False
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "valid":
            continue
        values = row[group_column].split("|") if split_categories else [row[group_column]]
        values = [value for value in values if value] or ["UNCLASSIFIED"]
        for value in values:
            grouped[value].append(row)

    result: list[dict[str, Any]] = []
    for value, group_rows in grouped.items():
        counts = diagnostic_counts(group_rows)
        total = len(group_rows)
        result.append(
            {
                "value": value,
                "valid_cases": total,
                "usable_cases": counts["usable"],
                "unreliable_cases": counts["unreliable"],
                "unavailable_cases": counts["unavailable"],
                "unreliable_rate": counts["unreliable"] / total if total else None,
            }
        )
    return sorted(result, key=lambda item: (-item["unreliable_cases"], -item["unreliable_rate"], item["value"]))


def pattern_metric_summary(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[exclusive_pattern(parse_reasons(row))].append(row)
    result: list[dict[str, Any]] = []
    for pattern, pattern_rows in grouped.items():
        result.append(
            {
                "pattern": pattern,
                "case_count": len(pattern_rows),
                "metrics": {metric: distribution(pattern_rows, metric) for metric in METRIC_COLUMNS},
            }
        )
    return sorted(result, key=lambda item: (-item["case_count"], item["pattern"]))


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    table_rows = list(rows)
    if not table_rows:
        return "（无）\n"
    header = "| " + " | ".join(headers) + " |\n"
    separator = "| " + " | ".join("---" for _ in headers) + " |\n"
    body = "".join("| " + " | ".join(row) + " |\n" for row in table_rows)
    return header + separator + body


def write_ranked_cases(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "reason_count",
        "reason_families",
        "exclusive_pattern",
        *rows[0].keys(),
    ]
    ranked = sorted(
        rows,
        key=lambda row: (-len(parse_reasons(row)), row["case"]),
    )
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            reasons = parse_reasons(row)
            writer.writerow(
                {
                    "reason_count": len(reasons),
                    "reason_families": "|".join(sorted({reason_family(reason) for reason in reasons})),
                    "exclusive_pattern": exclusive_pattern(reasons),
                    **row,
                }
            )


def write_markdown(
    path: Path,
    total_rows: int,
    valid_rows: list[dict[str, str]],
    unreliable_rows: list[dict[str, str]],
    reason_counts: Counter[str],
    combination_counts: Counter[str],
    pattern_counts: Counter[str],
    case_type_rates: list[dict[str, Any]],
    category_rates: list[dict[str, Any]],
    metric_comparison: dict[str, dict[str, dict[str, float | int | None]]],
    evidence_pattern_metrics: list[dict[str, Any]],
) -> None:
    valid_count = len(valid_rows)
    unreliable_count = len(unreliable_rows)
    lines = [
        "# P-1 `unreliable` 样本诊断",
        "",
        "本报告只读取 `p1_cases.csv`，不读取或导出任何源图像。失败原因可重叠，因此下列单项原因计数不能相加。",
        "",
        "## 总览",
        "",
        f"- 全部 case：{total_rows}",
        f"- 有效输入：{valid_count}",
        f"- `unreliable`：{unreliable_count}（有效输入中的 {unreliable_count / valid_count:.1%}）",
        "- 诊断类别来自 P-1 固定报告规则，不是生产判定策略。",
        "",
        "## 失败原因（允许重叠）",
        "",
        markdown_table(
            ["原因", "case 数", "占 unreliable"],
            [[reason, str(count), f"{count / unreliable_count:.1%}"] for reason, count in reason_counts.most_common()],
        ),
        "## 互斥原因组合",
        "",
        markdown_table(
            ["组合", "case 数", "占 unreliable"],
            [[pattern, str(count), f"{count / unreliable_count:.1%}"] for pattern, count in combination_counts.most_common()],
        ),
        "## 互斥证据模式",
        "",
        markdown_table(
            ["模式", "case 数", "占 unreliable"],
            [[pattern, str(count), f"{count / unreliable_count:.1%}"] for pattern, count in pattern_counts.most_common()],
        ),
        "说明：`only_low_ecc` 是只有已收敛 ECC 相关性低于 0.20 的 case；它特别值得人工抽样，因为 ECC 当前在全图（含边缘填充）上计算，且不参与最终几何变换。",
        "",
        "## 各证据模式的中位数对照",
        "",
        markdown_table(
            ["模式", "case 数", "匹配数", "内点率", "空间覆盖", "有效重叠", "ECC 相关性"],
            [
                [
                    item["pattern"],
                    str(item["case_count"]),
                    format_number(item["metrics"]["feature_match_count"]["median"]),
                    format_number(item["metrics"]["inlier_rate"]["median"]),
                    format_number(item["metrics"]["spatial_coverage"]["median"]),
                    format_number(item["metrics"]["valid_overlap_ratio"]["median"]),
                    format_number(item["metrics"]["ecc_correlation"]["median"]),
                ]
                for item in evidence_pattern_metrics
            ],
        ),
        "这张表用于区分两类现象：若只有 ECC 低而匹配/覆盖/重叠正常，优先复核 ECC 的全图计算方式；若只有空间覆盖低而内点率和重投影误差正常，说明特征集中在局部区域，需结合 ROI 大小决定 0.02 是否过严。",
        "",
        "## 按案例类型（仅有效输入）",
        "",
        markdown_table(
            ["案例类型", "有效", "unreliable", "比例", "usable", "unavailable"],
            [
                [
                    item["value"],
                    str(item["valid_cases"]),
                    str(item["unreliable_cases"]),
                    f"{item['unreliable_rate']:.1%}",
                    str(item["usable_cases"]),
                    str(item["unavailable_cases"]),
                ]
                for item in case_type_rates
            ],
        ),
        "## 组件类别前 20（按 unreliable 数；仅有效输入）",
        "",
        markdown_table(
            ["组件类别", "有效", "unreliable", "比例", "usable", "unavailable"],
            [
                [
                    item["value"],
                    str(item["valid_cases"]),
                    str(item["unreliable_cases"]),
                    f"{item['unreliable_rate']:.1%}",
                    str(item["usable_cases"]),
                    str(item["unavailable_cases"]),
                ]
                for item in category_rates[:20]
            ],
        ),
        "## 关键指标：unreliable 与 usable 对照",
        "",
        markdown_table(
            ["指标", "unreliable 中位数", "unreliable P95", "usable 中位数", "usable P95"],
            [
                [
                    metric,
                    format_number(metric_comparison[metric]["unreliable"]["median"]),
                    format_number(metric_comparison[metric]["unreliable"]["p95"]),
                    format_number(metric_comparison[metric]["usable"]["median"]),
                    format_number(metric_comparison[metric]["usable"]["p95"]),
                ]
                for metric in METRIC_COLUMNS
            ],
        ),
        "`p95` 不是阈值，只是当前分布的第 95 百分位。完整的最小值、均值、最大值和每个样本的原因见同目录 JSON 与排序 CSV。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    arguments = parse_args()
    rows = read_rows(arguments.cases)
    unreliable_rows = [row for row in rows if row["status"] == "valid" and row["alignment_diagnostic"] == "unreliable"]
    valid_rows = [row for row in rows if row["status"] == "valid"]
    if not unreliable_rows:
        raise ValueError("no valid rows are classified as unreliable")

    reason_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    for row in unreliable_rows:
        reasons = parse_reasons(row)
        reason_counts.update(reasons)
        combination_counts[" + ".join(reasons)] += 1
        pattern_counts[exclusive_pattern(reasons)] += 1

    case_type_rates = grouped_diagnostic_rates(rows, "case_type")
    category_rates = grouped_diagnostic_rates(rows, "component_categories", split_categories=True)
    usable_rows = [row for row in rows if row["status"] == "valid" and row["alignment_diagnostic"] == "usable"]
    metric_comparison = {
        metric: {
            "unreliable": distribution(unreliable_rows, metric),
            "usable": distribution(usable_rows, metric),
        }
        for metric in METRIC_COLUMNS
    }
    evidence_pattern_metrics = pattern_metric_summary(unreliable_rows)

    report_policy: dict[str, Any] | None = None
    report_path = arguments.report or arguments.cases.with_name("p1_characterization_report.json")
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as input_file:
            report_policy = json.load(input_file).get("alignment_diagnostic_policy")

    arguments.output.mkdir(parents=True, exist_ok=True)
    diagnosis = {
        "input": {"case_report": str(arguments.cases), "report": str(report_path) if report_path.is_file() else None},
        "scope": "local report analysis only; no source images were read",
        "counts": {
            "all_cases": len(rows),
            "valid_cases": len(valid_rows),
            "unreliable_cases": len(unreliable_rows),
            "unreliable_rate_among_valid": len(unreliable_rows) / len(valid_rows),
        },
        "alignment_diagnostic_policy": report_policy,
        "overlapping_reason_counts": dict(reason_counts.most_common()),
        "exclusive_reason_combinations": dict(combination_counts.most_common()),
        "exclusive_evidence_patterns": dict(pattern_counts.most_common()),
        "evidence_pattern_metrics": evidence_pattern_metrics,
        "case_type_rates": case_type_rates,
        "component_category_rates": category_rates,
        "metric_comparison": metric_comparison,
    }
    with (arguments.output / "unreliable_diagnosis.json").open("w", encoding="utf-8") as output_file:
        json.dump(diagnosis, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    write_ranked_cases(arguments.output / "unreliable_cases_ranked.csv", unreliable_rows)
    write_markdown(
        arguments.output / "unreliable_diagnosis.md",
        len(rows),
        valid_rows,
        unreliable_rows,
        reason_counts,
        combination_counts,
        pattern_counts,
        case_type_rates,
        category_rates,
        metric_comparison,
        evidence_pattern_metrics,
    )
    print(f"diagnosed {len(unreliable_rows)} unreliable cases")
    print(f"markdown: {(arguments.output / 'unreliable_diagnosis.md')}")
    print(f"json: {(arguments.output / 'unreliable_diagnosis.json')}")
    print(f"ranked cases: {(arguments.output / 'unreliable_cases_ranked.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
