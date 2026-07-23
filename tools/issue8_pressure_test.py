#!/usr/bin/env python3
"""Run controlled Issue 8 synthetic cases without reading local case images.

The report records code paths and evidence counts.  It is not an accuracy,
recall, false-positive, or missed-detection claim for unlabelled local data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imagecmp_py"))

from imagecmp import ComparisonState, ImageComparisonService  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "imagecmp_py" / "configs" / "calibrated-20260723-issue8-v1.json",
    )
    return parser.parse_args()


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def _scene() -> np.ndarray:
    random = np.random.RandomState(20260728)
    image = random.randint(25, 210, (360, 480, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.rectangle(image, (95, 72), (385, 285), (85, 95, 115), -1)
    cv2.rectangle(image, (108, 84), (372, 273), (185, 185, 185), 3)
    cv2.circle(image, (165, 140), 18, (230, 230, 230), -1)
    cv2.circle(image, (315, 140), 18, (230, 230, 230), -1)
    cv2.circle(image, (165, 230), 18, (230, 230, 230), -1)
    cv2.circle(image, (315, 230), 18, (230, 230, 230), -1)
    cv2.putText(image, "I8", (215, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
    for value in range(30, 480, 45):
        cv2.line(image, (value, 0), (value, 359), (100, 100, 100), 1)
    return image


def _gamma(image: np.ndarray, exponent: float) -> np.ndarray:
    table = np.rint((np.arange(256, dtype=np.float32) / 255.0) ** exponent * 255.0)
    return cv2.LUT(image, table.astype(np.uint8))


def _cases(standard: np.ndarray) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {
        "unchanged": standard.copy(),
        "illumination": _gamma(standard, 1.6),
        "blur": cv2.GaussianBlur(standard, (9, 9), 1.8),
        "background_only": standard.copy(),
        "local_colour_change": standard.copy(),
        "oil_film": standard.copy(),
        "bolt_missing": standard.copy(),
        "crack": standard.copy(),
        "untrusted_alignment": np.full_like(standard, 127),
    }
    cv2.rectangle(variants["background_only"], (5, 5), (65, 65), (0, 0, 255), -1)
    cv2.rectangle(variants["local_colour_change"], (225, 160), (275, 215), (0, 255, 0), -1)
    overlay = variants["oil_film"].copy()
    cv2.ellipse(overlay, (240, 205), (72, 32), 0, 0, 360, (35, 90, 145), -1)
    variants["oil_film"] = cv2.addWeighted(overlay, 0.58, variants["oil_film"], 0.42, 0)
    cv2.circle(variants["bolt_missing"], (315, 230), 19, (85, 95, 115), -1)
    cv2.line(variants["crack"], (205, 128), (282, 238), (10, 10, 10), 3, cv2.LINE_AA)
    return variants


def _fallback_config(source: Path, output: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["version"] = "calibrated-20260723-issue8-fallback-evaluation-v1"
    payload["alignment"]["orb_feature_count"] = 4
    payload["alignment"]["superpoint_lightglue_fallback_enabled"] = True
    destination = output / "fallback-evaluation-config.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def _row(name: str, result) -> dict:
    metrics = result.alignment_metrics
    return {
        "case": name,
        "state": result.state.value,
        "alignment_diagnostic": metrics.get("diagnostic"),
        "alignment_method": metrics.get("alignment_correspondence_method"),
        "fallback_attempted": metrics.get("alignment_fallback_attempted", 0),
        "raw_candidate_count": metrics.get("raw_candidate_count", 0),
        "decision_candidate_count": metrics.get("decision_candidate_count", 0),
        "small_candidate_count": metrics.get("small_candidate_count", 0),
        "colour_weight": metrics.get("colour_weight"),
        "unavailable_reason": result.unavailable_reason.value if result.unavailable_reason else None,
    }


def main() -> int:
    arguments = _arguments()
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = arguments.config.resolve()
    standard = _scene()
    variants = _cases(standard)
    standard_path = output / "standard.png"
    _write(standard_path, standard)
    service = ImageComparisonService()
    rows: list[dict] = []
    passed = True
    expected_normal = {"unchanged", "illumination", "background_only"}
    expected_change = {"local_colour_change", "oil_film", "bolt_missing", "crack"}
    expected_unavailable = {"blur", "untrusted_alignment"}
    for name, live in variants.items():
        live_path = output / f"{name}_live.png"
        _write(live_path, live)
        result = service.compare(
            standard_path=standard_path,
            live_path=live_path,
            roi="equipment 0.5 0.5 0.62 0.60",
            output_dir=output / name,
            config_path=config,
        )
        rows.append(_row(name, result))
        if name in expected_normal and result.state != ComparisonState.NO_CHANGE_HIGH_CONFIDENCE:
            passed = False
        if name in expected_change and result.state != ComparisonState.CHANGE_DETECTED:
            passed = False
        if name in expected_unavailable and result.state != ComparisonState.DETECTION_UNAVAILABLE:
            passed = False

    rotation = cv2.getRotationMatrix2D((240, 180), 11.0, 1.0)
    fallback_live = cv2.warpAffine(standard, rotation, (480, 360), borderMode=cv2.BORDER_REFLECT)
    fallback_live_path = output / "fallback_live.png"
    _write(fallback_live_path, fallback_live)
    fallback_result = service.compare(
        standard_path=standard_path,
        live_path=fallback_live_path,
        roi="equipment 0.5 0.5 0.62 0.60",
        output_dir=output / "superpoint_lightglue_fallback",
        config_path=_fallback_config(config, output),
    )
    rows.append(_row("superpoint_lightglue_fallback", fallback_result))
    if (fallback_result.state == ComparisonState.DETECTION_UNAVAILABLE
            or fallback_result.alignment_metrics.get("alignment_correspondence_method")
            != "superpoint_lightglue"):
        passed = False

    report = {
        "config": str(config),
        "passed": passed,
        "note": (
            "全部图片由程序生成。结果只验证既定合成场景的链路、门禁和证据，"
            "不对未标注本地图片作准确率或漏检率声明。"
        ),
        "results": rows,
    }
    report_path = output / "issue8_pressure_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['case']:30} {row['state']:28} "
            f"raw={row['raw_candidate_count']} decision={row['decision_candidate_count']} "
            f"method={row['alignment_method']}"
        )
    print(f"report: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
