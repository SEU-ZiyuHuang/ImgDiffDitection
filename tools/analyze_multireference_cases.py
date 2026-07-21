#!/usr/bin/env python3
"""Discover and score all same-case reference images for local multi-reference comparison.

The dataset convention was audited before this tool was written:

* every case is a direct child directory of the dataset root;
* ``对比截图.jpg`` is the live image;
* ``标准源图.jpg`` is the original reference;
* zero or more contiguous ``新增标准源图N.jpg`` files are additional references;
* ``标准源图坐标.txt`` contains YOLO ROIs shared by the case.

The tool never compares images across case directories and never writes source
images.  It records a pairwise geometric score for every (live, reference)
combination, an illumination-style distance, masked raw/CLAHE ECC observations,
and a geometry-first recommendation of up to K reference candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


LIVE_NAME = "对比截图.jpg"
PRIMARY_REFERENCE_NAME = "标准源图.jpg"
ROI_NAME = "标准源图坐标.txt"
ADDED_REFERENCE_PATTERN = re.compile(r"^新增标准源图(\d+)\.jpg$", re.IGNORECASE)
ROI_BOUNDARY_TOLERANCE = 0.01

MIN_FEATURE_MATCHES = 12
MIN_INLIERS = 8
MIN_INLIER_RATE = 0.40
MAX_REPROJECTION_ERROR_PIXELS = 3.0
MIN_SPATIAL_COVERAGE = 0.02
MIN_PROJECTED_AREA_RATIO = 0.20
MAX_PROJECTED_AREA_RATIO = 5.0
MIN_VALID_OVERLAP_RATIO = 0.60


@dataclass(frozen=True)
class ReferenceImage:
    reference_id: str
    filename: str
    path: Path


@dataclass(frozen=True)
class CaseFiles:
    case_directory: Path
    case_id: str
    case_type: str
    live_path: Path
    roi_path: Path
    references: tuple[ReferenceImage, ...]
    discovery_errors: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="local dataset root")
    parser.add_argument("--output", required=True, type=Path, help="local report directory")
    parser.add_argument("--top-k", type=int, default=3, help="recommended candidates per case (default: 3)")
    parser.add_argument("--max-cases", type=int, help="optional deterministic smoke-test limit")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="optional exact case-directory name; may be supplied more than once for focused analysis",
    )
    parser.add_argument("--ecc-max-side", type=int, default=640, help="max side used only for ECC (default: 640)")
    parser.add_argument("--ecc-iterations", type=int, default=30, help="ECC iteration cap (default: 30)")
    arguments = parser.parse_args()
    if arguments.top_k <= 0 or arguments.ecc_max_side <= 0 or arguments.ecc_iterations <= 0:
        parser.error("--top-k, --ecc-max-side, and --ecc-iterations must be positive")
    return arguments


def case_type_for(case_id: str) -> str:
    if "_" not in case_id:
        return "UNSPECIFIED"
    suffix = case_id.rsplit("_", 1)[1]
    return suffix or "UNSPECIFIED"


def discover_case(case_directory: Path) -> CaseFiles:
    files = {entry.name: entry for entry in case_directory.iterdir() if entry.is_file()}
    errors: list[str] = []
    live_path = case_directory / LIVE_NAME
    roi_path = case_directory / ROI_NAME
    if not live_path.is_file():
        errors.append("missing live image")
    if not roi_path.is_file():
        errors.append("missing ROI file")

    references: list[ReferenceImage] = []
    primary = case_directory / PRIMARY_REFERENCE_NAME
    if primary.is_file():
        references.append(ReferenceImage("standard", PRIMARY_REFERENCE_NAME, primary))
    else:
        errors.append("missing primary reference image")

    added: list[tuple[int, Path]] = []
    for name, path in files.items():
        match = ADDED_REFERENCE_PATTERN.fullmatch(name)
        if match:
            added.append((int(match.group(1)), path))
    added.sort()
    actual_indices = [index for index, _ in added]
    if actual_indices != list(range(len(actual_indices))):
        errors.append("non-contiguous additional reference indices")
    for index, path in added:
        references.append(ReferenceImage(f"added_{index}", path.name, path))

    return CaseFiles(
        case_directory=case_directory,
        case_id=case_directory.name,
        case_type=case_type_for(case_directory.name),
        live_path=live_path,
        roi_path=roi_path,
        references=tuple(references),
        discovery_errors=tuple(errors),
    )


def read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def parse_rois(path: Path) -> tuple[int, list[str], int]:
    errors: list[str] = []
    normalized_lines = 0
    valid_count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="gb18030").splitlines()
    except OSError:
        return 0, ["cannot read ROI file"], 0

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"invalid ROI at line {line_number}")
            continue
        try:
            _, center_x, center_y, width, height = fields
            center_x, center_y, width, height = map(float, (center_x, center_y, width, height))
        except ValueError:
            errors.append(f"invalid ROI at line {line_number}")
            continue
        if not all(math.isfinite(value) for value in (center_x, center_y, width, height)) or width <= 0 or height <= 0:
            errors.append(f"invalid ROI at line {line_number}")
            continue
        left, right = center_x - width / 2.0, center_x + width / 2.0
        top, bottom = center_y - height / 2.0, center_y + height / 2.0
        if left < -ROI_BOUNDARY_TOLERANCE or right > 1.0 + ROI_BOUNDARY_TOLERANCE or top < -ROI_BOUNDARY_TOLERANCE or bottom > 1.0 + ROI_BOUNDARY_TOLERANCE:
            errors.append(f"invalid ROI at line {line_number}")
            continue
        clipped = (max(0.0, left), min(1.0, right), max(0.0, top), min(1.0, bottom))
        if clipped[1] <= clipped[0] or clipped[3] <= clipped[2]:
            errors.append(f"invalid ROI at line {line_number}")
            continue
        if clipped != (left, right, top, bottom):
            normalized_lines += 1
        valid_count += 1
    if valid_count == 0 and not errors:
        errors.append("ROI file contains no ROI")
    return valid_count, errors, normalized_lines


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def quantile(values: np.ndarray, fraction: float) -> float:
    return float(np.quantile(values, fraction, method="linear"))


def illumination_signature(image: np.ndarray) -> dict[str, float]:
    image_gray = gray(image)
    sample = cv2.resize(image_gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    bgr_mean = image.reshape(-1, 3).mean(axis=0)
    return {
        "luma_mean": float(sample.mean()),
        "luma_std": float(sample.std()),
        "luma_p10": quantile(sample, 0.10),
        "luma_p50": quantile(sample, 0.50),
        "luma_p90": quantile(sample, 0.90),
        "blue_mean": float(bgr_mean[0]),
        "green_mean": float(bgr_mean[1]),
        "red_mean": float(bgr_mean[2]),
    }


def illumination_band(signature: dict[str, float], thresholds: tuple[float, float]) -> str:
    if signature["luma_p50"] <= thresholds[0]:
        return "dark"
    if signature["luma_p50"] <= thresholds[1]:
        return "medium"
    return "bright"


def style_distance(live: dict[str, float], reference: dict[str, float]) -> float:
    luma_component = (
        0.15 * abs(live["luma_p10"] - reference["luma_p10"]) / 255.0
        + 0.35 * abs(live["luma_p50"] - reference["luma_p50"]) / 255.0
        + 0.15 * abs(live["luma_p90"] - reference["luma_p90"]) / 255.0
        + 0.15 * abs(live["luma_std"] - reference["luma_std"]) / 128.0
    )
    color_delta = math.sqrt(
        (live["blue_mean"] - reference["blue_mean"]) ** 2
        + (live["green_mean"] - reference["green_mean"]) ** 2
        + (live["red_mean"] - reference["red_mean"]) ** 2
    ) / (math.sqrt(3.0) * 255.0)
    return luma_component + 0.20 * color_delta


def finite_homography(homography: np.ndarray) -> bool:
    return homography.shape == (3, 3) and bool(np.isfinite(homography).all())


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def geometric_pair_metrics(reference: np.ndarray, live: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "reference_width": int(reference.shape[1]),
        "reference_height": int(reference.shape[0]),
        "live_width": int(live.shape[1]),
        "live_height": int(live.shape[0]),
        "feature_match_count": 0,
        "inlier_count": 0,
        "inlier_rate": None,
        "reprojection_error_pixels": None,
        "spatial_coverage": None,
        "center_displacement_pixels": None,
        "center_displacement_relative_diagonal": None,
        "corner_displacement_median_pixels": None,
        "projected_corners_in_live_frame": 0,
        "projected_area_ratio": None,
        "projected_geometry_valid": False,
        "homography_available": False,
        "valid_overlap_available": False,
        "valid_overlap_ratio": None,
        "geometry_class": "unavailable",
        "geometry_reasons": [],
        "aligned_live": None,
        "valid_mask": None,
    }
    reference_gray, live_gray = gray(reference), gray(live)
    orb = cv2.ORB_create(nfeatures=2000)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_gray, None)
    live_keypoints, live_descriptors = orb.detectAndCompute(live_gray, None)
    if reference_descriptors is None or live_descriptors is None:
        add_reason(metrics["geometry_reasons"], "missing ORB descriptors")
        return metrics

    neighbours = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(reference_descriptors, live_descriptors, k=2)
    matches = [pair[0] for pair in neighbours if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    metrics["feature_match_count"] = len(matches)
    if len(matches) < 4:
        add_reason(metrics["geometry_reasons"], "fewer than four ratio-test matches")
        return metrics

    reference_points = np.float32([reference_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    live_points = np.float32([live_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(reference_points, live_points, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        add_reason(metrics["geometry_reasons"], "homography unavailable")
        return metrics
    homography = homography.astype(np.float64)
    metrics["homography_available"] = True
    if not finite_homography(homography):
        add_reason(metrics["geometry_reasons"], "homography contains a non-finite coefficient")
        return metrics

    inlier_indices = inlier_mask.reshape(-1).astype(bool)
    inlier_reference = reference_points[inlier_indices]
    inlier_live = live_points[inlier_indices]
    metrics["inlier_count"] = int(inlier_indices.sum())
    metrics["inlier_rate"] = metrics["inlier_count"] / metrics["feature_match_count"]
    if metrics["inlier_count"] == 0:
        add_reason(metrics["geometry_reasons"], "homography has no inliers")
        return metrics

    projected_inliers = cv2.perspectiveTransform(inlier_reference, homography)
    metrics["reprojection_error_pixels"] = float(np.linalg.norm(projected_inliers - inlier_live, axis=2).mean())
    if metrics["inlier_count"] >= 3:
        hull = cv2.convexHull(inlier_reference.reshape(-1, 2))
        metrics["spatial_coverage"] = abs(float(cv2.contourArea(hull))) / (reference_gray.shape[0] * reference_gray.shape[1])

    reference_corners = np.float32(
        [[0, 0], [reference_gray.shape[1] - 1, 0], [reference_gray.shape[1] - 1, reference_gray.shape[0] - 1], [0, reference_gray.shape[0] - 1]]
    ).reshape(-1, 1, 2)
    reference_center = np.float32([[(reference_gray.shape[1] - 1) / 2.0, (reference_gray.shape[0] - 1) / 2.0]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(np.concatenate((reference_corners, reference_center), axis=0), homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        add_reason(metrics["geometry_reasons"], "projected image geometry is non-finite")
        return metrics
    projected_corners, projected_center = projected[:4], projected[4]
    projected_area = abs(float(cv2.contourArea(projected_corners.astype(np.float32))))
    if not math.isfinite(projected_area) or projected_area <= 1.0 or not cv2.isContourConvex(projected_corners.astype(np.float32)):
        add_reason(metrics["geometry_reasons"], "projected image geometry is degenerate")
        return metrics

    metrics["projected_geometry_valid"] = True
    metrics["projected_area_ratio"] = projected_area / (live_gray.shape[0] * live_gray.shape[1])
    live_center = np.array([(live_gray.shape[1] - 1) / 2.0, (live_gray.shape[0] - 1) / 2.0])
    metrics["center_displacement_pixels"] = float(np.linalg.norm(projected_center - live_center))
    metrics["center_displacement_relative_diagonal"] = metrics["center_displacement_pixels"] / math.hypot(live_gray.shape[1], live_gray.shape[0])
    live_corners = np.array(
        [[0, 0], [live_gray.shape[1] - 1, 0], [live_gray.shape[1] - 1, live_gray.shape[0] - 1], [0, live_gray.shape[0] - 1]],
        dtype=np.float32,
    )
    metrics["corner_displacement_median_pixels"] = median([float(np.linalg.norm(projected_corners[index] - live_corners[index])) for index in range(4)])
    metrics["projected_corners_in_live_frame"] = int(
        sum(0 <= point[0] <= live_gray.shape[1] - 1 and 0 <= point[1] <= live_gray.shape[0] - 1 for point in projected_corners)
    )

    invertible, inverse_homography = cv2.invert(homography, cv2.DECOMP_SVD)
    if invertible == 0.0:
        add_reason(metrics["geometry_reasons"], "homography cannot be inverted")
        return metrics
    live_mask = np.full(live_gray.shape, 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        live_mask,
        inverse_homography,
        (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    metrics["valid_overlap_available"] = True
    metrics["valid_overlap_ratio"] = float(np.count_nonzero(valid_mask)) / valid_mask.size
    metrics["aligned_live"] = cv2.warpPerspective(
        live_gray,
        inverse_homography,
        (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    metrics["valid_mask"] = valid_mask

    quality_reasons = metrics["geometry_reasons"]
    if metrics["feature_match_count"] < MIN_FEATURE_MATCHES:
        add_reason(quality_reasons, "feature matches below geometry minimum (12)")
    if metrics["inlier_count"] < MIN_INLIERS:
        add_reason(quality_reasons, "inliers below geometry minimum (8)")
    if metrics["inlier_rate"] is None or metrics["inlier_rate"] < MIN_INLIER_RATE:
        add_reason(quality_reasons, "inlier rate below geometry minimum (0.40)")
    if metrics["reprojection_error_pixels"] is None or metrics["reprojection_error_pixels"] > MAX_REPROJECTION_ERROR_PIXELS:
        add_reason(quality_reasons, "reprojection error exceeds geometry maximum (3 px)")
    if metrics["spatial_coverage"] is None or metrics["spatial_coverage"] < MIN_SPATIAL_COVERAGE:
        add_reason(quality_reasons, "inlier spatial coverage below geometry minimum (0.02)")
    if metrics["projected_area_ratio"] is None or not MIN_PROJECTED_AREA_RATIO <= metrics["projected_area_ratio"] <= MAX_PROJECTED_AREA_RATIO:
        add_reason(quality_reasons, "projected area ratio outside geometry range [0.20, 5.00]")
    if metrics["valid_overlap_ratio"] is None or metrics["valid_overlap_ratio"] < MIN_VALID_OVERLAP_RATIO:
        add_reason(quality_reasons, "valid overlap below geometry minimum (0.60)")
    metrics["geometry_class"] = "usable" if not quality_reasons else "degraded"
    return metrics


def resize_ecc_inputs(template: np.ndarray, aligned: np.ndarray, mask: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_max_side = max(template.shape)
    if current_max_side <= max_side:
        return template, aligned, mask
    scale = max_side / current_max_side
    size = (max(1, round(template.shape[1] * scale)), max(1, round(template.shape[0] * scale)))
    return (
        cv2.resize(template, size, interpolation=cv2.INTER_AREA),
        cv2.resize(aligned, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
    )


def ecc_correlation(template: np.ndarray, aligned: np.ndarray, mask: np.ndarray, max_side: int, iterations: int, clahe: bool) -> tuple[float | None, str | None]:
    template, aligned, mask = resize_ecc_inputs(template, aligned, mask, max_side)
    if np.count_nonzero(mask) < 16:
        return None, "too few valid-overlap pixels for ECC"
    if clahe:
        normalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        template = normalizer.apply(template)
        aligned = normalizer.apply(aligned)
    try:
        correlation, _ = cv2.findTransformECC(
            template,
            aligned,
            np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, 1e-5),
            inputMask=mask,
        )
    except cv2.error as error:
        return None, str(error).splitlines()[0]
    return float(correlation), None


def pair_record(case: CaseFiles, reference: ReferenceImage, live_signature: dict[str, float], reference_signature: dict[str, float], band_thresholds: tuple[float, float], ecc_max_side: int, ecc_iterations: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case": case.case_id,
        "case_type": case.case_type,
        "reference_id": reference.reference_id,
        "reference_filename": reference.filename,
        "reference_path": str(reference.path),
        "live_path": str(case.live_path),
        "reference_count": len(case.references),
        "live_illumination_band": illumination_band(live_signature, band_thresholds),
        "reference_illumination_band": illumination_band(reference_signature, band_thresholds),
        "style_band_match": illumination_band(live_signature, band_thresholds) == illumination_band(reference_signature, band_thresholds),
        "style_distance": style_distance(live_signature, reference_signature),
        **{f"live_{key}": value for key, value in live_signature.items()},
        **{f"reference_{key}": value for key, value in reference_signature.items()},
        "roi_count": None,
        "roi_normalized_lines": None,
        "errors": [],
        "geometry_class": "unavailable",
        "geometry_reasons": [],
        "ecc_masked_raw": None,
        "ecc_masked_clahe": None,
        "ecc_raw_error": None,
        "ecc_clahe_error": None,
    }
    roi_count, roi_errors, roi_normalized_lines = parse_rois(case.roi_path)
    result["roi_count"] = roi_count
    result["roi_normalized_lines"] = roi_normalized_lines
    if case.discovery_errors or roi_errors:
        result["errors"] = list(case.discovery_errors) + roi_errors
        return result

    reference_image = read_image(reference.path)
    live_image = read_image(case.live_path)
    if reference_image is None:
        result["errors"].append("unreadable reference image")
    if live_image is None:
        result["errors"].append("unreadable live image")
    if result["errors"]:
        return result

    geometry = geometric_pair_metrics(reference_image, live_image)
    aligned_live = geometry.pop("aligned_live")
    valid_mask = geometry.pop("valid_mask")
    result.update(geometry)
    if result["geometry_class"] != "usable" or aligned_live is None or valid_mask is None:
        return result

    reference_gray = gray(reference_image)
    result["ecc_masked_raw"], result["ecc_raw_error"] = ecc_correlation(
        reference_gray, aligned_live, valid_mask, ecc_max_side, ecc_iterations, clahe=False
    )
    result["ecc_masked_clahe"], result["ecc_clahe_error"] = ecc_correlation(
        reference_gray, aligned_live, valid_mask, ecc_max_side, ecc_iterations, clahe=True
    )
    return result


def quality_rank(record: dict[str, Any]) -> int:
    return {"usable": 0, "degraded": 1, "unavailable": 2}.get(record["geometry_class"], 3)


def candidate_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        quality_rank(record),
        record["style_distance"],
        -float(record.get("valid_overlap_ratio") or 0.0),
        -float(record.get("spatial_coverage") or 0.0),
        -float(record.get("inlier_rate") or 0.0),
        record["reference_id"],
    )


def select_references(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    analyzed = [record for record in records if not record["errors"]]
    usable = [record for record in analyzed if record["geometry_class"] == "usable"]
    degraded = [record for record in analyzed if record["geometry_class"] == "degraded"]
    pool = usable or degraded
    ordered = sorted(pool, key=candidate_sort_key)
    recommendations = ordered[:top_k]
    for rank, record in enumerate(recommendations, start=1):
        record["candidate_rank"] = rank
        record["recommended"] = True
    return {
        "selected_reference_id": recommendations[0]["reference_id"] if recommendations else None,
        "selected_reference_filename": recommendations[0]["reference_filename"] if recommendations else None,
        "selected_geometry_class": recommendations[0]["geometry_class"] if recommendations else "unavailable",
        "recommended_reference_ids": "|".join(record["reference_id"] for record in recommendations),
        "selection_pool": "usable" if usable else "degraded" if degraded else "unavailable",
        "usable_reference_count": len(usable),
        "degraded_reference_count": len(degraded),
        "unavailable_reference_count": sum(record["geometry_class"] == "unavailable" for record in records),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {key: "|".join(value) if isinstance(value, list) else value for key, value in row.items()}
            writer.writerow(json_safe(serializable))


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    lines = [
        "# 多参考图识别与候选配对报告",
        "",
        "该报告只读取本地图片计算特征和本地 CSV/JSON；不复制或导出源图像。`dark`/`medium`/`bright` 是按全数据集亮度中位数三分位得到的视觉风格桶，不是有 EXIF 证据的白天/夜晚标签。",
        "",
        "## 命名识别",
        "",
        f"- case 目录：{counts['case_directories']}",
        f"- 原始参考图：{counts['primary_references']}",
        f"- 补充参考图：{counts['additional_references']}",
        f"- 含多个参考图的 case：{counts['multi_reference_cases']}",
        f"- 可读实时图：{counts['readable_live_images']}",
        f"- 可读参考图：{counts['readable_reference_images']}",
        "",
        "## 候选选择规则",
        "",
        "1. 仅在同一 case 目录内配对实时图与每张真实参考图；",
        "2. 每对独立估计单应性和有效重叠；",
        "3. 先按几何类别（`usable` > `degraded` > `unavailable`）筛选；",
        "4. 在同一几何类别中，以连续 `style_distance` 最小者优先，再以重叠、空间覆盖和内点率打破并列；",
        "5. 输出至多 3 个推荐候选，不用 ROI 差分值参与选择，避免真实异动影响参考图选择。",
        "",
        "`usable` 仅由匹配、内点、重投影误差、空间覆盖、投影面积和有效重叠确定；ECC 只作掩码化的昼夜敏感性观测，不会降低几何类别。",
        "",
        "## 结果",
        "",
        f"- 计算的实时图—参考图配对：{counts['pair_records']}",
        f"- 几何 `usable` 配对：{counts['geometry_usable_pairs']}",
        f"- 选中补充参考图为第一候选的 case：{counts['selected_additional_reference_cases']}",
        f"- 有多个几何 `usable` 候选的 case：{counts['cases_with_multiple_usable_references']}",
        "",
        "详见 `multireference_pairs.csv`（每个配对）和 `multireference_selection.csv`（每个 case 的推荐参考图）。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    arguments = parse_args()
    if not arguments.dataset.is_dir():
        raise ValueError("--dataset must be an existing directory")
    case_directories = sorted(entry for entry in arguments.dataset.iterdir() if entry.is_dir())
    if arguments.case_ids:
        case_by_name = {case_directory.name: case_directory for case_directory in case_directories}
        missing_cases = [case_id for case_id in arguments.case_ids if case_id not in case_by_name]
        if missing_cases:
            raise ValueError(f"--case not found: {', '.join(missing_cases)}")
        case_directories = [case_by_name[case_id] for case_id in arguments.case_ids]
    elif arguments.max_cases is not None:
        case_directories = case_directories[: arguments.max_cases]
    cases = [discover_case(case_directory) for case_directory in case_directories]

    signatures: dict[Path, dict[str, float]] = {}
    unreadable_paths: set[Path] = set()
    for case in cases:
        candidates = [case.live_path, *(reference.path for reference in case.references)]
        for path in candidates:
            if path in signatures or path in unreadable_paths:
                continue
            image = read_image(path)
            if image is None:
                unreadable_paths.add(path)
            else:
                signatures[path] = illumination_signature(image)
    if not signatures:
        raise ValueError("no readable images found")
    p50_values = np.array([signature["luma_p50"] for signature in signatures.values()], dtype=np.float32)
    band_thresholds = (quantile(p50_values, 1.0 / 3.0), quantile(p50_values, 2.0 / 3.0))

    pair_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        records: list[dict[str, Any]] = []
        live_signature = signatures.get(case.live_path)
        if live_signature is None:
            records = [
                {
                    "case": case.case_id,
                    "case_type": case.case_type,
                    "reference_id": reference.reference_id,
                    "reference_filename": reference.filename,
                    "reference_path": str(reference.path),
                    "live_path": str(case.live_path),
                    "reference_count": len(case.references),
                    "errors": ["unreadable live image"],
                    "geometry_class": "unavailable",
                    "geometry_reasons": [],
                    "candidate_rank": None,
                    "recommended": False,
                }
                for reference in case.references
            ]
        else:
            for reference in case.references:
                reference_signature = signatures.get(reference.path)
                if reference_signature is None:
                    records.append(
                        {
                            "case": case.case_id,
                            "case_type": case.case_type,
                            "reference_id": reference.reference_id,
                            "reference_filename": reference.filename,
                            "reference_path": str(reference.path),
                            "live_path": str(case.live_path),
                            "reference_count": len(case.references),
                            "errors": ["unreadable reference image"],
                            "geometry_class": "unavailable",
                            "geometry_reasons": [],
                            "candidate_rank": None,
                            "recommended": False,
                        }
                    )
                    continue
                records.append(
                    pair_record(
                        case,
                        reference,
                        live_signature,
                        reference_signature,
                        band_thresholds,
                        arguments.ecc_max_side,
                        arguments.ecc_iterations,
                    )
                )
        for record in records:
            record.setdefault("candidate_rank", None)
            record.setdefault("recommended", False)
        selection = select_references(records, arguments.top_k)
        selection_rows.append(
            {
                "case": case.case_id,
                "case_type": case.case_type,
                "reference_count": len(case.references),
                "discovery_errors": list(case.discovery_errors),
                **selection,
            }
        )
        pair_rows.extend(records)
        if index % 10 == 0 or index == len(cases):
            print(f"processed {index}/{len(cases)} cases", flush=True)

    geometry_counts = Counter(row["geometry_class"] for row in pair_rows)
    selected_additional = sum(row["selected_reference_id"].startswith("added_") for row in selection_rows if row["selected_reference_id"])
    summary = {
        "schema_version": "multireference-pairing-v1",
        "scope": {"execution": "local-only", "source_images_copied": False, "source_images_exported": False},
        "naming_rules": {
            "live": LIVE_NAME,
            "primary_reference": PRIMARY_REFERENCE_NAME,
            "additional_reference_pattern": ADDED_REFERENCE_PATTERN.pattern,
            "roi": ROI_NAME,
            "case_directory_depth": "direct child only",
        },
        "illumination_style": {
            "method": "luma median terciles over all readable live and reference images",
            "dark_medium_threshold": band_thresholds[0],
            "medium_bright_threshold": band_thresholds[1],
            "warning": "style bands are not verified day/night labels because no EXIF timestamps were found",
        },
        "geometry_policy": {
            "feature_match_count_min": MIN_FEATURE_MATCHES,
            "inlier_count_min": MIN_INLIERS,
            "inlier_rate_min": MIN_INLIER_RATE,
            "reprojection_error_pixels_max": MAX_REPROJECTION_ERROR_PIXELS,
            "spatial_coverage_min": MIN_SPATIAL_COVERAGE,
            "projected_area_ratio_range": [MIN_PROJECTED_AREA_RATIO, MAX_PROJECTED_AREA_RATIO],
            "valid_overlap_ratio_min": MIN_VALID_OVERLAP_RATIO,
            "ecc_role": "masked raw and CLAHE ECC are observations only, not geometry gates",
            "ecc_max_side": arguments.ecc_max_side,
            "ecc_iterations": arguments.ecc_iterations,
        },
        "counts": {
            "case_directories": len(cases),
            "primary_references": sum(1 for case in cases if any(reference.reference_id == "standard" for reference in case.references)),
            "additional_references": sum(1 for case in cases for reference in case.references if reference.reference_id.startswith("added_")),
            "multi_reference_cases": sum(len(case.references) > 1 for case in cases),
            "readable_live_images": sum(case.live_path in signatures for case in cases),
            "readable_reference_images": sum(reference.path in signatures for case in cases for reference in case.references),
            "unreadable_images": len(unreadable_paths),
            "pair_records": len(pair_rows),
            "geometry_usable_pairs": geometry_counts["usable"],
            "geometry_degraded_pairs": geometry_counts["degraded"],
            "geometry_unavailable_pairs": geometry_counts["unavailable"],
            "selected_additional_reference_cases": selected_additional,
            "cases_with_multiple_usable_references": sum(row["usable_reference_count"] > 1 for row in selection_rows),
        },
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output / "multireference_pairs.csv", pair_rows)
    write_csv(arguments.output / "multireference_selection.csv", selection_rows)
    with (arguments.output / "multireference_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(json_safe(summary), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    write_markdown(arguments.output / "multireference_summary.md", summary)
    print(json.dumps(json_safe(summary["counts"]), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
