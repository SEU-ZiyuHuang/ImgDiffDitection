"""Versioned, validated configuration for image comparison.

The built-in configuration is intended only for local development and test
fixtures.  A daily-detection deployment should pass an explicit, calibrated
JSON file and retain the reported version with every result.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


_DEVELOPMENT_CONFIG_VERSION = "development-default-v1"


@dataclass(frozen=True)
class AlignmentThresholds:
    """All quality gates and numeric alignment parameters.

    They are configuration rather than business conclusions.  Calibration is
    responsible for replacing the development values before daily use.
    """

    orb_feature_count: int = 2000
    match_ratio_test_max: float = 0.75
    ransac_reprojection_threshold_pixels: float = 3.0
    ecc_max_iterations: int = 50
    ecc_epsilon: float = 1e-5

    feature_match_count_min: int = 12
    inlier_count_min: int = 8
    inlier_rate_min: float = 0.40
    reprojection_error_pixels_max: float = 3.0
    spatial_coverage_min: float = 0.02
    projected_area_ratio_min: float = 0.20
    projected_area_ratio_max: float = 5.0
    valid_overlap_ratio_min: float = 0.60
    roi_valid_overlap_ratio_min: float = 0.60
    ecc_correlation_min_when_converged: float = 0.20


@dataclass(frozen=True)
class DetectionThresholds:
    """Difference-channel parameters and calibrated decision boundaries."""

    lab_delta_e_threshold: float = 8.0
    gradient_magnitude_diff_threshold: float = 30.0
    difference_decision_threshold: float = 0.50
    morphology_kernel_size: int = 1
    min_detection_area_pixels: int = 100
    min_detection_confidence: float = 0.30


@dataclass(frozen=True)
class CalibratedConfig:
    """A complete, versioned configuration used for one comparison call."""

    version: str = _DEVELOPMENT_CONFIG_VERSION
    alignment: AlignmentThresholds = field(default_factory=AlignmentThresholds)
    detection: DetectionThresholds = field(default_factory=DetectionThresholds)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"configuration field {name!r} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"configuration field {name!r} must be finite")
    return value


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    numeric = _finite_number(value, name)
    if not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"configuration field {name!r} must be an integer >= {minimum}")
    return int(numeric)


def _merge_dataclass(raw: Any, defaults: Any, section: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"configuration section {section!r} must be an object")

    allowed = {entry.name for entry in fields(defaults)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"unknown configuration field(s) in {section}: {', '.join(unknown)}"
        )
    missing = sorted(allowed - set(raw))
    if missing:
        raise ValueError(
            f"missing configuration field(s) in {section}: {', '.join(missing)}"
        )
    return {entry.name: raw[entry.name] for entry in fields(defaults)}


def _parse_alignment(raw: Any) -> AlignmentThresholds:
    values = _merge_dataclass(raw, AlignmentThresholds(), "alignment")
    parsed = AlignmentThresholds(
        orb_feature_count=_positive_int(values["orb_feature_count"], "alignment.orb_feature_count", 4),
        match_ratio_test_max=_finite_number(values["match_ratio_test_max"], "alignment.match_ratio_test_max"),
        ransac_reprojection_threshold_pixels=_finite_number(
            values["ransac_reprojection_threshold_pixels"],
            "alignment.ransac_reprojection_threshold_pixels",
        ),
        ecc_max_iterations=_positive_int(values["ecc_max_iterations"], "alignment.ecc_max_iterations"),
        ecc_epsilon=_finite_number(values["ecc_epsilon"], "alignment.ecc_epsilon"),
        feature_match_count_min=_positive_int(
            values["feature_match_count_min"], "alignment.feature_match_count_min", 4
        ),
        inlier_count_min=_positive_int(values["inlier_count_min"], "alignment.inlier_count_min", 4),
        inlier_rate_min=_finite_number(values["inlier_rate_min"], "alignment.inlier_rate_min"),
        reprojection_error_pixels_max=_finite_number(
            values["reprojection_error_pixels_max"],
            "alignment.reprojection_error_pixels_max",
        ),
        spatial_coverage_min=_finite_number(
            values["spatial_coverage_min"], "alignment.spatial_coverage_min"
        ),
        projected_area_ratio_min=_finite_number(
            values["projected_area_ratio_min"], "alignment.projected_area_ratio_min"
        ),
        projected_area_ratio_max=_finite_number(
            values["projected_area_ratio_max"], "alignment.projected_area_ratio_max"
        ),
        valid_overlap_ratio_min=_finite_number(
            values["valid_overlap_ratio_min"], "alignment.valid_overlap_ratio_min"
        ),
        roi_valid_overlap_ratio_min=_finite_number(
            values["roi_valid_overlap_ratio_min"], "alignment.roi_valid_overlap_ratio_min"
        ),
        ecc_correlation_min_when_converged=_finite_number(
            values["ecc_correlation_min_when_converged"],
            "alignment.ecc_correlation_min_when_converged",
        ),
    )

    if not 0.0 < parsed.match_ratio_test_max <= 1.0:
        raise ValueError("alignment.match_ratio_test_max must be in (0, 1]")
    if parsed.ransac_reprojection_threshold_pixels <= 0.0:
        raise ValueError("alignment.ransac_reprojection_threshold_pixels must be > 0")
    if parsed.ecc_epsilon <= 0.0:
        raise ValueError("alignment.ecc_epsilon must be > 0")
    if not 0.0 <= parsed.inlier_rate_min <= 1.0:
        raise ValueError("alignment.inlier_rate_min must be in [0, 1]")
    if parsed.reprojection_error_pixels_max <= 0.0:
        raise ValueError("alignment.reprojection_error_pixels_max must be > 0")
    if not 0.0 <= parsed.spatial_coverage_min <= 1.0:
        raise ValueError("alignment.spatial_coverage_min must be in [0, 1]")
    if (parsed.projected_area_ratio_min <= 0.0
            or parsed.projected_area_ratio_max < parsed.projected_area_ratio_min):
        raise ValueError("alignment projected-area ratio range is invalid")
    if not 0.0 <= parsed.valid_overlap_ratio_min <= 1.0:
        raise ValueError("alignment.valid_overlap_ratio_min must be in [0, 1]")
    if not 0.0 <= parsed.roi_valid_overlap_ratio_min <= 1.0:
        raise ValueError("alignment.roi_valid_overlap_ratio_min must be in [0, 1]")
    if not -1.0 <= parsed.ecc_correlation_min_when_converged <= 1.0:
        raise ValueError("alignment.ecc_correlation_min_when_converged must be in [-1, 1]")
    return parsed


def _parse_detection(raw: Any) -> DetectionThresholds:
    values = _merge_dataclass(raw, DetectionThresholds(), "detection")
    parsed = DetectionThresholds(
        lab_delta_e_threshold=_finite_number(
            values["lab_delta_e_threshold"], "detection.lab_delta_e_threshold"
        ),
        gradient_magnitude_diff_threshold=_finite_number(
            values["gradient_magnitude_diff_threshold"],
            "detection.gradient_magnitude_diff_threshold",
        ),
        difference_decision_threshold=_finite_number(
            values["difference_decision_threshold"],
            "detection.difference_decision_threshold",
        ),
        morphology_kernel_size=_positive_int(
            values["morphology_kernel_size"], "detection.morphology_kernel_size"
        ),
        min_detection_area_pixels=_positive_int(
            values["min_detection_area_pixels"], "detection.min_detection_area_pixels"
        ),
        min_detection_confidence=_finite_number(
            values["min_detection_confidence"], "detection.min_detection_confidence"
        ),
    )
    if parsed.lab_delta_e_threshold <= 0.0:
        raise ValueError("detection.lab_delta_e_threshold must be > 0")
    if parsed.gradient_magnitude_diff_threshold <= 0.0:
        raise ValueError("detection.gradient_magnitude_diff_threshold must be > 0")
    if not 0.0 < parsed.difference_decision_threshold <= 1.0:
        raise ValueError("detection.difference_decision_threshold must be in (0, 1]")
    if parsed.morphology_kernel_size % 2 != 1:
        raise ValueError("detection.morphology_kernel_size must be odd")
    if not 0.0 <= parsed.min_detection_confidence <= 1.0:
        raise ValueError("detection.min_detection_confidence must be in [0, 1]")
    return parsed


def load_config(
    path: Optional[Path], *, allow_development: bool = True
) -> CalibratedConfig:
    """Load and validate a configuration.

    ``None`` selects a clearly-labelled development configuration for
    calibration observations and local synthetic tests.  Daily detection
    passes ``allow_development=False`` so it cannot publish a business result
    from that profile.  A path that was explicitly supplied must exist and be
    valid; silently falling back would hide deployment mistakes as detection
    outcomes.
    """
    if path is None:
        print(
            "imagecmp: using development defaults; provide a calibrated config "
            "before daily detection deployment",
            file=sys.stderr,
        )
        return CalibratedConfig()
    if not path.is_file():
        raise FileNotFoundError(f"calibration config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in calibration config {path}: {exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read calibration config {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = sorted(set(raw) - {"version", "alignment", "detection"})
    if unknown:
        raise ValueError(f"unknown configuration section(s): {', '.join(unknown)}")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("configuration version must be a non-empty string")

    config = CalibratedConfig(
        version=version.strip(),
        alignment=_parse_alignment(raw.get("alignment")),
        detection=_parse_detection(raw.get("detection")),
    )
    if not allow_development and config.version == _DEVELOPMENT_CONFIG_VERSION:
        raise ValueError(
            "development configuration cannot be used for daily detection; "
            "supply a versioned configuration produced by calibration"
        )
    return config


def default_config() -> CalibratedConfig:
    """Return the development-only configuration for synthetic tests."""
    return CalibratedConfig()
