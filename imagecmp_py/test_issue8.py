"""Issue 8 behaviour tests using programmatically generated images only.

The tests use the public configuration, alignment and comparison seams.  They
never read the controlled local validation set.
"""

from __future__ import annotations

import sys
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imagecmp.config import default_config
from imagecmp.compare import compare_aligned
from imagecmp import ComparisonState, ImageComparisonService
from imagecmp.alignment import AlignmentDiagnostic, align


def test_issue8_defaults_keep_neural_alignment_fallback_disabled() -> None:
    """A model with pending deployment approval is opt-in, never implicit."""
    config = default_config()

    assert config.alignment.superpoint_lightglue_fallback_enabled is False
    assert config.detection.illumination_normalization_enabled is True


def test_illumination_shift_is_normalized_before_colour_decision() -> None:
    """A global exposure change is evidence, but not a colour-only anomaly."""
    random = np.random.RandomState(20260723)
    standard = random.randint(20, 220, (160, 200, 3), dtype=np.uint8)
    cv2.rectangle(standard, (30, 35), (115, 120), (30, 170, 220), -1)
    cv2.putText(standard, "I8", (122, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    lookup = np.rint((np.arange(256, dtype=np.float32) / 255.0) ** 1.65 * 255.0)
    live = cv2.LUT(standard, lookup.astype(np.uint8))

    result = compare_aligned(
        standard_bgr=standard,
        aligned_live_bgr=live,
        valid_mask=np.full(standard.shape[:2], 255, dtype=np.uint8),
        roi_rect=(0, 0, standard.shape[1], standard.shape[0]),
        config=default_config(),
    )

    assert result.evidence_metrics["illumination_normalization_applied"] == 1
    assert result.evidence_metrics["colour_weight"] < 1.0
    assert result.decision_regions == []


def test_strong_exposure_does_not_reverse_a_thin_edge_into_an_anomaly() -> None:
    """A thin line remains the same structure after a monotonic exposure curve."""
    standard = np.full((180, 220, 3), 165, dtype=np.uint8)
    for x in range(18, 210, 28):
        cv2.line(standard, (x, 12), (x, 168), (100, 100, 100), 1)
    lookup = np.rint((np.arange(256, dtype=np.float32) / 255.0) ** 1.6 * 255.0)
    live = cv2.LUT(standard, lookup.astype(np.uint8))

    result = compare_aligned(
        standard_bgr=standard,
        aligned_live_bgr=live,
        valid_mask=np.full(standard.shape[:2], 255, dtype=np.uint8),
        roi_rect=(0, 0, standard.shape[1], standard.shape[0]),
        config=default_config(),
    )

    assert result.decision_regions == []


def test_tiny_candidate_is_retained_without_becoming_a_decision() -> None:
    """A one-pixel colour signal stays inspectable but cannot alert by itself."""
    standard = np.full((80, 80, 3), 120, dtype=np.uint8)
    live = standard.copy()
    live[40, 40] = (0, 255, 0)

    result = compare_aligned(
        standard_bgr=standard,
        aligned_live_bgr=live,
        valid_mask=np.full(standard.shape[:2], 255, dtype=np.uint8),
        roi_rect=(0, 0, 80, 80),
        config=default_config(),
    )

    assert len(result.detection_regions) == 1
    assert result.decision_regions == []
    assert "small_candidate" in result.detection_regions[0].evidence_channels
    assert "low_confidence_candidate" in result.detection_regions[0].evidence_channels
    assert result.detection_regions[0].decision_eligible is False


def test_daily_result_preserves_low_confidence_candidate_without_alerting() -> None:
    """The public daily result exposes tiny evidence without calling it an anomaly."""
    random = np.random.RandomState(20260724)
    standard = random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    cv2.rectangle(standard, (20, 30), (180, 150), (220, 220, 220), 3)
    cv2.putText(standard, "I8", (210, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    live = standard.copy()
    live[120, 160] = (0, 255, 0)

    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        standard_path = directory / "standard.png"
        live_path = directory / "live.png"
        config_path = directory / "issue8-test.json"
        assert cv2.imwrite(str(standard_path), standard)
        assert cv2.imwrite(str(live_path), live)
        payload = asdict(default_config())
        payload["version"] = "issue8-test-v1"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        result = ImageComparisonService().compare(
            standard_path=standard_path,
            live_path=live_path,
            roi="equipment 0.5 0.5 0.8 0.8",
            output_dir=directory / "out",
            config_path=config_path,
        )

    assert result.state == ComparisonState.NO_CHANGE_HIGH_CONFIDENCE
    assert any("small_candidate" in item.evidence_channels for item in result.detection_regions)


def test_local_colour_change_is_decision_eligible_inside_valid_roi() -> None:
    """A sizeable synthetic colour change remains a reportable difference."""
    random = np.random.RandomState(20260726)
    standard = random.randint(10, 240, (180, 220, 3), dtype=np.uint8)
    live = standard.copy()
    cv2.rectangle(live, (82, 66), (142, 126), (0, 255, 0), -1)
    config = replace(
        default_config(),
        detection=replace(
            default_config().detection,
            illumination_luma_shift_full_weight=255.0,
        ),
    )

    result = compare_aligned(
        standard_bgr=standard,
        aligned_live_bgr=live,
        valid_mask=np.full(standard.shape[:2], 255, dtype=np.uint8),
        roi_rect=(30, 30, 160, 130),
        config=config,
    )

    assert result.decision_regions
    assert all(item.decision_eligible for item in result.decision_regions)
    assert any(
        "lab_colour_illumination_normalized" in item.evidence_channels
        for item in result.decision_regions
    )


def test_background_change_outside_component_roi_creates_no_evidence() -> None:
    """Pixels outside the expected component ROI cannot make a candidate."""
    random = np.random.RandomState(20260727)
    standard = random.randint(10, 240, (180, 220, 3), dtype=np.uint8)
    live = standard.copy()
    cv2.rectangle(live, (0, 0), (25, 25), (0, 0, 255), -1)

    result = compare_aligned(
        standard_bgr=standard,
        aligned_live_bgr=live,
        valid_mask=np.full(standard.shape[:2], 255, dtype=np.uint8),
        roi_rect=(40, 40, 120, 100),
        config=default_config(),
    )

    assert result.detection_regions == []


def test_global_blur_is_reported_as_insufficient_image_quality() -> None:
    """A blurred frame cannot be silently returned as a high-confidence normal."""
    random = np.random.RandomState(20260729)
    standard = random.randint(0, 255, (180, 220, 3), dtype=np.uint8)
    live = cv2.GaussianBlur(standard, (9, 9), 1.8)

    result = compare_aligned(
        standard_bgr=standard,
        aligned_live_bgr=live,
        valid_mask=np.full(standard.shape[:2], 255, dtype=np.uint8),
        roi_rect=(0, 0, standard.shape[1], standard.shape[0]),
        config=default_config(),
    )

    assert result.comparison_quality_usable is False
    assert result.evidence_metrics["sharpness_ratio"] < (
        default_config().detection.image_quality_sharpness_ratio_min
    )


def test_neural_fallback_failure_never_bypasses_global_alignment_gate() -> None:
    """A bad model hash leaves a featureless input detection-unavailable."""
    alignment_config = replace(
        default_config().alignment,
        superpoint_lightglue_fallback_enabled=True,
        superpoint_lightglue_model_sha256="0" * 64,
    )
    config = replace(default_config(), alignment=alignment_config)
    featureless = np.full((96, 128), 127, dtype=np.uint8)

    result = align(featureless, featureless, config)

    assert result.diagnostic == AlignmentDiagnostic.UNAVAILABLE
    assert result.as_metrics_dict()["alignment_fallback_attempted"] == 1
    assert "hash" in result.as_metrics_dict()["alignment_fallback_detail"]


def test_neural_fallback_can_recover_then_pass_the_same_global_gates() -> None:
    """Neural correspondences must still earn a usable result through all gates."""
    random = np.random.RandomState(20260725)
    standard = random.randint(0, 255, (360, 480), dtype=np.uint8)
    cv2.rectangle(standard, (70, 80), (250, 250), 240, 4)
    cv2.circle(standard, (350, 150), 45, 20, -1)
    cv2.putText(standard, "I8", (160, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    rotation = cv2.getRotationMatrix2D((240, 180), 11.0, 1.0)
    live = cv2.warpAffine(standard, rotation, (480, 360), borderMode=cv2.BORDER_REFLECT)
    alignment_config = replace(
        default_config().alignment,
        orb_feature_count=4,
        superpoint_lightglue_fallback_enabled=True,
    )
    result = align(standard, live, replace(default_config(), alignment=alignment_config))

    assert result.diagnostic == AlignmentDiagnostic.USABLE
    assert result.correspondence_method == "superpoint_lightglue"
    assert result.as_metrics_dict()["alignment_fallback_attempted"] == 1
    assert result.as_metrics_dict()["ecc_converged"] == 1
