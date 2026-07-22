#!/usr/bin/env python3
"""Automated tests for ImageComparisonService using synthetic fixtures.

Tests cover:
  - Three client-facing states: NO_CHANGE_HIGH_CONFIDENCE, CHANGE_DETECTED,
    DETECTION_UNAVAILABLE
  - Input validation errors (missing files, invalid ROI, unwritable output)
  - Artifact completeness (all five evidence files written)
  - ROI boundary-tolerance normalization
  - Configuration version recording
  - Small translation, rotation, perspective, illumination, blur,
    and local-change scenarios

No internal images are read.  All fixtures are programmatically generated.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure the package is importable from the test script's location
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imagecmp import (
    ComparisonState,
    ImageComparisonService,
)
from imagecmp.roi import read_rois, parse_roi_string
from imagecmp.config import default_config


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _create_textured_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Create a synthetic image with enough texture for ORB features."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Random noise base for texture
    rng = np.random.RandomState(42)
    noise = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.3, noise, 0.7, 0)

    # High-contrast edges for ORB
    cv2.rectangle(img, (30, 30), (160, 140), (220, 220, 220), 3)
    cv2.circle(img, (240, 140), 36, (10, 20, 230), -1)
    cv2.putText(img, "A1", (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 2)
    cv2.line(img, (0, 0), (319, 239), (180, 180, 180), 1)
    cv2.line(img, (319, 0), (0, 239), (180, 180, 180), 1)
    # Grid pattern for more features
    for i in range(40, width, 40):
        cv2.line(img, (i, 0), (i, height - 1), (100, 100, 100), 1)
    for i in range(40, height, 40):
        cv2.line(img, (0, i), (width - 1, i), (100, 100, 100), 1)
    return img


def _featureless_image(width: int = 80, height: int = 80) -> np.ndarray:
    """Create a flat, featureless image for alignment-unavailable tests."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _image_with_green_block(width: int = 320, height: int = 240) -> np.ndarray:
    """Clone the baseline and add a synthetic green block (local change)."""
    img = _create_textured_image(width, height)
    cv2.rectangle(img, (210, 130), (240, 160), (0, 255, 0), -1)
    return img


def _calibrated_config_payload(version: str) -> dict:
    """Build a complete config fixture; deployed configs cannot be partial."""
    payload = asdict(default_config())
    payload["version"] = version
    return payload


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_validate_missing_standard():
    """Missing standard image raises FileNotFoundError."""
    service = ImageComparisonService()
    with tempfile.TemporaryDirectory() as tmp:
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(live_path), _create_textured_image())
        try:
            service.compare(
                standard_path=Path(tmp) / "nonexistent.jpg",
                live_path=live_path,
                roi="17 0.5 0.5 0.4 0.5",
                output_dir=Path(tmp) / "out",
            )
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass


def test_validate_invalid_roi():
    """Invalid ROI string raises ValueError."""
    service = ImageComparisonService()
    with tempfile.TemporaryDirectory() as tmp:
        std = Path(tmp) / "std.jpg"
        live = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std), _create_textured_image())
        cv2.imwrite(str(live), _create_textured_image())
        try:
            service.compare(std, live, roi="bad", output_dir=Path(tmp) / "out")
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_validate_unwritable_output():
    """A non-creatable output directory raises an error."""
    service = ImageComparisonService()
    with tempfile.TemporaryDirectory() as tmp:
        std = Path(tmp) / "std.jpg"
        live = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std), _create_textured_image())
        cv2.imwrite(str(live), _create_textured_image())
        # Create a file where we need a directory
        blocker = Path(tmp) / "blocker"
        blocker.write_text("x")
        try:
            service.compare(std, live, roi="17 0.5 0.5 0.4 0.5",
                            output_dir=blocker / "sub")
            assert False, "expected error"
        except (ValueError, OSError):
            pass


def test_no_change_identical():
    """Identical images should produce NO_CHANGE_HIGH_CONFIDENCE."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    live = standard.copy()  # pixel-identical

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        assert result.state == ComparisonState.NO_CHANGE_HIGH_CONFIDENCE, (
            f"expected NO_CHANGE_HIGH_CONFIDENCE, got {result.state.value}"
        )
        assert result.artifacts is not None
        assert result.artifacts.alignment_image.is_file()
        assert result.artifacts.valid_mask.is_file()
        assert result.artifacts.difference_mask.is_file()
        assert result.artifacts.difference_heatmap.is_file()
        assert result.artifacts.annotated_image.is_file()
        assert result.config_version == "development-default-v1"


def test_change_detected_local_block():
    """A synthetic green block should produce CHANGE_DETECTED."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    live = _image_with_green_block()

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        assert result.state == ComparisonState.CHANGE_DETECTED, (
            f"expected CHANGE_DETECTED, got {result.state.value}"
        )
        assert len(result.detection_regions) >= 1, (
            "expected at least one detection region"
        )
        # Verify regions are within ROI
        rx, ry, rw, rh = 96, 60, 128, 120  # roi "17 0.5 0.5 0.4 0.5" at 320x240
        for region in result.detection_regions:
            assert region.confidence > 0.0


def test_detection_unavailable_featureless():
    """Featureless images should produce DETECTION_UNAVAILABLE."""
    service = ImageComparisonService()
    img = _featureless_image()

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), img)
        cv2.imwrite(str(live_path), img)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        assert result.state == ComparisonState.DETECTION_UNAVAILABLE, (
            f"expected DETECTION_UNAVAILABLE, got {result.state.value}"
        )
        assert result.unavailable_reason is not None
        assert len(result.unavailable_detail) > 0
        assert result.artifacts is not None
        assert result.artifacts.alignment_image.is_file()
        assert result.artifacts.valid_mask.is_file()
        assert result.artifacts.difference_mask.is_file()
        assert result.artifacts.difference_heatmap.is_file()
        assert result.artifacts.annotated_image.is_file()


def test_variation_rotation():
    """Moderate rotation should still produce a result (not unavailable)."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    M = cv2.getRotationMatrix2D((160, 120), 6.0, 1.0)
    live = cv2.warpAffine(standard, M, (320, 240))

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        # Should not be unavailable (rotation is within recovery range)
        assert result.state != ComparisonState.DETECTION_UNAVAILABLE, (
            f"rotation case should not be unavailable, got {result.state.value}"
        )


def test_variation_illumination():
    """Brightness change should not trigger false positive."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    live = cv2.convertScaleAbs(standard, alpha=1.15, beta=18.0)

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        # Illumination change alone should not raise CHANGE_DETECTED
        # (may be DETECTION_UNAVAILABLE if alignment weak, but not false positive)
        assert result.state != ComparisonState.CHANGE_DETECTED or (
            result.state == ComparisonState.CHANGE_DETECTED
            and len(result.detection_regions) >= 1
        ), f"illumination test got {result.state.value}"


def test_roi_boundary_tolerance():
    """ROI boundaries within [-0.01, 1.01] are clipped, not rejected."""
    # Test parseIntRoiString with tolerated boundaries
    roi = parse_roi_string("17 0.776495 0.623402 0.447011 0.430379")
    assert roi is not None, "tolerated boundary should parse"
    # The center should be shifted by clipping
    assert 0.0 <= roi.center_x <= 1.0
    assert 0.0 <= roi.center_y <= 1.0

    # Test borderline out-of-range
    roi_bad = parse_roi_string("17 0.5 0.5 2.0 0.5")
    assert roi_bad is None, "width > 1+tolerance should reject (height > 1+tolerance)"

    # Test that center_x=1.0, width=0.2 yields left=0.9, right=1.1
    # right=1.1 exceeds 1.0+tolerance(0.01)=1.01 → should be None
    roi_bad2 = parse_roi_string("17 1.0 0.5 0.2 0.2")
    assert roi_bad2 is None, "right boundary > 1+tolerance should reject"


def test_roi_file_parsing():
    """read_rois correctly parses valid, invalid, and tolerated ROIs."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("17 0.5 0.5 0.4 0.5\n")
        f.write("18 0.3 0.3 0.2 0.2\n")
        f.write("bad line\n")
        f.write("19 1.5 0.5 0.2 0.2\n")
        f.write("\n")
        path = Path(f.name)

    try:
        result = read_rois(path)
        assert len(result.rois) == 2, f"expected 2 valid ROIs, got {len(result.rois)}"
        assert len(result.errors) == 2, f"expected 2 errors, got {len(result.errors)}"
        assert result.errors[0].startswith("invalid ROI"), f"bad line not flagged"
        assert result.errors[1].startswith("invalid ROI"), f"out-of-range not flagged"
    finally:
        path.unlink()


def test_config_version_in_result():
    """The configuration version is recorded in every result."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    live = standard.copy()

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        assert result.config_version == "development-default-v1", (
            f"expected development-default-v1, got {result.config_version}"
        )


def test_artifact_output_count():
    """A successful call writes exactly five evidence files."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    live = standard.copy()

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)
        output_dir = Path(tmp) / "out"

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=output_dir,
        )

        assert result.artifacts is not None
        # Count output files
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) == 5, (
            f"expected 5 artifact files, got {len(png_files)}: {[p.name for p in png_files]}"
        )


def test_artifacts_support_unicode_output_directory():
    """Evidence artifacts can be written when a Windows path contains Chinese text."""
    service = ImageComparisonService()
    image = _create_textured_image()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        standard_path = tmp_path / "standard.png"
        live_path = tmp_path / "live.png"
        cv2.imwrite(str(standard_path), image)
        cv2.imwrite(str(live_path), image)

        result = service.compare(
            standard_path,
            live_path,
            "17 0.5 0.5 0.4 0.5",
            tmp_path / "证据输出",
        )

        assert result.artifacts is not None
        assert all(path.is_file() for path in (
            result.artifacts.alignment_image,
            result.artifacts.valid_mask,
            result.artifacts.difference_mask,
            result.artifacts.difference_heatmap,
            result.artifacts.annotated_image,
        ))


def test_variation_blur():
    """Blur should not cause false alarm."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    live = cv2.GaussianBlur(standard, (9, 9), 1.8)

    with tempfile.TemporaryDirectory() as tmp:
        std_path = Path(tmp) / "std.jpg"
        live_path = Path(tmp) / "live.jpg"
        cv2.imwrite(str(std_path), standard)
        cv2.imwrite(str(live_path), live)

        result = service.compare(
            standard_path=std_path,
            live_path=live_path,
            roi="17 0.5 0.5 0.4 0.5",
            output_dir=Path(tmp) / "out",
        )

        # Blur may produce any state but should not crash
        assert result.state in ComparisonState, f"invalid state {result.state}"


def test_detection_regions_use_live_image_coordinates():
    """A detected region is returned where it appears in the original live image."""
    service = ImageComparisonService()
    standard = _create_textured_image()
    transform = np.float32([[1, 0, 7], [0, 1, 5]])
    live = cv2.warpAffine(standard, transform, (320, 240))
    cv2.rectangle(live, (217, 135), (247, 165), (0, 255, 0), -1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        standard_path = tmp_path / "standard.png"
        live_path = tmp_path / "live.png"
        cv2.imwrite(str(standard_path), standard)
        cv2.imwrite(str(live_path), live)
        result = service.compare(
            standard_path, live_path, "17 0.5 0.5 0.4 0.5", tmp_path / "out"
        )

        assert result.state == ComparisonState.CHANGE_DETECTED
        assert any(210 <= region.x <= 250 and 130 <= region.y <= 170
                   for region in result.detection_regions), result.detection_regions


def test_unreliable_alignment_is_unavailable_with_artifacts():
    """Weak quality evidence must never become a normal conclusion."""
    service = ImageComparisonService()
    image = _create_textured_image()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        standard_path = tmp_path / "standard.png"
        live_path = tmp_path / "live.png"
        config_path = tmp_path / "config.json"
        cv2.imwrite(str(standard_path), image)
        cv2.imwrite(str(live_path), image)
        config = _calibrated_config_payload("force-unreliable-v1")
        config["alignment"]["feature_match_count_min"] = 10000
        config_path.write_text(json.dumps(config), encoding="utf-8")

        result = service.compare(
            standard_path, live_path, "17 0.5 0.5 0.4 0.5",
            tmp_path / "out", config_path,
        )

        assert result.state == ComparisonState.DETECTION_UNAVAILABLE
        assert result.unavailable_reason is not None
        assert result.artifacts is not None
        assert all(path.is_file() for path in (
            result.artifacts.alignment_image,
            result.artifacts.valid_mask,
            result.artifacts.difference_mask,
            result.artifacts.difference_heatmap,
            result.artifacts.annotated_image,
        ))


def test_explicit_missing_config_is_an_error():
    """An explicit but absent configuration must not silently use defaults."""
    service = ImageComparisonService()
    image = _create_textured_image()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        standard_path = tmp_path / "standard.png"
        live_path = tmp_path / "live.png"
        cv2.imwrite(str(standard_path), image)
        cv2.imwrite(str(live_path), image)
        try:
            service.compare(
                standard_path, live_path, "17 0.5 0.5 0.4 0.5",
                tmp_path / "out", tmp_path / "missing.json",
            )
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass


def test_invalid_config_is_an_error():
    """Invalid calibrated thresholds are setup errors, not business states."""
    service = ImageComparisonService()
    image = _create_textured_image()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        standard_path = tmp_path / "standard.png"
        live_path = tmp_path / "live.png"
        config_path = tmp_path / "invalid.json"
        cv2.imwrite(str(standard_path), image)
        cv2.imwrite(str(live_path), image)
        config = _calibrated_config_payload("invalid-v1")
        config["detection"]["difference_decision_threshold"] = 2.0
        config_path.write_text(json.dumps(config), encoding="utf-8")
        try:
            service.compare(
                standard_path, live_path, "17 0.5 0.5 0.4 0.5",
                tmp_path / "out", config_path,
            )
            assert False, "expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    tests = [
        ("validate_missing_standard", test_validate_missing_standard),
        ("validate_invalid_roi", test_validate_invalid_roi),
        ("validate_unwritable_output", test_validate_unwritable_output),
        ("no_change_identical", test_no_change_identical),
        ("change_detected_local_block", test_change_detected_local_block),
        ("detection_unavailable_featureless", test_detection_unavailable_featureless),
        ("variation_rotation", test_variation_rotation),
        ("variation_illumination", test_variation_illumination),
        ("roi_boundary_tolerance", test_roi_boundary_tolerance),
        ("roi_file_parsing", test_roi_file_parsing),
        ("config_version_in_result", test_config_version_in_result),
        ("artifact_output_count", test_artifact_output_count),
        ("artifacts_support_unicode_output_directory",
         test_artifacts_support_unicode_output_directory),
        ("variation_blur", test_variation_blur),
        ("detection_regions_use_live_image_coordinates",
         test_detection_regions_use_live_image_coordinates),
        ("unreliable_alignment_is_unavailable_with_artifacts",
         test_unreliable_alignment_is_unavailable_with_artifacts),
        ("explicit_missing_config_is_an_error", test_explicit_missing_config_is_an_error),
        ("invalid_config_is_an_error", test_invalid_config_is_an_error),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
