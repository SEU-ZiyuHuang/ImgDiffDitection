"""The public single-component comparison service.

Daily detection requires a calibrated configuration.  Calibration callers use
``observe`` instead, which returns measurements and evidence without exposing
a normal/anomaly business conclusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .alignment import AlignmentDiagnostic, AlignmentResult, align, clear_alignment_cache
from .artifacts import write_all_artifacts
from .compare import CompareResult, compare_aligned
from .config import CalibratedConfig, default_config, load_config
from .mapping import ComponentMappingEvidence, evaluate_component_mapping
from .references import read_color_image
from .result import (
    ArtifactSet,
    CalibrationComponentObservation,
    ComparisonResult,
    ComparisonState,
    DetectionRegion,
    UnavailableReason,
)
from .roi import parse_roi_string, roi_to_pixel_rect


class ImageComparisonService:
    """Compare one caller-specified expected component."""

    def __init__(self) -> None:
        # Reuse one alignment within this service instance, but never carry
        # evidence across independently created services.
        clear_alignment_cache()

    def compare(
        self,
        standard_path: Path,
        live_path: Path,
        roi: str,
        output_dir: Path,
        config_path: Optional[Path] = None,
        live_roi_str: Optional[str] = None,
    ) -> ComparisonResult:
        """Run daily detection for one component.

        A valid, versioned configuration is mandatory.  This prevents a
        caller from accidentally using development defaults to issue a normal
        or anomaly business conclusion.
        """
        if config_path is None:
            raise ValueError(
                "daily detection requires a valid versioned calibration configuration"
            )
        return self._compare_with_config(
            standard_path=standard_path,
            live_path=live_path,
            roi=roi,
            output_dir=output_dir,
            config=load_config(Path(config_path), allow_development=False),
            live_roi_str=live_roi_str,
        )

    def observe(
        self,
        standard_path: Path,
        live_path: Path,
        roi: str,
        output_dir: Path,
        processing_config_path: Optional[Path] = None,
        component_index: int = 0,
    ) -> CalibrationComponentObservation:
        """Collect one component's calibration observations and evidence.

        The returned object deliberately has no ``ComparisonState`` field.
        A caller may inspect alignment metrics and difference candidates, but
        must not mistake them for a normal or anomaly business conclusion.
        """
        parsed_roi = parse_roi_string(roi)
        if parsed_roi is None:
            raise ValueError(
                f"invalid ROI string: {roi!r}; expected 'class_id center_x center_y width height'"
            )
        config = (load_config(Path(processing_config_path))
                  if processing_config_path is not None else default_config())
        result = self._compare_with_config(
            standard_path=standard_path,
            live_path=live_path,
            roi=roi,
            output_dir=output_dir,
            config=config,
        )
        detail = (
            result.image_mismatch_detail
            or result.unavailable_detail
            or "alignment and difference observations collected"
        )
        return CalibrationComponentObservation(
            component_index=component_index,
            category=parsed_roi.category,
            alignment_metrics=result.alignment_metrics,
            difference_candidate_count=len(result.detection_regions),
            difference_regions=result.detection_regions,
            observation_detail=detail,
            artifacts=result.artifacts,
        )

    def _compare_with_config(
        self,
        standard_path: Path,
        live_path: Path,
        roi: str,
        output_dir: Path,
        config: CalibratedConfig,
        live_roi_str: Optional[str] = None,
    ) -> ComparisonResult:
        """Perform the shared image-comparison work with an explicit config."""
        standard_path = Path(standard_path)
        live_path = Path(live_path)
        output_dir = Path(output_dir)
        if not standard_path.is_file():
            raise FileNotFoundError(f"standard image not found: {standard_path}")
        if not live_path.is_file():
            raise FileNotFoundError(f"live image not found: {live_path}")

        parsed_roi = parse_roi_string(roi)
        if parsed_roi is None:
            raise ValueError(
                f"invalid ROI string: {roi!r}; expected 'class_id center_x center_y width height'"
            )
        parsed_live_roi = None
        if live_roi_str is not None:
            parsed_live_roi = parse_roi_string(live_roi_str)
            if parsed_live_roi is None:
                raise ValueError(f"invalid live ROI string: {live_roi_str!r}")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"cannot create output directory {output_dir}: {exc}") from exc
        if not output_dir.is_dir():
            raise ValueError(f"output location is not a directory: {output_dir}")

        standard = read_color_image(standard_path)
        live = read_color_image(live_path)
        if standard is None:
            raise RuntimeError(f"cannot decode standard image: {standard_path}")
        if live is None:
            raise RuntimeError(f"cannot decode live image: {live_path}")

        standard_gray = cv2.cvtColor(standard, cv2.COLOR_BGR2GRAY)
        live_gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
        standard_rect = roi_to_pixel_rect(parsed_roi, standard.shape[1], standard.shape[0])
        _validate_pixel_rect(standard_rect, standard.shape[1], standard.shape[0], "standard ROI")
        supplied_live_rect = (
            roi_to_pixel_rect(parsed_live_roi, live.shape[1], live.shape[0])
            if parsed_live_roi is not None else None
        )
        if supplied_live_rect is not None:
            _validate_pixel_rect(supplied_live_rect, live.shape[1], live.shape[0], "live ROI")

        alignment = align(standard_gray, live_gray, config)
        metrics = alignment.as_metrics_dict()
        standard_to_live = alignment.standard_to_live
        projected_live_rect = (
            _project_rect(standard_rect, standard_to_live, live.shape[:2])
            if standard_to_live is not None else supplied_live_rect
        )
        mapping: Optional[ComponentMappingEvidence] = None
        aligned_live: Optional[np.ndarray] = None
        if standard_to_live is not None:
            try:
                live_to_standard = np.linalg.inv(standard_to_live)
            except np.linalg.LinAlgError:
                detail = "final standard-to-live transform is singular"
                artifacts = _write_unavailable_artifacts(
                    standard, live, alignment, projected_live_rect, output_dir, detail
                )
                return ComparisonResult(
                    state=ComparisonState.DETECTION_UNAVAILABLE,
                    unavailable_reason=UnavailableReason.ALIGNMENT_FAILED,
                    unavailable_detail=detail,
                    artifacts=artifacts,
                    alignment_metrics=metrics,
                    config_version=config.version,
                )

            aligned_live = cv2.warpPerspective(
                live,
                live_to_standard,
                (standard.shape[1], standard.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            # Even when the image-level alignment later proves unreliable,
            # retain every component-level observation that the available map
            # permits.  This keeps unavailable results diagnosable.
            mapping = evaluate_component_mapping(
                standard_bgr=standard,
                aligned_live_bgr=aligned_live,
                valid_mask=alignment.valid_mask,
                standard_to_live=standard_to_live,
                roi_rect=standard_rect,
                live_shape=live.shape[:2],
                config=config,
            )
            metrics.update(mapping.as_metrics_dict())
            if mapping.candidate_live_rect is not None:
                projected_live_rect = mapping.candidate_live_rect

        if alignment.diagnostic != AlignmentDiagnostic.USABLE:
            detail = "; ".join(alignment.diagnostic_reasons) or "alignment quality is unavailable"
            artifacts = _write_unavailable_artifacts(
                standard=standard,
                live=live,
                alignment=alignment,
                live_roi_rect=projected_live_rect,
                output_dir=output_dir,
                detail=detail,
            )
            return ComparisonResult(
                state=ComparisonState.DETECTION_UNAVAILABLE,
                unavailable_reason=_unavailable_reason(alignment),
                unavailable_detail=detail,
                artifacts=artifacts,
                alignment_metrics=metrics,
                config_version=config.version,
            )

        if standard_to_live is None or aligned_live is None:
            detail = "alignment reported usable without a usable standard-to-live map"
            artifacts = _write_unavailable_artifacts(
                standard, live, alignment, projected_live_rect, output_dir, detail
            )
            return ComparisonResult(
                state=ComparisonState.DETECTION_UNAVAILABLE,
                unavailable_reason=UnavailableReason.ALIGNMENT_FAILED,
                unavailable_detail=detail,
                artifacts=artifacts,
                alignment_metrics=metrics,
                config_version=config.version,
            )

        if mapping is None or not mapping.usable:
            detail = (
                mapping.failure_detail
                if mapping is not None and mapping.failure_detail
                else "expected-component mapping evidence is unavailable"
            )
            reason = (
                mapping.failure_reason
                if mapping is not None and mapping.failure_reason is not None
                else UnavailableReason.MATCH_UNCERTAIN
            )
            artifacts = _write_unavailable_artifacts(
                standard, live, alignment, projected_live_rect, output_dir, detail
            )
            if mapping is not None and mapping.appearance_mismatch_confirmed:
                return ComparisonResult(
                    state=ComparisonState.IMAGE_MISMATCH_DETECTED,
                    image_mismatch_detail=detail,
                    artifacts=artifacts,
                    alignment_metrics=metrics,
                    config_version=config.version,
                )
            return ComparisonResult(
                state=ComparisonState.DETECTION_UNAVAILABLE,
                unavailable_reason=reason,
                unavailable_detail=detail,
                artifacts=artifacts,
                alignment_metrics=metrics,
                config_version=config.version,
            )

        comparison = compare_aligned(
            standard_bgr=standard,
            aligned_live_bgr=aligned_live,
            valid_mask=alignment.valid_mask,
            roi_rect=standard_rect,
            config=config,
            processing_scale=mapping.comparison_scale,
        )
        metrics["roi_valid_overlap_ratio"] = comparison.valid_pixel_ratio
        metrics.update(comparison.evidence_metrics)

        live_regions = _map_regions_to_live(
            comparison.detection_regions, standard_to_live, live.shape[:2]
        )
        live_decision_regions = _map_regions_to_live(
            comparison.decision_regions, standard_to_live, live.shape[:2]
        )
        if live_regions is None or live_decision_regions is None:
            detail = "a detected candidate could not be mapped into the live-image frame"
            artifacts = _write_artifacts(
                standard, live, standard_to_live, alignment, comparison,
                projected_live_rect, [], output_dir, detail,
            )
            return ComparisonResult(
                state=ComparisonState.DETECTION_UNAVAILABLE,
                unavailable_reason=UnavailableReason.ALIGNMENT_FAILED,
                unavailable_detail=detail,
                artifacts=artifacts,
                alignment_metrics=metrics,
                config_version=config.version,
            )

        artifacts = _write_artifacts(
            standard, live, standard_to_live, alignment, comparison,
            projected_live_rect, live_regions, output_dir, "",
        )
        if not comparison.comparison_quality_usable:
            return ComparisonResult(
                state=ComparisonState.DETECTION_UNAVAILABLE,
                unavailable_reason=UnavailableReason.IMAGE_QUALITY_INSUFFICIENT,
                unavailable_detail=comparison.comparison_quality_detail,
                detection_regions=live_regions,
                artifacts=artifacts,
                alignment_metrics=metrics,
                config_version=config.version,
            )
        state = (
            ComparisonState.CHANGE_DETECTED
            if live_decision_regions else ComparisonState.NO_CHANGE_HIGH_CONFIDENCE
        )
        return ComparisonResult(
            state=state,
            detection_regions=live_regions,
            artifacts=artifacts,
            alignment_metrics=metrics,
            config_version=config.version,
        )

    def compare_to_json(
        self,
        standard_path: Path,
        live_path: Path,
        roi: str,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ) -> str:
        """Run daily detection and serialize the public result to JSON."""
        return _result_to_json(
            self.compare(standard_path, live_path, roi, output_dir, config_path)
        )


def _validate_pixel_rect(
    rect: tuple[int, int, int, int], image_width: int, image_height: int, name: str
) -> None:
    x, y, width, height = rect
    if (width <= 0 or height <= 0 or x < 0 or y < 0
            or x + width > image_width or y + height > image_height):
        raise ValueError(f"{name} does not contain at least one in-bounds pixel")


def _project_rect(
    rect: tuple[int, int, int, int], homography: np.ndarray, live_shape: tuple[int, int]
) -> Optional[tuple[int, int, int, int]]:
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        return None
    x, y, width, height = rect
    corners = np.float32([
        [x, y], [x + width - 1, y],
        [x + width - 1, y + height - 1], [x, y + height - 1],
    ]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return None
    live_height, live_width = live_shape
    x0 = max(0, int(np.floor(projected[:, 0].min())))
    y0 = max(0, int(np.floor(projected[:, 1].min())))
    x1 = min(live_width - 1, int(np.ceil(projected[:, 0].max())))
    y1 = min(live_height - 1, int(np.ceil(projected[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _map_regions_to_live(
    standard_regions: list[DetectionRegion],
    homography: np.ndarray,
    live_shape: tuple[int, int],
) -> Optional[list[DetectionRegion]]:
    regions: list[DetectionRegion] = []
    for region in standard_regions:
        projected = _project_rect(
            (region.x, region.y, region.width, region.height), homography, live_shape
        )
        if projected is None:
            return None
        x, y, width, height = projected
        regions.append(DetectionRegion(
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=region.confidence,
            evidence_channels=region.evidence_channels,
            decision_eligible=region.decision_eligible,
        ))
    return regions


def _unavailable_reason(alignment: AlignmentResult) -> UnavailableReason:
    return (
        UnavailableReason.MATCH_UNCERTAIN
        if alignment.homography is None or alignment.suspected_zoom_parent_child_mismatch
        else UnavailableReason.ALIGNMENT_FAILED
    )


def _write_artifacts(
    standard: np.ndarray,
    live: np.ndarray,
    standard_to_live: Optional[np.ndarray],
    alignment: AlignmentResult,
    comparison: CompareResult,
    live_roi_rect: Optional[tuple[int, int, int, int]],
    detections: list[DetectionRegion],
    output_dir: Path,
    detail: str,
) -> ArtifactSet:
    try:
        return write_all_artifacts(
            standard_bgr=standard,
            live_bgr=live,
            H=standard_to_live,
            valid_mask=alignment.valid_mask,
            difference_mask=comparison.difference_mask,
            difference_heatmap=comparison.difference_heatmap,
            live_roi_rect=live_roi_rect,
            detections=detections,
            output_dir=output_dir,
            status_text=detail,
            inlier_standard_points=alignment.inlier_standard_points,
            inlier_live_points=alignment.inlier_live_points,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to write evidence artifacts: {exc}") from exc


def _write_unavailable_artifacts(
    standard: np.ndarray,
    live: np.ndarray,
    alignment: AlignmentResult,
    live_roi_rect: Optional[tuple[int, int, int, int]],
    output_dir: Path,
    detail: str,
) -> ArtifactSet:
    empty = CompareResult(
        difference_mask=np.zeros(standard.shape[:2], dtype=np.uint8),
        difference_heatmap=np.zeros(standard.shape[:2], dtype=np.float32),
        valid_pixel_ratio=0.0,
    )
    return _write_artifacts(
        standard, live, alignment.standard_to_live, alignment, empty,
        live_roi_rect, [], output_dir, detail,
    )


def _result_to_json(result: ComparisonResult) -> str:
    artifacts = result.artifacts
    obj = {
        "state": result.state.value,
        "unavailable_reason": (
            result.unavailable_reason.value if result.unavailable_reason else None
        ),
        "unavailable_detail": result.unavailable_detail,
        "image_mismatch_detail": result.image_mismatch_detail,
        "detection_regions": [
            {
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "confidence": region.confidence,
                "evidence_channels": region.evidence_channels,
                "decision_eligible": region.decision_eligible,
            }
            for region in result.detection_regions
        ],
        "artifacts": {
            "alignment_image": str(artifacts.alignment_image) if artifacts else None,
            "valid_mask": str(artifacts.valid_mask) if artifacts else None,
            "difference_mask": str(artifacts.difference_mask) if artifacts else None,
            "difference_heatmap": str(artifacts.difference_heatmap) if artifacts else None,
            "annotated_image": str(artifacts.annotated_image) if artifacts else None,
        },
        "alignment_metrics": result.alignment_metrics,
        "config_version": result.config_version,
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)
