"""Daily and calibration workflows for all components in one image.

One reference image is selected at image level from the case's primary and
additional references.  Every component then uses that same reference so the
image-level conclusion has one coherent coordinate system and audit trail.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .alignment import AlignmentDiagnostic, align
from .config import CalibratedConfig, default_config, load_config
from .references import CaseInput, ReferenceImage, discover_case_input, read_color_image
from .result import (
    CalibrationObservation,
    ComparisonState,
    ComponentConclusion,
    ImageComparisonResult,
    ReferenceAttempt,
)
from .roi import Roi, RoiResult, read_rois
from .service import ImageComparisonService


@dataclass(frozen=True)
class _ReferenceSelection:
    selected_reference: ReferenceImage
    attempts: list[ReferenceAttempt]


class MultiComponentImageComparisonService:
    """Run calibration or daily detection for all expected components."""

    def __init__(self, component_service: Optional[ImageComparisonService] = None) -> None:
        self._component_service = component_service or ImageComparisonService()

    def compare_daily(
        self,
        live_path: Path,
        references: Sequence[ReferenceImage],
        rois: Sequence[Roi],
        output_dir: Path,
        config_path: Optional[Path] = None,
    ) -> ImageComparisonResult:
        """Run daily detection with a mandatory calibrated configuration."""
        if config_path is None:
            raise ValueError(
                "daily detection requires a valid versioned calibration configuration"
            )
        config = load_config(Path(config_path), allow_development=False)
        live_path = Path(live_path)
        output_dir = _prepare_output_directory(output_dir)
        _validate_request_inputs(live_path, references, rois)

        selection = self._select_reference(live_path, references, config)
        conclusions: list[ComponentConclusion] = []
        for index, roi in enumerate(rois):
            result = self._component_service.compare(
                standard_path=selection.selected_reference.path,
                live_path=live_path,
                roi=_format_roi(roi),
                output_dir=output_dir / "components" / f"component-{index:03d}",
                config_path=Path(config_path),
            )
            conclusions.append(ComponentConclusion(
                component_index=index,
                category=roi.category,
                state=result.state,
                unavailable_reason=result.unavailable_reason,
                unavailable_detail=result.unavailable_detail,
                image_mismatch_detail=result.image_mismatch_detail,
                detection_regions=result.detection_regions,
                artifacts=result.artifacts,
                alignment_metrics=result.alignment_metrics,
                config_version=result.config_version,
            ))

        image_state = aggregate_component_conclusions(conclusions)
        manifest_path = output_dir / "daily_result.json"
        _write_json(manifest_path, {
            "mode": "daily_detection",
            "state": image_state.value,
            "config_version": config.version,
            "selected_reference": {
                "id": selection.selected_reference.reference_id,
                "path": str(selection.selected_reference.path),
            },
            "reference_attempts": [_reference_attempt_json(attempt) for attempt in selection.attempts],
            "components": [_component_conclusion_json(conclusion) for conclusion in conclusions],
        })
        return ImageComparisonResult(
            state=image_state,
            selected_reference_id=selection.selected_reference.reference_id,
            selected_reference_path=selection.selected_reference.path,
            reference_attempts=selection.attempts,
            component_conclusions=conclusions,
            manifest_path=manifest_path,
            config_version=config.version,
        )

    def compare_case_daily(
        self,
        case_directory: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ) -> ImageComparisonResult:
        """Run daily detection for a case using all its discovered references."""
        case = discover_case_input(case_directory)
        return self.compare_daily(
            live_path=case.live_path,
            references=case.references,
            rois=_read_valid_rois(case),
            output_dir=output_dir,
            config_path=config_path,
        )

    def calibrate(
        self,
        live_path: Path,
        references: Sequence[ReferenceImage],
        rois: Sequence[Roi],
        output_dir: Path,
        processing_config_path: Optional[Path] = None,
    ) -> CalibrationObservation:
        """Collect local observations without returning any business state."""
        config = (load_config(Path(processing_config_path))
                  if processing_config_path is not None else default_config())
        live_path = Path(live_path)
        output_dir = _prepare_output_directory(output_dir)
        _validate_request_inputs(live_path, references, rois)

        selection = self._select_reference(live_path, references, config)
        observations = []
        for index, roi in enumerate(rois):
            observation = self._component_service.observe(
                standard_path=selection.selected_reference.path,
                live_path=live_path,
                roi=_format_roi(roi),
                output_dir=output_dir / "components" / f"component-{index:03d}",
                processing_config_path=processing_config_path,
                component_index=index,
            )
            observations.append(observation)

        manifest_path = output_dir / "calibration_observation.json"
        _write_json(manifest_path, {
            "mode": "calibration",
            "processing_profile_version": config.version,
            "selected_reference": {
                "id": selection.selected_reference.reference_id,
                "path": str(selection.selected_reference.path),
            },
            "reference_attempts": [_reference_attempt_json(attempt) for attempt in selection.attempts],
            "components": [
                {
                    "component_index": item.component_index,
                    "category": item.category,
                    "alignment_metrics": item.alignment_metrics,
                    "difference_candidate_count": item.difference_candidate_count,
                    "difference_regions": [_region_json(region) for region in item.difference_regions],
                    "observation_detail": item.observation_detail,
                    "artifacts": _artifacts_json(item.artifacts),
                }
                for item in observations
            ],
        })
        return CalibrationObservation(
            selected_reference_id=selection.selected_reference.reference_id,
            selected_reference_path=selection.selected_reference.path,
            reference_attempts=selection.attempts,
            component_observations=observations,
            manifest_path=manifest_path,
            processing_profile_version=config.version,
        )

    def calibrate_case(
        self,
        case_directory: Path,
        output_dir: Path,
        processing_config_path: Optional[Path] = None,
    ) -> CalibrationObservation:
        """Collect calibration observations for one case and all its references."""
        case = discover_case_input(case_directory)
        return self.calibrate(
            live_path=case.live_path,
            references=case.references,
            rois=_read_valid_rois(case),
            output_dir=output_dir,
            processing_config_path=processing_config_path,
        )

    def _select_reference(
        self,
        live_path: Path,
        references: Sequence[ReferenceImage],
        config: CalibratedConfig,
    ) -> _ReferenceSelection:
        live = read_color_image(live_path)
        if live is None:
            raise RuntimeError(f"cannot decode live image: {live_path}")
        live_gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)

        evaluated: list[tuple[ReferenceImage, ReferenceAttempt, tuple[float, ...]]] = []
        for order, reference in enumerate(references):
            standard = read_color_image(reference.path)
            if standard is None:
                attempt = ReferenceAttempt(
                    reference_id=reference.reference_id,
                    reference_path=reference.path,
                    alignment_diagnostic="reference_unreadable",
                    alignment_metrics={},
                )
                evaluated.append((reference, attempt, (-1.0, -float(order))))
                continue
            alignment = align(cv2.cvtColor(standard, cv2.COLOR_BGR2GRAY), live_gray, config)
            attempt = ReferenceAttempt(
                reference_id=reference.reference_id,
                reference_path=reference.path,
                alignment_diagnostic=alignment.diagnostic.value,
                alignment_metrics=alignment.as_metrics_dict(),
            )
            evaluated.append((reference, attempt, _reference_score(alignment, order)))

        usable = [entry for entry in evaluated if entry[1].alignment_diagnostic != "reference_unreadable"]
        if not usable:
            raise RuntimeError("cannot decode any reference image")
        selected_index = max(range(len(evaluated)), key=lambda index: evaluated[index][2])
        selected_reference = evaluated[selected_index][0]
        attempts = [
            ReferenceAttempt(
                reference_id=attempt.reference_id,
                reference_path=attempt.reference_path,
                alignment_diagnostic=attempt.alignment_diagnostic,
                alignment_metrics=attempt.alignment_metrics,
                selected=index == selected_index,
            )
            for index, (_, attempt, _) in enumerate(evaluated)
        ]
        return _ReferenceSelection(selected_reference=selected_reference, attempts=attempts)


def aggregate_component_conclusions(
    conclusions: Sequence[ComponentConclusion],
) -> ComparisonState:
    """Aggregate safely: confirmed image mismatch wins, then change, unavailable, normal."""
    if not conclusions:
        raise ValueError("an image comparison requires at least one expected component")
    states = {conclusion.state for conclusion in conclusions}
    if ComparisonState.IMAGE_MISMATCH_DETECTED in states:
        return ComparisonState.IMAGE_MISMATCH_DETECTED
    if ComparisonState.CHANGE_DETECTED in states:
        return ComparisonState.CHANGE_DETECTED
    if ComparisonState.DETECTION_UNAVAILABLE in states:
        return ComparisonState.DETECTION_UNAVAILABLE
    return ComparisonState.NO_CHANGE_HIGH_CONFIDENCE


def _reference_score(alignment, order: int) -> tuple[float, ...]:
    diagnostic_rank = {
        AlignmentDiagnostic.USABLE: 3.0,
        AlignmentDiagnostic.UNRELIABLE: 2.0,
        AlignmentDiagnostic.UNAVAILABLE: 1.0,
    }[alignment.diagnostic]
    metrics = alignment.as_metrics_dict()
    return (
        diagnostic_rank,
        _finite_metric(metrics.get("valid_overlap_ratio")),
        float(metrics.get("inlier_count", 0)),
        _finite_metric(metrics.get("inlier_rate")),
        -_finite_metric(metrics.get("reprojection_error_pixels"), default=1e12),
        _finite_metric(metrics.get("spatial_coverage")),
        -float(order),
    )


def _finite_metric(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    return numeric if math.isfinite(numeric) else default


def _prepare_output_directory(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create output directory {output_dir}: {exc}") from exc
    if not output_dir.is_dir():
        raise ValueError(f"output location is not a directory: {output_dir}")
    return output_dir


def _validate_request_inputs(
    live_path: Path,
    references: Sequence[ReferenceImage],
    rois: Sequence[Roi],
) -> None:
    if not live_path.is_file():
        raise FileNotFoundError(f"live image not found: {live_path}")
    if not references:
        raise ValueError("at least one reference image is required")
    for reference in references:
        if not reference.path.is_file():
            raise FileNotFoundError(f"reference image not found: {reference.path}")
    if not rois:
        raise ValueError("at least one expected component ROI is required")


def _read_valid_rois(case: CaseInput) -> list[Roi]:
    roi_result: RoiResult = read_rois(case.roi_path)
    if roi_result.errors:
        raise ValueError(
            f"invalid ROI file {case.roi_path}: {'; '.join(roi_result.errors)}"
        )
    if not roi_result.rois:
        raise ValueError(f"ROI file contains no expected component: {case.roi_path}")
    return roi_result.rois


def _format_roi(roi: Roi) -> str:
    return (
        f"{roi.category} {roi.center_x:.12g} {roi.center_y:.12g} "
        f"{roi.width:.12g} {roi.height:.12g}"
    )


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot write result manifest {path}: {exc}") from exc


def _reference_attempt_json(attempt: ReferenceAttempt) -> dict:
    return {
        "reference_id": attempt.reference_id,
        "reference_path": str(attempt.reference_path),
        "alignment_diagnostic": attempt.alignment_diagnostic,
        "alignment_metrics": attempt.alignment_metrics,
        "selected": attempt.selected,
    }


def _component_conclusion_json(conclusion: ComponentConclusion) -> dict:
    return {
        "component_index": conclusion.component_index,
        "category": conclusion.category,
        "state": conclusion.state.value,
        "unavailable_reason": (
            conclusion.unavailable_reason.value if conclusion.unavailable_reason else None
        ),
        "unavailable_detail": conclusion.unavailable_detail,
        "image_mismatch_detail": conclusion.image_mismatch_detail,
        "detection_regions": [_region_json(region) for region in conclusion.detection_regions],
        "artifacts": _artifacts_json(conclusion.artifacts),
        "alignment_metrics": conclusion.alignment_metrics,
        "config_version": conclusion.config_version,
    }


def _region_json(region) -> dict:
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
        "confidence": region.confidence,
        "evidence_channels": region.evidence_channels,
        "decision_eligible": region.decision_eligible,
    }


def _artifacts_json(artifacts) -> Optional[dict]:
    if artifacts is None:
        return None
    return {
        "alignment_image": str(artifacts.alignment_image),
        "valid_mask": str(artifacts.valid_mask),
        "difference_mask": str(artifacts.difference_mask),
        "difference_heatmap": str(artifacts.difference_heatmap),
        "annotated_image": str(artifacts.annotated_image),
    }
