"""Public, explicit result contract for image comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ComparisonState(Enum):
    """The only client-facing comparison outcomes."""

    NO_CHANGE_HIGH_CONFIDENCE = "no_change_high_confidence"
    CHANGE_DETECTED = "change_detected"
    DETECTION_UNAVAILABLE = "detection_unavailable"


class UnavailableReason(Enum):
    """Why a valid request could not produce a reliable conclusion."""

    MATCH_UNCERTAIN = "match_uncertain"
    ALIGNMENT_FAILED = "alignment_failed"


class ComparisonMode(Enum):
    """Public operating modes.

    Calibration mode intentionally has no business-state result.  Daily
    detection mode is the only mode that may return ``ComparisonState``.
    """

    CALIBRATION = "calibration"
    DAILY_DETECTION = "daily_detection"


@dataclass(frozen=True)
class DetectionRegion:
    """A detected region in the original live-image pixel coordinate frame."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    evidence_channels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactSet:
    """Complete local evidence set written by every valid comparison call."""

    alignment_image: Path
    valid_mask: Path
    difference_mask: Path
    difference_heatmap: Path
    annotated_image: Path


@dataclass(frozen=True)
class ComparisonResult:
    """Structured output for one expected component comparison."""

    state: ComparisonState
    unavailable_reason: Optional[UnavailableReason] = None
    unavailable_detail: str = ""
    detection_regions: list[DetectionRegion] = field(default_factory=list)
    artifacts: Optional[ArtifactSet] = None
    alignment_metrics: dict = field(default_factory=dict)
    config_version: str = ""


@dataclass(frozen=True)
class ComponentConclusion:
    """A daily-detection conclusion for one expected component."""

    component_index: int
    category: str
    state: ComparisonState
    unavailable_reason: Optional[UnavailableReason] = None
    unavailable_detail: str = ""
    detection_regions: list[DetectionRegion] = field(default_factory=list)
    artifacts: Optional[ArtifactSet] = None
    alignment_metrics: dict = field(default_factory=dict)
    config_version: str = ""


@dataclass(frozen=True)
class ReferenceAttempt:
    """Alignment evidence for one reference-image candidate.

    The selected reference is chosen once at image level and is then used for
    every component.  This prevents a single image result from combining
    incompatible coordinate systems from different reference images.
    """

    reference_id: str
    reference_path: Path
    alignment_diagnostic: str
    alignment_metrics: dict = field(default_factory=dict)
    selected: bool = False


@dataclass(frozen=True)
class ImageComparisonResult:
    """Daily-detection result for all expected components in one image."""

    state: ComparisonState
    selected_reference_id: str
    selected_reference_path: Path
    reference_attempts: list[ReferenceAttempt]
    component_conclusions: list[ComponentConclusion]
    manifest_path: Path
    config_version: str


@dataclass(frozen=True)
class CalibrationComponentObservation:
    """Raw observation for one component; it is not a business conclusion."""

    component_index: int
    category: str
    alignment_metrics: dict = field(default_factory=dict)
    difference_candidate_count: int = 0
    difference_regions: list[DetectionRegion] = field(default_factory=list)
    observation_detail: str = ""
    artifacts: Optional[ArtifactSet] = None


@dataclass(frozen=True)
class CalibrationObservation:
    """Calibration-mode output with no normal/anomaly business state."""

    selected_reference_id: str
    selected_reference_path: Path
    reference_attempts: list[ReferenceAttempt]
    component_observations: list[CalibrationComponentObservation]
    manifest_path: Path
    processing_profile_version: str
