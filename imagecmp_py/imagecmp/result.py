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
    """A component-level conclusion, reserved for image-level aggregation."""

    state: ComparisonState
    unavailable_reason: Optional[UnavailableReason] = None
    unavailable_detail: str = ""
    detection_regions: list[DetectionRegion] = field(default_factory=list)
    alignment_metrics: dict = field(default_factory=dict)
