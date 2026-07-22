"""imagecmp — single-component image comparison for power-equipment anomaly detection.

Public API::

    from imagecmp import ImageComparisonService, ComparisonState, ComparisonResult
"""

from __future__ import annotations

from .result import (
    ArtifactSet,
    CalibrationComponentObservation,
    CalibrationObservation,
    ComparisonMode,
    ComparisonResult,
    ComparisonState,
    ComponentConclusion,
    DetectionRegion,
    ImageComparisonResult,
    ReferenceAttempt,
    UnavailableReason,
)
from .multi_component import MultiComponentImageComparisonService, aggregate_component_conclusions
from .references import CaseInput, ReferenceImage, discover_case_input
from .service import ImageComparisonService

__all__ = [
    "ImageComparisonService",
    "MultiComponentImageComparisonService",
    "ComparisonState",
    "ComparisonMode",
    "ComparisonResult",
    "ImageComparisonResult",
    "UnavailableReason",
    "DetectionRegion",
    "ArtifactSet",
    "ComponentConclusion",
    "CalibrationComponentObservation",
    "CalibrationObservation",
    "ReferenceAttempt",
    "ReferenceImage",
    "CaseInput",
    "discover_case_input",
    "aggregate_component_conclusions",
]
