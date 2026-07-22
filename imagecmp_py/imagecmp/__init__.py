"""imagecmp — single-component image comparison for power-equipment anomaly detection.

Public API::

    from imagecmp import ImageComparisonService, ComparisonState, ComparisonResult
"""

from __future__ import annotations

from .result import (
    ArtifactSet,
    ComparisonResult,
    ComparisonState,
    ComponentConclusion,
    DetectionRegion,
    UnavailableReason,
)
from .service import ImageComparisonService

__all__ = [
    "ImageComparisonService",
    "ComparisonState",
    "ComparisonResult",
    "UnavailableReason",
    "DetectionRegion",
    "ArtifactSet",
    "ComponentConclusion",
]
