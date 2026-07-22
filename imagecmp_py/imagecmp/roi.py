"""YOLO ROI parsing with boundary-tolerance normalization.

Mirrors the P-1 C++ readRois logic exactly:
- Five-column format: class_id center_x center_y width height
- Boundary tolerance of ±0.01 around [0, 1]
- Clipping + coordinate reconstruction when tolerated
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_ROI_BOUNDARY_TOLERANCE = 0.01


@dataclass
class Roi:
    """A single parsed YOLO ROI with normalized coordinates."""

    category: str
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class RoiResult:
    """Result of parsing a YOLO ROI file."""

    rois: list[Roi] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    boundary_normalized_lines: int = 0


def _trim(value: str) -> str:
    return value.strip(" \t\r\n")


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def read_rois(path: Path) -> RoiResult:
    """Parse a YOLO ROI file with boundary-tolerance normalization.

    Each non-empty line must contain exactly five whitespace-separated fields:
        class_id  center_x  center_y  width  height

    Width and height must be positive.  Boundaries are allowed within
    [-0.01, 1.01]; if within tolerance but outside [0, 1], the boundary
    is clipped and the center/size are reconstructed from the clipped
    coordinates.  Boundaries outside the tolerance range are rejected.
    """
    result = RoiResult()

    if not path.is_file():
        result.errors.append("cannot read ROI file")
        return result

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result.errors.append(f"cannot read ROI file: {exc}")
        return result

    for line_number, raw in enumerate(lines, start=1):
        text = _trim(raw)
        if not text:
            continue

        fields = text.split()
        if len(fields) != 5:
            result.errors.append(f"invalid ROI at line {line_number}")
            continue

        try:
            category = fields[0]
            center_x = float(fields[1])
            center_y = float(fields[2])
            width = float(fields[3])
            height = float(fields[4])
        except ValueError:
            result.errors.append(f"invalid ROI at line {line_number}")
            continue

        if not _finite(center_x, center_y, width, height):
            result.errors.append(f"invalid ROI at line {line_number}")
            continue

        if width <= 0.0 or height <= 0.0:
            result.errors.append(f"invalid ROI at line {line_number}")
            continue

        left = center_x - width / 2.0
        right = center_x + width / 2.0
        top = center_y - height / 2.0
        bottom = center_y + height / 2.0

        if (left < -_ROI_BOUNDARY_TOLERANCE
                or right > 1.0 + _ROI_BOUNDARY_TOLERANCE
                or top < -_ROI_BOUNDARY_TOLERANCE
                or bottom > 1.0 + _ROI_BOUNDARY_TOLERANCE):
            result.errors.append(f"invalid ROI at line {line_number}")
            continue

        normalized_left = max(0.0, min(1.0, left))
        normalized_right = max(0.0, min(1.0, right))
        normalized_top = max(0.0, min(1.0, top))
        normalized_bottom = max(0.0, min(1.0, bottom))

        if (normalized_right <= normalized_left
                or normalized_bottom <= normalized_top):
            result.errors.append(f"invalid ROI at line {line_number}")
            continue

        if (normalized_left != left or normalized_right != right
                or normalized_top != top or normalized_bottom != bottom):
            result.warnings.append(
                f"ROI boundary normalized at line {line_number}"
            )
            result.boundary_normalized_lines += 1
            roi = Roi(
                category=category,
                center_x=(normalized_left + normalized_right) / 2.0,
                center_y=(normalized_top + normalized_bottom) / 2.0,
                width=normalized_right - normalized_left,
                height=normalized_bottom - normalized_top,
            )
        else:
            roi = Roi(
                category=category,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )

        result.rois.append(roi)

    if not result.rois and not result.errors:
        result.errors.append("ROI file contains no ROI")

    return result


def parse_roi_string(roi_str: str) -> Optional[Roi]:
    """Parse a single ROI from a command-line string like '17 0.5 0.5 0.4 0.5'.

    Applies the same boundary-tolerance rules as read_rois.
    Returns None if the string is not a valid single ROI.
    """
    fields = roi_str.strip().split()
    if len(fields) != 5:
        return None
    try:
        category = fields[0]
        center_x = float(fields[1])
        center_y = float(fields[2])
        width = float(fields[3])
        height = float(fields[4])
    except ValueError:
        return None
    if not _finite(center_x, center_y, width, height):
        return None
    if width <= 0.0 or height <= 0.0:
        return None

    # Boundary tolerance check (same as read_rois)
    left = center_x - width / 2.0
    right = center_x + width / 2.0
    top = center_y - height / 2.0
    bottom = center_y + height / 2.0
    if (left < -_ROI_BOUNDARY_TOLERANCE
            or right > 1.0 + _ROI_BOUNDARY_TOLERANCE
            or top < -_ROI_BOUNDARY_TOLERANCE
            or bottom > 1.0 + _ROI_BOUNDARY_TOLERANCE):
        return None

    # Clamp within [0,1] and reconstruct
    n_left = max(0.0, min(1.0, left))
    n_right = max(0.0, min(1.0, right))
    n_top = max(0.0, min(1.0, top))
    n_bottom = max(0.0, min(1.0, bottom))
    if n_right <= n_left or n_bottom <= n_top:
        return None

    return Roi(
        category=category,
        center_x=(n_left + n_right) / 2.0,
        center_y=(n_top + n_bottom) / 2.0,
        width=n_right - n_left,
        height=n_bottom - n_top,
    )


def roi_to_pixel_rect(roi: Roi, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Convert a normalised ROI to pixel coordinates.

    Returns (x, y, width, height) in integer pixels.
    """
    x = int(roi.center_x * image_width - roi.width * image_width / 2.0)
    y = int(roi.center_y * image_height - roi.height * image_height / 2.0)
    w = int(roi.width * image_width)
    h = int(roi.height * image_height)
    return x, y, w, h
