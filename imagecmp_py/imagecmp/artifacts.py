"""Local evidence-artifact writer.

Every valid comparison call, including ``detection_unavailable``, emits the
same five named files.  This lets callers inspect why a conclusion was not
available instead of receiving an empty result that looks like a normal one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .result import ArtifactSet, DetectionRegion


def _write_image(path: Path, image: np.ndarray) -> Path:
    """Encode with OpenCV, then write bytes through Python's Unicode-safe path API."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"OpenCV could not encode {path.name}")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise OSError(f"cannot write {path}: {exc}") from exc
    return path


def write_alignment_image(
    standard_bgr: np.ndarray,
    live_bgr: np.ndarray,
    H: Optional[np.ndarray],
    output_dir: Path,
    status_text: str = "",
    filename: str = "alignment.png",
) -> Path:
    """Write an alignment diagnostic, with a clear fallback when no map exists."""
    standard_height, standard_width = standard_bgr.shape[:2]
    live_height, live_width = live_bgr.shape[:2]
    canvas = np.zeros((max(standard_height, live_height), standard_width + live_width, 3), dtype=np.uint8)
    canvas[:standard_height, :standard_width] = standard_bgr
    canvas[:live_height, standard_width:standard_width + live_width] = live_bgr

    if H is not None and H.shape == (3, 3) and np.all(np.isfinite(H)):
        corners = np.float32([
            [0, 0], [standard_width - 1, 0],
            [standard_width - 1, standard_height - 1], [0, standard_height - 1],
        ]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        for index in range(4):
            start = tuple(corners[index].ravel().astype(int))
            end = (int(round(projected[index, 0])) + standard_width,
                   int(round(projected[index, 1])))
            cv2.line(canvas, start, end, (0, 255, 0), 2)
    else:
        status_text = status_text or "Alignment unavailable: no trusted geometric map"

    if status_text:
        cv2.putText(
            canvas,
            status_text[:150],
            (10, max(20, canvas.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return _write_image(output_dir / filename, canvas)


def write_valid_mask(valid_mask: np.ndarray, output_dir: Path, filename: str = "valid_mask.png") -> Path:
    return _write_image(output_dir / filename, valid_mask)


def write_difference_mask(
    difference_mask: np.ndarray, output_dir: Path, filename: str = "difference_mask.png"
) -> Path:
    return _write_image(output_dir / filename, difference_mask)


def write_difference_heatmap(
    heatmap: np.ndarray, output_dir: Path, filename: str = "difference_heatmap.png"
) -> Path:
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    if h_max > h_min:
        scaled = ((heatmap - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    else:
        scaled = np.zeros_like(heatmap, dtype=np.uint8)
    return _write_image(output_dir / filename, cv2.applyColorMap(scaled, cv2.COLORMAP_JET))


def write_annotated_image(
    live_bgr: np.ndarray,
    roi_rect: Optional[tuple[int, int, int, int]],
    detections: list[DetectionRegion],
    output_dir: Path,
    status_text: str = "",
    filename: str = "annotated.png",
) -> Path:
    """Write the original live image annotated in its own coordinate frame."""
    annotated = live_bgr.copy()
    if roi_rect is not None:
        x, y, width, height = roi_rect
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)

    for det in detections:
        cv2.rectangle(
            annotated,
            (det.x, det.y),
            (det.x + det.width, det.y + det.height),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"{det.confidence:.2f}",
            (det.x, max(det.y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if status_text:
        cv2.putText(
            annotated,
            status_text[:150],
            (10, max(20, annotated.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return _write_image(output_dir / filename, annotated)


def write_all_artifacts(
    standard_bgr: np.ndarray,
    live_bgr: np.ndarray,
    H: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    difference_mask: Optional[np.ndarray],
    difference_heatmap: Optional[np.ndarray],
    live_roi_rect: Optional[tuple[int, int, int, int]],
    detections: list[DetectionRegion],
    output_dir: Path,
    status_text: str = "",
) -> ArtifactSet:
    """Write the complete evidence set, synthesising diagnostics where needed."""
    height, width = standard_bgr.shape[:2]
    valid_mask = valid_mask if valid_mask is not None else np.zeros((height, width), dtype=np.uint8)
    difference_mask = (difference_mask if difference_mask is not None
                       else np.zeros((height, width), dtype=np.uint8))
    difference_heatmap = (difference_heatmap if difference_heatmap is not None
                          else np.zeros((height, width), dtype=np.float32))
    return ArtifactSet(
        alignment_image=write_alignment_image(
            standard_bgr, live_bgr, H, output_dir, status_text=status_text
        ),
        valid_mask=write_valid_mask(valid_mask, output_dir),
        difference_mask=write_difference_mask(difference_mask, output_dir),
        difference_heatmap=write_difference_heatmap(difference_heatmap, output_dir),
        annotated_image=write_annotated_image(
            live_bgr, live_roi_rect, detections, output_dir, status_text=status_text
        ),
    )
