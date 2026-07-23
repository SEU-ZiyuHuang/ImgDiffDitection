"""SuperPoint + LightGlue correspondence adapter for Issue 8.0.

This module deliberately returns only image-coordinate correspondence evidence.
It does not estimate trust, locate a YOLO box, or decide an anomaly.  The
alignment module owns those decisions and applies the same geometry and ECC
gates to ORB and neural correspondences.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import AlignmentThresholds


@dataclass(frozen=True)
class CorrespondenceEvidence:
    """Original-image correspondence points and traceable model evidence."""

    standard_points: np.ndarray
    live_points: np.ndarray
    standard_keypoints: int
    live_keypoints: int
    accepted_match_count: int
    elapsed_ms: float
    detail: str = ""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_model_path(configured: str) -> Path:
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    root_candidate = _repository_root() / candidate
    if root_candidate.is_file():
        return root_candidate
    return candidate.resolve()


@lru_cache(maxsize=4)
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:  # hash validation above remains usable without ORT
        raise RuntimeError("onnxruntime is not installed for SuperPoint+LightGlue fallback") from exc
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _empty(detail: str, started: float) -> CorrespondenceEvidence:
    empty = np.empty((0, 2), dtype=np.float32)
    return CorrespondenceEvidence(empty, empty, 0, 0, 0, (time.perf_counter() - started) * 1000.0, detail)


def match_global_correspondences(
    standard_gray: np.ndarray,
    live_gray: np.ndarray,
    thresholds: AlignmentThresholds,
) -> CorrespondenceEvidence:
    """Run the registered model and restore matched points to image pixels.

    The registered ONNX asset receives `[2, 1, H, W]` grayscale values in
    `[0, 1]` and emits `keypoints`, `matches`, and `mscores`.  This adapter
    verifies those names and shapes before trusting the tensor contents.
    """
    started = time.perf_counter()
    model_path = _resolve_model_path(thresholds.superpoint_lightglue_model_path)
    if not model_path.is_file():
        return _empty(f"SuperPoint+LightGlue model is missing: {model_path}", started)
    actual_hash = _sha256(str(model_path))
    if actual_hash.lower() != thresholds.superpoint_lightglue_model_sha256.lower():
        return _empty(
            "SuperPoint+LightGlue model hash does not match the configured SHA-256",
            started,
        )
    try:
        session = _session(str(model_path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "images":
            return _empty("SuperPoint+LightGlue input contract is not images", started)
        if inputs[0].type != "tensor(float)":
            return _empty("SuperPoint+LightGlue input type is not float32", started)
        if len(outputs) < 3 or [item.name for item in outputs[:3]] != [
            "keypoints", "matches", "mscores"
        ]:
            return _empty("SuperPoint+LightGlue output contract differs from the registered model", started)
        if [item.type for item in outputs[:3]] != [
            "tensor(int64)", "tensor(int64)", "tensor(float)"
        ]:
            return _empty("SuperPoint+LightGlue output types differ from the registered model", started)
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 1:
            return _empty("SuperPoint+LightGlue input shape is not [batch,1,H,W]", started)
        if isinstance(shape[0], int) and shape[0] not in {0, 2}:
            return _empty("SuperPoint+LightGlue input does not accept a two-image batch", started)
        model_width = thresholds.superpoint_lightglue_input_width
        model_height = thresholds.superpoint_lightglue_input_height
        if model_height <= 0 or model_width <= 0:
            return _empty("SuperPoint+LightGlue input size is invalid", started)
        standard_resized = cv2.resize(standard_gray, (model_width, model_height), interpolation=cv2.INTER_AREA)
        live_resized = cv2.resize(live_gray, (model_width, model_height), interpolation=cv2.INTER_AREA)
        batch = np.stack([standard_resized, live_resized]).astype(np.float32) / 255.0
        batch = batch[:, np.newaxis, :, :]
        keypoints, matches, scores = session.run(
            ["keypoints", "matches", "mscores"], {"images": batch}
        )
    except Exception as exc:  # runtime errors become audited fallback evidence
        return _empty(f"SuperPoint+LightGlue inference failed ({type(exc).__name__})", started)

    if (not isinstance(keypoints, np.ndarray) or keypoints.ndim != 3
            or keypoints.shape[0] != 2 or keypoints.shape[2] != 2
            or keypoints.dtype != np.int64):
        return _empty("SuperPoint+LightGlue keypoint tensor shape is invalid", started)
    if (not isinstance(matches, np.ndarray) or matches.dtype != np.int64
            or matches.size % 3 != 0):
        return _empty("SuperPoint+LightGlue match tensor is invalid", started)
    if not isinstance(scores, np.ndarray) or scores.dtype != np.float32:
        return _empty("SuperPoint+LightGlue score tensor is invalid", started)

    keypoint_count = int(keypoints.shape[1])
    matches = matches.reshape(-1, 3)
    scores = scores.reshape(-1)
    standard_points: list[tuple[float, float]] = []
    live_points: list[tuple[float, float]] = []
    limit = min(len(matches), len(scores))
    standard_scale = (standard_gray.shape[1] / model_width, standard_gray.shape[0] / model_height)
    live_scale = (live_gray.shape[1] / model_width, live_gray.shape[0] / model_height)
    for match, score in zip(matches[:limit], scores[:limit]):
        if not np.isfinite(score) or float(score) < thresholds.superpoint_lightglue_match_score_min:
            continue
        batch_index, standard_index, live_index = (int(match[0]), int(match[1]), int(match[2]))
        if batch_index != 0 or standard_index < 0 or live_index < 0:
            continue
        if standard_index >= keypoint_count or live_index >= keypoint_count:
            continue
        standard_point = keypoints[0, standard_index]
        live_point = keypoints[1, live_index]
        if not (np.all(np.isfinite(standard_point)) and np.all(np.isfinite(live_point))):
            continue
        standard_points.append((
            float(standard_point[0]) * standard_scale[0],
            float(standard_point[1]) * standard_scale[1],
        ))
        live_points.append((
            float(live_point[0]) * live_scale[0],
            float(live_point[1]) * live_scale[1],
        ))
    return CorrespondenceEvidence(
        standard_points=np.asarray(standard_points, dtype=np.float32).reshape(-1, 2),
        live_points=np.asarray(live_points, dtype=np.float32).reshape(-1, 2),
        standard_keypoints=keypoint_count,
        live_keypoints=keypoint_count,
        accepted_match_count=len(standard_points),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
