"""Difference detection on aligned images.

P0 implementation uses two evidence channels:
  - Lab colour-space ΔE (CIE76)
  - Local gradient-magnitude difference

P2 will extend this to include edge-distance and perceptual-feature evidence,
with conservative OR-fusion across channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CalibratedConfig
from .result import DetectionRegion


@dataclass
class CompareResult:
    """Output of the difference-detection stage."""

    difference_mask: np.ndarray          # uint8 binary mask, 255 where change detected
    difference_heatmap: np.ndarray       # float32 heatmap (max across channels)
    valid_pixel_ratio: float
    detection_regions: list[DetectionRegion] = field(default_factory=list)


def compare_aligned(
    standard_bgr: np.ndarray,
    aligned_live_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray],
    roi_rect: tuple[int, int, int, int],
    config: CalibratedConfig,
) -> CompareResult:
    """Detect changes between aligned standard and live images.

    Args:
        standard_bgr: Standard (reference) image in BGR.
        aligned_live_bgr: Live image warped to standard coordinates (BGR).
        valid_mask: uint8 mask, 255 where pixels are valid for comparison,
                    or None to use the whole image.
        roi_rect: (x, y, w, h) in standard-image pixel coordinates.
        config: Loaded calibration configuration.

    Returns:
        CompareResult with the difference mask, heatmap, and detected regions.
    """
    th = config.detection
    rx, ry, rw, rh = roi_rect

    # Clamp ROI to image bounds
    H, W = standard_bgr.shape[:2]
    rx = max(0, rx)
    ry = max(0, ry)
    rw = min(rw, W - rx)
    rh = min(rh, H - ry)
    if rw <= 0 or rh <= 0:
        raise ValueError("ROI has no pixels inside the standard image")

    # Crop
    std_roi = standard_bgr[ry:ry + rh, rx:rx + rw].astype(np.float32)
    live_roi = aligned_live_bgr[ry:ry + rh, rx:rx + rw].astype(np.float32)

    if valid_mask is not None:
        mask_roi = valid_mask[ry:ry + rh, rx:rx + rw]
        valid = (mask_roi > 0).astype(np.float32)
    else:
        valid = np.ones((rh, rw), dtype=np.float32)
    valid_pixel_ratio = float(np.mean(valid))

    # ----- Channel 1: Lab ΔE (CIE76) ---------------------------------------
    std_lab = cv2.cvtColor((std_roi / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)
    live_lab = cv2.cvtColor((live_roi / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)
    lab_diff = np.sqrt(np.sum((std_lab - live_lab) ** 2, axis=2))
    lab_diff[valid < 0.5] = 0.0

    # Normalize to [0, 1] using threshold as scale reference
    lab_score = np.clip(lab_diff / max(th.lab_delta_e_threshold, 1.0), 0.0, 1.0)

    # ----- Channel 2: Gradient magnitude difference -------------------------
    # Keep grayscale values in the 0–255 range so the configured gradient
    # threshold has the same unit as Sobel magnitudes.  Dividing the images by
    # 255 here would make a threshold such as 30 effectively unreachable.
    std_gray = cv2.cvtColor(std_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    live_gray = cv2.cvtColor(live_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    std_gx = cv2.Sobel(std_gray, cv2.CV_32F, 1, 0, ksize=3)
    std_gy = cv2.Sobel(std_gray, cv2.CV_32F, 0, 1, ksize=3)
    live_gx = cv2.Sobel(live_gray, cv2.CV_32F, 1, 0, ksize=3)
    live_gy = cv2.Sobel(live_gray, cv2.CV_32F, 0, 1, ksize=3)

    std_grad = np.sqrt(std_gx ** 2 + std_gy ** 2)
    live_grad = np.sqrt(live_gx ** 2 + live_gy ** 2)
    grad_diff = np.abs(std_grad - live_grad)
    grad_diff[valid < 0.5] = 0.0

    grad_score = np.clip(
        grad_diff / max(th.gradient_magnitude_diff_threshold, 1.0), 0.0, 1.0
    )

    # ----- Fusion: max across channels (conservative OR) --------------------
    heatmap = np.maximum(lab_score, grad_score)

    # Threshold
    binary = (heatmap >= th.difference_decision_threshold).astype(np.uint8) * 255

    # Morphological cleaning
    # A size of one intentionally disables morphology.  This is the safe
    # default: a small candidate remains visible rather than being erased and
    # silently converted into a no-change conclusion.
    if th.morphology_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (th.morphology_kernel_size, th.morphology_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ----- Connected components → DetectionRegion ---------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    regions: list[DetectionRegion] = []
    for i in range(1, num_labels):  # skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Confidence = mean heatmap value within this region
        region_mask = (labels == i)
        confidence = float(np.mean(heatmap[region_mask]))

        # Map coordinates back to full standard-image frame
        evidence_channels = []
        mean_lab = float(np.mean(lab_score[region_mask]))
        mean_grad = float(np.mean(grad_score[region_mask]))
        if mean_lab > 0.5:
            evidence_channels.append("lab_color")
        if mean_grad > 0.5:
            evidence_channels.append("gradient_magnitude")
        if area < th.min_detection_area_pixels:
            evidence_channels.append("small_candidate")
        if confidence < th.min_detection_confidence:
            evidence_channels.append("low_confidence_candidate")

        regions.append(DetectionRegion(
            x=rx + x,
            y=ry + y,
            width=w,
            height=h,
            confidence=confidence,
            evidence_channels=evidence_channels,
        ))

    # Build full-size masks for artifact writing
    full_binary = np.zeros((H, W), dtype=np.uint8)
    if rh > 0 and rw > 0:
        full_binary[ry:ry + rh, rx:rx + rw] = binary

    full_heatmap = np.zeros((H, W), dtype=np.float32)
    if rh > 0 and rw > 0:
        full_heatmap[ry:ry + rh, rx:rx + rw] = heatmap

    return CompareResult(
        difference_mask=full_binary,
        difference_heatmap=full_heatmap,
        valid_pixel_ratio=valid_pixel_ratio,
        detection_regions=regions,
    )
