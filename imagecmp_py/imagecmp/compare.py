"""Issue 8 colour and local-structure evidence on trusted aligned images.

The caller has already established a usable whole-image alignment and a
usable expected-component mapping.  This module therefore never changes an
alignment or mapping conclusion; it only records difference evidence inside
the valid component ROI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CalibratedConfig
from .result import DetectionRegion


@dataclass
class CompareResult:
    """Difference evidence, including raw and decision-eligible candidates."""

    difference_mask: np.ndarray
    difference_heatmap: np.ndarray
    valid_pixel_ratio: float
    detection_regions: list[DetectionRegion] = field(default_factory=list)
    decision_regions: list[DetectionRegion] = field(default_factory=list)
    comparison_quality_usable: bool = True
    comparison_quality_detail: str = ""
    evidence_metrics: dict[str, float | int] = field(default_factory=dict)


def _histogram_match_luminance(
    live_luminance: np.ndarray,
    standard_luminance: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Map the live luminance distribution to the standard distribution.

    Only valid ROI pixels contribute to either histogram.  The deterministic
    256-bin lookup is recorded indirectly through the input/output evidence
    metrics and avoids a dependency on an external image-processing package.
    """
    live_u8 = np.clip(np.rint(live_luminance * 255.0 / 100.0), 0, 255).astype(np.uint8)
    standard_u8 = np.clip(np.rint(standard_luminance * 255.0 / 100.0), 0, 255).astype(np.uint8)
    valid_bool = valid > 0.5
    if int(np.count_nonzero(valid_bool)) < 32:
        return live_luminance.copy()

    live_hist = np.bincount(live_u8[valid_bool], minlength=256).astype(np.float64)
    standard_hist = np.bincount(standard_u8[valid_bool], minlength=256).astype(np.float64)
    live_cdf = np.cumsum(live_hist) / max(float(live_hist.sum()), 1.0)
    standard_cdf = np.cumsum(standard_hist) / max(float(standard_hist.sum()), 1.0)
    lookup = np.searchsorted(standard_cdf, live_cdf, side="left").clip(0, 255).astype(np.uint8)
    matched_u8 = cv2.LUT(live_u8, lookup)
    return matched_u8.astype(np.float32) * (100.0 / 255.0)


def compare_aligned(
    standard_bgr: np.ndarray,
    aligned_live_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray],
    roi_rect: tuple[int, int, int, int],
    config: CalibratedConfig,
    processing_scale: float = 1.0,
) -> CompareResult:
    """Return colour and structure evidence for an already trusted component.

    All raw candidates remain in ``detection_regions``.  ``decision_regions``
    is the calibrated subset that may change the business conclusion; a tiny
    high-contrast speck therefore remains visible to an operator without
    becoming a standalone anomaly decision.
    """
    th = config.detection
    if not 0.0 < processing_scale <= 1.0:
        raise ValueError("processing_scale must be in (0, 1]")
    rx, ry, rw, rh = roi_rect
    image_height, image_width = standard_bgr.shape[:2]
    rx = max(0, rx)
    ry = max(0, ry)
    rw = min(rw, image_width - rx)
    rh = min(rh, image_height - ry)
    if rw <= 0 or rh <= 0:
        raise ValueError("ROI has no pixels inside the standard image")

    standard_roi = standard_bgr[ry:ry + rh, rx:rx + rw].astype(np.float32)
    live_roi = aligned_live_bgr[ry:ry + rh, rx:rx + rw].astype(np.float32)
    if valid_mask is None:
        valid = np.ones((rh, rw), dtype=np.float32)
    else:
        valid = (valid_mask[ry:ry + rh, rx:rx + rw] > 0).astype(np.float32)
    valid_pixel_ratio = float(np.mean(valid))

    process_width = max(1, int(round(rw * processing_scale)))
    process_height = max(1, int(round(rh * processing_scale)))
    scale_x = rw / process_width
    scale_y = rh / process_height
    if (process_width, process_height) != (rw, rh):
        standard_roi = cv2.resize(standard_roi, (process_width, process_height), interpolation=cv2.INTER_AREA)
        live_roi = cv2.resize(live_roi, (process_width, process_height), interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid, (process_width, process_height), interpolation=cv2.INTER_NEAREST)

    standard_lab = cv2.cvtColor((standard_roi / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)
    live_lab = cv2.cvtColor((live_roi / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)
    valid_bool = valid > 0.5
    standard_luma = standard_lab[..., 0].astype(np.float32) / 100.0
    live_luma = live_lab[..., 0].astype(np.float32) / 100.0
    if int(np.count_nonzero(valid_bool)) >= 32:
        standard_laplacian = np.abs(cv2.Laplacian(standard_luma, cv2.CV_32F))
        live_laplacian = np.abs(cv2.Laplacian(live_luma, cv2.CV_32F))
        standard_sharpness = float(np.mean(standard_laplacian[valid_bool]))
        live_sharpness = float(np.mean(live_laplacian[valid_bool]))
        sharpness_ratio = min(standard_sharpness, live_sharpness) / max(
            standard_sharpness, live_sharpness, 1e-6
        )
    else:
        standard_sharpness = 0.0
        live_sharpness = 0.0
        sharpness_ratio = 0.0
    comparison_quality_usable = (
        not th.image_quality_blur_gate_enabled
        or sharpness_ratio >= th.image_quality_sharpness_ratio_min
    )
    comparison_quality_detail = ""
    if not comparison_quality_usable:
        comparison_quality_detail = (
            "live component is too blurred for a reliable difference conclusion "
            f"(sharpness ratio {sharpness_ratio:.3f} < "
            f"{th.image_quality_sharpness_ratio_min:.3f})"
        )
    if int(np.count_nonzero(valid_bool)):
        luma_shift = float(abs(
            np.median(standard_lab[..., 0][valid_bool]) - np.median(live_lab[..., 0][valid_bool])
        ) * 255.0 / 100.0)
    else:
        luma_shift = 0.0
    illumination_strength = min(1.0, luma_shift / th.illumination_luma_shift_full_weight)
    colour_weight = max(
        th.illumination_color_weight_floor,
        1.0 - illumination_strength,
    )

    normalized_live_lab = live_lab.copy()
    if th.illumination_normalization_enabled:
        normalized_live_lab[..., 0] = _histogram_match_luminance(
            live_lab[..., 0], standard_lab[..., 0], valid
        )
    normalized_live_bgr = cv2.cvtColor(normalized_live_lab, cv2.COLOR_Lab2BGR) * 255.0

    # Colour evidence is deliberately calculated after luminance matching.
    lab_diff = np.sqrt(np.sum((standard_lab - normalized_live_lab) ** 2, axis=2))
    lab_diff[~valid_bool] = 0.0
    colour_score = np.clip(
        lab_diff / max(th.lab_delta_e_threshold, 1.0), 0.0, 1.0
    ) * colour_weight

    # Subtracting a broad local background removes gradual shadows and keeps
    # the sign of real edges.  A local z-score was deliberately not used:
    # under a strong exposure curve it can reverse a thin edge's sign and
    # falsely describe the same straight line as a structural change.
    #
    # Structure uses raw Lab L rather than the quantised histogram-matched L.
    # A monotonic exposure change retains edge direction; quantising it first
    # can erase an edge and create a false "missing edge" signal.
    standard_gray = standard_luma
    live_gray = live_luma
    local_kernel = th.local_illumination_kernel_size
    standard_structure = standard_gray - cv2.GaussianBlur(
        standard_gray, (local_kernel, local_kernel), 0
    )
    live_structure = live_gray - cv2.GaussianBlur(
        live_gray, (local_kernel, local_kernel), 0
    )
    standard_structure = cv2.GaussianBlur(standard_structure, (3, 3), 0.8)
    live_structure = cv2.GaussianBlur(live_structure, (3, 3), 0.8)
    standard_gx = cv2.Sobel(standard_structure, cv2.CV_32F, 1, 0, ksize=3)
    standard_gy = cv2.Sobel(standard_structure, cv2.CV_32F, 0, 1, ksize=3)
    live_gx = cv2.Sobel(live_structure, cv2.CV_32F, 1, 0, ksize=3)
    live_gy = cv2.Sobel(live_structure, cv2.CV_32F, 0, 1, ksize=3)
    standard_gradient = np.sqrt(standard_gx * standard_gx + standard_gy * standard_gy)
    live_gradient = np.sqrt(live_gx * live_gx + live_gy * live_gy)
    # A monotonic illumination change can alter edge magnitude substantially
    # while leaving the edge direction intact.  Direction disagreement is
    # therefore the structure signal; a missing/new edge naturally produces
    # a cosine near zero and remains visible.
    direction_cosine = (
        standard_gx * live_gx + standard_gy * live_gy
    ) / (standard_gradient * live_gradient + 1e-6)
    edge_strength = np.clip(
        np.maximum(standard_gradient, live_gradient)
        / th.local_structure_gradient_threshold,
        0.0,
        1.0,
    )
    structure_score = 0.5 * (1.0 - np.clip(direction_cosine, -1.0, 1.0)) * edge_strength
    structure_score[~valid_bool] = 0.0

    heatmap = np.maximum(colour_score, structure_score)
    binary = (heatmap >= th.difference_decision_threshold).astype(np.uint8) * 255
    binary[~valid_bool] = 0
    if th.morphology_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (th.morphology_kernel_size, th.morphology_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[DetectionRegion] = []
    decision_regions: list[DetectionRegion] = []
    small_count = 0
    for index in range(1, label_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        region_mask = labels == index
        raw_confidence = float(np.mean(heatmap[region_mask]))
        area_weight = min(1.0, math.sqrt(area / th.candidate_area_confidence_reference_pixels))
        confidence = raw_confidence * (
            th.candidate_area_confidence_floor
            + (1.0 - th.candidate_area_confidence_floor) * area_weight
        )
        channels: list[str] = []
        if float(np.mean(colour_score[region_mask])) >= th.difference_decision_threshold:
            channels.append("lab_colour_illumination_normalized")
        if float(np.mean(structure_score[region_mask])) >= th.difference_decision_threshold:
            channels.append("local_structure_gradient")
        below_decision_area = area < th.min_detection_area_pixels
        if below_decision_area:
            small_count += 1
            channels.append("small_candidate")
        if confidence < th.min_detection_confidence:
            channels.append("low_confidence_candidate")
        decision_eligible = (
            confidence >= th.min_detection_confidence and not below_decision_area
        )
        region = DetectionRegion(
            x=rx + int(math.floor(x * scale_x)),
            y=ry + int(math.floor(y * scale_y)),
            width=max(1, int(math.ceil(width * scale_x))),
            height=max(1, int(math.ceil(height * scale_y))),
            confidence=confidence,
            evidence_channels=channels,
            decision_eligible=decision_eligible,
        )
        regions.append(region)
        # Area is not an evidence-deletion rule: small candidates are returned
        # and rendered.  It is only a conservative business-decision guard so
        # that isolated texture specks cannot alert on their own.
        if decision_eligible:
            decision_regions.append(region)

    full_binary = np.zeros((image_height, image_width), dtype=np.uint8)
    full_binary[ry:ry + rh, rx:rx + rw] = cv2.resize(binary, (rw, rh), interpolation=cv2.INTER_NEAREST)
    full_heatmap = np.zeros((image_height, image_width), dtype=np.float32)
    full_heatmap[ry:ry + rh, rx:rx + rw] = cv2.resize(heatmap, (rw, rh), interpolation=cv2.INTER_LINEAR)

    return CompareResult(
        difference_mask=full_binary,
        difference_heatmap=full_heatmap,
        valid_pixel_ratio=valid_pixel_ratio,
        detection_regions=regions,
        decision_regions=decision_regions,
        comparison_quality_usable=comparison_quality_usable,
        comparison_quality_detail=comparison_quality_detail,
        evidence_metrics={
            "illumination_normalization_applied": int(th.illumination_normalization_enabled),
            "illumination_luma_median_shift": luma_shift,
            "illumination_strength": illumination_strength,
            "colour_weight": colour_weight,
            "comparison_processing_scale": processing_scale,
            "standard_laplacian_sharpness": standard_sharpness,
            "live_laplacian_sharpness": live_sharpness,
            "sharpness_ratio": sharpness_ratio,
            "comparison_quality_usable": int(comparison_quality_usable),
            "raw_candidate_count": len(regions),
            "decision_candidate_count": len(decision_regions),
            "small_candidate_count": small_count,
        },
    )
