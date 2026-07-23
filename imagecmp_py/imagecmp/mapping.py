"""Expected-component mapping evidence and comparison eligibility.

This module is the seam between image-level alignment and difference
detection.  It turns a global standard-to-live geometric map into evidence
that the caller's *expected component* has a plausible live-image candidate,
enough valid pixels, consistent post-alignment appearance, and sufficient
native live-image resolution.  A failed check is deliberately an unavailable
comparison, never an implicit no-change result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CalibratedConfig
from .result import UnavailableReason


_NAN = float("nan")


@dataclass
class ComponentMappingEvidence:
    """Evidence required before an expected component may be compared.

    ``candidate_live_rect`` is only a candidate established by the global
    geometric map.  It becomes comparable only after every gate in this
    record has passed.  ``comparison_scale`` is the linear scale used by the
    difference module; it never exceeds one, so a coarse live component is
    never treated as though upsampling restored missing detail.
    """

    candidate_live_rect: Optional[tuple[int, int, int, int]] = None
    candidate_localized: bool = False
    candidate_in_frame_ratio: float = _NAN
    candidate_projected_area_pixels: float = _NAN

    roi_valid_overlap_ratio: float = _NAN
    appearance_ncc: float = _NAN
    appearance_ssim: float = _NAN
    appearance_histogram_ncc: float = _NAN
    appearance_local_contrast_ncc: float = _NAN
    appearance_gradient_ncc: float = _NAN
    appearance_mismatch_confirmed: bool = False

    effective_resolution_scale: float = _NAN
    effective_resolution_anisotropy: float = _NAN
    effective_live_width_pixels: float = _NAN
    effective_live_height_pixels: float = _NAN
    comparison_scale: float = 1.0

    failure_reason: Optional[UnavailableReason] = None
    failure_detail: str = ""
    stage_diagnostics: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.failure_reason is None

    def as_metrics_dict(self) -> dict:
        """Return serializable component-mapping evidence for result manifests."""
        fields = [
            ("candidate_localized", 1 if self.candidate_localized else 0),
            ("candidate_in_frame_ratio", self.candidate_in_frame_ratio),
            ("candidate_projected_area_pixels", self.candidate_projected_area_pixels),
            ("roi_valid_overlap_ratio", self.roi_valid_overlap_ratio),
            ("appearance_ncc", self.appearance_ncc),
            ("appearance_ssim", self.appearance_ssim),
            ("appearance_histogram_ncc", self.appearance_histogram_ncc),
            ("appearance_local_contrast_ncc", self.appearance_local_contrast_ncc),
            ("appearance_gradient_ncc", self.appearance_gradient_ncc),
            ("appearance_mismatch_confirmed", 1 if self.appearance_mismatch_confirmed else 0),
            ("effective_resolution_scale", self.effective_resolution_scale),
            ("effective_resolution_anisotropy", self.effective_resolution_anisotropy),
            ("effective_live_width_pixels", self.effective_live_width_pixels),
            ("effective_live_height_pixels", self.effective_live_height_pixels),
            ("comparison_processing_scale", self.comparison_scale),
            ("component_mapping_usable", 1 if self.usable else 0),
        ]
        metrics = {
            name: value
            for name, value in fields
            if isinstance(value, int) or math.isfinite(value)
        }
        metrics["component_mapping_stages"] = list(self.stage_diagnostics)
        if self.failure_reason is not None:
            metrics["component_mapping_failure_reason"] = self.failure_reason.value
        if self.failure_detail:
            metrics["component_mapping_failure_detail"] = self.failure_detail
        return metrics


def evaluate_component_mapping(
    standard_bgr: np.ndarray,
    aligned_live_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray],
    standard_to_live: np.ndarray,
    roi_rect: tuple[int, int, int, int],
    live_shape: tuple[int, int],
    config: CalibratedConfig,
) -> ComponentMappingEvidence:
    """Build the pre-difference evidence chain for one expected component.

    The function intentionally returns a structured failed record instead of
    raising for inadequate visual evidence.  Invalid call inputs remain
    programming/configuration errors, while ordinary image uncertainty is a
    client-visible ``DETECTION_UNAVAILABLE`` outcome selected by the caller.
    """
    evidence = ComponentMappingEvidence()
    th = config.alignment

    if standard_bgr.shape != aligned_live_bgr.shape:
        return _fail(
            evidence,
            UnavailableReason.ALIGNMENT_FAILED,
            "post-alignment image shape differs from the standard image",
        )
    if valid_mask is None or valid_mask.shape != standard_bgr.shape[:2]:
        return _fail(
            evidence,
            UnavailableReason.ALIGNMENT_FAILED,
            "valid comparison mask is unavailable for the expected component",
        )
    if standard_to_live.shape != (3, 3) or not np.all(np.isfinite(standard_to_live)):
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "no finite standard-to-live transform is available to localize the expected component",
        )

    x, y, width, height = roi_rect
    if width <= 0 or height <= 0:
        raise ValueError("expected-component ROI must have positive dimensions")

    evidence.stage_diagnostics.append("candidate_localization")
    projected = _project_rect_corners(roi_rect, standard_to_live)
    if projected is None:
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "the expected-component ROI could not be projected into the live image",
        )
    candidate = _candidate_from_projected_corners(projected, live_shape)
    if candidate is None:
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "the projected expected-component candidate has invalid geometry",
        )
    rect, in_frame_ratio, projected_area = candidate
    evidence.candidate_live_rect = rect
    evidence.candidate_in_frame_ratio = in_frame_ratio
    evidence.candidate_projected_area_pixels = projected_area
    evidence.candidate_localized = in_frame_ratio > 0.0
    if not evidence.candidate_localized:
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "the expected-component candidate falls outside the live-image frame",
        )
    if in_frame_ratio < th.candidate_in_frame_ratio_min:
        return _fail(
            evidence,
            UnavailableReason.ALIGNMENT_FAILED,
            "the expected-component candidate has insufficient in-frame coverage "
            f"({in_frame_ratio:.3f} < {th.candidate_in_frame_ratio_min:.3f})",
        )

    evidence.stage_diagnostics.append("valid_comparison_region")
    roi_mask = valid_mask[y:y + height, x:x + width]
    evidence.roi_valid_overlap_ratio = float(np.count_nonzero(roi_mask) / roi_mask.size)
    if evidence.roi_valid_overlap_ratio < th.roi_valid_overlap_ratio_min:
        return _fail(
            evidence,
            UnavailableReason.ALIGNMENT_FAILED,
            "ROI valid overlap below configured minimum "
            f"({evidence.roi_valid_overlap_ratio:.3f} < {th.roi_valid_overlap_ratio_min:.3f})",
        )

    evidence.stage_diagnostics.append("effective_resolution")
    resolution = _effective_resolution(standard_to_live, roi_rect, projected)
    if resolution is None:
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "the expected-component candidate has non-finite local sampling geometry",
        )
    (evidence.effective_resolution_scale,
     evidence.effective_resolution_anisotropy,
     evidence.effective_live_width_pixels,
     evidence.effective_live_height_pixels) = resolution
    evidence.comparison_scale = min(1.0, evidence.effective_resolution_scale)

    if evidence.effective_resolution_scale < th.effective_resolution_scale_min:
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "live expected-component resolution is too coarse for comparison "
            f"(scale {evidence.effective_resolution_scale:.3f} < "
            f"{th.effective_resolution_scale_min:.3f})",
        )
    if (evidence.effective_live_width_pixels < th.effective_component_width_min_pixels
            or evidence.effective_live_height_pixels < th.effective_component_height_min_pixels):
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "live expected-component resolution is too coarse for comparison "
            f"({evidence.effective_live_width_pixels:.1f}×"
            f"{evidence.effective_live_height_pixels:.1f} px; minimum "
            f"{th.effective_component_width_min_pixels}×"
            f"{th.effective_component_height_min_pixels} px)",
        )

    evidence.stage_diagnostics.append("post_alignment_appearance")
    appearance_mask = roi_mask > 0
    standard_gray = cv2.cvtColor(standard_bgr[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(aligned_live_bgr[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    evidence.appearance_ncc = _masked_ncc(standard_gray, live_gray, appearance_mask)
    evidence.appearance_ssim = _masked_ssim(standard_gray, live_gray, appearance_mask)
    if not math.isfinite(evidence.appearance_ncc) or not math.isfinite(evidence.appearance_ssim):
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "post-alignment appearance consistency could not be measured",
        )
    ncc_failed = evidence.appearance_ncc < th.appearance_ncc_min
    ssim_failed = evidence.appearance_ssim < th.appearance_ssim_min
    if not ncc_failed and not ssim_failed:
        return evidence

    # A low raw appearance score alone is not proof that the live image shows
    # a different component: illumination, contrast and exposure can cause it.
    # Confirm an image mismatch only when the global alignment and all earlier
    # component gates passed, the raw NCC failed, and every illumination-
    # resistant structural check also remains below the same NCC floor.
    evidence.stage_diagnostics.append("appearance_mismatch_confirmation")
    histogram_matched_live = _histogram_match(live_gray, standard_gray, appearance_mask)
    evidence.appearance_histogram_ncc = _masked_ncc(
        standard_gray, histogram_matched_live, appearance_mask
    )
    standard_local = _local_contrast(standard_gray)
    live_local = _local_contrast(live_gray)
    evidence.appearance_local_contrast_ncc = _masked_ncc(
        standard_local, live_local, appearance_mask
    )
    standard_gradient = _gradient_magnitude(standard_gray)
    live_gradient = _gradient_magnitude(live_gray)
    evidence.appearance_gradient_ncc = _masked_ncc(
        standard_gradient, live_gradient, appearance_mask
    )
    confirmation_scores = (
        evidence.appearance_histogram_ncc,
        evidence.appearance_local_contrast_ncc,
        evidence.appearance_gradient_ncc,
    )
    if ncc_failed and all(
        math.isfinite(score) and score < th.appearance_ncc_min
        for score in confirmation_scores
    ):
        evidence.appearance_mismatch_confirmed = True
        return _fail(
            evidence,
            UnavailableReason.MATCH_UNCERTAIN,
            "post-alignment appearance remains inconsistent after histogram, "
            "local-contrast, and edge checks; image mismatch is confirmed "
            f"(raw NCC {evidence.appearance_ncc:.3f}; "
            f"histogram {evidence.appearance_histogram_ncc:.3f}; "
            f"local contrast {evidence.appearance_local_contrast_ncc:.3f}; "
            f"edge {evidence.appearance_gradient_ncc:.3f})",
        )
    return _fail(
        evidence,
        UnavailableReason.MATCH_UNCERTAIN,
        "post-alignment appearance did not meet the raw NCC/SSIM gate, but "
        "illumination-resistant checks still retain structural similarity; "
        "the component mapping is not reliable enough for a conclusion",
    )


def _fail(
    evidence: ComponentMappingEvidence,
    reason: UnavailableReason,
    detail: str,
) -> ComponentMappingEvidence:
    evidence.failure_reason = reason
    evidence.failure_detail = detail
    evidence.stage_diagnostics.append("rejected")
    return evidence


def _project_rect_corners(
    rect: tuple[int, int, int, int], homography: np.ndarray
) -> Optional[np.ndarray]:
    x, y, width, height = rect
    corners = np.float32([
        [x, y], [x + width - 1, y],
        [x + width - 1, y + height - 1], [x, y + height - 1],
    ]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return None
    return projected.astype(np.float32)


def _candidate_from_projected_corners(
    projected: np.ndarray,
    live_shape: tuple[int, int],
) -> Optional[tuple[tuple[int, int, int, int], float, float]]:
    live_height, live_width = live_shape
    if live_height <= 0 or live_width <= 0 or not cv2.isContourConvex(projected):
        return None
    projected_area = abs(float(cv2.contourArea(projected)))
    if not math.isfinite(projected_area) or projected_area <= 1.0:
        return None

    frame = np.float32([
        [0, 0], [live_width - 1, 0],
        [live_width - 1, live_height - 1], [0, live_height - 1],
    ])
    intersection_area, _ = cv2.intersectConvexConvex(projected, frame)
    in_frame_ratio = max(0.0, min(1.0, float(intersection_area) / projected_area))
    if in_frame_ratio <= 0.0:
        return None

    x0 = max(0, int(math.floor(float(projected[:, 0].min()))))
    y0 = max(0, int(math.floor(float(projected[:, 1].min()))))
    x1 = min(live_width - 1, int(math.ceil(float(projected[:, 0].max()))))
    y1 = min(live_height - 1, int(math.ceil(float(projected[:, 1].max()))))
    if x1 < x0 or y1 < y0:
        return None
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), in_frame_ratio, projected_area


def _effective_resolution(
    homography: np.ndarray,
    roi_rect: tuple[int, int, int, int],
    projected: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    """Measure native live sampling density around the expected component."""
    x, y, width, height = roi_rect
    cx = x + (width - 1) / 2.0
    cy = y + (height - 1) / 2.0
    h00, h01, h02 = homography[0]
    h10, h11, h12 = homography[1]
    h20, h21, h22 = homography[2]
    denominator = h20 * cx + h21 * cy + h22
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-9:
        return None
    numerator_x = h00 * cx + h01 * cy + h02
    numerator_y = h10 * cx + h11 * cy + h12
    denominator_squared = denominator * denominator
    jacobian = np.array([
        [(h00 * denominator - numerator_x * h20) / denominator_squared,
         (h01 * denominator - numerator_x * h21) / denominator_squared],
        [(h10 * denominator - numerator_y * h20) / denominator_squared,
         (h11 * denominator - numerator_y * h21) / denominator_squared],
    ], dtype=np.float64)
    if not np.all(np.isfinite(jacobian)):
        return None
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if (singular_values.shape != (2,) or not np.all(np.isfinite(singular_values))
            or singular_values[1] <= 0.0):
        return None

    width_pixels = 0.5 * (
        float(np.linalg.norm(projected[1] - projected[0]))
        + float(np.linalg.norm(projected[2] - projected[3]))
    )
    height_pixels = 0.5 * (
        float(np.linalg.norm(projected[3] - projected[0]))
        + float(np.linalg.norm(projected[2] - projected[1]))
    )
    if not (math.isfinite(width_pixels) and math.isfinite(height_pixels)):
        return None
    return (
        float(singular_values[1]),
        float(singular_values[0] / singular_values[1]),
        width_pixels,
        height_pixels,
    )


def _masked_ncc(
    standard_gray: np.ndarray,
    live_gray: np.ndarray,
    mask: np.ndarray,
) -> float:
    values_standard = standard_gray[mask].astype(np.float64)
    values_live = live_gray[mask].astype(np.float64)
    if values_standard.size < 2:
        return _NAN
    centered_standard = values_standard - values_standard.mean()
    centered_live = values_live - values_live.mean()
    denominator = float(np.linalg.norm(centered_standard) * np.linalg.norm(centered_live))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(values_standard, values_live) else _NAN
    return float(np.dot(centered_standard, centered_live) / denominator)


def _histogram_match(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Map source brightness levels to the reference brightness distribution."""
    source_values = source_gray[mask].astype(np.uint8)
    reference_values = reference_gray[mask].astype(np.uint8)
    if source_values.size < 2 or reference_values.size < 2:
        return source_gray.copy()
    source_levels, source_counts = np.unique(source_values, return_counts=True)
    reference_levels, reference_counts = np.unique(reference_values, return_counts=True)
    source_quantiles = np.cumsum(source_counts, dtype=np.float64) / source_values.size
    reference_quantiles = np.cumsum(reference_counts, dtype=np.float64) / reference_values.size
    mapped_levels = np.interp(source_quantiles, reference_quantiles, reference_levels)
    lookup = np.arange(256, dtype=np.float32)
    lookup[source_levels] = mapped_levels.astype(np.float32)
    return cv2.LUT(source_gray, np.clip(np.rint(lookup), 0, 255).astype(np.uint8))


def _local_contrast(gray: np.ndarray) -> np.ndarray:
    """Remove slow brightness variation while retaining local texture."""
    image = gray.astype(np.float32)
    mean = cv2.GaussianBlur(image, (15, 15), 0)
    variance = cv2.GaussianBlur(image * image, (15, 15), 0) - mean * mean
    return (image - mean) / (np.sqrt(np.maximum(variance, 0.0)) + 1.0)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Return brightness-insensitive edge strength for structural comparison."""
    image = gray.astype(np.float32)
    horizontal = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(horizontal, vertical)


def _masked_ssim(
    standard_gray: np.ndarray,
    live_gray: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute local luminance/structure SSIM without an extra dependency."""
    if int(np.count_nonzero(mask)) < 2:
        return _NAN
    standard = standard_gray.astype(np.float32)
    live = live_gray.astype(np.float32)
    mu_standard = cv2.GaussianBlur(standard, (11, 11), 1.5)
    mu_live = cv2.GaussianBlur(live, (11, 11), 1.5)
    sigma_standard = cv2.GaussianBlur(standard * standard, (11, 11), 1.5) - mu_standard * mu_standard
    sigma_live = cv2.GaussianBlur(live * live, (11, 11), 1.5) - mu_live * mu_live
    covariance = cv2.GaussianBlur(standard * live, (11, 11), 1.5) - mu_standard * mu_live
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    denominator = (mu_standard * mu_standard + mu_live * mu_live + c1) * (sigma_standard + sigma_live + c2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ssim_map = ((2.0 * mu_standard * mu_live + c1) * (2.0 * covariance + c2)) / denominator
    values = ssim_map[mask]
    if values.size == 0 or not np.all(np.isfinite(values)):
        return _NAN
    return float(np.mean(values))
