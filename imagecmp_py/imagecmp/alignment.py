"""Alignment cascade: ORB feature matching → RANSAC homography → ECC refinement.

Mirrors the P-1 C++ analyzeFeatureAndAlignmentEvidence + finalizeAlignmentDiagnostic
logic exactly.  The alignment diagnostic classifies evidence into three tiers
(unavailable / unreliable / usable) for reporting, NOT for production decisions.

All thresholds come from the loaded CalibratedConfig, not hard-coded constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .config import CalibratedConfig

# ---------------------------------------------------------------------------
# Sentinel for unavailable numeric fields
# ---------------------------------------------------------------------------
_NAN = float("nan")


class AlignmentDiagnostic(Enum):
    """Three-tier evidence classification (same as P-1 C++)."""

    UNAVAILABLE = "unavailable"
    UNRELIABLE = "unreliable"
    USABLE = "usable"


@dataclass
class AlignmentResult:
    """Complete alignment evidence for a pair of images.

    The homography H maps standard-image coordinates to live-image
    coordinates (standard → live).  H_inv does the reverse.
    """

    # Core outputs
    homography: Optional[np.ndarray] = None          # initial 3×3 float64, standard → live
    homography_inv: Optional[np.ndarray] = None       # initial inverse, live → standard
    standard_to_live: Optional[np.ndarray] = None     # final map, including ECC when available
    aligned_live: Optional[np.ndarray] = None         # warped live in std coords
    valid_mask: Optional[np.ndarray] = None           # uint8, 255 where valid

    # Feature evidence
    standard_keypoints: int = 0
    live_keypoints: int = 0
    feature_match_count: int = 0
    inlier_count: int = 0
    inlier_rate: float = _NAN
    reprojection_error_pixels: float = _NAN
    spatial_coverage: float = _NAN

    # Projected geometry (computed only when H is finite and well-behaved)
    projected_geometry_valid: bool = False
    center_displacement_pixels: float = _NAN
    center_displacement_relative_diagonal: float = _NAN
    corner_displacement_median_pixels: float = _NAN
    projected_corners_in_live_frame: int = 0
    projected_area_ratio: float = _NAN

    # Valid overlap + ECC
    valid_overlap_available: bool = False
    valid_overlap_ratio: float = _NAN
    ecc_converged: bool = False
    ecc_correlation: float = _NAN

    # Diagnostic classification
    diagnostic: AlignmentDiagnostic = AlignmentDiagnostic.UNAVAILABLE
    diagnostic_reasons: list[str] = field(default_factory=list)

    def as_metrics_dict(self) -> dict:
        """Return observed alignment evidence as a flat dict.

        Only finite values are included; NaN fields are omitted.
        """
        fields = [
            ("standard_keypoints", self.standard_keypoints),
            ("live_keypoints", self.live_keypoints),
            ("feature_match_count", self.feature_match_count),
            ("inlier_count", self.inlier_count),
            ("inlier_rate", self.inlier_rate),
            ("reprojection_error_pixels", self.reprojection_error_pixels),
            ("spatial_coverage", self.spatial_coverage),
            ("center_displacement_pixels", self.center_displacement_pixels),
            ("center_displacement_relative_diagonal", self.center_displacement_relative_diagonal),
            ("corner_displacement_median_pixels", self.corner_displacement_median_pixels),
            ("projected_corners_in_live_frame", self.projected_corners_in_live_frame),
            ("projected_area_ratio", self.projected_area_ratio),
            ("valid_overlap_ratio", self.valid_overlap_ratio),
            ("ecc_converged", 1 if self.ecc_converged else 0),
            ("ecc_correlation", self.ecc_correlation),
            ("diagnostic", self.diagnostic.value),
        ]
        return {k: v for k, v in fields if isinstance(v, (int, str)) or math.isfinite(v)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_finite_homography(H: np.ndarray) -> bool:
    if H.shape != (3, 3) or H.dtype != np.float64:
        return False
    return bool(np.all(np.isfinite(H)))


def _median(values: list[float]) -> float:
    if not values:
        return _NAN
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


# ---------------------------------------------------------------------------
# Main alignment entry point
# ---------------------------------------------------------------------------

def align(
    standard_gray: np.ndarray,
    live_gray: np.ndarray,
    config: CalibratedConfig,
) -> AlignmentResult:
    """Run the full alignment cascade and return structured evidence.

    Args:
        standard_gray: Single-channel standard (reference) image.
        live_gray: Single-channel live (inspection) image.
        config: Loaded calibration configuration with alignment thresholds.

    Returns:
        AlignmentResult with homography, valid mask, metrics, and diagnostic.
        The result.aligned_live is the live image warped into standard
        coordinates via H_inv.  result.valid_mask marks pixels that came
        from the live image (not border fill).
    """
    result = AlignmentResult()
    th = config.alignment

    # ----- 1. ORB feature extraction ---------------------------------------
    orb = cv2.ORB.create(nfeatures=th.orb_feature_count)
    std_kp, std_desc = orb.detectAndCompute(standard_gray, None)
    live_kp, live_desc = orb.detectAndCompute(live_gray, None)

    result.standard_keypoints = len(std_kp)
    result.live_keypoints = len(live_kp)

    if std_desc is None or live_desc is None:
        result.diagnostic_reasons.append("no descriptors extracted")
        return result

    # ----- 2. Ratio-test matching ------------------------------------------
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    neighbours = bf.knnMatch(std_desc, live_desc, k=2)

    matches: list[cv2.DMatch] = []
    for pair in neighbours:
        if (len(pair) == 2
                and pair[0].distance < th.match_ratio_test_max * pair[1].distance):
            matches.append(pair[0])

    result.feature_match_count = len(matches)
    if len(matches) < 4:
        result.diagnostic_reasons.append(
            f"insufficient matches for homography ({len(matches)} < 4)"
        )
        return result

    # ----- 3. RANSAC homography --------------------------------------------
    src_pts = np.float32([std_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([live_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, th.ransac_reprojection_threshold_pixels
    )
    if H is None or H.size == 0 or inlier_mask is None:
        result.diagnostic_reasons.append("homography estimation returned empty")
        return result

    H = H.astype(np.float64)
    result.homography = H
    result.standard_to_live = H.copy()

    # ----- 4. Inlier evidence ----------------------------------------------
    inlier_src = src_pts[inlier_mask.ravel() == 1]
    inlier_dst = dst_pts[inlier_mask.ravel() == 1]
    result.inlier_count = len(inlier_src)
    result.inlier_rate = result.inlier_count / result.feature_match_count

    if len(inlier_src) > 0:
        projected_inliers = cv2.perspectiveTransform(
            inlier_src.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected_inliers - inlier_dst.reshape(-1, 2), axis=1)
        result.reprojection_error_pixels = float(np.mean(errors))

        if len(inlier_src) >= 3:
            hull = cv2.convexHull(inlier_src.reshape(-1, 2).astype(np.float32))
            result.spatial_coverage = cv2.contourArea(hull) / (
                standard_gray.shape[1] * standard_gray.shape[0]
            )

    # ----- 5. Projected geometry -------------------------------------------
    _compute_projected_geometry(result, standard_gray, live_gray)
    if not result.projected_geometry_valid:
        result.diagnostic_reasons.append("projected geometry is degenerate")
        return result

    # ----- 6. Valid overlap ------------------------------------------------
    H_inv = None
    if _is_finite_homography(H):
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            result.diagnostic_reasons.append("homography is singular")
    if H_inv is not None and np.all(np.isfinite(H_inv)):
        result.homography_inv = H_inv
        live_mask = np.full(live_gray.shape[:2], 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            live_mask, H_inv,
            (standard_gray.shape[1], standard_gray.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        result.valid_mask = valid_mask
        result.valid_overlap_available = True
        result.valid_overlap_ratio = float(
            cv2.countNonZero(valid_mask) / valid_mask.size
        )

        # Warp live into standard coordinates for downstream comparison
        aligned_live = cv2.warpPerspective(
            live_gray, H_inv,
            (standard_gray.shape[1], standard_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        result.aligned_live = aligned_live

        # ----- 7. ECC refinement -------------------------------------------
        # findTransformECC produces a map from the final template coordinates
        # to the current input (already homography-warped live) coordinates.
        # WARP_INVERSE_MAP therefore resamples the input into template space.
        try:
            ecc_warp = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                th.ecc_max_iterations,
                th.ecc_epsilon,
            )
            ecc_result = cv2.findTransformECC(
                standard_gray, aligned_live, ecc_warp,
                cv2.MOTION_AFFINE, criteria,
                inputMask=valid_mask,
            )
            # OpenCV 5.x returns (retval, warpMatrix); 4.x returns float.
            if isinstance(ecc_result, tuple):
                result.ecc_correlation = float(ecc_result[0])
            else:
                result.ecc_correlation = float(ecc_result)
            result.ecc_converged = True

            result.aligned_live = cv2.warpAffine(
                aligned_live,
                ecc_warp,
                (standard_gray.shape[1], standard_gray.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            result.valid_mask = cv2.warpAffine(
                valid_mask,
                ecc_warp,
                (standard_gray.shape[1], standard_gray.shape[0]),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            result.valid_overlap_ratio = float(
                cv2.countNonZero(result.valid_mask) / result.valid_mask.size
            )

            ecc_homography = np.vstack([ecc_warp, np.array([0.0, 0.0, 1.0])])
            result.standard_to_live = H @ ecc_homography
        except (cv2.error, TypeError, RuntimeError):
            # Non-convergence is a diagnostic observation.  The homography
            # result remains available for the quality gate below.
            pass

    # ----- 8. Diagnostic classification ------------------------------------
    _classify_diagnostic(result, th)
    return result


def _compute_projected_geometry(
    result: AlignmentResult,
    standard_gray: np.ndarray,
    live_gray: np.ndarray,
) -> None:
    """Compute real projected-geometry metrics from the homography.

    Mirrors P-1 addProjectedGeometryMetrics exactly.
    """
    H = result.homography
    if H is None or not _is_finite_homography(H):
        result.diagnostic_reasons.append("homography contains a non-finite coefficient")
        return

    W_s, H_s = standard_gray.shape[1], standard_gray.shape[0]
    W_l, H_l = live_gray.shape[1], live_gray.shape[0]

    std_corners = np.float32([
        [0, 0], [W_s - 1, 0], [W_s - 1, H_s - 1], [0, H_s - 1],
    ]).reshape(-1, 1, 2)
    center_pt = np.float32([[(W_s - 1) / 2.0, (H_s - 1) / 2.0]]).reshape(-1, 1, 2)
    ref_pts = np.vstack([std_corners, center_pt])

    projected = cv2.perspectiveTransform(ref_pts, H)
    if projected is None or not np.all(np.isfinite(projected)):
        result.diagnostic_reasons.append("projected image geometry is non-finite")
        return

    proj_corners = projected[:4].reshape(4, 2)
    proj_center = projected[4].ravel()

    # Convexity and area check
    area = abs(cv2.contourArea(proj_corners.astype(np.float32)))
    if not (math.isfinite(area) and area > 1.0):
        result.diagnostic_reasons.append("projected image geometry is degenerate")
        return
    if not cv2.isContourConvex(proj_corners.astype(np.float32)):
        result.diagnostic_reasons.append("projected image geometry is degenerate")
        return

    result.projected_geometry_valid = True
    result.projected_area_ratio = area / (W_l * H_l)

    live_center = np.float32([(W_l - 1) / 2.0, (H_l - 1) / 2.0])
    result.center_displacement_pixels = float(np.linalg.norm(proj_center - live_center))
    result.center_displacement_relative_diagonal = (
        result.center_displacement_pixels / math.hypot(W_l, H_l)
    )

    live_corners = np.float32([
        [0, 0], [W_l - 1, 0], [W_l - 1, H_l - 1], [0, H_l - 1],
    ])
    displacements = [
        float(np.linalg.norm(proj_corners[i] - live_corners[i]))
        for i in range(4)
    ]
    result.corner_displacement_median_pixels = _median(displacements)

    result.projected_corners_in_live_frame = sum(
        1 for pt in proj_corners
        if 0.0 <= pt[0] <= W_l - 1 and 0.0 <= pt[1] <= H_l - 1
    )


def _classify_diagnostic(result: AlignmentResult, th) -> None:
    """Apply P-1 triage rules to produce unavailable / unreliable / usable.

    Args:
        th: AlignmentThresholds from the loaded config.
    """
    reasons: list[str] = []

    # Structural blockers → unavailable
    if result.homography is None:
        reasons.append("homography unavailable")
        result.diagnostic = AlignmentDiagnostic.UNAVAILABLE
        result.diagnostic_reasons = reasons
        return

    if not result.projected_geometry_valid:
        reasons.append("projected geometry unavailable")
        result.diagnostic = AlignmentDiagnostic.UNAVAILABLE
        result.diagnostic_reasons = reasons
        return

    if not result.valid_overlap_available:
        reasons.append("valid overlap unavailable")
        result.diagnostic = AlignmentDiagnostic.UNAVAILABLE
        result.diagnostic_reasons = reasons
        return

    # Evidence-quality checks → unreliable
    if result.feature_match_count < th.feature_match_count_min:
        reasons.append(
            f"feature matches below diagnostic minimum ({th.feature_match_count_min})"
        )
    if result.inlier_count < th.inlier_count_min:
        reasons.append(
            f"inliers below diagnostic minimum ({th.inlier_count_min})"
        )
    if not math.isfinite(result.inlier_rate) or result.inlier_rate < th.inlier_rate_min:
        reasons.append(
            f"inlier rate below diagnostic minimum ({th.inlier_rate_min})"
        )
    if (not math.isfinite(result.reprojection_error_pixels)
            or result.reprojection_error_pixels > th.reprojection_error_pixels_max):
        reasons.append(
            f"reprojection error exceeds diagnostic maximum ({th.reprojection_error_pixels_max} px)"
        )
    if (not math.isfinite(result.spatial_coverage)
            or result.spatial_coverage < th.spatial_coverage_min):
        reasons.append(
            f"inlier spatial coverage below diagnostic minimum ({th.spatial_coverage_min})"
        )
    if (not math.isfinite(result.projected_area_ratio)
            or result.projected_area_ratio < th.projected_area_ratio_min
            or result.projected_area_ratio > th.projected_area_ratio_max):
        reasons.append(
            f"projected area ratio outside diagnostic range "
            f"[{th.projected_area_ratio_min}, {th.projected_area_ratio_max}]"
        )
    if (not math.isfinite(result.valid_overlap_ratio)
            or result.valid_overlap_ratio < th.valid_overlap_ratio_min):
        reasons.append(
            f"valid overlap below diagnostic minimum ({th.valid_overlap_ratio_min})"
        )
    if (result.ecc_converged
            and result.ecc_correlation < th.ecc_correlation_min_when_converged):
        reasons.append(
            "ECC correlation below diagnostic minimum after convergence "
            f"({th.ecc_correlation_min_when_converged})"
        )

    if reasons:
        result.diagnostic = AlignmentDiagnostic.UNRELIABLE
    else:
        result.diagnostic = AlignmentDiagnostic.USABLE

    result.diagnostic_reasons = reasons
