#include "imagecmp/image_comparison_service.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace imagecmp {
namespace {

constexpr double kNotAvailable = std::numeric_limits<double>::quiet_NaN();
constexpr double kRoiBoundaryTolerance = 0.01;
// These guards classify the observed alignment evidence for P-1 reporting only.
// They are deliberately not a production decision policy.
constexpr int kMinimumDiagnosticFeatureMatches = 12;
constexpr int kMinimumDiagnosticInliers = 8;
constexpr double kMinimumDiagnosticInlierRate = 0.40;
constexpr double kMaximumDiagnosticReprojectionErrorPixels = 3.0;
constexpr double kMinimumDiagnosticSpatialCoverage = 0.02;
constexpr double kMinimumDiagnosticProjectedAreaRatio = 0.20;
constexpr double kMaximumDiagnosticProjectedAreaRatio = 5.0;
constexpr double kMinimumDiagnosticOverlapRatio = 0.60;
constexpr double kMinimumDiagnosticEccCorrelationWhenConverged = 0.20;

struct Roi {
    std::string category;
    double center_x = 0.0;
    double center_y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct RoiGeometry {
    std::size_t count = 0;
    double mean_relative_area = kNotAvailable;
    double mean_aspect_ratio = kNotAvailable;
};

struct RoiTotals {
    double relative_area_sum = 0.0;
    double aspect_ratio_sum = 0.0;
};

enum class CaseStatus { kValid, kIncomplete, kInvalid };
enum class AlignmentDiagnostic { kUnavailable, kUnreliable, kUsable };

struct CaseRecord {
    std::string relative_case_path;
    std::string case_type;
    std::vector<std::string> component_categories;
    std::map<std::string, RoiGeometry> roi_geometry_by_category;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    CaseStatus status = CaseStatus::kIncomplete;

    std::string standard_format;
    std::string live_format;
    int standard_width = 0;
    int standard_height = 0;
    int standard_channels = 0;
    int live_width = 0;
    int live_height = 0;
    int live_channels = 0;
    std::size_t roi_count = 0;
    std::size_t roi_boundary_normalized_lines = 0;
    bool image_analysis_complete = false;

    double standard_brightness = kNotAvailable;
    double live_brightness = kNotAvailable;
    double standard_contrast = kNotAvailable;
    double live_contrast = kNotAvailable;
    double standard_sharpness = kNotAvailable;
    double live_sharpness = kNotAvailable;
    double brightness_delta = kNotAvailable;
    double contrast_delta = kNotAvailable;
    double sharpness_delta = kNotAvailable;
    double raw_luma_mad = kNotAvailable;
    double aspect_ratio_delta = kNotAvailable;
    double mean_roi_relative_area = kNotAvailable;
    double mean_roi_aspect_ratio = kNotAvailable;

    int standard_keypoints = 0;
    int live_keypoints = 0;
    int feature_match_count = 0;
    int inlier_count = 0;
    double inlier_rate = kNotAvailable;
    double reprojection_error_pixels = kNotAvailable;
    double spatial_coverage = kNotAvailable;
    double center_displacement_pixels = kNotAvailable;
    double center_displacement_relative_diagonal = kNotAvailable;
    double corner_displacement_median_pixels = kNotAvailable;
    int projected_corners_in_live_frame = 0;
    double projected_area_ratio = kNotAvailable;
    bool projected_geometry_valid = false;
    bool homography_available = false;
    bool ecc_converged = false;
    double ecc_correlation = kNotAvailable;
    bool valid_overlap_available = false;
    double valid_overlap_ratio = kNotAvailable;
    AlignmentDiagnostic alignment_diagnostic = AlignmentDiagnostic::kUnavailable;
    std::vector<std::string> alignment_diagnostic_reasons;
};

using MetricSamples = std::map<std::string, std::vector<double>>;

struct GroupAccumulator {
    std::size_t case_count = 0;
    std::size_t valid_cases = 0;
    std::size_t incomplete_cases = 0;
    std::size_t invalid_cases = 0;
    std::size_t roi_boundary_normalized_cases = 0;
    std::size_t roi_boundary_normalized_lines = 0;
    std::size_t homography_available_cases = 0;
    std::size_t ecc_converged_cases = 0;
    std::size_t valid_overlap_available_cases = 0;
    std::size_t alignment_unavailable_cases = 0;
    std::size_t alignment_unreliable_cases = 0;
    std::size_t alignment_usable_cases = 0;
    MetricSamples metrics;
};

struct Distribution {
    std::size_t count = 0;
    double minimum = kNotAvailable;
    double maximum = kNotAvailable;
    double mean = kNotAvailable;
    double median = kNotAvailable;
    double p05 = kNotAvailable;
    double p95 = kNotAvailable;
};

std::string pathToUtf8(const std::filesystem::path& path) {
    const auto value = path.generic_u8string();
    return std::string(value.begin(), value.end());
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string imageFormat(const std::filesystem::path& path) {
    std::string extension = pathToUtf8(path.extension());
    if (!extension.empty() && extension.front() == '.') {
        extension.erase(extension.begin());
    }
    return lower(extension);
}

std::string caseTypeFor(const std::filesystem::path& caseDirectory) {
    const std::string name = pathToUtf8(caseDirectory.filename());
    const std::size_t separator = name.rfind('_');
    if (separator == std::string::npos || separator + 1 >= name.size()) {
        return "UNSPECIFIED";
    }
    return name.substr(separator + 1);
}

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<Roi> readRois(const std::filesystem::path& path, std::vector<std::string>& errors,
                          std::vector<std::string>& warnings) {
    std::ifstream input(path);
    if (!input) {
        errors.emplace_back("cannot read ROI file");
        return {};
    }

    std::vector<Roi> rois;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (trim(line).empty()) {
            continue;
        }
        Roi roi;
        std::string surplus;
        std::istringstream values(line);
        if (!(values >> roi.category >> roi.center_x >> roi.center_y >> roi.width >> roi.height) ||
            (values >> surplus) || !std::isfinite(roi.center_x) || !std::isfinite(roi.center_y) ||
            !std::isfinite(roi.width) || !std::isfinite(roi.height) || roi.width <= 0.0 ||
            roi.height <= 0.0) {
            errors.emplace_back("invalid ROI at line " + std::to_string(lineNumber));
            continue;
        }
        const double left = roi.center_x - roi.width / 2.0;
        const double right = roi.center_x + roi.width / 2.0;
        const double top = roi.center_y - roi.height / 2.0;
        const double bottom = roi.center_y + roi.height / 2.0;
        if (left < -kRoiBoundaryTolerance || right > 1.0 + kRoiBoundaryTolerance ||
            top < -kRoiBoundaryTolerance || bottom > 1.0 + kRoiBoundaryTolerance) {
            errors.emplace_back("invalid ROI at line " + std::to_string(lineNumber));
            continue;
        }

        const double normalizedLeft = std::clamp(left, 0.0, 1.0);
        const double normalizedRight = std::clamp(right, 0.0, 1.0);
        const double normalizedTop = std::clamp(top, 0.0, 1.0);
        const double normalizedBottom = std::clamp(bottom, 0.0, 1.0);
        if (normalizedRight <= normalizedLeft || normalizedBottom <= normalizedTop) {
            errors.emplace_back("invalid ROI at line " + std::to_string(lineNumber));
            continue;
        }
        if (normalizedLeft != left || normalizedRight != right || normalizedTop != top ||
            normalizedBottom != bottom) {
            warnings.emplace_back("ROI boundary normalized at line " + std::to_string(lineNumber));
            roi.center_x = (normalizedLeft + normalizedRight) / 2.0;
            roi.center_y = (normalizedTop + normalizedBottom) / 2.0;
            roi.width = normalizedRight - normalizedLeft;
            roi.height = normalizedBottom - normalizedTop;
        }
        rois.push_back(std::move(roi));
    }

    if (rois.empty() && errors.empty()) {
        errors.emplace_back("ROI file contains no ROI");
    }
    return rois;
}

cv::Mat toGray(const cv::Mat& image) {
    cv::Mat gray;
    switch (image.channels()) {
        case 1:
            gray = image;
            break;
        case 3:
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            break;
        case 4:
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
            break;
        default:
            throw std::runtime_error("unsupported image channel count");
    }
    return gray;
}

cv::Mat readImage(const std::filesystem::path& path) {
    // OpenCV's path overload is narrow-character based on some Windows builds.
    // Reading through std::filesystem keeps UTF-8 default case filenames usable.
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        return {};
    }
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> encoded(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char*>(encoded.data()), size)) {
        return {};
    }
    return cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
}

void addQualityMetrics(const cv::Mat& gray, double& brightness, double& contrast, double& sharpness) {
    cv::Scalar mean;
    cv::Scalar standardDeviation;
    cv::meanStdDev(gray, mean, standardDeviation);
    brightness = mean[0];
    contrast = standardDeviation[0];

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::meanStdDev(laplacian, mean, standardDeviation);
    sharpness = standardDeviation[0] * standardDeviation[0];
}

void addError(CaseRecord& record, const std::string& error) {
    record.errors.push_back(error);
}

bool isFiniteHomography(const cv::Mat& homography) {
    if (homography.rows != 3 || homography.cols != 3 || homography.type() != CV_64F) {
        return false;
    }
    for (int row = 0; row < homography.rows; ++row) {
        for (int column = 0; column < homography.cols; ++column) {
            if (!std::isfinite(homography.at<double>(row, column))) {
                return false;
            }
        }
    }
    return true;
}

bool isFinitePoint(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

bool isPointInImage(const cv::Point2f& point, const cv::Mat& image) {
    return point.x >= 0.0F && point.x <= static_cast<float>(image.cols - 1) && point.y >= 0.0F &&
           point.y <= static_cast<float>(image.rows - 1);
}

double medianOf(std::vector<double> values) {
    if (values.empty()) {
        return kNotAvailable;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

void addAlignmentDiagnosticReason(CaseRecord& record, const std::string& reason) {
    if (std::find(record.alignment_diagnostic_reasons.begin(), record.alignment_diagnostic_reasons.end(), reason) ==
        record.alignment_diagnostic_reasons.end()) {
        record.alignment_diagnostic_reasons.push_back(reason);
    }
}

void addProjectedGeometryMetrics(const cv::Mat& homography, const cv::Mat& standardGray, const cv::Mat& liveGray,
                                 CaseRecord& record) {
    if (!isFiniteHomography(homography)) {
        addAlignmentDiagnosticReason(record, "homography contains a non-finite coefficient");
        return;
    }

    const std::vector<cv::Point2f> standardCorners = {
        {0.0F, 0.0F}, {static_cast<float>(standardGray.cols - 1), 0.0F},
        {static_cast<float>(standardGray.cols - 1), static_cast<float>(standardGray.rows - 1)},
        {0.0F, static_cast<float>(standardGray.rows - 1)}};
    std::vector<cv::Point2f> referencePoints = standardCorners;
    referencePoints.emplace_back(static_cast<float>(standardGray.cols - 1) / 2.0F,
                                 static_cast<float>(standardGray.rows - 1) / 2.0F);
    std::vector<cv::Point2f> projectedPoints;
    cv::perspectiveTransform(referencePoints, projectedPoints, homography);
    if (projectedPoints.size() != referencePoints.size() ||
        !std::all_of(projectedPoints.begin(), projectedPoints.end(), isFinitePoint)) {
        addAlignmentDiagnosticReason(record, "projected image geometry is non-finite");
        return;
    }

    const std::vector<cv::Point2f> projectedCorners(projectedPoints.begin(), projectedPoints.begin() + 4);
    const double projectedArea = std::abs(cv::contourArea(projectedCorners));
    if (!std::isfinite(projectedArea) || projectedArea <= 1.0 || !cv::isContourConvex(projectedCorners)) {
        addAlignmentDiagnosticReason(record, "projected image geometry is degenerate");
        return;
    }

    record.projected_geometry_valid = true;
    record.projected_area_ratio = projectedArea / static_cast<double>(liveGray.cols * liveGray.rows);
    const cv::Point2f liveCenter(static_cast<float>(liveGray.cols - 1) / 2.0F,
                                 static_cast<float>(liveGray.rows - 1) / 2.0F);
    record.center_displacement_pixels = cv::norm(projectedPoints.back() - liveCenter);
    record.center_displacement_relative_diagonal =
        record.center_displacement_pixels / std::hypot(static_cast<double>(liveGray.cols), static_cast<double>(liveGray.rows));

    const std::vector<cv::Point2f> liveCorners = {
        {0.0F, 0.0F}, {static_cast<float>(liveGray.cols - 1), 0.0F},
        {static_cast<float>(liveGray.cols - 1), static_cast<float>(liveGray.rows - 1)},
        {0.0F, static_cast<float>(liveGray.rows - 1)}};
    std::vector<double> cornerDisplacements;
    cornerDisplacements.reserve(projectedCorners.size());
    for (std::size_t index = 0; index < projectedCorners.size(); ++index) {
        cornerDisplacements.push_back(cv::norm(projectedCorners[index] - liveCorners[index]));
        if (isPointInImage(projectedCorners[index], liveGray)) {
            ++record.projected_corners_in_live_frame;
        }
    }
    record.corner_displacement_median_pixels = medianOf(std::move(cornerDisplacements));
}

void finalizeAlignmentDiagnostic(CaseRecord& record) {
    if (!record.homography_available) {
        addAlignmentDiagnosticReason(record, "homography unavailable");
        record.alignment_diagnostic = AlignmentDiagnostic::kUnavailable;
        return;
    }
    if (!record.projected_geometry_valid) {
        addAlignmentDiagnosticReason(record, "projected geometry unavailable");
        record.alignment_diagnostic = AlignmentDiagnostic::kUnavailable;
        return;
    }
    if (!record.valid_overlap_available) {
        addAlignmentDiagnosticReason(record, "valid overlap unavailable");
        record.alignment_diagnostic = AlignmentDiagnostic::kUnavailable;
        return;
    }

    if (record.feature_match_count < kMinimumDiagnosticFeatureMatches) {
        addAlignmentDiagnosticReason(record, "feature matches below diagnostic minimum (12)");
    }
    if (record.inlier_count < kMinimumDiagnosticInliers) {
        addAlignmentDiagnosticReason(record, "inliers below diagnostic minimum (8)");
    }
    if (!std::isfinite(record.inlier_rate) || record.inlier_rate < kMinimumDiagnosticInlierRate) {
        addAlignmentDiagnosticReason(record, "inlier rate below diagnostic minimum (0.40)");
    }
    if (!std::isfinite(record.reprojection_error_pixels) ||
        record.reprojection_error_pixels > kMaximumDiagnosticReprojectionErrorPixels) {
        addAlignmentDiagnosticReason(record, "reprojection error exceeds diagnostic maximum (3 px)");
    }
    if (!std::isfinite(record.spatial_coverage) || record.spatial_coverage < kMinimumDiagnosticSpatialCoverage) {
        addAlignmentDiagnosticReason(record, "inlier spatial coverage below diagnostic minimum (0.02)");
    }
    if (!std::isfinite(record.projected_area_ratio) ||
        record.projected_area_ratio < kMinimumDiagnosticProjectedAreaRatio ||
        record.projected_area_ratio > kMaximumDiagnosticProjectedAreaRatio) {
        addAlignmentDiagnosticReason(record, "projected area ratio outside diagnostic range [0.20, 5.00]");
    }
    if (!std::isfinite(record.valid_overlap_ratio) || record.valid_overlap_ratio < kMinimumDiagnosticOverlapRatio) {
        addAlignmentDiagnosticReason(record, "valid overlap below diagnostic minimum (0.60)");
    }
    if (record.ecc_converged && record.ecc_correlation < kMinimumDiagnosticEccCorrelationWhenConverged) {
        addAlignmentDiagnosticReason(record, "ECC correlation below diagnostic minimum after convergence (0.20)");
    }

    record.alignment_diagnostic = record.alignment_diagnostic_reasons.empty()
                                      ? AlignmentDiagnostic::kUsable
                                      : AlignmentDiagnostic::kUnreliable;
}

void analyzeFeatureAndAlignmentEvidence(const cv::Mat& standardGray, const cv::Mat& liveGray,
                                        CaseRecord& record) {
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> standardKeypoints;
    std::vector<cv::KeyPoint> liveKeypoints;
    cv::Mat standardDescriptors;
    cv::Mat liveDescriptors;
    orb->detectAndCompute(standardGray, cv::noArray(), standardKeypoints, standardDescriptors);
    orb->detectAndCompute(liveGray, cv::noArray(), liveKeypoints, liveDescriptors);
    record.standard_keypoints = static_cast<int>(standardKeypoints.size());
    record.live_keypoints = static_cast<int>(liveKeypoints.size());

    if (standardDescriptors.empty() || liveDescriptors.empty()) {
        return;
    }

    std::vector<std::vector<cv::DMatch>> neighbours;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(standardDescriptors, liveDescriptors, neighbours, 2);
    std::vector<cv::DMatch> matches;
    for (const auto& candidates : neighbours) {
        if (candidates.size() == 2 && candidates[0].distance < 0.75F * candidates[1].distance) {
            matches.push_back(candidates[0]);
        }
    }
    record.feature_match_count = static_cast<int>(matches.size());
    if (matches.size() < 4) {
        return;
    }

    std::vector<cv::Point2f> standardPoints;
    std::vector<cv::Point2f> livePoints;
    standardPoints.reserve(matches.size());
    livePoints.reserve(matches.size());
    for (const cv::DMatch& match : matches) {
        standardPoints.push_back(standardKeypoints[match.queryIdx].pt);
        livePoints.push_back(liveKeypoints[match.trainIdx].pt);
    }

    cv::Mat inlierMask;
    cv::Mat homography = cv::findHomography(standardPoints, livePoints, cv::RANSAC, 3.0, inlierMask);
    if (homography.empty()) {
        return;
    }

    homography.convertTo(homography, CV_64F);
    record.homography_available = true;
    std::vector<cv::Point2f> inlierStandardPoints;
    std::vector<cv::Point2f> inlierLivePoints;
    for (std::size_t index = 0; index < matches.size(); ++index) {
        if (inlierMask.at<unsigned char>(static_cast<int>(index)) != 0U) {
            inlierStandardPoints.push_back(standardPoints[index]);
            inlierLivePoints.push_back(livePoints[index]);
        }
    }
    record.inlier_count = static_cast<int>(inlierStandardPoints.size());
    record.inlier_rate = static_cast<double>(record.inlier_count) /
                         static_cast<double>(record.feature_match_count);
    if (inlierStandardPoints.empty()) {
        return;
    }

    std::vector<cv::Point2f> projectedPoints;
    cv::perspectiveTransform(inlierStandardPoints, projectedPoints, homography);
    double totalReprojectionError = 0.0;
    for (std::size_t index = 0; index < projectedPoints.size(); ++index) {
        totalReprojectionError += cv::norm(projectedPoints[index] - inlierLivePoints[index]);
    }
    record.reprojection_error_pixels = totalReprojectionError / projectedPoints.size();

    if (inlierStandardPoints.size() >= 3) {
        std::vector<cv::Point2f> hull;
        cv::convexHull(inlierStandardPoints, hull);
        record.spatial_coverage = std::abs(cv::contourArea(hull)) /
                                  static_cast<double>(standardGray.cols * standardGray.rows);
    }

    addProjectedGeometryMetrics(homography, standardGray, liveGray, record);
    if (!record.projected_geometry_valid) {
        return;
    }

    cv::Mat inverseHomography;
    if (cv::invert(homography, inverseHomography, cv::DECOMP_SVD) == 0.0) {
        addAlignmentDiagnosticReason(record, "homography cannot be inverted");
        return;
    }

    cv::Mat liveMask(liveGray.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat validMask;
    cv::warpPerspective(liveMask, validMask, inverseHomography, standardGray.size(), cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar(0));
    record.valid_overlap_available = true;
    record.valid_overlap_ratio = static_cast<double>(cv::countNonZero(validMask)) /
                                 static_cast<double>(validMask.total());

    cv::Mat alignedLive;
    cv::warpPerspective(liveGray, alignedLive, inverseHomography, standardGray.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat eccWarp = cv::Mat::eye(2, 3, CV_32F);
    try {
        record.ecc_correlation = cv::findTransformECC(
            standardGray, alignedLive, eccWarp, cv::MOTION_AFFINE,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 1e-5));
        record.ecc_converged = true;
    } catch (const cv::Exception&) {
        // A convergence failure is evidence in the report, not a failed P-1 run.
    }
}

void analyzeImages(const std::filesystem::path& standardPath, const std::filesystem::path& livePath,
                   const std::vector<Roi>& rois, CaseRecord& record) {
    const cv::Mat standard = readImage(standardPath);
    const cv::Mat live = readImage(livePath);
    if (standard.empty()) {
        addError(record, "unreadable standard image");
    }
    if (live.empty()) {
        addError(record, "unreadable live image");
    }
    if (!record.errors.empty()) {
        return;
    }

    record.standard_width = standard.cols;
    record.standard_height = standard.rows;
    record.standard_channels = standard.channels();
    record.live_width = live.cols;
    record.live_height = live.rows;
    record.live_channels = live.channels();
    record.standard_format = imageFormat(standardPath);
    record.live_format = imageFormat(livePath);

    const cv::Mat standardGray = toGray(standard);
    const cv::Mat liveGray = toGray(live);
    addQualityMetrics(standardGray, record.standard_brightness, record.standard_contrast,
                      record.standard_sharpness);
    addQualityMetrics(liveGray, record.live_brightness, record.live_contrast, record.live_sharpness);
    record.brightness_delta = std::abs(record.standard_brightness - record.live_brightness);
    record.contrast_delta = std::abs(record.standard_contrast - record.live_contrast);
    record.sharpness_delta = std::abs(record.standard_sharpness - record.live_sharpness);

    cv::Mat resizedLive;
    cv::resize(liveGray, resizedLive, standardGray.size(), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat rawDifference;
    cv::absdiff(standardGray, resizedLive, rawDifference);
    record.raw_luma_mad = cv::mean(rawDifference)[0];
    const double standardAspect = static_cast<double>(standard.cols) / standard.rows;
    const double liveAspect = static_cast<double>(live.cols) / live.rows;
    record.aspect_ratio_delta = std::abs(standardAspect - liveAspect);

    record.roi_count = rois.size();
    double areaTotal = 0.0;
    double aspectTotal = 0.0;
    std::map<std::string, RoiTotals> categoryTotals;
    for (const Roi& roi : rois) {
        const double relativeArea = roi.width * roi.height;
        const double aspectRatio = roi.width / roi.height;
        areaTotal += relativeArea;
        aspectTotal += aspectRatio;
        auto& totals = categoryTotals[roi.category];
        totals.relative_area_sum += relativeArea;
        totals.aspect_ratio_sum += aspectRatio;
        ++record.roi_geometry_by_category[roi.category].count;
    }
    record.mean_roi_relative_area = areaTotal / rois.size();
    record.mean_roi_aspect_ratio = aspectTotal / rois.size();
    for (const auto& entry : categoryTotals) {
        RoiGeometry& geometry = record.roi_geometry_by_category[entry.first];
        geometry.mean_relative_area = entry.second.relative_area_sum / geometry.count;
        geometry.mean_aspect_ratio = entry.second.aspect_ratio_sum / geometry.count;
    }
    record.image_analysis_complete = true;

    analyzeFeatureAndAlignmentEvidence(standardGray, liveGray, record);
    finalizeAlignmentDiagnostic(record);
}

void addMetric(MetricSamples& samples, const std::string& name, double value) {
    if (std::isfinite(value)) {
        samples[name].push_back(value);
    }
}

MetricSamples metricsFor(const CaseRecord& record) {
    MetricSamples metrics;
    if (!record.image_analysis_complete) {
        return metrics;
    }
    addMetric(metrics, "standard_width", record.standard_width == 0 ? kNotAvailable : record.standard_width);
    addMetric(metrics, "standard_height", record.standard_height == 0 ? kNotAvailable : record.standard_height);
    addMetric(metrics, "live_width", record.live_width == 0 ? kNotAvailable : record.live_width);
    addMetric(metrics, "live_height", record.live_height == 0 ? kNotAvailable : record.live_height);
    addMetric(metrics, "standard_channels", record.standard_channels == 0 ? kNotAvailable : record.standard_channels);
    addMetric(metrics, "live_channels", record.live_channels == 0 ? kNotAvailable : record.live_channels);
    addMetric(metrics, "roi_count", record.roi_count == 0 ? kNotAvailable : record.roi_count);
    addMetric(metrics, "standard_brightness", record.standard_brightness);
    addMetric(metrics, "live_brightness", record.live_brightness);
    addMetric(metrics, "standard_contrast", record.standard_contrast);
    addMetric(metrics, "live_contrast", record.live_contrast);
    addMetric(metrics, "standard_sharpness", record.standard_sharpness);
    addMetric(metrics, "live_sharpness", record.live_sharpness);
    addMetric(metrics, "brightness_delta", record.brightness_delta);
    addMetric(metrics, "contrast_delta", record.contrast_delta);
    addMetric(metrics, "sharpness_delta", record.sharpness_delta);
    addMetric(metrics, "raw_luma_mad", record.raw_luma_mad);
    addMetric(metrics, "aspect_ratio_delta", record.aspect_ratio_delta);
    addMetric(metrics, "mean_roi_relative_area", record.mean_roi_relative_area);
    addMetric(metrics, "mean_roi_aspect_ratio", record.mean_roi_aspect_ratio);
    addMetric(metrics, "standard_keypoints", record.standard_keypoints);
    addMetric(metrics, "live_keypoints", record.live_keypoints);
    addMetric(metrics, "feature_match_count", record.feature_match_count);
    addMetric(metrics, "inlier_count", record.inlier_count);
    addMetric(metrics, "inlier_rate", record.inlier_rate);
    addMetric(metrics, "reprojection_error_pixels", record.reprojection_error_pixels);
    addMetric(metrics, "spatial_coverage", record.spatial_coverage);
    addMetric(metrics, "center_displacement_pixels", record.center_displacement_pixels);
    addMetric(metrics, "center_displacement_relative_diagonal", record.center_displacement_relative_diagonal);
    addMetric(metrics, "corner_displacement_median_pixels", record.corner_displacement_median_pixels);
    addMetric(metrics, "projected_corners_in_live_frame", record.projected_corners_in_live_frame);
    addMetric(metrics, "projected_area_ratio", record.projected_area_ratio);
    addMetric(metrics, "ecc_correlation", record.ecc_correlation);
    addMetric(metrics, "valid_overlap_ratio", record.valid_overlap_ratio);
    return metrics;
}

const std::array<const char*, 34>& metricNames() {
    static const std::array<const char*, 34> names = {
        "standard_width", "standard_height", "live_width", "live_height", "standard_channels", "live_channels",
        "roi_count", "standard_brightness", "live_brightness", "standard_contrast", "live_contrast",
        "standard_sharpness", "live_sharpness", "brightness_delta", "contrast_delta", "sharpness_delta",
        "raw_luma_mad", "aspect_ratio_delta", "mean_roi_relative_area", "mean_roi_aspect_ratio",
        "standard_keypoints", "live_keypoints", "feature_match_count", "inlier_count", "inlier_rate",
        "reprojection_error_pixels", "spatial_coverage", "center_displacement_pixels",
        "center_displacement_relative_diagonal", "corner_displacement_median_pixels",
        "projected_corners_in_live_frame", "projected_area_ratio", "ecc_correlation", "valid_overlap_ratio"};
    return names;
}

std::string alignmentDiagnosticName(AlignmentDiagnostic diagnostic) {
    switch (diagnostic) {
        case AlignmentDiagnostic::kUnavailable:
            return "unavailable";
        case AlignmentDiagnostic::kUnreliable:
            return "unreliable";
        case AlignmentDiagnostic::kUsable:
            return "usable";
    }
    return "unavailable";
}

void addRecord(GroupAccumulator& group, const CaseRecord& record) {
    ++group.case_count;
    if (record.status == CaseStatus::kValid) {
        ++group.valid_cases;
    } else if (record.status == CaseStatus::kIncomplete) {
        ++group.incomplete_cases;
    } else {
        ++group.invalid_cases;
    }
    if (record.homography_available) {
        ++group.homography_available_cases;
    }
    if (record.ecc_converged) {
        ++group.ecc_converged_cases;
    }
    if (record.valid_overlap_available) {
        ++group.valid_overlap_available_cases;
    }
    switch (record.alignment_diagnostic) {
        case AlignmentDiagnostic::kUnavailable:
            ++group.alignment_unavailable_cases;
            break;
        case AlignmentDiagnostic::kUnreliable:
            ++group.alignment_unreliable_cases;
            break;
        case AlignmentDiagnostic::kUsable:
            ++group.alignment_usable_cases;
            break;
    }
    if (record.roi_boundary_normalized_lines != 0) {
        ++group.roi_boundary_normalized_cases;
        group.roi_boundary_normalized_lines += record.roi_boundary_normalized_lines;
    }
    for (const auto& entry : metricsFor(record)) {
        group.metrics[entry.first].insert(group.metrics[entry.first].end(), entry.second.begin(), entry.second.end());
    }
}

CaseRecord recordForComponentCategory(const CaseRecord& record, const std::string& category) {
    CaseRecord categoryRecord = record;
    const auto geometry = record.roi_geometry_by_category.find(category);
    if (geometry != record.roi_geometry_by_category.end()) {
        categoryRecord.roi_count = geometry->second.count;
        categoryRecord.mean_roi_relative_area = geometry->second.mean_relative_area;
        categoryRecord.mean_roi_aspect_ratio = geometry->second.mean_aspect_ratio;
    }
    return categoryRecord;
}

double quantile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return kNotAvailable;
    }
    const double index = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(index));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(index));
    const double remainder = index - lowerIndex;
    return sorted[lowerIndex] + (sorted[upperIndex] - sorted[lowerIndex]) * remainder;
}

Distribution distributionFor(std::vector<double> values) {
    Distribution result;
    result.count = values.size();
    if (values.empty()) {
        return result;
    }
    std::sort(values.begin(), values.end());
    result.minimum = values.front();
    result.maximum = values.back();
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    result.mean = sum / values.size();
    result.median = quantile(values, 0.5);
    result.p05 = quantile(values, 0.05);
    result.p95 = quantile(values, 0.95);
    return result;
}

void writeJsonString(std::ostream& output, const std::string& value) {
    output << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '\\':
                output << "\\\\";
                break;
            case '"':
                output << "\\\"";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec << std::setfill(' ');
                } else {
                    output << character;
                }
        }
    }
    output << '"';
}

void writeJsonNumber(std::ostream& output, double value) {
    if (std::isfinite(value)) {
        output << std::setprecision(10) << value;
    } else {
        output << "null";
    }
}

void writeDistribution(std::ostream& output, const Distribution& distribution) {
    output << "{\"count\": " << distribution.count << ", \"min\": ";
    writeJsonNumber(output, distribution.minimum);
    output << ", \"max\": ";
    writeJsonNumber(output, distribution.maximum);
    output << ", \"mean\": ";
    writeJsonNumber(output, distribution.mean);
    output << ", \"median\": ";
    writeJsonNumber(output, distribution.median);
    output << ", \"p05\": ";
    writeJsonNumber(output, distribution.p05);
    output << ", \"p95\": ";
    writeJsonNumber(output, distribution.p95);
    output << '}';
}

void writeMetricDistributions(std::ostream& output, const MetricSamples& metrics, const std::string& indent) {
    MetricSamples reportMetrics = metrics;
    for (const char* name : metricNames()) {
        reportMetrics.try_emplace(name);
    }
    output << "{";
    if (!reportMetrics.empty()) {
        output << '\n';
    }
    for (auto iterator = reportMetrics.begin(); iterator != reportMetrics.end(); ++iterator) {
        output << indent << "  ";
        writeJsonString(output, iterator->first);
        output << ": ";
        writeDistribution(output, distributionFor(iterator->second));
        if (std::next(iterator) != reportMetrics.end()) {
            output << ',';
        }
        output << '\n';
    }
    output << indent << '}';
}

void writeGroup(std::ostream& output, const std::string& value, const GroupAccumulator& group,
                const std::string& indent) {
    output << indent << "{\n" << indent << "  \"value\": ";
    writeJsonString(output, value);
    output << ",\n" << indent << "  \"case_count\": " << group.case_count << ",\n" << indent
           << "  \"valid_cases\": " << group.valid_cases << ",\n" << indent
           << "  \"incomplete_cases\": " << group.incomplete_cases << ",\n" << indent
           << "  \"invalid_cases\": " << group.invalid_cases << ",\n" << indent
           << "  \"roi_boundary_normalization\": {\"cases\": "
           << group.roi_boundary_normalized_cases << ", \"lines\": "
           << group.roi_boundary_normalized_lines << "},\n" << indent
           << "  \"alignment_evidence\": {\"homography_available_cases\": "
           << group.homography_available_cases << ", \"ecc_converged_cases\": "
           << group.ecc_converged_cases << ", \"valid_overlap_available_cases\": "
           << group.valid_overlap_available_cases << "},\n" << indent
           << "  \"alignment_diagnostic\": {\"unavailable_cases\": "
           << group.alignment_unavailable_cases << ", \"unreliable_cases\": "
           << group.alignment_unreliable_cases << ", \"usable_cases\": " << group.alignment_usable_cases
           << "},\n" << indent << "  \"metrics\": ";
    writeMetricDistributions(output, group.metrics, indent + "  ");
    output << "\n" << indent << '}';
}

void writeGroups(std::ostream& output, const std::map<std::string, GroupAccumulator>& groups,
                 const std::string& indent) {
    output << "[";
    if (!groups.empty()) {
        output << '\n';
    }
    for (auto iterator = groups.begin(); iterator != groups.end(); ++iterator) {
        writeGroup(output, iterator->first, iterator->second, indent + "  ");
        if (std::next(iterator) != groups.end()) {
            output << ',';
        }
        output << '\n';
    }
    output << indent << ']';
}

std::string statusName(CaseStatus status) {
    switch (status) {
        case CaseStatus::kValid:
            return "valid";
        case CaseStatus::kIncomplete:
            return "incomplete";
        case CaseStatus::kInvalid:
            return "invalid";
    }
    return "invalid";
}

std::string join(const std::vector<std::string>& values, const std::string& separator) {
    std::ostringstream result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            result << separator;
        }
        result << values[index];
    }
    return result.str();
}

void writeCsvValue(std::ostream& output, const std::string& value) {
    output << '"';
    for (char character : value) {
        if (character == '"') {
            output << '"';
        }
        output << character;
    }
    output << '"';
}

void writeCsvNumber(std::ostream& output, double value) {
    if (std::isfinite(value)) {
        output << std::setprecision(10) << value;
    }
}

void writeCaseReport(const std::filesystem::path& path, const std::vector<CaseRecord>& cases) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write per-case report");
    }
    output << "case,case_type,component_categories,status,errors,warnings,roi_boundary_normalized_lines,standard_format,live_format,"
              "standard_width,standard_height,standard_channels,live_width,live_height,live_channels,roi_count,"
              "standard_brightness,live_brightness,standard_contrast,live_contrast,standard_sharpness,live_sharpness,"
              "brightness_delta,contrast_delta,sharpness_delta,raw_luma_mad,aspect_ratio_delta,mean_roi_relative_area,"
              "mean_roi_aspect_ratio,standard_keypoints,"
              "live_keypoints,feature_match_count,inlier_count,inlier_rate,reprojection_error_pixels,spatial_coverage,"
              "center_displacement_pixels,center_displacement_relative_diagonal,corner_displacement_median_pixels,"
              "projected_corners_in_live_frame,projected_area_ratio,projected_geometry_valid,alignment_diagnostic,"
              "alignment_diagnostic_reasons,homography_available,ecc_converged,ecc_correlation,"
              "valid_overlap_available,valid_overlap_ratio\n";
    for (const CaseRecord& record : cases) {
        writeCsvValue(output, record.relative_case_path);
        output << ',';
        writeCsvValue(output, record.case_type);
        output << ',';
        writeCsvValue(output, join(record.component_categories, "|"));
        output << ',';
        writeCsvValue(output, statusName(record.status));
        output << ',';
        writeCsvValue(output, join(record.errors, "; "));
        output << ',';
        writeCsvValue(output, join(record.warnings, "; "));
        output << ',' << record.roi_boundary_normalized_lines << ',';
        writeCsvValue(output, record.standard_format);
        output << ',';
        writeCsvValue(output, record.live_format);
        output << ',' << record.standard_width << ',' << record.standard_height << ',' << record.standard_channels
               << ',' << record.live_width << ',' << record.live_height << ',' << record.live_channels << ','
               << record.roi_count << ',';
        const std::array<double, 25> values = {
            record.standard_brightness, record.live_brightness, record.standard_contrast, record.live_contrast,
            record.standard_sharpness, record.live_sharpness, record.brightness_delta, record.contrast_delta,
            record.sharpness_delta, record.raw_luma_mad, record.aspect_ratio_delta, record.mean_roi_relative_area,
            record.mean_roi_aspect_ratio,
            static_cast<double>(record.standard_keypoints), static_cast<double>(record.live_keypoints),
            static_cast<double>(record.feature_match_count), static_cast<double>(record.inlier_count),
            record.inlier_rate, record.reprojection_error_pixels, record.spatial_coverage,
            record.center_displacement_pixels, record.center_displacement_relative_diagonal,
            record.corner_displacement_median_pixels, static_cast<double>(record.projected_corners_in_live_frame),
            record.projected_area_ratio};
        for (double value : values) {
            writeCsvNumber(output, value);
            output << ',';
        }
        output << (record.projected_geometry_valid ? "true" : "false") << ',';
        writeCsvValue(output, alignmentDiagnosticName(record.alignment_diagnostic));
        output << ',';
        writeCsvValue(output, join(record.alignment_diagnostic_reasons, "; "));
        output << ',' << (record.homography_available ? "true" : "false") << ','
               << (record.ecc_converged ? "true" : "false") << ',';
        writeCsvNumber(output, record.ecc_correlation);
        output << ',' << (record.valid_overlap_available ? "true" : "false") << ',';
        writeCsvNumber(output, record.valid_overlap_ratio);
        output << '\n';
    }
}

double distributionMean(const GroupAccumulator& group, const std::string& metric) {
    const auto found = group.metrics.find(metric);
    return found == group.metrics.end() ? kNotAvailable : distributionFor(found->second).mean;
}

void writeGroupReport(const std::filesystem::path& path,
                      const std::map<std::string, GroupAccumulator>& componentGroups,
                      const std::map<std::string, GroupAccumulator>& caseTypeGroups) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write grouped report");
    }
    output << "group_dimension,group_value,case_count,valid_cases,incomplete_cases,invalid_cases,"
              "roi_boundary_normalized_cases,roi_boundary_normalized_lines,"
              "homography_available_cases,ecc_converged_cases,valid_overlap_available_cases,"
              "alignment_unavailable_cases,alignment_unreliable_cases,alignment_usable_cases,"
              "mean_raw_luma_mad,mean_feature_match_count,mean_inlier_rate,mean_reprojection_error_pixels,"
              "mean_spatial_coverage,mean_center_displacement_relative_diagonal,mean_projected_area_ratio,"
              "mean_ecc_correlation,mean_valid_overlap_ratio\n";
    const auto writeDimension = [&output](const std::string& dimension,
                                          const std::map<std::string, GroupAccumulator>& groups) {
        for (const auto& entry : groups) {
            const GroupAccumulator& group = entry.second;
            writeCsvValue(output, dimension);
            output << ',';
            writeCsvValue(output, entry.first);
            output << ',' << group.case_count << ',' << group.valid_cases << ',' << group.incomplete_cases << ','
                   << group.invalid_cases << ',' << group.roi_boundary_normalized_cases << ','
                   << group.roi_boundary_normalized_lines << ',' << group.homography_available_cases << ','
                   << group.ecc_converged_cases << ',' << group.valid_overlap_available_cases << ','
                   << group.alignment_unavailable_cases << ',' << group.alignment_unreliable_cases << ','
                   << group.alignment_usable_cases << ',';
            const std::array<double, 9> means = {
                distributionMean(group, "raw_luma_mad"), distributionMean(group, "feature_match_count"),
                distributionMean(group, "inlier_rate"), distributionMean(group, "reprojection_error_pixels"),
                distributionMean(group, "spatial_coverage"),
                distributionMean(group, "center_displacement_relative_diagonal"),
                distributionMean(group, "projected_area_ratio"), distributionMean(group, "ecc_correlation"),
                distributionMean(group, "valid_overlap_ratio")};
            for (std::size_t index = 0; index < means.size(); ++index) {
                writeCsvNumber(output, means[index]);
                if (index + 1 != means.size()) {
                    output << ',';
                }
            }
            output << '\n';
        }
    };
    writeDimension("component_category", componentGroups);
    writeDimension("case_type", caseTypeGroups);
}

void writeAggregateReport(const std::filesystem::path& path, const DatasetAnalysisSummary& summary,
                          const GroupAccumulator& allCases,
                          const std::map<std::string, GroupAccumulator>& componentGroups,
                          const std::map<std::string, GroupAccumulator>& caseTypeGroups,
                          const std::map<std::string, std::size_t>& standardFormats,
                          const std::map<std::string, std::size_t>& liveFormats,
                          const std::map<std::string, std::size_t>& validationErrors) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write aggregate report");
    }

    const auto writeStringCounts = [&output](const std::map<std::string, std::size_t>& counts) {
        output << '{';
        for (auto iterator = counts.begin(); iterator != counts.end(); ++iterator) {
            writeJsonString(output, iterator->first);
            output << ": " << iterator->second;
            if (std::next(iterator) != counts.end()) {
                output << ", ";
            }
        }
        output << '}';
    };

    output << "{\n"
           << "  \"schema_version\": \"p1-characterization-v2\",\n"
           << "  \"scope\": {\n"
           << "    \"execution\": \"local-only\",\n"
           << "    \"network_operations\": \"none\",\n"
           << "    \"source_images_copied\": false,\n"
           << "    \"source_images_exported\": false\n"
           << "  },\n"
           << "  \"limitations\": [\n"
           << "    \"This is a descriptive P-1 dataset characterization report, not a detection result.\",\n"
           << "    \"It makes no anomaly recall, missed-detection, precision, or production-readiness claim.\",\n"
           << "    \"ORB ratio matching, RANSAC homography, and ECC are observed for alignment selection only; their values are not production decision thresholds.\",\n"
           << "    \"The alignment diagnostic is a fixed P-1 reporting triage, not a deployment policy; it requires review against labelled change data before production use.\"\n"
           << "  ],\n"
           << "  \"alignment_diagnostic_policy\": {\n"
           << "    \"purpose\": \"P-1 descriptive triage of homography evidence; not a production decision policy\",\n"
           << "    \"usable_requires\": {\"feature_match_count_min\": "
           << kMinimumDiagnosticFeatureMatches << ", \"inlier_count_min\": " << kMinimumDiagnosticInliers
           << ", \"inlier_rate_min\": " << kMinimumDiagnosticInlierRate
           << ", \"reprojection_error_pixels_max\": " << kMaximumDiagnosticReprojectionErrorPixels
           << ", \"spatial_coverage_min\": " << kMinimumDiagnosticSpatialCoverage
           << ", \"projected_area_ratio_min\": " << kMinimumDiagnosticProjectedAreaRatio
           << ", \"projected_area_ratio_max\": " << kMaximumDiagnosticProjectedAreaRatio
           << ", \"valid_overlap_ratio_min\": " << kMinimumDiagnosticOverlapRatio << "},\n"
           << "    \"ecc_rule\": \"ECC is optional; if it converges, correlation must be at least "
           << kMinimumDiagnosticEccCorrelationWhenConverged << " for the usable diagnostic class\"\n"
           << "  },\n"
           << "  \"summary\": {\n"
           << "    \"total_cases\": " << summary.total_cases << ",\n"
           << "    \"valid_cases\": " << summary.valid_cases << ",\n"
           << "    \"incomplete_cases\": " << summary.incomplete_cases << ",\n"
           << "    \"invalid_cases\": " << summary.invalid_cases << ",\n"
           << "    \"validation_errors\": ";
    writeStringCounts(validationErrors);
    output << ",\n    \"image_formats\": {\"standard\": ";
    writeStringCounts(standardFormats);
    output << ", \"live\": ";
    writeStringCounts(liveFormats);
    output << "},\n"
           << "    \"roi_boundary_normalization\": {\"cases\": "
           << allCases.roi_boundary_normalized_cases << ", \"lines\": "
           << allCases.roi_boundary_normalized_lines << "},\n"
           << "    \"alignment_evidence\": {\"homography_available_cases\": "
           << allCases.homography_available_cases << ", \"ecc_converged_cases\": "
           << allCases.ecc_converged_cases << ", \"valid_overlap_available_cases\": "
           << allCases.valid_overlap_available_cases << "},\n"
           << "    \"alignment_diagnostic\": {\"unavailable_cases\": "
           << allCases.alignment_unavailable_cases << ", \"unreliable_cases\": "
           << allCases.alignment_unreliable_cases << ", \"usable_cases\": "
           << allCases.alignment_usable_cases << "},\n"
           << "    \"metrics\": ";
    writeMetricDistributions(output, allCases.metrics, "    ");
    output << "\n  },\n"
           << "  \"groups\": {\n"
           << "    \"component_category\": ";
    writeGroups(output, componentGroups, "    ");
    output << ",\n    \"case_type\": ";
    writeGroups(output, caseTypeGroups, "    ");
    output << "\n  }\n}\n";
}

CaseRecord analyzeCase(const std::filesystem::path& datasetRoot, const std::filesystem::path& caseDirectory,
                       const DatasetAnalysisRequest& request) {
    CaseRecord record;
    record.relative_case_path = pathToUtf8(std::filesystem::relative(caseDirectory, datasetRoot));
    record.case_type = caseTypeFor(caseDirectory);

    const std::filesystem::path standardPath = caseDirectory / std::filesystem::u8path(request.standard_image_name);
    const std::filesystem::path livePath = caseDirectory / std::filesystem::u8path(request.live_image_name);
    const std::filesystem::path roiPath = caseDirectory / std::filesystem::u8path(request.roi_name);
    bool missingRequiredFile = false;
    if (!std::filesystem::is_regular_file(standardPath)) {
        addError(record, "missing standard image");
        missingRequiredFile = true;
    }
    if (!std::filesystem::is_regular_file(livePath)) {
        addError(record, "missing live image");
        missingRequiredFile = true;
    }
    if (!std::filesystem::is_regular_file(roiPath)) {
        addError(record, "missing ROI file");
        missingRequiredFile = true;
    }
    if (missingRequiredFile) {
        record.status = CaseStatus::kIncomplete;
        return record;
    }

    const std::vector<Roi> rois = readRois(roiPath, record.errors, record.warnings);
    record.roi_boundary_normalized_lines = record.warnings.size();
    std::set<std::string> categories;
    for (const Roi& roi : rois) {
        categories.insert(roi.category);
    }
    record.component_categories.assign(categories.begin(), categories.end());
    if (!record.errors.empty()) {
        record.status = CaseStatus::kInvalid;
        return record;
    }

    try {
        analyzeImages(standardPath, livePath, rois, record);
    } catch (const cv::Exception& error) {
        addError(record, std::string("OpenCV image analysis failed: ") + error.what());
    } catch (const std::exception& error) {
        addError(record, std::string("image analysis failed: ") + error.what());
    }
    record.status = record.errors.empty() ? CaseStatus::kValid : CaseStatus::kInvalid;
    return record;
}

}  // namespace

DatasetAnalysisResult ImageComparisonService::analyzeDataset(const DatasetAnalysisRequest& request) const {
    namespace fs = std::filesystem;
    if (request.dataset_root.empty() || !fs::is_directory(request.dataset_root)) {
        throw std::invalid_argument("dataset_root must be an existing local directory");
    }
    if (request.output_directory.empty()) {
        throw std::invalid_argument("output_directory must be specified");
    }
    if (request.standard_image_name.empty() || request.live_image_name.empty() || request.roi_name.empty()) {
        throw std::invalid_argument("input file names must not be empty");
    }

    std::vector<fs::path> caseDirectories;
    for (const fs::directory_entry& entry : fs::directory_iterator(request.dataset_root)) {
        if (entry.is_directory()) {
            caseDirectories.push_back(entry.path());
        }
    }
    std::sort(caseDirectories.begin(), caseDirectories.end());

    std::error_code outputError;
    fs::create_directories(request.output_directory, outputError);
    if (outputError || !fs::is_directory(request.output_directory)) {
        throw std::runtime_error("cannot create local output directory");
    }

    std::vector<CaseRecord> cases;
    cases.reserve(caseDirectories.size());
    DatasetAnalysisSummary summary;
    GroupAccumulator allCases;
    std::map<std::string, GroupAccumulator> componentGroups;
    std::map<std::string, GroupAccumulator> caseTypeGroups;
    std::map<std::string, std::size_t> standardFormats;
    std::map<std::string, std::size_t> liveFormats;
    std::map<std::string, std::size_t> validationErrors;
    for (const fs::path& caseDirectory : caseDirectories) {
        CaseRecord record = analyzeCase(request.dataset_root, caseDirectory, request);
        ++summary.total_cases;
        if (record.status == CaseStatus::kValid) {
            ++summary.valid_cases;
        } else if (record.status == CaseStatus::kIncomplete) {
            ++summary.incomplete_cases;
        } else {
            ++summary.invalid_cases;
        }
        addRecord(allCases, record);
        addRecord(caseTypeGroups[record.case_type], record);
        for (const std::string& category : record.component_categories) {
            addRecord(componentGroups[category], recordForComponentCategory(record, category));
        }
        if (record.component_categories.empty()) {
            // A missing or invalid ROI has no defensible component category; keep
            // the failure visible instead of silently excluding it from the view.
            addRecord(componentGroups["UNCLASSIFIED"], record);
        }
        if (!record.standard_format.empty()) {
            ++standardFormats[record.standard_format];
        }
        if (!record.live_format.empty()) {
            ++liveFormats[record.live_format];
        }
        for (const std::string& error : record.errors) {
            ++validationErrors[error];
        }
        cases.push_back(std::move(record));
    }

    DatasetAnalysisResult result;
    result.summary = summary;
    result.report_path = request.output_directory / "p1_characterization_report.json";
    result.case_report_path = request.output_directory / "p1_cases.csv";
    result.group_report_path = request.output_directory / "p1_groups.csv";
    writeCaseReport(result.case_report_path, cases);
    writeGroupReport(result.group_report_path, componentGroups, caseTypeGroups);
    writeAggregateReport(result.report_path, summary, allCases, componentGroups, caseTypeGroups, standardFormats,
                         liveFormats, validationErrors);
    return result;
}

}  // namespace imagecmp
