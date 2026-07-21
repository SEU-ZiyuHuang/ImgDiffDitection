#include "imagecmp/image_comparison_service.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void writeImage(const std::filesystem::path& path, const cv::Mat& image) {
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".jpg", image, encoded)) {
        throw std::runtime_error("cannot encode test image");
    }
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
}

cv::Mat createStandardImage() {
    cv::Mat standard(240, 320, CV_8UC3);
    cv::RNG(1234).fill(standard, cv::RNG::UNIFORM, 0, 255);
    cv::rectangle(standard, cv::Rect(40, 40, 140, 140), cv::Scalar(220, 220, 220), 3);
    cv::circle(standard, cv::Point(230, 144), 36, cv::Scalar(10, 20, 230), -1);
    cv::putText(standard, "A1", cv::Point(70, 210), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 0, 0), 2);
    return standard;
}

void writeValidCase(const std::filesystem::path& root, const std::string& name, const cv::Mat& standard,
                    const cv::Mat& live, const std::string& rois = "17 0.5 0.5 0.4 0.5\n") {
    namespace fs = std::filesystem;
    const fs::path caseDirectory = root / name;
    fs::create_directories(caseDirectory);
    writeImage(caseDirectory / std::filesystem::u8path(u8"标准源图.jpg"), standard);
    writeImage(caseDirectory / std::filesystem::u8path(u8"对比截图.jpg"), live);
    std::ofstream roi(caseDirectory / std::filesystem::u8path(u8"标准源图坐标.txt"));
    roi << rois;
}

void writeCompleteCase(const std::filesystem::path& root) {
    const cv::Mat standard = createStandardImage();
    cv::Mat translated;
    cv::warpAffine(standard, translated, cv::Matx23d(1, 0, 3, 0, 1, 2), standard.size());
    writeValidCase(root, "station_component_translation", standard, translated,
                   "17 0.5 0.5 0.4 0.5\n18 0.3 0.3 0.2 0.2\n");
}

void writeVariationCases(const std::filesystem::path& root) {
    const cv::Mat standard = createStandardImage();
    cv::Mat live;

    cv::warpAffine(standard, live, cv::getRotationMatrix2D(cv::Point2f(160, 120), 6.0, 1.0), standard.size());
    writeValidCase(root, "station_component_rotation", standard, live);

    cv::warpAffine(standard, live, cv::getRotationMatrix2D(cv::Point2f(160, 120), 0.0, 1.08), standard.size());
    writeValidCase(root, "station_component_scale", standard, live);

    const std::vector<cv::Point2f> source = {{0, 0}, {319, 0}, {319, 239}, {0, 239}};
    const std::vector<cv::Point2f> destination = {{10, 5}, {310, 0}, {318, 235}, {0, 239}};
    cv::warpPerspective(standard, live, cv::getPerspectiveTransform(source, destination), standard.size());
    writeValidCase(root, "station_component_perspective", standard, live);

    standard.convertTo(live, -1, 1.15, 18.0);
    writeValidCase(root, "station_component_illumination", standard, live);

    live = standard.clone();
    cv::rectangle(live, cv::Rect(0, 0, live.cols / 2, live.rows), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(standard, 0.55, live, 0.45, 0.0, live);
    writeValidCase(root, "station_component_shadow", standard, live);

    cv::GaussianBlur(standard, live, cv::Size(9, 9), 1.8);
    writeValidCase(root, "station_component_blur", standard, live);

    live = standard.clone();
    cv::rectangle(live, cv::Rect(205, 125, 24, 24), cv::Scalar(0, 255, 0), -1);
    writeValidCase(root, "station_component_local_change", standard, live);
}

void writeIncompleteCase(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    const fs::path caseDirectory = root / "station_component_routine";
    fs::create_directories(caseDirectory);
    cv::Mat live(40, 40, CV_8UC3, cv::Scalar(100, 100, 100));
    writeImage(caseDirectory / std::filesystem::u8path(u8"对比截图.jpg"), live);
    std::ofstream roi(caseDirectory / std::filesystem::u8path(u8"标准源图坐标.txt"));
    roi << "17 0.5 0.5 0.4 0.5\n";
}

void writeInvalidCase(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    const fs::path caseDirectory = root / "station_component_invalid";
    fs::create_directories(caseDirectory);
    cv::Mat image(40, 40, CV_8UC3, cv::Scalar(100, 100, 100));
    writeImage(caseDirectory / fs::u8path(u8"标准源图.jpg"), image);
    writeImage(caseDirectory / fs::u8path(u8"对比截图.jpg"), image);
    std::ofstream roi(caseDirectory / fs::u8path(u8"标准源图坐标.txt"));
    roi << "17 0.9 0.5 0.4 0.5\n";
}

void writeToleratedRoiCase(const std::filesystem::path& root) {
    const cv::Mat standard = createStandardImage();
    writeValidCase(root, "station_component_boundary_tolerated", standard, standard,
                   "17 0.776495 0.623402 0.447011 0.430379\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path fixtureRoot = fs::temp_directory_path() / "imagecmp_next_fixture";
    const fs::path outputRoot = fs::temp_directory_path() / "imagecmp_next_report";
    fs::remove_all(fixtureRoot);
    fs::remove_all(outputRoot);

    writeCompleteCase(fixtureRoot);
    writeVariationCases(fixtureRoot);
    writeIncompleteCase(fixtureRoot);
    writeInvalidCase(fixtureRoot);
    writeToleratedRoiCase(fixtureRoot);

    // The public data-characterization operation observes the controlled
    // variation suite, validates incomplete input, and never exports images.
    imagecmp::ImageComparisonService service;
    imagecmp::DatasetAnalysisRequest request;
    request.dataset_root = fixtureRoot;
    request.output_directory = outputRoot;
    const auto result = service.analyzeDataset(request);

    if (result.summary.total_cases != 11 || result.summary.valid_cases != 9 ||
        result.summary.incomplete_cases != 1 || result.summary.invalid_cases != 1 ||
        !fs::is_regular_file(result.report_path) ||
        !fs::is_regular_file(result.case_report_path) || !fs::is_regular_file(result.group_report_path)) {
        std::cerr << "Expected local P-1 report files and pair-completeness summary.\n";
        return EXIT_FAILURE;
    }

    const std::string report = readFile(result.report_path);
    const std::string cases = readFile(result.case_report_path);
    if (report.find("\"component_category\"") == std::string::npos ||
        report.find("\"case_type\"") == std::string::npos ||
        report.find("\"valid_overlap_ratio\"") == std::string::npos ||
        report.find("\"value\": \"UNCLASSIFIED\"") == std::string::npos ||
        report.find("\"value\": \"18\"") == std::string::npos ||
        report.find("\"roi_boundary_normalization\": {\"cases\": 1, \"lines\": 1}") == std::string::npos ||
        report.find("\"mean_roi_relative_area\": {\"count\": 1, \"min\": 0.04") == std::string::npos ||
        cases.find("missing standard image") == std::string::npos ||
        cases.find("invalid ROI at line 1") == std::string::npos ||
        cases.find("ROI boundary normalized at line 1") == std::string::npos ||
        cases.find("station_component_rotation") == std::string::npos ||
        cases.find("station_component_scale") == std::string::npos ||
        cases.find("station_component_perspective") == std::string::npos ||
        cases.find("station_component_illumination") == std::string::npos ||
        cases.find("station_component_shadow") == std::string::npos ||
        cases.find("station_component_blur") == std::string::npos ||
        cases.find("station_component_local_change") == std::string::npos) {
        std::cerr << "Expected local reports to preserve grouping, alignment evidence, and errors.\n";
        return EXIT_FAILURE;
    }

    std::size_t outputFiles = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(outputRoot)) {
        if (!entry.is_regular_file() || entry.path().extension() == ".jpg") {
            std::cerr << "Expected reports only; source images must not be copied.\n";
            return EXIT_FAILURE;
        }
        ++outputFiles;
    }
    if (outputFiles != 3) {
        std::cerr << "Expected exactly the three local report files.\n";
        return EXIT_FAILURE;
    }

    fs::remove_all(fixtureRoot);
    fs::remove_all(outputRoot);
    return EXIT_SUCCESS;
}
