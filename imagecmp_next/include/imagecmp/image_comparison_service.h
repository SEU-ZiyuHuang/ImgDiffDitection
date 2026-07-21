#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace imagecmp {

// P-1 is descriptive analysis only. It reads each case from a local directory
// and writes reports, never copies images or performs network I/O.
struct DatasetAnalysisRequest {
    std::filesystem::path dataset_root;
    std::filesystem::path output_directory;
    std::string standard_image_name = "标准源图.jpg";
    std::string live_image_name = "对比截图.jpg";
    std::string roi_name = "标准源图坐标.txt";
};

struct DatasetAnalysisSummary {
    std::size_t total_cases = 0;
    std::size_t valid_cases = 0;
    std::size_t incomplete_cases = 0;
    std::size_t invalid_cases = 0;
};

struct DatasetAnalysisResult {
    DatasetAnalysisSummary summary;
    std::filesystem::path report_path;
    std::filesystem::path case_report_path;
    std::filesystem::path group_report_path;
};

// The public seam for P-1. Future P0/P1 operations extend this service rather
// than exposing a matcher or alignment implementation to callers.
class ImageComparisonService {
public:
    DatasetAnalysisResult analyzeDataset(const DatasetAnalysisRequest& request) const;
};

}  // namespace imagecmp
