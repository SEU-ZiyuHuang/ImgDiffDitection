#include "imagecmp/image_comparison_service.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage(std::ostream& output) {
    output << "Usage: imagecmp_characterize --dataset <local-directory> --output <local-directory>"
              " [--standard-name <filename>] [--live-name <filename>] [--roi-name <filename>]\n";
}

std::string nextValue(int& index, int argc, char* argv[], const char* option) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("Missing value for ") + option);
    }
    return argv[index];
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        imagecmp::DatasetAnalysisRequest request;
        bool hasDataset = false;
        bool hasOutput = false;

        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--help" || option == "-h") {
                printUsage(std::cout);
                return 0;
            }
            if (option == "--dataset") {
                request.dataset_root = nextValue(index, argc, argv, "--dataset");
                hasDataset = true;
            } else if (option == "--output") {
                request.output_directory = nextValue(index, argc, argv, "--output");
                hasOutput = true;
            } else if (option == "--standard-name") {
                request.standard_image_name = nextValue(index, argc, argv, "--standard-name");
            } else if (option == "--live-name") {
                request.live_image_name = nextValue(index, argc, argv, "--live-name");
            } else if (option == "--roi-name") {
                request.roi_name = nextValue(index, argc, argv, "--roi-name");
            } else {
                throw std::invalid_argument("Unknown option: " + option);
            }
        }

        if (!hasDataset || !hasOutput) {
            printUsage(std::cerr);
            return 2;
        }

        const imagecmp::DatasetAnalysisResult result =
            imagecmp::ImageComparisonService().analyzeDataset(request);
        std::cout << "P-1 local characterization complete\n"
                  << "  cases: " << result.summary.total_cases << " (valid "
                  << result.summary.valid_cases << ", incomplete "
                  << result.summary.incomplete_cases << ", invalid "
                  << result.summary.invalid_cases << ")\n"
                  << "  aggregate report: " << result.report_path.string() << "\n"
                  << "  per-case report: " << result.case_report_path.string() << "\n"
                  << "  group report: " << result.group_report_path.string() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Characterization failed: " << error.what() << '\n';
        return 1;
    }
}
