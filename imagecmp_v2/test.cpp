// ONNX Runtime 后端的一次性端到端验证程序。
//
// 示例（在包含三个 .onnx 文件的目录执行）：
//   imagecmp_test --live test/live.jpg --template test/temp.jpg --yolo test/yolo.txt --model-dir .
//
// 它逐行读取 YOLO 标注并调用主 C ABI。程序只执行一轮，不再像历史版本那样无限循环，
// 因而适合人工验收和后续 CI。ret=1 表示得到有效异动框，ret=0 表示未得到异动框。
#include "imagecmp.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct YoloRect {
    int class_id = 0;
    float x_center = 0.0f;
    float y_center = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

struct Options {
    std::string live_path = "test/live.jpg";
    std::string template_path = "test/temp.jpg";
    std::string yolo_path = "test/yolo.txt";
    std::string model_dir = ".";
    std::string output_dir;
    // 默认把可视化结果放入项目的 test/，便于直接打开检查。
    std::string annotation_dir = "test";
    float threshold = 0.70f;
};

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--live <image>] [--template <image>] [--yolo <labels.txt>]"
              << " [--model-dir <dir>] [--threshold <0..1>] [--output-dir <existing-dir>]"
              << " [--annotation-dir <existing-dir>]\n";
}

bool read_value(int& index, int argc, char** argv, std::string& output) {
    if (index + 1 >= argc) return false;
    output = argv[++index];
    return true;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        std::string value;
        if (argument == "--help" || argument == "-h") {
            return false;
        } else if (argument == "--live" && read_value(index, argc, argv, value)) {
            options.live_path = value;
        } else if (argument == "--template" && read_value(index, argc, argv, value)) {
            options.template_path = value;
        } else if (argument == "--yolo" && read_value(index, argc, argv, value)) {
            options.yolo_path = value;
        } else if (argument == "--model-dir" && read_value(index, argc, argv, value)) {
            options.model_dir = value;
        } else if (argument == "--output-dir" && read_value(index, argc, argv, value)) {
            options.output_dir = value;
        } else if (argument == "--annotation-dir" && read_value(index, argc, argv, value)) {
            options.annotation_dir = value;
        } else if (argument == "--threshold" && read_value(index, argc, argv, value)) {
            std::istringstream stream(value);
            if (!(stream >> options.threshold) || options.threshold < 0.0f || options.threshold > 1.0f) {
                std::cerr << "--threshold must be between 0 and 1\n";
                return false;
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << argument << '\n';
            return false;
        }
    }
    return true;
}

std::vector<YoloRect> load_yolo_rects(const std::string& file_path) {
    std::ifstream file(file_path.c_str());
    if (!file) throw std::runtime_error("cannot open YOLO label file: " + file_path);

    std::vector<YoloRect> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;
        std::istringstream stream(line);
        YoloRect rect;
        if (!(stream >> rect.class_id >> rect.x_center >> rect.y_center >> rect.width >> rect.height)) {
            throw std::runtime_error("invalid YOLO label at line " + std::to_string(line_number));
        }
        result.push_back(rect);
    }
    if (result.empty()) throw std::runtime_error("YOLO label file contains no component rectangles");
    return result;
}

void set_model_directory(const std::string& model_dir) {
#ifdef _WIN32
    if (_putenv_s("IMAGECMP_MODEL_DIR", model_dir.c_str()) != 0) {
        throw std::runtime_error("failed to set IMAGECMP_MODEL_DIR");
    }
#else
    if (setenv("IMAGECMP_MODEL_DIR", model_dir.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set IMAGECMP_MODEL_DIR");
    }
#endif
}

std::string library_error() {
    char* message = 0;
    const int status = lxGetLastErrorMessage(&message);
    std::string result = (status == 1 && message) ? message : "(no detailed library error)";
    lxFreePtr(message);
    return result;
}

cv::Rect clip_rect(const cv::Rect& rect, const cv::Mat& image) {
    return rect & cv::Rect(0, 0, image.cols, image.rows);
}

// 将算法输出的实时 YOLO 框还原成像素框，用蓝色标出“系统实际定位到的部件”。
cv::Rect live_yolo_to_rect(const float live_yolo[4], const cv::Mat& image) {
    const int width = static_cast<int>(std::lround(live_yolo[2] * image.cols));
    const int height = static_cast<int>(std::lround(live_yolo[3] * image.rows));
    const int x = static_cast<int>(std::lround(live_yolo[0] * image.cols - width * 0.5));
    const int y = static_cast<int>(std::lround(live_yolo[1] * image.rows - height * 0.5));
    return clip_rect(cv::Rect(x, y, width, height), image);
}

// 生成便于人工验收的叠加图：蓝框是定位部件，红框是局部异动，绿色文字表示未发现有效异动。
bool save_annotation(const std::string& live_path, const std::string& output_path,
                     int component_index, int class_id, int result, float similarity,
                     const float live_yolo[4], const Box& difference_box) {
    cv::Mat image = cv::imread(live_path, cv::IMREAD_COLOR);
    if (image.empty()) return false;

    const cv::Rect located_component = live_yolo_to_rect(live_yolo, image);
    if (located_component.width > 0 && located_component.height > 0) {
        cv::rectangle(image, located_component, cv::Scalar(255, 0, 0), 3);  // 蓝色：定位部件
        cv::putText(image, "LOCATED COMPONENT", cv::Point(located_component.x,
                    std::max(24, located_component.y - 8)), cv::FONT_HERSHEY_SIMPLEX,
                    0.75, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }

    std::ostringstream caption;
    caption << "component=" << component_index << " class=" << class_id
            << " sim=" << std::fixed << std::setprecision(3) << similarity;
    if (result == 1) {
        const cv::Rect anomaly = clip_rect(
            cv::Rect(difference_box.x, difference_box.y, difference_box.w, difference_box.h), image);
        if (anomaly.width > 0 && anomaly.height > 0) {
            cv::rectangle(image, anomaly, cv::Scalar(0, 0, 255), 4);  // 红色：异动区域
            cv::putText(image, "CHANGE", cv::Point(anomaly.x, std::max(48, anomaly.y - 10)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }
        caption << "  RESULT=CHANGE";
    } else if (result == 0) {
        caption << "  RESULT=NO EFFECTIVE CHANGE";
    } else {
        caption << "  RESULT=FAILED";
    }

    const int banner_height = 42;
    cv::rectangle(image, cv::Rect(0, 0, image.cols, banner_height), cv::Scalar(30, 30, 30), cv::FILLED);
    cv::putText(image, caption.str(), cv::Point(16, 29), cv::FONT_HERSHEY_SIMPLEX,
                0.75, result == 1 ? cv::Scalar(0, 0, 255) :
                (result == 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255)), 2, cv::LINE_AA);
    return cv::imwrite(output_path, image);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        set_model_directory(options.model_dir);
        const std::vector<YoloRect> labels = load_yolo_rects(options.yolo_path);
        const int cpu_backend_id = 0;  // ONNX 后端忽略此值；为兼容原 ABI 仍传入一个设备号。
        if (lxInitAIModel(&cpu_backend_id, 1, 1) != 1) {
            std::cerr << "Model initialization failed: " << library_error() << '\n';
            return 3;
        }

        std::cout << "ONNX model directory: " << options.model_dir << '\n'
                  << "Components to test: " << labels.size() << "\n\n";
        int changed_count = 0;
        int unchanged_count = 0;
        int failed_count = 0;

        for (size_t index = 0; index < labels.size(); ++index) {
            const YoloRect& label = labels[index];
            float similarity = 0.0f;
            Box box = {0, 0, 0, 0};
            float live_yolo[4] = {0, 0, 0, 0};
            char* live_base64 = 0;
            char* temp_base64 = 0;
            const std::string crop_path = options.output_dir.empty() ? "" :
                options.output_dir + "/component_" + std::to_string(index) + ".jpg";
            const std::string annotation_path = options.annotation_dir.empty() ? "" :
                options.annotation_dir + "/annotated_component_" + std::to_string(index) + ".jpg";

            const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            const int result = lxImageCmpOnnx(
                options.live_path.c_str(), options.template_path.c_str(),
                label.x_center, label.y_center, label.width, label.height,
                options.threshold, &similarity, &box, crop_path.c_str(), 640,
                live_yolo, &live_base64, &temp_base64, 0);
            const long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started).count();
            lxFreePtr(live_base64);
            lxFreePtr(temp_base64);

            // 不论异动、无异动还是报错，都留下实时原图标注，便于人工核对定位和返回码。
            const bool annotation_saved = annotation_path.empty() ? true :
                save_annotation(options.live_path, annotation_path, static_cast<int>(index), label.class_id,
                                result, similarity, live_yolo, box);

            std::cout << "component=" << index << " class=" << label.class_id
                      << " ret=" << result << " similarity=" << std::fixed << std::setprecision(4)
                      << similarity << " live_yolo=[" << live_yolo[0] << ',' << live_yolo[1] << ','
                      << live_yolo[2] << ',' << live_yolo[3] << "] box=[" << box.x << ',' << box.y
                      << ',' << box.w << ',' << box.h << "] elapsed_ms=" << elapsed_ms;
            if (!annotation_path.empty()) {
                std::cout << " annotated=" << (annotation_saved ? annotation_path : "WRITE_FAILED");
            }
            if (result < 0) {
                ++failed_count;
                std::cout << " error=" << library_error();
            } else if (result == 1) {
                ++changed_count;
            } else {
                ++unchanged_count;
            }
            std::cout << '\n';
        }

        lxUninit();
        std::cout << "\nSummary: changed=" << changed_count << " unchanged=" << unchanged_count
                  << " failed=" << failed_count << '\n';
        return failed_count == 0 ? 0 : 4;
    } catch (const std::exception& exception) {
        std::cerr << "Test setup failed: " << exception.what() << '\n';
        lxUninit();
        return 5;
    }
}
