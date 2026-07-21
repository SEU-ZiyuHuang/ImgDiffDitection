// ONNX Runtime 后端的三个推理模块。
//
// 这些类刻意使用 Onnx 前缀，避免与现有 Ascend 版 ImageMatcher、LDC、
// DeepDifferenceDetector 同名冲突。它们只依赖 OpenCV 和 ONNX Runtime，
// 可在不具备 Ascend NPU 的开发环境运行。
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace imagecmp {
namespace onnx {

// 两图关键点匹配结果。left 为样本图，right 为实时图。
struct MatchResult {
    std::vector<cv::Point2f> keypoints_left;
    std::vector<cv::Point2f> keypoints_right;
    std::vector<cv::Point2f> matched_left;
    std::vector<cv::Point2f> matched_right;
};

// 进程内共享一个 ORT Environment；Session 可以安全地各自持有。
Ort::Env& runtime_environment();

// 从 ORT Session 读取节点名并保存为 std::string，避免 AllocatedStringPtr 生命周期问题。
std::vector<std::string> session_input_names(Ort::Session& session);
std::vector<std::string> session_output_names(Ort::Session& session);
std::vector<const char*> c_string_views(const std::vector<std::string>& names);

class OnnxImageMatcher {
public:
    explicit OnnxImageMatcher(const std::string& model_path);

    // 对两张大图进行联合特征匹配。
    MatchResult get_matches(const cv::Mat& left_img, const cv::Mat& right_img);

    // 将样本图中的 src_box 映射为实时图中的矩形。返回 false 表示定位证据不足。
    bool locate_box(const cv::Rect& src_box, int src_width, int src_height,
                    const MatchResult& matches, cv::Rect& target_box) const;

private:
    std::vector<float> preprocess_one(const cv::Mat& image) const;
    cv::Rect resize_rect(const cv::Rect& rect, int src_width, int src_height,
                         int dst_width, int dst_height) const;
    cv::Rect restore_rect(const cv::Rect& rect, int dst_width, int dst_height) const;

    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int width_ = 512;
    int height_ = 512;

    const float confidence_threshold_ = 0.5f;
    const float distance_threshold_ = 150.0f;
    const int min_global_matches_ = 20;
    const int min_box_matches_ = 5;
};

class OnnxLdc {
public:
    explicit OnnxLdc(const std::string& model_path);

    // 生成与输入图同尺寸的 8 位单通道轮廓图。
    void detect(const cv::Mat& source, cv::Mat& edge_image);

private:
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int input_width_ = 0;
    int input_height_ = 0;
};

class OnnxDeepDifferenceDetector {
public:
    explicit OnnxDeepDifferenceDetector(const std::string& model_path);

    // 在大图中定位小图。max_value 是归一化相关系数，语义仅为定位可信度。
    static cv::Rect template_matching(const cv::Mat& large_image, const cv::Mat& small_image,
                                      double& max_value,
                                      int method = cv::TM_CCOEFF_NORMED);

    // 返回两个输入图的全局 ResNet 特征余弦相似度。
    float compute_similarity(const cv::Mat& left, const cv::Mat& right,
                             const cv::Size& target_size = cv::Size(224, 224));

    // 计算局部特征差异，返回输入图片像素坐标中的候选异动框。
    std::vector<cv::Rect> detect_differences(const cv::Mat& left, const cv::Mat& right,
                                             float difference_threshold = 0.7f,
                                             float roi_difference_threshold = 0.7f,
                                             int min_contour_area = 500,
                                             const cv::Size& target_size = cv::Size(224, 224));

private:
    std::pair<cv::Mat, cv::Mat> preprocess_image(const cv::Mat& image,
                                                   const cv::Size& target_size) const;
    cv::Mat extract_features(const cv::Mat& rgb_image);
    cv::Mat compute_dense_difference(const cv::Mat& left_features, const cv::Mat& right_features,
                                     const cv::Size& output_size) const;

    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int feature_height_ = 0;
    int feature_width_ = 0;

    const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std_dev_ = {0.229f, 0.224f, 0.225f};
};

}  // namespace onnx
}  // namespace imagecmp
