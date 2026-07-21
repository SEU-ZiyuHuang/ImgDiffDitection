// ONNX Runtime 后端的模型加载、推理和特征后处理实现。
#include "onnx_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace imagecmp {
namespace onnx {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

size_t element_count(const std::vector<int64_t>& shape) {
    size_t result = 1;
    for (int64_t dimension : shape) {
        require(dimension > 0, "ONNX model contains an unresolved dynamic tensor dimension");
        result *= static_cast<size_t>(dimension);
    }
    return result;
}

cv::Mat row_l2_normalize(const cv::Mat& input) {
    cv::Mat normalized(input.rows, input.cols, CV_32F);
    for (int row = 0; row < input.rows; ++row) {
        const cv::Mat src_row = input.row(row);
        const float norm = static_cast<float>(cv::norm(src_row, cv::NORM_L2));
        if (norm > 1e-8f) {
            src_row.copyTo(normalized.row(row));
            normalized.row(row) /= norm;
        } else {
            normalized.row(row).setTo(0.0f);
        }
    }
    return normalized;
}

float global_cosine_similarity(const cv::Mat& left, const cv::Mat& right) {
    require(left.size() == right.size(), "Feature maps have different shapes");
    const cv::Mat left_flat = left.reshape(1, 1);
    const cv::Mat right_flat = right.reshape(1, 1);
    const float left_norm = static_cast<float>(cv::norm(left_flat, cv::NORM_L2));
    const float right_norm = static_cast<float>(cv::norm(right_flat, cv::NORM_L2));
    if (left_norm <= 1e-8f || right_norm <= 1e-8f) {
        return 0.0f;
    }
    return static_cast<float>(left_flat.dot(right_flat) / (left_norm * right_norm));
}

bool point_in_rect(const cv::Point2f& point, const cv::Rect& rect) {
    return point.x >= rect.x && point.x <= rect.x + rect.width &&
           point.y >= rect.y && point.y <= rect.y + rect.height;
}

// ONNX Runtime 在 Windows 使用 wchar_t 路径，Linux 使用 UTF-8 char 路径。
// 模型目录常包含中文，不能直接按字节转换为 std::wstring。
std::unique_ptr<Ort::Session> create_session(const std::string& model_path,
                                             const Ort::SessionOptions& options) {
#ifdef _WIN32
    const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, model_path.c_str(), -1,
                                             nullptr, 0);
    require(required > 0, "Cannot convert UTF-8 ONNX model path to UTF-16");
    // required 包含字符串末尾的 NUL，先保留这一个位置给 Win32 API 写入。
    std::wstring wide_path(static_cast<size_t>(required), L'\0');
    const int written = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, model_path.c_str(), -1,
                                            &wide_path[0], required);
    require(written == required, "Cannot convert UTF-8 ONNX model path to UTF-16");
    wide_path.resize(static_cast<size_t>(required - 1));
    return std::unique_ptr<Ort::Session>(
        new Ort::Session(runtime_environment(), wide_path.c_str(), options));
#else
    return std::unique_ptr<Ort::Session>(
        new Ort::Session(runtime_environment(), model_path.c_str(), options));
#endif
}

}  // namespace

Ort::Env& runtime_environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "imagecmp_onnx");
    return environment;
}

std::vector<std::string> session_input_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    const size_t count = session.GetInputCount();
    names.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(index, allocator);
        require(name != nullptr, "Unable to read ONNX input name");
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<std::string> session_output_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    const size_t count = session.GetOutputCount();
    names.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(index, allocator);
        require(name != nullptr, "Unable to read ONNX output name");
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<const char*> c_string_views(const std::vector<std::string>& names) {
    std::vector<const char*> result;
    result.reserve(names.size());
    for (const std::string& name : names) {
        result.push_back(name.c_str());
    }
    return result;
}

OnnxImageMatcher::OnnxImageMatcher(const std::string& model_path) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = create_session(model_path, options);
    input_names_ = session_input_names(*session_);
    output_names_ = session_output_names(*session_);

    require(input_names_.size() == 1, "SuperPoint+LightGlue model must have exactly one input");
    require(output_names_.size() >= 3, "SuperPoint+LightGlue model must have keypoints, matches and scores outputs");
    require(input_names_[0] == "images" && output_names_[0] == "keypoints" &&
            output_names_[1] == "matches" && output_names_[2] == "mscores",
            "Unexpected SuperPoint+LightGlue ONNX node names");

    const std::vector<int64_t> input_shape =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    require(input_shape.size() == 4, "SuperPoint+LightGlue input must be [2,1,H,W]");
    if (input_shape[2] > 0) height_ = static_cast<int>(input_shape[2]);
    if (input_shape[3] > 0) width_ = static_cast<int>(input_shape[3]);
    require(width_ > 0 && height_ > 0, "SuperPoint+LightGlue input size is invalid");
}

std::vector<float> OnnxImageMatcher::preprocess_one(const cv::Mat& image) const {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    cv::Mat normalized;
    gray.convertTo(normalized, CV_32F, 1.0 / 255.0);
    return std::vector<float>(reinterpret_cast<float*>(normalized.data),
                              reinterpret_cast<float*>(normalized.data) + normalized.total());
}

MatchResult OnnxImageMatcher::get_matches(const cv::Mat& left_image, const cv::Mat& right_image) {
    require(!left_image.empty() && !right_image.empty(), "ImageMatcher received an empty image");

    cv::Mat left_resized;
    cv::Mat right_resized;
    cv::resize(left_image, left_resized, cv::Size(width_, height_));
    cv::resize(right_image, right_resized, cv::Size(width_, height_));

    const std::vector<float> left = preprocess_one(left_resized);
    const std::vector<float> right = preprocess_one(right_resized);
    std::vector<float> input;
    input.reserve(left.size() + right.size());
    input.insert(input.end(), left.begin(), left.end());
    input.insert(input.end(), right.begin(), right.end());

    const std::vector<int64_t> shape = {2, 1, height_, width_};
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), shape.data(), shape.size());

    const std::vector<const char*> input_views = c_string_views(input_names_);
    const std::vector<const char*> output_views = c_string_views(output_names_);
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_views.data(), &tensor, 1,
        output_views.data(), output_views.size());
    require(outputs.size() >= 3, "SuperPoint+LightGlue returned fewer than three output tensors");

    const Ort::TensorTypeAndShapeInfo keypoint_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> keypoint_shape = keypoint_info.GetShape();
    require(keypoint_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            "SuperPoint+LightGlue keypoints must use int64 coordinates");
    require(keypoint_shape.size() == 3 && keypoint_shape[0] == 2 && keypoint_shape[2] == 2,
            "SuperPoint+LightGlue keypoints must have shape [2,N,2]");
    const size_t keypoint_count = static_cast<size_t>(keypoint_shape[1]);
    const int64_t* keypoints = outputs[0].GetTensorData<int64_t>();

    MatchResult result;
    result.keypoints_left.reserve(keypoint_count);
    result.keypoints_right.reserve(keypoint_count);
    for (size_t index = 0; index < keypoint_count; ++index) {
        result.keypoints_left.emplace_back(static_cast<float>(keypoints[index * 2]),
                                           static_cast<float>(keypoints[index * 2 + 1]));
        const size_t right_offset = keypoint_count * 2 + index * 2;
        result.keypoints_right.emplace_back(static_cast<float>(keypoints[right_offset]),
                                            static_cast<float>(keypoints[right_offset + 1]));
    }

    const Ort::TensorTypeAndShapeInfo match_info = outputs[1].GetTensorTypeAndShapeInfo();
    require(match_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            "SuperPoint+LightGlue matches must use int64 indices");
    const size_t match_value_count = element_count(match_info.GetShape());
    require(match_value_count % 3 == 0, "SuperPoint+LightGlue matches must be triplets");
    const size_t match_count = match_value_count / 3;
    const int64_t* match_data = outputs[1].GetTensorData<int64_t>();

    const Ort::TensorTypeAndShapeInfo score_info = outputs[2].GetTensorTypeAndShapeInfo();
    require(score_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "SuperPoint+LightGlue match scores must use float32");
    const size_t score_count = element_count(score_info.GetShape());
    const float* scores = outputs[2].GetTensorData<float>();

    const size_t count = std::min(match_count, score_count);
    result.matched_left.reserve(count);
    result.matched_right.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        if (scores[index] <= confidence_threshold_) continue;
        if (match_data[index * 3] != 0) continue;

        const int64_t left_index = match_data[index * 3 + 1];
        const int64_t right_index = match_data[index * 3 + 2];
        if (left_index < 0 || right_index < 0 ||
            static_cast<size_t>(left_index) >= result.keypoints_left.size() ||
            static_cast<size_t>(right_index) >= result.keypoints_right.size()) {
            continue;
        }

        const cv::Point2f left_point = result.keypoints_left[static_cast<size_t>(left_index)];
        const cv::Point2f right_point = result.keypoints_right[static_cast<size_t>(right_index)];
        if (cv::norm(left_point - right_point) > distance_threshold_) continue;
        result.matched_left.push_back(left_point);
        result.matched_right.push_back(right_point);
    }
    return result;
}

cv::Rect OnnxImageMatcher::resize_rect(const cv::Rect& rect, int src_width, int src_height,
                                       int dst_width, int dst_height) const {
    require(src_width > 0 && src_height > 0 && dst_width > 0 && dst_height > 0,
            "Invalid image dimensions when resizing a rectangle");
    const float scale_x = static_cast<float>(dst_width) / src_width;
    const float scale_y = static_cast<float>(dst_height) / src_height;
    int x = static_cast<int>(std::round(rect.x * scale_x));
    int y = static_cast<int>(std::round(rect.y * scale_y));
    int width = static_cast<int>(std::round(rect.width * scale_x));
    int height = static_cast<int>(std::round(rect.height * scale_y));
    x = std::max(0, std::min(x, dst_width - 1));
    y = std::max(0, std::min(y, dst_height - 1));
    width = std::max(1, std::min(width, dst_width - x));
    height = std::max(1, std::min(height, dst_height - y));
    return cv::Rect(x, y, width, height);
}

cv::Rect OnnxImageMatcher::restore_rect(const cv::Rect& rect, int dst_width, int dst_height) const {
    return resize_rect(rect, width_, height_, dst_width, dst_height);
}

bool OnnxImageMatcher::locate_box(const cv::Rect& src_box, int src_width, int src_height,
                                   const MatchResult& matches, cv::Rect& target_box) const {
    target_box = cv::Rect();
    if (matches.matched_left.size() != matches.matched_right.size() ||
        matches.matched_left.size() < static_cast<size_t>(min_global_matches_)) {
        return false;
    }

    const cv::Rect model_box = resize_rect(src_box, src_width, src_height, width_, height_);
    std::vector<cv::Point2f> source_points;
    std::vector<cv::Point2f> target_points;
    for (size_t index = 0; index < matches.matched_left.size(); ++index) {
        if (point_in_rect(matches.matched_left[index], model_box)) {
            source_points.push_back(matches.matched_left[index]);
            target_points.push_back(matches.matched_right[index]);
        }
    }
    if (source_points.size() < static_cast<size_t>(min_box_matches_)) {
        return false;
    }

    const cv::Mat homography = cv::findHomography(source_points, target_points, cv::RANSAC, 3.0);
    if (homography.empty()) {
        return false;
    }

    std::vector<cv::Point2f> corners = {
        cv::Point2f(static_cast<float>(model_box.x), static_cast<float>(model_box.y)),
        cv::Point2f(static_cast<float>(model_box.x + model_box.width), static_cast<float>(model_box.y)),
        cv::Point2f(static_cast<float>(model_box.x + model_box.width), static_cast<float>(model_box.y + model_box.height)),
        cv::Point2f(static_cast<float>(model_box.x), static_cast<float>(model_box.y + model_box.height))};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(corners, projected, homography);
    if (projected.size() != 4) return false;

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const cv::Point2f& point : projected) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }
    const cv::Rect projected_box(static_cast<int>(std::floor(min_x)),
                                 static_cast<int>(std::floor(min_y)),
                                 static_cast<int>(std::ceil(max_x - min_x)),
                                 static_cast<int>(std::ceil(max_y - min_y)));
    if (projected_box.width <= 2 || projected_box.height <= 2) return false;

    target_box = restore_rect(projected_box, src_width, src_height);
    return target_box.width > 2 && target_box.height > 2;
}

OnnxLdc::OnnxLdc(const std::string& model_path) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    session_ = create_session(model_path, options);
    input_names_ = session_input_names(*session_);
    output_names_ = session_output_names(*session_);
    require(input_names_.size() == 1, "LDC model must have exactly one input");
    require(!output_names_.empty(), "LDC model has no outputs");
    require(input_names_[0] == "input_image", "Unexpected LDC ONNX input name");

    const std::vector<int64_t> input_shape =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    require(input_shape.size() == 4 && input_shape[0] == 1 && input_shape[1] == 3,
            "LDC input must have shape [1,3,H,W]");
    input_height_ = static_cast<int>(input_shape[2]);
    input_width_ = static_cast<int>(input_shape[3]);
    require(input_width_ > 0 && input_height_ > 0, "LDC input shape contains dynamic dimensions");
}

void OnnxLdc::detect(const cv::Mat& source, cv::Mat& edge_image) {
    require(!source.empty() && source.channels() == 3, "LDC requires a non-empty BGR image");
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(input_width_, input_height_));
    std::vector<float> input(static_cast<size_t>(3 * input_width_ * input_height_));
    for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < input_height_; ++row) {
            for (int column = 0; column < input_width_; ++column) {
                input[static_cast<size_t>(channel * input_height_ * input_width_ + row * input_width_ + column)] =
                    static_cast<float>(resized.ptr<uchar>(row)[column * 3 + (2 - channel)]);
            }
        }
    }

    const std::vector<int64_t> shape = {1, 3, input_height_, input_width_};
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), shape.data(), shape.size());
    const std::vector<const char*> input_views = c_string_views(input_names_);
    const std::vector<const char*> output_views = c_string_views(output_names_);
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_views.data(), &tensor, 1,
        output_views.data(), output_views.size());

    cv::Mat accumulator = cv::Mat::zeros(source.rows, source.cols, CV_32FC1);
    int valid_outputs = 0;
    for (Ort::Value& output : outputs) {
        const Ort::TensorTypeAndShapeInfo info = output.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> output_shape = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_shape.size() != 4 || output_shape[2] <= 0 || output_shape[3] <= 0) {
            continue;
        }
        const int output_height = static_cast<int>(output_shape[2]);
        const int output_width = static_cast<int>(output_shape[3]);
        const float* values = output.GetTensorData<float>();
        cv::Mat logits(output_height, output_width, CV_32FC1, const_cast<float*>(values));
        cv::Mat exponent;
        cv::exp(-logits, exponent);
        cv::Mat mask = 1.0 / (1.0 + exponent);
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(mask, &min_value, &max_value);
        mask = (mask - min_value) * 255.0 / (max_value - min_value + 1e-12);
        mask.convertTo(mask, CV_8UC1);
        cv::bitwise_not(mask, mask);
        cv::resize(mask, mask, source.size());
        cv::accumulate(mask, accumulator);
        ++valid_outputs;
    }
    require(valid_outputs > 0, "LDC returned no usable float edge outputs");
    accumulator /= static_cast<float>(valid_outputs);
    accumulator.convertTo(edge_image, CV_8UC1);
}

OnnxDeepDifferenceDetector::OnnxDeepDifferenceDetector(const std::string& model_path) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = create_session(model_path, options);
    input_names_ = session_input_names(*session_);
    output_names_ = session_output_names(*session_);
    require(input_names_.size() == 1, "ResNet18 feature model must have exactly one input");
    require(!output_names_.empty(), "ResNet18 feature model has no outputs");
    require(input_names_[0] == "input" && output_names_[0] == "output",
            "Unexpected ResNet18 ONNX node names");
}

cv::Rect OnnxDeepDifferenceDetector::template_matching(const cv::Mat& large_image,
                                                        const cv::Mat& small_image,
                                                        double& max_value, int method) {
    require(!large_image.empty() && !small_image.empty(), "Template matching received an empty image");
    require(small_image.rows <= large_image.rows && small_image.cols <= large_image.cols,
            "Template image is larger than search image");

    int scale = (large_image.cols > 1000 || large_image.rows > 1000) ? 4 : 1;
    cv::Mat search = large_image;
    cv::Mat templ = small_image;
    if (scale > 1) {
        cv::resize(large_image, search, cv::Size(large_image.cols / scale, large_image.rows / scale));
        cv::resize(small_image, templ, cv::Size(small_image.cols / scale, small_image.rows / scale));
        if (templ.rows >= search.rows || templ.cols >= search.cols) {
            scale = 1;
            search = large_image;
            templ = small_image;
        }
    }

    cv::Mat result;
    cv::matchTemplate(search, templ, result, method);
    double min_value = 0.0;
    cv::Point min_location;
    cv::Point max_location;
    cv::minMaxLoc(result, &min_value, &max_value, &min_location, &max_location);
    const cv::Point coarse = (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) ?
        min_location : max_location;
    if (scale == 1) {
        return cv::Rect(coarse.x, coarse.y, small_image.cols, small_image.rows);
    }

    const int offset = scale * 2;
    const int rough_x = coarse.x * scale;
    const int rough_y = coarse.y * scale;
    const int roi_x = std::max(0, rough_x - offset);
    const int roi_y = std::max(0, rough_y - offset);
    const int roi_width = std::min(large_image.cols - roi_x, small_image.cols + 2 * offset);
    const int roi_height = std::min(large_image.rows - roi_y, small_image.rows + 2 * offset);
    cv::Mat roi = large_image(cv::Rect(roi_x, roi_y, roi_width, roi_height));
    cv::matchTemplate(roi, small_image, result, method);
    cv::minMaxLoc(result, &min_value, &max_value, &min_location, &max_location);
    const cv::Point fine = (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) ?
        min_location : max_location;
    return cv::Rect(roi_x + fine.x, roi_y + fine.y, small_image.cols, small_image.rows);
}

std::pair<cv::Mat, cv::Mat> OnnxDeepDifferenceDetector::preprocess_image(
    const cv::Mat& image, const cv::Size& target_size) const {
    require(!image.empty(), "ResNet18 requires a non-empty image");
    cv::Mat rgb;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 1) {
        // 主流程会将 LDC 的单通道轮廓图送入 ResNet 计算结构相似度；
        // 复制到 RGB 三个通道即可满足 ImageNet 归一化和 [1,3,H,W] 模型输入。
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    } else {
        throw std::runtime_error("ResNet18 accepts only one-channel edge images or three-channel BGR images");
    }
    cv::Mat resized_rgb;
    cv::resize(rgb, resized_rgb, target_size, 0, 0, cv::INTER_AREA);
    cv::Mat resized_bgr;
    cv::resize(image, resized_bgr, target_size, 0, 0, cv::INTER_AREA);
    return std::make_pair(resized_rgb, resized_bgr);
}

cv::Mat OnnxDeepDifferenceDetector::extract_features(const cv::Mat& rgb_image) {
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);
    for (int channel = 0; channel < 3; ++channel) {
        channels[channel] = (channels[channel] - mean_[channel]) / std_dev_[channel];
    }

    std::vector<float> input;
    input.reserve(static_cast<size_t>(3 * rgb_image.rows * rgb_image.cols));
    for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < rgb_image.rows; ++row) {
            const float* ptr = channels[channel].ptr<float>(row);
            input.insert(input.end(), ptr, ptr + rgb_image.cols);
        }
    }
    const std::vector<int64_t> shape = {1, 3, rgb_image.rows, rgb_image.cols};
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), shape.data(), shape.size());
    const std::vector<const char*> input_views = c_string_views(input_names_);
    const std::vector<const char*> output_views = c_string_views(output_names_);
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_views.data(), &tensor, 1,
        output_views.data(), output_views.size());
    require(!outputs.empty(), "ResNet18 returned no outputs");

    const Ort::TensorTypeAndShapeInfo info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> output_shape = info.GetShape();
    require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            output_shape.size() == 4 && output_shape[0] == 1 && output_shape[1] > 0 &&
            output_shape[2] > 0 && output_shape[3] > 0,
            "ResNet18 output must be float [1,C,H,W]");
    const int channels_out = static_cast<int>(output_shape[1]);
    feature_height_ = static_cast<int>(output_shape[2]);
    feature_width_ = static_cast<int>(output_shape[3]);
    const float* values = outputs[0].GetTensorData<float>();

    cv::Mat features(feature_height_ * feature_width_, channels_out, CV_32F);
    for (int channel = 0; channel < channels_out; ++channel) {
        for (int row = 0; row < feature_height_; ++row) {
            for (int column = 0; column < feature_width_; ++column) {
                const int source_index = channel * feature_height_ * feature_width_ + row * feature_width_ + column;
                features.at<float>(row * feature_width_ + column, channel) = values[source_index];
            }
        }
    }
    return features;
}

cv::Mat OnnxDeepDifferenceDetector::compute_dense_difference(const cv::Mat& left_features,
                                                              const cv::Mat& right_features,
                                                              const cv::Size& output_size) const {
    require(left_features.size() == right_features.size(), "Feature maps have different shapes");
    require(feature_height_ > 0 && feature_width_ > 0 &&
            feature_height_ * feature_width_ == left_features.rows,
            "ResNet18 feature-map geometry is invalid");
    const cv::Mat left_normalized = row_l2_normalize(left_features);
    const cv::Mat right_normalized = row_l2_normalize(right_features);
    cv::Mat absolute_difference;
    cv::absdiff(left_normalized, right_normalized, absolute_difference);

    cv::Mat mean_difference(left_features.rows, 1, CV_32F);
    for (int row = 0; row < absolute_difference.rows; ++row) {
        mean_difference.at<float>(row, 0) = static_cast<float>(cv::mean(absolute_difference.row(row))[0]);
    }
    cv::Mat spatial = mean_difference.reshape(1, feature_height_);
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(spatial, &min_value, &max_value);
    spatial = (spatial - min_value) / (max_value - min_value + 1e-8);
    cv::Mat upsampled;
    cv::resize(spatial, upsampled, output_size, 0, 0, cv::INTER_LINEAR);
    return upsampled;
}

float OnnxDeepDifferenceDetector::compute_similarity(const cv::Mat& left, const cv::Mat& right,
                                                      const cv::Size& target_size) {
    const std::pair<cv::Mat, cv::Mat> left_preprocessed = preprocess_image(left, target_size);
    const std::pair<cv::Mat, cv::Mat> right_preprocessed = preprocess_image(right, target_size);
    const cv::Mat left_features = extract_features(left_preprocessed.first);
    const cv::Mat right_features = extract_features(right_preprocessed.first);
    return global_cosine_similarity(left_features, right_features);
}

std::vector<cv::Rect> OnnxDeepDifferenceDetector::detect_differences(
    const cv::Mat& left, const cv::Mat& right, float difference_threshold,
    float roi_difference_threshold, int min_contour_area, const cv::Size& target_size) {
    require(!left.empty() && !right.empty(), "Difference detector received an empty image");
    const int source_width = left.cols;
    const int source_height = left.rows;
    const std::pair<cv::Mat, cv::Mat> left_preprocessed = preprocess_image(left, target_size);
    const std::pair<cv::Mat, cv::Mat> right_preprocessed = preprocess_image(right, target_size);
    const cv::Mat left_features = extract_features(left_preprocessed.first);
    const cv::Mat right_features = extract_features(right_preprocessed.first);
    const cv::Mat difference = compute_dense_difference(
        left_features, right_features, cv::Size(target_size.width, target_size.height));

    cv::Mat smoothed;
    cv::GaussianBlur(difference, smoothed, cv::Size(5, 5), 3);
    cv::Mat binary = smoothed > difference_threshold;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> result;
    for (const std::vector<cv::Point>& contour : contours) {
        if (cv::contourArea(contour) <= min_contour_area) continue;
        const cv::Rect rect = cv::boundingRect(contour);
        const float roi_difference = static_cast<float>(cv::mean(difference(rect))[0]);
        if (roi_difference <= roi_difference_threshold) continue;
        result.emplace_back(
            static_cast<int>(static_cast<float>(rect.x) / target_size.width * source_width),
            static_cast<int>(static_cast<float>(rect.y) / target_size.height * source_height),
            static_cast<int>(static_cast<float>(rect.width) / target_size.width * source_width),
            static_cast<int>(static_cast<float>(rect.height) / target_size.height * source_height));
    }
    return result;
}

}  // namespace onnx
}  // namespace imagecmp
