// ONNX Runtime 后端的主编排与 C ABI 实现。
//
// 该文件与 yidong_main.cpp 对应，但不依赖 Ascend CANN 或 .om 文件。它复用
// imagecmp.cpp 的 YOLO 坐标转换/候选框合并工具，并以 CPU ONNX Runtime 运行
// SuperPoint+LightGlue、LDC 和 ResNet18 三个 ONNX 模型。
//
// 关键约束：模板匹配的高分只表示“在实时图中找到了该部件”，绝不直接表示
// “无异动”。只要定位成功，仍必须执行 LDC + 稠密特征差异检测。
#include "imagecmp.h"
#include "base64.h"
#include "onnx_runtime/onnx_pipeline.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// imagecmp.cpp 中的公共图像工具。它们不在公共头文件暴露，但会与本文件一起编译。
extern cv::Rect merge_rectangles(const std::vector<cv::Rect>& rectangles, int threshold);
extern cv::Mat resizeAndPad(const cv::Mat& input, int target_size);

namespace {

using imagecmp::onnx::MatchResult;
using imagecmp::onnx::OnnxDeepDifferenceDetector;
using imagecmp::onnx::OnnxImageMatcher;
using imagecmp::onnx::OnnxLdc;

struct ModelPipeline {
    std::unique_ptr<OnnxImageMatcher> matcher;
    std::unique_ptr<OnnxLdc> ldc;
    std::unique_ptr<OnnxDeepDifferenceDetector> detector;
};

// ONNX Runtime Session 可被多个线程调用，但三个模型一次检测会形成一个完整事务。
// 池化能够限制 CPU 并发、避免同一请求的三个 Session 与其他请求交错占满线程池。
class ModelPool {
public:
    void init(int requested_instances) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 若 lxUninit 正在等待旧请求结束，必须等待旧池完全释放后才能创建新池。
        cv_.wait(lock, [this] { return !destroying_; });
        if (initialized_) return;

        const int instances = std::max(1, requested_instances);
        std::queue<std::unique_ptr<ModelPipeline> > new_pool;
        for (int index = 0; index < instances; ++index) {
            std::unique_ptr<ModelPipeline> pipeline(new ModelPipeline());
            pipeline->matcher.reset(new OnnxImageMatcher(model_path("superpoint_lightglue_pipeline.onnx")));
            pipeline->ldc.reset(new OnnxLdc(model_path("LDC_640x360.onnx")));
            pipeline->detector.reset(new OnnxDeepDifferenceDetector(model_path("resnet18.onnx")));
            new_pool.push(std::move(pipeline));
        }
        pool_.swap(new_pool);
        initialized_ = true;
        cv_.notify_all();
    }

    std::unique_ptr<ModelPipeline> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !initialized_ || !pool_.empty(); });
        if (!initialized_) {
            throw std::runtime_error("ONNX model pool is not initialized");
        }
        std::unique_ptr<ModelPipeline> pipeline = std::move(pool_.front());
        pool_.pop();
        ++active_requests_;
        return pipeline;
    }

    void release(std::unique_ptr<ModelPipeline> pipeline) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_requests_ > 0) --active_requests_;
        // lxUninit 等待在途请求归还，故这里始终回收实例；随后 destroy 会统一释放。
        if (pipeline) pool_.push(std::move(pipeline));
        cv_.notify_all();
    }

    void destroy() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!initialized_) return;
        // 先拒绝新的 acquire，再等待正在执行的调用归还流水线，避免 uninit 与推理竞争。
        destroying_ = true;
        initialized_ = false;
        cv_.notify_all();
        cv_.wait(lock, [this] { return active_requests_ == 0; });
        while (!pool_.empty()) pool_.pop();
        destroying_ = false;
        cv_.notify_all();
    }

    bool initialized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_;
    }

private:
    static std::string model_path(const char* model_name) {
        const char* configured_dir = std::getenv("IMAGECMP_MODEL_DIR");
        const std::string directory = (configured_dir && configured_dir[0]) ? configured_dir : ".";
        const char last = directory.empty() ? '\0' : directory[directory.size() - 1];
        const std::string separator = (last == '/' || last == '\\') ? "" : "/";
        const std::string path = directory + separator + model_name;
        std::ifstream stream(path.c_str(), std::ios::binary);
        if (!stream.good()) {
            throw std::runtime_error("ONNX model file not found: " + path +
                                     " (set IMAGECMP_MODEL_DIR to the model directory)");
        }
        return path;
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::unique_ptr<ModelPipeline> > pool_;
    bool initialized_ = false;
    bool destroying_ = false;
    size_t active_requests_ = 0;
};

ModelPool g_model_pool;
thread_local std::string g_last_error_message;

void set_last_error(const std::string& message) {
    g_last_error_message = message;
}

void ensure_models_loaded() {
    if (!g_model_pool.initialized()) {
        g_model_pool.init(1);
    }
}

class PipelineGuard {
public:
    PipelineGuard() : pipeline_(g_model_pool.acquire()) {}
    ~PipelineGuard() {
        if (pipeline_) g_model_pool.release(std::move(pipeline_));
    }
    ModelPipeline& get() { return *pipeline_; }

private:
    std::unique_ptr<ModelPipeline> pipeline_;
};

bool valid_rect(const cv::Rect& rect, const cv::Mat& image, int min_edge = 1) {
    return rect.x >= 0 && rect.y >= 0 && rect.width >= min_edge && rect.height >= min_edge &&
           rect.x + rect.width <= image.cols && rect.y + rect.height <= image.rows;
}

float rect_iou(const cv::Rect& first, const cv::Rect& second) {
    const cv::Rect intersection = first & second;
    if (intersection.empty()) return 0.0f;
    const float union_area = static_cast<float>(first.area() + second.area() - intersection.area());
    return union_area > 0.0f ? static_cast<float>(intersection.area()) / union_area : 0.0f;
}

cv::Rect yolo_rect(float yolo_x, float yolo_y, float yolo_w, float yolo_h,
                   const cv::Mat& image) {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    yolo2rect(yolo_x, yolo_y, yolo_w, yolo_h, image.cols, image.rows, x, y, width, height);
    const cv::Rect result(x, y, width, height);
    if (!valid_rect(result, image, 4)) {
        throw std::invalid_argument("invalid or out-of-range YOLO rectangle");
    }
    return result;
}

void rect_to_yolo(const cv::Rect& rect, const cv::Mat& image, float output[4]) {
    output[0] = (rect.x + rect.width * 0.5f) / image.cols;
    output[1] = (rect.y + rect.height * 0.5f) / image.rows;
    output[2] = rect.width / static_cast<float>(image.cols);
    output[3] = rect.height / static_cast<float>(image.rows);
}

void clear_result(float* similarity, Box* box, float live_rect_yolo[4],
                  char** live_b64, char** temp_b64) {
    if (similarity) *similarity = 0.0f;
    if (box) std::memset(box, 0, sizeof(Box));
    if (live_rect_yolo) std::memset(live_rect_yolo, 0, sizeof(float) * 4);
    if (live_b64) *live_b64 = 0;
    if (temp_b64) *temp_b64 = 0;
}

void save_crop(const char* output_path, const cv::Mat& image, const cv::Rect& rect, int resize) {
    if (!output_path || output_path[0] == '\0') return;
    if (!valid_rect(rect, image)) {
        throw std::invalid_argument("cannot save an out-of-range crop rectangle");
    }
    cv::Mat crop = image(rect).clone();
    if (resize > 0) crop = resizeAndPad(crop, resize);
    if (!cv::imwrite(output_path, crop)) {
        throw std::runtime_error(std::string("failed to write crop image: ") + output_path);
    }
}

char* jpeg_base64(const cv::Mat& image) {
    std::vector<uchar> encoded;
    if (!cv::imencode(".jpg", image, encoded, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 90})) {
        throw std::runtime_error("failed to JPEG encode crop image");
    }
    const std::string text = "data:image/jpeg;base64," + Base64Encode(encoded.data(),
        static_cast<unsigned int>(encoded.size()));
    char* result = static_cast<char*>(std::malloc(text.size() + 1));
    if (!result) throw std::bad_alloc();
    std::memcpy(result, text.c_str(), text.size() + 1);
    return result;
}

// 返回实时图中的部件框。模板匹配仅作为快速定位；位置置信度不足时回退到
// SuperPoint+LightGlue。调用方随后始终执行异动检测，不能根据 score 提前返回。
bool locate_component(ModelPipeline& pipeline, const cv::Mat& live_image, const cv::Mat& temp_image,
                      const cv::Mat& temp_crop, const cv::Rect& temp_rect, cv::Rect& live_rect,
                      double& location_score) {
    cv::Rect template_rect = OnnxDeepDifferenceDetector::template_matching(
        live_image, temp_crop, location_score, cv::TM_CCOEFF_NORMED);
    if (valid_rect(template_rect, live_image) && location_score >= 0.85 &&
        rect_iou(template_rect, temp_rect) >= 0.10f) {
        live_rect = template_rect;
        return true;
    }

    const MatchResult matches = pipeline.matcher->get_matches(temp_image, live_image);
    if (!pipeline.matcher->locate_box(temp_rect, temp_image.cols, temp_image.rows, matches, live_rect)) {
        return false;
    }
    return valid_rect(live_rect, live_image, 4);
}

std::vector<cv::Rect> detect_component_changes(ModelPipeline& pipeline, const cv::Mat& live_crop,
                                                const cv::Mat& temp_crop, float* similarity) {
    cv::Mat live_edges;
    cv::Mat temp_edges;
    pipeline.ldc->detect(live_crop, live_edges);
    pipeline.ldc->detect(temp_crop, temp_edges);
    if (similarity) *similarity = pipeline.detector->compute_similarity(live_edges, temp_edges);

    // 注意：不再以 similarity/threshold 直接短路。全局特征相似时，小面积异动仍可能存在；
    // 稠密差异图正是用来定位这类局部变化的。
    return pipeline.detector->detect_differences(live_crop, temp_crop, 0.70f, 0.70f, 500,
                                                  cv::Size(224, 224));
}

int copy_difference_result(const std::vector<cv::Rect>& candidates, const cv::Rect& live_rect,
                           Box* output) {
    const cv::Rect merged = merge_rectangles(candidates, 50);
    if (merged.empty()) return 0;

    const float crop_area = static_cast<float>(live_rect.area());
    const float diff_area = static_cast<float>(merged.area());
    // 过滤过小的噪声和几乎覆盖整个部件的失配结果；两者都不输出“具体异动框”。
    if (crop_area <= 0.0f || diff_area < 100.0f || diff_area / crop_area > 0.80f) return 0;

    output->x = live_rect.x + merged.x;
    output->y = live_rect.y + merged.y;
    output->w = merged.width;
    output->h = merged.height;
    return 1;
}

int main_detection(const char* live_image_path, const char* temp_image_path,
                   float yolo_x, float yolo_y, float yolo_w, float yolo_h,
                   float threshold, float* similarity, Box* output_box,
                   const char* output_crop_path, int resize, float live_rect_yolo[4],
                   char** live_b64, char** temp_b64, int generate_base64) {
    (void)threshold;  // 保留 ABI 兼容；不再把全局相似度作为局部异动的短路条件。
    if (!live_image_path || !temp_image_path || !similarity || !output_box || !live_rect_yolo ||
        !live_b64 || !temp_b64) {
        set_last_error("lxImageCmpOnnx received a null required output/input pointer");
        return -7;
    }
    clear_result(similarity, output_box, live_rect_yolo, live_b64, temp_b64);

    const cv::Mat live_image = cv::imread(live_image_path, cv::IMREAD_COLOR);
    const cv::Mat temp_image = cv::imread(temp_image_path, cv::IMREAD_COLOR);
    if (live_image.empty() || temp_image.empty()) {
        set_last_error("failed to read live or template image");
        return -2;
    }
    if (live_image.size() != temp_image.size()) {
        set_last_error("live and template images must have the same dimensions");
        return -3;
    }

    cv::Rect temp_rect;
    try {
        temp_rect = yolo_rect(yolo_x, yolo_y, yolo_w, yolo_h, temp_image);
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return -4;
    }
    const cv::Mat temp_crop = temp_image(temp_rect).clone();

    try {
        PipelineGuard guard;
        cv::Rect live_rect;
        double location_score = 0.0;
        if (!locate_component(guard.get(), live_image, temp_image, temp_crop, temp_rect,
                              live_rect, location_score)) {
            set_last_error("component location failed: both template and feature matching lacked evidence");
            return -6;
        }
        if (!valid_rect(live_rect, live_image, 4)) {
            set_last_error("located component rectangle is out of range");
            return -5;
        }
        rect_to_yolo(live_rect, live_image, live_rect_yolo);
        const cv::Mat live_crop = live_image(live_rect).clone();

        const std::vector<cv::Rect> candidates = detect_component_changes(
            guard.get(), live_crop, temp_crop, similarity);
        save_crop(output_crop_path, live_image, live_rect, resize);

        if (copy_difference_result(candidates, live_rect, output_box) == 0) {
            return 0;
        }
        if (generate_base64) {
            // 先构造临时指针，两个编码均成功后再交给调用方，避免部分成功导致泄漏。
            std::unique_ptr<char, decltype(&std::free)> temp_encoded(jpeg_base64(temp_crop), &std::free);
            std::unique_ptr<char, decltype(&std::free)> live_encoded(jpeg_base64(live_crop), &std::free);
            *temp_b64 = temp_encoded.release();
            *live_b64 = live_encoded.release();
        }
        return 1;
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        clear_result(similarity, output_box, live_rect_yolo, live_b64, temp_b64);
        return -1;
    }
}

}  // namespace

extern "C" {

LXAPI int lxGetLastErrorMessage(char** message) {
    if (!message) return -1;
    *message = 0;
    if (g_last_error_message.empty()) return 0;
    char* copy = static_cast<char*>(std::malloc(g_last_error_message.size() + 1));
    if (!copy) return -2;
    std::memcpy(copy, g_last_error_message.c_str(), g_last_error_message.size() + 1);
    *message = copy;
    return 1;
}

LXAPI void lxFreePtr(char* pointer) {
    std::free(pointer);
}

LXAPI int lxInitAIModel(const int* device_ids, int device_count, int instances_per_device) {
    (void)device_ids;  // ONNX Runtime CPU 后端不使用 Ascend 设备 ID，保留原 ABI。
    set_last_error("");
    try {
        const int count = device_count > 0 ? device_count : 1;
        g_model_pool.init(std::max(1, count * std::max(1, instances_per_device)));
        return 1;
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return 0;
    }
}

LXAPI int lxUninit() {
    set_last_error("");
    try {
        g_model_pool.destroy();
        return 1;
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return 0;
    }
}

LXAPI int lxImageCmpOnnx(const char* live_image_path, const char* temp_image_path,
                          float yolo_x, float yolo_y, float yolo_w, float yolo_h,
                          float threshold, float* similarity, Box* output_box,
                          const char* output_crop_path, int resize, float live_rect_yolo[4],
                          char** live_b64, char** temp_b64, int generate_base64) {
    set_last_error("");
    if (!live_image_path || !temp_image_path || !similarity || !output_box || !live_rect_yolo ||
        !live_b64 || !temp_b64) {
        set_last_error("lxImageCmpOnnx received a null required output/input pointer");
        return -7;
    }
    // 即使模型文件缺失或加载失败，也不给调用方留下未初始化的输出字段。
    clear_result(similarity, output_box, live_rect_yolo, live_b64, temp_b64);
    try {
        ensure_models_loaded();
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return -1;
    }
    return main_detection(live_image_path, temp_image_path, yolo_x, yolo_y, yolo_w, yolo_h,
                          threshold, similarity, output_box, output_crop_path, resize,
                          live_rect_yolo, live_b64, temp_b64, generate_base64);
}

LXAPI int lxImageYolo2Rect(const char* source_path, float yolo_x, float yolo_y,
                            float yolo_w, float yolo_h, int* x, int* y, int* width, int* height) {
    if (!source_path || !x || !y || !width || !height) return -7;
    const cv::Mat image = cv::imread(source_path, cv::IMREAD_COLOR);
    if (image.empty()) return -2;
    try {
        const cv::Rect rect = yolo_rect(yolo_x, yolo_y, yolo_w, yolo_h, image);
        *x = rect.x; *y = rect.y; *width = rect.width; *height = rect.height;
        return 1;
    } catch (...) {
        return -4;
    }
}

LXAPI int lxImageDraw(const char* source_path, int x, int y, int width, int height,
                      const char* output_path) {
    if (!source_path || !output_path) return -7;
    try {
        cv::Mat image = cv::imread(source_path, cv::IMREAD_COLOR);
        const cv::Rect rect(x, y, width, height);
        if (image.empty() || !valid_rect(rect, image)) return -2;
        cv::rectangle(image, rect, cv::Scalar(0, 255, 255), 2);
        return cv::imwrite(output_path, image) ? 1 : -1;
    } catch (...) {
        return -1;
    }
}

LXAPI int lxImageCrop(const char* source_path, int x, int y, int width, int height,
                      const char* output_path) {
    if (!source_path || !output_path) return -7;
    try {
        const cv::Mat image = cv::imread(source_path, cv::IMREAD_COLOR);
        const cv::Rect rect(x, y, width, height);
        if (image.empty() || !valid_rect(rect, image)) return -2;
        return cv::imwrite(output_path, image(rect)) ? 1 : -1;
    } catch (...) {
        return -1;
    }
}

LXAPI int lxImageOnnxLDC(const char* source_path, const char* output_path) {
    if (!source_path || !output_path) return -7;
    try {
        ensure_models_loaded();
        const cv::Mat image = cv::imread(source_path, cv::IMREAD_COLOR);
        if (image.empty()) return -2;
        PipelineGuard guard;
        cv::Mat edges;
        guard.get().ldc->detect(image, edges);
        return cv::imwrite(output_path, edges) ? 1 : -1;
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return -1;
    }
}

LXAPI int lxImage_MatcherDet(const char* live_image_path, const char* temp_image_path,
                             float yolo_x, float yolo_y, float yolo_w, float yolo_h,
                             Box* output_box) {
    if (!live_image_path || !temp_image_path || !output_box) return -7;
    std::memset(output_box, 0, sizeof(Box));
    try {
        ensure_models_loaded();
        const cv::Mat live_image = cv::imread(live_image_path, cv::IMREAD_COLOR);
        const cv::Mat temp_image = cv::imread(temp_image_path, cv::IMREAD_COLOR);
        if (live_image.empty() || temp_image.empty()) return -2;
        if (live_image.size() != temp_image.size()) return -3;
        const cv::Rect temp_rect = yolo_rect(yolo_x, yolo_y, yolo_w, yolo_h, temp_image);
        PipelineGuard guard;
        const MatchResult matches = guard.get().matcher->get_matches(temp_image, live_image);
        cv::Rect live_rect;
        if (!guard.get().matcher->locate_box(temp_rect, temp_image.cols, temp_image.rows,
                                             matches, live_rect) ||
            !valid_rect(live_rect, live_image, 4)) return 0;
        output_box->x = live_rect.x; output_box->y = live_rect.y;
        output_box->w = live_rect.width; output_box->h = live_rect.height;
        return 1;
    } catch (const std::invalid_argument&) {
        return -4;
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return -1;
    }
}

LXAPI int lxImage_DeepDifferenceDetector(const char* live_image_path, const char* temp_image_path,
                                         float yolo_x, float yolo_y, float yolo_w, float yolo_h,
                                         int live_x, int live_y, int live_width, int live_height,
                                         float* out_score, Box* out_box) {
    if (!live_image_path || !temp_image_path || !out_score || !out_box) return -7;
    *out_score = 0.0f;
    std::memset(out_box, 0, sizeof(Box));
    try {
        ensure_models_loaded();
        const cv::Mat live_image = cv::imread(live_image_path, cv::IMREAD_COLOR);
        const cv::Mat temp_image = cv::imread(temp_image_path, cv::IMREAD_COLOR);
        if (live_image.empty() || temp_image.empty()) return -2;
        if (live_image.size() != temp_image.size()) return -3;
        const cv::Rect temp_rect = yolo_rect(yolo_x, yolo_y, yolo_w, yolo_h, temp_image);
        const cv::Rect live_rect(live_x, live_y, live_width, live_height);
        if (!valid_rect(live_rect, live_image, 4)) return -5;
        PipelineGuard guard;
        const std::vector<cv::Rect> candidates = detect_component_changes(
            guard.get(), live_image(live_rect).clone(), temp_image(temp_rect).clone(), out_score);
        return copy_difference_result(candidates, live_rect, out_box);
    } catch (const std::invalid_argument&) {
        return -4;
    } catch (const std::exception& exception) {
        set_last_error(exception.what());
        return -1;
    }
}

}  // extern "C"
