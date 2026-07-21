// ResNet18 特征差异检测器。
//
// computer_sim 给出两个部件的全局特征余弦相似度；detect_differences 则生成
// 局部稠密差异图，并返回差异区域的像素框。
#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <utility>
#include <vector>

#include "ascend_infer.h"

class DeepDifferenceDetector {
public:
    explicit DeepDifferenceDetector(const std::string& om_model_path = "resnet18.om", uint32_t device_id = 0);

    // BGR 转 RGB、缩放，返回模型输入图及用于尺寸映射的缩放后 BGR 图。
    std::pair<cv::Mat, cv::Mat> preprocess_image(const cv::Mat& img, const cv::Size& target_size = cv::Size(224, 224));
    cv::Mat extract_features(const cv::Mat& pil_image);
    cv::Mat compute_dense_difference(const cv::Mat& feat_map1, const cv::Mat& feat_map2, const cv::Size& original_img_shape);

    // 以差异阈值、ROI 平均差异和最小轮廓面积过滤，输出原始输入尺寸下的框。
    std::vector<cv::Rect> detect_differences(const cv::Mat& srcimg1,
                                             const cv::Mat& srcimg2,
                                             float difference_threshold = 0.5f,
                                             float roi_diff_threshold = 0.2f,
                                             int min_contour_area = 1000,
                                             const cv::Size& target_img_size = cv::Size(224, 224));

    std::vector<cv::Rect> readYoloFormatToRects(const std::string& filePath, int imageWidth, int imageHeight);

    // 在大图中寻找小图；maxVal 为定位得分，而非“无异动”得分。
    static cv::Rect templateMatching(const cv::Mat& largeImg,
                                     const cv::Mat& smallImg,
                                     double& maxVal,
                                     int method = cv::TM_CCOEFF_NORMED);

    // 返回全局特征余弦相似度；该分数衡量整体结构相近程度。
    float computer_sim(const cv::Mat& srcimg1, const cv::Mat& srcimg2, const cv::Size& target_img_size = cv::Size(224, 224));

private:
    uint32_t device_id_;
    AscendRuntime ascend_;
    AclModel model_;

    const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std_dev_ = {0.229f, 0.224f, 0.225f};
};
