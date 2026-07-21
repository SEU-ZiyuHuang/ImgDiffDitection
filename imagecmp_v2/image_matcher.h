// SuperPoint + LightGlue 部件定位器。
// 输入样本大图和实时大图，取得关键点匹配后把样本 YOLO 框投影到实时图。
// 它的职责是“定位同一部件”，不是直接判定部件是否发生异动。
#pragma once

#include <opencv2/opencv.hpp>

#include <mutex>
#include <string>
#include <vector>

#include "ascend_infer.h"

struct MatchResult {
    // 左/右图所有关键点，以及通过置信度和距离筛选后的匹配点对。
    std::vector<cv::Point2f> keypoints_left;
    std::vector<cv::Point2f> keypoints_right;
    std::vector<cv::Point2f> matched_left;
    std::vector<cv::Point2f> matched_right;
};

class ImageMatcher {
private:
    std::string model_path_;
    int height_;
    int width_;

    float DISTANCE_THRESHOLD = 150.0f;
    float CONFIDENCE_THRESHOLD = 0.5f;
    int MAX_KEYPOINTS = 1024;
    int MIN_GLOBAL_MATCHES = 150;
    int MIN_BOX_MATCHES = 5;
    int MAX_BOX_AREA = 2000;

    uint32_t device_id_;
    AscendRuntime ascend_;
    AclModel model_;
    std::mutex session_mutex_;

public:
    explicit ImageMatcher(const std::string& model_path, uint32_t device_id = 0);

    // 执行组合模型，输出样本图(left)到实时图(right)的匹配关系。
    MatchResult get_matches(const cv::Mat& left_img, const cv::Mat& right_img);

    cv::Rect resizePoint(const cv::Rect& srcRect, int srcW, int srcH, int dstW, int dstH);

    // 用样本框内的特征和单应性矩阵，计算实时图中的目标框。
    bool feature_match(int x_min, int y_min, int x_max, int y_max, std::vector<cv::Point2f>& matched_kp_l,
                       std::vector<cv::Point2f>& match_l, std::vector<cv::Point2f>& match_r, cv::Rect& tarrec);

    bool feature_match(cv::Rect srcbox, int bW, int bH, std::vector<cv::Point2f>& matched_kp_l,
                       std::vector<cv::Point2f>& match_l, std::vector<cv::Point2f>& match_r, cv::Rect& tarrec);

    cv::Rect outRect(cv::Rect srcRect, int dstW, int dstH);

private:
    std::vector<float> preprocess_one(const cv::Mat& img);
};
