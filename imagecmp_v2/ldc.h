// LDC（Lightweight Dense CNN）边缘/轮廓提取器。
// 输入 BGR 部件图，输出同尺寸单通道轮廓强度图，供后续结构相似度计算。
#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "ascend_infer.h"

class LDC {
public:
    explicit LDC(std::string modelpath, uint32_t device_id = 0);
    void detect(cv::Mat srcimg, cv::Mat& average_image);

private:
    void preprocess(const cv::Mat& src, std::vector<float>& data);

    uint32_t device_id_;
    AscendRuntime ascend_;
    AclModel model_;

    std::vector<float> input_image_;
    int inpWidth_;
    int inpHeight_;
    int num_outs_;
};
