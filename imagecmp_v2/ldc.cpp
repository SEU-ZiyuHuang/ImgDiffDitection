// LDC 轮廓模型实现：多输出尺度经过 sigmoid、归一化、反色和融合，
// 形成与原图同尺寸的单通道轮廓图。
#include "ldc.h"

#include <algorithm>
#include <cstring>

LDC::LDC(std::string modelpath, uint32_t device_id)
    : device_id_(device_id), ascend_(device_id_), model_(ascend_, modelpath), inpWidth_(0), inpHeight_(0), num_outs_(0) {
    const aclmdlIODims& in_dims = model_.input_dims(0);
    if (in_dims.dimCount >= 4) {
        inpHeight_ = static_cast<int>(in_dims.dims[2]);
        inpWidth_ = static_cast<int>(in_dims.dims[3]);
    }
    num_outs_ = static_cast<int>(model_.num_outputs());
}

// BGR 转 RGB 并按 NCHW 顺序展开。该模型使用原始 0~255 float 像素值。
void LDC::preprocess(const cv::Mat& src, std::vector<float>& data) {
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(inpWidth_, inpHeight_));

    data.resize(3 * inpHeight_ * inpWidth_);

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < inpHeight_; i++) {
            for (int j = 0; j < inpWidth_; j++) {
                uchar pix = dst.ptr<uchar>(i)[j * 3 + (2 - c)];
                data[c * inpHeight_ * inpWidth_ + i * inpWidth_ + j] = static_cast<float>(pix);
            }
        }
    }
}

// 对每个模型输出尺度生成边缘概率图，再平均融合为最终轮廓图。
void LDC::detect(cv::Mat srcimg, cv::Mat& average_image) {
    average_image = cv::Mat::zeros(srcimg.rows, srcimg.cols, CV_32FC1);

    preprocess(srcimg, input_image_);

    const size_t expected = model_.input_size(0) / sizeof(float);
    if (expected != input_image_.size()) {
        std::cerr << "LDC input size mismatch: expected floats=" << expected << " got=" << input_image_.size() << std::endl;
        std::exit(-1);
    }

    std::memcpy(model_.input_host(0), input_image_.data(), model_.input_size(0));
    model_.Execute();

    for (int n = 0; n < num_outs_; n++) {
        const aclmdlIODims& od = model_.output_dims(static_cast<size_t>(n));
        int outHeight = (od.dimCount >= 4) ? static_cast<int>(od.dims[2]) : 0;
        int outWidth = (od.dimCount >= 4) ? static_cast<int>(od.dims[3]) : 0;
        if (outHeight <= 0 || outWidth <= 0) {
            std::cerr << "LDC invalid output dims at index " << n << std::endl;
            std::exit(-1);
        }

        float* pred = reinterpret_cast<float*>(model_.output_host(static_cast<size_t>(n)));
        cv::Mat result(outHeight, outWidth, CV_32FC1, pred);

        cv::Mat TmpExp;
        cv::exp(-result, TmpExp);
        cv::Mat mask = 1.0 / (1.0 + TmpExp);

        double min_val = 0;
        double max_val = 0;
        cv::minMaxLoc(mask, &min_val, &max_val);
        mask = (mask - min_val) * 255.0 / (max_val - min_val + 1e-12);
        mask.convertTo(mask, CV_8UC1);

        cv::bitwise_not(mask, mask);
        cv::resize(mask, mask, srcimg.size());

        cv::accumulate(mask, average_image);
    }

    average_image = average_image / static_cast<float>(std::max(1, num_outs_));
    average_image.convertTo(average_image, CV_8UC1);
}
