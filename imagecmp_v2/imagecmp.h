// 对外 C ABI 定义。
//
// 上层服务需提供实时图、同机位样本图及样本图中的 YOLO 归一化部件框。
// 主算法会返回部件相似度和检测到的异动矩形。Box 坐标均为实时大图像素坐标。
// C 调用方若取得 Base64 返回指针，必须通过 lxFreePtr（实现在 yidong_main.cpp）释放。
#ifndef IMAGE_CMP_H
#define IMAGE_CMP_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>


#if defined(_WIN32) || defined(WIN32)

#define LXAPI __declspec(dllexport)
#else
#define LXAPI __attribute__((visibility("default")))
#endif

#define CONFIG_DEBUG_SHOW 0

#define CONFIG_DEBUG_OUT_IMG 0


double calculate_hist_similarity(cv::Mat& img1, cv::Mat& img2);

//对其图像
cv::Mat align_images(cv::Mat& base_gray, cv::Mat& target_gray,
	cv::Mat& target_img);

std::vector<cv::Rect> find_differences(cv::Mat& base_gray, cv::Mat& aligned_gray);

extern "C" {

	struct Box {
		// 异动框左上角及宽高；全 0 表示未输出有效异动区域。
		int x;
		int y;
		int w;
		int h;
	};

	LXAPI void yolo2rect(float yolo_x, float yolo_y, float yolo_w, float yolo_h,
		int imagew, int imageh, int& x, int& y, int& w, int& h);

	// 初始化AI模型，可以选择加载到哪些NPU设备，以及每个设备加载多少个实例
	// device_ids: NPU设备ID数组，如 {0, 1}
	// device_count: NPU设备数量
	// instances_per_device: 每个设备加载的实例数（并发数）
	// 返回 1 表示成功， 0 表示失败
	LXAPI int lxInitAIModel(const int* device_ids, int device_count, int instances_per_device);

	// 取得当前线程最近一次库内错误信息。返回的字符串由 lxFreePtr 释放。
	LXAPI int lxGetLastErrorMessage(char** msg);
	LXAPI void lxFreePtr(char* ptr);

	// 释放模型资源，通常在程序退出前主动调用
	// 返回 1 表示成功， 0 表示失败
	LXAPI int lxUninit();

	//liveImagePath: 实时图像地址
	//tempImagePath: 样本图像地址
	//yolo(x,y,w,h): yolo坐标点(对应样本图片)
	//float threshold,
	//outCropImagePath: 输出的切图地址(需要缩放成640)
	//return -1; notfind, -2 error; 0 没有找到; 1 找到
	// 历史“图找图”接口。当前 CMake 未编译 imagecmpfind.cpp，不能将此
	// 声明视为当前动态库的稳定导出；新接入方应使用 lxImageCmpOnnx。
	LXAPI int lxImageFind(
		const char* liveImagePath,
		const char* tempImagePath,
		float yolo_x, float yolo_y, float yolo_w, float yolo_h,
		float threshold,
		float* simres,
		Box* box,
		const char* outCropImagePath,
		int resize);

	//截图对其
	LXAPI int lxImageCmp3(const char* img_path1, 
		const char* img_path2, float yolo_x,
		float yolo_y, float yolo_w, float yolo_h,
		float threshold,
		float* simres, Box* box, const char* cropPath, int resize);

	//大图对其
	//img1 样本图像
	//img2 实时图像
	//yolo: yolo标签
	//threshold: 对比阈值
	//simres: 切图的直方对比阈值
	//box: 大图中的坐标
	//cropPath 实时图像切图路径 长度 >0 时 保存: /tmp/aaaa.jpg
	//resize: 缩放:640,0不缩放
	LXAPI int lxImageCmp2(const char* img_path1, const char* img_path2, float yolo_x,
		float yolo_y, float yolo_w, float yolo_h, float threshold,
		float* simres, Box* box, const char* cropPath, int resize);

	//@simres: 0~1
	//return 0:success !0: error
	LXAPI int lxImageCmp(const char* path1, const char* path2, float threshold,
		float* simres, Box* box);

	// 当前主入口。名称沿用 ONNX 历史版本；实际后端由 CMake 的
	// IMAGECMP_USE_ONNXRUNTIME 选项选择 Ascend .om 或 ONNX Runtime。
	// liveRectYolo 输出定位到的实时部件框（YOLO 归一化格式）；
	// generate_base64 非零时，仅在发现异动后返回样本/实时部件 JPEG Base64。
	LXAPI int lxImageCmpOnnx(
		const char* liveImagePath,
		const char* tempImagePath,
		float yolo_x, float yolo_y, float yolo_w, float yolo_h,
		float threshold,
		float* simres,
		struct Box* box,
		const char* outCropImagePath,
		int resize,
		float liveRectYolo[4],
		char** oImgLiveB64,
		char** oImgTmpB64,
		int generate_base64);

	LXAPI int lxImageCrop(const char* srcpath, int x, int y, int w, int h, const char* outpath);

	// 便于上层联调的辅助接口。
	LXAPI int lxImageYolo2Rect(const char* srcpath, float yolo_x, float yolo_y,
		float yolo_w, float yolo_h, int* x, int* y, int* w, int* h);
	LXAPI int lxImageDraw(const char* srcpath, int x, int y, int w, int h, const char* outpath);
	LXAPI int lxImageOnnxLDC(const char* srcpath, const char* outpath);
	LXAPI int lxImage_MatcherDet(const char* liveImagePath, const char* tempImagePath,
		float yolo_x, float yolo_y, float yolo_w, float yolo_h, struct Box* box);
	LXAPI int lxImage_DeepDifferenceDetector(const char* liveImagePath, const char* tempImagePath,
		float yolo_x, float yolo_y, float yolo_w, float yolo_h,
		int livebox_x, int livebox_y, int livebox_w, int livebox_h,
		float* out_score, struct Box* out_box);
	LXAPI int lxLaplacian(const char* inpath, double threshold, int* is_blurred, double* score);
	LXAPI int lxLaplacian2(const char* inpath, float yolo_x, float yolo_y,
		float yolo_w, float yolo_h, double threshold, int* is_blurred, double* score);


}

#endif
