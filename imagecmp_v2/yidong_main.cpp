// 异动检测库的主编排与 C ABI 实现。
//
// 典型调用链：样本 YOLO 框 -> 模板匹配/特征匹配定位 -> LDC 轮廓 -> ResNet18 相似度
// -> 稠密差异图 -> 异动框、裁剪图和可选 Base64 结果。
// 本文件不负责 SFTP、YOLO 推理、部件类别判断或异动结果上传；这些均由上层业务服务完成。
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include "base64.h"
#include "imagecmp.h"


#define CONFIG_HAVE_DEEP_DIFF 1
#define CONFIG_HAVE_LDC 1
#define CONFIG_HAVE_ImageMatcher 1

#include "deep_difference_detector.h"
#include "ldc.h"
#include "image_matcher.h"

#include <queue>
#include <condition_variable>

// 定义一个模型流水线结构体，将三个 AI 模型打包在一起。
// 每个流水线实例绑定在一个特定的 NPU 设备 (device_id) 上，
// 用于在多线程并发时，各个线程能独占一套模型，实现真正的并行推理。
// 一套流水线中的三个模型共用同一个 NPU 设备。ModelPool 以“整套借出、整套归还”
// 的方式避免同一模型实例被多个请求同时写入输入/输出缓冲区。
struct ModelPipeline {
	std::unique_ptr<ImageMatcher> matcher;
	std::unique_ptr<LDC> ldc;
	std::unique_ptr<DeepDifferenceDetector> detector;
	int device_id;
};

// 模型对象池管理器。
// 负责在程序启动时预先创建多套 AI 模型，多线程调用时从池中取用，用完归还。
class ModelPool {
public:
	// 初始化模型池
	// device_ids: 允许使用的 NPU 设备列表 (例如 [0, 1])
	// device_count: 设备总数
	// instances_per_device: 每个设备上创建几套模型 (决定了单卡的并发度)
	void init(const int* device_ids, int device_count, int instances_per_device) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (initialized_) return;
		
		// 模型文件按相对路径加载，部署时必须放在进程工作目录或由运行环境保证可见。
		for (int i = 0; i < device_count; i++) {
			int dev_id = device_ids[i];
			for (int j = 0; j < instances_per_device; j++) {
				auto pipeline = std::make_unique<ModelPipeline>();
				pipeline->device_id = dev_id;
				pipeline->matcher = std::make_unique<ImageMatcher>("superpoint_lightglue_pipeline_512x512_linux_aarch64.om", dev_id);
				pipeline->ldc = std::make_unique<LDC>("LDC_640x360.om", dev_id);
				pipeline->detector = std::make_unique<DeepDifferenceDetector>("resnet18.om", dev_id);
				pool_.push(std::move(pipeline));
			}
		}
		initialized_ = true;
	}

	// 阻塞获取一套空闲的模型流水线。如果池子为空，线程会在这里等待直到有其他线程归还模型。
	std::unique_ptr<ModelPipeline> acquire() {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [this]() { return !pool_.empty(); });
		auto pipeline = std::move(pool_.front());
		pool_.pop();
		return pipeline;
	}

	// 归还模型流水线到池中，并唤醒一个可能在等待的线程。
	void release(std::unique_ptr<ModelPipeline> pipeline) {
		std::lock_guard<std::mutex> lock(mutex_);
		pool_.push(std::move(pipeline));
		cv_.notify_one();
	}
	
	// 销毁模型池，释放所有资源
	void destroy() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (!initialized_) return;
		
		// 清空队列，这会自动触发每个 ModelPipeline 及其内部 unique_ptr 的析构
		while (!pool_.empty()) {
			pool_.pop();
		}
		initialized_ = false;
	}
	
	bool is_initialized() const { return initialized_; }

private:
	std::queue<std::unique_ptr<ModelPipeline>> pool_;
	std::mutex mutex_;
	std::condition_variable cv_; // 条件变量，用于队列为空时的阻塞和唤醒
	bool initialized_ = false;
};

static ModelPool g_model_pool;
static std::once_flag g_models_once;

// 默认初始化逻辑：当上层(如 C#)没有显式调用 lxInitAIModel 初始化时，
// 默认在 NPU 0 卡上初始化 1 套模型以防报错。
static void InitModelsOnce() {
	int default_device = 0;
	g_model_pool.init(&default_device, 1, 1);
}

// 确保模型已经被加载的守护函数。
static void EnsureModelsLoaded() {
	if (!g_model_pool.is_initialized()) {
		std::call_once(g_models_once, InitModelsOnce);
	}
}

// RAII 风格的模型生命周期守护类（类似于 std::lock_guard）。
// 构造时自动向模型池 acquire 申请一套模型，
// 析构（离开作用域）时自动 release 归还模型。
// 推理过程中持有一套模型；无论正常返回还是抛出异常，析构时都会归还到模型池。
class PipelineGuard {
public:
	PipelineGuard() {
		pipeline_ = g_model_pool.acquire();
	}
	~PipelineGuard() {
		if (pipeline_) {
			g_model_pool.release(std::move(pipeline_));
		}
	}
	ModelPipeline* get() { return pipeline_.get(); }
private:
	std::unique_ptr<ModelPipeline> pipeline_;
};

uint64_t os_time_ms()
{
	// 使用 C++11 的 std::chrono 来获取当前时间戳
	auto now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	return milliseconds.count(); // 返回毫秒数
}


float AreaIOU(int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh) {
	int ax2 = ax + aw;
	int ay2 = ay + ah;
	int bx2 = bx + bw;
	int by2 = by + bh;

	// 计算交集区域
	int interX1 = std::max(ax, bx);
	int interY1 = std::max(ay, by);
	int interX2 = std::min(ax2, bx2);
	int interY2 = std::min(ay2, by2);

	int interW = interX2 - interX1;
	int interH = interY2 - interY1;

	if (interW <= 0 || interH <= 0)
		return 0.0f;  // 没有交集

	int interArea = interW * interH;

	int areaA = aw * ah;
	int areaB = bw * bh;
	int unionArea = areaA + areaB - interArea;

	return static_cast<float>(interArea) / unionArea;
}

float AreaIOU(cv::Rect rect1, cv::Rect rect2)
{
	return AreaIOU(rect1.x, rect1.y, rect1.width, rect1.height, rect2.x, rect2.y, rect2.width, rect2.height);
}

// 定义一个结构来存储每个步骤的信息
struct StepInfo
{
	std::string name;
	uint64_t start_time;
	uint64_t end_time;
};

// 定义一个类来管理步骤的耗时
class StepTimer
{
public:
	// 添加步骤的开始时间
	void push_start(const std::string& step_name, uint64_t start_time)
	{
		step_info_.push_back({ step_name, start_time, 0 });
	}

	// 添加步骤的结束时间
	void push_end(const std::string& step_name, uint64_t end_time)
	{
		for (auto& info : step_info_)
		{
			if (info.name == step_name && info.end_time == 0)
			{
				info.end_time = end_time;
				break;
			}
		}
	}

	// 打印所有步骤的耗时
	void print_step_times() const
	{
		for (const auto& info : step_info_)
		{
			if (info.end_time != 0)
			{
				uint64_t duration = info.end_time - info.start_time;
				std::cout << "Step: " << info.name << " took " << duration << " ms" << std::endl;
			}
		}
	}

private:
	std::vector<StepInfo> step_info_; // 存储每个步骤的信息
};


std::vector<uchar> cvMat2jpgString(cv::Mat& img) {
	// 创建一个字节流向量
	std::vector<uchar> buf;

	// 编码参数（JPEG 格式）
	std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 90 };

	// 将图像编码为 JPEG 格式
	bool success = cv::imencode(".jpg", img, buf, params);
	if (!success) {
		throw std::runtime_error("Failed to encode image as JPEG");
	}
	return buf;
}

// 将实时图中的像素框转换回归一化 YOLO 格式，便于上层服务与检测标注体系对接。
void cvRect2YoloPoint(const cv::Rect& rect, int w, int h, float v[4]) {
	if (w <= 0 || h <= 0) {
		throw std::invalid_argument("Width and height must be positive");
	}

	// 计算中心点坐标
	float x_center = (rect.x + rect.width / 2.0f) / w;
	float y_center = (rect.y + rect.height / 2.0f) / h;

	// 计算宽度和高度
	float rect_width = rect.width / static_cast<float>(w);
	float rect_height = rect.height / static_cast<float>(h);

	// 确保值在 [0, 1] 范围内
	x_center = std::max(0.0f, std::min(1.0f, x_center));
	y_center = std::max(0.0f, std::min(1.0f, y_center));
	rect_width = std::max(0.0f, std::min(1.0f, rect_width));
	rect_height = std::max(0.0f, std::min(1.0f, rect_height));

	// 将结果存储到数组中
	v[0] = x_center;
	v[1] = y_center;
	v[2] = rect_width;
	v[3] = rect_height;
}

//一个图形中查找另一个图片
extern std::pair<cv::Rect, double> find_best_match(cv::Mat& large_img, cv::Mat& template_img, int& findType);

//合并坐标
extern cv::Rect merge_rectangles(const std::vector<cv::Rect>& rectangles, int threshold = 50);

//缩放
extern cv::Mat resizeAndPad(const cv::Mat& input, int target_size);

//计算两张图形的相似度阈值
extern double calculate_hist_similarity(cv::Mat& img1, cv::Mat& img2);

/////////////////////////////////////////////////////////////////////////////////////
#if 0
int main(int argc, char* argv[]) {
    try {
        DeepDifferenceDetector detector("resnet18.onnx");

        std::string img1_path = argv[1];
        std::string img2_path = argv[2];
        std::string yolo_txt = argv[3];

        cv::Mat simg1 = cv::imread(img1_path); //实时图片
        cv::Mat simg2 = cv::imread(img2_path); //模板
        int lw = simg1.cols, lh = simg1.rows;

        // 从yolo文件获取坐标
        std::vector<cv::Rect> reccs = detector.readYoloFormatToRects(yolo_txt, lw, lh);

        for (int i = 0; i < reccs.size(); ++i)
        {
            // 实时图片切图
            cv::Mat timg = simg1(reccs[i]);
            // cv::imwrite("tarimgs/" + std::to_string(i) + "_out.jpg", timg);
            // 匹配图像
            cv::Rect matchrect = detector.templateMatching(simg2, timg);

            cv::Mat tempCropMat = simg2(matchrect);

            float sim_value = detector.computer_sim(timg, tempCropMat);

            if (sim_value > 0.9)
            {
                continue;
            }
            
            if (sim_value >= 0.5)
            {
                std::vector<cv::Rect> tarv = detector.detect_differences(
                    timg,
                    tempCropMat,
                    0.7f,        // difference_threshold (对应Python版本)
                    0.7f,        // roi_diff_threshold (对应Python版本)
                    500,         // min_contour_area (对应Python版本)                    
                    cv::Size(224, 224)  // target_img_size (对应Python版本)
                );
                //陈坚处理

            }
            else
            {
                float new_sim_value = detector.computer_sim(timg, simg2(reccs[i]));
                //陈坚处理

            }


        }

        // cv::imwrite("tarimgs/match_out.jpg", simg2);

        /*
            std::vector<cv::Rect> tarv = detector.detect_differences(
                img1_path,
                img2_path,
                0.7f,        // difference_threshold (对应Python版本)
                0.7f,        // roi_diff_threshold (对应Python版本)
                500,         // min_contour_area (对应Python版本)
                "results",   // output_dir (对应Python版本)
                cv::Size(224, 224)  // target_img_size (对应Python版本)
            );
        */

        // cv::Mat srcimg = cv::imread(img1_path);
        // for(cv::Rect rec : tarv) cv::rectangle(srcimg, rec, cv::Scalar(0,255,0), 2);

        // cv::imwrite("results/out.jpg", srcimg);

    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

#endif

//////////////////////////////////////////////////////
// 输出定位到的实时部件图。resize>0 时以目标尺寸居中填白；较大图片会等比例缩小。
void saveimage(const char *outCropImagePath, cv::Mat &img, cv::Rect rect, int resize) {
	if (outCropImagePath && strlen(outCropImagePath) > 0) {
		bool wret = false;
		if (resize > 0) {

			if (rect.width > resize || rect.height > resize) {
				cv::Mat cropImg = img(rect).clone();
				cv::Mat out = resizeAndPad(cropImg, resize);
				wret = cv::imwrite(outCropImagePath, out);
			} else {
				 // 计算中心
				cv::Point center = (rect.tl() + rect.br()) / 2;

				// 计算新左上角坐标
				int x0 = center.x - resize / 2;
				int y0 = center.y - resize / 2;

				// 与图像边界做裁剪
				x0 = std::max(0, std::min(x0, img.cols - resize));
				y0 = std::max(0, std::min(y0, img.rows - resize));

				// 生成新的 640×640 矩形
				rect = cv::Rect(x0, y0, resize, resize);
				cv::Mat cropImg = img(rect).clone();
				wret = cv::imwrite(outCropImagePath, cropImg);
			}
		}
		if (!wret) {
			std::cout << "file write error　" << outCropImagePath << std::endl;
		}
	} else {
		std::cout << "not set out file path" << std::endl;
	}
}

//图片中查找图片 模板匹配 特征匹配中
// 在实时大图中定位样本图中的目标部件。
// findtype=0 表示 OpenCV 模板匹配；findtype=1 表示 SuperPoint+LightGlue 回退定位。
// value 在模板路径中为模板相关系数；特征路径中仅用于兼容旧接口，不能作为异动分数。
int ImgFindImg(cv::Mat& imgLive, cv::Mat& imgTemp, cv::Mat& tmpCrop, cv::Rect rectTmpBox, cv::Rect& rectOut, double& value, int& findtype, StepTimer& stepTimer, std::unique_ptr<PipelineGuard>& guard) {

	//模板匹配
	findtype = 0;
	stepTimer.push_start("cv::matchTemplate", os_time_ms());
	cv::Rect rect = DeepDifferenceDetector::templateMatching(imgLive,
		tmpCrop,
		value,
		cv::TM_CCOEFF_NORMED);
	stepTimer.push_end("cv::matchTemplate", os_time_ms());
	
	float iou = AreaIOU(rect, rectTmpBox);

	printf("模板匹配结果 %.4f 坐标:%d,%d,%d,%d  iou=%.2f\n", value, rect.x, rect.y, rect.width, rect.height, iou);
	// 此阈值只说明“定位可信”。不能单独证明部件没有局部异动，后续业务应将
	// 定位成功和无异动判定解耦。
	if (value > 0.85 && iou > 0.1) {
		rectOut = rect;
		return 0;
	}

	findtype = 1;

	printf("特征匹配中...\n");
	stepTimer.push_start("ImageMatcher_Infer", os_time_ms());
	
	// 在需要推理时，才从池中获取模型流水线实例（避免占用NPU等待CPU匹配）
	if (!guard) {
		guard = std::make_unique<PipelineGuard>();
	}
	ImageMatcher& im = *(guard->get()->matcher);

	cv::Mat left_img = imgTemp;// imgLive;	
	cv::Mat right_img = imgLive; // imgTemp;

	// 获取全局匹配结果
	MatchResult mres = im.get_matches(left_img, right_img);
	//MatchResult mres = im.get_matches(right_img, left_img);

	// 匹配小图结果保存
	cv::Rect tarrect;
	bool matchres = im.feature_match(rectTmpBox, imgTemp.cols, imgTemp.rows, mres.keypoints_left, mres.matched_left, mres.matched_right, tarrect);
	stepTimer.push_end("ImageMatcher_Infer", os_time_ms());

	// 匹配成功
	if (matchres)
	{
		value = 0;

		printf("特征匹配成功\n");
		rectOut = im.outRect(tarrect, imgTemp.cols, imgTemp.rows);
		//rectOut需要与yolo坐标框 进行互补?是否需要?
		return 0;
	} else {
		printf("ImageMatcher模型特征匹配失败\n");
	}
	return 0;
}

extern "C" {

	thread_local std::string g_last_error_msg;

	static inline void set_last_error(const std::string& s) {
		g_last_error_msg = s;
	}

	LXAPI int lxGetLastErrorMessage(char** msg) {
		if (!msg) return -1;
		*msg = 0;
		if (g_last_error_msg.empty()) return 0;
		char* p = (char*)malloc(g_last_error_msg.size() + 1);
		if (!p) return -2;
		memcpy(p, g_last_error_msg.c_str(), g_last_error_msg.size() + 1);
		*msg = p;
		return 1;
	}

	LXAPI void lxFreePtr(char* ptr) {
		if (ptr)
			free(ptr);
	}
	
	LXAPI int lxInitAIModel(const int* device_ids, int device_count, int instances_per_device) {
		set_last_error("");
		try {
			g_model_pool.init(device_ids, device_count, instances_per_device);
			return 1;
		} catch (const std::exception& e) {
			set_last_error(e.what());
			return 0;
		}
	}

	LXAPI int lxUninit() {
		set_last_error("");
		std::cout<<"释放模型资源"<<std::endl;

		try {
			g_model_pool.destroy();
			return 1;
		} catch (const std::exception& e) {
			set_last_error(e.what());
			return 0;
		}
	}

	//liveImagePath: 实时图像地址
	//tempImagePath: 样本图像地址
	//yolo(x,y,w,h): yolo坐标点(对应样本图片)
	//float threshold,
	//outCropImagePath: 输出的切图地址(需要缩放成640)
	//return -1; notfind, -2 error; 0 没有找到; 1 找到
	// 主检测接口。
	// 返回：1=发现有效异动框，0=未发现有效异动，负数=输入/定位/运行时错误。
	// 注意：名称中的 Onnx 是历史遗留；当前实现经 Ascend ACL 执行 .om 模型。
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
		int generate_base64) {
		set_last_error("");
		try {
			EnsureModelsLoaded();
		} catch (const std::exception& e) {
			set_last_error(e.what());
			return -1;
		}

		// 第 1 步：读入同尺寸的实时图和样本图。尺寸不一致时无法直接使用同一 YOLO 框。
		StepTimer stepTimer;
		stepTimer.push_start("cv::imread", os_time_ms());
		cv::Mat imgLive = cv::imread(liveImagePath); //BGR
		cv::Mat imgTemp = cv::imread(tempImagePath); //BGR 模板图片
		stepTimer.push_end("cv::imread", os_time_ms());

		*oImgLiveB64 = 0;
		*oImgTmpB64 = 0;

		memset(liveRectYolo, 0, sizeof(liveRectYolo));
		memset(box, 0, sizeof(Box));
		if (imgLive.empty() || imgTemp.empty()) {
			printf("error: not open image\n");
			set_last_error("error: not open image");
			return -2;
		}

		if (imgLive.rows != imgTemp.rows || imgLive.cols != imgTemp.cols) {
			printf("image size not equ\n");
			set_last_error("image size not equal");
			return -3;
		}

		try {
			int img_width = imgTemp.cols;
			int img_height = imgTemp.rows;
			cv::Rect rectBox;
			{
				int x1, y1, box_width, box_height;
				yolo2rect(yolo_x, yolo_y, yolo_w, yolo_h, img_width, img_height, x1, y1, box_width,
					box_height);

				printf("yolo转真实坐标:(%.5f,%.5f,%.5f,%.5f)(%d,%d)-> %d,%d,%d,%d\n",
					yolo_x, yolo_y, yolo_w, yolo_h,
					img_width, img_height,
					x1, y1, box_width, box_height);

				if (x1 < 0 || y1 < 0 || box_width <= 3 || box_height <= 3) {
					std::cerr << "坐标或尺寸无效！" << std::endl;
					set_last_error("invalid yolo rect");
					return -4;
				}
				if (x1 + box_width > img_width || y1 + box_height > img_height) {
					std::cerr << "范围超过图像大小！" << std::endl;
					set_last_error("yolo rect out of range");
					return -4;
				}

				rectBox.x = x1;
				rectBox.y = y1;
				rectBox.width = box_width;
				rectBox.height = box_height;
			}

			//样本截图
			cv::Mat imgTempCrop = imgTemp(rectBox).clone();

			if (imgTempCrop.empty()) {
				std::cerr << "截图为空" << std::endl;
				set_last_error("template crop empty");
				return -1;
			}

			//需要寻找
			//查找最佳匹配物体
			//int findType = -1;
			//std::pair<cv::Rect, double> result = find_best_match(imgLive, imgTempCrop, findType);
			//cv::Rect best_match = result.first;
			//double match_score = result.second;
			//模板匹配
			int imgFindType;
			cv::Rect imgFindRect;
			double templateMaxValue = 0.0;

			//模板匹配 & 特征匹配
			stepTimer.push_start("匹配", os_time_ms());
			std::unique_ptr<PipelineGuard> guard;
			int ret = ImgFindImg(imgLive, imgTemp, imgTempCrop, rectBox, imgFindRect, templateMaxValue, imgFindType, stepTimer, guard);
			stepTimer.push_end("匹配", os_time_ms());

			if (imgFindRect.empty()) {
				std::cout << "图片匹配未找到目标,type=" << imgFindType << ",templateMaxValue=" << templateMaxValue << std::endl;
				stepTimer.print_step_times();
				set_last_error("image match not found");
				return -6;
			}

			//实时图片样本匹配截图
			printf("图片匹配结果(方式%d) %.4f 坐标:%d,%d,%d,%d\n", imgFindType, templateMaxValue, imgFindRect.x, imgFindRect.y, imgFindRect.width, imgFindRect.height);

			if (imgFindRect.width > rectBox.width + 10 || imgFindRect.height > rectBox.height + 10) {
				printf("匹配图像大小超过yolo模板大小: %dx%d > %dx%d\n", imgFindRect.width, imgFindRect.height, rectBox.width, rectBox.height);
				set_last_error("matched rect larger than template rect");
				return -5;
			}

			//左边转换
			cvRect2YoloPoint(imgFindRect, imgLive.cols, imgLive.rows, liveRectYolo);

			// 历史快速返回：模板高度相似即认为无异动。该逻辑会跳过局部差异检测，
			// 因此存在漏检小范围变化的风险；后续 ONNX 改造时应改为仅表示定位成功。
			if (imgFindType == 0 && templateMaxValue >= 0.85) {
				//没有异动?
				saveimage(outCropImagePath, imgLive, imgFindRect, resize);
				stepTimer.print_step_times();
				return 0;
			}

			//模板匹配 图片
			cv::Mat imgLiveFindCrop = imgLive(imgFindRect).clone();
			if (imgLiveFindCrop.empty()) {
				std::cerr << "best_match截图为空" << std::endl;
				set_last_error("live crop empty");
				return -1;
			}

			stepTimer.push_start("LDC", os_time_ms());
			if (!guard) {
				guard = std::make_unique<PipelineGuard>();
			}
			LDC& ldcnet = *(guard->get()->ldc);

			//样本截图轮廓图
			cv::Mat imgTempCropLDC;
			ldcnet.detect(imgTempCrop, imgTempCropLDC);
			
			//实时样本图片匹配截图轮廓图
			cv::Mat imgLiveCropLDC;
			ldcnet.detect(imgLiveFindCrop, imgLiveCropLDC);

			stepTimer.push_end("LDC", os_time_ms());

			stepTimer.push_start("异动", os_time_ms());
			//还需要做异动对比
			DeepDifferenceDetector& detector = *(guard->get()->detector);
			{
				std::vector<cv::Rect> cmpRects;
				//轮廓图片对比
				float score = 0.0f;			
				score = detector.computer_sim(imgLiveCropLDC, imgTempCropLDC);
				
				*simres = score;
				if (score > threshold) {
					std::cout << "保存: 实时图片样本匹配截图(超过阈值)" << std::endl;
					saveimage(outCropImagePath, imgLive, imgFindRect, resize);
					memset(box, 0, sizeof(Box));
					stepTimer.print_step_times();
					return 0;
				}

				cmpRects = detector.detect_differences(imgLiveFindCrop,
					imgTempCrop,
					0.7f,		 // difference_threshold (对应Python版本)
					0.7f,        // roi_diff_threshold (对应Python版本)
					500,         // min_contour_area (对应Python版本)
					cv::Size(224, 224)  // target_img_size (对应Python版本)
				);
				

				std::cout << "保存: 实时图片样本匹配截图" << std::endl;
				saveimage(outCropImagePath, imgLive, imgFindRect, resize);

				//将相邻边距离小于50像素的进行合并，合并后过滤掉面积小于10x10，或者大于部件80% 的区域。
				//取剩余差异区域中面积最大的一个返回
				//
				// 合并矩形
				cv::Rect merged_rect = merge_rectangles(cmpRects, 50);
				if (merged_rect.x == 0 && merged_rect.y == 0 && merged_rect.width == 0
					&& merged_rect.height == 0) {
					stepTimer.print_step_times();
					return 0;
				}

				//过滤掉面积小于10x10，或者大于部件80%的区域
				float rect_area = merged_rect.width * merged_rect.height;
				float box_area = imgLiveFindCrop.cols * imgLiveFindCrop.rows;
				if (rect_area / box_area > 0.8 || rect_area < 10 * 10) {
					box->x = 0;
					box->y = 0;
					box->w = 0;
					box->h = 0;

					std::cout << "面积小于0.8 < 10*10" << std::endl;
					stepTimer.print_step_times();
					return 0;
				}
				printf("合并坐标:%d,%d,%d,%d\n", merged_rect.x, merged_rect.y, merged_rect.width, merged_rect.height);
				//位置需要匹配到大图中,故而增加截图区域的xy
				box->x = merged_rect.x + imgFindRect.x;
				box->y = merged_rect.y + imgFindRect.y;
				box->w = merged_rect.width;
				box->h = merged_rect.height;

				stepTimer.push_end("异动", os_time_ms());

				stepTimer.push_start("base64_encode", os_time_ms());
				//转base64
				if (generate_base64) {
					//两个小图片
					//模板图片转字符串
					std::vector<uchar> jpgTmp = cvMat2jpgString(imgTempCrop);

					//匹配图片转小图
					std::vector<uchar> jpglive = cvMat2jpgString(imgLiveFindCrop);

					std::string jpgB64Tmp = "data:image/jpeg;base64," + Base64Encode(jpgTmp.data(), jpgTmp.size());
					std::string jpgB64live = "data:image/jpeg;base64," + Base64Encode(jpglive.data(), jpglive.size());

					// 分配堆内存并复制字符串（C# 负责释放）
					char* str1 = (char*)malloc(jpgB64Tmp.size() + 1);
					if (str1) {
						memset(str1, 0, jpgB64Tmp.size() + 1);
						memcpy(str1, jpgB64Tmp.c_str(), jpgB64Tmp.size() + 1);
					};

					char* str2 = (char*)malloc(jpgB64live.size() + 1);
					if (str2) {
						memset(str2, 0, jpgB64live.size() + 1);
						memcpy(str2, jpgB64live.c_str(), jpgB64live.size() + 1);
					};

					*oImgTmpB64 = str1;
					*oImgLiveB64 = str2;
				}
				stepTimer.push_end("base64_encode", os_time_ms());

				//需要判断box是否在范围内
				// 检查左上角
				if (box->x < rectBox.x || box->y < rectBox.y) {
					//printf("异动坐标超过yolo坐标 左上角:%d,%d != yolo:%d,%d\n", box->x, box->y, rectBox.x, rectBox.y);
					//return -6;
				}

				// 检查右下角
				if (box->x + box->w > rectBox.x + rectBox.width || box->y + box->h > rectBox.y + rectBox.height) {
					//printf("异动坐标超过yolo坐标 右下角 %d,%d != yolo:%d,%d", box->x + box->w, box->y + box->h, rectBox.x + rectBox.width, rectBox.y + rectBox.height);
					//return -6;
				}
			}
			stepTimer.print_step_times();
		}
		catch (const std::exception& e) {
			std::cerr << "发生异常: Exception in lxImageFind: " << e.what() << std::endl;
			set_last_error(e.what());
			return -1; // 返回错误代码
		}

		return 1;
	}


LXAPI int lxImageYolo2Rect(const char *srcpath, float yolo_x, float yolo_y, float yolo_w,
		float yolo_h, int *x, int *y, int *w, int *h) {
	cv::Mat src1 = cv::imread(srcpath);
	int x1, y1, box_width, box_height;
	yolo2rect(yolo_x, yolo_y, yolo_w, yolo_h, src1.cols, src1.rows, x1, y1, box_width, box_height);
	*x = x1;
	*y = y1;
	*w = box_width;
	*h = box_height;
	return 0;
}

LXAPI int lxImageDraw(const char *srcpath, int x, int y, int w, int h, const char *outpath) {
	try {
		cv::Mat img = cv::imread(srcpath);
		cv::Rect rect(x, y, w, h);
		cv::rectangle(img, rect, cv::Scalar(0, 255, 255), 2);
		cv::imwrite(outpath, img);
		return 1;
	} catch (const std::exception &e) {
		std::cerr << "错误: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

LXAPI int lxImageCrop(const char *srcpath, int x, int y, int w, int h, const char *outpath) {
	try {
		cv::Mat img1 = cv::imread(srcpath);
		cv::imwrite(outpath, img1(cv::Rect(x, y, w, h)).clone());
		return 1;
	} catch (const std::exception &e) {
		std::cerr << "错误: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

LXAPI int lxImageOnnxLDC(const char *srcpath, const char *outpath) {
	LDC ldcnet("LDC_640x360.om");
	cv::Mat timg = cv::imread(srcpath);

	cv::Mat timg_average_image;
	ldcnet.detect(timg, timg_average_image);
	cv::imwrite(outpath, timg_average_image);
	return 0;
}

//特征匹配 
//@liveImagePath 实时图片路径
//@tempImagePath 模板图片路径
//yolo: yolo坐标点(模板图片中的左边框)
//box: 匹配到的坐标(实际图片中的坐标)
//return: 1 成功; 0 没有找到, <0 错误
//  -2 图片读取失败
//	-3 模板图片与实时图片大小不一样
//  -4 坐标或尺寸无效
//	-5 匹配到的box与实际大小不一致
// 仅执行 SuperPoint+LightGlue 定位的调试/拆分接口；供上层将定位和差异检测分步调用。
LXAPI int lxImage_MatcherDet(const char* liveImagePath,
		const char* tempImagePath,
		float yolo_x, float yolo_y, float yolo_w, float yolo_h,
		struct Box* box
	)
{
	StepTimer stepTimer;
	cv::Mat imgLive = cv::imread(liveImagePath); //BGR
	cv::Mat imgTemp = cv::imread(tempImagePath); //BGR 模板图片

	if (imgLive.empty() || imgTemp.empty()) {
		printf("error: not open image\n");
		set_last_error("error: not open image");
		return -2;
	}

	if (imgLive.rows != imgTemp.rows || imgLive.cols != imgTemp.cols) {
		printf("image size not equ\n");
		set_last_error("image size not equal");
		return -3;
	}

	cv::Rect rectBox;
	{
		int img_width = imgTemp.cols;
		int img_height = imgTemp.rows;
		int x1, y1, box_width, box_height;
		yolo2rect(yolo_x, yolo_y, yolo_w, yolo_h, img_width, img_height, x1, y1, box_width, box_height);

		printf("yolo转真实坐标:(%.5f,%.5f,%.5f,%.5f)(%d,%d)-> %d,%d,%d,%d\n",
			yolo_x, yolo_y, yolo_w, yolo_h,
			img_width, img_height,
			x1, y1, box_width, box_height);

		if (x1 < 0 || y1 < 0 || box_width <= 3 || box_height <= 3) {
			std::cerr << "坐标或尺寸无效！" << std::endl;
			set_last_error("invalid yolo rect");
			return -4;
		}

		if (x1 + box_width > img_width || y1 + box_height > img_height) {
			std::cerr << "范围超过图像大小！" << std::endl;
			set_last_error("yolo rect out of range");
			return -4;
		}

		rectBox.x = x1;
		rectBox.y = y1;
		rectBox.width = box_width;
		rectBox.height = box_height;		
	}

	//样本截图
	cv::Mat imgTempCrop = imgTemp(rectBox).clone();

	if (imgTempCrop.empty()) {
		std::cerr << "截图为空" << std::endl;
		set_last_error("template crop empty");
		return -1;
	}

	std::unique_ptr<PipelineGuard> guard;

	stepTimer.push_start("ImageMatcher_Infer", os_time_ms());
	// 在需要推理时，才从池中获取模型流水线实例（避免占用NPU等待CPU匹配）
	if (!guard) {
		guard = std::make_unique<PipelineGuard>();
	}
	ImageMatcher& im = *(guard->get()->matcher);

	cv::Mat left_img = imgTemp;// imgLive;	
	cv::Mat right_img = imgLive; // imgTemp;

	// 获取全局匹配结果
	MatchResult mres = im.get_matches(left_img, right_img);
	//MatchResult mres = im.get_matches(right_img, left_img);

	// 匹配小图结果保存
	cv::Rect tarrect;
	bool matchres = im.feature_match(rectBox, imgTemp.cols, imgTemp.rows, mres.keypoints_left, mres.matched_left, mres.matched_right, tarrect);
	stepTimer.push_end("ImageMatcher_Infer", os_time_ms());

	// 匹配成功
	if (matchres)
	{
		printf("ImageMatcher模型特征匹配成功\n");
		cv::Rect rectOut = im.outRect(tarrect, imgTemp.cols, imgTemp.rows);

		box->x = rectOut.x;
		box->y = rectOut.y;
		box->w = rectOut.width;
		box->h = rectOut.height;

		//需要判断box大小与yolo的大小是否一致
		if (rectOut.width > rectBox.width + 10 || rectOut.height > rectBox.height + 10) {
			printf("匹配图像大小超过yolo模板大小: %dx%d > %dx%d\n", rectOut.width, rectOut.height, rectBox.width, rectBox.height);
			set_last_error("matched rect larger than template rect");
			return -5;
		}

		return 1;
	} else {
		printf("ImageMatcher模型特征匹配失败\n");
	}
	return 0;
}


//轮廓检测
//@liveImagePath 实时图片路径
//@tempImagePath 模板图片路径
//yolo: yolo坐标点(模板图片中的左边框)
//live: 实时图片中的实际坐标(特征匹配返回的box坐标)
//out_score: 轮廓阈值
//out_box: 轮廓检测中的变化的区域坐标, 需要加上 live 坐标
// 仅执行 LDC + ResNet18 差异检测的调试/拆分接口。
// livebox_* 应来自前一步定位，坐标相对于实时大图。
LXAPI int lxImage_DeepDifferenceDetector(const char* liveImagePath,
		const char* tempImagePath,
		float yolo_x, float yolo_y, float yolo_w, float yolo_h,
		int livebox_x, int livebox_y, int livebox_w, int livebox_h,
		float *out_score,
		struct Box* out_box
	)
{
	StepTimer stepTimer;

	cv::Mat imgLive = cv::imread(liveImagePath); //BGR
	cv::Mat imgTemp = cv::imread(tempImagePath); //BGR 模板图片

	if (imgLive.empty() || imgTemp.empty()) {
		printf("error: not open image\n");
		set_last_error("error: not open image");
		return -2;
	}

	if (imgLive.rows != imgTemp.rows || imgLive.cols != imgTemp.cols) {
		printf("image size not equ\n");
		set_last_error("image size not equal");
		return -3;
	}

	//yolo坐标转实际坐标
	cv::Rect rectBox;
	{
		int img_width = imgTemp.cols;
		int img_height = imgTemp.rows;
		int x1, y1, box_width, box_height;
		yolo2rect(yolo_x, yolo_y, yolo_w, yolo_h, img_width, img_height, x1, y1, box_width, box_height);

		printf("yolo转真实坐标:(%.5f,%.5f,%.5f,%.5f)(%d,%d)-> %d,%d,%d,%d\n",
			yolo_x, yolo_y, yolo_w, yolo_h,
			img_width, img_height,
			x1, y1, box_width, box_height);

		if (x1 < 0 || y1 < 0 || box_width <= 3 || box_height <= 3) {
			std::cerr << "坐标或尺寸无效！" << std::endl;
			set_last_error("invalid yolo rect");
			return -4;
		}

		if (x1 + box_width > img_width || y1 + box_height > img_height) {
			std::cerr << "范围超过图像大小！" << std::endl;
			set_last_error("yolo rect out of range");
			return -4;
		}

		rectBox.x = x1;
		rectBox.y = y1;
		rectBox.width = box_width;
		rectBox.height = box_height;		
	}

	//样本截图
	cv::Mat imgTempCrop = imgTemp(rectBox).clone();
	if (imgTempCrop.empty()) {
		std::cerr << "截图为空" << std::endl;
		set_last_error("template crop empty");
		return -1;
	}

	cv::Rect imgFindRect;
	imgFindRect.x = livebox_x;
	imgFindRect.y = livebox_y;
	imgFindRect.width = livebox_w;
	imgFindRect.height = livebox_h;

	//模板匹配 图片
	cv::Mat imgLiveFindCrop = imgLive(imgFindRect).clone();
	if (imgLiveFindCrop.empty()) {
		std::cerr << "best_match截图为空" << std::endl;
		set_last_error("live crop empty");
		return -1;
	}

	std::unique_ptr<PipelineGuard> guard;

	stepTimer.push_start("LDC", os_time_ms());
	if (!guard) {
		guard = std::make_unique<PipelineGuard>();
	}
	LDC& ldcnet = *(guard->get()->ldc);

	//样本截图轮廓图
	cv::Mat imgTempCropLDC;
	ldcnet.detect(imgTempCrop, imgTempCropLDC);
	
	//实时样本图片匹配截图轮廓图
	cv::Mat imgLiveCropLDC;
	ldcnet.detect(imgLiveFindCrop, imgLiveCropLDC);

	stepTimer.push_end("LDC", os_time_ms());

	stepTimer.push_start("异动", os_time_ms());
	//还需要做异动对比
	DeepDifferenceDetector& detector = *(guard->get()->detector);
	
	std::vector<cv::Rect> cmpRects;
	//轮廓图片对比
	float score = 0.0f;			
	score = detector.computer_sim(imgLiveCropLDC, imgTempCropLDC);
	
	cmpRects = detector.detect_differences(imgLiveFindCrop,
					imgTempCrop,
					0.7f,		 // difference_threshold (对应Python版本)
					0.7f,        // roi_diff_threshold (对应Python版本)
					500,         // min_contour_area (对应Python版本)
					cv::Size(224, 224)  // target_img_size (对应Python版本)
				);

	*out_score = score;

	// 合并矩形
	cv::Rect merged_rect = merge_rectangles(cmpRects, 50);

	out_box->x = merged_rect.x;
	out_box->y = merged_rect.y;
	out_box->w = merged_rect.width;
	out_box->h = merged_rect.height;

	return 1;
}

}
