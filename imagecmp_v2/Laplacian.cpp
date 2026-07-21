// 基于拉普拉斯方差的图像清晰度检查。
// 图片被划分为 3x3 网格；若至少四个相邻网格的方差低于阈值，视为大面积模糊。
// 该模块应在上层的“是否重拍”流程中先行调用，主异动入口不会自动调用它。
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include "imagecmp.h"

struct BlurResult {
    double min_variance;
    bool has_large_blur_area;
    int max_connected_size;
};

// DFS函数用于计算连通区域大小
// 深度优先搜索二值模糊网格，统计一个四邻接模糊连通区域的规模。
void dfs(const std::vector<std::vector<int>>& matrix,
    std::vector<std::vector<bool>>& visited,
    int i, int j, int& count) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    if (i < 0 || i >= rows || j < 0 || j >= cols || visited[i][j] || matrix[i][j] == 0)
        return;

    visited[i][j] = true;
    count++;

    // 4个方向：上、下、左、右
    dfs(matrix, visited, i - 1, j, count);
    dfs(matrix, visited, i + 1, j, count);
    dfs(matrix, visited, i, j - 1, count);
    dfs(matrix, visited, i, j + 1, count);
}

// 查找最大连通区域
int find_max_connected_region(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return 0;
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    int max_region = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] == 1 && !visited[i][j]) {
                int count = 0;
                dfs(matrix, visited, i, j, count);
                if (count > max_region) {
                    max_region = count;
                }
            }
        }
    }
    return max_region;
}

// 多区域模糊检测函数
// 计算每个网格的拉普拉斯方差，并判断是否出现足够大的连续模糊区域。
BlurResult multi_region_blur_check(cv::Mat &image,
    int grid_rows = 3, int grid_cols = 3,
    double threshold = 100.0) {
    //cv::Mat image = cv::imread(image_path);
    //if (image.empty()) {
        //std::cerr << "Error: Could not read image " << image_path << std::endl;
        //return { 0, false, 0 };
    //    throw std::invalid_argument("图像不存在");
    //}

    int h = image.rows;
    int w = image.cols;
    std::vector<double> results;
    std::vector<std::vector<double>> score_matrix(grid_rows, std::vector<double>(grid_cols, 0.0));
    BlurResult blur_result;

    // 计算每个区域的拉普拉斯方差
    for (int i = 0; i < grid_rows; i++) {
        for (int j = 0; j < grid_cols; j++) {
            int y1 = i * h / grid_rows;
            int y2 = (i + 1) * h / grid_rows;
            int x1 = j * w / grid_cols;
            int x2 = (j + 1) * w / grid_cols;

            cv::Mat roi = image(cv::Range(y1, y2), cv::Range(x1, x2));

            cv::Mat laplacian;
            cv::Laplacian(roi, laplacian, CV_64F);

            cv::Scalar mean, stddev;
            cv::meanStdDev(laplacian, mean, stddev);
            double variance = stddev.val[0] * stddev.val[0];

            //std::cout << "region(" << i << "," << j << "), score=" << variance << std::endl;
            results.push_back(variance);
            score_matrix[i][j] = variance;
        }
    }

    // 创建二值模糊矩阵
    std::vector<std::vector<int>> blur_matrix(grid_rows, std::vector<int>(grid_cols, 0));
    for (int i = 0; i < grid_rows; i++) {
        for (int j = 0; j < grid_cols; j++) {
            blur_matrix[i][j] = (score_matrix[i][j] < threshold) ? 1 : 0;
        }
    }

    // 打印模糊矩阵
    //std::cout << "模糊区域矩阵:" << std::endl;
    for (const auto& row : blur_matrix) {
        for (int val : row) {
            //std::cout << val << " ";
        }
        //std::cout << std::endl;
    }

    // 查找最大连通区域
    int max_connected_size = find_max_connected_region(blur_matrix);
    //std::cout << "最大连通模糊区域大小: " << max_connected_size << std::endl;

    // 计算最小方差
    double min_variance = *std::min_element(results.begin(), results.end());
    bool has_large_blur_area = (max_connected_size >= 4);

    return { min_variance, has_large_blur_area, max_connected_size };
}

#if 0
int main(int argc, char** argv) {
    std::string input_path = "input.jpg";
    if (argc > 1) {
        input_path = argv[1];
    }

    double threshold = 100.0;
    BlurResult result = multi_region_blur_check(input_path, 3, 3, threshold);

    if (result.has_large_blur_area) {
        std::cout << "图像 " << input_path << " 检测到大面积模糊区域(连续"
            << result.max_connected_size << "块),最低score="
            << result.min_variance << std::endl;
    }
    else {
        std::cout << "图像 " << input_path << " 清晰,最低score="
            << result.min_variance << std::endl;
    }

    return 0;
}

#endif

std::pair<bool, double> is_blur(const std::string& image_path, double threshold = 100.0) {
    // 读取图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("图像未找到: " + image_path);
    }
    
    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // 使用拉普拉斯算子计算梯度
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    // 计算梯度的方差
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double variance = stddev.val[0] * stddev.val[0];
    
    // 判断是否模糊
    bool is_blurred = variance > threshold;
    return std::make_pair(is_blurred, variance);
}


extern "C" {

// 仅检查 YOLO 部件框区域的清晰度，避免背景清晰但目标部件失焦造成漏检。
LXAPI int lxLaplacian2(const char* inpath, 
	float yolo_x, float yolo_y, float yolo_w, float yolo_h,
	double threshold, 
	int *is_blurred,
	double *score)
{
	try {
		cv::Mat imgTemp = cv::imread(inpath); //BGR 模板图片
		if (imgTemp.empty()) {
		    printf("error: not open image:%s\n", inpath);
		    return -2;
		}

		int img_width = imgTemp.cols;
		int img_height = imgTemp.rows;

		int x1, y1, box_width, box_height;
		yolo2rect(yolo_x, yolo_y, yolo_w, yolo_h, img_width, img_height, x1, y1, box_width, box_height);
	   	//printf("yolo转真实坐标: %d,%d,%d,%d\n", x1, y1, box_width, box_height);
		if (x1 < 0 || y1 < 0 || box_width <= 3 || box_height <= 3) {
			std::cerr << "坐标或尺寸无效！" << std::endl;
			return -4;
		}
		if (x1 + box_width > img_width || y1 + box_height > img_height) {
			std::cerr << "范围超过图像大小！" << std::endl;
			return -5;
		}

		cv::Mat imgCrop = imgTemp(cv::Rect(x1, y1, box_width, box_height)).clone();
	 	if (imgCrop.empty())
		{
			//std::cerr << "截图为空" << std::endl;
			return -6;
		}
	
		//检测模糊
		BlurResult result = multi_region_blur_check(imgCrop, 3, 3, threshold);
		*score = result.min_variance;
		if (result.has_large_blur_area)
			*is_blurred = 0;
		else
			*is_blurred = 1;
		return 1;
	}
	catch (const std::exception& e) {
	    std::cerr << "错误: " << e.what() << std::endl;
	    return -1;
	}
return -2;
}

    //输入参数 本地图片路径，阈值
    //    检测得分是否小于阈值
    //    输出结果 是否小于阈值，得分
    //返回: 1 调用成功; 小于0失败
    // 检查整张图。当前约定：is_blurred=0 表示大面积模糊，1 表示清晰；
    // 命名和取值方向不直观，后续对外 API 应改为更明确的 is_clear 或枚举状态。
    LXAPI int lxLaplacian(const char* inpath, double threshold, int *is_blurred, double *score)
    {
        try {
            //std::string input_path = "input.jpg";           
            //double threshold = 100.0;
	    cv::Mat image = cv::imread(inpath);
	    if(image.empty()){
		return -1;
	    }
            BlurResult result = multi_region_blur_check(image, 3, 3, threshold);

            *score = result.min_variance;

            if (result.has_large_blur_area) {
                //std::cout << "图像 " << inpath << " 检测到大面积模糊区域(连续"
                //    << result.max_connected_size << "块),最低score="
                //    << result.min_variance << std::endl;
                *is_blurred = 0;
            }
            else {
                //std::cout << "图像 " << inpath << " 清晰,最低score="
                //    << result.min_variance << std::endl;
                *is_blurred = 1;
            }
            return 1;
        }
        catch (const std::exception& e) {
            //std::cerr << "错误: " << e.what() << std::endl;
            return -1;
        }
        return -2;
    }
}
