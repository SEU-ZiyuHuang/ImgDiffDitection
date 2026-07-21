# 异动检测动态库的统一 CMake 构建入口。
#
# 支持两种互斥后端：
#   IMAGECMP_USE_ONNXRUNTIME=OFF：原有 Linux/AArch64 + Ascend CANN/.om 后端；
#   IMAGECMP_USE_ONNXRUNTIME=ON ：新 ONNX Runtime 后端，可用于普通 CPU 开发环境。
#
# 两种后端保持同一套 C ABI（imagecmp.h），因此上层服务无需因推理后端切换而改调用方式。
cmake_minimum_required(VERSION 3.16)
project(lximagecmp LANGUAGES C CXX)

# ONNX Runtime 1.26 的 C++ 头文件使用 std::variant、结构化绑定等 C++17 特性。
# Ascend 旧代码同样可在 C++17 下编译，因此统一采用 C++17。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(IMAGECMP_USE_ONNXRUNTIME "Build the portable ONNX Runtime backend" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime package root containing include/ and lib/")
set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend CANN toolkit root")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # 项目源文件和中文注释均保存为 UTF-8；MSVC 默认按本地代码页读取会产生警告，
    # 在少数头文件场景还可能造成预处理连锁错误，因此显式指定 UTF-8。
    add_compile_options(/utf-8)
    # Visual Studio 是多配置生成器；显式指定目录让 DLL 和测试程序位于构建树内，
    # 而不是 CMake 的默认 Program Files 安装前缀。
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d features2d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# 两种后端都会使用的传统 OpenCV 工具和图像结果编码模块。
set(IMAGECMP_COMMON_SOURCES
    imagecmp.cpp
    base64.c
    Laplacian.cpp
)

if(IMAGECMP_USE_ONNXRUNTIME)
    # ONNX Runtime 没有统一的 CMake 包结构，优先使用 ONNXRUNTIME_ROOT；
    # 也允许开发者通过 CMAKE_PREFIX_PATH/find_path 自行提供头文件和库。
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}"
        PATH_SUFFIXES include include/onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}"
        PATH_SUFFIXES lib lib64)

    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime was not found. Configure -DONNXRUNTIME_ROOT=<package-root>, "
            "where <package-root>/include/onnxruntime_cxx_api.h and <package-root>/lib exist.")
    endif()

    add_library(${PROJECT_NAME} SHARED
        yidong_main_onnx.cpp
        onnx_runtime/onnx_pipeline.cpp
        ${IMAGECMP_COMMON_SOURCES}
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE IMAGECMP_BACKEND_ONNX=1)
    target_include_directories(${PROJECT_NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/onnx_runtime"
            "${ONNXRUNTIME_INCLUDE_DIR}"
            ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")

    # Windows 发布包需同时带上 onnxruntime.dll；Unix 运行时则由 rpath/LD_LIBRARY_PATH 管理。
    if(WIN32)
        find_file(ONNXRUNTIME_DLL NAMES onnxruntime.dll
            HINTS "${ONNXRUNTIME_ROOT}" PATH_SUFFIXES lib bin)
        if(ONNXRUNTIME_DLL)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
        endif()
        # OpenCV 官方 Windows 包的导入库位于 lib/，运行 DLL 位于相邻的 bin/。
        # 将实际链接到的 world DLL 复制到输出目录，使 imagecmp_test 不依赖临时 PATH。
        if(DEFINED OpenCV_LIB_PATH)
            get_filename_component(OPENCV_BIN_DIR "${OpenCV_LIB_PATH}/../bin" ABSOLUTE)
            file(GLOB OPENCV_RUNTIME_DLLS "${OPENCV_BIN_DIR}/opencv_world*.dll")
            foreach(OPENCV_RUNTIME_DLL IN LISTS OPENCV_RUNTIME_DLLS)
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${OPENCV_RUNTIME_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
            endforeach()
        endif()
    else()
        set_target_properties(${PROJECT_NAME} PROPERTIES
            BUILD_RPATH "$ORIGIN:$ORIGIN/libs"
            INSTALL_RPATH "$ORIGIN:$ORIGIN/libs")
    endif()

    # 独立于旧 C ABI 的模型治理检查。它在任何检测链路使用模型前验证
    # 已登记的哈希、张量契约及确定性参考推理，便于 CI 或发布前人工运行。
    add_executable(onnx_model_contract_check onnx_model_contract_check.cpp)
    target_include_directories(onnx_model_contract_check PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(onnx_model_contract_check PRIVATE "${ONNXRUNTIME_LIBRARY}")
    if(WIN32 AND ONNXRUNTIME_DLL)
        add_custom_command(TARGET onnx_model_contract_check POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:onnx_model_contract_check>")
    endif()
    set(IMAGECMP_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
        "Directory containing the governed ONNX model files")
    add_custom_target(verify_onnx_model_contracts
        COMMAND $<TARGET_FILE:onnx_model_contract_check> --model-dir "${IMAGECMP_MODEL_DIR}"
        DEPENDS onnx_model_contract_check
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Verifying governed ONNX model files and reference inference")
else()
    # 原始 Ascend 版：三个模型由 yidong_main.cpp 以 .om 文件形式加载。
    add_library(${PROJECT_NAME} SHARED
        yidong_main.cpp
        ascend_infer.cpp
        deep_difference_detector.cpp
        ldc.cpp
        image_matcher.cpp
        ${IMAGECMP_COMMON_SOURCES}
    )
    target_include_directories(${PROJECT_NAME}
        PRIVATE
            "${ASCEND_HOME}/acllib/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            ${OpenCV_INCLUDE_DIRS})
    target_link_directories(${PROJECT_NAME} PRIVATE "${ASCEND_HOME}/acllib/lib64")
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCV_LIBS}
        ascendcl
        acl_dvpp
        dl
        rt
        pthread)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "$ORIGIN:$ORIGIN/libs"
        INSTALL_RPATH "$ORIGIN:$ORIGIN/libs")
endif()

# 一次性命令行验证程序：支持 --live、--template、--yolo、--model-dir、--threshold。
add_executable(imagecmp_test test.cpp)
# imagecmp.h 公开了 cv::Mat 等 OpenCV 类型，测试程序即使只调用 C ABI 也必须能找到 OpenCV 头文件。
target_include_directories(imagecmp_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${OpenCV_INCLUDE_DIRS})
# 测试程序除调用动态库外，还直接使用 OpenCV 绘制并写入可视化标注图。
target_link_libraries(imagecmp_test PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS})

if(UNIX)
    target_link_libraries(imagecmp_test PRIVATE pthread dl rt)
    set_target_properties(imagecmp_test PROPERTIES BUILD_RPATH "$ORIGIN:$ORIGIN/libs")
endif()
