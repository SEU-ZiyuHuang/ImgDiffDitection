# 异动检测项目：源码导航与 ONNX Runtime 改造说明

## 1. 本文档的范围

本文说明工作区中的**项目自有代码**、当前算法调用链、可执行边界和 ONNX Runtime 改造路线。以下内容不属于业务代码，本文不逐行解释：

- `deps/`、`deps-cv4.4/`：第三方 OpenCV 头文件、库和许可证；
- `build/`、`build-acl/`：历史 CMake 生成物、ELF 二进制和发布包；
- `.svn/`：版本控制元数据；
- 嵌套目录中的重复依赖和构建产物。

权威源码目录为 `imagecmp_v2/`。`imagecmp_v2/imagecmp_v2/` 在整理开始前是其业务源码的逐文件相同副本，应视为历史镜像，不应再同时维护两套代码。

## 2. 项目要解决的问题

系统以“实时巡检图”和“同一预置机位的正常样本图”为输入。上层业务系统先给出样本图中待检部件的 YOLO 框；算法库在实时图中定位该部件，再比较其结构，输出是否存在有效异动及异动区域。

它不是单图目标检测器，也不负责从全图自动识别部件类别。其最小输入为：

```text
实时图路径
样本图路径
样本图中的 YOLO 框：[center_x, center_y, width, height]
深度结构相似度阈值
```

主接口的输出为：

```text
返回码：1=发现有效异动框，0=未发现有效异动，负数=调用/定位/运行时错误
simres：LDC + ResNet18 的全局结构相似度（ONNX 后端每次完成有效定位后都会赋值）
Box：实时大图像素坐标中的异动框 [x, y, w, h]
liveRectYolo：定位后的实时部件 YOLO 框
可选：样本部件和实时部件 JPEG Base64
```

## 3. 当前实际算法链路

构建时可选择两个后端：`IMAGECMP_USE_ONNXRUNTIME=OFF` 为原始 Ascend `.om` 后端；
`=ON` 为新完成的 ONNX Runtime 后端。二者共用 `imagecmp.h` 的 C ABI，但历史 Ascend
分支仍保留了“高相似度早退”行为；本文以下“当前 ONNX 后端”描述的是已改造后的逻辑。

```text
实时大图 + 样本大图 + 样本 YOLO 框
  │
  ├─ 1. 从样本图裁出待检部件
  │
  ├─ 2. 在实时图定位部件
  │     ├─ OpenCV TM_CCOEFF_NORMED 模板匹配
  │     └─ 不可信时：SuperPoint + LightGlue 特征匹配 + 单应性投影
  │
  ├─ 3. LDC 提取两张部件图的轮廓图
  │
  ├─ 4. ResNet18 计算轮廓图的全局特征余弦相似度，写入 simres
  │     └─ 不以此分数直接结束判定
  │
  ├─ 5. ResNet18 稠密特征差异图
  │     └─ 平滑 → 阈值化 → 形态学处理 → 轮廓 → 合并/面积过滤
  │
  └─ 6. 合并/面积过滤后输出最大有效异动框、裁剪图和可选 Base64
```

### 关键语义：定位与异动必须分开

模板匹配得分只能表达“样本部件在实时图的哪个位置、定位是否可信”。当前代码中，模板分数大于等于 `0.85` 会直接返回“无异动”，这是历史快速路径，可能漏掉小面积的螺栓缺失、指针偏移、颜色变化、局部破损等异动。

ONNX 后端已经调整为：

```text
定位成功（模板匹配或特征匹配）
  → 始终进入独立的结构/颜色/局部差异判定
  → 再输出“无异动”或“异动框”
```

## 4. 源码文件说明

| 文件 | 编译状态 | 主要职责 |
|---|---|---|
| `imagecmp_v2/CMakeLists.txt` | 当前主构建 | `IMAGECMP_USE_ONNXRUNTIME=OFF/ON` 二选一构建 Ascend 或 ONNX Runtime 动态库；会构建命令行测试程序。 |
| `imagecmp_v2/Makefile` | 历史备用 | 旧 Make 构建方式，路径假设和 CMake 不一致；已将 C++ 标准注释并修正为 C++17。 |
| `imagecmp_v2/imagecmp.h` | 公共头文件 | `Box`、YOLO 转像素框、图像对比主接口等 C ABI 声明。 |
| `imagecmp_v2/yidong_main.cpp` | 当前主实现 | 模型池、接口实现、部件定位、LDC/ResNet 编排、结果输出。 |
| `imagecmp_v2/yidong_main_onnx.cpp` | ONNX 构建时启用 | ONNX 模型池、完整 C ABI、定位与异动解耦、裁剪图/Base64 输出。模型目录取自 `IMAGECMP_MODEL_DIR`。 |
| `imagecmp_v2/onnx_runtime/onnx_pipeline.h/.cpp` | ONNX 构建时启用 | ONNX Runtime Session 封装：SuperPoint+LightGlue 定位、LDC 边缘、ResNet18 全局/稠密差异；启动时校验节点数、类型和形状。 |
| `imagecmp_v2/ascend_infer.h/.cpp` | 当前主实现 | 华为 ACL 初始化、设备 context/stream、`.om` 模型加载、内存和同步推理。 |
| `imagecmp_v2/image_matcher.h/.cpp` | 当前主实现 | Ascend 版 SuperPoint + LightGlue：两图关键点匹配和样本框到实时框的投影。 |
| `imagecmp_v2/ldc.h/.cpp` | 当前主实现 | Ascend 版 LDC：多尺度边缘预测融合为轮廓图。 |
| `imagecmp_v2/deep_difference_detector.h/.cpp` | 当前主实现 | Ascend 版 ResNet18：全局特征相似度与局部稠密差异框检测。 |
| `imagecmp_v2/imagecmp.cpp` | 当前主构建 | 传统 OpenCV 直方图、ORB 对齐、像素差、差异框合并、YOLO 转换、裁剪图填充。 |
| `imagecmp_v2/Laplacian.cpp` | 仅 CMake 当前构建 | 3x3 网格拉普拉斯清晰度检查，供上层重拍策略调用。 |
| `imagecmp_v2/base64.h/.c` | 当前主构建 | JPEG 二进制与 Base64 的转换。 |
| `imagecmp_v2/test.cpp` | 当前主构建 | 一次性命令行冒烟测试；逐部件输出返回码、相似度、定位框、异动框、耗时，并默认在 `test/` 输出蓝色定位框和红色异动框的标注图。 |
| `imagecmp_v2/imagecmpfind.cpp` | 未被当前 CMake 构建 | 历史“图找图 + ORB + 像素差”方案，包含旧 `lxImageFind`。 |
| `imagecmp_v2/deepseek_cpp_20250625_641801.cpp` | 未构建 | 早期、较简单的模糊检测备份。 |
| `imagecmp_v2/onnx/deep_difference_detector_onnx.cpp` | `#if 0` 禁用 | 历史 ONNX Runtime 版 ResNet18 差异检测参考实现。 |
| `imagecmp_v2/onnx/image_matcher_onnx.cpp` | `#if 0` 禁用 | 历史 ONNX Runtime 版 SuperPoint + LightGlue 参考实现。 |
| `imagecmp_v2/onnx/ldc_onnx.cpp` | `#if 0` 禁用 | 历史 ONNX Runtime 版 LDC 参考实现。 |

## 5. 接口与构建现状

### 当前 Ascend 版本

`yidong_main.cpp` 硬编码加载以下模型：

```text
superpoint_lightglue_pipeline_512x512_linux_aarch64.om
LDC_640x360.om
resnet18.om
```

它要求 Linux/AArch64、Ascend NPU、CANN/ACL、与构建匹配的 OpenCV 动态库。工作区中的 `liblximagecmp.so` 也是 AArch64 ELF，不能在当前 Windows 主机直接运行。

### 当前 ONNX 资产

工作区根目录已具备：

```text
superpoint_lightglue_pipeline.onnx
LDC_640x360.onnx
resnet18.onnx
```

当以 `IMAGECMP_USE_ONNXRUNTIME=ON` 构建时，这三份文件由
`yidong_main_onnx.cpp` 加载，文件名固定为以上名称，所在目录通过环境变量
`IMAGECMP_MODEL_DIR` 指定。未设置时默认使用进程工作目录。原始 Ascend 构建仍只加载
`.om` 文件，二者不会混用。

### 已知接口问题

- `imagecmp.h` 声明了 `lxImageFind`，但当前 CMake 未编译其实现 `imagecmpfind.cpp`；不能视为稳定导出。
- 原始版本中 `lxImage_MatcherDet`、`lxImage_DeepDifferenceDetector`、`lxLaplacian` 等实现未完整声明在公共头文件；本次已补齐声明，但 `lxImageFind` 仍不是当前构建的稳定导出。
- 历史 Ascend 分支的 `lxImageCmpOnnx` 在模板高分的快速返回分支没有写入 `simres`，且会
  跳过局部差异检测；ONNX 分支已修正。
- ONNX 模型池支持 `lxUninit()` 后重新初始化，并在释放时等待在途调用归还模型实例。

## 6. ONNX Runtime 改造完成情况

目标是支持普通 x86 Windows/Linux 或 Linux AArch64 上的 ONNX Runtime 推理，不再强依赖 Ascend CANN。现在已实现为可选后端：

```text
IMAGECMP_USE_ONNXRUNTIME=ON    # 开发、x86 测试环境
IMAGECMP_USE_ONNXRUNTIME=OFF   # 现场 Ascend 环境（如仍保留）
```

### 已实现：ONNX 推理层

1. 新建 `onnx_runtime/onnx_pipeline.*`，使用独立的 `OnnxImageMatcher`、`OnnxLdc`、`OnnxDeepDifferenceDetector` 命名，避免与 Ascend 类冲突。
2. CMake 通过 `ONNXRUNTIME_ROOT` 查找 `onnxruntime_cxx_api.h` 和 `onnxruntime` 库；Windows 构建后会复制 `onnxruntime.dll` 到动态库输出目录。
3. Session 创建支持 Windows 的 UTF-8 模型路径转 UTF-16，因此模型目录可以包含中文。
4. 已对工作区三份 ONNX 文件做静态 Protobuf 元数据核验；启动时还会校验输入输出名称、数量、数据类型和运行时形状：
   - SuperPoint+LightGlue：`images` 为动态 `[batch_size,1,height,width]`，实际传入 `[2,1,512,512]`；输出为 `keypoints`、`matches`、`mscores`；
   - LDC：`input_image` 为 `[1,3,360,640]`，共有 5 个 `[1,1,360,640]` 边缘输出；
   - ResNet18：`input` 为 `[1,3,224,224]`，`output` 为 `[1,256,14,14]` 特征图。

### 已实现：主编排与判定语义

1. ONNX `ModelPipeline` 持有三个推理器并按整套借出/归还；`lxInitAIModel` 的设备 ID 参数在 CPU ONNX 后端中仅为 ABI 兼容，实例数仍决定并发上限。
2. 模型路径不再硬编码在源代码工作目录，改由 `IMAGECMP_MODEL_DIR` 配置。
3. `lxImageCmpOnnx` 中模板匹配只作为快速定位；模板定位成功和 SuperPoint+LightGlue 回退定位成功后，都会执行 LDC、全局相似度和稠密差异图。
4. 兼容保留参数 `threshold`，但它不再触发“全局相似度高即无异动”的早退。当前是否异动由有效局部差异框决定；`simres` 作为可观测指标供后续标定。
5. 公共头文件已经补充 `lxFreePtr`、最近错误文本和拆分调试接口声明。调用方必须用 `lxFreePtr` 释放 Base64/错误文本。

### 已在 Windows x64 实测：构建与发布

当前工作区已安装/放置以下 Windows x64 依赖：

| 依赖 | 版本/位置 | 用途 |
|---|---|---|
| CMake | 4.4.0（系统安装） | 生成与驱动构建。 |
| Visual Studio 2022 Build Tools | 17.14，MSVC x64 + Windows SDK（系统安装） | C/C++17 编译与链接。 |
| ONNX Runtime | 1.26.0，`third_party/onnxruntime-win-x64-1.26.0/` | ONNX CPU 推理、头文件、导入库与 DLL。 |
| OpenCV | 4.12.0，`third_party/opencv-4.12.0/opencv/build/` | 图像读写、模板匹配、几何处理和后处理。 |

CMake 会将 `onnxruntime.dll`、`opencv_world4120.dll` 复制到生成目录，测试程序无需依赖临时
`PATH` 配置。若保留 Ascend 现场后端，应把它的模板高分早退也改为与 ONNX 后端一致的语义，
否则两种部署的判定结果会不同。

## 7. 测试计划

### 冒烟测试

新增的 `test/` 目录包含 `live.jpg`、`temp.jpg` 和 `yolo.txt`。测试程序已改为命令行形式：

```text
imagecmp_test --live test/live.jpg --template test/temp.jpg --yolo test/yolo.txt --model-dir .
```

每个 YOLO 框会输出：类别、结构相似度、实时 YOLO 定位框、返回码、异动框和耗时。默认还会在
`test/annotated_component_<序号>.jpg` 输出可视化标注：蓝框为定位到的部件，红框为有效异动区域。
可增加 `--output-dir <已有目录>` 保存各部件的 640×640 裁剪图；使用
`--annotation-dir <已有目录>` 可以改变标注图输出目录。

### 必须覆盖的回归样本

| 类别 | 期望结果 |
|---|---|
| 正常同机位图 | 无异动、无异动框。 |
| 轻微光照变化 | 不应误报。 |
| 图像模糊 | 清晰度模块识别并提示重拍。 |
| 机位偏移/旋转 | 模板匹配失败时由特征匹配定位成功。 |
| 小面积真实异动 | 发现异动并输出接近真实位置的框。 |
| 大面积遮挡/错位 | 返回定位异常或低可信结果，不能伪造局部异动。 |
| 呼吸器颜色变化 | 如要纳入需求，需单独定义颜色判定与标注标准。 |

### 验收指标

在有人工标注的验证集上统计：定位成功率、异动召回率、精确率、误报率、漏报率、框 IoU、单部件耗时和并发吞吐。未完成这一步前，不能用“模型能运行”替代“算法可上线”。

## 8. 构建命令与下一步

在 Windows x64（示例路径需按本机安装位置替换）可以使用：

```powershell
cmake -S imagecmp_v2 -B build-onnx `
  -DIMAGECMP_USE_ONNXRUNTIME=ON `
  -DONNXRUNTIME_ROOT=C:\third_party\onnxruntime-win-x64 `
  -DOpenCV_DIR=C:\opencv\build
cmake --build build-onnx --config Release
build-onnx\bin\Release\imagecmp_test.exe --live test\live.jpg --template test\temp.jpg `
  --yolo test\yolo.txt --model-dir . --output-dir test
```

Linux x86_64/AArch64 使用同样的 CMake 选项，并将 `ONNXRUNTIME_ROOT`、`OpenCV_DIR` 换为
Linux 包所在目录；运行时保证 `libonnxruntime.so` 和 OpenCV `.so` 可被动态链接器找到。

本工作站已使用上述依赖成功配置并构建 Windows x64 Release 版本：

```text
build-onnx/bin/Release/lximagecmp.dll
build-onnx/bin/Release/imagecmp_test.exe
```

并已执行下列冒烟测试：

```text
imagecmp_test --live test/live.jpg --template test/temp.jpg --yolo test/yolo.txt --model-dir .
```

本次样例共有两个 YOLO 部件，均完成推理且进程退出码为 0：

| 部件类别 | 返回码 | simres | 异动框（实时大图像素） | 耗时 |
|---|---:|---:|---|---:|
| 176 | 1 | 0.6318 | `[798, 27, 116, 39]` | 3640 ms |
| 177 | 1 | 0.6213 | `[945, 390, 82, 84]` | 3490 ms |

这只证明“模型、Windows 依赖、C ABI 和完整流水线能够实际运行”；由于这两张样例图没有提供
人工异动真值，不能据此断言两个 `ret=1` 都是业务上的正确告警。后续验收应按下列顺序进行：

1. 先执行 CMake 配置；若模型节点名、形状或数据类型与预期不一致，构造 Session 会报出明确错误。
2. 用 `test/` 样例完成每个 YOLO 框的一次性冒烟测试。
3. 补充人工标注的正常、轻微光照、小面积异动、机位偏移与遮挡样本，统计召回/误报/框 IoU。
4. 依据验证集校准特征匹配证据、局部差异阈值和面积过滤，而不是只调全局 `simres`。
