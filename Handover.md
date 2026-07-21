# 异动检测项目交接

更新日期：2026-07-21。本文写给没有任何上下文的新对话。

## 1. 当前任务

项目用于供电设备的“标准图 vs 实时图”比较。

业务目标：

1. 先确认实时图是否是预期设备/部件；不匹配则上报。
2. 若是同一部件，判断是否与标准状态存在异动。
3. 可接受误报，但不能把“不确定、未匹配或有异动”静默判成无异动。
4. 检出异动时，输出全部差异区域的框（最好同时输出 mask/证据图）。

约束：供电局数据涉密；当前没有可用于训练/验收的异动标注集。现有 SuperPoint、LightGlue、LDC、ResNet18 ONNX 都是开源预训练模型，并非项目训练所得。

## 2. 已完成

- 已完整审阅当前工程，并写过源码导航与 ONNX 改造说明：`项目源码导航与ONNX改造说明.md`。
- 已将主项目 `imagecmp_v2/` 改造成可在 Windows 上使用 ONNX Runtime 的版本；保留旧 Ascend ACL 路线。
- 已安装/接入 CMake、MSVC、OpenCV 4.12、ONNX Runtime 1.26，并完成 Release 编译。
- 已完成一次 ONNX 端到端冒烟运行；可执行文件：`build-onnx/bin/Release/imagecmp_test.exe`。
- 测试程序已能在 `test/` 输出标注图片：`annotated_component_*.jpg`。
- 已为主要业务代码补充中文注释。
- 已确认：本地没有训练脚本、训练数据、微调逻辑或异动真值标签。
- 已确认 `all_test/` 有 385 个案例目录、918 张图片、385 对标准/对比图、773 个 YOLO 框、72 个类别；它没有异动标签/异动框，不能当训练集或召回率验收集，但可在内网用于正常波动、匹配和阈值回归。

## 3. 当前代码真实流程

```text
标准图 YOLO 框
  -> OpenCV matchTemplate 快速定位
  -> 不可靠时：SuperPoint + LightGlue 找匹配点
  -> OpenCV findHomography 根据匹配点计算 H，并投影部件框
  -> 直接裁标准图/实时图的两个矩形
  -> LDC 边缘图计算 similarity（当前不参与最终框）
  -> ResNet18 原图特征差异生成单个候选框
```

术语：SuperPoint 找特征点；LightGlue 找可靠点对；`findHomography` 根据点对计算 H（两图如何移动/旋转/缩放的几何关系）。当前 H 只用于找实时图中的部件框，**没有用于把两张图真正对齐后再比较**。

关键文件：

- `imagecmp_v2/yidong_main_onnx.cpp`：当前 ONNX 主流程/C API。
- `imagecmp_v2/onnx_runtime/onnx_pipeline.cpp`：ONNX 模型封装、SuperPoint+LightGlue、LDC、ResNet18。
- `imagecmp_v2/test.cpp`：命令行测试及标注图输出。
- `imagecmp_v2/yidong_main.cpp`：旧 Ascend ACL 主流程，仍有历史风险逻辑。

## 4. 当前阶段

ONNX 推理原型已经可运行；现在处于“按新业务要求重构算法语义与对齐链路”的设计完成、尚未实施阶段。

尚未开始 P0/P1 重构。不要把当前结果宣称为满足“不能漏检”的生产方案。

## 5. 下一步计划（主路线，不以训练新模型为主）

### P0：先消除静默漏检

- 将结果从混杂的 `1 / 0 / 负数` 改为业务状态：`IMAGE_MISMATCH`、`MATCH_UNCERTAIN`、`ALIGNMENT_UNCERTAIN`、`CHANGE_SUSPECTED`、`NO_CHANGE_HIGH_CONFIDENCE`。
- 保留旧 C API 作为兼容包装；新 API 应将“函数是否报错”和“业务判断”分离。
- 支持多个差异框和 mask，不再只返回一个 `Box`。
- 将阈值移入配置并记录版本；输出每个候选框的触发原因。
- 补齐模型来源、SHA-256、导出脚本、输入预处理、输出形状、许可证台账。

### P1：真正做“先对齐，再检测”

- SuperPoint+LightGlue 后，保留并返回 H、内点数/率、重投影误差、点的空间覆盖度。
- 用 H 将实时图 warp 到标准图坐标；再用 OpenCV ECC 做小范围精调。
- 生成 valid-mask，排除旋转后黑边、无效重叠区和低质量区域。
- 匹配/配准证据不足时上报复核，绝不自动判正常。

### P2：多证据差异与高召回输出

- 在已对齐图上并行比较：Lab 颜色差（油污/变色）、局部结构/纹理差、边缘距离差（含 LDC）、ResNet 感知特征差。
- 使用“保守 OR 融合”：任一通道显著异常即保留候选；多通道重叠时提高置信度。
- 小区域标为低置信复核，不直接删除；大面积差异应报告“整体异动”或“对齐不可靠”，不得返回正常。

### P3：内网标定与验证

- 使用确认正常的参考图建立每设备/部件/机位的基线（中位数 + MAD），这属于现场标定，不是训练。
- 用 `all_test/` 做正常场景回归；增加内网合成的平移、旋转、透视、曝光、阴影、油迹/变色/缺件压力测试。
- 后续必须采集受控真实异动样本，才能验收“漏检率”；没有阳性真值，不能承诺零漏检。

## 6. 绝对不要再踩的坑

1. **模板匹配高分不等于无异动。** 它只能说明“目标大概在图中”。旧 Ascend 主流程的 `score >= 0.85` 直接返回正常是错误逻辑；ONNX 路线已移除该短路。
2. **不要把 H 和模板匹配混为一谈。** `matchTemplate` 是快速找位置；H 是 SuperPoint+LightGlue 找到点对后，由 OpenCV `findHomography` 算出的对齐关系。
3. **不要只投影 ROI 框后直接比较 crop。** 必须先把实时图对齐到标准图，才能可靠做像素/结构差异。
4. **不要把 `-6` 当成“没有异动”。** 它目前只是“没有成功定位”，可能是拍错、遮挡、视角问题或严重变化。
5. **不要继续保留“差异小于 100 像素或超过 ROI 80% 就返回 0”的逻辑。** 它会漏掉小裂纹/早期漏油，也会把整体变色、换件、严重遮挡静默判正常。位置：`imagecmp_v2/yidong_main_onnx.cpp` 的 `copy_difference_result()`。
6. **不要依赖当前固定 `0.70 / 0.70 / 500` 阈值。** 差异图按每一对图 min-max 归一化，固定分数没有跨场景业务含义。
7. **不要把当前模型说成“类别识别/通用缺陷模型”。** SuperPoint+LightGlue 是几何匹配；LDC 是边缘检测；ResNet18 当前只是通用特征差异基线。
8. **不要把 LDC 当作已参与最终异动框。** 当前它只产生 similarity；最终局部框来自原图的 ResNet 特征差异。
9. **不要假设 ONNX 预处理和权重来源正确。** LDC 的当前 RGB/0-255 输入、ResNet18 的来源/截断层均需对原始导出验证。
10. **不要把 `all_test/` 上传、外发或用于在线服务。** 它是涉密项目数据；只允许在授权内网做本地标定和测试。
11. **不要把当前 `all_test/` 当作带异动标签的数据集。** 它缺少异动类别、框/mask、预期结论。
12. **不要忽略许可证。** 尤其需核对当前 SuperPoint ONNX 的实际权重来源及商用许可；LDC/ResNet 来源也应书面确认。

## 7. 已验证的 ONNX 运行方式

在项目根目录执行：

```powershell
.\build-onnx\bin\Release\imagecmp_test.exe `
  --live test\live.jpg `
  --template test\temp.jpg `
  --yolo test\yolo.txt `
  --model-dir .
```

这只证明当前链路能运行，不证明检测结论业务正确。
