# ONNX 模型台账与本地契约验证

本台账覆盖仓库根目录内的全部受版本控制 ONNX 权重。它记录当前原型的实际运行契约，不能把这些通用预训练模型描述为设备身份识别或通用缺陷分类模型。

## 完整性基线

| 模型 | 根目录路径 | SHA-256 | 当前治理状态 |
| --- | --- | --- | --- |
| SuperPoint + LightGlue | `superpoint_lightglue_pipeline.onnx` | `228994cea8c010146fa2aef933baa3ffaa4bcdc522bc8aa560087fcff8134526` | 已验证发布资产；仍须按依赖链完成许可审批 |
| LDC | `LDC_640x360.onnx` | `1895fa66262c9caac1dfe0e4ff7b180f99ea2c1b5993906b6443bede4da4ac62` | **阻断：原始导出物和权重来源缺失** |
| ResNet18 特征提取器 | `resnet18.onnx` | `c812837bb5132a5757b42c63d07041e6f84a563f18b5d780d4ae8e5dfed37c2b` | **阻断：原始导出物、权重来源和许可证缺失** |

本地 Git 历史只能追溯到 `a0e4325` 对三份二进制文件的首次项目导入；没有包含上游下载地址、导出命令、checkpoint 哈希或许可证文件。因此 LDC 与 ResNet18 不得被新的 `imagecmp_next` 模块或生产发布依赖，直到补齐下方“解除阻断”材料。保留 `imagecmp_v2` 仅作为既有原型和诊断参考，不改变该限制。

## 来源与许可证记录

| 模型 | 可验证来源 | 许可证记录 | 结论 |
| --- | --- | --- | --- |
| SuperPoint + LightGlue | 上游 [LightGlue-ONNX v2.0 发布资产](https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v2.0) 中同名资产的大小为 `51,182,095` 字节；本地重新下载并计算的 SHA-256 与本台账完全相同。上游提供同名端到端导出路径。 | 上游仓库声明 [Apache-2.0](https://github.com/fabio-sim/LightGlue-ONNX/blob/main/LICENSE)。这只记录发布仓库的声明；其 SuperPoint 权重和任何下游再分发仍需纳入项目依赖许可证审查。 | 来源资产已验证；许可证审批待项目合规负责人确认。 |
| LDC | 图结构、节点名和输出形式与 [xavysp/LDC](https://github.com/xavysp/LDC) 的 LDC 边缘检测候选来源一致，但该仓库未提供可匹配的 ONNX 发布资产或导出记录。 | 候选仓库没有可由 GitHub 许可证 API 确认的许可证记录；本地也没有许可证文件。 | 不得推断许可证或声称该二进制来自候选仓库。 |
| ResNet18 特征提取器 | 本地只有截断特征图输出，`[1,256,14,14]`；没有 checkpoint、原始框架、导出脚本或上游 URL。 | 无记录。 | 不得推断其是否为 torchvision、其训练权重或许可证。 |

解除 LDC 或 ResNet18 阻断时，必须将以下材料与上述 SHA-256 对应保存到受控的本地或内网位置：不可变上游 URL 或内部制品 ID、下载/生成日期、原始 checkpoint SHA-256、导出脚本和依赖锁定文件、导出命令、上游提交或发布版本、以及书面许可证审批结论。若二进制变更，必须重新完成本台账和参考运行。

## 已验证的张量契约

所有下列预处理描述的是现有原型代码的实际行为，位置为 `imagecmp_v2/onnx_runtime/onnx_pipeline.cpp`；并不证明该预处理与原始训练或导出过程一致。

| 模型 | 输入 | 现有预处理 | 输出及现有解释 |
| --- | --- | --- | --- |
| SuperPoint + LightGlue | `images`: `float32[-1,1,-1,-1]`；参考运行使用 `[2,1,512,512]` | BGR 转灰度，除以 255；两张图片沿 batch 维拼接。 | `keypoints`: `int64[2,1024,2]`，`matches`: `int64[N,3]`，`mscores`: `float32[N]`。仅用作图像对应证据。 |
| LDC | `input_image`: `float32[1,3,360,640]` | BGR 转 RGB，缩放为 640×360，通道优先，数值保留 0–255。 | 5 个 `float32[1,1,360,640]` 输出：`onnx::Concat_241`、`onnx::Concat_244`、`onnx::Concat_250`、`onnx::Concat_259`、`282`。现有原型逐输出 sigmoid、归一化、反转并平均为边缘图。 |
| ResNet18 特征提取器 | `input`: `float32[1,3,224,224]` | BGR 转 RGB（灰度复制为 RGB），缩放至 224×224，除以 255，再用 ImageNet mean `[0.485,0.456,0.406]` 与 std `[0.229,0.224,0.225]` 归一化。 | `output`: `float32[1,256,14,14]`。当前代码将其作为局部感知特征，不是分类 logits。 |

## 可重复本地参考运行

`onnx_model_contract_check` 使用确定性合成张量，不读取 `all_test/` 或其他项目图片。它依次验证文件 SHA-256、节点名、输入/输出类型和形状，并运行一次 CPU ONNX Runtime 推理；同时校验输出元素和与最大绝对值。任何缺失文件、哈希变化、模型契约变化、非有限输出或参考结果变化都会非零退出。

当前基线为 Windows x64、ONNX Runtime 1.26.0、单线程顺序执行：

| 模型 | 合成输入摘要 | 输出元素和 | 输出最大绝对值 |
| --- | --- | ---: | ---: |
| SuperPoint + LightGlue | `[2,1,512,512]` | `1076416.000000` | `503.000000` |
| LDC | `[1,3,360,640]` | `-3233133.436660` | `10.780195` |
| ResNet18 特征提取器 | `[1,3,224,224]` | `5651.709883` | `1.832119` |

比较采用 `0.0001 + 1e-6 × |期望值|` 容差。构建 ONNX Runtime 后端并运行：

```powershell
cmake --build build-onnx --config Release --target verify_onnx_model_contracts
```

也可直接运行已生成的检查器：

```powershell
.\build-onnx\bin\Release\onnx_model_contract_check.exe --model-dir .
```

该命令是发布前和 CI 的模型契约边界；它的成功只证明受控二进制可由当前运行时按已登记契约执行，不构成异动检测准确率、召回率或生产可用性的声明。
