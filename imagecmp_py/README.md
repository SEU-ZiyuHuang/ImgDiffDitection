# imagecmp_py：供电设备图像异动检测

这是供电设备图像异动检测模块的 Python 实现。它比较标准图与实时图，先确认两张图能否可靠重合，再比较调用方指定的预期部件。

模块有两种明确的运行模式：

- **日常检测模式**：必须加载完整、带版本号的标定配置；输出部件和整图的业务结论。
- **标定模式**：只输出对齐指标、差异候选和证据图；绝不输出正常或异动的业务结论。

## 安装与测试

```powershell
python -m pip install -r .\imagecmp_py\requirements-dev.txt
python -m pytest .\imagecmp_py\test_service.py -q
```

自动测试仅使用程序生成的合成图片，不读取或外发 `all_test/` 中的内部样本。

## 日常检测模式

日常检测必须提供完整标定配置。配置缺失、配置不存在、配置字段不完整或数值非法时，程序直接报错，而不会使用开发默认值给出业务结论。

### 单个部件

```powershell
python .\imagecmp_py\cli.py compare `
  --standard 标准图.jpg `
  --live 实时图.jpg `
  --roi "17 0.5 0.5 0.4 0.5" `
  --config 标定配置.json `
  --output .\output\single-component
```

部件框格式为：`类别 中心横坐标 中心纵坐标 宽 高`，四个坐标均已归一化到零至一。

### 一个案例中的多个部件和多张参考图

案例目录支持以下本地文件约定：

```text
案例目录/
├─ 对比截图.jpg
├─ 标准源图.jpg
├─ 新增标准源图0.jpg          （可选）
├─ 新增标准源图1.jpg          （可选，编号必须连续）
└─ 标准源图坐标.txt
```

全部参考图共享同一份部件框文件。程序会先分别评估每张参考图与实时图的对齐质量，再选择一张最可信的参考图供**全部部件**使用；不会对不同部件任意混用不同参考图。

```powershell
python .\imagecmp_py\cli.py compare-case `
  --case-directory .\all_test\某个案例目录 `
  --config 标定配置.json `
  --output .\output\daily-case
```

整图汇总规则：

| 部件结果 | 整图结果 |
| --- | --- |
| 任意部件检出异动 | `change_detected` |
| 无异动但任意部件检测不可用 | `detection_unavailable` |
| 全部部件均高置信无异动 | `no_change_high_confidence` |

只有最后一种情况才允许整图显示正常。

## 标定模式

标定模式用于内部正常样本分析。它会记录参考图选择、特征匹配、对齐质量、有效比较区域、差异候选和证据图，但不会输出正常、异动或整图业务结论。

```powershell
python .\imagecmp_py\cli.py calibrate-case `
  --case-directory .\all_test\某个案例目录 `
  --output .\output\calibration-case
```

可选的 `--processing-config` 用于指定完整处理配置；省略时使用开发处理配置，仅供生成原始观察指标。

## 输出内容

每个部件目录都包含：

- `alignment.png`：对齐诊断图；
- `valid_mask.png`：可比较像素区域；
- `difference_mask.png`：差异候选区域；
- `difference_heatmap.png`：差异程度图；
- `annotated.png`：实时图中的部件框和差异框。

日常检测额外写入 `daily_result.json`，其中包含整图结论、参考图选择、全部部件结论、证据路径和配置版本。标定模式写入 `calibration_observation.json`，其中只包含观察结果和证据路径。

## 可信对齐与部件映射闸门

差异检测之前，系统必须为每个预期部件保留并通过以下证据链：

1. 特征匹配、RANSAC 内点、重投影误差与空间覆盖度建立标准图到实时图的几何变换；
2. ECC 局部精调必须收敛；近似恒等变换、低空间覆盖度和 ECC 未收敛会被识别为远近景/父子视图疑似，输出 `detection_unavailable` / `match_uncertain`；
3. 预期部件框投影到实时图后，候选区域必须在画面内且具有足够的有效重叠；
4. 在 `valid_mask ∩ ROI` 上记录 NCC 与 SSIM 外观一致性。整体外观不一致不会进入差异检测；
5. 通过单应变换估计实时部件的原生有效分辨率。比较会降至两图的较低共同分辨率；若低于配置下限，输出 `detection_unavailable`，不会以放大图做细微异动结论。

这些指标均位于 `alignment_metrics`，包括 `alignment_stages`、`component_mapping_stages`、ECC 状态、候选区域有效比例、NCC/SSIM 和有效分辨率。`alignment.png` 还会显示抽样的 RANSAC 内点连线，便于人工复核匹配证据。

相关门槛是完整标定配置中的 `alignment` 字段，开发默认值只用于本地观察和合成测试；正式日常检测仍必须由后续标定给出版本化配置。

## 当前边界

- 开发配置不是正式标定配置，不能用于现场生产判断。
- 内部样本、证据图和派生报告只能留在授权本地或内网环境。
- 当前不作异动准确率、召回率、漏检率或零漏检承诺；这些需要后续受控真实异动样本验证。
- 现有模型文件不是设备身份识别模型，也不是通用缺陷分类模型。
