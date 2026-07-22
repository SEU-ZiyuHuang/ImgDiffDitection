# imagecmp_py：供电设备单部件图像异动检测（#4）

这是新的 Python 实现边界。调用方提交标准图、实时图、标准图中的一个 YOLO ROI、版本化配置和本地证据目录；服务只比较这个预期部件，不推断设备、机位或部件身份。

它实现的是 #4 的单部件检测切片。整图多 ROI 聚合、内网标定与上线可用性门槛属于后续阶段。

## 安装与测试

```powershell
& "C:\Users\47326\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
& "C:\Users\47326\AppData\Local\Programs\Python\Python312\python.exe" test_service.py
```

测试只生成合成图片，不读取或外发 `all_test/` 中的涉密数据。

## 调用方式

```python
from pathlib import Path
from imagecmp import ImageComparisonService

result = ImageComparisonService().compare(
    standard_path=Path("标准图.jpg"),
    live_path=Path("实时图.jpg"),
    roi="17 0.5 0.5 0.4 0.5",
    config_path=Path("configs/development-default-v1.json"),
    output_dir=Path("./output"),
)

print(result.state.value)
for region in result.detection_regions:
    # 坐标相对于原始实时图
    print(region.x, region.y, region.width, region.height, region.confidence)
```

命令行入口：

```powershell
python cli.py compare `
  --standard 标准图.jpg `
  --live 实时图.jpg `
  --roi "17 0.5 0.5 0.4 0.5" `
  --config configs/development-default-v1.json `
  --output .\output
```

`config_path` 省略时只会使用带有 `development-default-v1` 版本号的开发配置，且会向标准错误输出警告；它只用于本地合成测试。日常检测必须传入完整、经内网标定的 JSON 配置。显式传入不存在、缺字段或无效的配置会抛出异常，绝不会静默改用默认值。

## 三种对外状态

| 状态 | 含义 |
| --- | --- |
| `no_change_high_confidence` | 配准质量和 ROI 有效重叠均通过配置门槛，未检出差异候选。 |
| `change_detected` | 配准可信，且至少一个颜色或梯度差异候选达到配置的判定边界。 |
| `detection_unavailable` | 图像可读且请求有效，但特征匹配或配准质量不足。结果包含 `match_uncertain` 或 `alignment_failed` 原因，绝不显示为正常。 |

缺失文件、图片无法解码、ROI 不合法、配置不合法或证据目录无法写入属于显式 Python 异常，不会被编码成上述业务状态。

每个有效比较调用（包括 `detection_unavailable`）都会在调用方指定的本地目录写入以下五个证据文件：

| 文件 | 内容 |
| --- | --- |
| `alignment.png` | 标准图与实时图的配准诊断；没有可信变换时会明确标示。 |
| `valid_mask.png` | 标准坐标系中可用于比较的重叠像素。 |
| `difference_mask.png` | 标准坐标系中的二值差异候选。 |
| `difference_heatmap.png` | 差异分数热力图。 |
| `annotated.png` | 原始实时图上的 ROI、差异框与不可用状态说明。 |

## 当前算法边界

当前实现先用 ORB 特征、RANSAC 单应矩阵和 ECC 精调将实时图变换到标准坐标系，生成有效重叠 mask；只有质量门槛通过后才比较 Lab 颜色差与梯度幅值差。差异通道采用保守 OR 融合：小候选和低置信候选仍被保留并标注，而不会因为面积过滤而被静默当作无异动。

仓库根目录的 ONNX 模型来源与输入/输出契约已由 #3 记录，但 #4 的 Python 实现不把这些预训练模型伪称为设备身份或缺陷分类器，也不因此作出召回率、漏检率或生产准确率承诺。`development-default-v1.json` 不是标定结果，不能用于生产上线。
