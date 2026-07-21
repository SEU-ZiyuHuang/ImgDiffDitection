# P-1 数据处理与配准诊断流程

本文档描述 `imagecmp_characterize` 当前实际执行的本地数据集表征流程（报告模式 `p1-characterization-v2`）。它不是异动检测器：不输出“有/无异动”结论，不计算召回率、漏检率或生产可用性。

实现的唯一公共入口是 `ImageComparisonService::analyzeDataset`；命令行程序仅把 `--dataset`、`--output` 和可选文件名传入这个入口。实现位于 `src/image_comparison_service.cpp`。

## 1. 范围、输入和输出

- 仅枚举 `dataset_root` 的**直接子目录**，每个子目录是一条 case；不会递归遍历更深的目录。
- 每条 case 需要一张标准图、一张对比图和一个 YOLO ROI 文本文件。缺任何文件为 `incomplete`；文件齐全但 ROI 或图像不可用为 `invalid`。
- 图像在内存中解码和计算；程序不会复制、上传或导出源图像。
- 输出目录只写三份本地报告：
  - `p1_characterization_report.json`：全量及按组件类别/案例类型分组的分布；
  - `p1_cases.csv`：每个 case 的原始观测值、诊断类别和原因；
  - `p1_groups.csv`：各分组的计数和均值。

目录名最后一个 `_` 后的文本为 `case_type`；ROI 的 YOLO 类别 ID 为 `component_category`。一条 case 有多个类别时，会在每个对应类别分组中各出现一次；ROI 无法解析时归入 `UNCLASSIFIED`，不会从失败统计中消失。

## 2. 处理流程

```text
直接子目录
  -> 文件完整性检查
  -> YOLO ROI 准入与边界归一化
  -> 图像解码、灰度与基础质量统计
  -> ORB 特征 + 比率匹配
  -> RANSAC 单应性 H（标准图 -> 对比图）
  -> 实际投影几何、有效重叠、ECC 诊断
  -> 配准诊断分级
  -> JSON / case CSV / group CSV
```

下文给出的符号为：标准图尺寸 `(W_s,H_s)`，对比图尺寸 `(W_l,H_l)`；标准图点为 `p=(x,y)`，对比图对应点为 `q`；单应性为 `H`。

## 3. 数据准入

### 3.1 文件与图像

程序先确认三份必需文件都是普通文件，再用 OpenCV `imdecode` 解码。标准图或对比图解码失败会记录 `unreadable standard image` 或 `unreadable live image`，该 case 为 `invalid`。这属于输入准入失败，绝不是“未检测到异动”。

### 3.2 YOLO ROI

每个非空行必须严格为五列：

```text
class_id center_x center_y width height
```

所有坐标必须有限，且 `width>0`、`height>0`。先将中心宽高换算为边界：

```text
left   = center_x - width / 2
right  = center_x + width / 2
top    = center_y - height / 2
bottom = center_y + height / 2
```

允许原始四条边落在 `[-0.01, 1.01]` 内。若只是在此容差内越界，程序裁剪到 `[0,1]`，再由裁剪后的四边重建 `center_x`、`center_y`、`width`、`height`，并记录 `ROI boundary normalized at line N`。超出容差、裁剪后宽/高不再为正、格式错误或无 ROI 时均为 `invalid ROI`。

这条规则防止因浮点序列化造成的约 `10^-6` 级边界误判，同时不会接受真正越出图像的大框。

## 4. 图像与 ROI 基础指标

解码后的 1/3/4 通道图像被转换为单通道灰度图。对每张灰度图 `I`：

- `brightness`：像素均值 `mean(I)`；
- `contrast`：像素标准差 `std(I)`；
- `sharpness`：拉普拉斯响应的方差 `Var(Laplacian(I))`。

成对差值 `brightness_delta`、`contrast_delta`、`sharpness_delta` 均为两张图相应指标之差的绝对值。

为描述未经配准时的差异，对比图先用线性插值缩放到标准图尺寸，再计算：

```text
raw_luma_mad = mean(|I_standard - resize(I_live)|)
```

它会同时受到视角、平移、光照、遮挡、模糊和真实异动的影响，因而只能作为原始数据差异描述，不能作为异动分数。

`aspect_ratio_delta = |W_s/H_s - W_l/H_l|`。每个 ROI 的相对面积为 `width*height`，宽高比为 `width/height`；case 级指标取全部 ROI 的算术平均。在组件类别分组中，ROI 面积和宽高比只使用该类别的 ROI；其它 case 级图像/匹配指标按该 case 复制到相应类别组。

## 5. 特征匹配与单应性

实现函数：`analyzeFeatureAndAlignmentEvidence`。

1. 标准图和对比图分别用 ORB 提取至多 2000 个关键点与二进制描述子。
2. 用 Hamming 距离做 2 近邻匹配。仅保留满足 Lowe 比率规则的第一近邻：

   ```text
   distance_1 < 0.75 * distance_2
   ```

   保留数为 `feature_match_count`。它是候选对应点数，不等于可靠匹配数。
3. 当候选数至少为 4 时，调用 `findHomography(standard_points, live_points, RANSAC, 3.0)`，估计从标准图到对比图的单应性 `H`。RANSAC 的 3.0 是内点重投影阈值，单位为像素。
4. `inlier_count` 是 RANSAC 内点数，`inlier_rate = inlier_count / feature_match_count`。
5. 对内点计算平均重投影误差：

   ```text
   reprojection_error_pixels = (1/N) * Σ ||project(H, p_i) - q_i||_2
   ```

6. 当内点不少于 3 时，对标准图内点取凸包，计算：

   ```text
   spatial_coverage = area(convex_hull(inlier_standard_points)) / (W_s * H_s)
   ```

低重投影误差只证明已有内点附近能拟合；若内点集中在很小区域，`spatial_coverage` 仍会很低，因此二者必须一起看。

## 6. 单应性的实际投影几何

单应性按齐次坐标作用：

```text
[u v w]^T = H * [x y 1]^T
project(H, (x,y)) = (u/w, v/w)
```

报告**不再**输出旧版的 `translation_pixels`、`scale_estimate`、`rotation_degrees`。旧版把 `H[0,2]`、`H[1,2]` 和左上 2×2 子矩阵直接当作平移、缩放、旋转；只要 `H` 含透视项，这些系数就不再是可解释的物理位移/缩放/旋转，退化矩阵可产生远大于图像尺寸的伪数值。

现在计算下列实际投影指标。若 `H` 的系数或投影点不有限、四角投影多边形不凸、或其面积不大于 1 像素，则 `projected_geometry_valid=false`。

| CSV 指标 | 计算 | 含义与正确性边界 |
| --- | --- | --- |
| `center_displacement_pixels` | `|| project(H, ((W_s-1)/2,(H_s-1)/2)) - ((W_l-1)/2,(H_l-1)/2) ||_2` | 标准图中心投影后相对对比图中心的真实坐标距离，单位为对比图像素。它是投影结果，不是矩阵系数。 |
| `center_displacement_relative_diagonal` | `center_displacement_pixels / sqrt(W_l^2+H_l^2)` | 无量纲化中心偏移，可跨分辨率比较。 |
| `corner_displacement_median_pixels` | 标准图四个角分别投影，与对比图相应四角的四个距离取中位数 | 描述整体边界偏离，抗单个角极端值。 |
| `projected_corners_in_live_frame` | 四个投影角中落入 `[0,W_l-1] × [0,H_l-1]` 的个数 | 描述值，不参与分级：即使只有很小的纯平移，也可能只有一个原始角仍在画面内。 |
| `projected_area_ratio` | `abs(area(projected_standard_corners)) / (W_l*H_l)` | 标准图经 `H` 变换后在对比图坐标系的面积比例。极小/极大值是几何退化或极端视场变化的证据。 |

中心和角点指标与 `H` 的方向一致：当前代码中的 `H` 始终是“标准图坐标 -> 对比图坐标”。

## 7. 有效重叠与 ECC

### 7.1 有效重叠

若 `H` 可逆，记 `H_inv = H^{-1}`。代码建立全 255 的对比图掩码，并用 `H_inv` 将它透视变换到标准图画布：

```text
M = warpPerspective(ones(W_l,H_l), H_inv, output_size=(W_s,H_s))
valid_overlap_ratio = count(M > 0) / (W_s * H_s)
```

它表示配准到标准图画布后，哪些像素仍来自原对比图，而不是边界填充。高重叠只说明可比较区域大，不证明特征对应正确；必须结合匹配、覆盖率和投影几何。

### 7.2 ECC

代码先以 `H_inv` 将对比灰度图投影到标准图画布，再调用：

```text
findTransformECC(standard_gray, aligned_live, initial_identity_affine,
                 MOTION_AFFINE, max_iterations=50, epsilon=1e-5)
```

ECC 在仿射增量变换下最大化两幅图的增强相关系数；`ecc_correlation` 是求解成功时的最终相关值，`ecc_converged` 表示没有抛出 OpenCV 收敛异常。当前实现只把它作为**诊断观测**：求得的 ECC 仿射矩阵没有再写回或用于生成最终对齐图；而且调用未传入有效重叠掩码，黑色边界也会参与该全图统计。因此：

- ECC 未收敛不单独证明配准失败；
- ECC 收敛也不单独证明对齐正确；
- ECC 很低且已收敛，是与其它证据联合判为不可靠的信号。

## 8. P-1 配准诊断分级

分级的实现函数为 `finalizeAlignmentDiagnostic`，其目的只是让本次数据集表征可审计。固定阈值会写入 JSON 的 `alignment_diagnostic_policy`，但不是生产策略，后续必须用带异动标注的数据校准。

| 类别 | 代码条件 |
| --- | --- |
| `unavailable` | 没有求得 `H`；或投影几何无效；或 `H` 无法得到有效重叠。 |
| `unreliable` | 有 `H`、投影几何和有效重叠，但下列任一诊断条件不满足。`alignment_diagnostic_reasons` 逐项列出原因。 |
| `usable` | 同时满足下列全部条件。它表示“本轮 P-1 证据足以作为可比较候选”，不表示无异动或生产批准。 |

`usable` 的全部条件如下：

```text
feature_match_count          >= 12
inlier_count                 >= 8
inlier_rate                  >= 0.40
reprojection_error_pixels    <= 3.0
spatial_coverage             >= 0.02
0.20 <= projected_area_ratio <= 5.00
valid_overlap_ratio          >= 0.60
if ECC converged: ecc_correlation >= 0.20
```

ECC 是条件性门槛：未收敛不会单独把 case 降级；一旦已收敛但相关性低于 0.20，会在 `alignment_diagnostic_reasons` 中留下明确原因。中心偏移和角点在画面内数量不设硬门槛，原因是它们会随合法的取景平移而变化；有效重叠与投影面积已经承担了更稳定的几何保护作用。

## 9. 聚合统计和复核方法

每个有限数值进入全量和分组分布。分位数使用线性插值：对排序后 `n` 个样本和分位 `p`，索引为 `p*(n-1)`，在相邻整数索引间插值。因此 JSON 的 `p05`、`median`、`p95` 都可由同一份 `p1_cases.csv` 复算。

复核某条 case 时，建议按以下顺序阅读 CSV：

1. `status`、`errors`、`warnings`：先排除输入准入问题；
2. `feature_match_count`、`inlier_count`、`inlier_rate`、`reprojection_error_pixels`、`spatial_coverage`：判断 `H` 的局部支持是否充分；
3. `projected_geometry_valid`、中心/角点投影指标、`projected_area_ratio`、`valid_overlap_ratio`：判断该 `H` 对整幅画面是否几何合理；
4. `ecc_converged`、`ecc_correlation`：只作为补充证据；
5. `alignment_diagnostic` 与 `alignment_diagnostic_reasons`：查看代码按固定 P-1 规则给出的可追溯归类。

## 10. 代码对应关系与测试

| 文档步骤 | 实现位置 |
| --- | --- |
| ROI 解析、容差裁剪 | `readRois` |
| 图像解码、灰度和基础质量指标 | `readImage`、`toGray`、`addQualityMetrics`、`analyzeImages` |
| ORB、比率测试、RANSAC、内点指标 | `analyzeFeatureAndAlignmentEvidence` |
| 实际中心/角点投影和投影面积 | `addProjectedGeometryMetrics` |
| 有效重叠和 ECC 观测 | `analyzeFeatureAndAlignmentEvidence` 中 `warpPerspective` / `findTransformECC` 调用 |
| 三类配准诊断及原因 | `finalizeAlignmentDiagnostic` |
| CSV、JSON、分组统计 | `writeCaseReport`、`writeGroupReport`、`writeAggregateReport` |

测试 `tests/image_comparison_service_test.cpp` 构造了一个已知小平移 case，并验证它使用新的中心投影位移而不是已删除的矩阵系数字段，且可获得 `usable` 诊断；另构造无特征图像，验证其为 `unavailable` 并写出原因。测试还覆盖完整/缺失/无效 ROI、边界容差归一化以及三份本地报告的生成。
