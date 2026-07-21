# P-1 `unreliable` 样本诊断

本报告只读取 `p1_cases.csv`，不读取或导出任何源图像。失败原因可重叠，因此下列单项原因计数不能相加。

## 总览

- 全部 case：385
- 有效输入：383
- `unreliable`：104（有效输入中的 27.2%）
- 诊断类别来自 P-1 固定报告规则，不是生产判定策略。

## 失败原因（允许重叠）

| 原因 | case 数 | 占 unreliable |
| --- | --- | --- |
| ECC correlation below diagnostic minimum after convergence (0.20) | 54 | 51.9% |
| inlier spatial coverage below diagnostic minimum (0.02) | 51 | 49.0% |
| inlier rate below diagnostic minimum (0.40) | 10 | 9.6% |
| inliers below diagnostic minimum (8) | 4 | 3.8% |
| valid overlap below diagnostic minimum (0.60) | 3 | 2.9% |
| projected area ratio outside diagnostic range [0.20, 5.00] | 3 | 2.9% |
| feature matches below diagnostic minimum (12) | 2 | 1.9% |

## 互斥原因组合

| 组合 | case 数 | 占 unreliable |
| --- | --- | --- |
| ECC correlation below diagnostic minimum after convergence (0.20) | 39 | 37.5% |
| inlier spatial coverage below diagnostic minimum (0.02) | 37 | 35.6% |
| inlier spatial coverage below diagnostic minimum (0.02) + ECC correlation below diagnostic minimum after convergence (0.20) | 10 | 9.6% |
| inlier rate below diagnostic minimum (0.40) | 8 | 7.7% |
| feature matches below diagnostic minimum (12) + inliers below diagnostic minimum (8) + valid overlap below diagnostic minimum (0.60) + ECC correlation below diagnostic minimum after convergence (0.20) | 1 | 1.0% |
| inliers below diagnostic minimum (8) + inlier spatial coverage below diagnostic minimum (0.02) + projected area ratio outside diagnostic range [0.20, 5.00] | 1 | 1.0% |
| inliers below diagnostic minimum (8) | 1 | 1.0% |
| inlier rate below diagnostic minimum (0.40) + inlier spatial coverage below diagnostic minimum (0.02) + valid overlap below diagnostic minimum (0.60) + ECC correlation below diagnostic minimum after convergence (0.20) | 1 | 1.0% |
| inlier rate below diagnostic minimum (0.40) + ECC correlation below diagnostic minimum after convergence (0.20) | 1 | 1.0% |
| projected area ratio outside diagnostic range [0.20, 5.00] + ECC correlation below diagnostic minimum after convergence (0.20) | 1 | 1.0% |
| projected area ratio outside diagnostic range [0.20, 5.00] | 1 | 1.0% |
| valid overlap below diagnostic minimum (0.60) | 1 | 1.0% |
| inliers below diagnostic minimum (8) + inlier spatial coverage below diagnostic minimum (0.02) + ECC correlation below diagnostic minimum after convergence (0.20) | 1 | 1.0% |
| feature matches below diagnostic minimum (12) + inlier spatial coverage below diagnostic minimum (0.02) | 1 | 1.0% |

## 互斥证据模式

| 模式 | case 数 | 占 unreliable |
| --- | --- | --- |
| only_low_ecc | 39 | 37.5% |
| only_low_spatial_coverage | 37 | 35.6% |
| low_ecc_and_low_spatial_coverage | 10 | 9.6% |
| only_match_support | 9 | 8.7% |
| multiple_or_other_evidence | 7 | 6.7% |
| only_global_geometry | 2 | 1.9% |

说明：`only_low_ecc` 是只有已收敛 ECC 相关性低于 0.20 的 case；它特别值得人工抽样，因为 ECC 当前在全图（含边缘填充）上计算，且不参与最终几何变换。

## 各证据模式的中位数对照

| 模式 | case 数 | 匹配数 | 内点率 | 空间覆盖 | 有效重叠 | ECC 相关性 |
| --- | --- | --- | --- | --- | --- | --- |
| only_low_ecc | 39 | 66 | 0.7977 | 0.1246 | 0.9984 | 0.1372 |
| only_low_spatial_coverage | 37 | 113 | 0.8333 | 0.008525 | 0.998 | 0.3595 |
| low_ecc_and_low_spatial_coverage | 10 | 89 | 0.8394 | 0.009744 | 0.9991 | 0.137 |
| only_match_support | 9 | 41 | 0.359 | 0.1948 | 0.9971 | 0.4265 |
| multiple_or_other_evidence | 7 | 16 | 0.56 | 0.01053 | 0.8812 | 0.07048 |
| only_global_geometry | 2 | 117 | 0.4819 | 0.09984 | 0.6846 | — |

这张表用于区分两类现象：若只有 ECC 低而匹配/覆盖/重叠正常，优先复核 ECC 的全图计算方式；若只有空间覆盖低而内点率和重投影误差正常，说明特征集中在局部区域，需结合 ROI 大小决定 0.02 是否过严。

## 按案例类型（仅有效输入）

| 案例类型 | 有效 | unreliable | 比例 | usable | unavailable |
| --- | --- | --- | --- | --- | --- |
| 日常 | 292 | 76 | 26.0% | 206 | 10 |
| 位置 | 16 | 8 | 50.0% | 7 | 1 |
| 测温 | 20 | 7 | 35.0% | 9 | 4 |
| UNSPECIFIED | 25 | 7 | 28.0% | 17 | 1 |
| 表计 | 23 | 5 | 21.7% | 18 | 0 |
| 守望位 | 1 | 1 | 100.0% | 0 | 0 |
| 位置 (1) | 1 | 0 | 0.0% | 1 | 0 |
| 日常 (1) | 1 | 0 | 0.0% | 1 | 0 |
| 日常1 | 1 | 0 | 0.0% | 1 | 0 |
| 表记 | 1 | 0 | 0.0% | 1 | 0 |
| 鸟害 | 2 | 0 | 0.0% | 2 | 0 |

## 组件类别前 20（按 unreliable 数；仅有效输入）

| 组件类别 | 有效 | unreliable | 比例 | usable | unavailable |
| --- | --- | --- | --- | --- | --- |
| 179 | 50 | 13 | 26.0% | 35 | 2 |
| 176 | 52 | 13 | 25.0% | 38 | 1 |
| 178 | 44 | 10 | 22.7% | 32 | 2 |
| 177 | 45 | 9 | 20.0% | 35 | 1 |
| 194 | 9 | 6 | 66.7% | 3 | 0 |
| 288 | 17 | 6 | 35.3% | 7 | 4 |
| 264 | 5 | 5 | 100.0% | 0 | 0 |
| 287 | 20 | 5 | 25.0% | 13 | 2 |
| 180 | 40 | 5 | 12.5% | 34 | 1 |
| 226 | 11 | 4 | 36.4% | 7 | 0 |
| 217 | 19 | 4 | 21.1% | 15 | 0 |
| 191 | 4 | 3 | 75.0% | 1 | 0 |
| 247 | 5 | 3 | 60.0% | 2 | 0 |
| 190 | 11 | 3 | 27.3% | 7 | 1 |
| 175 | 15 | 3 | 20.0% | 11 | 1 |
| 279 | 4 | 2 | 50.0% | 2 | 0 |
| 188 | 5 | 2 | 40.0% | 3 | 0 |
| 201 | 5 | 2 | 40.0% | 3 | 0 |
| 248 | 5 | 2 | 40.0% | 3 | 0 |
| 172 | 10 | 2 | 20.0% | 8 | 0 |

## 关键指标：unreliable 与 usable 对照

| 指标 | unreliable 中位数 | unreliable P95 | usable 中位数 | usable P95 |
| --- | --- | --- | --- | --- |
| feature_match_count | 70.5 | 519.8 | 150 | 810 |
| inlier_count | 55.5 | 456 | 125 | 727.6 |
| inlier_rate | 0.7649 | 0.9617 | 0.8394 | 0.9579 |
| reprojection_error_pixels | 0.3572 | 0.8823 | 0.3568 | 1.092 |
| spatial_coverage | 0.02061 | 0.2634 | 0.2064 | 0.5419 |
| center_displacement_relative_diagonal | 0.004461 | 0.2259 | 0.0003336 | 0.01013 |
| corner_displacement_median_pixels | 19.06 | 701.5 | 1.726 | 46.55 |
| projected_area_ratio | 0.9978 | 1.264 | 0.9986 | 1.014 |
| valid_overlap_ratio | 0.9976 | 1 | 0.9987 | 1 |
| ecc_correlation | 0.1794 | 0.48 | 0.5124 | 0.9086 |

`p95` 不是阈值，只是当前分布的第 95 百分位。完整的最小值、均值、最大值和每个样本的原因见同目录 JSON 与排序 CSV。
