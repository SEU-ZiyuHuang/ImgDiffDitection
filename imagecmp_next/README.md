# P-1 local dataset characterization

`imagecmp_characterize` analyzes only a directory already available in the authorized local or intranet environment. It never uploads, copies, or exposes source images. Its output is descriptive: it must not be used to claim anomaly recall, missed-detection rate, precision, or production readiness.

## Build

```powershell
cmake -S imagecmp_next -B imagecmp_next/build `
  -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build imagecmp_next/build --config Release
```

## Run

The default case layout is one directory per case, each containing `标准源图.jpg`, `对比截图.jpg`, and `标准源图坐标.txt` (YOLO: `class cx cy width height`). Different names can be supplied explicitly.

```powershell
.\imagecmp_next\build\Release\imagecmp_characterize.exe `
  --dataset .\all_test `
  --output .\local-analysis
```

The caller-selected output directory receives only local reports:

- `p1_characterization_report.json`: global distributions and the same distributions grouped by YOLO component category and directory-derived case type.
- `p1_cases.csv`: completeness/validation status and per-case image, ROI, quality, geometric, feature-match, alignment, and valid-overlap observations.
- `p1_groups.csv`: compact component-category and case-type aggregate view.

Component-category grouping uses the YOLO class ID. Cases without a parseable ROI are retained in the `UNCLASSIFIED` category so missing/invalid input remains visible in failure statistics. The report records raw, unaligned luma variation separately from alignment evidence. ORB feature matching, RANSAC homography, and ECC are measured to choose a future P1 alignment strategy; they do not make a normal/anomalous conclusion and are not production thresholds.

See [P1_DATA_PROCESSING.md](P1_DATA_PROCESSING.md) for the exact processing flow, formulas, report fields, and the non-production alignment diagnostic rules used by the current implementation.
