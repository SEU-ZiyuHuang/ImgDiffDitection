# -*- coding: utf-8 -*-
"""Full-population illumination-rescue audit for ALL NCC-blocked components
in the 20260722_135830 batch (local-only).

For every component blocked by the appearance NCC gate, recompute
post-alignment NCC after three illumination-robust variants and decide:
  rescued  (any robust variant >= 0.35) -> photometric-only mismatch
  blocked  (all variants < 0.35)        -> genuine mismatch

Writes the full per-component list to CSV and prints a bucket summary.
"""
import csv, io, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\47326\Desktop\异动检测\imagecmp_py')

from imagecmp.alignment import align
from imagecmp.config import load_config
from imagecmp.references import read_color_image
from imagecmp.roi import read_rois, roi_to_pixel_rect

ALL_TEST = Path(r'C:\Users\47326\Desktop\异动检测\all_test')
BATCH = Path(r'D:\异动检测批量输出\all_test_calibration_parallel_20260722_135830')
CSV_IN = BATCH / 'analysis' / 'component_observations.csv'
CSV_OUT = Path(r'D:\异动检测批量输出\ncc_rescue_full_20260722.csv')
PROGRESS = Path(r'C:\Users\47326\AppData\Local\Temp\opencode\rescue_progress.txt')
CONFIG = load_config(Path(r'C:\Users\47326\Desktop\异动检测\imagecmp_py\configs\development-default-v1.json'))
THRESH = 0.35

def num(x):
    try:
        return float(x)
    except Exception:
        return float('nan')

rows = list(csv.DictReader(io.open(CSV_IN, encoding='utf-8')))
blocked = [r for r in rows if 'NCC below' in (r['component_mapping_failure_detail'] or '')]
print(f'NCC-blocked components: {len(blocked)}')

def masked_ncc(a, b, mask):
    va = a[mask].astype(np.float64)
    vb = b[mask].astype(np.float64)
    if va.size < 2:
        return float('nan')
    ca = va - va.mean()
    cb = vb - vb.mean()
    d = np.linalg.norm(ca) * np.linalg.norm(cb)
    if d <= 1e-12:
        return float('nan')
    return float(ca @ cb / d)

def hist_match(src_vals, ref_vals):
    hs, _ = np.histogram(src_vals, bins=256, range=(0, 256), density=True)
    hr, _ = np.histogram(ref_vals, bins=256, range=(0, 256), density=True)
    cs = np.cumsum(hs)
    cr = np.cumsum(hr)
    lut = np.interp(cs, cr, np.arange(256))
    return np.clip(lut, 0, 255).astype(np.uint8)

def local_zscore(gray):
    f = gray.astype(np.float32)
    mu = cv2.GaussianBlur(f, (0, 0), 15)
    var = cv2.GaussianBlur(f * f, (0, 0), 15) - mu * mu
    return (f - mu) / (np.sqrt(np.maximum(var, 0)) + 1.0)

def grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)

def bucket(v):
    if v < 0:
        return 'A_neg(<0)'
    if v < 0.15:
        return 'B_[0,0.15)'
    if v < 0.25:
        return 'C_[0.15,0.25)'
    return 'D_[0.25,0.35)'

align_cache = {}
out_rows = []
errors = 0
for i, r in enumerate(blocked):
    case = r['case']
    idx = int(r['component_index'])
    rec_ncc = num(r['appearance_ncc'])
    b = bucket(rec_ncc)
    status = 'error'
    ncc_orig = ncc_hist = ncc_ln = ncc_grad = float('nan')
    rescued = 0
    try:
        if case not in align_cache:
            obs = json.load(io.open(BATCH / case / 'calibration_observation.json', encoding='utf-8'))
            ref_id = obs['selected_reference']['id']
            ref_file = '标准源图.jpg' if ref_id == 'primary' else f"新增标准源图{ref_id.split('_')[1]}.jpg"
            std = read_color_image(ALL_TEST / case / ref_file)
            live = read_color_image(ALL_TEST / case / '对比截图.jpg')
            if std is None or live is None:
                align_cache[case] = None
            else:
                res = align(cv2.cvtColor(std, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), CONFIG)
                align_cache[case] = (std, res)
        entry = align_cache[case]
        if entry is None:
            status = 'load_error'
        else:
            std, res = entry
            if res.aligned_live is None or res.valid_mask is None:
                status = 'align_unavailable'
            else:
                rois = read_rois(ALL_TEST / case / '标准源图坐标.txt').rois
                roi = rois[idx]
                H, W = std.shape[:2]
                x, y, w, h = roi_to_pixel_rect(roi, W, H)
                x = max(0, x); y = max(0, y)
                w = min(w, W - x); h = min(h, H - y)
                std_roi = cv2.cvtColor(std[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                live_roi = res.aligned_live[y:y + h, x:x + w]
                if live_roi.ndim == 3:
                    live_roi = cv2.cvtColor(live_roi, cv2.COLOR_BGR2GRAY)
                mask = res.valid_mask[y:y + h, x:x + w] > 0
                if mask.sum() < 50:
                    status = 'mask_too_small'
                else:
                    ncc_orig = masked_ncc(std_roi, live_roi, mask)
                    lut = hist_match(live_roi[mask], std_roi[mask])
                    ncc_hist = masked_ncc(std_roi, lut[live_roi], mask)
                    ncc_ln = masked_ncc(local_zscore(std_roi), local_zscore(live_roi), mask)
                    ncc_grad = masked_ncc(grad_mag(std_roi), grad_mag(live_roi), mask)
                    rescued = int(max(ncc_hist, ncc_ln, ncc_grad) >= THRESH)
                    status = 'rescued' if rescued else 'blocked'
    except Exception as exc:
        status = f'error:{type(exc).__name__}'
        errors += 1
    out_rows.append({
        'case': case, 'component_index': idx, 'case_type': r['case_type'],
        'recorded_ncc': rec_ncc, 'bucket': b,
        'ncc_orig': ncc_orig, 'ncc_hist': ncc_hist, 'ncc_ln': ncc_ln, 'ncc_grad': ncc_grad,
        'rescued': rescued, 'status': status,
    })
    if (i + 1) % 10 == 0 or i + 1 == len(blocked):
        done = i + 1
        PROGRESS.write_text(f'{done}/{len(blocked)}', encoding='utf-8')
        print(f'  ... {done}/{len(blocked)}', flush=True)

with io.open(CSV_OUT, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print('\n== bucket summary ==')
agg = defaultdict(lambda: [0, 0])
for r in out_rows:
    agg[r['bucket']][0] += 1
    agg[r['bucket']][1] += r['rescued']
total_n = total_r = 0
for b in sorted(agg):
    n_c, n_r = agg[b]
    total_n += n_c
    total_r += n_r
    print(f'  {b:14} n={n_c:3} rescued={n_r:3} ({n_r/n_c*100:5.1f}%)')
print(f'  {"TOTAL":14} n={total_n:3} rescued={total_r:3} ({total_r/total_n*100:5.1f}%)')
print(f'\nerrors/skips: {errors}')
print(f'full list written: {CSV_OUT}')
PROGRESS.write_text('DONE', encoding='utf-8')
