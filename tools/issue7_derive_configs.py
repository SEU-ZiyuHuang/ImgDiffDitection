# -*- coding: utf-8 -*-
"""Derive candidate calibrated configs from the 135830 batch observations
and simulate component availability under each candidate, offline.

Policy A (strict): min-metrics at p05 of normal distribution, max-metrics at p95.
Policy B (loose): min-metrics at p01, max-metrics at p99.
Policy C (per-case-type): Policy A rules per case-type group, global fallback
when a group has fewer than MIN_GROUP components.

Exclusions: 鸟害-type cases (user decision 2026-07-22).
"""
import csv, io, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
CSV = r'D:\异动检测批量输出\all_test_calibration_parallel_20260722_135830\analysis\component_observations.csv'
MIN_GROUP = 30

def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None

rows = list(csv.DictReader(io.open(CSV, encoding='utf-8')))

# ---- exclusion: 鸟害 ----
excluded = sorted({r['case'] for r in rows if r['case_type'] == '鸟害'})
rows = [r for r in rows if r['case_type'] != '鸟害']
print(f'excluded 鸟害 cases: {excluded}')
print(f'components after exclusion: {len(rows)}')

# ---- case-type grouping (merge near-duplicates) ----
def group_of(ct):
    if ct.startswith('日常'): return '日常'
    if ct in ('表计', '表记'): return '表计'
    if ct.startswith('位置'): return '位置'
    if ct == '测温': return '测温'
    if ct in ('', None): return 'unspecified'
    return ct  # unspecified / 守望位 stay as-is

for r in rows:
    r['_group'] = group_of(r['case_type'])

# ---- metric directions ----
MIN_METRICS = ['feature_match_count', 'inlier_count', 'inlier_rate', 'spatial_coverage',
               'valid_overlap_ratio', 'ecc_correlation',
               'candidate_in_frame_ratio', 'roi_valid_overlap_ratio',
               'effective_resolution_scale', 'effective_live_width_pixels',
               'effective_live_height_pixels', 'appearance_ncc', 'appearance_ssim']
MAX_METRICS = ['reprojection_error_pixels']

def quantile(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals: return None
    k = min(len(vals) - 1, max(0, int(q * len(vals))))
    return vals[k]

def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return {'n': len(vals), 'p01': quantile(vals, .01), 'p05': quantile(vals, .05),
            'p50': quantile(vals, .5), 'p95': quantile(vals, .95), 'p99': quantile(vals, .99)}

groups = defaultdict(list)
for r in rows:
    groups[r['_group']].append(r)
print('groups:', {k: len(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))})

# ---- derive thresholds ----
def derive(subset_rows, qa_min, qa_max):
    t = {}
    for m in MIN_METRICS:
        s = stats([num(r.get(m)) for r in subset_rows])
        t[m] = s[qa_min] if s else None
    for m in MAX_METRICS:
        s = stats([num(r.get(m)) for r in subset_rows])
        t[m] = s[qa_max] if s else None
    return t

confA = derive(rows, 'p05', 'p95')          # strict
confB = derive(rows, 'p01', 'p99')          # loose
confC = {}
for g, grows in groups.items():
    if len(grows) >= MIN_GROUP:
        confC[g] = derive(grows, 'p05', 'p95')
    else:
        confC[g] = confA                    # fallback to global strict

def th_for(config, row, metric):
    if isinstance(config, dict) and '_per_group' in config:
        gtab = config['_per_group']
        return gtab.get(row['_group'], confA).get(metric)
    return config.get(metric)

confC_wrap = {'_per_group': confC}

# ---- simulate mapping-level availability ----
# Structural prerequisites (transform/mask/projection) are threshold-independent:
# components that failed there keep failing under every candidate config.
def simulate(config):
    usable, blocked_by = 0, defaultdict(int)
    for r in rows:
        detail = r['component_mapping_failure_detail'] or ''
        structural = any(s in detail for s in (
            'shape differs', 'mask is unavailable', 'no finite standard-to-live',
            'could not be projected', 'invalid geometry', 'outside the live-image frame',
            'non-finite local sampling', 'could not be measured'))
        if structural:
            blocked_by['structural'] += 1
            continue
        ok = True
        for m, label in [('candidate_in_frame_ratio', 'B_in_frame'),
                         ('roi_valid_overlap_ratio', 'B_roi_overlap'),
                         ('effective_resolution_scale', 'C_res_scale'),
                         ('effective_live_width_pixels', 'C_res_w'),
                         ('effective_live_height_pixels', 'C_res_h'),
                         ('appearance_ncc', 'D_ncc'),
                         ('appearance_ssim', 'D_ssim')]:
            v = num(r.get(m))
            t = th_for(config, r, m) if '_per_group' in config else config.get(m)
            if v is None or t is None:
                if v is None and m in ('appearance_ncc', 'appearance_ssim'):
                    ok = False; blocked_by[label + '_nan'] += 1
                    break
                continue
            if v < t:
                ok = False; blocked_by[label] += 1
                break
        if ok:
            usable += 1
    return usable, blocked_by

resA = simulate(confA)
resB = simulate(confB)
resC = simulate(confC_wrap)

n = len(rows)
print('\n== simulated component availability (mapping level) ==')
for tag, (u, bb) in [('A strict(p05/p95)', resA), ('B loose (p01/p99)', resB), ('C per-type      ', resC)]:
    print(f'  {tag}: usable {u}/{n} = {u/n*100:.1f}%   blocked {n-u}')
    for k, v in sorted(bb.items(), key=lambda kv: -kv[1]):
        print(f'      {v:4} {k}')

# ---- derived thresholds comparison ----
print('\n== derived thresholds (global) ==')
print(f'{"metric":34} {"dev-current":>12} {"A p05/p95":>12} {"B p01/p99":>12}')
cur = {'feature_match_count': 12, 'inlier_count': 8, 'inlier_rate': 0.4,
       'reprojection_error_pixels': 3.0, 'spatial_coverage': 0.02,
       'valid_overlap_ratio': 0.6, 'ecc_correlation': 0.2,
       'candidate_in_frame_ratio': 0.6, 'roi_valid_overlap_ratio': 0.6,
       'effective_resolution_scale': 0.25, 'effective_live_width_pixels': 24,
       'effective_live_height_pixels': 24, 'appearance_ncc': 0.35, 'appearance_ssim': 0.2}
for m in MIN_METRICS + MAX_METRICS:
    a = confA.get(m); b = confB.get(m); c = cur.get(m)
    fa = f'{a:.4g}' if a is not None else '-'
    fb = f'{b:.4g}' if b is not None else '-'
    fc = f'{c:.4g}' if c is not None else '-'
    print(f'{m:34} {fc:>12} {fa:>12} {fb:>12}')

print('\n== per-type Policy-C thresholds for appearance_ncc ==')
for g in sorted(confC):
    v = confC[g].get('appearance_ncc')
    print(f'  {g:12} n={len(groups[g]):4}  ncc_min={v:.4f}' if v is not None else f'  {g}: -')

# ---- availability by case type under each policy ----
print('\n== simulated usable rate by case type ==')
def simulate_by_type(config):
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        detail = r['component_mapping_failure_detail'] or ''
        structural = any(s in detail for s in (
            'shape differs', 'mask is unavailable', 'no finite standard-to-live',
            'could not be projected', 'invalid geometry', 'outside the live-image frame',
            'non-finite local sampling', 'could not be measured'))
        ok = not structural
        if ok:
            for m in ('candidate_in_frame_ratio', 'roi_valid_overlap_ratio',
                      'effective_resolution_scale', 'effective_live_width_pixels',
                      'effective_live_height_pixels', 'appearance_ncc', 'appearance_ssim'):
                v = num(r.get(m))
                t = th_for(config, r, m) if '_per_group' in config else config.get(m)
                if v is None:
                    if m in ('appearance_ncc', 'appearance_ssim'):
                        ok = False; break
                    continue
                if t is not None and v < t:
                    ok = False; break
        out[r['_group']][1] += 1
        if ok: out[r['_group']][0] += 1
    return out

sa, sb, sc = simulate_by_type(confA), simulate_by_type(confB), simulate_by_type(confC_wrap)
print(f'{"type":12} {"n":>4} {"A strict":>10} {"B loose":>10} {"C per-type":>10}')
for g in sorted(sa, key=lambda k: -sa[k][1]):
    u, t = sa[g]; ub, _ = sb[g]; uc, _ = sc[g]
    print(f'{g:12} {t:>4} {u/t*100:>9.1f}% {ub/t*100:>9.1f}% {uc/t*100:>9.1f}%')
