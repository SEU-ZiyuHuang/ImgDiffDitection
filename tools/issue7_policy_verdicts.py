# -*- coding: utf-8 -*-
"""Per-sample verdict simulation under policies A/B/C/D with the business rule:
'a live image that clearly mismatches the reference IS an anomaly'.

Verdict classes per component:
  无异动          = passed all gates AND compared AND zero candidates
  异动-部件差异   = passed all gates AND compared AND candidates > 0
  异动-图像不匹配 = failed NCC/SSIM gate, or candidate largely out of frame
                    (confident mismatch per the new business rule)
  检测不可用      = structural failures (no transform/mask) + too-coarse
                    resolution + unmeasurable appearance
  比较未运行      = would pass gates under a looser policy but difference
                    stage never ran for it in the dev batch
"""
import csv, io, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\异动检测批量输出\all_test_calibration_parallel_20260722_135830\analysis\\'
rows = list(csv.DictReader(io.open(BASE + 'component_observations.csv', encoding='utf-8')))
rows = [r for r in rows if r['case_type'] != '鸟害']

def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None

def group_of(ct):
    if ct.startswith('日常'): return '日常'
    if ct in ('表计', '表记'): return '表计'
    if ct.startswith('位置'): return '位置'
    if ct in ('', None): return 'unspecified'
    return ct

for r in rows:
    r['_g'] = group_of(r['case_type'])

POLICIES = {
    'A_strict': {'candidate_in_frame_ratio': 0.9857, 'roi_valid_overlap_ratio': 0.9895,
                 'effective_resolution_scale': 0.9211, 'effective_live_width_pixels': 68.96,
                 'effective_live_height_pixels': 69.08, 'appearance_ncc': -0.1754,
                 'appearance_ssim': 0.1171},
    'B_loose': {'candidate_in_frame_ratio': 0.7222, 'roi_valid_overlap_ratio': 0.8787,
                'effective_resolution_scale': 0.6646, 'effective_live_width_pixels': 46.06,
                'effective_live_height_pixels': 50.92, 'appearance_ncc': -0.3626,
                'appearance_ssim': 0.03764},
    'C_pertype': None,  # same as A for geometry except ncc/ssim per type; approximated below
    'D_recommend': {'candidate_in_frame_ratio': 0.7, 'roi_valid_overlap_ratio': 0.85,
                    'effective_resolution_scale': 0.4, 'effective_live_width_pixels': 48,
                    'effective_live_height_pixels': 48, 'appearance_ncc': 0.35,
                    'appearance_ssim': 0.2},
}
# Policy C: like A but ncc/ssim per-type (unspecified: -0.2958, others global A: -0.1754)
C_NCC = {'unspecified': -0.2958}
C_A = POLICIES['A_strict']

def verdict(r, th, policy):
    detail = r['component_mapping_failure_detail'] or ''
    structural = any(s in detail for s in (
        'shape differs', 'mask is unavailable', 'no finite standard-to-live',
        'could not be projected', 'invalid geometry', 'outside the live-image frame',
        'non-finite local sampling'))
    if structural:
        return '检测不可用'
    if 'could not be measured' in detail:
        return '检测不可用'
    # gate order mirrors mapping.py: in-frame -> roi overlap -> resolution -> appearance
    v = num(r.get('candidate_in_frame_ratio'))
    if v is not None and v < th['candidate_in_frame_ratio']:
        return '异动-图像不匹配'          # expected component mostly outside live view
    v = num(r.get('roi_valid_overlap_ratio'))
    if v is not None and v < th['roi_valid_overlap_ratio']:
        return '检测不可用'                # too few valid pixels to judge
    for m in ('effective_resolution_scale', 'effective_live_width_pixels',
              'effective_live_height_pixels'):
        v = num(r.get(m))
        if v is not None and v < th[m]:
            return '检测不可用'            # too coarse to judge
    ncc_min = th['appearance_ncc']
    ssim_min = th['appearance_ssim']
    if policy == 'C_pertype':
        ncc_min = C_NCC.get(r['_g'], C_A['appearance_ncc'])
    ncc = num(r.get('appearance_ncc'))
    ssim = num(r.get('appearance_ssim'))
    if ncc is None or ssim is None:
        return '检测不可用'
    if ncc < ncc_min or ssim < ssim_min:
        return '异动-图像不匹配'          # confident mismatch (new business rule)
    # passed all gates -> comparison stage
    dev_compared = r.get('component_mapping_usable') == '1'
    if not dev_compared:
        return '比较未运行'                # needs re-run under this policy
    cands = num(r.get('difference_candidate_count')) or 0
    return '异动-部件差异' if cands > 0 else '无异动'

out_cols = ['case', 'component_index', 'case_type', 'category']
verdicts = {}
for pname, th in POLICIES.items():
    counts = defaultdict(int)
    by_type = defaultdict(lambda: defaultdict(int))
    for r in rows:
        v = verdict(r, th or C_A, pname)
        r['v_' + pname] = v
        counts[v] += 1
        by_type[r['_g']][v] += 1
    verdicts[pname] = (counts, by_type)

n = len(rows)
print(f'components: {n} (771 - 2 鸟害)\n')
order = ['无异动', '异动-部件差异', '异动-图像不匹配', '检测不可用', '比较未运行']
for pname in POLICIES:
    counts, by_type = verdicts[pname]
    print(f'== {pname} ==')
    for k in order:
        c = counts.get(k, 0)
        print(f'   {k:12} {c:4} ({c/n*100:5.1f}%)')
    print()

print('== by case type (D_recommend) ==', )
counts, by_type = verdicts['D_recommend']
print(f'{"type":12} {"n":>4}' + ''.join(f'{k:>14}' for k in order))
for g in sorted(by_type, key=lambda k: -sum(by_type[k].values())):
    tot = sum(by_type[g].values())
    print(f'{g:12} {tot:>4}' + ''.join(f'{by_type[g].get(k,0):>14}' for k in order))

# per-sample CSV with all four verdicts
out_csv = r'D:\异动检测批量输出\policy_verdicts_ABCD_20260722.csv'
with io.open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case', 'component_index', 'case_type', 'category',
                'appearance_ncc', 'appearance_ssim', 'spatial_coverage', 'ecc_converged',
                'difference_candidate_count',
                'A_strict', 'B_loose', 'C_pertype', 'D_recommend'])
    for r in rows:
        w.writerow([r['case'], r['component_index'], r['case_type'], r['category'],
                    r['appearance_ncc'], r['appearance_ssim'], r['spatial_coverage'],
                    r['ecc_converged'], r['difference_candidate_count'],
                    r['v_A_strict'], r['v_B_loose'], r['v_C_pertype'], r['v_D_recommend']])
print(f'\nper-sample verdicts written: {out_csv}')

# image-level aggregation for D
img = defaultdict(lambda: defaultdict(int))
for r in rows:
    img[r['case']][r['v_D_recommend']] += 1
img_counts = defaultdict(int)
for case, cc in img.items():
    if cc.get('异动-部件差异') or cc.get('异动-图像不匹配'):
        img_counts['异动'] += 1
    elif cc.get('检测不可用'):
        img_counts['检测不可用'] += 1
    elif cc.get('比较未运行'):
        img_counts['比较未运行'] += 1
    else:
        img_counts['无异动'] += 1
print('\n== image-level conclusion (D_recommend, cases) ==')
for k in ['无异动', '异动', '检测不可用', '比较未运行']:
    print(f'   {k:10} {img_counts[k]:4} / {len(img)}')
