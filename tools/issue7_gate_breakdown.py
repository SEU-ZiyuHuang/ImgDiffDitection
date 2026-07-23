# -*- coding: utf-8 -*-
"""Per-gate breakdown of mapping blocks in the 135830 batch."""
import csv, io, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
PATH = r'D:\异动检测批量输出\all_test_calibration_parallel_20260722_135830\analysis\component_observations.csv'
rows = list(csv.DictReader(io.open(PATH, encoding='utf-8')))

def gate(detail):
    if not detail:
        return '(compared - 通过)'
    if 'shape differs' in detail: return 'A0 对齐后形状不一致'
    if 'mask is unavailable' in detail: return 'A0 无有效比较mask'
    if 'no finite standard-to-live' in detail: return 'B1 无有效标准→实时变换'
    if 'could not be projected' in detail: return 'B1 ROI无法投影到实时图'
    if 'invalid geometry' in detail: return 'B1 投影几何非法'
    if 'outside the live-image frame' in detail: return 'B2 候选完全落在实时图外'
    if 'in-frame coverage' in detail: return 'B3 候选在框内比例 < 0.6'
    if 'ROI valid overlap' in detail: return 'B4 ROI内有效像素重叠 < 0.6'
    if 'non-finite local sampling' in detail: return 'C1 局部采样几何非法'
    if 'too coarse' in detail: return 'C2 实时图有效分辨率不足'
    if 'could not be measured' in detail: return 'D0 表观一致性无法测量'
    if 'NCC below' in detail: return 'D1 配准后表观 NCC < 0.35'
    if 'SSIM below' in detail: return 'D2 配准后表观 SSIM < 0.2'
    return 'OTHER: ' + detail[:50]

c = Counter(gate(r['component_mapping_failure_detail']) for r in rows)
blocked = sum(v for k, v in c.items() if not k.startswith('(compared'))
print(f'771 个部件：通过 {c["(compared - 通过)"]}，被拦 {blocked}')
print('== 各门禁条件触发分布 ==')
for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
    print(f'  {v:4} ({v/771*100:4.1f}%)  {k}')

# NCC 被拦部件的分桶（供救援实验分层抽样）
def num(x):
    try: return float(x)
    except Exception: return float('nan')
ncc_blocked = [r for r in rows if 'NCC below' in (r['component_mapping_failure_detail'] or '')]
print()
print(f'== D1 NCC 被拦 {len(ncc_blocked)} 个部件的 NCC 分桶 ==')
buckets = Counter()
for r in ncc_blocked:
    v = num(r['appearance_ncc'])
    if v != v: buckets['nan'] += 1
    elif v < 0: buckets['<0'] += 1
    elif v < 0.15: buckets['0~0.15'] += 1
    elif v < 0.25: buckets['0.15~0.25'] += 1
    else: buckets['0.25~0.35'] += 1
for k, v in sorted(buckets.items()):
    print(f'  {v:4}  NCC {k}')
