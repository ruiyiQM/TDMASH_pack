"""Measure accepted CN/PT-CN settings in four same-core timing pairs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

import euler_c2h4_cn_reference as benchmark

DURATION = 10.


def select(summary):
    if not summary['cn_reference_validated']:
        raise ValueError('Timing requires a validated short CN reference.')
    selected = {}
    for method in ('accelerated', 'pt_cn'):
        rows = [r for r in summary['records'] if r['role'] == 'candidate' and r['method'] == method and r['passed']]
        if not rows:
            raise ValueError('Both methods need an accepted candidate before timing.')
        row = max(rows, key=lambda r: r['dt'])
        selected[method] = {key: row[key] for key in ('backend', 'method', 'alpha', 'dt', 'electronic_dt', 'tolerance')}
    return selected


def run(root):
    benchmark.study.verify(root)
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Paired timing requires exactly one allocated CPU.')
    source = json.loads((root / 'protocol.json').read_text())
    parent = Path(source['timing_source_root'])
    summary_path = parent / 'compact_summary.json'
    selected = select(json.loads(summary_path.read_text()))
    duration = DURATION
    if any(not np.isclose(duration / row['dt'], round(duration / row['dt']), rtol=0, atol=1e-12) for row in selected.values()):
        raise ValueError('The timing duration must cover complete nuclear steps.')
    warmups = []
    for method, case in selected.items():
        benchmark.study.SAMPLE_INTERVAL = case['dt']
        result, _ = benchmark.study.integrate(root, case, 2 * case['dt'])
        if not result['completed']:
            raise RuntimeError('Timing warmup failed: ' + str(result['failure']))
        warmups.append(dict(result, method=method))
    records = []
    for repeat, order in enumerate((('accelerated', 'pt_cn'), ('pt_cn', 'accelerated'),
                                   ('pt_cn', 'accelerated'), ('accelerated', 'pt_cn'))):
        for rank, method in enumerate(order):
            case = selected[method]
            benchmark.study.SAMPLE_INTERVAL = case['dt']
            record, _ = benchmark.study.integrate(root, case, duration)
            records.append(dict(record, repeat=repeat, rank=rank, method=method))
            print(json.dumps(records[-1]), flush=True)
    ratios = []
    for repeat in range(4):
        pair = {r['method']: r for r in records if r['repeat'] == repeat}
        if all(r['completed'] for r in pair.values()):
            ratios.append(pair['accelerated']['propagation_seconds'] / pair['pt_cn']['propagation_seconds'])
    result = dict(host=platform.node(), affinity=sorted(os.sched_getaffinity(0)), job_id=os.environ.get('SLURM_JOB_ID'),
                  duration_au=duration, duration_fs=duration * benchmark.study.AU_FS, selected=selected,
                  short_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  warmups=warmups, records=records, cn_over_pt_pairs=ratios,
                  cn_over_pt_median=float(np.median(ratios)) if ratios else None,
                  cn_over_pt_range=[min(ratios), max(ratios)] if ratios else None,
                  scope='Same-core short-window propagation with common outputs; setup reported separately; diagnostic SCFs and archive writes excluded. Long accuracy is not inferred.')
    with (root / 'timing_summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)


def self_check():
    assert all(np.isclose(DURATION / h, round(DURATION / h), rtol=0, atol=1e-10)
               for h in (.03125, .0390625, .05))
    rows = [dict(role='candidate', method=m, passed=True, backend='uks', alpha=1., dt=h, electronic_dt=h, tolerance=1e-9)
            for m in ('accelerated', 'pt_cn') for h in (.015625, .03125)]
    selected = select(dict(cn_reference_validated=True, records=rows))
    assert all(row['dt'] == .03125 for row in selected.values())
    rows[-1]['passed'] = False
    assert select(dict(cn_reference_validated=True, records=rows))['pt_cn']['dt'] == .015625
    try:
        select(dict(cn_reference_validated=False, records=rows))
    except ValueError:
        pass
    else:
        raise AssertionError('Unvalidated references must not produce speedup claims.')
    print('Timing selection requires accepted candidates and a validated CN reference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    else:
        run(args.root)
