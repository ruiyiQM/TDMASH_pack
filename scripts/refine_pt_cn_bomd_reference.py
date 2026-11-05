"""Refine only the BOMD references and reuse the completed 100 fs trajectories."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from pyscf import lib

import euler_pt_cn_bomd as study

REFERENCE_DT = .625


def check_grid():
    positions = np.repeat(np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]), 5, axis=0)
    reference = dict(time=np.arange(5) * REFERENCE_DT, positions=positions,
                     forces=np.zeros_like(positions), energy=np.zeros(5))
    coarse = {key: value[::2] for key, value in reference.items()}
    assert study.metrics(coarse, reference)['oh_rms_angstrom'] == 0


def verify_parent(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if hashlib.sha256((root / 'code' / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen study source changed: {name}')
    original = ast.parse((root / 'code/scripts/euler_pt_cn_bomd.py').read_text())
    current = ast.parse(Path(study.__file__).read_text())
    for name in ('setup', 'native_bomd', 'geometry_observables'):
        old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(old) != ast.dump(new):
            raise RuntimeError(f'Physical benchmark implementation changed: {name}')


def run(root, number):
    verify_parent(root)
    check_grid()
    backend = ('rhf', 'rks')[number]
    output = root / 'reference_refinement' / backend
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    mf, masses, velocity = study.setup(backend)
    record, trajectory = study.native_bomd(mf, masses, velocity, REFERENCE_DT, study.DURATION, 1e-11)
    study.check_trajectory(trajectory, REFERENCE_DT, study.DURATION)
    np.savez_compressed(output / 'trajectory.npz', **trajectory)
    record.update(backend=backend, dt=REFERENCE_DT, tolerance=1e-11, duration_au=study.DURATION,
                  total_seconds=time.perf_counter() - started, host=platform.node(),
                  job_id=os.environ.get('SLURM_JOB_ID'),
                  trajectory_sha256=hashlib.sha256((output / 'trajectory.npz').read_bytes()).hexdigest())
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def analyze(root):
    verify_parent(root)
    parent = root / 'summary.json'
    summary = json.loads(parent.read_text())
    summary['parent_summary_sha256'] = hashlib.sha256(parent.read_bytes()).hexdigest()
    summary['original_reference_checks'] = summary.pop('references')
    summary['references'], summary['reference_refinement'] = {}, []
    summary['observables_reference_dt'] = REFERENCE_DT
    for backend in ('rhf', 'rks'):
        folder = root / 'reference_refinement' / backend
        record = json.loads((folder / 'result.json').read_text())
        raw = folder / 'trajectory.npz'
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record['trajectory_sha256']:
            raise RuntimeError('Refined reference checksum mismatch.')
        reference = dict(np.load(raw))
        study.check_trajectory(reference, REFERENCE_DT, study.DURATION)
        summary['reference_refinement'].append(record)
        previous = next(r for r in summary['records'] if r['backend'] == backend and r['reference'] and r['dt'] == 1.25)
        previous_data = dict(np.load(root / 'runs' / f"case_{previous['case_id']:03d}" / 'trajectory.npz'))
        convergence = study.metrics(previous_data, reference)
        summary['references'][backend] = dict(validated=convergence['oh_rms_angstrom'] < 1e-4,
                                               compared_dt=[1.25, REFERENCE_DT], metrics=convergence)
        for candidate in summary['records']:
            if candidate['backend'] != backend or not candidate['completed']:
                continue
            raw = root / 'runs' / f"case_{candidate['case_id']:03d}" / 'trajectory.npz'
            if hashlib.sha256(raw.read_bytes()).hexdigest() != candidate['trajectory_sha256']:
                raise RuntimeError('Original candidate checksum mismatch.')
            trajectory = dict(np.load(raw))
            study.check_trajectory(trajectory, candidate['dt'], study.DURATION)
            candidate['observables'] = study.metrics(trajectory, reference)
    summary['postprocess_sha256'] = {
        Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (__file__, study.__file__)}
    output = root / 'reference_refinement'
    with (output / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    lines = ['# 101.59 fs comparison with refined BOMD references', '',
             'Original 42 trajectories are reused unchanged. References: dt 0.625 au, checked against 1.25 au.', '',
             '| Backend | Method | dt / au | Tolerance | Energy maximum / meV per atom | OH RMS / Angstrom | Angle RMS / degree | Position RMS / Angstrom |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['records']:
        m = row.get('observables')
        if m:
            lines.append(f"| {row['backend']} | {row['method']} | {row['dt']} | {row['tolerance']:.0e} | "
                         f"{m['energy_max_mev_per_atom']:.6g} | {m['oh_rms_angstrom']:.6g} | "
                         f"{m['angle_rms_degree']:.6g} | {m['position_rms_angstrom']:.6g} |")
    lines += ['', 'Reference checks:', '', '```json', json.dumps(summary['references'], indent=2), '```', '',
              'Single cross-node timings do not establish a speedup. PT-Ehrenfest and BOMD need not describe identical electronic responses.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary['references'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'analyze'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', type=int, choices=(0, 1))
    args = parser.parse_args()
    lib.num_threads(1)
    if args.action == 'run':
        if args.case is None:
            parser.error('--case is required for run')
        run(args.root, args.case)
    else:
        analyze(args.root)


if __name__ == '__main__':
    main()
