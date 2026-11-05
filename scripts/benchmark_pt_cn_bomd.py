"""Measure paired PT-CN/BOMD timings using the frozen 100 fs Euler study."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from pyscf import lib

import euler_pt_cn_bomd as study
from refine_pt_cn_bomd_reference import verify_parent

CELLS = [(backend, dt) for backend in ('rhf', 'rks') for dt in (5., 10., 20.)]
ORDERS = [('bomd', 'pt_cn'), ('pt_cn', 'bomd'), ('pt_cn', 'bomd'), ('bomd', 'pt_cn')]
TOLERANCES = dict(bomd=1e-9, pt_cn=1e-6)
KEYS = ('time', 'positions', 'velocities', 'forces', 'dipoles', 'energy')


def prepare(root):
    verify_parent(root)
    output = root / 'timing'
    output.mkdir(exist_ok=False)
    protocol = dict(cells=CELLS, orders=ORDERS, tolerances=TOLERANCES,
                    duration_au=study.DURATION, duration_fs=study.DURATION * study.AU_FS,
                    timing='Fresh initial SCF plus driver construction and propagation; imports, warmups, validation and archive writes excluded',
                    propagation='Includes driver construction, forces, electronic solution, endpoint energies/dipoles, common in-memory trajectory arrays',
                    diagnostics='No AO-density history or XYZ/energy file writes for either method',
                    pairing='Four fresh pairs per cell, AB/BA/BA/AB, on one allocated physical CPU; two-step warmup for each method',
                    outputs=list(KEYS) + ['total_dipoles'],
                    accuracy='Same dt, initial state, masses and backend; errors versus validated BOMD dt0.625 are reported, no new acceptance budget',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    parent_summary_sha256=hashlib.sha256((root / 'large_steps/summary.json').read_bytes()).hexdigest())
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))


def timed_run(backend, method, dt, duration):
    started = time.perf_counter()
    mf, masses, velocity = study.setup(backend)
    initialized = time.perf_counter()
    diagnostics = {}
    if method == 'bomd':
        result, trajectory = study.native_bomd(mf, masses, velocity, dt, duration, TOLERANCES[method])
        diagnostics['scf_iterations'] = result['scf_iterations']
    else:
        simulation = study.TDRHFSimulation if backend == 'rhf' else study.TDRKSSimulation
        sim = simulation(mf, dt=dt, nuclei_dt=dt, steps=round(duration / dt), ehrenfest=True,
                         propagation_method='pt_cn', convergence_threshold=TOLERANCES[method],
                         max_iterations=200, initial_velocity=velocity.copy(),
                         output_interval=0, output_filename=os.devnull, energy_filename=None)
        sim.mass = masses.copy()
        iterations, residuals = [], []
        propagate = sim.propagate_electrons

        def measured_propagate(*args, **kwargs):
            propagate(*args, **kwargs)
            if kwargs.get('dt') != 0:
                iterations.append(sim.pt_cn_iterations)
                residuals.append(sim.pt_cn_residual)

        sim.propagate_electrons = measured_propagate
        sim.run_ehrenfest_midpoint_decoupled()
        trajectory = dict(time=np.asarray(sim.times), positions=np.asarray(sim.positions_history),
                          velocities=np.asarray(sim.velocities_history), forces=np.asarray(sim.forces_history),
                          dipoles=np.asarray(sim.dipoles), energy=np.asarray(sim.total_energies))
        diagnostics.update(pt_iterations=sum(iterations), max_pt_iterations=max(iterations),
                           max_pt_residual=max(residuals))
    trajectory['total_dipoles'] = np.einsum('a,tac->tc', mf.mol.atom_charges(), trajectory['positions']) - trajectory['dipoles']
    finished = time.perf_counter()
    return dict(backend=backend, method=method, dt=dt, tolerance=TOLERANCES[method],
                setup_seconds=initialized - started, propagation_seconds=finished - initialized,
                total_seconds=finished - started, **diagnostics), trajectory


def load_checked(folder, expected=None):
    raw = folder / 'trajectory.npz'
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if expected is None:
        expected = json.loads((folder / 'result.json').read_text())['trajectory_sha256']
    if digest != expected:
        raise RuntimeError(f'Trajectory checksum mismatch: {folder}')
    with np.load(raw) as archive:
        return dict(archive)


def check_reproduction(trajectory, previous):
    for key in KEYS:
        tolerance = 1e-10 if key == 'velocities' else 1e-8
        np.testing.assert_allclose(trajectory[key], previous[key], rtol=0, atol=tolerance,
                                   err_msg=f'Timing run changed the validated {key} trajectory')


def run(root, number):
    verify_parent(root)
    protocol = json.loads((root / 'timing/protocol.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol['source_sha256']:
        raise RuntimeError('Timing script changed after protocol preparation.')
    parent = root / 'large_steps/summary.json'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != protocol['parent_summary_sha256']:
        raise RuntimeError('Parent summary changed.')
    summary = json.loads(parent.read_text())
    if not all(r['validated'] for r in summary['references'].values()):
        raise RuntimeError('Reference accuracy checks must pass first.')
    backend, dt = CELLS[number]
    output = root / 'timing' / f'cell_{number:02d}'
    output.mkdir(exist_ok=False)
    reference = load_checked(root / 'reference_refinement' / backend)
    previous = {}
    for method in TOLERANCES:
        row = next(r for r in summary['records'] if r['backend'] == backend and r['dt'] == dt
                   and r['method'] == method and r['tolerance'] == TOLERANCES[method])
        base = root / 'large_steps' if row.get('study_group') == 'large_steps' else root
        index = row.get('extension_case_id', row['case_id'])
        previous[method] = load_checked(base / 'runs' / f'case_{index:03d}', row['trajectory_sha256'])
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if affinity is None or len(affinity) != 1:
        raise RuntimeError('Timing requires exactly one allocated CPU.')
    metadata = dict(host=platform.node(), affinity=affinity, job_id=os.environ.get('SLURM_JOB_ID'),
                    python=platform.python_version(), numpy=np.__version__, scipy=study.scipy.__version__,
                    pyscf=__import__('pyscf').__version__, cell=number,
                    cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                   if line.startswith('model name')))
    for method in TOLERANCES:
        _, warmup = timed_run(backend, method, dt, dt * 2)
        study.check_trajectory(warmup, dt, dt * 2)
    for repeat, order in enumerate(ORDERS):
        for rank, method in enumerate(order):
            folder = output / f'{repeat}_{method}'
            folder.mkdir(exist_ok=False)
            try:
                record, trajectory = timed_run(backend, method, dt, study.DURATION)
                np.savez_compressed(folder / 'trajectory.npz', **trajectory)
                study.check_trajectory(trajectory, dt, study.DURATION)
                check_reproduction(trajectory, previous[method])
                record['observables'] = study.metrics(trajectory, reference)
                record.update(metadata, repeat=repeat, rank=rank, completed=True,
                              trajectory_sha256=hashlib.sha256((folder / 'trajectory.npz').read_bytes()).hexdigest())
            except Exception as error:
                (folder / 'result.json').write_text(json.dumps(dict(metadata, completed=False, repeat=repeat,
                                                                   method=method, failure=f'{type(error).__name__}: {error}'), indent=2))
                raise
            (folder / 'result.json').write_text(json.dumps(record, indent=2))
            print(backend, dt, repeat, method, f"{record['total_seconds']:.3f}s", flush=True)


def paired_stats(records, field):
    ratios = []
    for repeat in range(len(ORDERS)):
        pair = {r['method']: r for r in records if r['repeat'] == repeat}
        if set(pair) != set(TOLERANCES) or len([r for r in records if r['repeat'] == repeat]) != 2:
            raise ValueError('Each repeat must contain exactly one complete BOMD/PT-CN pair.')
        if any(not r['completed'] or not np.isfinite(r[field]) or r[field] <= 0 for r in pair.values()):
            raise ValueError('Only complete positive timings can be compared.')
        ratios.append(pair['bomd'][field] / pair['pt_cn'][field])
    return dict(paired_bomd_over_pt=ratios, median=float(np.median(ratios)),
                minimum=min(ratios), maximum=max(ratios),
                bomd_first_median=float(np.median([ratios[i] for i, order in enumerate(ORDERS) if order[0] == 'bomd'])),
                pt_first_median=float(np.median([ratios[i] for i, order in enumerate(ORDERS) if order[0] == 'pt_cn'])))


def analyze(root):
    verify_parent(root)
    protocol = json.loads((root / 'timing/protocol.json').read_text())
    records, cells = [], []
    for number, (backend, dt) in enumerate(CELLS):
        cell = []
        for repeat, order in enumerate(ORDERS):
            for rank, method in enumerate(order):
                folder = root / 'timing' / f'cell_{number:02d}' / f'{repeat}_{method}'
                row = json.loads((folder / 'result.json').read_text())
                if not row['completed'] or (row['backend'], row['dt'], row['repeat'], row['rank'], row['method']) != (backend, dt, repeat, rank, method):
                    raise ValueError(f'Incomplete or mismatched timing record: {folder}')
                if len(row['affinity']) != 1:
                    raise ValueError('Incorrect CPU affinity.')
                study.check_trajectory(load_checked(folder, row['trajectory_sha256']), dt, study.DURATION)
                cell.append(row)
        if len({(r['host'], tuple(r['affinity']), r['job_id']) for r in cell}) != 1:
            raise ValueError('A timing cell must run sequentially in one allocation on one CPU.')
        records.extend(cell)
        result = dict(backend=backend, dt=dt, total=paired_stats(cell, 'total_seconds'),
                      propagation=paired_stats(cell, 'propagation_seconds'), methods={})
        for method in TOLERANCES:
            rows = [r for r in cell if r['method'] == method]
            result['methods'][method] = dict(
                median_seconds={key: float(np.median([r[key] for r in rows])) for key in
                                ('setup_seconds', 'propagation_seconds', 'total_seconds')},
                seconds_per_fs=float(np.median([r['total_seconds'] for r in rows]) / (study.DURATION * study.AU_FS)),
                observables=rows[0]['observables'])
        cells.append(result)
    with (root / 'timing/summary.json').open('x') as stream:
        json.dump(dict(protocol=protocol, cells=cells, records=records), stream, indent=2)
    print(json.dumps(cells, indent=2))


def self_check():
    synthetic = [dict(repeat=i, method=m, completed=True, seconds=4 if m == 'bomd' else 2)
                 for i, order in enumerate(ORDERS) for m in order]
    assert paired_stats(synthetic, 'seconds')['median'] == 2
    try:
        paired_stats(synthetic[:-1], 'seconds')
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete timing pairs must be rejected.')
    for backend in ('rhf', 'rks'):
        trajectories = []
        for method in TOLERANCES:
            record, trajectory = timed_run(backend, method, 1., 2.)
            study.check_trajectory(trajectory, 1., 2.)
            assert set(trajectory) == set(KEYS) | {'total_dipoles'}
            assert record['total_seconds'] > record['propagation_seconds'] > 0
            if method == 'pt_cn':
                assert record['max_pt_residual'] <= TOLERANCES[method]
            trajectories.append(trajectory)
        np.testing.assert_allclose(trajectories[0]['positions'][0], trajectories[1]['positions'][0], atol=1e-14)
        np.testing.assert_allclose(trajectories[0]['velocities'][0], trajectories[1]['velocities'][0], atol=1e-14)
        check_reproduction(trajectories[0], trajectories[0])
        changed = dict(trajectories[0], positions=trajectories[0]['positions'] + 1e-5)
        try:
            check_reproduction(changed, trajectories[0])
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed trajectories must be rejected.')
    print('Paired timing and common-output checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze', 'self-check'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cell', type=int, choices=range(len(CELLS)))
    args = parser.parse_args()
    lib.num_threads(1)
    if args.action == 'self-check':
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    elif args.action == 'run':
        if args.cell is None:
            parser.error('--cell is required for run')
        run(args.root, args.cell)
    else:
        globals()[args.action](args.root)


if __name__ == '__main__':
    main()
