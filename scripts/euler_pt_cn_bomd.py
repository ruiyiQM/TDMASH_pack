"""Run and summarize a 100 fs PT-CN versus native PySCF Velocity Verlet BOMD study."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import scipy
from pyscf import data, dft, gto, lib, md, scf
from pyscf.lib import param
from scipy.signal import find_peaks

from tdmash import TDRHFSimulation, TDRKSSimulation
from validate_pt_cn_timestep import run_case

DURATION = 4200.0
AU_FS = 0.024188843265857
EV = data.nist.HARTREE2EV
GEOMETRY = 'O 0 0 0; H 1.4 0 0.9; H -1.2 0 1.1'


def cases():
    result = []
    for backend in ('rhf', 'rks'):
        for dt in (1.25, 2.5):
            result.append(dict(backend=backend, method='bomd', dt=dt, reference=True, tolerance=1e-11))
        for dt in (5., 10., 15., 20., 25., 30., 40.):
            result.append(dict(backend=backend, method='bomd', dt=dt, reference=False, tolerance=1e-9))
        for tolerance in (1e-6, 1e-5):
            for dt in (2.5, 5., 10., 12.5, 15.):
                result.append(dict(backend=backend, method='pt_cn', dt=dt, reference=False, tolerance=tolerance))
        for method, tolerance in [('pt_cn', 1e-9), ('accelerated', 1e-8)]:
            result.append(dict(backend=backend, method=method, dt=1.25, reference=False, tolerance=tolerance))
    return [dict(case_id=i, **case) for i, case in enumerate(result)]


def setup(backend):
    mol = gto.M(atom=GEOMETRY, basis='sto-3g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol) if backend == 'rhf' else dft.RKS(mol)
    if backend == 'rks':
        mf.xc, mf.grids.level, mf.grids.prune, mf.small_rho_cutoff = 'lda,vwn', 0, None, 0
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Initial SCF did not converge.')
    masses = np.array([data.elements.COMMON_ISOTOPE_MASSES[z] * data.nist.AMU2AU
                       for z in mol.atom_charges()])
    velocity = np.array([[0., 0., 0.], [8e-4, 0., 2e-4], [-5e-4, 1e-4, -3e-4]])
    velocity -= np.sum(masses[:, None] * velocity, axis=0) / masses.sum()
    return mf, masses, velocity


def native_bomd(mf, masses, velocity, dt, duration, tolerance):
    """Use PySCF's native integrator, including its initial frame in the step count."""
    gradient = mf.nuc_grad_method()
    if hasattr(gradient, 'grid_response'):
        gradient.grid_response = True
    dynamics = md.NVE(gradient, dt=dt, steps=round(duration / dt) + 1, veloc=velocity.copy(), verbose=0)
    dynamics.scanner.base.conv_tol = tolerance
    dynamics.scanner.base.conv_tol_grad = 1e-7 if tolerance <= 1e-11 else 1e-5
    iterations = [0]
    dynamics.scanner.base.callback = lambda env: iterations.__setitem__(0, iterations[0] + 1)
    result = {key: [] for key in ('time', 'positions', 'velocities', 'forces', 'dipoles', 'energy')}

    def record(env):
        result['time'].append(dynamics.time)
        result['positions'].append(dynamics.mol.atom_coords())
        result['velocities'].append(dynamics.veloc.copy())
        result['forces'].append(dynamics.accel * masses[:, None])
        D = dynamics.scanner.base.make_rdm1()
        result['dipoles'].append(np.einsum('ij,xji->x', D.real, dynamics.mol.intor('int1e_r', comp=3)))
        result['energy'].append(dynamics.epot + dynamics.ekin)

    dynamics.callback = record
    started = time.perf_counter()
    # PySCF's MD kernel prints vectors even at verbosity zero.
    with open(os.devnull, 'w') as quiet:
        dynamics.stdout = quiet
        dynamics.kernel(dump_flags=False)
    if not np.array_equal(dynamics._masses, masses):
        raise RuntimeError('BOMD and PT-CN masses do not match.')
    result = {key: np.asarray(value) for key, value in result.items()}
    return dict(completed=True, wall_seconds=time.perf_counter() - started,
                scf_iterations=iterations[0], completed_steps=round(duration / dt)), result


def check_trajectory(trajectory, dt, duration):
    expected = np.arange(round(duration / dt) + 1) * dt
    if (len(trajectory['time']) != len(expected)
            or not np.allclose(trajectory['time'], expected, rtol=0, atol=1e-8)):
        raise ValueError('Trajectory has missing frames or an incorrect physical endpoint.')
    if any(not np.isfinite(value).all() for value in trajectory.values()):
        raise ValueError('Nonfinite trajectory data.')


def geometry_observables(positions):
    bonds = positions[:, 1:] - positions[:, :1]
    lengths = np.linalg.norm(bonds, axis=-1)
    cosine = np.einsum('ti,ti->t', bonds[:, 0], bonds[:, 1]) / np.prod(lengths, axis=1)
    return lengths * param.BOHR, np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def metrics(trajectory, reference):
    reference_dt = reference['time'][1] - reference['time'][0]
    if reference_dt <= 0 or not np.allclose(np.diff(reference['time']), reference_dt, rtol=0, atol=1e-8):
        raise ValueError('Reference times must form a positive uniform grid.')
    indices = np.rint((trajectory['time'] - reference['time'][0]) / reference_dt).astype(int)
    if (np.any(indices >= len(reference['time']))
            or not np.allclose(reference['time'][indices], trajectory['time'], rtol=0, atol=1e-8)):
        raise ValueError('Comparison requires shared physical time nodes.')
    R = trajectory['positions'] - reference['positions'][indices]
    bonds, angles = geometry_observables(trajectory['positions'])
    ref_bonds, ref_angles = geometry_observables(reference['positions'][indices])
    force_error = (trajectory['forces'] - reference['forces'][indices]) * EV / param.BOHR
    ref_forces = reference['forces'][indices] * EV / param.BOHR
    energy = (trajectory['energy'] - trajectory['energy'][0]) * EV * 1000 / 3
    result = dict(position_rms_angstrom=float(np.sqrt(np.mean(np.sum(R**2, axis=-1))) * param.BOHR),
                  position_max_angstrom=float(np.max(np.linalg.norm(R, axis=-1)) * param.BOHR),
                  oh_rms_angstrom=float(np.sqrt(np.mean((bonds - ref_bonds)**2))),
                  oh_max_angstrom=float(np.max(np.abs(bonds - ref_bonds))),
                  angle_rms_degree=float(np.sqrt(np.mean((angles - ref_angles)**2))),
                  force_rms_ev_per_angstrom=float(np.sqrt(np.mean(np.sum(force_error**2, axis=-1)))),
                  force_reference_rms_ev_per_angstrom=float(np.sqrt(np.mean(np.sum(ref_forces**2, axis=-1)))),
                  energy_max_mev_per_atom=float(np.max(np.abs(energy))),
                  energy_peak_to_peak_mev_per_atom=float(np.ptp(energy)),
                  energy_final_mev_per_atom=float(energy[-1]))
    result['energy_slope_mev_per_atom_per_ps'] = float(np.polyfit(trajectory['time'] * AU_FS / 1000, energy, 1)[0])
    periods = []
    times_fs = trajectory['time'] * AU_FS
    for column in range(2):
        distance = max(1, round(3 / (times_fs[1] - times_fs[0])))
        peaks, _ = find_peaks(bonds[:, column], distance=distance, prominence=.005)
        periods.append(float(np.median(np.diff(times_fs[peaks]))) if len(peaks) >= 3 else None)
    result['sampled_oh_peak_period_fs'] = periods
    return result


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]
    files = [*sorted((code / 'src/tdmash').glob('*.py')), *sorted((code / 'tests').glob('*.py')),
             code / 'scripts/euler_pt_cn_bomd.py', code / 'scripts/validate_pt_cn_timestep.py']
    protocol = dict(duration_au=DURATION, duration_fs=DURATION * AU_FS, geometry=GEOMETRY,
                    basis='sto-3g', rks_xc='lda,vwn', rks_grids='level 0, no pruning, full response',
                    masses='PySCF common isotope masses, identical for both integrators',
                    initial_velocity='As in setup(); initial COM velocity removed once for all methods',
                    model='Field-free, mu=1, closed-shell ground-state initial condition, no thermostat',
                    purpose='Observable error orders versus Velocity Verlet BOMD, without the previous pointwise electronic acceptance gate',
                    reference='BOMD dt1.25 versus dt2.5; require OH RMS difference <1e-4 Angstrom before interpreting candidate errors',
                    timing='Single allocated core; initialization and propagation reported separately; cross-node wall times are descriptive',
                    source_sha256={str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    for name, value in [('cases.json', cases()), ('protocol.json', protocol)]:
        with (root / name).open('x') as stream:
            json.dump(value, stream, indent=2)


def run(root, number):
    case = json.loads((root / 'cases.json').read_text())[number]
    protocol = json.loads((root / 'protocol.json').read_text())
    code = Path(__file__).resolve().parents[1]
    for name, digest in protocol['source_sha256'].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source hash mismatch: {name}')
    output = root / 'runs' / f'case_{number:03d}'
    output.mkdir(parents=True, exist_ok=False)
    record = dict(case, host=platform.node(), platform=platform.platform(),
                  job_id=os.environ.get('SLURM_JOB_ID'), array_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  pyscf=__import__('pyscf').__version__)
    if Path('/proc/cpuinfo').exists():
        record['cpu_model'] = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                   if line.startswith('model name'))
    started = time.perf_counter()
    try:
        mf, masses, velocity = setup(case['backend'])
        record['initial_scf_seconds'] = time.perf_counter() - started
        if case['method'] == 'bomd':
            result, trajectory = native_bomd(mf, masses, velocity, case['dt'], DURATION, case['tolerance'])
        else:
            simulation = TDRHFSimulation if case['backend'] == 'rhf' else TDRKSSimulation

            def factory(*args, **kwargs):
                kwargs.update(convergence_threshold=case['tolerance'], max_iterations=200,
                              initial_velocity=velocity.copy())
                sim = simulation(*args, **kwargs)
                sim.mass = masses.copy()
                return sim

            result, trajectory = run_case(mf, factory, case['method'], case['dt'], DURATION, output, 'trajectory')
        record.update(result)
        if record['completed']:
            check_trajectory(trajectory, case['dt'], DURATION)
            trajectory['total_dipoles'] = np.einsum('a,tac->tc', mf.mol.atom_charges(), trajectory['positions']) - trajectory['dipoles']
            np.savez_compressed(output / 'trajectory.npz', **trajectory)
            record['trajectory_sha256'] = hashlib.sha256((output / 'trajectory.npz').read_bytes()).hexdigest()
            record['final_time_au'] = float(trajectory['time'][-1])
    except Exception as error:
        record.update(completed=False, failure=f'{type(error).__name__}: {error}')
    record['total_seconds'] = time.perf_counter() - started
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    if not record['completed']:
        raise RuntimeError(record.get('failure', 'Incomplete propagation.'))


def analyze(root):
    manifest = json.loads((root / 'cases.json').read_text())
    records, datasets = [], {}
    for case in manifest:
        path = root / 'runs' / f"case_{case['case_id']:03d}"
        if not (path / 'result.json').exists():
            records.append(dict(case, completed=False, failure='Missing result; inspect Slurm state.'))
            continue
        record = json.loads((path / 'result.json').read_text())
        if record['completed']:
            raw = path / 'trajectory.npz'
            if hashlib.sha256(raw.read_bytes()).hexdigest() != record['trajectory_sha256']:
                raise RuntimeError('Trajectory hash mismatch.')
            datasets[case['case_id']] = dict(np.load(raw))
            check_trajectory(datasets[case['case_id']], case['dt'], DURATION)
        records.append(record)
    summary = dict(protocol=json.loads((root / 'protocol.json').read_text()), records=records, references={})
    for backend in ('rhf', 'rks'):
        references = [r for r in records if r['backend'] == backend and r['reference']]
        if not all(r['completed'] for r in references):
            summary['references'][backend] = dict(validated=False, reason='Incomplete BOMD references')
            continue
        fine_record = next(r for r in references if r['dt'] == 1.25)
        coarse_record = next(r for r in references if r['dt'] == 2.5)
        fine = datasets[fine_record['case_id']]
        reference_metrics = metrics(datasets[coarse_record['case_id']], fine)
        summary['references'][backend] = dict(validated=reference_metrics['oh_rms_angstrom'] < 1e-4, metrics=reference_metrics)
        for record in records:
            if record['backend'] == backend and record['completed']:
                record['observables'] = metrics(datasets[record['case_id']], fine)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# 100 fs PT-CN and Velocity Verlet BOMD comparison', '',
             f'Physical duration: {DURATION * AU_FS:.6f} fs. Raw trajectories remain on Euler scratch.', '',
             'Single-run wall times are descriptive; the reference must pass its convergence check.', '',
             '| Backend | Method | dt / au | Solver tolerance | Complete | Max energy / meV per atom | OH RMS / Angstrom | Angle RMS / degree | Position RMS / Angstrom | Propagation / s |',
             '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        observable = record.get('observables', {})
        fields = [record['backend'], record['method'], str(record['dt']), f"{record['tolerance']:.0e}",
                  str(record['completed'])]
        for key in ('energy_max_mev_per_atom', 'oh_rms_angstrom', 'angle_rms_degree', 'position_rms_angstrom'):
            fields.append(f'{observable[key]:.6g}' if key in observable else '--')
        fields.append(f"{record.get('wall_seconds', 0):.3f}")
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', 'BOMD reference checks:', '', json.dumps(summary['references'], indent=2), '',
              'PT-CN and BOMD solve different electronic models. Compare near-BO observables and timestep trends; do not interpret every difference as an implementation error.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


def self_check():
    manifest = cases()
    assert len(manifest) == 42 and len({c['case_id'] for c in manifest}) == 42
    assert all(abs(DURATION / c['dt'] - round(DURATION / c['dt'])) < 1e-10 for c in manifest)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepare(root)
        assert len(json.loads((root / 'cases.json').read_text())) == 42
    for backend in ('rhf', 'rks'):
        mf, masses, velocity = setup(backend)
        _, trajectory = native_bomd(mf, masses, velocity, 1.25, 5., 1e-11)
        check_trajectory(trajectory, 1.25, 5.)
        bonds, angles = geometry_observables(trajectory['positions'])
        assert bonds.shape == (5, 2) and angles.shape == (5,)
        assert np.linalg.norm(np.sum(masses[:, None] * velocity, axis=0)) < 1e-10
        finer = {key: np.repeat(value[:1], 9, axis=0) for key, value in trajectory.items()}
        finer['time'] = np.arange(9) * .625
        coarse = {key: value[::2] for key, value in finer.items()}
        assert metrics(coarse, finer)['oh_rms_angstrom'] == 0
    print('Manifest, native BOMD clock, isotope masses and geometry checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze', 'self-check'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    lib.num_threads(1)
    if args.action == 'self-check':
        self_check()
    elif args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'run':
        if args.case is None or args.case not in range(42):
            parser.error('--case must be in 0..41')
        run(args.root, args.case)
    else:
        analyze(args.root)


if __name__ == '__main__':
    main()
