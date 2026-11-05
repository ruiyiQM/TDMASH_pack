"""Validate physical-time excited-state PT-CN against refined conventional CN."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

import euler_c2h4_ehrenfest as study
from compare_c2h4_prefix import compare, load

DURATION = 200.
STRICT_BUDGET = dict(position_rms_angstrom=1e-4, force_rms_hartree_per_bohr=1e-4,
              total_dipole_rms_au=1e-4, spin_density_rms_frobenius=1e-4,
              max_energy_difference_hartree=1e-4,
              shared_node_mean_cc_error_angstrom=1e-4, shared_node_mean_ch_max_error_angstrom=1e-4)
BUDGET = {key: value for key, value in STRICT_BUDGET.items()
          if key not in ('force_rms_hartree_per_bohr', 'spin_density_rms_frobenius')}


def cases():
    rows = [dict(method='accelerated', dt=n, electronic_dt=e, role='reference')
            for n, e in ((.5, .0625), (.5, .03125), (.25, .03125), (.25, .015625))]
    for method in ('pt_cn', 'accelerated'):
        grid = (.25, .5, 1., 2., 2.5, 4., 5.) if method == 'pt_cn' else (.0625, .125, .25, .5, 1., 2., 2.5, 4., 5.)
        rows += [dict(method=method, dt=dt, electronic_dt=dt, role='candidate') for dt in grid]
    rows += [dict(method='pt_cn', dt=.5, electronic_dt=.5, role='tolerance_control', tolerance=1e-5)]
    for i, row in enumerate(rows):
        row.update(case_id=i, backend='uks', alpha=1.)
        row.setdefault('tolerance', 1e-9 if row['method'] == 'pt_cn' else 1e-10)
    return rows


def prepare(root):
    study.prepare(root)
    code = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol.update(duration_au=DURATION, duration_fs=DURATION * study.AU_FS,
                    question='At mu=1, reproduce refined conventional-CN Ehrenfest dynamics from the same unrestricted DeltaSCF initial state; compare largest accurate timestep.',
                    reference='CN electron refinement: 0 vs 1 and 2 vs 3; nuclear refinement: 1 vs 2; finest reference is case 3.',
                    sampling='Occupied spin coefficients at every nuclear endpoint; compare orthogonal-basis spin densities, not orbital phases.',
                    budget=BUDGET, acceptance='Complete common time window, validated CN reference, trajectory/dipole/energy budgets; force and spin density remain diagnostics and legacy strict checks.',
                    tolerances=dict(pt_cn=1e-9, conventional_cn_mo_change=1e-10, pt_tolerance_control=1e-5),
                    timing='Descriptive propagation/Fock counts only; short trajectory results do not establish 100 fs accuracy or speedup.')
    for name in ('euler_c2h4_cn_reference.py', 'compare_c2h4_prefix.py', 'benchmark_c2h4_cn_pt.py'):
        path = code / 'scripts' / name
        protocol['source_sha256'][str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (root / 'cases.json').write_text(json.dumps(cases(), indent=2))


def run(root, number):
    study.verify(root)
    case = json.loads((root / 'cases.json').read_text())[number]
    duration = json.loads((root / 'protocol.json').read_text())['duration_au']
    if hasattr(os, 'sched_getaffinity') and len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('The comparison requires one allocated CPU.')
    output = root / 'runs' / f'case_{number:03d}'
    output.mkdir(parents=True, exist_ok=False)
    study.SAMPLE_INTERVAL = case['dt']
    record, trajectory = study.integrate(root, case, duration)
    record.update(case, duration_au=duration, host=platform.node(), job_id=os.environ.get('SLURM_JOB_ID'))
    np.savez_compressed(output / 'trajectory.npz', **trajectory)
    record['trajectory_sha256'] = hashlib.sha256((output / 'trajectory.npz').read_bytes()).hexdigest()
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def orthogonal_densities(root, case, trajectory):
    mf, _, _ = study.load_initial(root, case['backend'])
    densities = []
    for time, C in zip(trajectory['sample_time'], trajectory['sample_coefficients']):
        index = round(time / case['dt'])
        mol = mf.mol.set_geom_(trajectory['positions'][index], unit='Bohr', inplace=False)
        B, _ = study.half_overlap(mol)
        X = B @ C
        densities.append(X @ X.conj().swapaxes(-1, -2))
    if not np.allclose(trajectory['sample_time'], trajectory['time'], rtol=0, atol=1e-10):
        raise RuntimeError('Density snapshots must cover every nuclear endpoint.')
    return np.array(densities)


def metrics(candidate, reference, density, reference_density, interval=None):
    reference_time = reference['time']
    indices = np.minimum(np.searchsorted(reference_time, candidate['time']), len(reference_time) - 1)
    left = np.maximum(indices - 1, 0)
    indices = np.where(abs(reference_time[left] - candidate['time']) < abs(reference_time[indices] - candidate['time']), left, indices)
    shared = np.isclose(candidate['time'], reference_time[indices], rtol=0, atol=1e-10)
    if interval is not None:
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('The comparison interval must be finite and positive.')
        shared &= np.isclose(candidate['time'] / interval, np.rint(candidate['time'] / interval), rtol=0, atol=1e-8)
    candidate = {key: value[shared] for key, value in candidate.items()
                 if key in ('time', 'positions', 'forces', 'dipoles', 'energy')}
    density = density[shared]
    result = compare(candidate, reference)
    if not result['comparable']:
        return result
    keep = candidate['time'] <= reference['time'][-1] + 1e-10
    time = candidate['time'][keep]
    indices = np.rint(time / reference['time'][1]).astype(int)
    if not np.allclose(time, reference['time'][indices], rtol=0, atol=1e-10):
        raise ValueError('CN comparison requires shared time nodes.')
    force = candidate['forces'][keep] - reference['forces'][indices]
    delta = density[keep] - reference_density[indices]
    density_norm = np.sqrt(np.sum(abs(delta)**2, axis=(1, 2, 3)))
    result.update(force_rms_hartree_per_bohr=float(np.sqrt(np.mean(np.sum(force**2, axis=-1)))),
                  spin_density_rms_frobenius=float(np.sqrt(np.mean(density_norm**2))),
                  spin_density_max_frobenius=float(np.max(density_norm)))
    return result


def passes(result, budget=None):
    budget = BUDGET if budget is None else budget
    if (not set(BUDGET).issubset(budget) or not set(budget).issubset(STRICT_BUDGET)
            or any(not np.isfinite(value) or value <= 0 for value in budget.values())):
        raise ValueError('An observable budget must retain trajectory, dipole and energy metrics with positive finite limits.')
    return result.get('comparable', False) and all(result[key] <= value for key, value in budget.items())


def analyze(root):
    study.verify(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    budget = protocol.get('budget', BUDGET)
    configuration = json.loads((root / 'cases.json').read_text())
    references = [load(root, i) for i in range(4)]
    densities = [orthogonal_densities(root, configuration[i], references[i][1]) for i in range(4)]
    checks = []
    for coarse, fine, kind in ((0, 1, 'electronic'), (1, 2, 'nuclear'), (2, 3, 'electronic')):
        fixed = configuration[coarse]['dt' if kind == 'electronic' else 'electronic_dt']
        label = f'{kind}_refinement_at_fixed_step_{fixed}'
        result = metrics(references[coarse][1], references[fine][1], densities[coarse], densities[fine])
        complete = references[coarse][0]['completed'] and references[fine][0]['completed']
        checks.append(dict(label=label, coarse=coarse, fine=fine, metrics=result,
                           passed=complete and passes(result, budget), passed_strict=complete and passes(result, STRICT_BUDGET)))
    validated = all(check['passed'] for check in checks)
    strict_validated = all(check['passed_strict'] for check in checks)
    rows = []
    for case in configuration:
        record, trajectory = load(root, case['case_id'])
        density = densities[case['case_id']] if case['case_id'] < 4 else orthogonal_densities(root, case, trajectory)
        result = metrics(trajectory, references[3][1], density, densities[3])
        interval = protocol.get('common_comparison_interval_au')
        common = metrics(trajectory, references[3][1], density, densities[3], interval) if interval is not None else None
        # A common sampling grid supplements the original full shared-node budget.
        accepted = validated and record['completed'] and passes(result, budget) and (common is None or passes(common, budget))
        strict = strict_validated and record['completed'] and passes(result, STRICT_BUDGET) and (common is None or passes(common, STRICT_BUDGET))
        rows.append(dict(record, metrics=result, common_grid_metrics=common, passed=bool(accepted), passed_strict=bool(strict)))
    bounds = {method: max((r['dt'] for r in rows if r['role'] == 'candidate' and r['method'] == method and r['passed']), default=None)
              for method in ('pt_cn', 'accelerated')}
    summary = dict(protocol=protocol, cn_reference_validated=validated,
                   cn_reference_validated_strict=strict_validated, strict_budget=STRICT_BUDGET,
                   reference_checks=checks, largest_tested_accurate_dt=bounds,
                   largest_tested_accurate_dt_strict={method: max((r['dt'] for r in rows if r['role'] == 'candidate' and r['method'] == method and r['passed_strict']), default=None)
                                                     for method in ('pt_cn', 'accelerated')}, records=rows)
    with (root / 'compact_summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(dict(cn_reference_validated=validated, largest_tested_accurate_dt=bounds)), flush=True)


def self_check():
    rows = cases()
    assert len(rows) == 21 and all(row['alpha'] == 1 for row in rows)
    assert all(DURATION % row['dt'] == 0 and row['dt'] % row['electronic_dt'] == 0 for row in rows)
    trajectory = dict(time=np.array([0., .25, .5]), positions=np.ones((3, 6, 3)),
                      dipoles=np.zeros((3, 3)), energy=np.zeros(3), forces=np.zeros((3, 6, 3)))
    density = np.zeros((3, 2, 2, 2))
    assert passes(metrics(trajectory, trajectory, density, density))
    result = metrics(trajectory, trajectory, density, density)
    result['spin_density_rms_frobenius'] = 1e-2
    result['force_rms_hartree_per_bohr'] = 1e-2
    assert not passes(result, STRICT_BUDGET) and passes(result)
    result['total_dipole_rms_au'] = 1.5e-4
    assert not passes(result)
    try:
        passes(result, {'spin_density_rms_frobenius': 2e-4})
    except ValueError:
        pass
    else:
        raise AssertionError('An observable budget must retain trajectory, dipole and energy checks.')
    changed = density.copy()
    changed[-1, 0, 0, 0] = .01
    assert not passes(metrics(trajectory, trajectory, changed, density), STRICT_BUDGET)
    changed = density.copy()
    changed[1, 0, 0, 0] = .01
    assert not passes(metrics(trajectory, trajectory, changed, density), STRICT_BUDGET)
    common = metrics(trajectory, trajectory, changed, density, interval=.5)
    assert common['frames'] == 2 and passes(common, STRICT_BUDGET)
    for interval in (0, -1, np.inf, np.nan):
        try:
            metrics(trajectory, trajectory, density, density, interval)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid comparison intervals must be rejected.')
    angle = .37
    X = np.eye(3, dtype=complex)[:, :2]
    U = np.array([[np.cos(angle), 1j*np.sin(angle)], [1j*np.sin(angle), np.cos(angle)]])
    assert np.linalg.norm((X @ U) @ (X @ U).conj().T - X @ X.conj().T) < 1e-14
    print('Physical alpha, refinement matrix, density errors and occupied-gauge invariance checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze', 'self-check'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    if args.action == 'self-check':
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    elif args.action == 'run':
        if args.case is None:
            parser.error('--case is required')
        run(args.root, args.case)
    else:
        globals()[args.action](args.root)
