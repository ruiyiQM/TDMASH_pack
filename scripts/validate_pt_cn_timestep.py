"""Compare Ehrenfest timesteps at a fixed observable error budget."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from pyscf import dft, gto, lib, scf

from tdmash import TDRHFSimulation, TDRKSSimulation


BUDGET = dict(position_bohr=1e-5, force_hartree_per_bohr=1e-5,
              dipole_e_bohr=1e-4, density_frobenius=1e-4)
ENERGY_BUDGET = 1e-5
TIMESTEPS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
REFERENCE_TOLERANCES = (1e-9, 1e-11)


def errors(candidate, reference):
    """Compare complete trajectories on exactly shared times, without interpolation."""
    right = np.minimum(np.searchsorted(reference['time'], candidate['time']), len(reference['time']) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(reference['time'][left] - candidate['time'])
                       < abs(reference['time'][right] - candidate['time']), left, right)
    if not np.allclose(reference['time'][indices], candidate['time'], rtol=0, atol=1e-10):
        raise ValueError("Trajectories must share exact physical time nodes.")
    result = {}
    for key, metric in (('positions', 'position_bohr'), ('forces', 'force_hartree_per_bohr'),
                        ('dipoles', 'dipole_e_bohr'), ('density', 'density_frobenius')):
        difference = candidate[key] - reference[key][indices]
        axes = (-2, -1) if key == 'density' else -1
        result[metric] = float(np.max(np.linalg.norm(difference, axis=axes)))
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite trajectory comparison.")
    return result


def self_check():
    reference = dict(time=np.array([0., 0.5, 1.]), positions=np.zeros((3, 2, 3)),
                     forces=np.zeros((3, 2, 3)), dipoles=np.zeros((3, 3)),
                     density=np.zeros((3, 2, 2)))
    candidate = {key: value[::2].copy() for key, value in reference.items()}
    assert max(errors(candidate, reference).values()) == 0
    candidate['positions'][1, 0, 0] = 0.02
    assert errors(candidate, reference)['position_bohr'] == 0.02
    candidate['time'][1] = 0.9
    try:
        errors(candidate, reference)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonshared time nodes must be rejected.")


def run_case(mf, simulation, method, dt, duration, output, label):
    densities, iterations, residuals = [], [], []
    counts = {'fock_builds': 0}
    started = time.perf_counter()
    sim = simulation(mf, dt=dt, nuclei_dt=dt, steps=round(duration / dt), ehrenfest=True,
                     propagation_method=method, convergence_threshold=1e-11, max_iterations=160,
                     initial_velocity=np.array([[0., 0., 0.], [8e-4, 0., 2e-4], [-5e-4, 1e-4, -3e-4]]),
                     output_filename=str(output / f'{label}.xyz'))
    original_fock = sim.get_fock_matrix
    original_dipole = sim.calculate_dipole
    original_propagate = sim.propagate_electrons

    def fock(*args, **kwargs):
        counts['fock_builds'] += 1
        return original_fock(*args, **kwargs)

    def dipole(D):
        densities.append(sim.overlap_half @ D @ sim.overlap_half)
        return original_dipole(D)

    def propagate(*args, **kwargs):
        original_propagate(*args, **kwargs)
        if kwargs.get('dt') != 0:
            iterations.append(sim.pt_cn_iterations if method == 'pt_cn' else sim.pc_iterations)
            if method == 'pt_cn':
                residuals.append(sim.pt_cn_residual)

    sim.get_fock_matrix, sim.calculate_dipole, sim.propagate_electrons = fock, dipole, propagate
    record = dict(method=method, dt=dt, duration=duration)
    try:
        sim.run_ehrenfest_midpoint_decoupled()
        record['completed'] = True
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        record.update(completed=False, failure=str(exc))
    record['wall_seconds'] = time.perf_counter() - started
    data = dict(time=np.asarray(sim.times), positions=np.asarray(sim.positions_history),
                velocities=np.asarray(sim.velocities_history), forces=np.asarray(sim.forces_history),
                dipoles=np.asarray(sim.dipoles), density=np.asarray(densities), energy=np.asarray(sim.total_energies))
    if record['completed'] and (len(data['time']) != round(duration / dt) + 1
                                or not np.isclose(data['time'][-1], duration, rtol=0, atol=1e-10)):
        raise RuntimeError("A case did not cover its complete physical interval.")
    record.update(counts, completed_steps=len(iterations), final_time=float(sim.t),
                  max_iterations=max(iterations, default=0),
                  max_pt_residual=max(residuals, default=None),
                  energy_drift_hartree_per_atom=float(np.max(np.abs(data['energy'] - data['energy'][0])) / sim.mol.natm))
    record['data_file'] = f'{label}.npz'
    np.savez_compressed(output / record['data_file'], **data)
    print(label, 'complete' if record['completed'] else 'FAILED',
          f"{record['wall_seconds']:.2f}s", flush=True)
    return record, data



def run_reference(mf, simulation, duration, output, label, tolerance):
    """Integrate the independent density/nuclear ODE in orthogonal coordinates."""
    started = time.perf_counter()
    sim = simulation(mf, dt=.1, ehrenfest=True)
    shape = sim.overlap.shape
    size = sim.overlap.size
    natm = sim.mol.natm
    D_initial = sim.overlap_half @ sim.make_rdm1(sim.mo_coeff, sim.mo_occ) @ sim.overlap_half
    initial = np.concatenate((D_initial.ravel(), sim.positions.ravel(),
                              np.array([[0., 0., 0.], [8e-4, 0., 2e-4], [-5e-4, 1e-4, -3e-4]]).ravel()))

    def state(flat):
        density = flat[:size].reshape(shape)
        positions = flat[size:size + 3 * natm].real.reshape(natm, 3)
        velocities = flat[size + 3 * natm:].real.reshape(natm, 3)
        sim.update_mol(positions)
        sim.velocities = velocities
        D = sim.overlap_half_inv @ density @ sim.overlap_half_inv
        sim.F = sim.get_fock_matrix(D)
        sim.force = -sim.energy_grad(sim.F, D, lowdin=True)
        return density, D

    def derivative(t, flat):
        density, _ = state(flat)
        H = sim.overlap_half_inv @ sim.F @ sim.overlap_half_inv
        derivative_density = -1j * (H @ density - density @ H)
        return np.concatenate((derivative_density.ravel(), sim.velocities.ravel(),
                               (sim.force / sim.mass[:, None]).ravel()))

    times = np.linspace(0, duration, round(duration / .1) + 1)
    solution = solve_ivp(derivative, (0, duration), initial, method='DOP853', t_eval=times,
                         rtol=tolerance, atol=tolerance * .01)
    if not solution.success or len(solution.t) != len(times):
        raise RuntimeError(f'Independent reference failed: {solution.message}')
    data = {key: [] for key in ('positions', 'velocities', 'forces', 'dipoles', 'density', 'energy')}
    for flat in solution.y.T:
        density, D = state(flat)
        data['positions'].append(sim.positions.copy())
        data['velocities'].append(sim.velocities.copy())
        data['forces'].append(sim.force.copy())
        data['dipoles'].append(sim.calculate_dipole(D))
        data['density'].append(density.copy())
        data['energy'].append(sim.energy_tot(dm_total=D) + .5 * np.sum(sim.mass[:, None] * sim.velocities**2))
    data = {key: np.asarray(value) for key, value in data.items()}
    data['time'] = times
    record = dict(method='DOP853 density/nuclear ODE', rtol=tolerance, atol=tolerance * .01,
                  completed=True, duration=duration, rhs_evaluations=solution.nfev,
                  wall_seconds=time.perf_counter() - started,
                  energy_drift_hartree_per_atom=float(np.max(np.abs(data['energy'] - data['energy'][0])) / natm),
                  data_file=f'{label}.npz')
    np.savez_compressed(output / record['data_file'], **data)
    print(label, 'complete', f"{record['wall_seconds']:.2f}s", flush=True)
    return record, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/pt_cn_timestep'))
    parser.add_argument('--duration', type=float, default=32.0)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('Trajectory comparison self-check passed.')
        return
    if (not np.isfinite(args.duration) or args.duration <= 0
            or any(not np.isclose(args.duration / dt, round(args.duration / dt), rtol=0, atol=1e-9)
                   for dt in TIMESTEPS)):
        parser.error('Duration must be a positive common multiple of all tested timesteps.')
    args.output.mkdir(parents=True, exist_ok=False)
    lib.num_threads(1)
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / 'src/tdmash').glob('*.py')), Path(__file__).resolve()]
    summary = dict(protocol=dict(duration=args.duration, mu=1, field='none', basis='sto-3g',
                                budgets=BUDGET, energy_drift_hartree_per_atom=ENERGY_BUDGET,
                                reference_budget_fraction=.1, nonlinear_tolerance=1e-11,
                                reference_method='DOP853 density/nuclear ODE', reference_tolerances=REFERENCE_TOLERANCES,
                                timesteps=TIMESTEPS, timing='Single local run including simulation setup and propagation; excludes shared SCF and archive writing.'),
                   environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                    pyscf=__import__('pyscf').__version__, platform=platform.platform(),
                                    threads=1, base_commit=subprocess.check_output(
                                        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()),
                   source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   backends={})
    (args.output / 'protocol.json').write_text(json.dumps(summary, indent=2))
    for backend, factory, simulation in [('rhf', scf.RHF, TDRHFSimulation), ('rks', dft.RKS, TDRKSSimulation)]:
        mf = factory(gto.M(atom='O 0 0 0; H 1.4 0 0.9; H -1.2 0 1.1',
                          unit='Bohr', basis='sto-3g', verbose=0))
        if backend == 'rks':
            mf.xc, mf.grids.level, mf.grids.prune, mf.small_rho_cutoff = 'lda,vwn', 0, None, 0
        mf.conv_tol = 1e-12
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Benchmark initial SCF did not converge.')
        result = dict(references=[], candidates=[])
        references = []
        for tolerance in REFERENCE_TOLERANCES:
            record, data = run_reference(mf, simulation, args.duration, args.output,
                                         f'{backend}_reference_rtol{tolerance}', tolerance)
            if not record['completed']:
                raise RuntimeError('Reference propagation failed.')
            result['references'].append(record)
            references.append(data)
        reference_error = errors(references[0], references[1])
        result['reference_errors'] = reference_error
        if (any(reference_error[key] > .1 * limit for key, limit in BUDGET.items())
                or any(r['energy_drift_hartree_per_atom'] > .1 * ENERGY_BUDGET for r in result['references'])):
            (args.output / f'{backend}_rejected_reference.json').write_text(json.dumps(result, indent=2))
            raise RuntimeError(f'{backend}: independent reference tolerance convergence is insufficient.')
        for method in ('accelerated', 'pt_cn'):
            for dt in TIMESTEPS:
                record, data = run_case(mf, simulation, method, dt, args.duration,
                                        args.output, f'{backend}_{method}_dt{dt}')
                if record['completed']:
                    record['errors'] = errors(data, references[-1])
                    record['passed_accuracy'] = (all(record['errors'][key] <= limit for key, limit in BUDGET.items())
                                                 and record['energy_drift_hartree_per_atom'] <= ENERGY_BUDGET)
                else:
                    record['passed_accuracy'] = False
                result['candidates'].append(record)
                (args.output / f'{backend}_progress.json').write_text(json.dumps(result, indent=2))
        result['largest_passing_dt'] = {
            method: max((r['dt'] for r in result['candidates'] if r['method'] == method and r['passed_accuracy']), default=None)
            for method in ('accelerated', 'pt_cn')}
        summary['backends'][backend] = result
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({backend: result['largest_passing_dt'] for backend, result in summary['backends'].items()}, indent=2))


if __name__ == '__main__':
    main()
