"""Replay an immutable ethylene failure with controlled mixing and Newton-Krylov."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.optimize import newton_krylov
from scipy.optimize._nonlin import NoConvergence
from scipy.sparse.linalg import LinearOperator

import euler_c2h4_ehrenfest as study
import tdmash.PT as pt
from tdmash.Anderson import AndersonAcceleration
from tdmash import TDUKSSimulation


def state(root, number):
    folder = root / 'runs' / f'case_{number:03d}'
    record = json.loads((folder / 'result.json').read_text())
    raw = folder / 'trajectory.npz'
    if hashlib.sha256(raw.read_bytes()).hexdigest() != record['trajectory_sha256']:
        raise RuntimeError('Replay trajectory checksum mismatch.')
    with np.load(raw) as archive:
        data = dict(archive)
    snapshot_time = float(data['sample_time'][-1])
    index = round(snapshot_time / record['dt'])
    mf, masses, _ = study.load_initial(root, record['backend'])
    mf.reset(mf.mol.set_geom_(data['positions'][index], unit='Bohr', inplace=False))
    steps = round((float(data['time'][-1]) - snapshot_time) / record['dt'])
    sim = TDUKSSimulation(mf, dt=record['dt'], nuclei_dt=record['dt'], steps=steps,
                         propagation_method='pt_cn', mu=record['alpha'], ehrenfest=True,
                         initial_velocity=data['velocities'][index].copy(), convergence_threshold=1e-6,
                         max_iterations=120, output_filename=os.devnull)
    sim.mass, sim.mo_coeff = masses.copy(), data['sample_coefficients'][-1].copy()
    sim.run()
    assert np.linalg.norm(sim.positions - data['positions'][-1]) < 1e-8
    sim.steps = 1
    return sim, record, float(data['time'][-1])


def action(sim, C, F):
    return np.array([sim.mu * sim.overlap_inv @ F[s] @ C[s]
                     - C[s] @ (C[s].conj().T @ (sim.mu * F[s]) @ C[s]) for s in (0, 1)])


def replay(root, number, tolerance, history=6, clip=False, absolute=False):
    sim, original, start_time = state(root, number)
    sim.convergence_threshold = tolerance
    trace, last_fock = [], [sim.F]
    get_fock = sim.get_fock_matrix

    def fock(*args, **kwargs):
        last_fock[0] = get_fock(*args, **kwargs)
        return last_fock[0]

    sim.get_fock_matrix = fock

    class ControlledMixing:
        def __init__(self, **kwargs):
            self.inner = AndersonAcceleration(history_size=history, regularization=1e-12)
            self.regularization = 1e-12
            self.shape = sim.mo_coeff.shape
            self.rhs = sim.mo_coeff - .5j * sim.dt * action(sim, sim.mo_coeff, last_fock[0])

        def compute(self, current, candidate):
            C = (current[0] + 1j * current[1]).reshape(self.shape)
            residual = self.rhs - C - .5j * sim.dt * action(sim, C, last_fock[0])
            trace.append(float(np.linalg.norm(residual)))
            self.inner.regularization = 1e-12 if absolute else self.regularization
            mixed = self.inner.compute(current, candidate)
            if clip:
                length = np.linalg.norm(mixed - current)
                limit = 2 * np.linalg.norm(candidate - current)
                if length > limit:
                    mixed = current + (mixed - current) * (limit / length)
            return mixed

    result = dict(case_id=number, method='anderson', tolerance=tolerance, history=history,
                  clipped=clip, absolute_regularization=absolute, replay_time_au=start_time)
    started = time.perf_counter()
    try:
        with patch.object(pt, 'AndersonAcceleration', ControlledMixing):
            sim.run()
        result.update(completed=True, residual=float(sim.pt_cn_residual), iterations=sim.pt_cn_iterations,
                      energy_change_hartree=float(sim.total_energies[-1] - sim.total_energies[0]))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        result.update(completed=False, failure=str(error)[:400])
    result.update(seconds=time.perf_counter() - started, minimum_mixed_residual=min(trace, default=None),
                  first_residual=trace[0] if trace else None, last_residual=trace[-1] if trace else None)
    result['minimum_residual'] = min(trace + ([result['residual']] if result['completed'] else []), default=None)
    return result


def frozen_guess(S, F, C, mu, a):
    """Include the normal CN displacement in the initial iterate."""
    energies, rotation = eigh(C.conj().T @ (mu * F) @ C)
    rotated = C @ rotation
    rhs = S @ rotated - 1j * a * (mu * F @ rotated - (S @ rotated) * energies)
    columns = [np.linalg.solve(S + 1j * a * (mu * F - e * S), rhs[:, j])
               for j, e in enumerate(energies)]
    return np.column_stack(columns) @ rotation.conj().T


def self_check():
    S = np.array([[1.2, .1], [.1, .8]])
    F = np.array([[.7, .2j], [-.2j, -.3]])
    C = np.array([[1.], [.2j]])
    C /= np.sqrt((C.conj().T @ S @ C).item().real)
    mu, a = 10., 2.5

    def act(block):
        return np.linalg.solve(S, mu * F @ block) - block @ (block.conj().T @ (mu * F) @ block)

    guess = frozen_guess(S, F, C, mu, a)
    assert np.linalg.norm(guess.conj().T @ S @ guess - np.eye(1)) < 1e-12
    assert np.linalg.norm(guess + 1j * a * act(guess) - C + 1j * a * act(C)) < 1e-11
    assert np.linalg.norm(frozen_guess(S, F, C, mu, 0) - C) < 1e-12
    print('Frozen-Fock single-orbital PT-CN predictor checks passed.')


def newton(root, number, tolerance, predictor=False):
    sim, original, start_time = state(root, number)
    sim.calculate_forces(lowdin=True)
    v_half = sim.velocities + .5 * sim.dt * sim.force / sim.mass[:, None]
    sim.update_mol(sim.positions + .5 * sim.dt * v_half)
    C0 = sim.mo_coeff.copy()
    F0 = sim.get_fock_matrix(sim.make_rdm1(C0, sim.mo_occ))
    rhs = C0 - .5j * sim.dt * action(sim, C0, F0)
    size = C0.size
    preconditioners = [pt._preconditioner(sim.overlap, F0[s], C0[s], sim.mu, .5 * sim.dt) for s in (0, 1)]
    calls, errors = [0], []

    def pack(C):
        return np.concatenate((C.real.ravel(), C.imag.ravel()))

    def unpack(vector):
        return (vector[:size] + 1j * vector[size:]).reshape(C0.shape)

    def residual(vector):
        C = unpack(vector)
        F = sim.get_fock_matrix(sim.make_rdm1(C, sim.mo_occ))
        R = C + .5j * sim.dt * action(sim, C, F) - rhs
        calls[0] += 1
        errors.append(float(np.linalg.norm(R)))
        return pack(R)

    def inverse(vector):
        blocks = unpack(vector)
        return pack(np.array([p(C) for p, C in zip(preconditioners, blocks)]))

    operator = LinearOperator((2 * size, 2 * size), matvec=inverse, dtype=float)
    outer = [0]

    def accepted_step(x, f):
        outer[0] += 1

    seed = np.array([frozen_guess(sim.overlap, F0[s], C0[s], sim.mu, .5 * sim.dt)
                     for s in (0, 1)]) if predictor else C0
    result = dict(case_id=number, method='newton_krylov', tolerance=tolerance,
                  frozen_fock_predictor=predictor, replay_time_au=start_time)
    started = time.perf_counter()
    try:
        vector = newton_krylov(residual, pack(seed), method='lgmres', inner_M=operator,
                               inner_maxiter=30, maxiter=30, rdiff=1e-6, f_tol=tolerance,
                               tol_norm=np.linalg.norm, callback=accepted_step)
        final_residual = float(np.linalg.norm(residual(vector)))
        assert final_residual <= tolerance
        C = unpack(vector)
        gram_error = max(float(np.linalg.norm(block.conj().T @ sim.overlap @ block - np.eye(block.shape[1]))) for block in C)
        for s, block in enumerate(C):
            root_factor = cholesky(block.conj().T @ sim.overlap @ block, lower=True)
            C[s] = solve_triangular(root_factor, block.conj().T, lower=True).conj().T
        result.update(completed=True, residual=final_residual, gram_error_before_normalization=gram_error,
                      transported_start_fidelity=study.fidelity(C0, C, sim.overlap))
    except (NoConvergence, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        result.update(completed=False, failure=type(error).__name__)
    result.update(seconds=time.perf_counter() - started, fock_evaluations=calls[0], outer_iterations=outer[0],
                  first_residual=errors[0] if errors else None, minimum_residual=min(errors, default=None),
                  last_residual=errors[-1] if errors else None)
    return result


def substeps(root, number, count):
    sim, original, start_time = state(root, number)
    sim.convergence_threshold = 1e-5
    sim.calculate_forces(lowdin=True)
    v_half = sim.velocities + .5 * sim.dt * sim.force / sim.mass[:, None]
    sim.update_mol(sim.positions + .5 * sim.dt * v_half)
    initial = sim.mo_coeff.copy()
    initial_time = sim.t
    F = sim.get_fock_matrix(sim.make_rdm1(sim.mo_coeff, sim.mo_occ))
    h = sim.dt / count
    result = dict(case_id=number, method='fixed_midpoint_electronic_substeps', tolerance=1e-5,
                  nuclear_dt=sim.dt, electronic_dt=h, substeps=count, replay_time_au=start_time)
    residuals, iterations = [], []
    started = time.perf_counter()
    try:
        for i in range(count):
            sim.propagate_electrons(F, 0, dt=h)
            residuals.append(float(sim.pt_cn_residual))
            iterations.append(sim.pt_cn_iterations)
            sim.t = initial_time + (i + 1) * h
            F = sim.F
        assert np.isclose(sim.t - initial_time, sim.dt, rtol=0, atol=1e-12)
        assert max(residuals) <= sim.convergence_threshold
        result.update(completed=True, max_residual=max(residuals),
                      transported_start_fidelity=study.fidelity(initial, sim.mo_coeff, sim.overlap))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        result.update(completed=False, failure=str(error)[:400])
    result.update(seconds=time.perf_counter() - started, completed_substeps=len(residuals),
                  electronic_iterations=sum(iterations))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', type=int, choices=(0, 1))
    parser.add_argument('--predictor', action='store_true')
    parser.add_argument('--substeps', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None or args.case is None:
        parser.error('--root and --case are required for a replay.')
    if args.predictor and args.substeps:
        parser.error('--predictor and --substeps select separate experiments.')
    study.verify(args.root)
    records = []
    policies = () if args.predictor or args.substeps else ((1e-6, 6, False, False), (1e-5, 6, False, False),
                                               (1e-5, 6, False, True), (1e-6, 12, True, False),
                                               (1e-5, 12, True, False))
    for tolerance, history, clip, absolute in policies:
        result = replay(args.root, args.case, tolerance, history, clip, absolute)
        records.append(result)
        print(json.dumps(result), flush=True)
    experiments = [substeps(args.root, args.case, count) for count in (1, 2, 4, 8, 16)] if args.substeps else [
        newton(args.root, args.case, tolerance, predictor=args.predictor) for tolerance in (1e-6, 1e-5)]
    for result in experiments:
        records.append(result)
        print(json.dumps(result), flush=True)
    folder = 'solver_substep_probe' if args.substeps else ('solver_predictor_probe' if args.predictor else 'solver_probe')
    output = args.root / folder / f'case_{args.case:03d}.json'
    output.parent.mkdir(exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(host=platform.node(), affinity=sorted(os.sched_getaffinity(0)), records=records,
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)


if __name__ == '__main__':
    main()
