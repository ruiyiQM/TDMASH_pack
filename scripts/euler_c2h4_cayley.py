"""Independent fixed-nuclei test of Cayley dispersion in CN and PT-CN."""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from pyscf import lib
from pyscf.tdscf.uhf import get_ab
from scipy.linalg import eig, eigh, expm

import euler_c2h4_ehrenfest as study
from tdmash import TDUKSSimulation

DURATION = 1200.
PILOT_DURATION = 30.
TIMESTEPS = (.05, .5, 1., 2., 5., 10., 15.)
KICKS = (0., 1e-4, 5e-4)
DIRECTION = np.array([1., 2., 3.]) / np.sqrt(14.)
TOLERANCE = 1e-11
ITERATIONS = 120


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def cases():
    rows = []
    for state in ('ground', 'dscf'):
        settings = [('accelerated', h, 'reference') for h in (.01, .005)]
        settings += [(method, h, 'candidate') for method in ('accelerated', 'pt_cn') for h in TIMESTEPS]
        for method, h, role in settings:
            rows.append(dict(case_id=len(rows), state=state, method=method, dt=h, role=role))
    return rows


def prepare(root, source):
    """Copy the frozen determinant without changing the paused study."""
    root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / 'initial.json').read_text())
    if digest(source / 'initial.npz') != metadata['sha256']:
        raise RuntimeError('Frozen initial-state checksum mismatch.')
    for name in ('initial.npz', 'initial.json'):
        if (root / name).exists():
            raise FileExistsError(root / name)
        shutil.copyfile(source / name, root / name)
    code = Path(__file__).resolve().parents[1]
    files = [*sorted((code / 'src/tdmash').glob('*.py')),
             code / 'scripts/euler_c2h4_ehrenfest.py', Path(__file__),
             code / 'scripts/analyze_c2h4_cayley.py', code / 'scripts/euler_c2h4_cayley.sbatch']
    protocol = dict(independent_study=True, resumes_paused_goal=False, paused_goal_refinements_consumed=0,
                    geometry_bohr=study.GEOMETRY, basis=study.BASIS, xc='lda,vwn', grid_level=0,
                    grid_pruning=None, nuclei='fixed', alpha=1., field_after_kick='none',
                    frozen_initial_root=str(source), frozen_initial_sha256=metadata['sha256'],
                    duration_au=DURATION, pilot_duration_au=PILOT_DURATION,
                    kick_strengths=KICKS, kick_direction=DIRECTION.tolist(),
                    kick='Finite-basis length-gauge exp(-i*kappa*Q*r.e*Q), mapped back with Q=S^(-1/2).',
                    baseline='An independent zero-kick trajectory for every state, method and dt.',
                    solver=dict(tolerance=TOLERANCE, max_iterations=ITERATIONS,
                                acceptance='Both methods must satisfy a common relative endpoint-equation residual <=1e-10 before PT orthonormalization; post-projection defects are diagnostic only.',
                                no_retry_or_subcycling=True),
                    spectral='Multichannel matrix-pencil harmonic inversion; independent fits at multiple ranks/windows; no frequency snapping to theory.',
                    response='Full unrestricted A/B response about the same density; report complex eigenvalues separately.',
                    source_sha256={str(p.relative_to(code)): digest(p) for p in files})
    write_json(root / 'protocol.json', protocol)
    write_json(root / 'cases.json', cases())


def verify(root):
    study.verify(root)
    p = json.loads((root / 'protocol.json').read_text())
    if p.get('independent_study') is not True or p.get('resumes_paused_goal') is not False:
        raise RuntimeError('This workflow must remain an independent study.')


def load_state(root, state):
    if state == 'dscf':
        return study.load_initial(root, 'uks')[0]
    metadata = json.loads((root / 'ground.json').read_text())
    if digest(root / 'ground.npz') != metadata['sha256']:
        raise RuntimeError('Ground-state checksum mismatch.')
    mf = study.load_initial(root, 'uks')[0]
    with np.load(root / 'ground.npz', allow_pickle=False) as data:
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = (data[key].copy() for key in ('coeff', 'occ', 'mo_energy'))
        mf.e_tot = float(data['energy'])
    return mf


def response(mf, destination):
    """Use occupied/virtual semicanonical rotations without changing the density."""
    mf = mf.copy()
    density = mf.make_rdm1()
    F = mf.get_hcore() + mf.get_veff(dm=density)
    C, energies = np.array(mf.mo_coeff).copy(), np.array(mf.mo_energy).copy()
    for spin in (0, 1):
        for occupied in (True, False):
            indices = np.flatnonzero((mf.mo_occ[spin] > .5) == occupied)
            block = C[spin][:, indices]
            values, vectors = eigh(block.T @ F[spin] @ block)
            C[spin][:, indices] = block @ vectors
            energies[spin, indices] = values
    mf.mo_coeff, mf.mo_energy = C, energies
    if np.linalg.norm(mf.make_rdm1() - density) > 1e-11:
        raise RuntimeError('Semicanonicalization changed the frozen density.')
    a, b = get_ab(mf)
    def full(blocks):
        aa, ab, bb = (block.reshape(block.shape[0] * block.shape[1], -1) for block in blocks)
        return np.block([[aa, ab], [ab.T, bb]])
    A, B = full(a), full(b)
    L = np.block([[A, B], [-B, -A]])
    values, vectors = eig(L)
    r = mf.mol.intor('int1e_r', comp=3)
    transitions = np.concatenate([np.einsum('pi,xpq,qa->xia',
        C[s][:, mf.mo_occ[s] > .5], r, C[s][:, mf.mo_occ[s] < .5]).reshape(3, -1) for s in (0, 1)], axis=1)
    kick = transitions.T @ DIRECTION
    initial = np.r_[-1j * kick, 1j * kick]
    amplitudes = -(np.c_[transitions, transitions] @ vectors) * np.linalg.solve(vectors, initial)[None, :]
    strengths = np.linalg.norm(amplitudes, axis=0)
    growing = np.flatnonzero(values.imag > 1e-6)
    bright = sorted((i for i, value in enumerate(values) if value.real > .02 and abs(value.imag) < 1e-6),
                    key=lambda i: strengths[i], reverse=True)[:12]
    modes = lambda indices: [dict(omega=float(values[i].real), growth_rate=float(values[i].imag),
                                  dipole_amplitude_per_kappa=float(2 * strengths[i])) for i in indices]
    np.savez_compressed(destination.with_suffix('.npz'), A=A, B=B, eigenvalues=values,
                        dipole_amplitudes=amplitudes, energies=energies, occupations=mf.mo_occ)
    record = dict(dimension=len(L), orbital_gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))),
                  unstable=bool(len(growing)), growing_modes=modes(growing), bright_real_modes=modes(bright),
                  max_growth_rate=float(max(values.imag)), raw_sha256=digest(destination.with_suffix('.npz')))
    write_json(destination.with_suffix('.json'), record)
    return record


def initialize(root):
    verify(root)
    excited = study.load_initial(root, 'uks')[0]
    ground = study.mean_field(excited.mol.copy(), 'uks')
    ground.conv_tol, ground.conv_tol_grad = 1e-13, 1e-9
    ground = study.converge(ground)
    gradient = float(np.linalg.norm(ground.get_grad(ground.mo_coeff, ground.mo_occ)))
    if gradient > 1e-7 or ground.e_tot >= excited.e_tot:
        raise RuntimeError('The stable-control ground state was not converged.')
    with (root / 'ground.npz').open('xb') as stream:
        np.savez_compressed(stream, coeff=ground.mo_coeff, occ=ground.mo_occ,
                            mo_energy=ground.mo_energy, energy=ground.e_tot)
    write_json(root / 'ground.json', dict(sha256=digest(root / 'ground.npz'), energy=ground.e_tot,
                                        orbital_gradient=gradient, spin_square=float(ground.spin_square()[0])))
    records = {state: response(mf, root / ('response_' + state))
               for state, mf in (('ground', ground), ('dscf', excited))}
    print(json.dumps(records), flush=True)


def apply_kick(sim, kappa, r):
    Q, B = sim.overlap_half_inv, sim.overlap_half
    generator = Q @ np.einsum('x,xij->ij', DIRECTION, r) @ Q
    sim.mo_coeff = np.array([Q @ expm(-1j * kappa * generator) @ B @ C for C in sim.mo_coeff])


def endpoint_residual(sim, C0, C1, F0, F1, dt, method):
    if method == 'pt_cn':
        def action(C, F):
            HC = F @ C
            return sim.overlap_inv @ HC - C @ (C.conj().T @ HC)
        residual = np.array([new - old + .5j * dt * (action(old, f0) + action(new, f1))
                             for old, new, f0, f1 in zip(C0, C1, F0, F1)])
    else:
        Favg = .5 * (F0 + F1)
        residual = sim.overlap @ (C1 - C0) + .5j * dt * Favg @ (C0 + C1)
    return float(np.linalg.norm(residual) / max(1., np.linalg.norm(sim.overlap @ C0)))


def checked_residuals(sim, C0, F0, dt, method, recent_C, recent_F):
    post = endpoint_residual(sim, C0, sim.mo_coeff, F0, sim.F, dt, method)
    if method == 'pt_cn':
        if len(recent_C) != 2 or len(recent_F) != 2:
            raise RuntimeError('The converged PT trial and its projection were not captured.')
        # PT solves before orthonormalization; a projection defect is not a solver residual.
        raw = endpoint_residual(sim, C0, recent_C[0], F0, recent_F[0], dt, method)
        correction = float(np.linalg.norm(sim.mo_coeff - recent_C[0]))
    else:
        raw, correction = post, 0.
    return raw, post, correction


def propagate(root, case, kappa, duration, folder):
    mf = load_state(root, case['state'])
    sim = TDUKSSimulation(mf, dt=case['dt'], mu=1., propagation_method=case['method'],
                         convergence_threshold=TOLERANCE, max_iterations=ITERATIONS,
                         field_type='none', ehrenfest=False, output_filename=os.devnull)
    r = mf.mol.intor('int1e_r', comp=3)
    apply_kick(sim, kappa, r)
    initial_C = sim.mo_coeff.copy()
    dm = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
    nuclear = mf.mol.atom_charges() @ mf.mol.atom_coords()
    def moment(density):
        return nuclear - np.einsum('xij,sji->x', r, density).real
    native_error = float(np.max(abs(moment(dm) - mf.dip_moment(dm=dm, unit='AU', verbose=0))))
    legacy_error = float(np.max(abs(moment(dm) - (nuclear - sim.calculate_dipole(dm)))))
    if max(native_error, legacy_error) > 1e-11:
        raise RuntimeError('Independent and native total dipoles disagree.')
    count = dict(fock_builds=0)
    recent_C, recent_F = deque(maxlen=2), deque(maxlen=2)
    original_density = sim.make_rdm1
    def density(mo_coeff, mo_occ):
        recent_C.append(np.array(mo_coeff).copy())
        return original_density(mo_coeff, mo_occ)
    sim.make_rdm1 = density
    original = sim.get_fock_matrix
    def fock(*args, **kwargs):
        count['fock_builds'] += 1
        result = original(*args, **kwargs)
        recent_F.append(result.copy())
        return result
    sim.get_fock_matrix = fock
    F = fock(dm)
    steps = round(duration / case['dt'])
    data = dict(time=[], dipole=[], iterations=[], endpoint_residual=[], solver_residual=[],
                post_projection_endpoint_residual=[], projection_norm=[],
                diagnostic_time=[], energy=[], density_displacement=[], orthogonality_error=[])
    initial_X = sim.overlap_half @ dm @ sim.overlap_half
    failure = None
    started = time.perf_counter()
    for step in range(steps + 1):
        data['time'].append(step * case['dt'])
        data['dipole'].append(moment(dm))
        if step % max(1, round(1. / case['dt'])) == 0 or step == steps:
            data['diagnostic_time'].append(step * case['dt'])
            data['energy'].append(float(sim.energy_tot(dm_total=dm)))
            data['density_displacement'].append(float(np.linalg.norm(sim.overlap_half @ dm @ sim.overlap_half - initial_X)))
            data['orthogonality_error'].append(float(np.linalg.norm(sim.mo_coeff.conj().swapaxes(-1, -2) @ sim.overlap @ sim.mo_coeff - np.eye(8))))
        if step == steps:
            break
        C0, F0 = sim.mo_coeff.copy(), F.copy()
        recent_C.clear()
        recent_F.clear()
        try:
            sim.propagate_electrons(F, 0., dt=case['dt'])
            niter = sim.pt_cn_iterations if case['method'] == 'pt_cn' else sim.pc_iterations
            if case['method'] != 'pt_cn' and not sim.pc_converged:
                raise RuntimeError('Self-consistent CN did not converge.')
            F1 = sim.F.copy()
            residual, post, correction = checked_residuals(sim, C0, F0, case['dt'], case['method'], recent_C, recent_F)
            if not np.isfinite(residual) or residual > 1e-10:
                raise RuntimeError(f'Common endpoint residual exceeded tolerance: {residual:.3e}.')
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            message = str(error)
            failure = dict(step=step + 1, attempted_time_au=(step + 1) * case['dt'],
                           iterations=ITERATIONS if 'did not converge' in message else int(getattr(sim, 'pt_cn_iterations' if case['method'] == 'pt_cn' else 'pc_iterations', 0)), message=message)
            sim.mo_coeff, sim.F = C0, F0
            break
        sim.t = (step + 1) * case['dt']
        F, dm = F1, sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
        data['iterations'].append(niter)
        data['endpoint_residual'].append(residual)
        data['post_projection_endpoint_residual'].append(post)
        data['projection_norm'].append(correction)
        data['solver_residual'].append(float(sim.pt_cn_residual) if case['method'] == 'pt_cn' else -1.)
        if step % max(1, round(20. / case['dt'])) == 0:
            write_path = folder / 'progress.json'
            write_path.write_text(json.dumps(dict(time_au=sim.t, wall_seconds=time.perf_counter() - started)))
    elapsed = time.perf_counter() - started
    arrays = {key: np.asarray(value) for key, value in data.items()}
    arrays.update(initial_coefficients=initial_C, final_coefficients=sim.mo_coeff)
    raw = folder / 'trajectory.npz'
    with raw.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    record = dict(case, kappa=kappa, duration_au=duration, completed=failure is None, failure=failure,
                  final_time_au=data['time'][-1], completed_steps=len(data['iterations']), wall_seconds=elapsed,
                  fock_builds=count['fock_builds'], mean_iterations=float(np.mean(data['iterations'])) if data['iterations'] else None,
                  max_iterations=max(data['iterations'], default=0), max_endpoint_residual=max(data['endpoint_residual'], default=0.),
                  max_post_projection_endpoint_residual=max(data['post_projection_endpoint_residual'], default=0.),
                  max_projection_norm=max(data['projection_norm'], default=0.),
                  max_orthogonality_error=max(data['orthogonality_error']),
                  max_energy_drift_hartree=float(max(abs(arrays['energy'] - arrays['energy'][0]))),
                  native_dipole_error_au=native_error, legacy_dipole_error_au=legacy_error,
                  trajectory_sha256=digest(raw), host=platform.node(), job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(folder / 'result.json', record)
    print(json.dumps(record), flush=True)
    return record


def run(root, number, pilot=False):
    verify(root)
    case = json.loads((root / 'cases.json').read_text())[number]
    if hasattr(os, 'sched_getaffinity') and len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Exactly one allocated CPU is required.')
    for index, kappa in enumerate(KICKS):
        folder = root / ('pilot' if pilot else 'runs') / f'case_{number:03d}' / f'kick_{index}'
        folder.mkdir(parents=True, exist_ok=False)
        propagate(root, case, kappa, PILOT_DURATION if pilot else DURATION, folder)


def pilot_summary(root):
    verify(root)
    records, estimates = [], []
    for case in cases():
        group = [json.loads((root / 'pilot' / f"case_{case['case_id']:03d}" / f'kick_{i}' / 'result.json').read_text()) for i in range(3)]
        complete = all(r['completed'] for r in group)
        estimate = sum(r['wall_seconds'] * DURATION / PILOT_DURATION for r in group) if complete else None
        estimates.append(estimate or 0)
        records.append(dict(case, pilot_completed=complete, estimated_main_group_hours=estimate / 3600 if estimate else None,
                            pilots=group))
    summary = dict(records=records, estimated_total_cpu_hours=sum(estimates) / 3600,
                   estimated_longest_group_hours=max(estimates) / 3600,
                   caveat='Extrapolated from 30 au; physical instability or changed iteration counts can increase cost.')
    write_json(root / 'pilot_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


def self_check():
    assert len(cases()) == 32 and len(KICKS) * len(cases()) == 96
    assert np.isclose(np.linalg.norm(DIRECTION), 1)
    assert all(np.isclose(DURATION / c['dt'], round(DURATION / c['dt'])) for c in cases())
    # Gauge changes commute with the exact flow, but not with a finite CN step.
    eps_i, eps_a, h = -.7, .4, 5.
    cn = 2 / h * (np.arctan(h * eps_a / 2) - np.arctan(h * eps_i / 2))
    pt = 2 / h * np.arctan(h * (eps_a - eps_i) / 2)
    assert abs(cn - pt) > .1
    for method, expected in (('accelerated', cn), ('pt_cn', pt)):
        sim = TDUKSSimulation.__new__(TDUKSSimulation)
        sim.overlap = sim.overlap_inv = np.eye(2)
        sim.mo_coeff = np.array([[[np.sqrt(1 - 1e-10)], [1e-5]]] * 2, dtype=complex)
        sim.mo_occ = np.ones((2, 1))
        sim.n_virtual, sim.mu, sim.t, sim.dt = 0, 1., 0., h
        sim.imaginary_time, sim.count = False, False
        sim.propagation_method = method
        sim.convergence_threshold, sim.max_iterations = 1e-13, ITERATIONS
        sim.electric_field = lambda t: 0.
        F = np.array([np.diag([eps_i, eps_a])] * 2)
        sim.get_fock_matrix = lambda dm, field=0.: F.copy()
        C0 = sim.mo_coeff.copy()
        sim.propagate_electrons(F, 0.)
        phase = -np.angle(sim.mo_coeff[0, 1, 0] / sim.mo_coeff[0, 0, 0]) / h
        aliased = -np.angle(np.exp(-1j * expected * h)) / h
        assert abs(phase - aliased) < 1e-8
        assert endpoint_residual(sim, C0, sim.mo_coeff, F, F, h, method) < 1e-11
    sim.overlap = sim.overlap_inv = np.eye(4)
    generator = np.random.default_rng(19).normal(size=(4, 4))
    generator += generator.T
    C0 = expm(-.01j * generator)[:, :2]
    sim.mo_coeff, sim.mo_occ = np.array([C0, C0]), np.ones((2, 2))
    F = np.array([np.diag([-.7, -.3, .1, .4])] * 2)
    recent_C, recent_F = deque(maxlen=2), deque(maxlen=2)
    original_density = sim.make_rdm1
    def density(mo_coeff, mo_occ):
        recent_C.append(np.array(mo_coeff).copy())
        return original_density(mo_coeff, mo_occ)
    def fock(dm, field=0.):
        recent_F.append(F.copy())
        return F.copy()
    sim.make_rdm1, sim.get_fock_matrix = density, fock
    initial = sim.mo_coeff.copy()
    sim.propagate_electrons(F, 0.)
    raw, post, correction = checked_residuals(sim, initial, F, h, 'pt_cn', recent_C, recent_F)
    assert raw < 1e-11 and post > 1e-9 and correction > 1e-9
    C = np.array([[1.], [0.]], dtype=complex)
    R = np.array([[.1, .3], [.3, -.2]])
    kicked = expm(-1j * 1e-4 * R) @ C
    assert np.linalg.norm(kicked.conj().T @ kicked - 1) < 1e-14
    print('Independent matrix, kick unitarity, CN/PT dispersion and pre-projection solver checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'initialize', 'pilot', 'run', 'pilot-summary', 'self-check'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    lib.num_threads(1)
    if args.action == 'self-check':
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    elif args.action == 'prepare' and args.source is not None:
        prepare(args.root, args.source)
    elif args.action in ('pilot', 'run') and args.case is not None:
        run(args.root, args.case, args.action == 'pilot')
    elif args.action in ('initialize', 'pilot-summary'):
        globals()[args.action.replace('-', '_')](args.root)
    else:
        parser.error('prepare requires --source; pilot/run require --case')
