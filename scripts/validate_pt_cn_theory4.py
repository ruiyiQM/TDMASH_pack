"""Local tests of PT-CN alpha*dt scaling and unrestricted DeltaSCF stability."""

import argparse
from collections import deque
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import time

import numpy as np
from pyscf import lib
from pyscf.tdscf.uhf import get_ab
from scipy.linalg import eig, eigh, expm

import euler_c2h4_ehrenfest as study
from euler_c2h4_cayley import apply_kick
from euler_c2h4_mom_recovery import from_occupied, reference_scf
from compare_c2h4_prefix import load
import probe_c2h4_pt_solver as probe
from tdmash import TDUKSSimulation
from collect_c2h4_cayley import HOST, OPTIONS

TAUS = (5., 10., 20., 30., 40., 50.)
ALPHAS = (1., 2., 5., 10.)
TOLERANCE = 1e-6
REPLAY_SOURCE = '/cluster/scratch/zhourui/tdmash_validation/c2h4_dscf_pt_20261102_qCKJNK'
WINDOW_SOURCE = '/cluster/scratch/zhourui/tdmash_validation/c2h4_theory4_inputs_20261104_aaw60ky1/frequency_window.npz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, allow_nan=False, separators=(',', ':'))


def verify_inputs(root):
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if sha(root / name) != expected:
            raise RuntimeError('Checkpoint input changed: ' + name)
    return manifest


def fetch_inputs(root):
    """Stage immutable checkpoints only in an empty temporary directory."""
    root = root.resolve()
    if not root.is_relative_to(Path('/private/tmp')) or (root.exists() and any(root.iterdir())):
        raise ValueError('Checkpoint inputs require an empty directory below /private/tmp.')
    root.mkdir(parents=True, exist_ok=True)
    def fetch(names, destination):
        subprocess.run(['scp', *OPTIONS, *[HOST + ':' + REPLAY_SOURCE + '/' + name for name in names],
                        str(destination)], check=True, timeout=120)
    fetch(['initial.npz', 'initial.json', 'protocol.json'], root)
    for number in (0, 1, 3):
        folder = root / 'runs' / f'case_{number:03d}'
        recovered = root / 'recovered/runs' / f'case_{number:03d}'
        folder.mkdir(parents=True);recovered.mkdir(parents=True)
        fetch([f'runs/case_{number:03d}/result.json', f'runs/case_{number:03d}/trajectory.npz'], folder)
        fetch([f'recovered/runs/case_{number:03d}/result.json'], recovered)
        load(root, number)
    initial = json.loads((root / 'initial.json').read_text())
    if sha(root / 'initial.npz') != initial['sha256']:
        raise RuntimeError('Frozen initial-state checksum mismatch.')
    subprocess.run(['scp', *OPTIONS, HOST + ':' + WINDOW_SOURCE, str(root / 'frequency_window.npz')],
                   check=True, timeout=120)
    if sha(root / 'frequency_window.npz') != '8dbb4106e963c337a2250125cb37c090fc68e839992858b4cff3e2674d47018f':
        raise RuntimeError('Frozen dipole-window checksum mismatch.')
    manifest = dict(source_root=REPLAY_SOURCE,
                    files_sha256={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()})
    write(root / 'source_manifest.json', manifest)
    print(json.dumps(dict(temporary_inputs=str(root), checked=True)), flush=True)


def simulation(mf, C=None, alpha=1., dt=1.):
    sim = TDUKSSimulation(mf.copy(), dt=dt, mu=alpha, propagation_method='pt_cn',
                         convergence_threshold=TOLERANCE, max_iterations=120,
                         field_type='none', ehrenfest=False, output_filename=os.devnull)
    if C is not None:
        sim.mo_coeff = C.copy()
    return sim


def metric_norm(B, value):
    return float(np.linalg.norm(B @ value))


def action_metrics(B, A0, A1):
    start, finish = metric_norm(B, A0), metric_norm(B, A1)
    resolved = start > 1e-7
    inner = np.vdot(B @ A0, B @ A1) / (start * finish) if resolved and finish > 1e-14 else None
    return dict(start_norm=start, end_norm=finish, initial_action_resolved=resolved,
                norm_ratio=finish / start if resolved else None,
                anti_action_relative=metric_norm(B, A0 + A1) / (start + finish) if resolved else None,
                correlation_real=float(inner.real) if inner is not None else None,
                correlation_imag=float(inner.imag) if inner is not None else None,
                phase_deg=float(np.angle(inner, deg=True)) if inner is not None else None)


def solve_step(sim):
    """Capture the actual production iterates, separating the final projection."""
    C0, alpha, dt = np.array(sim.mo_coeff).copy(), sim.mu, sim.dt
    F0 = sim.get_fock_matrix(sim.make_rdm1(C0, sim.mo_occ))
    A0 = probe.action(sim, C0, F0) / alpha
    rhs = C0 - .5j * dt * probe.action(sim, C0, F0)
    last, recent, trace = [None], deque(maxlen=2), []
    density, fock = sim.make_rdm1, sim.get_fock_matrix
    def tracked_density(mo_coeff, mo_occ):
        last[0] = np.array(mo_coeff).copy()
        return density(mo_coeff, mo_occ)
    def tracked_fock(*args, **kwargs):
        F = fock(*args, **kwargs)
        C = last[0]
        trace.append(float(np.linalg.norm(rhs - C - .5j * dt * probe.action(sim, C, F))))
        recent.append((C.copy(), F.copy()))
        return F
    sim.make_rdm1, sim.get_fock_matrix = tracked_density, tracked_fock
    started, failure = time.perf_counter(), None
    try:
        sim.propagate_electrons(F0, 0., dt=dt)
        completed = True
        Cpre, Fpre = recent[0]
        raw_trace, post_residual = trace[:-1], trace[-1]
        if len(raw_trace) != sim.pt_cn_iterations:
            raise RuntimeError('The captured trial history does not match the solver iteration count.')
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        completed, failure = False, str(error)
        Cpre, Fpre = recent[-1] if recent else (C0, F0)
        raw_trace, post_residual = trace, None
    Apre = probe.action(sim, Cpre, Fpre) / alpha
    tau = alpha * dt
    identity = Apre + A0 - 2j / tau * (Cpre - C0)
    record = dict(alpha=alpha, dt=dt, tau=tau, completed=completed, failure=failure,
                  iterations=len(raw_trace), residual_trace=raw_trace,
                  final_raw_residual=raw_trace[-1] if raw_trace else None,
                  post_projection_residual=post_residual, seconds=time.perf_counter() - started,
                  pre_projection=action_metrics(sim.overlap_half, A0, Apre),
                  endpoint_identity_error=metric_norm(sim.overlap_half, identity),
                  pre_projection_gram_error=float(np.linalg.norm(Cpre.conj().swapaxes(-1, -2) @ sim.overlap @ Cpre - np.eye(Cpre.shape[-1]))))
    Cpost = np.array(sim.mo_coeff).copy() if completed else None
    if completed:
        Apost = probe.action(sim, Cpost, sim.F) / alpha
        record.update(post_projection=action_metrics(sim.overlap_half, A0, Apost),
                      projection_norm=metric_norm(sim.overlap_half, Cpost - Cpre))
    return record, dict(C0=C0, F0=F0, Cpre=Cpre, Fpre=Fpre, Cpost=Cpost,
                        overlap_inv=sim.overlap_inv, B=sim.overlap_half)


def scaling_check(rows, states):
    reference = states[0]
    C0, trial, F0, Ftrial = (reference[key] for key in ('C0', 'Cpre', 'F0', 'Fpre'))
    residuals = []
    for row in rows:
        stub = SimpleNamespace(mu=row['alpha'], overlap_inv=reference['overlap_inv'])
        residuals.append(trial - C0 + .5j * row['dt'] *
                         (probe.action(stub, C0, F0) + probe.action(stub, trial, Ftrial)))
    trace_differences, density_differences = [], []
    for row, state in zip(rows[1:], states[1:]):
        size = min(len(rows[0]['residual_trace']), len(row['residual_trace']))
        left, right = np.array(rows[0]['residual_trace'][:size]), np.array(row['residual_trace'][:size])
        trace_differences.append(float(np.linalg.norm(right - left) / max(np.linalg.norm(left), 1e-15)))
        if reference['Cpost'] is not None and state['Cpost'] is not None:
            D0 = reference['Cpost'] @ reference['Cpost'].conj().swapaxes(-1, -2)
            D1 = state['Cpost'] @ state['Cpost'].conj().swapaxes(-1, -2)
            density_differences.append(float(np.linalg.norm(reference['B'] @ (D1 - D0) @ reference['B'])))
    return dict(shared_trial_equation_max_difference=max(float(np.linalg.norm(R - residuals[0])) for R in residuals),
                same_outcome=len({row['completed'] for row in rows}) == 1,
                iterations_min=min(row['iterations'] for row in rows), iterations_max=max(row['iterations'] for row in rows),
                max_relative_trace_difference=max(trace_differences, default=0.),
                max_accepted_density_difference=max(density_differences, default=None))


def contexts(root):
    original, data = load(root, 0)
    index = int(np.flatnonzero(np.isclose(data['sample_time'], 40.))[0])
    R = data['positions'][round(40 / original['dt'])]
    initial = study.load_initial(root, 'uks')[0]
    early_mol = initial.mol.set_geom_(R, unit='Bohr', inplace=False)
    result = [dict(name='early_t40', time_au=40., mf=from_occupied(early_mol, data['sample_coefficients'][index]),
                   C=data['sample_coefficients'][index].copy(), source_case=0, geometry='saved endpoint')]
    for number in (0, 1):
        sim, record, start_time = probe.state(root, number)
        sim.calculate_forces(lowdin=True)
        v_half = sim.velocities + .5 * sim.dt * sim.force / sim.mass[:, None]
        sim.update_mol(sim.positions + .5 * sim.dt * v_half)
        result.append(dict(name=f'failure_dt{record["dt"]:g}', time_au=start_time, mf=sim.mf,
                           C=sim.mo_coeff.copy(), source_case=number, geometry='replayed failing electronic midpoint'))
    return result


def tau_matrix(root):
    records, checks, controls = [], [], []
    for context in contexts(root):
        ground = reference_scf(study.mean_field(context['mf'].mol.copy(), 'uks'))
        blank, _ = solve_step(simulation(ground, alpha=1., dt=50.))
        controls.append(dict(context=context['name'], state='stationary_ground', **blank))
        kicked = simulation(ground)
        apply_kick(kicked, 1e-3, ground.mol.intor('int1e_r', comp=3))
        for name, mf, C in (('ground_kick_1e-3', ground, kicked.mo_coeff),
                            ('evolved_dscf', context['mf'], context['C'])):
            for tau in TAUS:
                group, states = [], []
                for alpha in ALPHAS:
                    row, state = solve_step(simulation(mf, C, alpha, tau / alpha))
                    row.update(context=context['name'], source_case=context['source_case'],
                               snapshot_time_au=context['time_au'], geometry=context['geometry'], state=name)
                    records.append(row); group.append(row); states.append(state)
                checks.append(dict(context=context['name'], state=name, tau=tau, **scaling_check(group, states)))
            print(json.dumps(dict(stage='tau', context=context['name'], state=name,
                                  completed=sum(row['completed'] for row in records if row['context'] == context['name'] and row['state'] == name))), flush=True)
    return dict(records=records, equivalence_checks=checks, stationary_controls=controls)


def semicanonical(mf):
    mf = mf.copy()
    density = mf.make_rdm1()
    F = mf.get_hcore() + mf.get_veff(dm=density)
    C, energies = np.array(mf.mo_coeff).real.copy(), np.array(mf.mo_energy).copy()
    for spin in (0, 1):
        for occupied in (True, False):
            indices = np.flatnonzero((mf.mo_occ[spin] > .5) == occupied)
            block = C[spin][:, indices]
            energies[spin, indices], rotation = eigh(block.T @ F[spin] @ block)
            C[spin][:, indices] = block @ rotation
    mf.mo_coeff, mf.mo_energy = C, energies
    if np.linalg.norm(mf.make_rdm1() - density) > 1e-10:
        raise RuntimeError('Semicanonicalization changed the reference density.')
    return mf


def full_ab(mf):
    a, b = get_ab(mf)
    def blocks(parts, hermitian):
        aa, ab, bb = (part.reshape(part.shape[0] * part.shape[1], -1) for part in parts)
        return np.block([[aa, ab], [ab.conj().T if hermitian else ab.T, bb]])
    A, B = blocks(a, True), blocks(b, False)
    labels = []
    for spin in (0, 1):
        for i in np.flatnonzero(mf.mo_occ[spin] == 1):
            for a in np.flatnonzero(mf.mo_occ[spin] == 0):
                labels.append(dict(spin=spin, occupied=int(i), virtual=int(a),
                                   gap=float(mf.mo_energy[spin, a] - mf.mo_energy[spin, i])))
    return A, B, labels


def verify_ab(mf, A, B, labels, permutation=False):
    rng = np.random.default_rng(20261104)
    gaps = np.array([item['gap'] for item in labels])
    errors = []
    for kind in ('real', 'imaginary', 'alpha', 'beta'):
        z = rng.normal(size=len(labels)) + 1j * rng.normal(size=len(labels))
        if kind == 'real': z = z.real.astype(complex)
        if kind == 'imaginary': z = 1j * z.imag
        if kind in ('alpha', 'beta'):
            z[[item['spin'] != (0 if kind == 'alpha' else 1) for item in labels]] = 0
        z /= np.linalg.norm(z)
        expected = A @ z + B @ z.conj() - gaps * z
        for epsilon in (1e-4, 5e-5):
            focks = []
            for sign in (1, -1):
                densities = []
                for spin in (0, 1):
                    generator = np.zeros((mf.mol.nao_nr(),) * 2, complex)
                    for value, item in zip(z, labels):
                        if item['spin'] == spin:
                            i, a = item['occupied'], item['virtual']
                            generator[a, i], generator[i, a] = value, -value.conjugate()
                    orbitals = mf.mo_coeff[spin] @ expm(sign * epsilon * generator)
                    occupied = orbitals[:, mf.mo_occ[spin] > .5]
                    densities.append(occupied @ occupied.conj().T)
                focks.append(mf.get_hcore() + mf.get_veff(dm=np.array(densities)))
            derivative = (focks[0] - focks[1]) / (2 * epsilon)
            actual = np.array([mf.mo_coeff[item['spin']][:, item['virtual']].conj() @ derivative[item['spin']]
                               @ mf.mo_coeff[item['spin']][:, item['occupied']] for item in labels])
            error = float(np.linalg.norm(actual - expected) / max(1., np.linalg.norm(expected)))
            errors.append(dict(direction=kind, epsilon=epsilon, relative_action_error=error))
    result = dict(finite_difference_checks=errors, max_finite_difference_error=max(row['relative_action_error'] for row in errors))
    if permutation:
        permuted = mf.copy()
        permutations = [rng.permutation(mf.mol.nao_nr()) for _ in (0, 1)]
        permuted.mo_coeff = np.array([mf.mo_coeff[s][:, permutations[s]] for s in (0, 1)])
        permuted.mo_occ = np.array([mf.mo_occ[s][permutations[s]] for s in (0, 1)])
        permuted.mo_energy = np.array([mf.mo_energy[s][permutations[s]] for s in (0, 1)])
        Ap, Bp, pairs = full_ab(permuted)
        lookup = {(item['spin'], item['occupied'], item['virtual']): i for i, item in enumerate(labels)}
        indices = [lookup[(item['spin'], int(permutations[item['spin']][item['occupied']]),
                           int(permutations[item['spin']][item['virtual']]))] for item in pairs]
        result['permutation_error'] = float(max(np.linalg.norm(Ap - A[np.ix_(indices, indices)]),
                                                np.linalg.norm(Bp - B[np.ix_(indices, indices)])) / max(1., np.linalg.norm(A)))
        if result['permutation_error'] > 1e-10:
            raise RuntimeError('The occupation permutation check failed.')
    if result['max_finite_difference_error'] > 2e-6:
        raise RuntimeError('The full response failed independent finite differences.')
    return result


def rpa_spectrum(A, B):
    L = np.block([[A, B], [-B.conj(), -A.conj()]])
    metric = np.r_[np.ones(len(A)), -np.ones(len(A))]
    pseudo_error = np.linalg.norm(L.conj().T * metric[None, :] - metric[:, None] * L) / max(1., np.linalg.norm(L))
    values, vectors = eig(L)
    residual = np.linalg.norm(L @ vectors - vectors * values[None, :], axis=0) / np.maximum(1., abs(values))
    norms = np.sum(metric[:, None] * abs(vectors)**2, axis=0).real
    noise = max(1e-6, 100 * float(max(residual)))
    if pseudo_error > 1e-10 or max(residual) > 1e-8:
        raise RuntimeError('Dense RPA diagonalization failed its algebraic checks.')
    return values, vectors, norms, dict(dimension=len(L), eigenvalues_computed=len(values),
        pseudo_hermiticity_error=float(pseudo_error), max_eigenpair_residual=float(max(residual)),
        imaginary_noise_threshold=noise, growing_count=int(np.sum(values.imag > noise)),
        max_growth_rate=float(max(values.imag)), negative_norm_positive_real_frequency_count=int(np.sum((values.real > noise) & (abs(values.imag) <= noise) & (norms < -1e-7))))


def mode_report(index, values, vectors, norms, labels, mf):
    vector, value = vectors[:, index], values[index]
    size = len(labels)
    X, Y = vector[:size], vector[size:]
    weights = abs(X)**2 + abs(Y)**2
    components = []
    for i in np.argsort(-weights)[:10]:
        components.append(dict(labels[i], weight=float(weights[i]), x_weight=float(abs(X[i])**2),
                               y_weight=float(abs(Y[i])**2), negative_gap=labels[i]['gap'] < 0))
    densities = []
    Bmetric, _ = study.half_overlap(mf.mol)
    for spin in (0, 1):
        keep = [i for i, item in enumerate(labels) if item['spin'] == spin]
        Co, Cv = mf.mo_coeff[spin][:, mf.mo_occ[spin] == 1], mf.mo_coeff[spin][:, mf.mo_occ[spin] == 0]
        x, y = X[keep].reshape(Co.shape[1], Cv.shape[1]), Y[keep].reshape(Co.shape[1], Cv.shape[1])
        densities.append(Bmetric @ (Cv @ x.T @ Co.T + Co @ y @ Cv.T) @ Bmetric)
    return dict(omega_real=float(value.real), omega_imag=float(value.imag), j_norm=float(norms[index]),
                krein_norm_sign='null/complex' if abs(value.imag) > 1e-6 else ('positive' if norms[index] > 1e-7 else 'negative' if norms[index] < -1e-7 else 'near_zero'),
                energy_form=float((value * norms[index]).real),
                alpha_weight=float(sum(weights[i] for i, item in enumerate(labels) if item['spin'] == 0)),
                beta_weight=float(sum(weights[i] for i, item in enumerate(labels) if item['spin'] == 1)),
                charge_density_norm=float(np.linalg.norm(densities[0] + densities[1])),
                longitudinal_spin_density_norm=float(np.linalg.norm(densities[0] - densities[1])),
                alpha_beta_phase_deg=float(np.angle(np.vdot(densities[0], densities[1]), deg=True)),
                leading_pairs=components)


def analyze_reference(mf, label, check_permutation=False):
    started = time.perf_counter()
    mf = semicanonical(mf)
    A, B, pairs = full_ab(mf)
    validation = verify_ab(mf, A, B, pairs, check_permutation)
    values, vectors, norms, statistics = rpa_spectrum(A, B)
    growing = [i for i, value in enumerate(values) if value.imag > statistics['imaginary_noise_threshold'] and value.real >= -statistics['imaginary_noise_threshold']]
    anomalous = sorted((i for i, value in enumerate(values) if value.real > 0 and abs(value.imag) < statistics['imaginary_noise_threshold'] and norms[i] < -1e-7), key=lambda i: values[i].real)[:8]
    record = dict(label=label, **statistics, validation=validation,
                  orbital_gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))),
                  energy=float(mf.e_tot), spin_square=float(mf.spin_square()[0]),
                  negative_gap_count=sum(item['gap'] < 0 for item in pairs),
                  negative_gap_pairs=[item for item in pairs if item['gap'] < 0],
                  growing_modes=[mode_report(i, values, vectors, norms, pairs, mf) for i in growing],
                  anomalous_real_modes=[mode_report(i, values, vectors, norms, pairs, mf) for i in anomalous],
                  spectrum_columns=['omega_real', 'omega_imag', 'j_norm'],
                  spectrum=np.column_stack((values.real, values.imag, norms)).tolist(), seconds=time.perf_counter() - started)
    print(json.dumps(dict(stage='rpa', label=label, seconds=record['seconds'], growing_count=record['growing_count'], max_growth_rate=record['max_growth_rate'])), flush=True)
    return record


def stationary_references(root):
    _, trajectory = load(root, 3)
    initial = study.load_initial(root, 'uks')[0]
    frozen = analyze_reference(initial, 'initial_frozen_dscf', True)
    source_mol, source_C = initial.mol, study.occupied(initial)
    records, branch = [], []
    for t in (0., 40., 80., 116.):
        R = trajectory['positions'][round(t)]
        start = source_mol.atom_coords()
        segments = max(1, int(np.ceil(np.max(np.linalg.norm(R - start, axis=1)) / .015)))
        for fraction in np.arange(1, segments + 1) / segments:
            mol = source_mol.set_geom_(start + fraction * (R - start), unit='Bohr', inplace=False)
            B0, _ = study.half_overlap(source_mol)
            _, Q1 = study.half_overlap(mol)
            guess = Q1 @ B0 @ source_C
            target = study.mean_field(mol, 'uks')
            study.attach_mom(target, source_mol, source_C)
            target = reference_scf(target, np.array([C @ C.conj().T for C in guess]).real)
            fidelity = study.fidelity(guess, study.occupied(target), mol.intor('int1e_ovlp'))
            if fidelity < .9:
                raise RuntimeError('MOM target continuation switched branch.')
            source_mol, source_C = mol, study.occupied(target)
        ground = reference_scf(study.mean_field(source_mol.copy(), 'uks'))
        for name, mf in (('ground', ground), ('dscf_target', target)):
            record = analyze_reference(mf, f'{name}_t{t:g}', t == 0)
            record.update(state=name, snapshot_time_au=t)
            records.append(record)
        sample = np.flatnonzero(np.isclose(trajectory['sample_time'], t))
        branch.append(dict(time_au=t, target_energy=float(target.e_tot), ground_energy=float(ground.e_tot),
                           excitation_hartree=float(target.e_tot - ground.e_tot),
                           trajectory_target_fidelity=study.fidelity(trajectory['sample_coefficients'][sample[0]], source_C, source_mol.intor('int1e_ovlp')) if len(sample) else None,
                           continuation_last_segment_fidelity=float(fidelity)))
    return dict(frozen_initial=frozen, references=records, target_continuation=branch,
                spin_scope='Collinear spin-conserving alpha-alpha and beta-beta orbital transitions with full alpha-beta density coupling; transverse spin-flip rotations are not in this basis.',
                krein_convention='Euclidean-normalized eigenvector; J norm = X^dagger X - Y^dagger Y. Complex modes are null, not positive/negative. Real-mode energy sign uses omega*J_norm.',
                krein_collision='Complex modes and negative-energy channels alone do not establish a specific parameter-continuation collision.')


def growth_crosscheck(window):
    if window is None or not window.exists():
        return dict(status='pending_frequency_window')
    from analyze_c2h4_cayley import harmonic_inversion
    with np.load(window, allow_pickle=False) as data:
        time_grid, metadata = data['time'], json.loads(str(data['metadata']))
        baseline = data['mu_0']
        small, large = (data['mu_1'] - baseline) / 1e-4, (data['mu_2'] - baseline) / 5e-4
        rows = []
        for kick, kappa in ((1, 1e-4), (2, 5e-4)):
            response = (data[f'mu_{kick}'] - baseline) / kappa
            for duration in (40., 60., 80.):
                keep = time_grid <= duration + 1e-9
                fit = harmonic_inversion(time_grid[keep], response[keep], rank=96)
                modes = [mode for mode in fit['modes'] if .15 < mode['omega'] < .3 and mode['growth_rate'] > .005]
                selected = max(modes, key=lambda mode: mode['amplitude']) if modes else None
                disagreement = float(np.linalg.norm(small[keep] - large[keep]) / max(1e-20, .5 * (np.linalg.norm(small[keep]) + np.linalg.norm(large[keep]))))
                rows.append(dict(kappa=kappa, duration_au=duration, fit_error=fit['relative_fit_error'],
                                 normalized_kick_disagreement=disagreement, linear_kick_check=disagreement < .05, measured=selected))
        return dict(status='measured', source=metadata, window_sha256=sha(window), fits=rows)


def pilot(root, output):
    started = time.perf_counter()
    initial = study.load_initial(root, 'uks')[0]
    reference = analyze_reference(initial, 'pilot_frozen_dscf', True)
    step, _ = solve_step(simulation(initial, alpha=10., dt=4.))
    result = dict(seconds=time.perf_counter() - started, rpa_seconds=reference['seconds'],
                  one_step_seconds=step['seconds'], max_growth_rate=reference['max_growth_rate'],
                  fd_error=reference['validation']['max_finite_difference_error'], permutation_error=reference['validation']['permutation_error'])
    write(output / 'pt_cn_theory4_pilot_20261104.json', result)
    print(json.dumps(result), flush=True)


def run(root, output, window):
    started = time.perf_counter()
    manifest = verify_inputs(root)
    scaling = tau_matrix(root)
    stability = stationary_references(root)
    crosscheck = growth_crosscheck(window)
    code = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), code / 'src/tdmash/PT.py', code / 'scripts/probe_c2h4_pt_solver.py',
               code / 'scripts/euler_c2h4_mom_recovery.py', code / 'scripts/analyze_c2h4_cayley.py']
    result = dict(protocol=dict(independent_study=True, resumes_paused_goal=False, paused_goal_refinements_consumed=0,
                    priority='Frequency-test collection and analysis take precedence; its code and outputs are not modified.',
                    execution='Local single-thread computation; source checkpoint files are temporary inputs.',
                    software={name: version(name) for name in ('numpy', 'scipy', 'pyscf')},
                    taus=TAUS, alphas=ALPHAS, tolerance=TOLERANCE, max_iterations=120,
                    step_count=144, stationary_ground_controls=3,
                    action='Unscaled PT action in the solver endpoint gauge; pre/post orthonormalization reported separately.',
                    ground_control='Stationary blank plus a fixed 1e-3 length-gauge kick in the main matrix.',
                    dscf_control='Unmodified evolved orbitals from the replayed DeltaSCF-initialized trajectories, not new SCF solutions.',
                    rpa='Complete dense scipy.linalg.eig of the full unrestricted response; no Davidson.',
                    source_inputs=manifest, sources_sha256={str(p.relative_to(code)): sha(p) for p in sources}),
                  scaling=scaling, stability=stability, frequency_growth_crosscheck=crosscheck,
                  total_seconds=time.perf_counter() - started)
    write(output / 'pt_cn_theory4_20261104.json', result)
    plot(result, output / 'pt_cn_theory4_20261104.png')
    print(json.dumps(dict(completed=True, seconds=result['total_seconds'], source_step_count=len(scaling['records']))), flush=True)


def plot(result, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    rows = result['scaling']['records']
    for context, style in (('early_t40', '-'), ('failure_dt4', '--'), ('failure_dt5', ':')):
        for state, color in (('ground_kick_1e-3', '#0072B2'), ('evolved_dscf', '#D55E00')):
            selected = [row for row in rows if row['context'] == context and row['state'] == state and row['alpha'] == 1]
            label = context + (' ground kick' if state.startswith('ground') else ' evolved DeltaSCF')
            axes[0, 0].plot([row['tau'] for row in selected], [row['iterations'] for row in selected], style, color=color, marker='o', label=label)
            converged = [row for row in selected if row['completed'] and row['pre_projection']['initial_action_resolved']]
            axes[1, 0].plot([row['tau'] for row in converged], [row['pre_projection']['norm_ratio'] for row in converged], style, color=color, marker='o', label=label)
            axes[1, 1].plot([row['tau'] for row in converged], [row['pre_projection']['correlation_real'] for row in converged], style, color=color, marker='o')
    for row in rows:
        if row['state'] == 'evolved_dscf' and ((row['context'] == 'failure_dt4' and row['tau'] == 40) or (row['context'] == 'failure_dt5' and row['tau'] == 50)):
            axes[0, 1].semilogy(np.arange(1, len(row['residual_trace']) + 1), row['residual_trace'],
                                '--' if row['tau'] == 40 else '-', lw=.9, label=f"tau={row['tau']:g}, alpha={row['alpha']:g}")
    for row in result['stability']['references']:
        spectrum = np.array(row['spectrum'])
        keep = (spectrum[:, 0] > 0) & (spectrum[:, 0] < 1.2)
        axes[2, 0].scatter(spectrum[keep, 0], spectrum[keep, 1], s=9,
                           marker='.' if row['state'] == 'ground' else 'x', label=row['label'], alpha=.7)
    for kappa, marker in ((1e-4, 'o'), (5e-4, 's')):
        fits = [row for row in result['frequency_growth_crosscheck'].get('fits', []) if row['kappa'] == kappa and row['measured']]
        axes[2, 1].plot([row['duration_au'] for row in fits], [row['measured']['growth_rate'] for row in fits], marker=marker, label=f'kappa={kappa:g}')
    gamma = result['stability']['frozen_initial']['max_growth_rate']
    axes[2, 1].axhline(gamma, ls='--', color='black', label='Full RPA Im omega')
    settings = [(r'Effective step $\tau=\alpha\Delta t$', 'Iterations (120 = limit)', 'Production nonlinear solve'),
                ('Iteration', 'Raw AO residual', 'Same-tau residual histories'),
                (r'$\tau$', r'$\|A_{n+1}\|_S/\|A_n\|_S$', 'Converged pre-projection actions'),
                (r'$\tau$', 'Real normalized action overlap', 'Action anti-alignment; endpoint gauge'),
                (r'Re $\omega$ (a.u.)', r'Im $\omega$ (a.u.)', 'Full unrestricted response: ground vs target'),
                ('Fit-window duration (a.u.)', 'Growth rate (a.u.)', 'Independent dipole-growth fits')]
    for axis, (x, y, title) in zip(axes.flat, settings):
        axis.set(xlabel=x, ylabel=y, title=title);axis.grid(alpha=.2)
    axes[1, 0].axhline(1, color='gray', lw=.8)
    axes[1, 0].set_yscale('log')
    axes[1, 1].axhline(-1, color='gray', lw=.8)
    for index in (0, 1, 4, 5):axes.flat[index].legend(fontsize=6.8, ncol=2)
    fig.suptitle('Independent theory test 4: effective step and DeltaSCF stability', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch', 'pilot', 'run'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/validation'))
    parser.add_argument('--frequency-window', type=Path)
    args = parser.parse_args()
    lib.num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'fetch':fetch_inputs(args.source)
    else:
        verify_inputs(args.source)
        if args.action == 'pilot':pilot(args.source, args.output)
        else:run(args.source, args.output, args.frequency_window)
