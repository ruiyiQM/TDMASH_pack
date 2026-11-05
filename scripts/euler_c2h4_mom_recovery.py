"""Test CN/PT orbital predictors at replayed MOM-SCF failure geometries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import null_space
from pyscf import gto

import euler_c2h4_ehrenfest as study
from compare_c2h4_prefix import load
from tdmash import TDUKSSimulation


def from_occupied(mol, C):
    mf = study.mean_field(mol, 'uks')
    B, Q = study.half_overlap(mol)
    mf.mo_coeff = np.array([np.column_stack((block, Q @ null_space((B @ block).conj().T))) for block in C])
    mf.mo_occ = np.array([np.r_[np.ones(block.shape[1]), np.zeros(mol.nao_nr() - block.shape[1])] for block in C])
    F = mf.get_hcore() + mf.get_veff(dm=np.array([block @ block.conj().T for block in C]))
    mf.mo_energy = np.array([np.diag(full.conj().T @ H @ full).real for full, H in zip(mf.mo_coeff, F)])
    for block in mf.mo_coeff:
        if not np.allclose(block.conj().T @ mol.intor('int1e_ovlp') @ block, np.eye(mol.nao_nr()), atol=1e-8):
            raise RuntimeError('Occupied checkpoint completion is not orthonormal.')
    return mf


def replay(root, number):
    protocol = json.loads((root / 'protocol.json').read_text())
    if 'context_source_root' in protocol:
        source = Path(protocol['context_source_root']) / 'runs' / f'case_{number:03d}'
        metadata = json.loads((source / 'result.json').read_text())['replay']
        raw = source / 'context.npz'
        if hashlib.sha256(raw.read_bytes()).hexdigest() != metadata['context_sha256']:
            raise RuntimeError('Frozen MOM failure context checksum mismatch.')
        initial, _, _ = study.load_initial(root, 'uks')
        result = {key: value for key, value in metadata.items() if not key.startswith('reference_') and key != 'context_sha256'}
        with np.load(raw) as arrays:
            for name in ('previous', 'target', 'reference'):
                result[name + '_mol'] = initial.mol.set_geom_(arrays[name + '_R'], unit='Bohr', inplace=False)
            for name in ('previous', 'reference'):
                result[name + '_C'] = arrays[name + '_C'].copy()
        result['context_reused_from'] = str(source)
        return result
    parent = Path(protocol['mom_source_root'])
    record, trajectory = load(parent, number)
    if record['method'] not in ('mom_follow', 'mom_fixed') or record['completed']:
        raise ValueError('The replay source must be a failed MOM trajectory.')
    initial, masses, _ = study.load_initial(root, 'uks')
    initial_mol, initial_C = initial.mol.copy(), study.occupied(initial)
    checkpoint_time = float(trajectory['sample_time'][-1])
    first = round(checkpoint_time / record['dt'])
    checkpoint_mol = initial.mol.set_geom_(trajectory['positions'][first], unit='Bohr', inplace=False)
    scanner = from_occupied(checkpoint_mol, trajectory['sample_coefficients'][-1]).as_scanner()
    fixed = record['method'] == 'mom_fixed'
    source_mol = initial_mol if fixed else checkpoint_mol
    source_C = initial_C if fixed else trajectory['sample_coefficients'][-1]
    advance = study.attach_mom(scanner, source_mol, source_C)
    dt = record['dt']
    attempted = trajectory['positions'][-1] + dt * trajectory['velocities'][-1] + .5 * dt**2 * trajectory['forces'][-1] / masses[:, None]
    targets = list(trajectory['positions'][first + 1:]) + [attempted]
    previous_mol, previous_C = checkpoint_mol, source_C.copy() if not fixed else trajectory['sample_coefficients'][-1].copy()
    for i, positions in enumerate(targets):
        target_mol = previous_mol.set_geom_(positions, unit='Bohr', inplace=False)
        if not fixed:
            source_mol, source_C = previous_mol, previous_C
        try:
            scanner(target_mol)
            converged = bool(scanner.converged)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            converged = False
        if not converged or i == len(targets) - 1:
            return dict(previous_mol=previous_mol, previous_C=previous_C, target_mol=target_mol,
                        reference_mol=source_mol, reference_C=source_C, dt=dt,
                        checkpoint_time_au=checkpoint_time, attempted_time_au=checkpoint_time + (i + 1) * dt,
                        original_attempted_time_au=float(trajectory['time'][-1] + dt), replay_steps=i + 1,
                        native_replay_failed=not converged, original_method=record['method'],
                        source_trajectory_sha256=record['trajectory_sha256'])
        previous_mol, previous_C = scanner.mol.copy(), study.occupied(scanner).copy()
        if not fixed:
            advance()


def transport(left_mol, C, right_mol):
    B, _ = study.half_overlap(left_mol)
    _, Q = study.half_overlap(right_mol)
    return Q @ B @ C


def predictor(context, method, mu, electronic_dt):
    mol, C = context['previous_mol'], context['previous_C']
    mf = from_occupied(mol, C)
    dt = context['dt']
    count = round(dt / electronic_dt)
    if count < 1 or not np.isclose(count * electronic_dt, dt, rtol=0, atol=1e-12):
        raise ValueError('Predictor electronic steps must cover the nuclear interval.')
    sim = TDUKSSimulation(mf, dt=electronic_dt, nuclei_dt=dt, steps=1, mu=mu,
                         propagation_method=method, ehrenfest=True, initial_velocity=np.zeros_like(mol.atom_coords()),
                         convergence_threshold=1e-9 if method == 'pt_cn' else 1e-10,
                         max_iterations=120, output_filename=os.devnull)
    sim.mo_coeff = C.copy()
    sim.update_mol(.5 * (mol.atom_coords() + context['target_mol'].atom_coords()))
    iterations = 0
    for i in range(count):
        F = sim.get_fock_matrix(sim.make_rdm1(sim.mo_coeff, sim.mo_occ))
        sim.propagate_electrons(F, 0, dt=electronic_dt)
        if method != 'pt_cn' and not sim.pc_converged:
            raise RuntimeError('CN orbital predictor did not converge.')
        iterations += sim.pt_cn_iterations if method == 'pt_cn' else sim.pc_iterations
        sim.t = (i + 1) * electronic_dt
    sim.update_mol(context['target_mol'].atom_coords())
    return sim.make_rdm1(sim.mo_coeff, sim.mo_occ), iterations


def seed_for(context, strategy):
    info = dict(predictor_iterations=0)
    if strategy in ('raw_ao', 'newton_mom'):
        C = context['previous_C']
    elif strategy == 'lowdin_only':
        C = transport(context['previous_mol'], context['previous_C'], context['target_mol'])
    elif strategy in ('cn_predictor', 'pt_predictor', 'pt_alpha10_predictor'):
        method = 'accelerated' if strategy == 'cn_predictor' else 'pt_cn'
        mu = 10. if strategy == 'pt_alpha10_predictor' else 1.
        h = min(.125, context['dt']) if method == 'accelerated' else context['dt']
        if mu == 10:
            h = min(.25, context['dt'])
        density, info['predictor_iterations'] = predictor(context, method, mu, h)
        info.update(predictor_mu=mu, predictor_dt=h)
        return density, info
    else:
        raise ValueError('Unknown MOM seed strategy: ' + strategy)
    return np.array([block @ block.conj().T for block in C]), info


def reference_scf(mf, density=None):
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-8, 200
    mf.kernel(dm0=density)
    norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if not mf.converged or norm > 1e-7:
        refined = mf.newton()
        refined.ah_lindep, refined.ah_conv_tol = 1e-22, 1e-14
        refined.kernel(mf.mo_coeff, mf.mo_occ)
        mf = refined.undo_soscf()
        norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    energy = float(mf.energy_tot(dm=mf.make_rdm1()))
    if not np.isfinite([norm, energy, mf.e_tot]).all() or norm > 1e-7 or abs(energy - mf.e_tot) > 1e-9:
        raise RuntimeError(f'Independent reference gradient did not converge: {norm:.3e}.')
    return mf


def lowdin_gradient(mf):
    sim = TDUKSSimulation(mf, dt=.1, propagation_method='pt_cn', output_filename=os.devnull)
    density = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
    F = sim.get_fock_matrix(density)
    return sim.energy_grad(F, density, lowdin=True)


def target_reference(context, segments=8):
    if not isinstance(segments, int) or segments < 1:
        raise ValueError('The reference needs a positive number of geometry segments.')
    mol, C = context['previous_mol'], context['previous_C']
    start, finish = mol.atom_coords(), context['target_mol'].atom_coords()
    fidelities = []
    for fraction in np.arange(1, segments + 1) / segments:
        target = mol.set_geom_(start + fraction * (finish - start), unit='Bohr', inplace=False)
        guess = transport(mol, C, target)
        mf = study.mean_field(target, 'uks')
        study.attach_mom(mf, mol, C)
        mf = reference_scf(mf, np.array([block @ block.conj().T for block in guess]))
        fidelities.append(study.fidelity(C, study.occupied(mf), gto.intor_cross('int1e_ovlp', mol, target)))
        mol, C = target, study.occupied(mf)
    ground = reference_scf(study.mean_field(mol, 'uks'))
    S = mol.intor('int1e_ovlp')
    ambiguous = min(fidelities) < .99 or study.fidelity(C, study.occupied(ground), S) >= .5 or ground.e_tot > mf.e_tot + 1e-8
    return dict(C=C, energy=float(mf.e_tot), ground_C=study.occupied(ground),
                ambiguous=bool(ambiguous), gradient=lowdin_gradient(mf), minimum_step_fidelity=min(fidelities),
                native_scf_flag=bool(mf.converged), orbital_gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))))


def electronic_errors(mol, C, reference_C):
    density = np.array([block @ block.conj().T for block in C])
    reference = np.array([block @ block.conj().T for block in reference_C])
    B, _ = study.half_overlap(mol)
    difference = density - reference
    dipole = np.einsum('sij,xji->x', difference, mol.intor('int1e_r', comp=3)).real
    return dict(spin_density_frobenius=float(np.linalg.norm(B @ difference @ B)),
                total_dipole_difference_au=float(np.linalg.norm(dipole)))


def correct(context, density, newton=False):
    mf = study.mean_field(context['target_mol'], 'uks')
    study.attach_mom(mf, context['reference_mol'], context['reference_C'])
    cycles = []
    mf.callback = lambda env: cycles.append(1)
    mf.kernel(dm0=density)
    norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if newton and (not mf.converged or norm > 1e-5):
        refined = mf.newton()
        refined.ah_lindep, refined.ah_conv_tol = 1e-22, 1e-14
        refined.kernel(mf.mo_coeff, mf.mo_occ)
        mf = refined.undo_soscf()
    norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if not np.isfinite([norm, mf.e_tot]).all():
        raise RuntimeError('MOM corrector returned nonfinite energy or gradient.')
    return mf, dict(scf_converged=bool(mf.converged), orbital_gradient=norm,
                    qualified=bool(mf.converged and norm <= 1e-5), scf_iterations=len(cycles), energy_hartree=float(mf.e_tot))


def run(root, number):
    study.verify(root)
    context = replay(root, number)
    output = root / 'runs' / f'case_{number:03d}'
    output.mkdir(parents=True, exist_ok=False)
    metadata = {key: value for key, value in context.items() if not key.endswith('_mol') and not key.endswith('_C')}
    np.savez_compressed(output / 'context.npz', previous_R=context['previous_mol'].atom_coords(),
                        previous_C=context['previous_C'], target_R=context['target_mol'].atom_coords(),
                        reference_R=context['reference_mol'].atom_coords(), reference_C=context['reference_C'])
    metadata['context_sha256'] = hashlib.sha256((output / 'context.npz').read_bytes()).hexdigest()
    try:
        protocol = json.loads((root / 'protocol.json').read_text())
        if 'reference_source_root' in protocol:
            source = Path(protocol['reference_source_root']) / 'runs' / f'case_{number:03d}'
            previous = json.loads((source / 'result.json').read_text())['replay']
            raw = source / 'reference.npz'
            if hashlib.sha256(raw.read_bytes()).hexdigest() != previous['reference_sha256']:
                raise RuntimeError('Frozen target reference checksum mismatch.')
            with np.load(raw) as arrays:
                reference = dict(C=arrays['target_C'].copy(), ground_C=arrays['ground_C'].copy(),
                                 gradient=arrays['gradient'].copy(), energy=float(arrays['energy']))
            metadata.update(reference_reused_from=str(source), source_reference_sha256=previous['reference_sha256'])
            reference.update(ambiguous=previous['reference_ambiguous'], minimum_step_fidelity=previous['reference_minimum_step_fidelity'],
                             native_scf_flag=previous['reference_native_scf_flag'], orbital_gradient=previous['reference_orbital_gradient'])
        else:
            reference = target_reference(context)
        metadata['reference_ambiguous'] = reference['ambiguous']
        metadata['reference_minimum_step_fidelity'] = reference['minimum_step_fidelity']
        metadata['reference_native_scf_flag'] = reference['native_scf_flag']
        metadata['reference_orbital_gradient'] = reference['orbital_gradient']
        np.savez_compressed(output / 'reference.npz', target_C=reference['C'], ground_C=reference['ground_C'],
                            gradient=reference['gradient'], energy=reference['energy'])
        metadata['reference_sha256'] = hashlib.sha256((output / 'reference.npz').read_bytes()).hexdigest()
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        reference = None
        metadata['reference_failure'] = str(error)[:400]
    rows = []
    strategies = ('raw_ao', 'lowdin_only', 'cn_predictor', 'pt_predictor', 'pt_alpha10_predictor', 'newton_mom')
    for name in strategies:
        started = time.perf_counter()
        row = dict(strategy=name, predictor_iterations=0)
        try:
            seed, info = seed_for(context, name)
            row.update(info)
            row['predictor_seconds'] = time.perf_counter() - started
            mf, status = correct(context, seed, name == 'newton_mom')
            row.update(status)
            if reference is not None:
                S = mf.mol.intor('int1e_ovlp')
                row.update(electronic_errors(mf.mol, study.occupied(mf), reference['C']))
                row.update(target_fidelity=study.fidelity(study.occupied(mf), reference['C'], S),
                           ground_fidelity=study.fidelity(study.occupied(mf), reference['ground_C'], S),
                           target_energy_difference_hartree=abs(float(mf.e_tot) - reference['energy']))
                if row['qualified']:
                    row['force_rms_hartree_per_bohr'] = float(np.sqrt(np.mean((lowdin_gradient(mf) - reference['gradient'])**2)))
                row['faithful'] = bool(row['qualified'] and not reference['ambiguous'] and row['target_fidelity'] >= .99
                                       and row['target_energy_difference_hartree'] <= 1e-5
                                       and row['spin_density_frobenius'] <= 1e-4 and row['total_dipole_difference_au'] <= 1e-4
                                       and row.get('force_rms_hartree_per_bohr', np.inf) <= 1e-4)
        except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError) as error:
            row.update(qualified=False, failure=str(error)[:400])
        row['total_seconds'] = time.perf_counter() - started
        rows.append(row)
        print(json.dumps(row), flush=True)
    baseline_failed = not rows[0].get('qualified', False)
    for row in rows:
        row['confirmed_rescue'] = bool(baseline_failed and row.get('faithful', False))
        row['confirmed_branch_rescue'] = bool(not baseline_failed and not rows[0].get('faithful', False) and row.get('faithful', False))
    result = dict(source_case=number, replay=metadata, baseline_failure_reproduced=baseline_failed,
                  acceptance='Baseline MOM must fail; corrected SCF gradient <=1e-5, target fidelity >=0.99, energy difference <=1e-5 Ha, force RMS <=1e-4 Ha/Bohr, spin density and dipole differences <=1e-4; same MOM occupation reference for all seeds.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=rows)
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)


def self_check():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    _, Q = study.half_overlap(mol)
    C = np.stack([Q[:, :1]] * 2)
    mf = from_occupied(mol, C)
    assert np.linalg.norm(mf.make_rdm1() - np.array([block @ block.conj().T for block in C])) < 1e-12
    positions = mol.atom_coords()
    positions[1, 0] += .001
    target = mol.set_geom_(positions, unit='Bohr', inplace=False)
    context = dict(previous_mol=mol, previous_C=C, target_mol=target, dt=.04)
    for method in ('accelerated', 'pt_cn'):
        density, count = predictor(context, method, 1., .02)
        assert np.isfinite(density).all() and count >= 2
        assert np.allclose(np.einsum('sij,ji->s', density, target.intor('int1e_ovlp')), [1, 1], atol=1e-11)
    ground = study.converge(study.mean_field(mol, 'uks'))
    expected_gradient = lowdin_gradient(ground)
    ground.mo_coeff = ground.mo_coeff.astype(complex)
    assert np.allclose(lowdin_gradient(ground), expected_gradient, rtol=0, atol=1e-11)
    C = study.occupied(ground)
    context.update(previous_C=C, reference_mol=mol, reference_C=C.copy())
    for strategy in ('raw_ao', 'lowdin_only', 'cn_predictor', 'pt_predictor', 'pt_alpha10_predictor', 'newton_mom'):
        seed, _ = seed_for(context, strategy)
        assert seed.shape == (2, mol.nao_nr(), mol.nao_nr()) and np.isfinite(seed).all()
    moved = transport(mol, C, target)
    _, status = correct(context, np.array([block @ block.conj().T for block in moved]))
    assert status['qualified']
    assert np.array_equal(context['reference_C'], C)
    assert target_reference(context)['ambiguous']
    rotated = C.astype(complex) * np.exp(.37j)
    errors = electronic_errors(mol, rotated, C)
    assert max(errors.values()) < 1e-13
    changed = C.copy()
    changed[0, :, 0] = np.cos(.01) * C[0, :, 0] + np.sin(.01) * ground.mo_coeff[0, :, 1].real
    errors = electronic_errors(mol, changed, C)
    assert errors['spin_density_frobenius'] > 1e-3 and errors['total_dipole_difference_au'] > 1e-5
    print('Checkpoint completion, predictors, fixed-reference MOM correction and ground-state ambiguity checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', type=int)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is None or args.case is None:
        parser.error('--root and --case are required')
    else:
        run(args.root, args.case)
