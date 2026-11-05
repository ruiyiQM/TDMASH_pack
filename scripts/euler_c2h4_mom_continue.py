"""Continue MOM trajectories from the last verified original excited-state frame."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from pyscf import data, gto, lib, md

import euler_c2h4_mom_recovery as mom
from compare_c2h4_prefix import load


class RecoveryFailure(RuntimeError):
    pass


def restart(root, number):
    protocol = json.loads((root / 'protocol.json').read_text())
    parent = Path(protocol['mom_source_root'])
    record, trajectory = load(parent, number)
    if record['method'] not in ('mom_follow', 'mom_fixed'):
        raise ValueError('Continuation requires a MOM source trajectory.')
    diagnostic_path = parent / 'recovered/runs' / f'case_{number:03d}' / 'result.json'
    history = json.loads(diagnostic_path.read_text())['branch_diagnostics']
    valid = [row for row in history if 'failure' not in row and not row['reference_ambiguous']
             and row['target_fidelity'] >= .99 and row['density_frobenius'] <= 1e-4
             and row['force_rms_hartree_per_bohr'] <= 1e-4]
    if not valid:
        raise RecoveryFailure('No verified original-branch checkpoint is available.')
    last = max(valid, key=lambda row: row['time_au'])
    checkpoint_time = last['time_au']
    index = round(checkpoint_time / record['dt'])
    sample = np.flatnonzero(np.isclose(trajectory['sample_time'], checkpoint_time, rtol=0, atol=1e-10))
    if len(sample) != 1:
        raise RecoveryFailure('The verified checkpoint lacks a unique occupied snapshot.')
    initial, masses, _ = mom.study.load_initial(root, 'uks')
    mol = initial.mol.set_geom_(trajectory['positions'][index], unit='Bohr', inplace=False)
    bad = next((row['time_au'] for row in history if row.get('target_fidelity', 1) < .99), None)
    return dict(mol=mol, C=trajectory['sample_coefficients'][sample[0]].copy(),
                velocity=trajectory['velocities'][index].copy(), force=trajectory['forces'][index].copy(),
                masses=masses, dt=record['dt'], time=checkpoint_time, original_method=record['method'],
                fixed_mol=initial.mol.copy(), fixed_C=mom.study.occupied(initial),
                source_case=number, original_failure_time=float(trajectory['time'][-1] + record['dt']),
                original_wrong_branch_time=bad, original_global_diagnostic=last,
                source_trajectory_sha256=record['trajectory_sha256'],
                global_diagnostic_sha256=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest())


def diagnostic(mf, gradient, reference, status):
    C = mom.study.occupied(mf)
    S = mf.mol.intor('int1e_ovlp')
    result = dict(status, **mom.electronic_errors(mf.mol, C, reference['C']),
                  target_fidelity=mom.study.fidelity(C, reference['C'], S),
                  ground_fidelity=mom.study.fidelity(C, reference['ground_C'], S),
                  energy_difference_hartree=abs(float(mf.e_tot) - reference['energy']),
                  force_rms_hartree_per_bohr=float(np.sqrt(np.mean((gradient - reference['gradient'])**2))),
                  reference_ambiguous=reference['ambiguous'], reference_orbital_gradient=reference['orbital_gradient'])
    result['passed'] = bool(status['qualified'] and not reference['ambiguous'] and result['target_fidelity'] >= .99
                            and result['spin_density_frobenius'] <= 1e-4 and result['total_dipole_difference_au'] <= 1e-4
                            and result['energy_difference_hartree'] <= 1e-5 and result['force_rms_hartree_per_bohr'] <= 1e-4)
    return result


def reference_step(context):
    try:
        reference = mom.target_reference(context, segments=1)
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError):
        return mom.target_reference(context, segments=8)
    return mom.target_reference(context, segments=8) if reference['ambiguous'] else reference


class RecoveryScanner(lib.GradScanner):
    def __init__(self, initial, strategy):
        self.mol = initial['mol'].copy()
        self.previous_mol, self.previous_C = self.mol.copy(), initial['C'].copy()
        self.strategy, self.dt, self.start_time = strategy, initial['dt'], initial['time']
        self.fixed = initial['original_method'] == 'mom_fixed' and strategy != 'follow_reference'
        self.fixed_mol, self.fixed_C = initial['fixed_mol'].copy(), initial['fixed_C'].copy()
        self.base = mom.from_occupied(self.previous_mol, self.previous_C)
        self.base.e_tot = float(self.base.energy_tot(dm=self.base.make_rdm1()))
        context = dict(previous_mol=self.previous_mol, previous_C=self.previous_C, target_mol=self.previous_mol)
        self.reference = reference_step(context)
        self.reference_mol, self.reference_C = self.previous_mol.copy(), self.reference['C'].copy()
        norm = float(np.linalg.norm(self.base.get_grad(self.base.mo_coeff, self.base.mo_occ)))
        self.initial_gradient = -initial['force'].copy()
        self.force_restart_difference = float(np.max(abs(mom.lowdin_gradient(self.base) - self.initial_gradient)))
        first = diagnostic(self.base, self.initial_gradient, self.reference,
                           dict(qualified=norm <= 1e-5, orbital_gradient=norm, checkpoint_from_accepted_source=True))
        first.update(time_au=self.start_time, accepted=first['passed'])
        self.diagnostics = [first]
        if not first['passed']:
            raise RecoveryFailure('The restart checkpoint failed the independent original-branch check.')
        self.accepted, self.started, self.accepted_steps = True, False, 0
        self.last_attempt_C, self.last_attempt_R = None, None

    @property
    def converged(self):
        return self.accepted

    def __call__(self, molecule):
        target = molecule.copy()
        if not self.started:
            if not np.allclose(target.atom_coords(), self.previous_mol.atom_coords(), rtol=0, atol=1e-12):
                raise RecoveryFailure('Native NVE did not start at the verified checkpoint.')
            self.started = True
            return self.base.e_tot, self.initial_gradient.copy()
        number = self.accepted_steps + 1
        row = dict(time_au=self.start_time + number * self.dt, strategy=self.strategy, accepted=False)
        self.last_attempt_R = target.atom_coords().copy()
        self.last_attempt_C = None
        try:
            context = dict(previous_mol=self.previous_mol, previous_C=self.previous_C, target_mol=target, dt=self.dt,
                           reference_mol=self.fixed_mol if self.fixed else self.previous_mol,
                           reference_C=self.fixed_C if self.fixed else self.previous_C)
            started = time.perf_counter()
            seed, info = mom.seed_for(context, 'raw_ao' if self.strategy == 'follow_reference' else self.strategy)
            row.update(info, predictor_seconds=time.perf_counter() - started)
            started = time.perf_counter()
            mf, status = mom.correct(context, seed, self.strategy == 'newton_mom')
            row.update(status, corrector_seconds=time.perf_counter() - started)
            self.last_attempt_C = mom.study.occupied(mf).copy()
            if not status['qualified']:
                raise RecoveryFailure('scf_not_qualified')
            gradient = mom.lowdin_gradient(mf)
            reference_context = dict(previous_mol=self.reference_mol, previous_C=self.reference_C, target_mol=target)
            started = time.perf_counter()
            reference = reference_step(reference_context)
            row.update(diagnostic(mf, gradient, reference, status), reference_seconds=time.perf_counter() - started)
            if not row['passed']:
                if row['ground_fidelity'] > .9:
                    raise RecoveryFailure('ground_collapse')
                if row['reference_ambiguous']:
                    raise RecoveryFailure('target_reference_ambiguous')
                raise RecoveryFailure('target_observable_budget')
            self.base, self.mol = mf, target
            self.previous_mol, self.previous_C = target.copy(), mom.study.occupied(mf).copy()
            self.reference, self.reference_mol, self.reference_C = reference, target.copy(), reference['C'].copy()
            self.accepted_steps = number
            row['accepted'] = True
            self.diagnostics.append(row)
            if number == 1 or number % max(1, round(40 / self.dt)) == 0:
                print(json.dumps(dict(time_au=row['time_au'], strategy=self.strategy,
                                      density_error=row['spin_density_frobenius'], dipole_error=row['total_dipole_difference_au'])), flush=True)
            return mf.e_tot, gradient
        except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError) as error:
            self.accepted = False
            row['failure'] = str(error)[:500]
            self.diagnostics.append(row)
            raise RecoveryFailure(row['failure']) from error


def integrate(initial, strategy, end_time):
    scanner = RecoveryScanner(initial, strategy)
    count = round((end_time - initial['time']) / initial['dt'])
    if count < 1 or not np.isclose(initial['time'] + count * initial['dt'], end_time, rtol=0, atol=1e-10):
        raise ValueError('The target time must contain complete nuclear steps.')
    dynamics = md.NVE(scanner, dt=initial['dt'], steps=count + 1, time=initial['time'],
                      veloc=initial['velocity'].copy(), verbose=0)
    values = {key: [] for key in ('time', 'positions', 'velocities', 'forces', 'energy', 'dipoles', 'sample_coefficients', 'reference_coefficients')}

    def record(env):
        values['time'].append(dynamics.time)
        values['positions'].append(dynamics.mol.atom_coords().copy())
        values['velocities'].append(dynamics.veloc.copy())
        values['forces'].append(dynamics.accel * initial['masses'][:, None])
        values['energy'].append(dynamics.epot + dynamics.ekin)
        density = scanner.base.make_rdm1()
        values['dipoles'].append(np.einsum('sij,xji->x', density.real, dynamics.mol.intor('int1e_r', comp=3)))
        values['sample_coefficients'].append(mom.study.occupied(scanner.base).copy())
        values['reference_coefficients'].append(scanner.reference_C.copy())

    dynamics.callback = record
    started = time.perf_counter()
    failure = None
    try:
        with open(os.devnull, 'w') as quiet:
            dynamics.stdout = quiet
            dynamics.kernel(dump_flags=False)
        if not np.array_equal(dynamics._masses, initial['masses']):
            raise RecoveryFailure('Restart and NVE masses differ.')
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError) as error:
        failure = str(error)[:500]
    trajectory = {key: np.asarray(value) for key, value in values.items()}
    completed = failure is None and len(trajectory['time']) == count + 1 and np.isclose(trajectory['time'][-1], end_time)
    return dict(completed=bool(completed), failure=failure, completed_steps=max(0, len(values['time']) - 1),
                last_accepted_time_au=float(values['time'][-1]) if values['time'] else initial['time'],
                force_restart_max_difference=scanner.force_restart_difference, seconds=time.perf_counter() - started), trajectory, scanner


def run(root, number, strategy):
    mom.study.verify(root)
    initial = restart(root, number)
    output = root / 'runs' / f'case_{number:03d}_{strategy}'
    output.mkdir(parents=True, exist_ok=False)
    try:
        status, trajectory, scanner = integrate(initial, strategy, 4000.)
        np.savez_compressed(output / 'trajectory.npz', **trajectory)
        status['trajectory_sha256'] = hashlib.sha256((output / 'trajectory.npz').read_bytes()).hexdigest()
        (output / 'diagnostics.json').write_text(json.dumps(scanner.diagnostics, indent=2))
        if scanner.last_attempt_C is not None:
            np.savez_compressed(output / 'last_attempt.npz', positions=scanner.last_attempt_R, coefficients=scanner.last_attempt_C)
        status['passed_original_scf_failure_time'] = bool(status['last_accepted_time_au'] >= initial['original_failure_time'])
        if initial['original_wrong_branch_time'] is not None:
            status['passed_original_branch_failure_time'] = bool(status['last_accepted_time_au'] >= initial['original_wrong_branch_time'])
        verified = [row for row in scanner.diagnostics if row['accepted']]
        status.update(max_density_error=max(row['spin_density_frobenius'] for row in verified),
                      max_dipole_error_au=max(row['total_dipole_difference_au'] for row in verified),
                      max_force_error_hartree_per_bohr=max(row['force_rms_hartree_per_bohr'] for row in verified),
                      min_target_fidelity=min(row['target_fidelity'] for row in verified),
                      checked_frames=len(verified))
        if failure_row := next((row for row in reversed(scanner.diagnostics) if not row['accepted']), None):
            status['failure_diagnostic'] = failure_row
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError) as error:
        status = dict(completed=False, failure=str(error)[:500], completed_steps=0,
                      last_accepted_time_au=initial['time'], failure_stage='restart_validation')
    status.update(source_case=number, strategy=strategy, checkpoint_time_au=initial['time'],
                  source_mom_reference=initial['original_method'],
                  original_scf_failure_time_au=initial['original_failure_time'],
                  original_wrong_branch_time_au=initial['original_wrong_branch_time'],
                  source_global_diagnostic=initial['original_global_diagnostic'],
                  source_trajectory_sha256=initial['source_trajectory_sha256'],
                  source_global_diagnostic_sha256=initial['global_diagnostic_sha256'],
                  target_time_au=4000., reference_checks='Independent original branch at every accepted frame, including density and dipole budgets.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (output / 'result.json').open('x') as stream:
        json.dump(status, stream, indent=2)
    print(json.dumps(status), flush=True)


def analyze(root):
    records = []
    for case in json.loads((root / 'cases.json').read_text()):
        path = root / 'runs' / f"case_{case['source_case']:03d}_{case['strategy']}" / 'result.json'
        records.append(json.loads(path.read_text()))
    result = dict(records=records, acceptance='Original excited-state branch checked at every continuation frame; no success inferred from SCF flags alone.')
    with (root / 'compact_summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(completed=sum(row['completed'] for row in records), cases=len(records))), flush=True)


def self_check():
    from unittest.mock import patch
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    ground = mom.study.converge(mom.study.mean_field(mol, 'uks'))
    C = mom.study.occupied(ground)
    C[0, :, 0] = ground.mo_coeff[0, :, 1]
    excited = mom.study.mean_field(mol, 'uks')
    mom.study.attach_mom(excited, mol, C)
    excited = mom.reference_scf(excited, np.array([block @ block.conj().T for block in C]))
    C = mom.study.occupied(excited)
    masses = np.array([data.elements.COMMON_ISOTOPE_MASSES[z] * data.nist.AMU2AU for z in mol.atom_charges()])
    initial = dict(mol=mol, C=C, velocity=np.zeros((2, 3)), force=-mom.lowdin_gradient(excited), masses=masses,
                   dt=.1, time=0., original_method='mom_follow', fixed_mol=mol, fixed_C=C)
    result, trajectory, _ = integrate(initial, 'lowdin_only', .2)
    assert result['completed'], result
    assert np.allclose(trajectory['time'], [0, .1, .2], rtol=0, atol=1e-15)
    gradient = excited.nuc_grad_method()
    gradient.grid_response = True
    native = md.NVE(gradient, dt=.1, steps=3, veloc=np.zeros((2, 3)), verbose=0)
    advance = mom.study.attach_mom(native.scanner.base, mol, C)
    positions = []
    native.callback = lambda env: (positions.append(native.mol.atom_coords().copy()), advance())
    with open(os.devnull, 'w') as quiet:
        native.stdout = quiet
        native.kernel(dump_flags=False)
    assert np.allclose(trajectory['positions'], positions, rtol=0, atol=2e-8)
    with patch.object(mom, 'correct', side_effect=RuntimeError('Injected corrector failure')):
        result, trajectory, _ = integrate(initial, 'lowdin_only', .2)
    assert not result['completed'] and len(trajectory['time']) == 1
    assert np.array_equal(trajectory['positions'][0], mol.atom_coords())
    def collapsed(context, density, newton=False):
        mf = mom.reference_scf(mom.study.mean_field(context['target_mol'], 'uks'))
        return mf, dict(qualified=True, scf_converged=True, orbital_gradient=0., scf_iterations=1)
    with patch.object(mom, 'correct', side_effect=collapsed):
        result, trajectory, scanner = integrate(initial, 'lowdin_only', .2)
    assert not result['completed'] and len(trajectory['time']) == 1
    assert scanner.diagnostics[-1]['failure'] == 'ground_collapse'
    print('Excited-state continuation matches native NVE; failed frames are not accepted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', type=int)
    parser.add_argument('--task', type=int)
    parser.add_argument('--strategy', choices=('raw_ao', 'lowdin_only', 'cn_predictor', 'pt_predictor', 'pt_alpha10_predictor', 'newton_mom', 'follow_reference'))
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    elif args.analyze:
        analyze(args.root)
    elif args.task is not None:
        cases = json.loads((args.root / 'cases.json').read_text())
        if args.task not in range(len(cases)):
            parser.error('--task must identify a configured continuation.')
        case = cases[args.task]
        run(args.root, case['source_case'], case['strategy'])
    elif args.case is None or args.strategy is None:
        parser.error('--root, --case and --strategy are required')
    else:
        run(args.root, args.case, args.strategy)
