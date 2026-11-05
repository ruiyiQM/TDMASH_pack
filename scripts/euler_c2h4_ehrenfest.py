"""Compare scaled unrestricted PT-CN and MOM dynamics on excited ethylene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from pyscf import data, dft, gto, lib, md, scf
from scipy.linalg import eigh

from tdmash import TDUHFSimulation, TDUKSSimulation

GEOMETRY = ('C .21748627 .03041187 .12785250; C -.18721957 -.02224824 2.63987900; '
            'H 1.97520305 -.03844242 -.73638100; H -1.43020513 -.17863585 -1.11986023; '
            'H -2.20993578 .13003341 3.13549369; H 1.30459025 -.01015813 3.87199466')
DURATION = 4000.
AU_FS = .024188843265857
SAMPLE_INTERVAL = 40.
BASIS = '6-31g*'
DT_GRID = (4., 5., 2., 1., 8., 10., 20.)


def cases():
    rows = [dict(backend='uks', method='pt_cn', alpha=a, dt=dt)
            for a in (10., 1., 5., 20., 80.) for dt in DT_GRID]
    rows += [dict(backend='uks', method=method, alpha=1., dt=dt)
             for method in ('mom_follow', 'mom_fixed') for dt in (.5, 1., 2., 4., 5., 8., 10., 20.)]
    rows += [dict(backend='uks', method='accelerated', alpha=10., dt=dt) for dt in (1., 4., 5., 10., 20.)]
    rows += [dict(backend='uhf', method='pt_cn', alpha=10., dt=dt) for dt in (4., 5.)]
    rows += [dict(backend='uhf', method='mom_follow', alpha=1., dt=dt) for dt in (.5, 4., 5.)]
    return [dict(case_id=i, **row) for i, row in enumerate(rows)]


def half_overlap(mol):
    values, vectors = eigh(mol.intor('int1e_ovlp'))
    if values.min() <= 1e-12:
        raise ValueError('Singular overlap in ethylene validation.')
    return (vectors * np.sqrt(values)) @ vectors.T, (vectors * values**-.5) @ vectors.T


def mean_field(mol, backend):
    mf = dft.UKS(mol) if backend == 'uks' else scf.UHF(mol)
    mf.chkfile = None
    mf._chkfile.close()
    if backend == 'uks':
        mf.xc, mf.grids.level, mf.grids.prune, mf.small_rho_cutoff = 'lda,vwn', 0, None, 0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-6, 150
    return mf


def occupied(mf):
    return np.array([mf.mo_coeff[s][:, mf.mo_occ[s] > .5] for s in (0, 1)])


def attach_mom(mf, source_mol, source_coefficients):
    """Bind MOM to the actual scanner, using cross-geometry occupied overlaps."""
    reference = dict(mol=source_mol.copy(), C=source_coefficients.copy(), key=None, overlap=None)

    def get_occ(mo_energy=None, mo_coeff=None):
        C = np.asarray(mf.mo_coeff if mo_coeff is None else mo_coeff)
        key = mf.mol.atom_coords().tobytes()
        if key != reference['key']:
            reference['overlap'] = gto.intor_cross('int1e_ovlp', reference['mol'], mf.mol)
            reference['key'] = key
        result = np.zeros((2, C.shape[-1]))
        for s in (0, 1):
            overlap = reference['C'][s].conj().T @ reference['overlap'] @ C[s]
            scores = np.sum(abs(overlap)**2, axis=0)
            result[s, np.argsort(-scores)[:reference['C'][s].shape[1]]] = 1
        return result

    def advance_reference():
        reference.update(mol=mf.mol.copy(), C=occupied(mf).copy(), key=None, overlap=None)

    mf.get_occ = get_occ
    return advance_reference


def converge(mf, density=None):
    mf.kernel(dm0=density)
    if not mf.converged:
        refined = mf.newton()
        refined.ah_lindep, refined.ah_conv_tol = 1e-22, 1e-14
        refined.kernel(mf.mo_coeff, mf.mo_occ)
        mf = refined.undo_soscf()
    if not mf.converged:
        raise RuntimeError('State-specific SCF did not converge.')
    gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if gradient > 1e-5:
        raise RuntimeError(f'SCF orbital gradient is too large: {gradient:.3e}.')
    return mf


def initial_velocity(mol):
    masses = np.array([data.elements.COMMON_ISOTOPE_MASSES[z] * data.nist.AMU2AU for z in mol.atom_charges()])
    R = mol.atom_coords()
    R = R - np.sum(masses[:, None] * R, axis=0) / masses.sum()
    velocity = np.random.default_rng(20261102).normal(size=R.shape) / np.sqrt(masses[:, None])
    velocity -= np.sum(masses[:, None] * velocity, axis=0) / masses.sum()
    inertia = sum(m * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) for m, r in zip(masses, R))
    omega = np.linalg.solve(inertia, np.sum(masses[:, None] * np.cross(R, velocity), axis=0))
    velocity -= np.cross(omega, R)
    kinetic = .5 * np.sum(masses[:, None] * velocity**2)
    target = .5 * (3 * mol.natm - 6) * (data.nist.BOLTZMANN / data.nist.HARTREE2J) * 300
    velocity *= np.sqrt(target / kinetic)
    return masses, velocity


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]
    files = [*sorted((code / 'src/tdmash').glob('*.py')), *sorted((code / 'tests').glob('*.py')),
             Path(__file__), code / 'scripts/euler_c2h4_ehrenfest.sbatch']
    protocol = dict(geometry_bohr=GEOMETRY, basis=BASIS, state='M_s=0 alpha HOMO-to-LUMO unrestricted MOM determinant',
                    duration_au=DURATION, duration_fs=DURATION * AU_FS, sample_interval_au=SAMPLE_INTERVAL,
                    alpha='Maps to simulation.mu and scales electrons only; forces and time retain physical units',
                    initial_velocity='300 K kinetic energy, seed 20261102, COM and rigid rotation removed; no thermostat',
                    uks='LDA/VWN, level 0, no pruning, full grid response',
                    tolerances=dict(pt_cn=1e-6, accelerated=1e-8, scf_energy=1e-10, scf_gradient=1e-6),
                    iteration_limit=120, pilot_duration_au=200.,
                    branch_reference='Independent MOM continuation on each sampled nuclear path, maximum atom displacement 0.03 Bohr per SCF segment',
                    collapse_definition='F_ground > 0.90, F_target < 0.50 and F_target_ground < 0.50',
                    dt_bounds='Largest tested converged dt and largest tested sampled target fidelity >=0.90/0.99; report continuous observable errors separately',
                    timing='Fresh driver and common outputs; diagnostic SCFs and writes excluded. Target dt4/5 have two-step warmups and four cyclic-order repeats over 400 au on one CPU; full scan times are descriptive.',
                    source_sha256={str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    for name, value in [('protocol.json', protocol), ('cases.json', cases())]:
        with (root / name).open('x') as stream:
            json.dump(value, stream, indent=2)


def verify(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    code = Path(__file__).resolve().parents[1]
    for name, digest in protocol['source_sha256'].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen source changed: {name}')


def initialize(root):
    verify(root)
    if (root / 'initial.json').exists():
        for backend in ('uks', 'uhf'):
            load_initial(root, backend)
        print('Reusing checksum-verified initial orbitals and velocities.', flush=True)
        return
    arrays, records = {}, []
    for backend in ('uks', 'uhf'):
        mol = gto.M(atom=GEOMETRY, basis=BASIS, unit='Bohr', verbose=0)
        ground = converge(mean_field(mol, backend))
        occupation = ground.mo_occ.copy()
        homo = np.flatnonzero(occupation[0] > .5)[-1]
        lumo = np.flatnonzero(occupation[0] < .5)[0]
        occupation[0, homo], occupation[0, lumo] = 0, 1
        seed = np.array([ground.mo_coeff[s][:, occupation[s] > .5] for s in (0, 1)])
        excited = mean_field(mol, backend)
        attach_mom(excited, mol, seed)
        excited = converge(excited, ground.make_rdm1(mo_occ=occupation))
        if excited.e_tot <= ground.e_tot + .001:
            raise RuntimeError('Initial excited determinant collapsed to the ground state.')
        arrays[backend + '_coeff'], arrays[backend + '_occ'] = excited.mo_coeff, excited.mo_occ
        arrays[backend + '_energy'], arrays[backend + '_mo_energy'] = np.array(excited.e_tot), excited.mo_energy
        records.append(dict(backend=backend, nao=mol.nao_nr(), ground_energy=float(ground.e_tot),
                            excited_energy=float(excited.e_tot), spin_square=float(excited.spin_square()[0]),
                            orbital_gradient=float(np.linalg.norm(excited.get_grad(excited.mo_coeff, excited.mo_occ)))))
    arrays['masses'], arrays['velocity'] = initial_velocity(mol)
    with (root / 'initial.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with (root / 'initial.json').open('x') as stream:
        json.dump(dict(references=records, sha256=hashlib.sha256((root / 'initial.npz').read_bytes()).hexdigest()), stream, indent=2)
    print(json.dumps(records, indent=2), flush=True)


def load_initial(root, backend):
    metadata = json.loads((root / 'initial.json').read_text())
    if hashlib.sha256((root / 'initial.npz').read_bytes()).hexdigest() != metadata['sha256']:
        raise RuntimeError('Initial-state checksum mismatch.')
    with np.load(root / 'initial.npz') as data_file:
        values = dict(data_file)
    mol = gto.M(atom=GEOMETRY, basis=BASIS, unit='Bohr', verbose=0)
    mf = mean_field(mol, backend)
    mf.mo_coeff, mf.mo_occ = values[backend + '_coeff'].copy(), values[backend + '_occ'].copy()
    mf.mo_energy, mf.e_tot, mf.converged = values[backend + '_mo_energy'].copy(), float(values[backend + '_energy']), True
    return mf, values['masses'], values['velocity']


def integrate(root, case, duration):
    started = time.perf_counter()
    mf, masses, velocity = load_initial(root, case['backend'])
    initialized = time.perf_counter()
    sample_time, sample_C = [], []

    def sample(t, mol, C):
        if abs(t / SAMPLE_INTERVAL - round(t / SAMPLE_INTERVAL)) < 1e-8:
            sample_time.append(t)
            sample_C.append(np.array(C).copy())

    diagnostics = dict(iterations=[], residuals=[], fock_builds=0)
    failure = None
    if case['method'] in ('pt_cn', 'accelerated'):
        simulation = TDUKSSimulation if case['backend'] == 'uks' else TDUHFSimulation
        sim = simulation(mf, dt=case.get('electronic_dt', case['dt']), nuclei_dt=case['dt'], steps=round(duration / case['dt']),
                         mu=case['alpha'], propagation_method=case['method'], ehrenfest=True,
                         initial_velocity=velocity.copy(), convergence_threshold=case.get('tolerance', 1e-6 if case['method'] == 'pt_cn' else 1e-8),
                         max_iterations=120, output_interval=0, output_filename=os.devnull)
        sim.mass = masses.copy()
        dipole, propagate, fock = sim.calculate_dipole, sim.propagate_electrons, sim.get_fock_matrix

        def record_dipole(dm):
            sample(sim.t, sim.mol, sim.mo_coeff)
            return dipole(dm)

        def count_fock(*args, **kwargs):
            diagnostics['fock_builds'] += 1
            return fock(*args, **kwargs)

        def count_propagation(*args, **kwargs):
            propagate(*args, **kwargs)
            if kwargs.get('dt') != 0:
                diagnostics['iterations'].append(sim.pt_cn_iterations if case['method'] == 'pt_cn' else sim.pc_iterations)
                if case['method'] == 'pt_cn':
                    diagnostics['residuals'].append(sim.pt_cn_residual)

        sim.calculate_dipole, sim.propagate_electrons, sim.get_fock_matrix = record_dipole, count_propagation, count_fock
        try:
            sim.run_ehrenfest_midpoint_decoupled()
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failure = f'{type(error).__name__}: {error}'
        trajectory = dict(time=np.asarray(sim.times), positions=np.asarray(sim.positions_history),
                          velocities=np.asarray(sim.velocities_history), forces=np.asarray(sim.forces_history),
                          dipoles=np.asarray(sim.dipoles), energy=np.asarray(sim.total_energies))
    else:
        gradient = mf.nuc_grad_method()
        if hasattr(gradient, 'grid_response'):
            gradient.grid_response = True
        dynamics = md.NVE(gradient, dt=case['dt'], steps=round(duration / case['dt']) + 1,
                          veloc=velocity.copy(), verbose=0)
        base = dynamics.scanner.base
        update_reference = attach_mom(base, mf.mol, occupied(mf))
        base.callback = lambda env: diagnostics['iterations'].append(1)
        values = {key: [] for key in ('time', 'positions', 'velocities', 'forces', 'dipoles', 'energy')}

        def record(env):
            if not base.converged:
                raise RuntimeError('MOM trajectory SCF did not converge; frame was not accepted.')
            dm = base.make_rdm1()
            values['time'].append(dynamics.time)
            values['positions'].append(dynamics.mol.atom_coords())
            values['velocities'].append(dynamics.veloc.copy())
            values['forces'].append(dynamics.accel * masses[:, None])
            values['dipoles'].append(np.einsum('ij,xji->x', dm.sum(axis=0).real, dynamics.mol.intor('int1e_r', comp=3)))
            values['energy'].append(dynamics.epot + dynamics.ekin)
            sample(dynamics.time, dynamics.mol, occupied(base))
            if case['method'] == 'mom_follow':
                update_reference()

        dynamics.callback = record
        try:
            with open(os.devnull, 'w') as quiet:
                dynamics.stdout = quiet
                dynamics.kernel(dump_flags=False)
            if not np.array_equal(dynamics._masses, masses):
                raise RuntimeError('MOM and PT-CN use different masses.')
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failure = f'{type(error).__name__}: {error}'
        trajectory = {key: np.asarray(value) for key, value in values.items()}
    finished = time.perf_counter()
    completed = failure is None
    if completed and (len(trajectory['time']) != round(duration / case['dt']) + 1
                      or not np.isclose(trajectory['time'][-1], duration, rtol=0, atol=1e-8)):
        raise RuntimeError('Trajectory endpoint is incorrect.')
    if any(not np.isfinite(value).all() for value in trajectory.values()):
        raise RuntimeError('Nonfinite trajectory values.')
    retained = [i for i, t in enumerate(sample_time) if len(trajectory['time']) and t <= trajectory['time'][-1] + 1e-8]
    trajectory.update(sample_time=np.array([sample_time[i] for i in retained]),
                      sample_coefficients=np.array([sample_C[i] for i in retained]))
    return dict(completed=completed, failure=failure, setup_seconds=initialized - started,
                propagation_seconds=finished - initialized, total_seconds=finished - started,
                completed_steps=max(0, len(trajectory['time']) - 1),
                electronic_iterations=sum(diagnostics['iterations']), fock_builds=diagnostics['fock_builds'],
                max_pt_residual=max(diagnostics['residuals'], default=None)), trajectory


def fidelity(left, right, overlap):
    # Singular values remain well defined for exactly orthogonal determinants.
    return float(np.prod([np.prod(np.linalg.svd(a.conj().T @ overlap @ b, compute_uv=False))**2
                          for a, b in zip(left, right)]))


def diagnose(root, case, trajectory):
    mf, masses, _ = load_initial(root, case['backend'])
    target_mol, target_C = mf.mol, occupied(mf)
    results = []
    for t, C in zip(trajectory['sample_time'], trajectory['sample_coefficients']):
        index = int(round(t / case['dt']))
        R = trajectory['positions'][index]
        previous_R = target_mol.atom_coords()
        segments = max(1, int(np.ceil(np.max(np.linalg.norm(R - previous_R, axis=1)) / .03)))
        try:
            for fraction in np.arange(1, segments + 1) / segments:
                next_mol = target_mol.set_geom_(previous_R + fraction * (R - previous_R), unit='Bohr', inplace=False)
                B, _ = half_overlap(target_mol)
                _, Q = half_overlap(next_mol)
                guess = np.array([Q @ B @ block for block in target_C])
                density = np.array([block @ block.conj().T for block in guess]).real
                target = mean_field(next_mol, case['backend'])
                attach_mom(target, target_mol, target_C)
                target = converge(target, density)
                target_mol, target_C = next_mol, occupied(target)
            ground = converge(mean_field(target_mol, case['backend']))
            S = target_mol.intor('int1e_ovlp')
            B, _ = half_overlap(target_mol)
            dm = np.array([block @ block.conj().T for block in C])
            ref_dm = target.make_rdm1()
            distance = np.linalg.norm(np.array([B @ (dm[s] - ref_dm[s]) @ B for s in (0, 1)]))
            ft, fg = fidelity(C, target_C, S), fidelity(C, occupied(ground), S)
            ftg = fidelity(target_C, occupied(ground), S)
            grad = target.nuc_grad_method()
            if hasattr(grad, 'grid_response'):
                grad.grid_response = True
            force_error = trajectory['forces'][index] + grad.kernel()
            kinetic = .5 * np.sum(masses[:, None] * trajectory['velocities'][index]**2)
            results.append(dict(time_au=float(t), target_fidelity=ft, ground_fidelity=fg,
                                target_ground_fidelity=ftg, reference_ambiguous=bool(ftg >= .5 or ground.e_tot > target.e_tot + 1e-8),
                                collapse=fg > .9 and ft < .5 and ftg < .5,
                                density_frobenius=float(distance),
                                target_excitation_ev=float((target.e_tot - ground.e_tot) * data.nist.HARTREE2EV),
                                excess_energy_ev=float((trajectory['energy'][index] - kinetic - target.e_tot) * data.nist.HARTREE2EV),
                                force_rms_hartree_per_bohr=float(np.sqrt(np.mean(force_error**2))),
                                target_spin_square=float(target.spin_square()[0])))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            results.append(dict(time_au=float(t), failure=f'{type(error).__name__}: {error}'))
            break
    return results


def run(root, number):
    verify(root)
    case = json.loads((root / 'cases.json').read_text())[number]
    output = root / 'runs' / f'case_{number:03d}'
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(case, host=platform.node(), affinity=sorted(os.sched_getaffinity(0)),
                    job_id=os.environ.get('SLURM_JOB_ID'), python=platform.python_version(),
                    cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                   if line.startswith('model name')))
    if len(metadata['affinity']) != 1:
        raise RuntimeError('Benchmark requires exactly one allocated CPU.')
    pilot, partial = integrate(root, case, 200.)
    (output / 'pilot.json').write_text(json.dumps(dict(metadata, **pilot), indent=2))
    print('pilot', number, pilot['completed'], pilot.get('failure'), flush=True)
    if pilot['completed']:
        result, trajectory = integrate(root, case, DURATION)
    else:
        result, trajectory = pilot, partial
    record = dict(metadata, **result, pilot_only=not pilot['completed'])
    np.savez_compressed(output / 'trajectory.npz', **trajectory)
    record['trajectory_sha256'] = hashlib.sha256((output / 'trajectory.npz').read_bytes()).hexdigest()
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    started = time.perf_counter()
    diagnostics = diagnose(root, case, trajectory)
    record.update(branch_diagnostics=diagnostics, diagnostic_seconds=time.perf_counter() - started)
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    print('finished', number, record['completed'], record['total_seconds'], flush=True)


def analyze(root):
    verify(root)
    records = []
    for case in json.loads((root / 'cases.json').read_text()):
        path = root / 'runs' / f"case_{case['case_id']:03d}" / 'result.json'
        record = json.loads(path.read_text()) if path.exists() else dict(case, completed=False, failure='Missing result; inspect Slurm state.')
        diagnostics = record.get('branch_diagnostics', [])
        valid = [r for r in diagnostics if 'failure' not in r]
        record['diagnostic_failures'] = sum('failure' in r for r in diagnostics)
        if valid:
            record.update(min_target_fidelity=min(r['target_fidelity'] for r in valid),
                          max_ground_fidelity=max(r['ground_fidelity'] for r in valid),
                          collapse_samples=sum(r['collapse'] for r in valid),
                          ambiguous_samples=sum(r['reference_ambiguous'] for r in valid),
                          max_density_frobenius=max(r['density_frobenius'] for r in valid),
                          max_force_rms_hartree_per_bohr=max(r['force_rms_hartree_per_bohr'] for r in valid))
        if record.get('completed'):
            raw = path.parent / 'trajectory.npz'
            if hashlib.sha256(raw.read_bytes()).hexdigest() != record['trajectory_sha256']:
                raise RuntimeError('Trajectory checksum mismatch.')
            with np.load(raw) as archive:
                energy = archive['energy']
                positions = archive['positions']
                bonds = np.stack([np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
                                  for i, j in ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5))], axis=1)
                record.update(energy_max_mev_per_atom=float(np.max(abs(energy - energy[0])) * data.nist.HARTREE2EV * 1000 / 6),
                              mean_bonds_angstrom=(np.trapz(bonds, archive['time'], axis=0) / DURATION * lib.param.BOHR).tolist(),
                              seconds_per_fs=record['total_seconds'] / (DURATION * AU_FS))
        records.append(record)
    references = {}
    for backend in ('uks', 'uhf'):
        fine = next((r for r in records if r['backend'] == backend and r['method'] == 'mom_follow'
                     and r['dt'] == .5 and r.get('completed')), None)
        if fine:
            with np.load(root / 'runs' / f"case_{fine['case_id']:03d}" / 'trajectory.npz') as archive:
                fine_time, fine_R = archive['time'], archive['positions']
            for row in records:
                if row['backend'] != backend or not row.get('completed'):
                    continue
                with np.load(root / 'runs' / f"case_{row['case_id']:03d}" / 'trajectory.npz') as archive:
                    indices = np.rint(archive['time'] / .5).astype(int)
                    if not np.allclose(fine_time[indices], archive['time'], rtol=0, atol=1e-8):
                        raise RuntimeError('Nuclear comparison needs shared physical time nodes.')
                    row['position_rms_angstrom'] = float(np.sqrt(np.mean(np.sum((archive['positions'] - fine_R[indices])**2, axis=-1))) * lib.param.BOHR)
                delta = np.array(row['mean_bonds_angstrom']) - fine['mean_bonds_angstrom']
                row['mean_cc_error_angstrom'], row['mean_ch_max_error_angstrom'] = float(abs(delta[0])), float(np.max(abs(delta[1:])))
            coarse = next((r for r in records if r['backend'] == backend and r['method'] == 'mom_follow'
                           and r['dt'] == 1 and r.get('completed')), None)
            references[backend] = dict(case_id=fine['case_id'], dt=.5,
                                       mean_geometry_validated=bool(coarse and coarse['mean_cc_error_angstrom'] < 1e-4
                                                                    and coarse['mean_ch_max_error_angstrom'] < 1e-4),
                                       branch_ambiguous_samples=fine.get('ambiguous_samples'),
                                       diagnostic_failures=fine.get('diagnostic_failures'))
    bounds = []
    for backend in ('uks', 'uhf'):
        for alpha in (1., 5., 10., 20., 80.):
            rows = [r for r in records if r['backend'] == backend and r['method'] == 'pt_cn' and r['alpha'] == alpha]
            if not rows:
                continue
            complete = [r for r in rows if r.get('completed')]
            entry = dict(backend=backend, alpha=alpha, max_completed_dt=max((r['dt'] for r in complete), default=None),
                         completed_dt_values=sorted(r['dt'] for r in complete),
                         incomplete_dt_values=sorted(r['dt'] for r in rows if not r.get('completed')),
                         completed_configurations=[dict(nuclear_dt=r['dt'], electronic_dt=r.get('electronic_dt', r['dt'])) for r in complete],
                         incomplete_configurations=[dict(nuclear_dt=r['dt'], electronic_dt=r.get('electronic_dt', r['dt'])) for r in rows if not r.get('completed')])
            for threshold in (.9, .99):
                passing = [r for r in complete if r.get('min_target_fidelity', 0) >= threshold
                           and not r.get('diagnostic_failures') and not r.get('ambiguous_samples') and r.get('branch_diagnostics')]
                entry[f'max_dt_fidelity_{threshold}'] = max((r['dt'] for r in passing), default=None)
            bounds.append(entry)
    result = dict(protocol=json.loads((root / 'protocol.json').read_text()),
                  initial=json.loads((root / 'initial.json').read_text()), records=records, tested_dt_bounds=bounds,
                  nuclear_references=references)
    result['paired_timings'] = [json.loads(p.read_text()) for p in sorted((root / 'timing').glob('cell_*/summary.json'))]
    with (root / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    compact = dict(result, records=[{k: v for k, v in row.items() if k != 'branch_diagnostics'} for row in records],
                   branch_diagnostics_retained_on_euler=True)
    with (root / 'compact_summary.json').open('x') as stream:
        json.dump(compact, stream, indent=2)
    print(json.dumps(bounds, indent=2))


def timing(root, number):
    verify(root)
    target = json.loads((root / 'cases.json').read_text())[number]
    if number not in (0, 1):
        raise ValueError('Paired timing is defined for alpha10 dt4 and dt5 only.')
    output = root / 'timing' / f'cell_{number:03d}'
    output.mkdir(parents=True, exist_ok=False)
    names = ('pt_cn', 'mom_follow', 'mom_fixed', 'accelerated')
    records, failed = [], set()
    warmups = []
    for method in names:
        case = dict(target, method=method, alpha=10. if method in ('pt_cn', 'accelerated') else 1.)
        if method != 'pt_cn':
            case.pop('tolerance', None)
        warmup, _ = integrate(root, case, 2 * target['dt'])
        warmups.append(dict(warmup, method=method))
    for repeat in range(4):
        order = names[repeat:] + names[:repeat]
        for rank, method in enumerate(order):
            if method in failed:
                records.append(dict(repeat=repeat, rank=rank, method=method, completed=False,
                                    failure='Not repeated after deterministic failure.'))
                continue
            case = dict(target, method=method, alpha=10. if method in ('pt_cn', 'accelerated') else 1.)
            if method != 'pt_cn':
                case.pop('tolerance', None)
            record, trajectory = integrate(root, case, 400.)
            record.update(repeat=repeat, rank=rank, method=method)
            records.append(record)
            if not record['completed']:
                failed.add(method)
            print('timing', number, repeat, method, record['completed'], record['total_seconds'], flush=True)
    ratios = {}
    for method in names[1:]:
        pairs = []
        for repeat in range(4):
            pair = {r['method']: r for r in records if r['repeat'] == repeat}
            if pair['pt_cn']['completed'] and pair[method]['completed']:
                pairs.append(pair[method]['total_seconds'] / pair['pt_cn']['total_seconds'])
        ratios[method + '_over_pt'] = dict(pairs=pairs, median=float(np.median(pairs)) if pairs else None,
                                         minimum=min(pairs, default=None), maximum=max(pairs, default=None))
    summary = dict(target=target, duration_au=400., duration_fs=400 * AU_FS,
                   host=platform.node(), affinity=sorted(os.sched_getaffinity(0)),
                   job_id=os.environ.get('SLURM_JOB_ID'), warmups=warmups, records=records, ratios=ratios)
    with (output / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2)


def self_check():
    rows = cases()
    assert rows[0]['alpha'] == 10 and rows[0]['dt'] == 4 and rows[1]['dt'] == 5
    assert len({(r['backend'], r['method'], r['alpha'], r['dt']) for r in rows}) == len(rows)
    assert all(DURATION % r['dt'] == 0 and 200 % r['dt'] == 0 and SAMPLE_INTERVAL % r['dt'] == 0 for r in rows)
    mol = gto.M(atom=GEOMETRY, basis='sto-3g', unit='Bohr', verbose=0)
    masses, velocity = initial_velocity(mol)
    assert_all = np.testing.assert_allclose
    assert_all(np.sum(masses[:, None] * velocity, axis=0), 0, atol=1e-12)
    R = mol.atom_coords() - np.sum(masses[:, None] * mol.atom_coords(), axis=0) / masses.sum()
    assert_all(np.sum(masses[:, None] * np.cross(R, velocity), axis=0), 0, atol=1e-12)
    B, Q = half_overlap(mol)
    assert_all(B @ Q, np.eye(mol.nao_nr()), atol=1e-12)
    C = np.stack([Q[:, :8]] * 2)
    mf = scf.UHF(mol)
    mf.chkfile = None
    attach_mom(mf, mol, C)
    permutation = np.arange(mol.nao_nr())[::-1]
    full = np.stack([Q[:, permutation]] * 2)
    occupation = mf.get_occ(mo_coeff=full)
    assert_all(occupation.sum(axis=1), [8, 8])
    selected = np.array([full[s][:, occupation[s] > .5] for s in (0, 1)])
    assert_all(fidelity(C, selected, mol.intor('int1e_ovlp')), 1, atol=1e-12)
    mf._chkfile.close()
    from unittest.mock import patch
    ground = converge(mean_field(mol, 'uhf'))
    seed = occupied(ground)
    seed[0, :, -1] = ground.mo_coeff[0, :, 8]
    excited = mean_field(mol, 'uhf')
    attach_mom(excited, mol, seed)
    excited = converge(excited, np.array([block @ block.conj().T for block in seed]))
    trajectory = dict(sample_time=np.array([0.]), sample_coefficients=occupied(excited)[None],
                      time=np.array([0.]), positions=mol.atom_coords()[None], velocities=np.zeros((1, 6, 3)),
                      forces=np.zeros((1, 6, 3)), energy=np.array([excited.e_tot]))
    with patch(__name__ + '.load_initial', return_value=(excited, masses, velocity)):
        diagnostic = diagnose(Path('/unused'), dict(backend='uhf', dt=1), trajectory)
        record, subcycled = integrate(Path('/unused'), dict(backend='uhf', method='pt_cn', alpha=1,
                                                           dt=.1, electronic_dt=.05, tolerance=1e-8), .2)
    assert diagnostic and 'failure' not in diagnostic[0]
    assert diagnostic[0]['target_fidelity'] > .99 and diagnostic[0]['target_ground_fidelity'] < .5
    json.dumps(diagnostic)
    assert record['completed'] and record['completed_steps'] == 2
    assert_all(subcycled['time'], [0, .1, .2], atol=1e-15)
    assert record['electronic_iterations'] >= 4
    print('Ethylene manifest, initial state, MOM diagnostics and electronic-substep workflow checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'initialize', 'run', 'analyze', 'timing', 'self-check'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    lib.num_threads(1)
    if args.action == 'self-check':
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    elif args.action in ('run', 'timing'):
        if args.case is None or args.case not in range(len(cases())):
            parser.error('--case must identify an existing case')
        globals()[args.action](args.root, args.case)
    else:
        globals()[args.action](args.root)


if __name__ == '__main__':
    main()
