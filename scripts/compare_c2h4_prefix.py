"""Compare accepted ethylene trajectory prefixes with the saved MOM reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyscf.lib import param

import euler_c2h4_ehrenfest as study


def compare(candidate, reference):
    time = candidate['time']
    keep = time <= reference['time'][-1] + 1e-10
    time = time[keep]
    if len(time) < 2:
        return dict(comparable=False, reason='Fewer than two shared frames.')
    right = np.minimum(np.searchsorted(reference['time'], time), len(reference['time']) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(reference['time'][left] - time) < abs(reference['time'][right] - time), left, right)
    if not np.allclose(reference['time'][indices], time, rtol=0, atol=1e-10):
        raise ValueError('Prefix comparison requires shared physical time nodes.')
    R, ref_R = candidate['positions'][keep], reference['positions'][indices]
    displacement = np.linalg.norm(R - ref_R, axis=-1) * param.BOHR
    bonds = lambda positions: np.stack([np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
                                       for i, j in ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5))], axis=1) * param.BOHR
    mean_bond_difference = np.trapz(bonds(R) - bonds(ref_R), time, axis=0) / (time[-1] - time[0])
    charges = np.array([6, 6, 1, 1, 1, 1])
    dipole = np.einsum('a,tac->tc', charges, R) - candidate['dipoles'][keep]
    ref_dipole = np.einsum('a,tac->tc', charges, ref_R) - reference['dipoles'][indices]
    energy = candidate['energy'][keep] - reference['energy'][indices]
    result = dict(comparable=True, frames=len(time), duration_au=float(time[-1] - time[0]),
                  duration_fs=float((time[-1] - time[0]) * study.AU_FS),
                  position_rms_angstrom=float(np.sqrt(np.mean(displacement**2))),
                  position_max_angstrom=float(np.max(displacement)),
                  shared_node_mean_cc_error_angstrom=float(abs(mean_bond_difference[0])),
                  shared_node_mean_ch_max_error_angstrom=float(np.max(abs(mean_bond_difference[1:]))),
                  total_dipole_rms_au=float(np.sqrt(np.mean(np.sum((dipole - ref_dipole)**2, axis=-1)))),
                  max_energy_difference_hartree=float(np.max(abs(energy))))
    if not np.isfinite([v for v in result.values() if isinstance(v, float)]).all():
        raise ValueError('Nonfinite prefix comparison.')
    return result


def load(root, number):
    folder = root / 'runs' / f'case_{number:03d}'
    record = json.loads((folder / 'result.json').read_text())
    raw = folder / 'trajectory.npz'
    if hashlib.sha256(raw.read_bytes()).hexdigest() != record['trajectory_sha256']:
        raise RuntimeError('Trajectory checksum mismatch.')
    with np.load(raw) as archive:
        return record, dict(archive)


def run(root):
    study.verify(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    parent = Path(protocol['parent_root'])
    if hashlib.sha256((root / 'initial.npz').read_bytes()).digest() != hashlib.sha256((parent / 'initial.npz').read_bytes()).digest():
        raise RuntimeError('Prefix comparison requires identical initial states.')
    references = {backend: load(parent, number) for backend, number in (('uks', 35), ('uhf', 58))}
    _, coarse = load(parent, 36)
    rows = []
    for case in json.loads((root / 'cases.json').read_text()):
        record, trajectory = load(root, case['case_id'])
        reference_record, reference = references[case['backend']]
        row = dict(case, completed=record['completed'], completed_steps=record['completed_steps'],
                   trajectory_sha256=record['trajectory_sha256'], reference_case=reference_record['case_id'],
                   reference_trajectory_sha256=reference_record['trajectory_sha256'],
                   reference_completed=reference_record['completed'], metrics=compare(trajectory, reference))
        if case['backend'] == 'uks':
            prefix = {k: v[coarse['time'] <= trajectory['time'][-1] + 1e-10] for k, v in coarse.items()
                      if k in ('time', 'positions', 'dipoles', 'energy')}
            row['reference_dt1_vs_dt0_5'] = compare(prefix, reference)
        rows.append(row)
    result = dict(parent_root=str(parent), partial_prefix_only=True,
                  limitation='Accepted shared-time prefixes only; failed long trajectories do not pass. MOM reference stationarity and branch identity are assessed separately.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=rows)
    with (root / 'prefix_comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(rows, indent=2))


def self_check():
    R = np.arange(18.).reshape(1, 6, 3) * np.ones((3, 1, 1))
    reference = dict(time=np.array([0., 1., 2.]), positions=R, dipoles=np.zeros((3, 3)), energy=np.ones(3))
    candidate = {key: value[::2].copy() for key, value in reference.items()}
    assert compare(candidate, reference)['position_max_angstrom'] == 0
    candidate['positions'][:, :, 0] += .01
    candidate['dipoles'][:, 0] += .16
    result = compare(candidate, reference)
    assert abs(result['position_rms_angstrom'] - .01 * param.BOHR) < 1e-14
    assert result['shared_node_mean_ch_max_error_angstrom'] < 1e-14
    assert result['total_dipole_rms_au'] < 1e-13
    candidate['time'][-1] = 1.5
    try:
        compare(candidate, reference)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonshared time nodes must be rejected.')
    assert not compare({key: value[:1] for key, value in reference.items()}, reference)['comparable']
    print('Shared-prefix, translation-invariant bonds and total-dipole checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    else:
        run(args.root)
