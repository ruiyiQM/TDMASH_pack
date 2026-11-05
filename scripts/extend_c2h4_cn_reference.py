"""Extend validated short CN/PT-CN configurations to the requested 100 fs scale."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import euler_c2h4_cn_reference as benchmark


def select(summary):
    if not summary['cn_reference_validated']:
        return [], 'Short CN reference did not pass refinement.'
    pt = sorted((r for r in summary['records'] if r['role'] == 'candidate' and r['method'] == 'pt_cn' and r['passed']),
                key=lambda r: (r['dt'], -r['tolerance']))
    distinct = {}
    for row in pt:
        distinct[row['dt']] = row
    pt = list(distinct.values())
    cn = sorted((r for r in summary['records'] if r['role'] == 'candidate' and r['method'] == 'accelerated' and r['passed']),
                key=lambda r: r['dt'])
    if not pt:
        return [], 'No short PT-CN candidate passed the observable budgets.'
    chosen = summary['records'][:4] + pt[-2:] + cn[-1:]
    return [{key: row[key] for key in ('backend', 'method', 'alpha', 'dt', 'electronic_dt', 'role', 'tolerance')}
            for row in chosen], 'Extend four reference controls and the largest accepted candidate steps.'


def merge_cases(selected, previous):
    keys = ('backend', 'method', 'alpha', 'dt', 'electronic_dt', 'role', 'tolerance')
    signature = lambda row: tuple(row[key] for key in keys)
    if len(previous) < 4 or [signature(row) for row in selected[:4]] != [signature(row) for row in previous[:4]]:
        raise ValueError('Reusing a long study requires identical reference configurations.')
    if any(row['case_id'] != i for i, row in enumerate(previous)):
        raise ValueError('Reused case IDs must match their trajectory directories.')
    combined = [dict(row) for row in previous]
    seen = {signature(row) for row in previous}
    for row in selected[4:]:
        if signature(row) not in seen:
            combined.append(dict(row, case_id=len(combined)))
            seen.add(signature(row))
    return combined, list(range(len(previous), len(combined)))


def verify_reuse(root, previous):
    protocols = [json.loads((path / 'protocol.json').read_text()) for path in (root, previous)]
    physical = [{name: digest for name, digest in protocol['source_sha256'].items()
                 if name.startswith('src/') or name == 'scripts/euler_c2h4_ehrenfest.py'} for protocol in protocols]
    if not physical[0] or physical[0] != physical[1] or protocols[1]['duration_au'] != 4000.:
        raise ValueError('Reuse requires identical physical sources and a 4000-au source study.')
    initial = []
    for path, sources in zip((root, previous), physical):
        for name, digest in sources.items():
            if hashlib.sha256((path / 'code' / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Reused physical source checksum mismatch: ' + name)
        digest = hashlib.sha256((path / 'initial.npz').read_bytes()).hexdigest()
        if digest != json.loads((path / 'initial.json').read_text())['sha256']:
            raise ValueError('Reused initial-state checksum mismatch.')
        initial.append(digest)
    if initial[0] != initial[1]:
        raise ValueError('Reuse requires identical initial orbitals and velocities.')


def reuse_dependencies(root, indices):
    sources = {}
    for i in indices:
        result = root / 'runs' / f'case_{i:03d}' / 'result.json'
        if not result.exists():
            study_root = result.parent.resolve().parents[1]
            job = json.loads((study_root / 'submission.json').read_text())['array_job_id'].split(';')[0]
            sources.setdefault(job, []).append(result)
    dependencies = []
    for job, results in sources.items():
        live = subprocess.run(['squeue', '-h', '-j', job, '-o', '%i'], capture_output=True, text=True)
        if live.stdout.strip():
            dependencies.append(job)
        elif any(not result.exists() for result in results):
            raise RuntimeError('Reused results are incomplete and their source job is not observable; inspect it before resubmitting.')
    return dependencies


def run(root, reuse_root=None):
    benchmark.study.verify(root)
    summary = json.loads((root / 'compact_summary.json').read_text())
    configuration, reason = select(summary)
    decision = dict(reason=reason, extended=bool(configuration),
                    short_summary_sha256=hashlib.sha256((root / 'compact_summary.json').read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    pending = list(range(len(configuration)))
    reused = []
    dependencies = []
    if configuration and reuse_root is not None:
        verify_reuse(root, reuse_root)
        previous = json.loads((reuse_root / 'cases.json').read_text())
        configuration, pending = merge_cases(configuration, previous)
        reused = list(range(len(previous)))
        decision.update(reuse_long_root=str(reuse_root), reused_case_ids=reused, new_case_ids=pending,
                        physical_sources_and_initial_state_verified=True)
        if not pending:
            configuration = []
            decision.update(extended=False, reason='Existing long study already covers all accepted settings.')
        else:
            dependencies = reuse_dependencies(reuse_root, reused)
    if configuration:
        output = root / 'long_validation'
        output.mkdir(exist_ok=False)
        (output / 'code').symlink_to(root / 'code')
        for name in ('initial.json', 'initial.npz'):
            (output / name).symlink_to(root / name)
        (output / 'logs').mkdir()
        (output / 'runs').mkdir()
        # Preserve source case IDs so reused result metadata stays consistent.
        for i in reused:
            (output / 'runs' / f'case_{i:03d}').symlink_to(reuse_root / 'runs' / f'case_{i:03d}', target_is_directory=True)
        configuration = [dict(row, case_id=i) for i, row in enumerate(configuration)]
        (output / 'cases.json').write_text(json.dumps(configuration, indent=2))
        protocol = dict(summary['protocol'], duration_au=4000., duration_fs=4000 * benchmark.study.AU_FS,
                        short_root=str(root), phase='Long validation after short CN refinement and PT-CN observable checks')
        if reuse_root is not None:
            protocol.update(reuse_long_root=str(reuse_root), reused_case_ids=reused, new_case_ids=pending)
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        (output / 'extension_script.py').write_bytes(Path(__file__).read_bytes())
        files = [output / 'code' / name for name in protocol['source_sha256']]
        files += [output / name for name in ('cases.json', 'protocol.json', 'initial.json', 'initial.npz', 'extension_script.py')]
        (output / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(output)) + '\n' for p in files))
        script = output / 'code/scripts/euler_c2h4_cn_reference.sbatch'
        resources = protocol.get('long_resources', {})
        scan = subprocess.check_output(['sbatch', '--parsable', '--job-name=c2cn-long', '--time=' + resources.get('walltime', '08:00:00'),
                                        '--mem-per-cpu=' + resources.get('run_memory', '4G'),
                                        '--array=' + ','.join(map(str, pending)), '--output=' + str(output / 'logs/run_%A_%a.out'),
                                        str(script), str(output), 'run'], text=True).strip().split(';')[0]
        decision.update(root=str(output), array_job_id=scan, cases=len(configuration), new_case_ids=pending)
        (output / 'submission.json').write_text(json.dumps(decision, indent=2))
        dependencies.insert(0, scan)
        analysis = subprocess.check_output(['sbatch', '--parsable', '--job-name=c2cn-long-analysis', '--mem-per-cpu=' + resources.get('analysis_memory', '8G'),
                                            '--dependency=afterany:' + ':'.join(dependencies), '--output=' + str(output / 'logs/analysis_%j.out'),
                                            str(script), str(output), 'analyze'], text=True).strip().split(';')[0]
        decision.update(analysis_job_id=analysis, analysis_dependencies=dependencies)
        (output / 'submission.json').write_text(json.dumps(decision, indent=2))
    with (root / 'extension_decision.json').open('x') as stream:
        json.dump(decision, stream, indent=2)
    print(json.dumps(decision), flush=True)


def self_check():
    from contextlib import redirect_stdout
    import io
    from unittest.mock import patch
    base = dict(backend='uks', alpha=1., dt=.25, electronic_dt=.25, role='reference', tolerance=1e-10, method='accelerated', passed=True)
    rows = [dict(base) for _ in range(4)]
    rows += [dict(base, role='candidate', method='pt_cn', dt=dt, electronic_dt=dt) for dt in (.5, 1, 2)]
    summary = dict(cn_reference_validated=True, records=rows)
    selected, _ = select(summary)
    assert [row['dt'] for row in selected[4:]] == [1, 2]
    rows.append(dict(rows[-1], tolerance=1e-5))
    selected, _ = select(summary)
    assert [row['dt'] for row in selected[4:]] == [1, 2]
    assert selected[-1]['tolerance'] == 1e-10
    previous = [dict(row, case_id=i) for i, row in enumerate(selected[:5])]
    combined, pending = merge_cases(selected, previous)
    assert combined[:5] == previous and pending == [5]
    assert combined[5]['case_id'] == 5 and combined[5]['dt'] == 2
    assert not merge_cases(selected[:5], previous)[1]
    wrong = [dict(row) for row in previous]
    wrong[0]['dt'] /= 2
    try:
        merge_cases(selected, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError('Different reference grids must not be silently reused.')
    with tempfile.TemporaryDirectory() as folder:
        paths = [Path(folder) / name for name in ('new', 'old')]
        source = b'VALUE = 1\n'
        initial = b'identical occupied orbitals and velocities\n'
        for path in paths:
            (path / 'code/src').mkdir(parents=True)
            (path / 'code/src/model.py').write_bytes(source)
            (path / 'protocol.json').write_text(json.dumps(dict(duration_au=4000., source_sha256={
                'src/model.py': hashlib.sha256(source).hexdigest()})))
            (path / 'initial.npz').write_bytes(initial)
            (path / 'initial.json').write_text(json.dumps(dict(sha256=hashlib.sha256(initial).hexdigest())))
        verify_reuse(*paths)
        (paths[1] / 'code/src/model.py').write_bytes(b'VALUE = 2\n')
        try:
            verify_reuse(*paths)
        except ValueError:
            pass
        else:
            raise AssertionError('Physical source drift must prevent trajectory reuse.')
        (paths[1] / 'code/src/model.py').write_bytes(source)
        changed = initial + b'changed velocity'
        (paths[1] / 'initial.npz').write_bytes(changed)
        (paths[1] / 'initial.json').write_text(json.dumps(dict(sha256=hashlib.sha256(changed).hexdigest())))
        try:
            verify_reuse(*paths)
        except ValueError:
            pass
        else:
            raise AssertionError('Different verified initial states must prevent trajectory reuse.')
        (paths[1] / 'initial.npz').write_bytes(initial)
        (paths[1] / 'initial.json').write_text(json.dumps(dict(sha256=hashlib.sha256(initial).hexdigest())))
        protocol = json.loads((paths[0] / 'protocol.json').read_text())
        (paths[0] / 'compact_summary.json').write_text(json.dumps(dict(summary, protocol=protocol)))
        (paths[1] / 'cases.json').write_text(json.dumps(previous))
        (paths[1] / 'submission.json').write_text(json.dumps(dict(array_job_id='42')))
        ancestor = Path(folder) / 'ancestor'
        (ancestor / 'runs/case_000').mkdir(parents=True)
        (ancestor / 'submission.json').write_text(json.dumps(dict(array_job_id='43')))
        (paths[1] / 'runs').mkdir()
        (paths[1] / 'runs/case_000').symlink_to(ancestor / 'runs/case_000', target_is_directory=True)
        live = subprocess.CompletedProcess([], 0, stdout='42_0\n', stderr='')
        with patch.object(benchmark.study, 'verify'), patch.object(subprocess, 'run', return_value=live), \
                patch.object(subprocess, 'check_output', side_effect=['51\n', '52\n']) as submit, redirect_stdout(io.StringIO()):
            run(paths[0], paths[1])
        assert '--array=5' in submit.call_args_list[0].args[0]
        dependency = next(arg for arg in submit.call_args_list[1].args[0] if arg.startswith('--dependency='))
        assert dependency.startswith('--dependency=afterany:') and set(dependency.split(':')[1:]) == {'51', '42', '43'}
        output = paths[0] / 'long_validation'
        stored = json.loads((output / 'cases.json').read_text())
        assert stored[:5] == previous and stored[5]['case_id'] == 5
        assert all((output / 'runs' / f'case_{i:03d}').is_symlink() for i in range(5))
        assert '  code/src/model.py\n' in (output / 'SHA256SUMS').read_text()
        with patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='', stderr='')):
            try:
                reuse_dependencies(paths[1], range(5))
            except RuntimeError:
                pass
            else:
                raise AssertionError('Missing source results without a live job require inspection.')
    summary['cn_reference_validated'] = False
    assert not select(summary)[0]
    summary['cn_reference_validated'] = True
    for row in rows[4:]:
        row['passed'] = False
    assert not select(summary)[0]
    print('Long-trajectory extension requires a validated reference and accepted PT candidates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--reuse-long-root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is None:
        parser.error('--root is required')
    else:
        run(args.root, args.reuse_long_root)
