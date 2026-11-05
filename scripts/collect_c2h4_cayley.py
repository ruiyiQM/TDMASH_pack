"""Collect only compact artifacts after the independent Euler analysis succeeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

HOST = 'zhourui@euler.ethz.ch'
OPTIONS = ['-o', 'BatchMode=yes',
           '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20']
if os.environ.get('TDMASH_SSH_IDENTITY'):
    OPTIONS[:0] = ['-i', os.environ['TDMASH_SSH_IDENTITY']]
REMOTE = """
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]);job=sys.argv[2]
status=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=State,ExitCode'],text=True).strip()
files={}
if status.startswith('COMPLETED|0:0'):
 for name in ('compact_summary.json','cayley_spectra.png'):
  p=root/name
  files[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(dict(status=status,files=files)))
"""


def report(summary):
    protocol = summary['protocol']
    if protocol.get('resumes_paused_goal') is not False or protocol.get('alpha') != 1.:
        raise ValueError('This collector accepts only the independent alpha=1 study.')
    rows = summary['records']
    completed = sum(row['completed'] for row in rows)
    response = summary['response']
    growth = response['dscf']['max_growth_rate']
    instability = (f"ΔSCF 存在增长模：最大 γ={growth:.6g} a.u.，e-folding 时间 {1/growth:.3f} a.u."
                   f"（{.024188843265857/growth:.3f} fs）；其非线性尾部不能作为稳定线性谱验收。" if growth > 1e-6
                   else 'ΔSCF 响应矩阵未检测到超过 1e-6 a.u. 的增长率；仍需结合双 kick 线性检查。')
    table = []
    for state, label in (('ground', '基态'), ('dscf', 'ΔSCF')):
        bounds = summary['largest_tested_completed_dt'][state]
        table.append(f"| {label} | {bounds['accelerated']} | {bounds['pt_cn']} | {summary['reference_checks'][state]['validated']} |")
    verdicts = []
    for method, label in (('accelerated', '常规 CN'), ('pt_cn', 'PT-CN')):
        selected = [r for r in rows if r['state'] == 'ground' and r['role'] == 'candidate' and r['method'] == method]
        support = [r['dt'] for r in selected if r.get('classification', '').startswith('mapping_supported')]
        rejected = [r['dt'] for r in selected if r.get('classification', '').startswith('mapping_not_supported')]
        unresolved = [r['dt'] for r in selected if r['completed'] and r['dt'] not in support + rejected]
        verdicts.append(f"{label}：可分辨模式支持映射的步长 {support}；不支持 {rejected}；未分辨或非线性 {unresolved}。逐峰误差、振幅比和分类依据见 JSON。")
    return '\n'.join([
        '# 固定核电子相干：Cayley 映射独立测试', '',
        '本测试不恢复此前暂停的 goal，也不消耗其步长细化次数。', '',
        '**设置。** 冻结 C2H4/6-31G*、UKS LDA/VWN、M_s=0 ΔSCF 初态及同几何 UKS 基态；'
        '固定核、α=1、斜向弱 kick，κ=1e-4、5e-4，另有每档零 kick 对照。'
        '细 CN 参考 dt=0.01/0.005；CN/PT 各测试 0.05、0.5、1、2、5、10、15 a.u.。'
        f'32 组、96 条传播，目标窗口 {protocol["duration_au"]:g} a.u.；其中 {completed}/32 组全部三条完成。', '',
        '**收敛边界。** 同一 120 次迭代上限、1e-11 求解阈值；独立核对正交化前方程残差，'
        '正交化后偏差单列，不当作迭代失败。下表是最大已测完成步长，不能解释为最大准确步长；'
        '收敛可能非单调，不能据此推断更小步长全部可用。', '',
        '| 电子态 | CN / a.u. | PT-CN / a.u. | 细参考通过验收 |',
        '|---|---:|---:|---|', *table, '',
        '**稳定性与频率解释。** ' + instability, '',
        *verdicts, '',
        '常规轨道 CN 的两能级相干相位是两个轨道 Cayley 相位之差；'
        'PT 连续规范等价不保证有限 dt 的谱一致。各 dt 的直接 CN/PT 峰位差、振幅比、增长率和失败位置均保存在摘要中。', '',
        '**证据及限制。** [谱、峰位与振幅图](c2h4_cayley_spectra_20261104.png)；'
        '[精简结果](c2h4_cayley_20261104.json)。峰位采用不锁定理论频率的三分量 matrix-pencil 谐波反演，'
        '核对不同秩、窗口及双 kick。拥挤峰、小于提取不确定性的频移单独标注；'
        '不能用不稳定或非线性长时间响应证明稳定线性频率映射。原始输出保留 Euler scratch，未 commit/push。', '',
        '此页由脚本汇总；图像和拥挤峰匹配仍需目视复核。', ''])


def collect(root, job, output, wait=False):
    if (not root.startswith('/cluster/scratch/zhourui/tdmash_validation/c2h4_cayley_independent_')
            or not job.isdigit()):
        raise ValueError('Expected an independent scratch study and a numeric analysis job ID.')
    deadline = time.monotonic() + 48 * 3600
    command = shlex.join(['python3', '-c', REMOTE, root, job])
    while True:
        try:
            reply = subprocess.run(['ssh', *OPTIONS, HOST, command], capture_output=True, text=True, timeout=60)
            state = json.loads(reply.stdout) if reply.returncode == 0 else dict(status='SSH_RETRY', files={})
        except (subprocess.TimeoutExpired, OSError, ValueError):
            state = dict(status='SSH_RETRY', files={})
        if state['files']:
            break
        if any(word in state['status'] for word in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY')):
            raise RuntimeError('Euler analysis did not succeed: ' + state['status'])
        if not wait or time.monotonic() >= deadline:
            print(json.dumps(state), flush=True)
            return False
        time.sleep(300)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cayley-', dir=output) as directory:
        staging = Path(directory)
        names = {'compact_summary.json': 'c2h4_cayley_20261104.json',
                 'cayley_spectra.png': 'c2h4_cayley_spectra_20261104.png'}
        for source, destination in names.items():
            if not 0 < state['files'][source]['bytes'] < 2_000_000:
                raise ValueError('Only small result artifacts may be downloaded.')
            target = staging / destination
            subprocess.run(['scp', *OPTIONS, HOST + ':' + root + '/' + source, str(target)], check=True, timeout=120)
            if hashlib.sha256(target.read_bytes()).hexdigest() != state['files'][source]['sha256']:
                raise RuntimeError('Collected artifact checksum mismatch.')
        summary = json.loads((staging / names['compact_summary.json']).read_text())
        summary['local_compaction'] = dict(full_summary_sha256=state['files']['compact_summary.json']['sha256'],
                                           retained_modes_per_case=12, full_results_on_euler=True)
        for row in summary['records']:
            if row.get('fit'):
                row['fit']['extracted_mode_count'] = len(row['fit']['modes'])
                row['fit']['modes'] = row['fit']['modes'][:12]
        (staging / names['compact_summary.json']).write_text(json.dumps(summary, indent=2, allow_nan=False))
        (staging / 'PT_CN_Cayley独立测试.md').write_text(report(summary))
        for path in staging.iterdir():
            path.replace(output / path.name)
    print(json.dumps(dict(collected=True, output=str(output), analysis_job_id=job)), flush=True)
    return True


def self_check():
    import runpy
    from unittest.mock import patch

    scripts = Path(__file__).resolve().parent
    for name in ('euler_c2h4_cayley.sbatch', 'euler_c2h4_ehrenfest.sbatch',
                 'euler_c2h4_cn_reference.sbatch'):
        assert (scripts / name).is_file(), 'Missing frozen Slurm template: ' + name

    for identity in ('', '/tmp/test identity'):
        with patch.dict(os.environ, {'TDMASH_SSH_IDENTITY': identity}):
            settings = runpy.run_path(__file__)
        reply = subprocess.CompletedProcess([], 0, stdout='{"status":"RUNNING","files":{}}')
        with patch.object(subprocess, 'run', return_value=reply) as ssh:
            assert not settings['collect'](
                '/cluster/scratch/zhourui/tdmash_validation/c2h4_cayley_independent_test',
                '1', Path('/tmp'))
        command = ssh.call_args.args[0]
        assert ('-i' in command) == bool(identity)
        if identity:
            assert command[command.index('-i') + 1] == identity

    summary = dict(protocol=dict(resumes_paused_goal=False, alpha=1., duration_au=1200.), records=[],
                   response=dict(dscf=dict(max_growth_rate=.05)),
                   largest_tested_completed_dt={state: dict(accelerated=2., pt_cn=15.) for state in ('ground', 'dscf')},
                   reference_checks={state: dict(validated=False) for state in ('ground', 'dscf')})
    text = report(summary)
    assert '20.000 a.u.' in text and '0/32' in text and '不恢复' in text
    try:
        collect('/tmp/unrelated', '1', Path('/tmp'))
    except ValueError:
        pass
    else:
        raise AssertionError('Collection must remain inside the independent scratch study.')
    print('Independent-study scope and compact Chinese report checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--job')
    parser.add_argument('--output', type=Path, default=Path('docs/validation'))
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root and args.job:
        args.output.mkdir(parents=True, exist_ok=True)
        status_file = args.output / 'c2h4_cayley_collection_status_20261104.json'
        try:
            complete = collect(args.root, args.job, args.output, args.wait)
        except Exception as error:
            status_file.write_text(json.dumps(dict(collected=False, error=str(error), analysis_job_id=args.job), indent=2))
            raise
        status_file.write_text(json.dumps(dict(collected=complete, analysis_job_id=args.job), indent=2))
    else:
        parser.error('--root/--job or --self-check is required')
