"""Frozen, fail-fast extensions; never select a new main model from test results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--gpu', type=int, default=5)
    a = p.parse_args()
    root, base = Path(a.root), Path(a.base)
    status = root/'queue_status.json'
    if status.exists():
        raise RuntimeError('Queue already exists; inspect before any explicit recovery')
    manifest = base/'repository_mixed_snn_cycle_20261015/jobs/mixed_snn748894_20261015/split.json'
    train = base/'reconstruction_v2_20261013/data/train/Train400'
    tests = base/'repo/data/test'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    completed = []

    def run(label, cmd):
        save(status, dict(state='running', current=label, completed=completed, gpu=a.gpu))
        with (root/(label+'.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', *cmd], env=env, cwd=CODE,
                                     stdout=log, stderr=subprocess.STDOUT)
            save(status, dict(state='running', current=label, child_pid=child.pid, completed=completed, gpu=a.gpu))
            rc = child.wait()
        if rc:
            raise RuntimeError(f'{label} exited {rc}; inspect retained log/checkpoints')
        completed.append(label)

    try:
        run('resolution_stages', ['scripts/evaluate_resolution_stages.py', '--base', str(base),
                                 '--out', str(root/'resolution_stages')])
        for tag, rate in [('mr010', '.10'), ('mr050', '.50')]:
            rate_root = root/tag
            rate_root.mkdir(exist_ok=False)
            common = ['scripts/train_reconstruction_v2.py', '--architecture', 'repository',
                      '--train-root', str(train), '--test-root', str(tests), '--manifest', str(manifest),
                      '--channels', '49', '--stages', '11', '--time-steps', '8', '--mr', rate,
                      '--block-size', '32', '--seed', '123', '--epochs', '200', '--seconds', '0',
                      '--selection-metric', 'ssim', '--structure-weight', '.4', '--full-val',
                      '--stability-guard', '--memory-fraction', '.15', '--structure-mode', 'metric',
                      '--intermediate-weight', '.05', '--spatial-dilation', '2', '--burst-calibration', 'rms',
                      '--learnable-sensing']
            teacher = rate_root/'teacher/best.pt'
            for branch, kind in [('teacher','ann'), ('spike','spike'), ('ann','ann')]:
                cmd = common + ['--out', str(rate_root/branch), '--kind', kind]
                if branch != 'teacher':
                    cmd += ['--freeze-sensing', '--init-from', str(teacher), '--quant-warmup', '40']
                if branch == 'spike':
                    cmd += ['--snn-spatial-stride', '2', '--snn-fullres-tail', '3']
                run(tag+'_'+branch, cmd)
                result = json.loads((rate_root/branch/'status.json').read_text())
                if result['state'] != 'completed':
                    raise RuntimeError('Incomplete epoch budget')
            ann = rate_root/'ann/best.pt'
            digest = hashlib.sha256(ann.read_bytes()).hexdigest()
            for ds in ('Set11','BSD68','Urban100'):
                run(tag+'_test_'+ds, ['scripts/evaluate_reconstruction_v3.py',
                    '--spike', str(rate_root/'spike/best.pt'), '--ann', str(ann),
                    '--fixed-ann-sha256', digest, '--manifest', str(manifest), '--split', 'test',
                    '--test-root', str(tests/ds), '--out', str(rate_root/('test_'+ds)), '--device', 'cuda'])
            run(tag+'_octuf', ['scripts/train_multirate_octuf.py', '--method', 'octuf',
                '--source', str(base/'ann_comparison_20261020/code/baselines'), '--manifest', str(manifest),
                '--sensing-checkpoint', str(teacher), '--test-root', str(tests),
                '--out', str(rate_root/'octuf'), '--epochs', '400', '--seed', '123', '--memory-fraction', '.15'])
        save(status, dict(state='completed', completed=completed, gpu=a.gpu))
    except BaseException as exc:
        save(status, dict(state='failed', error=repr(exc), completed=completed, gpu=a.gpu))
        raise


if __name__ == '__main__':
    main()
