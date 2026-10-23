"""One model per process, exclusive-device measurements; retain failed attempts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import threading
import time
from benchmark_reconstruction_power import query, ensure_idle, integrate_power


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def worker(a):
    info = query('uuid,name,driver_version', a.gpu)[0]
    uuid = info[0].strip()
    ensure_idle(uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from evaluate_reconstruction_v3 import load_model
    from multirate_octuf_adapter import OfficialCS
    torch.set_num_threads(4)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(.30)
    from unified_load import load
    model, phi, forward = load(a.checkpoint)
    for module in model.modules():
        if hasattr(module, 'collect_activity'):
            module.collect_activity = False
    # No second model and no GPU optimizer states are resident.
    torch.manual_seed(123)  # identical input independent of constructor RNG use
    x = torch.rand(1,1,a.size,a.size,device='cuda')
    with torch.inference_mode():
        ref = forward(x)
        assert torch.isfinite(ref).all() and ref.shape == x.shape
        torch.testing.assert_close(ref, forward(x), rtol=0, atol=0)
        del ref
        for _ in range(10):
            forward(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    samples, errors = [], []
    stop = threading.Event()
    def sample():
        try:
            while not stop.is_set():
                ensure_idle(uuid)
                samples.append((time.perf_counter(),float(query('power.draw',a.gpu)[0][0])))
                stop.wait(.1)
        except Exception as exc:
            errors.append(repr(exc))
    monitor = threading.Thread(target=sample)
    monitor.start()
    start = time.perf_counter()
    count = 0
    try:
        with torch.inference_mode():
            while time.perf_counter()-start < a.seconds:
                if errors:
                    raise RuntimeError(errors[0])
                forward(x)
                torch.cuda.synchronize()
                count += 1
        elapsed = time.perf_counter()-start
    finally:
        stop.set()
        monitor.join()
    ensure_idle(uuid)
    if errors:
        raise RuntimeError(errors[0])
    covered = samples[-1][0]-samples[0][0] if len(samples)>1 else 0
    if covered < elapsed*.8:
        raise RuntimeError('Insufficient power coverage')
    watts = integrate_power(samples)/covered
    save(Path(a.out),dict(kind=a.kind,checkpoint=a.checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),device=info,
        sensing_shape=list(phi.shape),sensing_sha256=hashlib.sha256(phi.contiguous().numpy().tobytes()).hexdigest(),
        size=a.size,precision='FP32',batch_size=1,images=count,elapsed_seconds=elapsed,
        latency_ms=elapsed/count*1000,throughput_images_s=count/elapsed,
        sampled_average_board_watts=watts,estimated_board_joules_per_image=watts*elapsed/count,
        power_samples=samples,cudnn_deterministic=True,scope='sensing plus reconstruction; excludes file IO and host transfer; board not system energy'))


def queue(a):
    root = Path(a.out)
    root.mkdir(exist_ok=False)
    plan = json.loads(Path(a.plan).read_text())
    completed = []
    save(root/'protocol.json',plan)
    for group in plan['groups']:
        # Missing future-rate checkpoints are dependencies, not training failures.
        while not all(Path(v['checkpoint']).is_file() for v in group['models']):
            save(root/'status.json',dict(state='waiting_checkpoints',group=group['name'],completed=completed))
            time.sleep(60)
        if group.get('training_completion'):
            while True:
                status_path=Path(group['training_completion'])
                if status_path.exists() and json.loads(status_path.read_text()).get('state')=='completed':
                    break
                save(root/'status.json',dict(state='waiting_training_completion',group=group['name'],completed=completed))
                time.sleep(60)
        attempt = 0
        while True:
            occupied={r[0].strip() for r in query('gpu_uuid,pid',compute=True)}
            idle=[int(r[0]) for r in query('index,uuid') if r[1].strip() not in occupied]
            if not idle:
                save(root/'status.json',dict(state='waiting_exclusive_gpu',group=group['name'],completed=completed))
                time.sleep(60)
                continue
            gpu = next((g for g in (6,4,0,1,2,3,7,5) if g in idle),idle[0])
            folder=root/(group['name']+'_attempt'+str(attempt))
            folder.mkdir(exist_ok=False)
            attempt+=1
            rows=[]
            failure=None
            for repeat in range(3):
                models=group['models'] if repeat%2==0 else list(reversed(group['models']))
                for model in models:
                    output=folder/(str(repeat)+'_'+model['kind']+'.json')
                    command=[sys.executable,'-u',__file__,'--worker','--gpu',str(gpu),
                        '--checkpoint',model['checkpoint'],'--kind',model['kind'],
                        '--source',plan['source'],'--size',str(group['size']),'--seconds','20','--out',str(output)]
                    with output.with_suffix('.log').open('x') as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        save(root/'status.json',dict(state='running',group=group['name'],gpu=gpu,
                            repeat=repeat,model=model['kind'],child_pid=child.pid,completed=completed))
                        rc=child.wait()
                    if rc:
                        failure=output.with_suffix('.log').read_text()
                        break
                    rows.append(json.loads(output.read_text()))
                if failure:
                    break
            if failure:
                save(folder/'failure.json',dict(valid=False,error=failure))
                if 'Competing compute PIDs' in failure:
                    save(root/'status.json',dict(state='waiting_exclusive_gpu',group=group['name'],invalid_attempt=str(folder),completed=completed))
                    time.sleep(60)
                    continue
                save(root/'status.json',dict(state='failed',group=group['name'],error=failure,completed=completed))
                raise RuntimeError('Measurement failed; no blind retry')
            for shape in {tuple(r['sensing_shape']) for r in rows}:
                assert len({r['sensing_sha256'] for r in rows if tuple(r['sensing_shape'])==shape})==1
            assert len({r['device'][0] for r in rows})==1
            save(folder/'summary.json',dict(valid=True,rows=rows,group=group,
                protocol='three alternating rounds, separate model process, FP32 batch1 >=20 seconds'))
            completed.append(dict(group=group['name'],summary=str(folder/'summary.json')))
            break
    save(root/'status.json',dict(state='completed',completed=completed))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--plan')
    p.add_argument('--checkpoint')
    p.add_argument('--source')
    p.add_argument('--kind')
    p.add_argument('--gpu',type=int)
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    worker(a) if a.worker else queue(a)
