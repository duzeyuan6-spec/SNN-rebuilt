"""Paired sampled GPU-board power measurement on an otherwise idle device.

Includes acquisition + reconstruction, FP32, resident input. No claim of total
system energy or event-driven hardware energy. Refuses competing compute jobs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def query(fields,gpu=None,compute=False):
    cmd=['nvidia-smi',f"--query-{'compute-apps' if compute else 'gpu'}={fields}",'--format=csv,noheader,nounits']
    if gpu is not None:cmd.extend(['-i',str(gpu)])
    return list(csv.reader(io.StringIO(subprocess.check_output(cmd,text=True))))


def ensure_idle(uuid):
    others=[int(pid.strip()) for uid,pid in query('gpu_uuid,pid',compute=True)
            if uid.strip()==uuid and int(pid.strip())!=os.getpid()]
    if others:raise RuntimeError(f'Competing compute PIDs {others}; power result would not isolate this model')


def integrate_power(samples):
    if len(samples)<2:raise ValueError('Need at least two power samples')
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(samples,samples[1:]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spike',required=True);p.add_argument('--ann',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--size',type=int,default=256)
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--fixed-ann-sha256')
    a=p.parse_args()
    if a.seconds<10 or a.repeats<3:p.error('Use at least 10 seconds and 3 paired repeats')
    info=query('uuid,name,driver_version',a.gpu)[0];uuid=info[0].strip();ensure_idle(uuid)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from evaluate_reconstruction_v3 import load_model,check_pair
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.15,0)
    pairs=[load_model(f,torch.device('cuda:0')) for f in (a.spike,a.ann)]
    check_pair([v[1] for v in pairs],[v[0] for v in pairs],hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
               fixed_ann_sha256=a.fixed_ann_sha256,ann_checkpoint_sha256=hashlib.sha256(Path(a.ann).read_bytes()).hexdigest())
    for model,_ in pairs:
        for module in model.modules():
            if hasattr(module,'collect_activity'):module.collect_activity=False
    if a.size%pairs[0][0].cs.block_size:p.error('size must be divisible by measurement block size')
    x=torch.rand(1,1,a.size,a.size,device='cuda');rows=[]
    for repeat in range(a.repeats):
        for index in ([0,1] if repeat%2==0 else [1,0]):
            model=pairs[index][0];ensure_idle(uuid)
            with torch.inference_mode():
                for _ in range(10):model(model.cs(x))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            samples=[];errors=[];stop=threading.Event()
            def sample():
                try:
                    while not stop.is_set():
                        ensure_idle(uuid)
                        power=float(query('power.draw',a.gpu)[0][0])
                        samples.append((time.perf_counter(),power))
                        stop.wait(.1)
                except Exception as e:errors.append(repr(e))
            monitor=threading.Thread(target=sample);monitor.start()
            start=time.perf_counter();count=0
            try:
                with torch.inference_mode():
                    while time.perf_counter()-start<a.seconds:
                        if errors:raise RuntimeError(errors[0])
                        model(model.cs(x));torch.cuda.synchronize();count+=1
                elapsed=time.perf_counter()-start
            finally:
                stop.set();monitor.join()
            ensure_idle(uuid)
            if errors:raise RuntimeError(errors[0])
            covered=samples[-1][0]-samples[0][0] if len(samples)>1 else 0
            if covered<elapsed*.8:raise RuntimeError('Insufficient power sampling coverage')
            average_power=integrate_power(samples)/covered
            rows.append(dict(kind=['spike','ann'][index],repeat=repeat,images=count,
                elapsed_seconds=elapsed,latency_ms=elapsed/count*1000,
                sampled_average_board_watts=average_power,
                estimated_board_joules_per_image=average_power*elapsed/count,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),power_samples=samples))
            (out/'samples.json').write_text(json.dumps(rows,indent=2))
    summary=dict(device=info,protocol=vars(a),rows=rows,
        activity_statistics='Disabled for both matched models where supported; inference outputs unchanged',
        checkpoint_sha256=[hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (a.spike,a.ann)],
        measurement='Sampled GPU board power integrated over sustained inference; includes idle board power',
        limitations=['Not total-system joules','Sampling may miss transients','No neuromorphic implementation measured',
                     'Checks other compute PIDs; dedicated non-display GPU and stable clocks still required'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
