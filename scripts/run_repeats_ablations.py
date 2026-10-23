"""Predeclared SNN-only repetitions and training-mechanism ablations.

No architecture search; no ANN updates; all tests occur after validation selection.
Seeds share a frozen teacher: conditional branch-training variance only.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def specifications():
    return [('seed124',124,1e-4,40),('seed125',125,1e-4,40),
            ('no_spike_regularizer',123,0.,40),('no_quant_warmup',123,1e-4,0)]

def command(base,out,name,seed,regularizer,warmup):
    teacher=base/'repository_sensing_cycle_20261015/jobs/sensing748894_20261015/teacher/best.pt'
    manifest=base/'repository_mixed_snn_cycle_20261015/jobs/mixed_snn748894_20261015/split.json'
    return [sys.executable,'-u',str(ROOT/'scripts/train_reconstruction_v2.py'),
        '--architecture','repository','--train-root',str(base/'reconstruction_v2_20261013/data/train/Train400'),
        '--test-root',str(base/'repo/data/test'),'--manifest',str(manifest),'--out',str(out/name),
        '--kind','spike','--channels','49','--stages','11','--time-steps','8','--mr','.25','--block-size','32',
        '--seed',str(seed),'--epochs','200','--seconds','0','--selection-metric','ssim','--structure-weight','.4',
        '--full-val','--stability-guard','--memory-fraction','.15','--structure-mode','metric','--intermediate-weight','.05',
        '--spatial-dilation','2','--burst-calibration','rms','--snn-spatial-stride','2','--snn-fullres-tail','3',
        '--learnable-sensing','--freeze-sensing','--init-from',str(teacher),'--quant-warmup',str(warmup),'--spike-reg',str(regularizer)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpu',type=int,default=4);p.add_argument('--launch',action='store_true');p.add_argument('--plan',action='store_true')
    a=p.parse_args();base=Path(a.base);out=Path(a.out).resolve()
    specs=specifications();commands=[command(base,out,*s) for s in specs]
    if a.plan:print(json.dumps(commands,indent=2));return
    if a.launch:
        out.mkdir(parents=True,exist_ok=False)
        protocol=dict(experiments=[dict(name=n,seed=s,spike_reg=r,quant_warmup=w) for n,s,r,w in specs],
            epochs_each=200,common_initialization_epochs=200,mr=.25,parameters_each=748894,
            purpose='conditional training-seed robustness and retrained mechanism controls, not model search',
            seed_scope='123 archived + 124 + 125 share frozen teacher, sensing matrix and split',
            ablation_scope='seed123; one training mechanism changed at a time; no architecture changes',
            selection='full image validation SSIM; frozen tests after training only',
            baseline='existing seed123 selected mixed SNN and fixed ANN; neither retrained nor replaced',
            commands=commands,source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('src','scripts') for f in (ROOT/folder).glob('*.py')})
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        with (out/'queue.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__,*[v for v in sys.argv[1:] if v!='--launch']],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'launch.json').write_text(json.dumps(dict(pid=child.pid)))
        print(json.dumps(dict(pid=child.pid,out=str(out))));return
    state=dict(state='running',completed=[],phase=None,child_pid=None)
    def update(**kw):
        state.update(kw);temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'status.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    try:
        for spec,cmd in zip(specs,commands):
            name=spec[0];update(phase=name)
            with (out/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                update(child_pid=child.pid);code=child.wait();update(child_pid=None)
            if code or json.loads((out/name/'status.json').read_text())['state']!='completed':
                raise RuntimeError(name+' training did not complete; preserve checkpoint and diagnose')
            evalcmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_snn_control.py'),'--checkpoint',str(out/name/'best.pt'),
                     '--base',str(base),'--out',str(out/name/'test')]
            with (out/(name+'_test.log')).open('x') as log:
                child=subprocess.Popen(evalcmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                update(phase=name+'_test',child_pid=child.pid);code=child.wait();update(child_pid=None)
            if code:raise RuntimeError(name+' test evaluation failed')
            state['completed'].append(name);update()
        update(state='completed',phase='done')
    except BaseException as e:update(state='failed',error=repr(e));raise

if __name__=='__main__':main()
