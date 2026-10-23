"""Sequential, fail-fast queue. Never overwrites or silently repeats a job."""
import argparse,json,subprocess,sys
from pathlib import Path
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);a=p.parse_args()
    root=Path(a.root);base=Path(a.base);completed=[]
    status=root/'queue_status.json'
    for method in ('ista_plus','ista_pp','octuf'):
        job=root/(method+'_seed123')
        if job.exists():raise RuntimeError(f'Refusing duplicate job: {job}')
        save(status,dict(state='running',current=method,completed=completed,planned=['ista_plus','ista_pp','octuf']))
        cmd=[sys.executable,'-u',str(root/'code/scripts/train_ann_comparators.py'),'--method',method,
             '--source',str(root/'code/baselines'),'--manifest',str(base/'repository_mixed_snn_cycle_20261015/jobs/mixed_snn748894_20261015/split.json'),
             '--sensing-checkpoint',str(base/'repository_sensing_cycle_20261015/jobs/sensing748894_20261015/teacher/best.pt'),
             '--test-root',str(base/'repo/data/test'),'--out',str(job),'--epochs','400','--memory-fraction','.20']
        with (root/(method+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            save(status,dict(state='running',current=method,pid=child.pid,completed=completed,planned=['ista_plus','ista_pp','octuf']))
            code=child.wait()
        if code:
            save(status,dict(state='failed',current=method,returncode=code,completed=completed));raise SystemExit(code)
        result=json.loads((job/'status.json').read_text());assert result['state']=='completed'
        completed.append(method)
    save(status,dict(state='completed',completed=completed))
if __name__=='__main__':main()
