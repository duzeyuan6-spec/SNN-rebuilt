"""Evaluate declared seed/loss controls without weakening the paired evaluator."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_reconstruction_v3 import load_model,evaluate_image
from train_reconstruction_v2 import image_hash,atomic_json
from src.data import list_images,load_gray,pil_to_tensor

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for k in ('checkpoint','base','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.15)
    selected=base/'repository_mixed_snn_cycle_20261015/jobs/mixed_snn748894_20261015'
    manifest_path=selected/'split.json';manifest=json.loads(manifest_path.read_text())
    model,state=load_model(a.checkpoint,'cuda')
    reference=torch.load(selected/'spike/best.pt',map_location='cpu',weights_only=False)
    allowed={'seed','spike_reg','quant_warmup','out','resume','device','memory_fraction'}
    for k,v in reference['args'].items():
        if k not in allowed and state['args'].get(k)!=v:raise ValueError('Undeclared protocol change: '+k)
    assert state['manifest_hash']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert sum(p.numel() for p in model.parameters())==748894
    assert torch.equal(model.cs.phi.cpu(),reference['model']['cs.phi'])
    forbidden={e['sha256'] for part in ('train','val') for e in manifest[part]}
    atomic_json(out/'protocol.json',dict(args=state['args'],epoch=state['epoch'],split='test',
        scope='SNN training control; not equal-seed paired ANN training',frozen_test_selection=True))
    summaries=[]
    for ds,n in [('Set11',11),('BSD68',68),('Urban100',100)]:
        files=list_images(base/'repo/data/test'/ds);assert len(files)==n
        assert all(image_hash(f) not in forbidden for f in files)
        rows=[]
        for i,f in enumerate(files):
            x=pil_to_tensor(load_gray(f))[None].cuda();r=evaluate_image(model,x)
            rows.append(dict(image=str(f),**r));atomic_json(out/'status.json',dict(state='running',dataset=ds,images=i+1,total=n))
        atomic_json(out/(ds+'_images.json'),rows)
        summaries.append(dict(dataset=ds,n_images=n,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows]))))
    atomic_json(out/'summary.json',summaries);atomic_json(out/'status.json',dict(state='completed'))

if __name__=='__main__':main()
