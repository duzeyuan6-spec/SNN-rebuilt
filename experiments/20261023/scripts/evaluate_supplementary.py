"""Frozen variants, validation sensitivity and preselected qualitative examples."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import list_images,load_gray,pil_to_tensor
from src.metrics import psnr,ssim
from src.spiking import BurstIF
from evaluate_reconstruction_v3 import load_model,pad_image
from train_reconstruction_v2 import atomic_json,image_hash


@torch.inference_mode()
def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.25)
    base=Path(a.base);mixed=base/'repository_mixed_snn_cycle_20261015/jobs/mixed_snn748894_20261015'
    manifest=json.loads((mixed/'split.json').read_text())
    checkpoints={
        'mixed':mixed/'spike/best.pt',
        'ann':base/'repository_sensing_cycle_20261015/jobs/sensing748894_20261015/ann/best.pt',
        'full_resolution':base/'repository_sensing_cycle_20261015/jobs/sensing748894_20261015/spike/best.pt',
        'half_resolution':base/'repository_stride_snn_cycle_20261015/jobs/stride_snn748894_20261015/spike/best.pt'}
    models={k:load_model(v,a.device)[0] for k,v in checkpoints.items()}
    for m in models.values():
        assert sum(p.numel() for p in m.parameters())==748894
        assert torch.equal(m.cs.phi,models['ann'].cs.phi)
    forbidden={e['sha256'] for part in ('train','val') for e in manifest[part]}
    testfiles={d:list_images(base/'repo/data/test'/d) for d in ('Set11','BSD68','Urban100')}
    assert all(image_hash(f) not in forbidden for fs in testfiles.values() for f in fs)
    # Fixed ordering, chosen before any visual ranking: first three file names.
    visual_selection={d:[str(f) for f in fs[:3]] for d,fs in testfiles.items()}
    atomic_json(out/'protocol.json',dict(checkpoints={k:str(v) for k,v in checkpoints.items()},
        visual_selection=visual_selection,levels=[2,4,8,16],measurement_snr_db=[30,20,10],
        sensitivity_split='validation only; frozen weights; no selection or retraining',
        levels_meaning='signed count resolution, NOT sequential simulation timesteps',
        noise_definition='add iid Gaussian to CS measurements; std=rms(y)/10^(snr_db/20), identical realization for both models',
        spatial_variants='separately trained existing checkpoints; same architecture except resolution; 1 seed',
        hardware='not measured during shared GPU evaluation'))
    rows=[];start=time.monotonic()
    def predict(m,x,noise=None):
        y=m.cs(pad_image(x,32))
        if noise is not None:y=y+noise
        return m(y)[...,:x.shape[-2],:x.shape[-1]]
    def record(dataset,path,mode,pred,x,split):
        if not torch.isfinite(pred).all():raise FloatingPointError(mode)
        rows.append(dict(dataset=dataset,image=str(path),mode=mode,split=split,psnr=psnr(pred,x),ssim=ssim(pred,x)))
    def progress(phase,done,total):
        atomic_json(out/'rows.json',rows)
        atomic_json(out/'status.json',dict(state='running',phase=phase,done=done,total=total,seconds=time.monotonic()-start))
    for ds,files in testfiles.items():
        for i,path in enumerate(files):
            x=pil_to_tensor(load_gray(path))[None].to(a.device)
            outputs={k:predict(m,x) for k,m in models.items()}
            for name,pred in outputs.items():record(ds,path,name,pred,x,'test')
            if str(path) in visual_selection[ds]:
                folder=out/'visuals'/ds/path.stem;folder.mkdir(parents=True)
                for name,pred in dict(gt=x,**outputs).items():
                    Image.fromarray((pred[0,0].clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(folder/(name+'.png'))
                # Absolute-error images share one fixed [0,.2] display range.
                for name,pred in outputs.items():
                    error=((pred.clamp(0,1)-x).abs()[0,0].cpu().numpy()/.2).clip(0,1)
                    Image.fromarray((error*255).round().astype('uint8')).save(folder/(name+'_error.png'))
            progress(ds,i+1,len(files))
    nodes=[m for m in models['mixed'].modules() if isinstance(m,BurstIF)]
    originals=[(n.levels,n.quantum.clone()) for n in nodes]
    noisegen=torch.Generator(device=a.device).manual_seed(20261016)
    for i,entry in enumerate(manifest['val']):
        path=entry['path'];assert image_hash(path)==entry['sha256']
        x=pil_to_tensor(load_gray(path))[None].to(a.device)
        for level in (2,4,8,16):
            for node,(old,q) in zip(nodes,originals):node.levels=level;node.quantum.copy_(q*old/level)
            record('Train400-val',path,'count_levels_'+str(level),predict(models['mixed'],x),x,'val')
        for node,(old,q) in zip(nodes,originals):node.levels=old;node.quantum.copy_(q)
        y=models['mixed'].cs(pad_image(x,32));z=torch.randn(y.shape,device=a.device,generator=noisegen)
        for snr in (30,20,10):
            noise=z*y.square().mean().sqrt()/10**(snr/20)
            for name in ('mixed','ann'):record('Train400-val',path,name+'_measurement_snr_'+str(snr),predict(models[name],x,noise),x,'val')
        progress('validation_sensitivity',i+1,len(manifest['val']))
    summary=[]
    for ds,mode in sorted({(r['dataset'],r['mode']) for r in rows}):
        group=[r for r in rows if r['dataset']==ds and r['mode']==mode]
        summary.append(dict(dataset=ds,mode=mode,split=group[0]['split'],n_images=len(group),
            psnr=float(np.mean([r['psnr'] for r in group])),ssim=float(np.mean([r['ssim'] for r in group]))))
    atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',dict(state='completed',rows=len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    try:run(args)
    except BaseException as e:
        if Path(args.out).is_dir():atomic_json(Path(args.out)/'failure.json',dict(error=repr(e)))
        raise
