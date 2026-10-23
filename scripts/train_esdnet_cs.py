"""Train the declared ESDNet-CS adaptation, select on validation only, then test.

Preserves the selected SNN and fixed ANN. Native ESDNet capacity is reported,
not disguised as a parameter-matched baseline. No test feedback enters training.
"""
import argparse,csv,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import PatchImageDataset,load_gray,pil_to_tensor,list_images
from src.metrics import psnr,ssim
from evaluate_reconstruction_v3 import pad_image
from train_reconstruction_v2 import atomic_json,image_hash
from esdnet_cs_adapter import ESDNetCS


def main():
    p=argparse.ArgumentParser()
    for k in ('source','manifest','sensing-checkpoint','test-root','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--epochs',type=int,default=400);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cuda');p.add_argument('--memory-fraction',type=float,default=.25)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    manifest=json.loads(Path(a.manifest).read_text());train=manifest['train'];val=manifest['val']
    assert len(train)==360 and len(val)==40
    forbidden={v['sha256'] for v in train+val}
    for v in train+val:assert image_hash(v['path'])==v['sha256']
    tests={ds:list_images(Path(a.test_root)/ds) for ds in ('Set11','BSD68','Urban100')}
    assert all(image_hash(f) not in forbidden for files in tests.values() for f in files)
    state=torch.load(a.sensing_checkpoint,map_location='cpu',weights_only=False)
    phi=state['model']['cs.phi'];assert phi.shape==(256,1024)
    model=ESDNetCS(a.source,phi).to(a.device)
    dataset=PatchImageDataset(Path(train[0]['path']).parent,patch_size=96,augment=True,repeats=2)
    dataset.files=[Path(v['path']) for v in train]
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,generator=generator)
    # Accumulate four samples per optimizer step, same 180 updates per epoch.
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=1e-7)
    protocol=dict(method='ESDNet-CS (official architecture, adapted task)',args={k:v for k,v in vars(a).items() if k!='resume'},
        upstream='https://github.com/MingTian99/ESDNet',
        decoder_parameters=sum(p.numel() for p in model.parameters()),sensing_coefficients=phi.numel(),
        capacity_comparison='native ESDNet; NOT parameter matched',
        adaptations=['grayscale input/output','frozen shared Phi and adjoint initialization','torch backend','device-safe bilinear upsampling'],
        training=dict(images=360,validation_images=40,epochs=a.epochs,patch=96,repeats=2,effective_batch=4,
                      loss='MSE',optimizer='AdamW 1e-3 cosine to 1e-7',initialization='random ESDNet weights; sensing from common teacher',
                      model_selection='full-image validation SSIM',test_feedback=False),
        source_sha256=hashlib.sha256(Path(a.source).read_bytes()).hexdigest())
    best=-float('inf');start=1
    if a.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol']==protocol
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        best=saved['best_ssim'];start=saved['epoch']+1
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        generator.set_state(saved['loader_rng'])
        if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
    atomic_json(out/'protocol.json',protocol)
    @torch.inference_mode()
    def evaluate(files):
        model.eval();rows=[]
        for path in files:
            x=pil_to_tensor(load_gray(path))[None].to(a.device);h,w=x.shape[-2:]
            pred=model(pad_image(x,32))[...,:h,:w]
            if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite evaluation')
            rows.append(dict(image=str(path),psnr=psnr(pred,x),ssim=ssim(pred,x)))
        model.reset();return rows
    began=time.monotonic()
    try:
        for epoch in range(start,a.epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
            for step,x in enumerate(loader,1):
                x=x.to(a.device);pred=model(x);loss=(pred-x).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/4).backward();loss_sum+=loss.item();model.reset()
                if step%4==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                if step%40==0:atomic_json(out/'status.json',dict(state='running',phase='training',epoch=epoch,epochs=a.epochs,step=step,steps=len(loader),best_ssim=best if np.isfinite(best) else None))
            rows=evaluate([v['path'] for v in val]);score=float(np.mean([r['ssim'] for r in rows]));quality=float(np.mean([r['psnr'] for r in rows]))
            scheduler.step();improved=score>best;best=max(best,score)
            save=dict(model=model.state_dict(),epoch=epoch,best_ssim=best,protocol=protocol,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                      python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [])
            torch.save(save,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
            if improved:torch.save(save,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            row=dict(epoch=epoch,loss=loss_sum/len(loader),val_psnr=quality,val_ssim=score,best_ssim=best,elapsed_seconds=time.monotonic()-began)
            with (out/'history.csv').open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=row)
                if epoch==1:writer.writeheader()
                writer.writerow(row)
            atomic_json(out/'status.json',dict(state='running',phase='validation',epochs=a.epochs,**row));print(json.dumps(row),flush=True)
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
        for ds,files in tests.items():
            rows=evaluate(files);atomic_json(out/(ds+'.json'),dict(dataset=ds,split='test',n_images=len(rows),best_epoch=saved['epoch'],
                psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),images=rows))
        atomic_json(out/'status.json',dict(state='completed',epochs=a.epochs,best_epoch=saved['epoch'],best_ssim=best))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':main()
