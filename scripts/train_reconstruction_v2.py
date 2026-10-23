"""Independent train/validation split, matched training, exact epoch resume.

No benchmark test metrics are consulted. All new runs train from scratch under
the same fixed sensing operator; old test-selected checkpoints are not reused.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PatchImageDataset, FullImageDataset, list_images, load_gray, pil_to_tensor
from src.metrics import psnr, ssim
from src.reconstruction_v2 import StagewiseRecon
from src.reconstruction_v3 import StableStagewiseRecon
from src.reconstruction_v4 import EfficientStagewiseRecon
from src.reconstruction_v5 import SignedStagewiseRecon
from src.energy import estimate_ops
from src.models import build_repository_reconstruction


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def image_hash(path):
    image = load_gray(path)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare_split(root, out, test_root=None, seed=20261013):
    root, out = Path(root).resolve(), Path(out)
    files = list_images(root)
    unique = {}
    for p in files:
        unique.setdefault(image_hash(p), str(p))
    if test_root:
        test_hashes = {image_hash(p) for p in list_images(test_root)}
        overlap = set(unique) & test_hashes
        if overlap:
            raise ValueError(f'Train/test image overlap: {len(overlap)}; choose clean training data')
    items = sorted(unique.items())
    random.Random(seed).shuffle(items)
    nval = max(2, round(len(items) * .1))
    if len(items) - nval < 4:
        raise ValueError('Need at least six unique training-source images')
    manifest = dict(seed=seed, source=str(root), duplicates_removed=len(files)-len(items),
                    train=[dict(sha256=h, path=p) for h,p in items[nval:]],
                    val=[dict(sha256=h, path=p) for h,p in items[:nval]],
                    test_overlap_checked=bool(test_root))
    if out.exists():
        if json.loads(out.read_text()) != manifest:
            raise ValueError('Refusing to replace a different frozen manifest')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out, manifest)
    return manifest


def structure_loss(pred, target):
    mu_p, mu_t = F.avg_pool2d(pred, 7, 1, 3), F.avg_pool2d(target, 7, 1, 3)
    vp = (F.avg_pool2d(pred.square(),7,1,3)-mu_p.square()).clamp_min(0)
    vt = (F.avg_pool2d(target.square(),7,1,3)-mu_t.square()).clamp_min(0)
    cov = F.avg_pool2d(pred*target,7,1,3)-mu_p*mu_t
    return 1-(((2*mu_p*mu_t+.01**2)*(2*cov+.03**2))/
              ((mu_p.square()+mu_t.square()+.01**2)*(vp+vt+.03**2))).mean()


def metric_structure_loss(pred,target):
    """7x7 uniform-window SSIM with sample covariance and valid interior.

    Matches the project's skimage metric formula for the same unclipped arrays.
    Training uses raw predictions to retain gradients outside the display range.
    """
    if min(pred.shape[-2:])<7:raise ValueError('SSIM needs dimensions >=7')
    pool=lambda value:F.avg_pool2d(value,7,1)
    mp,mt=pool(pred),pool(target)
    factor=49/48
    vp=(pool(pred.square())-mp.square())*factor
    vt=(pool(target.square())-mt.square())*factor
    cov=(pool(pred*target)-mp*mt)*factor
    score=((2*mp*mt+.01**2)*(2*cov+.03**2))/((mp.square()+mt.square()+.01**2)*(vp+vt+.03**2))
    return 1-score.mean()


def reconstruction_loss(pred, target, structure_weight=.1,structure_mode='legacy'):
    edge = F.l1_loss(pred[...,1:,:]-pred[...,:-1,:],target[...,1:,:]-target[...,:-1,:])
    edge += F.l1_loss(pred[...,:,1:]-pred[...,:,:-1],target[...,:,1:]-target[...,:,:-1])
    structural=metric_structure_loss(pred,target) if structure_mode=='metric' else structure_loss(pred,target)
    return F.l1_loss(pred,target) + structure_weight*structural + .02*edge


@torch.no_grad()
def validate(model, dataset, device, crop=128, train_intervention=None):
    model.eval()
    rows=[]
    for index,(x,name) in enumerate(dataset):
        if crop==0:x=pil_to_tensor(load_gray(dataset.files[index]))
        h,w=x.shape[-2:]
        if min(h,w) < model.cs.block_size:
            raise ValueError(f'Image too small: {name}')
        hh,ww=(min(crop,h),min(crop,w)) if crop else (h,w)
        top,left=(h-hh)//2,(w-ww)//2
        x=x[:,top:top+hh,left:left+ww][None].to(device)
        block=model.cs.block_size
        padded=F.pad(x,(0,(-ww)%block,0,(-hh)%block),mode='replicate')
        y=model.cs(padded)
        pred=model(y,intervention=train_intervention)[...,:hh,:ww]
        fr=model.firing_rate()
        identity=model(y,intervention='identity')[...,:hh,:ww]
        reset=model(y,intervention='reset_memory')[...,:hh,:ww] if model.kind=='spike' else pred
        rows.append(dict(image=name,psnr=psnr(pred,x),ssim=ssim(pred,x),
                         identity_psnr=psnr(identity,x), reset_psnr=psnr(reset,x),
                         correction_rms=float((pred-identity).square().mean().sqrt()),fr=fr))
    return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k!='image'}, rows


def save_state(path, model, optimizer, scheduler, epoch, best, stale, generator, args, manifest_hash, unstable_epochs=0, best_selection=None):
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
               epoch=epoch,best=best,stale=stale,args=vars(args),manifest_hash=manifest_hash,
               rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
               rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],best_selection=best_selection,
               loader_rng=generator.get_state(), precision='fp32',unstable_epochs=unstable_epochs,trainer_schema=2)
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-root',required=True);p.add_argument('--test-root')
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=['spike','ann'],default='spike')
    p.add_argument('--seed',type=int,default=123);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--channels',type=int,default=48);p.add_argument('--stages',type=int,default=5)
    p.add_argument('--time-steps',type=int,default=4);p.add_argument('--mr',type=float,default=.25)
    p.add_argument('--block-size',type=int,default=32);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--patch-size',type=int,default=96);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0002);p.add_argument('--spike-reg',type=float,default=.0001)
    p.add_argument('--seconds',type=float,default=36000,help='Wall time budget; 0 disables it');p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--memory-fraction',type=float,default=.20)
    p.add_argument('--architecture',choices=['v2','v3','v4','v5','repository'],default='v2')
    p.add_argument('--init-from',help='Same initialization checkpoint for both capacity-matched branches')
    p.add_argument('--quant-warmup',type=int,default=0,help='Repository burst training blend ramp; inference is always fully quantized')
    p.add_argument('--selection-metric',choices=['psnr','ssim'],default='psnr')
    p.add_argument('--structure-weight',type=float,default=.1)
    p.add_argument('--structure-mode',choices=['legacy','metric'],default='legacy')
    p.add_argument('--intermediate-weight',type=float,default=.2)
    p.add_argument('--spatial-dilation',type=int,default=1)
    p.add_argument('--learnable-sensing',action='store_true')
    p.add_argument('--freeze-sensing',action='store_true')
    p.add_argument('--burst-calibration',choices=['rms','mse'],default='rms')
    p.add_argument('--snn-spatial-stride',type=int,choices=[1,2],default=1)
    p.add_argument('--snn-fullres-tail',type=int,default=0)
    p.add_argument('--full-val',action='store_true',help='Original image dimensions, right/bottom replicate pad')
    p.add_argument('--train-intervention',choices=['none','reset_memory'],default='none',
                   help='Retrained memory ablation; applies to training and checkpoint selection')
    p.add_argument('--stability-guard',action='store_true')
    p.add_argument('--guard-drop-db',type=float,default=3.)
    p.add_argument('--guard-gradient',type=float,default=50.)
    p.add_argument('--guard-patience',type=int,default=2)
    a=p.parse_args()
    if a.freeze_sensing and not a.learnable_sensing:p.error('freeze-sensing requires learnable-sensing')
    if a.learnable_sensing and a.architecture!='repository':p.error('learnable-sensing is for repository experiment')
    if a.burst_calibration!='rms' and a.architecture!='repository':p.error('burst calibration requires repository architecture')
    if a.snn_spatial_stride!=1 and (a.architecture!='repository' or a.kind!='spike'):
        p.error('stride-2 synthesis is for repository SNN only')
    if a.snn_fullres_tail and (a.snn_spatial_stride!=2 or a.kind!='spike' or a.architecture!='repository'):
        p.error('full-resolution tail requires stride-2 repository SNN')
    if not 0<=a.snn_fullres_tail<a.stages:p.error('full-resolution tail must be smaller than stage count')
    if a.spatial_dilation<1:p.error('spatial-dilation must be positive')
    if a.spatial_dilation!=1 and a.architecture!='repository':p.error('spatial-dilation is for repository priors')
    if a.intermediate_weight<0:p.error('intermediate-weight must be nonnegative')
    if a.quant_warmup<0:p.error('quant-warmup must be nonnegative')
    if a.quant_warmup and a.architecture!='repository':p.error('quant-warmup is for repository burst priors')
    a.initialization_sha256=hashlib.sha256(Path(a.init_from).read_bytes()).hexdigest() if a.init_from else None
    if a.seconds<0:p.error('seconds must be nonnegative')
    if a.train_intervention!='none' and (a.kind!='spike' or a.architecture not in ('v3','v4','v5')):
        p.error('Retrained memory ablation requires V3/V4/V5 SNN')
    if a.structure_weight<0:p.error('structure-weight must be nonnegative')
    train_intervention=None if a.train_intervention=='none' else a.train_intervention
    if a.patch_size % a.block_size:
        p.error('patch-size must be a multiple of block-size')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:
        raise ValueError('Run already exists; use --resume or a new output directory')
    torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda':
        device=torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
    manifest=prepare_split(a.train_root,a.manifest,a.test_root)
    manifest_hash=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    train=PatchImageDataset(a.train_root,a.patch_size,True,a.repeats)
    train.files=[Path(r['path']) for r in manifest['train']]
    val=FullImageDataset(a.train_root,a.block_size);val.files=[Path(r['path']) for r in manifest['val']]
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0,generator=generator)
    architecture=dict(kind=a.kind,channels=a.channels,stages=a.stages,time_steps=a.time_steps,
                      block_size=a.block_size,measurement_rate=a.mr)
    if a.architecture=='repository':
        architecture.update(spatial_dilation=a.spatial_dilation,learnable_sensing=a.learnable_sensing,freeze_sensing=a.freeze_sensing,
                            burst_calibration=a.burst_calibration,snn_spatial_stride=a.snn_spatial_stride,
                            snn_fullres_tail=a.snn_fullres_tail)
    model_class={'v2':StagewiseRecon,'v3':StableStagewiseRecon,'v4':EfficientStagewiseRecon,'v5':SignedStagewiseRecon,
                 'repository':build_repository_reconstruction}[a.architecture]
    model=model_class(**architecture).to(device)
    if a.init_from and not a.resume:
        initial=torch.load(a.init_from,map_location='cpu',weights_only=False)
        if initial['manifest_hash']!=manifest_hash:raise ValueError('Initialization uses a different dataset split')
        for key in ('architecture','channels','stages','time_steps','block_size','mr'):
            if initial['args'][key]!=getattr(a,key):raise ValueError(f'Initialization mismatch: {key}')
        if initial['args'].get('learnable_sensing',False)!=a.learnable_sensing:
            raise ValueError('Initialization sensing parameterization mismatch')
        model.load_state_dict(initial['model'],strict=True)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=a.lr*.05)
    best=-float('inf');best_selection=-float('inf');stale=0;start_epoch=1;unstable_epochs=0
    if a.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['manifest_hash']!=manifest_hash:
            raise ValueError('Manifest changed on resume')
        if state['args'].get('architecture','v2')!=a.architecture:
            raise ValueError('Cannot resume across architecture versions')
        if state['args'].get('train_intervention','none')!=a.train_intervention:
            raise ValueError('Cannot resume across training interventions')
        for key,default in [('selection_metric','psnr'),('structure_weight',.1),('full_val',False)]:
            if state['args'].get(key,default)!=getattr(a,key):raise ValueError(f'Resume mismatch: {key}')
        for key,default in [('quant_warmup',0),('initialization_sha256',None)]:
            if state['args'].get(key,default)!=getattr(a,key):raise ValueError(f'Resume mismatch: {key}')
        for key,default in [('structure_mode','legacy'),('intermediate_weight',.2),('spatial_dilation',1),
                            ('learnable_sensing',False),('freeze_sensing',False),('burst_calibration','rms'),
                            ('snn_spatial_stride',1),('snn_fullres_tail',0)]:
            if state['args'].get(key,default)!=getattr(a,key):raise ValueError(f'Resume mismatch: {key}')
        if state.get('trainer_schema')!=2:
            raise ValueError('Use the archived trainer for old-schema checkpoints; do not mix history formats')
        for key in ('kind','channels','stages','time_steps','block_size','mr','seed','epochs',
                    'batch_size','patch_size','repeats','lr','spike_reg',
                    'stability_guard','guard_drop_db','guard_gradient','guard_patience'):
            if state['args'][key]!=getattr(a,key):raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        for slot in opt.state.values():
            for k,v in slot.items():
                if torch.is_tensor(v):slot[k]=v.to(device)
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch']);generator.set_state(state['loader_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['rng_cuda'])
        start_epoch=state['epoch']+1;best=state['best'];stale=state['stale']
        best_selection=state.get('best_selection')
        if best_selection is None:best_selection=best
        unstable_epochs=state['unstable_epochs']
        if a.stability_guard and unstable_epochs>=a.guard_patience:
            raise ValueError('This run already failed the stability guard; diagnose before starting a new experiment')
    atomic_json(out/'config.json',dict(args=vars(a),architecture=architecture,manifest_hash=manifest_hash,
                 parameters=sum(v.numel() for v in model.parameters()),
                 gradient_updated_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                 learned_sensing_parameters=model.cs.phi.numel() if a.learnable_sensing else 0,
                 torch=torch.__version__,
                 source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                for folder in ('src','scripts') for f in (ROOT/folder).glob('*.py')}))
    start=time.monotonic();finished=False
    atomic_json(out/'status.json',dict(state='running',phase='training',start_epoch=start_epoch))
    try:
        for epoch in range(start_epoch,a.epochs+1):
            if a.architecture=='repository':
                blend=min(1.,epoch/a.quant_warmup) if a.quant_warmup else 1.
                for prior in model.priors:prior.set_training_blend(blend)
            model.train();total=0.;grad_total=0.;grad_max=0.;clipped_steps=0;steps=0
            for x in loader:
                if a.seconds and time.monotonic()-start>a.seconds:
                    raise TimeoutError('Run time budget reached; resume last completed epoch')
                x=x.to(device);opt.zero_grad(set_to_none=True)
                pred,stages=model(model.cs(x),return_stages=True,intervention=train_intervention)
                loss=reconstruction_loss(pred,x,structure_weight=a.structure_weight,structure_mode=a.structure_mode)
                if len(stages)>1:
                    loss=loss+a.intermediate_weight*sum(F.l1_loss(v,x) for v in stages[:-1])/(len(stages)-1)
                if a.kind=='spike':loss=loss+a.spike_reg*model.spike_regularization()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                grad_max=max(grad_max,float(norm));clipped_steps+=int(float(norm)>1.)
                grad_total+=float(norm);opt.step();total+=float(loss.detach());steps+=1
                if steps % 50 == 0:
                    progress=dict(state='running',phase='training',epoch=epoch,step=steps,
                                  steps_per_epoch=len(loader),loss=total/steps,elapsed_seconds=time.monotonic()-start)
                    atomic_json(out/'status.json',progress)
                    print(json.dumps(progress),flush=True)
            if a.learnable_sensing and not a.freeze_sensing:
                # Retract trained sensing rows to an orthonormal subspace once
                # per epoch, before validation and before saving checkpoints.
                with torch.no_grad():
                    q,_=torch.linalg.qr(model.cs.phi.T,mode='reduced')
                    model.cs.phi.copy_(q.T)
            scores,rows=validate(model,val,device,crop=0 if a.full_val else 128,train_intervention=train_intervention)
            best=max(best,scores['psnr'])
            improved=scores[a.selection_metric]>best_selection
            if improved:best_selection=scores[a.selection_metric];stale=0
            else:stale+=1
            if a.stability_guard:
                unstable=(scores['psnr']<best-a.guard_drop_db or grad_total/steps>a.guard_gradient)
                unstable_epochs=unstable_epochs+1 if unstable else 0
            scheduler.step()
            save_state(out/'last.pt',model,opt,scheduler,epoch,best,stale,generator,a,manifest_hash,unstable_epochs,best_selection)
            if improved:
                save_state(out/'best.pt',model,opt,scheduler,epoch,best,stale,generator,a,manifest_hash,unstable_epochs,best_selection)
                atomic_json(out/'best_validation_images.json',rows)
            row=dict(epoch=epoch,loss=total/steps,gradient_norm=grad_total/steps,
                     gradient_max=grad_max,clipped_fraction=clipped_steps/steps,**scores,
                     gain_min=min(float(pr.gain.detach()) for pr in model.priors),
                     gain_max=max(float(pr.gain.detach()) for pr in model.priors),elapsed_seconds=time.monotonic()-start)
            with (out/'history.csv').open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row.keys())
                if f.tell()==0:w.writeheader()
                w.writerow(row)
            atomic_json(out/'status.json',dict(state='running',**row,best_psnr=best))
            print(json.dumps(row),flush=True)
            if a.stability_guard:
                if unstable_epochs>=a.guard_patience:
                    atomic_json(out/'status.json',dict(state='stability_stopped',**row,best_psnr=best,
                                reason='Persistent validation degradation or excessive pre-clip gradient'))
                    break
        else:
            finished=True
    except TimeoutError as e:
        atomic_json(out/'status.json',dict(state='time_budget',error=str(e),best_psnr=best))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='failed',error=repr(e)))
        raise
    if (out/'best.pt').exists() and train_intervention is None:
        ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model']);model.eval()
        x,_=val[0];x=x[None,:,:128,:128].to(device)
        atomic_json(out/'arithmetic_estimate.json',estimate_ops(model,model.cs(x)))
    if finished:atomic_json(out/'status.json',dict(state='completed',epochs=a.epochs,best_psnr=best,
                                                selection_metric=a.selection_metric,best_selection=best_selection))


if __name__=='__main__':main()
