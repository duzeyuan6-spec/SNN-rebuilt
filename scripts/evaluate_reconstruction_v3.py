"""Frozen-checkpoint paired evaluation; full images, explicit intervention scope.

Default input is the frozen validation manifest. Test evaluation is explicit,
and records disjointness against BOTH training and validation images. No tiling,
resizing or cropped-away borders: replicate pad right/bottom then unpad metrics.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import list_images, load_gray, pil_to_tensor
from src.metrics import psnr,ssim
from src.reconstruction_v3 import StableStagewiseRecon
from src.reconstruction_v4 import EfficientStagewiseRecon
from src.reconstruction_v5 import SignedStagewiseRecon
from src.models import build_repository_reconstruction
from train_reconstruction_v2 import atomic_json,image_hash


def pad_image(x,block):
    h,w=x.shape[-2:]
    return F.pad(x,(0,(-w)%block,0,(-h)%block),mode='replicate')


def paired_interval(deltas,seed=20261013,repeats=10000):
    values=np.asarray(deltas,dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Paired differences must be finite and nonempty')
    rng=np.random.default_rng(seed)
    means=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(repeats)])
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                n_images=len(values),scope='paired image bootstrap; not training-seed uncertainty')


def load_model(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    a=state['args']
    if a.get('architecture') not in ('v3','v4','v5','repository'):raise ValueError('Unsupported checkpoint architecture')
    model_class={'v3':StableStagewiseRecon,'v4':EfficientStagewiseRecon,'v5':SignedStagewiseRecon,
                 'repository':build_repository_reconstruction}[a['architecture']]
    extra=dict(spatial_dilation=a.get('spatial_dilation',1),learnable_sensing=a.get('learnable_sensing',False),
               freeze_sensing=a.get('freeze_sensing',False),burst_calibration=a.get('burst_calibration','rms'),
               snn_spatial_stride=a.get('snn_spatial_stride',1),snn_fullres_tail=a.get('snn_fullres_tail',0)) if a['architecture']=='repository' else {}
    model=model_class(kind=a['kind'],channels=a['channels'],stages=a['stages'],
          time_steps=a['time_steps'],block_size=a['block_size'],measurement_rate=a['mr'],**extra)
    model.load_state_dict(state['model'],strict=True)
    return model.to(device).eval(),state


def check_pair(states,models,manifest_hash,*,fixed_ann_sha256=None,ann_checkpoint_sha256=None):
    if fixed_ann_sha256 and fixed_ann_sha256!=ann_checkpoint_sha256:
        raise ValueError('Fixed ANN checkpoint SHA256 differs from registered baseline')
    for state in states:
        if state['manifest_hash']!=manifest_hash:raise ValueError('Checkpoint/manifest mismatch')
        if state['args'].get('train_intervention','none')!='none':
            raise ValueError('This paired evaluator expects unablated training; use the ablation-specific protocol')
    for key in ('seed','epochs','channels','stages','time_steps','block_size','mr','batch_size','patch_size','repeats','lr'):
        if states[0]['args'][key]!=states[1]['args'][key]:raise ValueError(f'Unmatched protocol: {key}')
    if [m.kind for m in models]!=['spike','ann']:raise ValueError('Expected SNN then ANN')
    if states[0]['args'].get('initialization_sha256')!=states[1]['args'].get('initialization_sha256'):
        raise ValueError('Different initialization provenance')
    if sum(p.numel() for p in models[0].parameters())!=sum(p.numel() for p in models[1].parameters()):
        if states[0]['args'].get('architecture')=='repository':raise ValueError('Repository paired comparison must have exactly equal parameter counts')
    for key,default in [('architecture','v3'),('selection_metric','psnr'),('structure_weight',.1),('full_val',False),
                        ('structure_mode','legacy'),('intermediate_weight',.2),('spatial_dilation',1),
                        ('learnable_sensing',False),('freeze_sensing',False),('burst_calibration','rms'),
                        ('snn_spatial_stride',1),('snn_fullres_tail',0)]:
        if fixed_ann_sha256 and key in ('burst_calibration','snn_spatial_stride','snn_fullres_tail'):continue
        if states[0]['args'].get(key,default)!=states[1]['args'].get(key,default):
            raise ValueError(f'Unmatched protocol: {key}')
    if not torch.equal(models[0].cs.phi,models[1].cs.phi):raise ValueError('Different sensing matrices')


@torch.inference_mode()
def evaluate_image(model,x,intervention=None):
    h,w=x.shape[-2:]
    padded=pad_image(x,model.cs.block_size)
    y=model.cs(padded)
    pred=model(y,intervention=intervention)[...,:h,:w]
    if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite reconstruction')
    return dict(psnr=psnr(pred,x),ssim=ssim(pred,x),
                raw_mse=float((pred-x).square().mean()),
                clipped_fraction=float(((pred<0)|(pred>1)).float().mean()),
                firing_rate=model.firing_rate() if intervention!='identity' else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spike',required=True);p.add_argument('--ann',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--test-root')
    p.add_argument('--device',default='cpu');p.add_argument('--memory-fraction',type=float,default=.15)
    p.add_argument('--target-ssim',type=float,default=.90)
    p.add_argument('--fixed-ann-sha256',help='Verify a frozen ANN baseline; allow declared SNN-only spatial/calibration changes')
    a=p.parse_args()
    if not 0<a.target_ssim<=1:p.error('target-ssim must be in (0,1]')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda':
        device=torch.device('cuda',device.index if device.index is not None else 0)
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
    manifest_path=Path(a.manifest);manifest=json.loads(manifest_path.read_text())
    mh=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pairs=[load_model(path,device) for path in (a.spike,a.ann)]
    models=[v[0] for v in pairs];states=[v[1] for v in pairs]
    check_pair(states,models,mh,fixed_ann_sha256=a.fixed_ann_sha256,
               ann_checkpoint_sha256=hashlib.sha256(Path(a.ann).read_bytes()).hexdigest())
    if a.split=='val':
        entries=manifest['val']
    else:
        if not a.test_root:raise ValueError('--test-root required for test split')
        forbidden={v['sha256'] for part in ('train','val') for v in manifest[part]}
        entries=[dict(path=str(f),sha256=image_hash(f)) for f in list_images(a.test_root)]
        if any(v['sha256'] in forbidden for v in entries):raise ValueError('Test overlaps train/validation')
    config=dict(args=vars(a),manifest_sha256=mh,checkpoint_epochs=[s['epoch'] for s in states],
        checkpoint_sha256=[hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (a.spike,a.ann)],
        padding='replicate right/bottom to block multiple; metrics on original size',
        metrics='mean per-image PSNR and skimage SSIM on clamped [0,1]; no border shave',
        interventions='inference perturbations, not retrained ablations',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [Path(__file__),ROOT/'src/models.py',ROOT/'src/spiking.py',ROOT/'src/energy.py',
                                 ROOT/'src/reconstruction_v3.py',ROOT/'src/reconstruction_v4.py',ROOT/'src/reconstruction_v5.py',ROOT/'src/metrics.py']},
        model_training_args=[s['args'] for s in states],
        active_parameters=[sum(p.numel() for p in m.parameters()) for m in models],
        comparison_scope='SNN-only candidate versus SHA-verified fixed ANN' if a.fixed_ann_sha256 else 'matched paired training')
    atomic_json(out/'protocol.json',config)
    start=time.monotonic();rows=[]
    try:
        for entry in entries:
            if image_hash(entry['path'])!=entry['sha256']:raise ValueError('Image changed since manifest')
            x=pil_to_tensor(load_gray(entry['path']))[None].to(device)
            if min(x.shape[-2:])<7:raise ValueError('SSIM requires dimensions >= 7')
            row=dict(image=entry['path'],sha256=entry['sha256'],height=x.shape[-2],width=x.shape[-1])
            for label,model,intervention in [('spike',models[0],None),('ann',models[1],None)]+[
                    (mode,models[0],mode) for mode in ('identity','reset_memory','zero_spikes','shift_spikes')]:
                row[label]=evaluate_image(model,x,intervention)
            rows.append(row)
            atomic_json(out/'images.json',rows)
            atomic_json(out/'status.json',dict(state='running',images=len(rows),total=len(entries),
                                             elapsed_seconds=time.monotonic()-start))
            print(json.dumps(dict(images=len(rows),total=len(entries),spike=row['spike']['psnr'],ann=row['ann']['psnr'])),flush=True)
        labels=('spike','ann','identity','reset_memory','zero_spikes','shift_spikes')
        summary=dict(split=a.split,n_images=len(rows),
            means={label:{metric:float(np.mean([r[label][metric] for r in rows]))
                          for metric in ('psnr','ssim','raw_mse','clipped_fraction')} for label in labels},
            paired_psnr={label:paired_interval([r['spike']['psnr']-r[label]['psnr'] for r in rows])
                         for label in labels if label!='spike'},
            limitations=['One training seed; image bootstrap does not establish seed robustness',
                          'Inference interventions change feature distribution; retraining controls still required'])
        summary['target_check']=dict(target_ssim=a.target_ssim,
            mean_ssim_reaches_target=summary['means']['spike']['ssim']>=a.target_ssim,
            fraction_images_reaching_target=float(np.mean([r['spike']['ssim']>=a.target_ssim for r in rows])),
            better_than_matched_ann_ssim=summary['means']['spike']['ssim']>summary['means']['ann']['ssim'],
            hardware_energy_verified=False,overall_requirement_verified=False,
            note='Mean SSIM is not a per-image guarantee; no lowest-energy claim without hardware measurements')
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'status.json',dict(state='completed',images=len(rows),elapsed_seconds=time.monotonic()-start))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='failed',images=len(rows),error=repr(e)))
        raise


if __name__=='__main__':main()
