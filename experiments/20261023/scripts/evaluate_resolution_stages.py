"""Descriptive stage-8..11 diagnostics on frozen mixed model; no model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_reconstruction_v3 import load_model, pad_image
from train_reconstruction_v2 import atomic_json, image_hash
from src.data import list_images, load_gray, pil_to_tensor
from src.metrics import psnr, ssim


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    base, out = Path(a.base), Path(a.out)
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    job = base/'repository_mixed_snn_cycle_20261015/jobs/mixed_snn748894_20261015'
    manifest = json.loads((job/'split.json').read_text())
    model, state = load_model(job/'spike/best.pt', 'cuda')
    assert state['manifest_hash'] == hashlib.sha256((job/'split.json').read_bytes()).hexdigest()
    for m in model.modules():
        if hasattr(m, 'collect_activity'):
            m.collect_activity = False
    forbidden = {e['sha256'] for part in ('train', 'val') for e in manifest[part]}
    atomic_json(out/'protocol.json', dict(stages=[8,9,10,11], checkpoint=str(job/'spike/best.pt'),
        fixed_model=True, test_feedback=False, split='test', edge_mask='GT forward-difference L1 gradient > 0.1',
        scope='Intermediate outputs of final-trained network; NOT separately trained early-exit models',
        limitation='Set11 used in historical project development; does not establish unseen-project test'))
    summaries = []
    for ds, expected in [('Set11',11), ('BSD68',68), ('Urban100',100)]:
        files = list_images(base/'repo/data/test'/ds)
        assert len(files) == expected
        rows = []
        for i, path in enumerate(files):
            assert image_hash(path) not in forbidden
            x = pil_to_tensor(load_gray(path))[None].cuda()
            h,w = x.shape[-2:]
            _, stages = model(model.cs(pad_image(x,32)), return_stages=True)
            gx = x[..., :-1, 1:] - x[..., :-1, :-1]
            gy = x[..., 1:, :-1] - x[..., :-1, :-1]
            mask = gx.abs()+gy.abs() > .1
            for stage in (8,9,10,11):
                pred = stages[stage-1][..., :h, :w]
                assert torch.isfinite(pred).all()
                clipped = pred.clamp(0,1)
                px = clipped[..., :-1, 1:]-clipped[..., :-1, :-1]
                py = clipped[..., 1:, :-1]-clipped[..., :-1, :-1]
                edge_error = (clipped-x).abs()[..., :-1, :-1][mask]
                rows.append(dict(image=str(path), stage=stage, psnr=psnr(pred,x), ssim=ssim(pred,x),
                    gradient_mae=float(((px-gx).abs()+(py-gy).abs()).mean()),
                    edge_pixel_mae=float(edge_error.mean()) if edge_error.numel() else None))
            atomic_json(out/'status.json', dict(state='running',dataset=ds,images=i+1,total=expected))
        atomic_json(out/(ds+'_images.json'), rows)
        for stage in (8,9,10,11):
            group = [r for r in rows if r['stage']==stage]
            summaries.append(dict(dataset=ds,stage=stage,n_images=len(group),
                **{key:float(np.mean([r[key] for r in group if r[key] is not None]))
                   for key in ('psnr','ssim','gradient_mae','edge_pixel_mae')}))
    atomic_json(out/'summary.json', summaries)
    atomic_json(out/'status.json', dict(state='completed'))


if __name__ == '__main__':
    main()
