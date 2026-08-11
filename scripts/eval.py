import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import FullImageDataset, collate_full_images
from src.energy import estimate_ops
from src.metrics import AverageMeter, psnr, ssim
from src.models import build_model
from src.utils import count_params, resolve_device


@torch.no_grad()
def eval_set(model, root, set_name, device):
    dataset = FullImageDataset(Path(root) / set_name, block_size=model.cs.block_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_full_images)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    for batch in tqdm(loader, desc=set_name, leave=False):
        for x, _name in batch:
            x = x.unsqueeze(0).to(device)
            y = model.cs(x)
            pred = model(y).clamp(0, 1)
            psnr_meter.update(psnr(pred, x))
            ssim_meter.update(ssim(pred, x))
    return psnr_meter.avg, ssim_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--sets", nargs="+", default=["Set11", "BSD68", "Urban100"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    if args.device is not None:
        config["device"] = args.device
    device = resolve_device(config.get("device", "cuda"))

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Params: {count_params(model) / 1e6:.3f} M")

    for set_name in args.sets:
        p, s = eval_set(model, args.data_root, set_name, device)
        print(f"{set_name}: PSNR {p:.2f}, SSIM {s:.4f}")

    dummy = torch.zeros(1, 1, config["data"]["patch_size"], config["data"]["patch_size"], device=device)
    sample_y = model.cs(dummy)
    ops = estimate_ops(model, sample_y)
    print(
        "Efficiency: "
        f"MACs={ops['macs']:.0f}, ACs={ops['acs']:.0f}, "
        f"FR={ops['firing_rate']:.4f}, Energy={ops['energy_pj'] / 1e6:.3f} uJ"
    )


if __name__ == "__main__":
    main()
