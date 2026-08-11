import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import FullImageDataset, PatchImageDataset, collate_full_images
from src.energy import estimate_ops
from src.metrics import AverageMeter, psnr, ssim
from src.models import build_model
from src.utils import count_params, ensure_dir, load_config, resolve_device, save_checkpoint, set_seed


def train_one_epoch(model, loader, optimizer, device, log_every, spike_reg=0.0):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)
    for step, x in enumerate(pbar, 1):
        x = x.to(device)
        y = model.cs(x)
        pred = model(y)
        loss = F.l1_loss(pred, x)
        if getattr(model, "kind", "ann") == "spike" and spike_reg > 0:
            loss = loss + spike_reg * model.firing_rate()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", fr=f"{model.firing_rate():.4f}")
    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    for batch in tqdm(loader, desc="val", leave=False):
        for x, _name in batch:
            x = x.unsqueeze(0).to(device)
            y = model.cs(x)
            pred = model(y).clamp(0, 1)
            psnr_meter.update(psnr(pred, x))
            ssim_meter.update(ssim(pred, x))
    return psnr_meter.avg, ssim_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from", default=None, help="initialize model weights from an existing checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 123))
    device = resolve_device(config.get("device", "cuda"))

    train_set = PatchImageDataset(
        config["data"]["train_dir"],
        patch_size=config["data"]["patch_size"],
        augment=config["data"].get("augment", True),
    )
    print(f"Train images: {len(train_set)} from {config['data']['train_dir']}")
    train_loader = DataLoader(
        train_set,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_set = FullImageDataset(config["data"]["val_dir"], block_size=config["cs"]["block_size"])
    print(f"Val images: {len(val_set)} from {config['data']['val_dir']}")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_full_images)

    model = build_model(config).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Initialized from {args.init_from}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )

    save_dir = config["train"]["save_dir"]
    ensure_dir(save_dir)
    print(f"Device: {device}")
    print(f"Params: {count_params(model) / 1e6:.3f} M")

    best_psnr = -1.0
    for epoch in range(1, config["train"]["epochs"] + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config["train"].get("log_every", 50),
            config["train"].get("spike_reg", 0.0),
        )
        val_psnr, val_ssim = validate(model, val_loader, device)

        sample = next(iter(train_loader)).to(device)
        sample_y = model.cs(sample[:1])
        ops = estimate_ops(model, sample_y)

        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | val PSNR {val_psnr:.2f} | "
            f"SSIM {val_ssim:.4f} | FR {ops['firing_rate']:.4f} | energy {ops['energy_pj'] / 1e6:.3f} uJ"
        )

        save_checkpoint(f"{save_dir}/last.pt", model, optimizer, epoch, best_psnr, config)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(f"{save_dir}/best.pt", model, optimizer, epoch, best_psnr, config)


if __name__ == "__main__":
    main()
