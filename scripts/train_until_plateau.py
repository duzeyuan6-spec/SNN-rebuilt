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


def train_one_epoch(model, loader, optimizer, device, spike_reg=0.0):
    model.train()
    loss_meter = AverageMeter()
    for x in tqdm(loader, desc="train", leave=False):
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
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    config["train"]["save_dir"] = args.save_dir
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    config["train"]["epochs"] = args.max_epochs

    set_seed(config.get("seed", 123))
    device = resolve_device(config.get("device", "cuda"))

    train_set = PatchImageDataset(
        config["data"]["train_dir"],
        patch_size=config["data"]["patch_size"],
        augment=config["data"].get("augment", True),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_set = FullImageDataset(config["data"]["val_dir"], block_size=config["cs"]["block_size"])
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
    else:
        print("Training from scratch")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )

    ensure_dir(args.save_dir)
    print(f"Device: {device}")
    print(f"Params: {count_params(model) / 1e6:.6f} M")
    print(f"Max epochs: {args.max_epochs}, patience: {args.patience}, monitor: val SSIM")

    best_ssim = -1.0
    best_psnr = -1.0
    epochs_without_ssim_gain = 0

    for epoch in range(1, args.max_epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config["train"].get("spike_reg", 0.0),
        )
        val_psnr, val_ssim = validate(model, val_loader, device)
        sample = next(iter(train_loader)).to(device)
        ops = estimate_ops(model, model.cs(sample[:1]))

        improved_ssim = val_ssim > best_ssim + args.min_delta
        improved_psnr = val_psnr > best_psnr
        if val_ssim > best_ssim:
            best_ssim = val_ssim
        if val_psnr > best_psnr:
            best_psnr = val_psnr

        save_checkpoint(f"{args.save_dir}/last.pt", model, optimizer, epoch, best_ssim, config)
        if improved_ssim:
            save_checkpoint(f"{args.save_dir}/best_ssim.pt", model, optimizer, epoch, best_ssim, config)
            epochs_without_ssim_gain = 0
        else:
            epochs_without_ssim_gain += 1
        if improved_psnr:
            save_checkpoint(f"{args.save_dir}/best_psnr.pt", model, optimizer, epoch, best_psnr, config)

        print(
            f"Epoch {epoch:04d} | loss {loss:.4f} | val PSNR {val_psnr:.2f} | "
            f"SSIM {val_ssim:.4f} | best SSIM {best_ssim:.4f} | no_gain {epochs_without_ssim_gain} | "
            f"FR {ops['firing_rate']:.4f} | energy {ops['energy_pj'] / 1e6:.3f} uJ"
        )

        if epochs_without_ssim_gain >= args.patience:
            print(f"Early stop: no SSIM gain >= {args.min_delta} for {args.patience} epochs")
            break


if __name__ == "__main__":
    main()

