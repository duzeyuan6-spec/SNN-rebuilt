import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval import eval_set
from src.energy import estimate_ops
from src.models import build_model
from src.utils import count_params, resolve_device


FIELDS = [
    "method",
    "mr",
    "set",
    "psnr",
    "ssim",
    "params_m",
    "fr",
    "energy_uj",
    "source",
]


def parse_checkpoint_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("checkpoint spec must be METHOD:MR:PATH")
    method, mr, path = parts
    return method, mr, path


def fmt_float(value, digits=4):
    if value in ("", None):
        return "N/A"
    if isinstance(value, str) and value.strip().upper() in {"N/A", "NA", "TODO"}:
        return value.strip()
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def read_external_csv(path):
    rows = []
    if path is None:
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "method": row.get("method", ""),
                    "mr": row.get("mr", ""),
                    "set": row.get("set", ""),
                    "psnr": row.get("psnr", ""),
                    "ssim": row.get("ssim", ""),
                    "params_m": row.get("params_m", ""),
                    "fr": row.get("fr", ""),
                    "energy_uj": row.get("energy_uj", ""),
                    "source": row.get("source", "external"),
                }
            )
    return rows


@torch.no_grad()
def eval_checkpoint(spec, data_root, sets, device_override):
    method, mr, path = parse_checkpoint_spec(spec)
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    if device_override:
        config["device"] = device_override
    device = resolve_device(config.get("device", "cuda"))

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 1, config["data"]["patch_size"], config["data"]["patch_size"], device=device)
    sample_y = model.cs(dummy)
    ops = estimate_ops(model, sample_y)
    params_m = count_params(model) / 1e6
    energy_uj = ops["energy_pj"] / 1e6
    fr = ops["firing_rate"]

    rows = []
    for set_name in sets:
        psnr, ssim = eval_set(model, data_root, set_name, device)
        rows.append(
            {
                "method": method,
                "mr": mr,
                "set": set_name,
                "psnr": psnr,
                "ssim": ssim,
                "params_m": params_m,
                "fr": fr,
                "energy_uj": energy_uj,
                "source": path,
            }
        )
    return rows


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_markdown(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = "| Method | MR | Set | PSNR | SSIM | Params(M) | FR | Energy(uJ) | Source |"
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {method} | {mr} | {set} | {psnr} | {ssim} | {params_m} | {fr} | {energy_uj} | {source} |".format(
                method=row.get("method", ""),
                mr=row.get("mr", ""),
                set=row.get("set", ""),
                psnr=fmt_float(row.get("psnr"), 2),
                ssim=fmt_float(row.get("ssim"), 4),
                params_m=fmt_float(row.get("params_m"), 3),
                fr=fmt_float(row.get("fr"), 4),
                energy_uj=fmt_float(row.get("energy_uj"), 3),
                source=row.get("source", ""),
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--sets", nargs="+", default=["Set11", "BSD68", "Urban100"])
    parser.add_argument("--checkpoint", action="append", default=[], help="METHOD:MR:PATH")
    parser.add_argument("--external-csv", default=None)
    parser.add_argument("--out-csv", default="results/comparison.csv")
    parser.add_argument("--out-md", default="results/comparison.md")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows = read_external_csv(args.external_csv)
    for spec in args.checkpoint:
        rows.extend(eval_checkpoint(spec, args.data_root, args.sets, args.device))

    rows.sort(key=lambda r: (float(r["mr"]) if str(r.get("mr", "")).replace(".", "", 1).isdigit() else 999, r.get("set", ""), r.get("method", "")))
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
