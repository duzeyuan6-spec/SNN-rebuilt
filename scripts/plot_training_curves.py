import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+val PSNR\s+([0-9.]+)\s+\|\s+SSIM\s+([0-9.]+)\s+\|\s+best SSIM\s+([0-9.]+).*?FR\s+([0-9.]+)\s+\|\s+energy\s+([0-9.]+)"
)

COLORS = {
    "ANN-Ours": "#059669",
    "ISTA-Prox": "#7C3AED",
    "CSNet-style-C36B8": "#2563EB",
    "SpikeCS": "#DC2626",
}


def parse_log(path):
    rows = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for match in LOG_RE.finditer(text):
        rows.append(
            {
                "epoch": int(match.group(1)),
                "loss": float(match.group(2)),
                "psnr": float(match.group(3)),
                "ssim": float(match.group(4)),
                "best_ssim": float(match.group(5)),
                "fr": float(match.group(6)),
                "energy_uj": float(match.group(7)),
            }
        )
    return rows


def plot_single(method, rows, out_dir):
    epochs = [r["epoch"] for r in rows]
    color = COLORS.get(method, "#334155")
    specs = [
        ("ssim", "Validation SSIM", f"{method} SSIM by Epoch", "ssim_by_epoch.png"),
        ("best_ssim", "Best Validation SSIM", f"{method} Best SSIM by Epoch", "best_ssim_by_epoch.png"),
        ("psnr", "Validation PSNR (dB)", f"{method} PSNR by Epoch", "psnr_by_epoch.png"),
        ("loss", "Train Loss", f"{method} Loss by Epoch", "loss_by_epoch.png"),
    ]
    method_dir = out_dir / method.replace("/", "_")
    method_dir.mkdir(parents=True, exist_ok=True)
    for key, ylabel, title, filename in specs:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
        ax.plot(epochs, [r[key] for r in rows], color=color, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        fig.tight_layout()
        fig.savefig(method_dir / filename, bbox_inches="tight")
        plt.close(fig)


def plot_combined(all_rows, key, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    for method, rows in all_rows.items():
        epochs = [r["epoch"] for r in rows]
        values = [r[key] for r in rows]
        ax.plot(epochs, values, label=method, color=COLORS.get(method), linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/figures/ssim_2000_training_curves")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="METHOD:PATH",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    lines = ["| Method | Epochs | Best epoch | Best SSIM | Best PSNR at best SSIM | Last SSIM | Last PSNR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for spec in args.log:
        method, path = spec.split(":", 1)
        rows = parse_log(path)
        if not rows:
            raise RuntimeError(f"No epoch rows parsed from {path}")
        all_rows[method] = rows
        plot_single(method, rows, out_dir)
        best = max(rows, key=lambda r: r["ssim"])
        last = rows[-1]
        lines.append(
            f"| {method} | {len(rows)} | {best['epoch']} | {best['ssim']:.4f} | {best['psnr']:.2f} | {last['ssim']:.4f} | {last['psnr']:.2f} |"
        )

    plot_combined(all_rows, "ssim", "Validation SSIM", "Validation SSIM by Epoch", out_dir / "all_models_ssim_by_epoch.png")
    plot_combined(all_rows, "best_ssim", "Best Validation SSIM", "Best Validation SSIM by Epoch", out_dir / "all_models_best_ssim_by_epoch.png")
    plot_combined(all_rows, "psnr", "Validation PSNR (dB)", "Validation PSNR by Epoch", out_dir / "all_models_psnr_by_epoch.png")
    plot_combined(all_rows, "loss", "Train Loss", "Train Loss by Epoch", out_dir / "all_models_loss_by_epoch.png")

    summary = out_dir / "training_curve_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote training curves to {out_dir}")
    for path in sorted(out_dir.rglob("*.png")):
        print(path)
    print(summary)


if __name__ == "__main__":
    main()
