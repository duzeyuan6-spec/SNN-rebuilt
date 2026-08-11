import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ["ReconNet-PyTorch", "CSNet-style", "ISTA-Net++", "ANN-Ours", "SpikeCS"]
COLORS = {
    "ReconNet-PyTorch": "#6B7280",
    "CSNet-style": "#2563EB",
    "ISTA-Net++": "#7C3AED",
    "ANN-Ours": "#059669",
    "SpikeCS": "#DC2626",
}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sorted_methods(rows):
    present = {row["method"] for row in rows}
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def sorted_sets(rows):
    order = ["Set11", "BSD68", "Urban100"]
    present = {row["set"] for row in rows}
    ordered = [s for s in order if s in present]
    ordered += sorted(present - set(ordered))
    return ordered


def plot_grouped_metric(rows, metric, ylabel, title, out_path):
    methods = sorted_methods(rows)
    sets = sorted_sets(rows)
    values = {
        (row["method"], row["set"]): to_float(row.get(metric, ""))
        for row in rows
    }

    x = np.arange(len(sets))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=180)
    for i, method in enumerate(methods):
        ys = [values.get((method, s), np.nan) for s in sets]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            ys,
            width,
            label=method,
            color=COLORS.get(method, None),
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, y in zip(bars, ys):
            if np.isnan(y):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{y:.2f}" if metric != "ssim" else f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if metric == "energy_uj" else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(ncol=min(len(methods), 5), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_method_single(rows, metric, ylabel, title, out_path):
    methods = sorted_methods(rows)
    values = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        vals = [to_float(row.get(metric, "")) for row in method_rows]
        vals = [v for v in vals if not np.isnan(v)]
        values.append(np.mean(vals) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    bars = ax.bar(
        np.arange(len(methods)),
        values,
        color=[COLORS.get(m, "#64748B") for m in methods],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, y in zip(bars, values):
        if np.isnan(y):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/fair_model_mr025.csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "figures" / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)

    plot_grouped_metric(rows, "psnr", "PSNR (dB)", "Reconstruction Quality: PSNR", out_dir / "psnr_by_set.png")
    plot_grouped_metric(rows, "ssim", "SSIM", "Reconstruction Quality: SSIM", out_dir / "ssim_by_set.png")
    plot_grouped_metric(rows, "energy_uj", "Energy (uJ)", "Estimated Energy", out_dir / "energy_by_set.png")
    plot_method_single(rows, "params_m", "Params (M)", "Model Size", out_dir / "params_by_method.png")

    print(f"Wrote figures to {out_dir}")
    for path in sorted(out_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
