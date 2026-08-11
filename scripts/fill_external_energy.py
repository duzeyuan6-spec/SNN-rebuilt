import argparse
import csv
from pathlib import Path

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

E_MAC_PJ = 4.6

# 用和 ANN baselines 一样的 MAC 能耗常数估计 ISTA-Net++。
# 这是 estimated energy，不是硬件实测。
def estimate_istanetpp_energy_uj(layer_num, set_name, channels=32):
    # 用各测试集的代表尺寸估计。主表里 source 会标注 energy_estimated_mac。
    if set_name == "Set11":
        h, w = 256, 256
    elif set_name == "BSD68":
        h, w = 321, 481
    elif set_name == "Urban100":
        h, w = 512, 512
    else:
        h, w = 256, 256

    pixels = h * w

    # ISTA-Net++ 每 phase 主要是多层 3x3 conv，这里按 dominant conv MAC 估计。
    convs_per_phase = 4
    macs = layer_num * convs_per_phase * pixels * channels * channels * 3 * 3

    # 输入/输出浅层变换
    macs += pixels * channels * 3 * 3 * 2

    return macs * E_MAC_PJ / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/fair_external.csv")
    parser.add_argument("--method", default="ISTA-Net++")
    parser.add_argument("--layer-num", type=int, default=20)
    parser.add_argument("--channels", type=int, default=32)
    args = parser.parse_args()

    path = Path(args.csv)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [{field: row.get(field, "") for field in FIELDS} for row in csv.DictReader(f)]

    for row in rows:
        if row["method"] != args.method:
            continue
        row["energy_uj"] = f"{estimate_istanetpp_energy_uj(args.layer_num, row['set'], args.channels):.3f}"
        if "energy_estimated_mac" not in row["source"]:
            row["source"] = f"{row['source']}|energy_estimated_mac"

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"updated energy for {args.method}: {path}")


if __name__ == "__main__":
    main()
