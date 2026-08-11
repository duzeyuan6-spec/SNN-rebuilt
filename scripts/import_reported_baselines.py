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


SOURCE_MMU_NET = "reported:MMU-Net_Table2_Entropy2023"
SOURCE_SENSORS = "reported:Sensors2023_Table1"


REPORTED = [
    # MR=0.25 values from MMU-Net Table 2: Set11, BSDS68, Urban100.
    # Non-spiking baselines have FR=0.0000. Energy is left as N/A because
    # those papers do not use our spike/ANN energy-estimation protocol.
    ("ISTA-Net+", "0.25", "Set11", "32.59", "0.9254", "0.380", "0.0000", "N/A", SOURCE_MMU_NET),
    ("ISTA-Net+", "0.25", "BSD68", "29.36", "0.8525", "0.380", "0.0000", "N/A", SOURCE_MMU_NET),
    ("ISTA-Net+", "0.25", "Urban100", "29.78", "0.8954", "0.380", "0.0000", "N/A", SOURCE_MMU_NET),
    ("CSNet+", "0.25", "Set11", "33.34", "0.9387", "N/A", "0.0000", "N/A", SOURCE_MMU_NET),
    ("CSNet+", "0.25", "BSD68", "31.12", "0.9060", "N/A", "0.0000", "N/A", SOURCE_MMU_NET),
    ("CSNet+", "0.25", "Urban100", "28.13", "0.8827", "N/A", "0.0000", "N/A", SOURCE_MMU_NET),
    ("AMP-Net", "0.25", "Set11", "34.42", "0.9513", "0.580", "0.0000", "N/A", SOURCE_MMU_NET),
    ("AMP-Net", "0.25", "BSD68", "31.46", "0.9053", "0.580", "0.0000", "N/A", SOURCE_MMU_NET),
    ("AMP-Net", "0.25", "Urban100", "30.89", "0.9202", "0.580", "0.0000", "N/A", SOURCE_MMU_NET),
    ("OPINE-Net+", "0.25", "Set11", "34.81", "0.9503", "0.620", "0.0000", "N/A", SOURCE_MMU_NET),
    ("OPINE-Net+", "0.25", "BSD68", "31.56", "0.9121", "0.620", "0.0000", "N/A", SOURCE_MMU_NET),
    ("OPINE-Net+", "0.25", "Urban100", "31.36", "0.9216", "0.620", "0.0000", "N/A", SOURCE_MMU_NET),
    # ReconNet has clean Set11 MR=0.25 reported values in several CS tables.
    # Do not invent BSD68/Urban100 reported values unless a matching source is added.
    ("ReconNet", "0.25", "Set11", "32.07", "0.9246", "0.309", "0.0000", "N/A", SOURCE_SENSORS),
]


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{field: row.get(field, "") for field in FIELDS} for row in csv.DictReader(f)]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r["mr"]), r["set"], r["method"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def upsert(rows, item):
    key = (item["method"], item["mr"], item["set"])
    for idx, row in enumerate(rows):
        if (row["method"], str(row["mr"]), row["set"]) == key:
            rows[idx] = item
            return "updated"
    rows.append(item)
    return "added"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/external_baselines.csv")
    parser.add_argument("--mr", default="0.25")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ISTA-Net+", "CSNet+", "AMP-Net", "OPINE-Net+", "ReconNet"],
    )
    args = parser.parse_args()

    rows = read_rows(args.csv)
    changed = []
    for record in REPORTED:
        method, mr, set_name, psnr, ssim, params_m, fr, energy_uj, source = record
        if mr != str(args.mr) or method not in args.methods:
            continue
        status = upsert(
            rows,
            {
                "method": method,
                "mr": mr,
                "set": set_name,
                "psnr": psnr,
                "ssim": ssim,
                "params_m": params_m,
                "fr": fr,
                "energy_uj": energy_uj,
                "source": source,
            },
        )
        changed.append((status, method, set_name))

    write_rows(args.csv, rows)
    for status, method, set_name in changed:
        print(f"{status}: {method} MR={args.mr} {set_name}")


if __name__ == "__main__":
    main()
