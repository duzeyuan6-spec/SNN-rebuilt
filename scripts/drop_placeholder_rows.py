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

PLACEHOLDERS = {"", "TODO", "xx.xx", "0.xxxx"}


def has_placeholder_metric(row):
    return any(str(row.get(field, "")).strip() in PLACEHOLDERS for field in ["psnr", "ssim", "params_m"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/external_baselines.csv")
    args = parser.parse_args()

    path = Path(args.csv)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [{field: row.get(field, "") for field in FIELDS} for row in csv.DictReader(f)]

    kept = [row for row in rows if not has_placeholder_metric(row)]
    removed = len(rows) - len(kept)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)

    print(f"removed placeholder rows: {removed}")
    print(f"kept rows: {len(kept)}")


if __name__ == "__main__":
    main()
