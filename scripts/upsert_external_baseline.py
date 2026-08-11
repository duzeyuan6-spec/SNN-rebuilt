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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/external_baselines.csv")
    parser.add_argument("--method", required=True)
    parser.add_argument("--mr", required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--psnr", required=True)
    parser.add_argument("--ssim", required=True)
    parser.add_argument("--params-m", default="TODO")
    parser.add_argument("--fr", default="0.0000")
    parser.add_argument("--energy-uj", default="N/A")
    parser.add_argument("--source", default="reported_or_reproduced")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    key = (args.method, str(args.mr), args.set)
    item = {
        "method": args.method,
        "mr": str(args.mr),
        "set": args.set,
        "psnr": args.psnr,
        "ssim": args.ssim,
        "params_m": args.params_m,
        "fr": args.fr,
        "energy_uj": args.energy_uj,
        "source": args.source,
    }

    replaced = False
    for idx, row in enumerate(rows):
        if (row["method"], str(row["mr"]), row["set"]) == key:
            rows[idx] = item
            replaced = True
            break
    if not replaced:
        rows.append(item)

    write_rows(args.csv, rows)
    print(f"{'updated' if replaced else 'added'}: {args.method} MR={args.mr} {args.set}")


if __name__ == "__main__":
    main()
