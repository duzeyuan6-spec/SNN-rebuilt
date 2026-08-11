import argparse
import csv
import re
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

PAIR_RE = re.compile(
    r"(?:PSNR|psnr|avg_psnr|Average PSNR|PSNR_predicted)\s*[:=]\s*"
    r"(?P<psnr>[0-9]+(?:\.[0-9]+)?).*?"
    r"(?:SSIM|ssim|avg_ssim|Average SSIM|SSIM_predicted)\s*[:=]\s*"
    r"(?P<ssim>[0-9]+(?:\.[0-9]+)?)",
    re.S,
)

NUMBER_LINE_RE = re.compile(r"(?P<psnr>[0-9]{2}\.[0-9]+)\s*/\s*(?P<ssim>0\.[0-9]+)")


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


def parse_metrics(text):
    matches = list(PAIR_RE.finditer(text))
    if matches:
        match = matches[-1]
        return match.group("psnr"), match.group("ssim")

    matches = list(NUMBER_LINE_RE.finditer(text))
    if matches:
        match = matches[-1]
        return match.group("psnr"), match.group("ssim")

    raise ValueError("Could not find PSNR/SSIM in log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/fair_external.csv")
    parser.add_argument("--log", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mr", required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--params-m", required=True)
    parser.add_argument("--fr", default="0.0000")
    parser.add_argument("--energy-uj", default="N/A")
    args = parser.parse_args()

    log_path = Path(args.log)
    psnr, ssim = parse_metrics(log_path.read_text(encoding="utf-8", errors="ignore"))

    rows = read_rows(args.csv)
    item = {
        "method": args.method,
        "mr": str(args.mr),
        "set": args.set,
        "psnr": psnr,
        "ssim": ssim,
        "params_m": args.params_m,
        "fr": args.fr,
        "energy_uj": args.energy_uj,
        "source": str(log_path),
    }

    key = (item["method"], item["mr"], item["set"])
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
    print(f"psnr={psnr}, ssim={ssim}")


if __name__ == "__main__":
    main()
