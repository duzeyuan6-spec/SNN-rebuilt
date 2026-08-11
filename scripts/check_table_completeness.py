import argparse
import csv
from pathlib import Path


REQUIRED_FIELDS = ["psnr", "ssim", "params_m", "fr", "energy_uj"]
MISSING_VALUES = {"", "TODO", "xx.xx", "0.xxxx"}
ENERGY_NA_OK_SOURCES = {"reported", "reproduced_by_plv_test"}


def is_missing(value):
    return str(value).strip() in MISSING_VALUES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/five_model_mr025.csv")
    args = parser.parse_args()

    path = Path(args.csv)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    missing = []
    for row in rows:
        for field in REQUIRED_FIELDS:
            if field == "energy_uj" and str(row.get("energy_uj", "")).upper() in {"N/A", "NA"}:
                source = str(row.get("source", ""))
                if any(token in source for token in ENERGY_NA_OK_SOURCES):
                    continue
            if is_missing(row.get(field, "")):
                missing.append((row.get("method", ""), row.get("mr", ""), row.get("set", ""), field))

    if not missing:
        print(f"complete: {path}")
        return

    print(f"incomplete: {path}")
    for method, mr, set_name, field in missing:
        print(f"  {method} MR={mr} {set_name}: missing {field}")


if __name__ == "__main__":
    main()
