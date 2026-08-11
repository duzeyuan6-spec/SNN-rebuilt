import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_results import (
    FIELDS,
    eval_checkpoint,
    read_external_csv,
    write_csv,
    write_markdown,
)


METRIC_RE = re.compile(
    r"(?P<set>Set11|BSD68|Urban100).*?"
    r"(?:PSNR|psnr)[:=\s]+(?P<psnr>[0-9]+(?:\.[0-9]+)?).*?"
    r"(?:SSIM|ssim)[:=\s]+(?P<ssim>[0-9]+(?:\.[0-9]+)?)",
    re.S,
)

PLV_SUMMARY_RE = re.compile(
    r"Dataset=\s*(?P<dataset>[^\n\r]+).*?"
    r"PSNR_predicted=(?P<psnr>[0-9]+(?:\.[0-9]+)?),"
    r"SSIM_predicted=(?P<ssim>[0-9]+(?:\.[0-9]+)?)",
    re.S,
)


def run_command(cmd, cwd, log_path):
    env = os.environ.copy()
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout


def parse_metrics_from_log(text, wanted_sets):
    rows = []
    for match in PLV_SUMMARY_RE.finditer(text):
        dataset = match.group("dataset")
        for set_name in wanted_sets:
            if set_name in dataset:
                rows.append(
                    {
                        "set": set_name,
                        "psnr": float(match.group("psnr")),
                        "ssim": float(match.group("ssim")),
                    }
                )

    for match in METRIC_RE.finditer(text):
        set_name = match.group("set")
        if set_name in wanted_sets:
            rows.append(
                {
                    "set": set_name,
                    "psnr": float(match.group("psnr")),
                    "ssim": float(match.group("ssim")),
                }
            )
    return rows


def parse_last_plv_summary(text):
    matches = list(PLV_SUMMARY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "psnr": float(match.group("psnr")),
        "ssim": float(match.group("ssim")),
    }


def run_plv(plv_dir, mr, wanted_sets, log_dir, baseline_mat_root):
    if not plv_dir.exists():
        print(f"[skip] PLV-CSNet not found: {plv_dir}")
        return []

    rows = []
    target_dir = plv_dir / "test" / "BSD68_mat"
    backup_dir = plv_dir / "test" / "BSD68_mat_autobak"

    if target_dir.exists() and not backup_dir.exists():
        target_dir.rename(backup_dir)

    for set_name in wanted_sets:
        src_dir = baseline_mat_root / f"{set_name}_mat"
        if not src_dir.exists():
            print(f"[skip] PLV data not found: {src_dir}")
            continue

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(src_dir, target_dir)

        log_path = log_dir / f"plv_{set_name.lower()}_mr{str(mr).replace('.', '')}.log"
        print(f"[run] PLV-CSNet {set_name}, log: {log_path}")
        code, output = run_command([sys.executable, "test.py"], plv_dir, log_path)
        if code != 0:
            print(f"[warn] PLV-CSNet {set_name} exited with code {code}. See {log_path}")

        item = parse_last_plv_summary(output)
        if item is None:
            print(f"[warn] Could not parse PLV {set_name} metrics. See {log_path}")
            continue

        rows.append(
            {
                "method": "PLV-CSNet",
                "mr": mr,
                "set": set_name,
                "psnr": item["psnr"],
                "ssim": item["ssim"],
                "params_m": "",
                "fr": "",
                "energy_uj": "",
                "source": str(log_path),
            }
        )

    if target_dir.exists():
        shutil.rmtree(target_dir)
    if backup_dir.exists():
        backup_dir.rename(target_dir)

    return rows


def ensure_fields(rows):
    fixed = []
    for row in rows:
        item = {field: row.get(field, "") for field in FIELDS}
        for field in FIELDS:
            if item[field] in ("", None):
                item[field] = "N/A"
        fixed.append(item)
    return fixed


def merge_duplicate_rows(rows):
    merged = {}
    order = []
    for row in rows:
        key = (row.get("method", ""), str(row.get("mr", "")), row.get("set", ""))
        if key not in merged:
            merged[key] = {}
            order.append(key)

        current = merged[key]
        for field in FIELDS:
            value = row.get(field, "")
            if value in ("", None, "N/A"):
                continue
            current[field] = value

        for field in FIELDS:
            current.setdefault(field, row.get(field, ""))

    return [merged[key] for key in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", default="0.25")
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--sets", nargs="+", default=["Set11", "BSD68", "Urban100"])
    parser.add_argument("--ann", required=True, help="ANN checkpoint path")
    parser.add_argument("--spike", required=True, help="SpikeCS checkpoint path")
    parser.add_argument("--external-csv", default="results/external_baselines.csv")
    parser.add_argument("--plv-dir", default="baselines/PLV-CSNet")
    parser.add_argument("--baseline-mat-root", default="baseline_data/mat")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    mr_tag = str(args.mr).replace(".", "")
    out_csv = args.out_csv or f"results/five_model_mr{mr_tag}.csv"
    out_md = args.out_md or f"results/five_model_mr{mr_tag}.md"
    log_dir = ROOT / "results" / "logs"

    rows = []
    rows.extend(read_external_csv(args.external_csv))
    rows.extend(run_plv(ROOT / args.plv_dir, args.mr, args.sets, log_dir, ROOT / args.baseline_mat_root))

    rows.extend(eval_checkpoint(f"ANN-Ours:{args.mr}:{args.ann}", args.data_root, args.sets, args.device))
    rows.extend(eval_checkpoint(f"SpikeCS:{args.mr}:{args.spike}", args.data_root, args.sets, args.device))

    rows = merge_duplicate_rows(rows)
    rows = [row for row in ensure_fields(rows) if str(row.get("mr")) == str(args.mr)]
    rows.sort(key=lambda r: (r.get("set", ""), r.get("method", "")))

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
