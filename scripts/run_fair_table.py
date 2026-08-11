import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_results import eval_checkpoint, read_external_csv, write_csv, write_markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", required=True)
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--sets", nargs="+", default=["Set11", "BSD68", "Urban100"])
    parser.add_argument("--reconnet", required=True)
    parser.add_argument("--csnet", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--spike", required=True)
    parser.add_argument("--external-csv", default=None, help="Optional CSV for reproduced ISTA-Net++ rows")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    mr_tag = str(args.mr).replace(".", "")
    out_csv = args.out_csv or f"results/fair_model_mr{mr_tag}.csv"
    out_md = args.out_md or f"results/fair_model_mr{mr_tag}.md"

    rows = []
    rows.extend(read_external_csv(args.external_csv))
    rows.extend(eval_checkpoint(f"ReconNet-PyTorch:{args.mr}:{args.reconnet}", args.data_root, args.sets, args.device))
    rows.extend(eval_checkpoint(f"CSNet-style:{args.mr}:{args.csnet}", args.data_root, args.sets, args.device))
    rows.extend(eval_checkpoint(f"ANN-Ours:{args.mr}:{args.ann}", args.data_root, args.sets, args.device))
    rows.extend(eval_checkpoint(f"SpikeCS:{args.mr}:{args.spike}", args.data_root, args.sets, args.device))
    rows = [row for row in rows if str(row.get("mr")) == str(args.mr)]
    rows.sort(key=lambda r: (r.get("set", ""), r.get("method", "")))

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
