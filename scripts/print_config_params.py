import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.utils import count_params, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    for cfg_path in args.configs:
        config = load_config(cfg_path)
        model = build_model(config)
        print(f"{cfg_path}: {count_params(model) / 1e6:.6f} M")


if __name__ == "__main__":
    main()
