import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_object(module_path, object_name):
    module_path = Path(module_path).resolve()
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, object_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="our ANN/SpikeCS checkpoint")
    parser.add_argument("--module", default=None, help="external Python file defining a model class")
    parser.add_argument("--object", default=None, help="model class/function name in --module")
    parser.add_argument("--prepend-path", action="append", default=[], help="extra import path, e.g. a baseline repo root")
    parser.add_argument("--args", nargs="*", default=[], help="positional args passed to external object")
    args = parser.parse_args()

    if args.checkpoint:
        from src.models import build_model

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        print(f"params={count_params(model)}")
        print(f"params_m={count_params(model) / 1e6:.6f}")
        return

    if args.module and args.object:
        for path in reversed(args.prepend_path):
            sys.path.insert(0, str(Path(path).resolve()))
        obj = load_object(args.module, args.object)
        model = obj(*args.args)
        print(f"params={count_params(model)}")
        print(f"params_m={count_params(model) / 1e6:.6f}")
        return

    raise SystemExit("Provide --checkpoint or --module + --object")


if __name__ == "__main__":
    main()
