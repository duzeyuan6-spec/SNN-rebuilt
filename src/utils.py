import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def check_validation_split(train_dir, val_dir):
    """Fail closed for legacy configs that selected checkpoints on test sets."""
    train, val = Path(train_dir).resolve(), Path(val_dir).resolve()
    forbidden = {'test', 'set11', 'bsd68', 'urban100'}
    if forbidden.intersection(part.lower() for part in val.parts):
        raise ValueError('Benchmark test sets cannot be used for checkpoint selection. '
                         'Use train_reconstruction_v2.py with a frozen train-source split.')
    if train == val or train in val.parents or val in train.parents:
        raise ValueError('Training and validation roots overlap')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, optimizer, epoch, best_metric, config):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )

