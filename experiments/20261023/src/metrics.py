import math

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn


def psnr(pred, target, eps=1e-10):
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    return 10 * math.log10(1.0 / max(mse, eps))


def ssim(pred, target):
    pred_np = pred.detach().clamp(0, 1).squeeze().cpu().numpy()
    target_np = target.detach().clamp(0, 1).squeeze().cpu().numpy()
    return float(ssim_fn(target_np, pred_np, data_range=1.0))


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)

