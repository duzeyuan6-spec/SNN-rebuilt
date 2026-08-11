import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x.abs() < 1.0).to(x.dtype)
        return grad_output * mask


class LIFNode(nn.Module):
    def __init__(self, threshold=1.0, decay=0.5):
        super().__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, x, mem):
        mem = self.decay * mem + x
        spike = SpikeFn.apply(mem - self.threshold)
        mem = mem - spike * self.threshold
        return spike, mem


class SpikeStats:
    def __init__(self):
        self.spikes = 0.0
        self.neurons = 0.0

    def add(self, spike):
        self.spikes += float(spike.detach().sum().cpu())
        self.neurons += spike.numel()

    @property
    def firing_rate(self):
        if self.neurons == 0:
            return 0.0
        return self.spikes / self.neurons

