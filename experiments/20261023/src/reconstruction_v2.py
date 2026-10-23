"""Stagewise SNN reconstruction without a pretrained ANN reconstruction bypass.

A bounded gain prevents a near-zero scalar gate; benefit still requires matched
ANN training and validation interventions. Sensing and readout remain analog.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .cs import BlockCSOperator
from .spiking import LIFNode


class ReconstructionPrior(nn.Module):
    def __init__(self, channels, time_steps, kind):
        super().__init__()
        self.kind, self.time_steps = kind, time_steps
        self.input = nn.Conv2d(1, channels, 3, padding=1)
        self.hidden = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(2)])
        self.output = nn.Conv2d(channels, 1, 3, padding=1)
        self.gain_raw = nn.Parameter(torch.tensor(math.log(.4 / .6)))
        if kind == 'spike':
            self.nodes = nn.ModuleList([LIFNode(.45, .7, True) for _ in range(3)])
            self.temporal_logits = nn.Parameter(torch.zeros(time_steps))
            for layer in self.hidden:
                layer.event_driven_candidate = True
        self.spike_penalty = None

    @property
    def gain(self):
        return .25 + 1.25 * self.gain_raw.sigmoid()

    def forward(self, x, intervention=None):
        if intervention == 'identity':
            return x
        drive = self.input(x)
        if self.kind == 'ann':
            h = F.relu(drive)
            for layer in self.hidden:
                h = F.relu(layer(h))
            residual = self.output(h)
            self.spike_penalty = x.new_zeros(())
        else:
            states = [torch.zeros_like(drive) for _ in self.nodes]
            readouts, spike_sums = [], []
            for _ in range(self.time_steps):
                if intervention == 'reset_memory':
                    states = [torch.zeros_like(drive) for _ in self.nodes]
                s, states[0] = self.nodes[0](drive, states[0])
                spike_sums.append(s.mean())
                for i, layer in enumerate(self.hidden):
                    s, states[i + 1] = self.nodes[i + 1](layer(s), states[i + 1])
                    spike_sums.append(s.mean())
                readouts.append(self.output(states[-1]))
            self.spike_penalty = torch.stack(spike_sums).mean()
            residual = sum(w * r for w, r in zip(self.temporal_logits.softmax(0), readouts))
        return x + self.gain * residual


class StagewiseRecon(nn.Module):
    def __init__(self, kind='spike', channels=48, stages=5, time_steps=4,
                 block_size=32, measurement_rate=.25, acquisition_seed=20261013):
        super().__init__()
        if kind not in ('ann', 'spike') or stages < 1 or time_steps < 1:
            raise ValueError('Invalid architecture')
        self.kind = kind
        self.cs = BlockCSOperator(block_size, measurement_rate, acquisition_seed, False)
        generator = torch.Generator().manual_seed(acquisition_seed)
        q, _ = torch.linalg.qr(torch.randn(self.cs.n, self.cs.m, generator=generator), mode='reduced')
        self.cs.phi.copy_(q.T)
        self.step_raw = nn.Parameter(torch.zeros(stages))
        self.priors = nn.ModuleList([ReconstructionPrior(channels, time_steps, kind) for _ in range(stages)])

    def forward(self, y, return_stages=False, intervention=None):
        x = self.cs.adjoint(y)
        outputs = []
        for i, prior in enumerate(self.priors):
            x = x - (2 * self.step_raw[i].sigmoid()) * self.cs.normal_residual(x, y)
            x = prior(x, intervention)
            outputs.append(x)
        return (x, outputs) if return_stages else x

    def spike_regularization(self):
        return torch.stack([p.spike_penalty for p in self.priors]).mean()

    def firing_rate(self):
        return float(self.spike_regularization().detach()) if self.kind == 'spike' else 0.
