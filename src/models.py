import torch
import torch.nn as nn

from .cs import BlockCSOperator
from .spiking import LIFNode, SpikeStats


class AnnPrior(nn.Module):
    def __init__(self, channels=32, blocks=2):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(blocks):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class SpikingPrior(nn.Module):
    def __init__(self, channels=32, blocks=2, time_steps=4, threshold=1.0, decay=0.5):
        super().__init__()
        self.time_steps = time_steps
        self.in_conv = nn.Conv2d(1, channels, 3, padding=1)
        self.hidden = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(blocks)])
        self.out_conv = nn.Conv2d(channels, 1, 3, padding=1)
        self.lif_in = LIFNode(threshold, decay)
        self.lif_hidden = nn.ModuleList([LIFNode(threshold, decay) for _ in range(blocks)])
        self.input_gain = nn.Parameter(torch.tensor(1.0))
        self.res_scale = nn.Parameter(torch.tensor(0.2))
        self.last_firing_rate = 0.0

    def forward(self, x):
        stats = SpikeStats()
        mem_in = torch.zeros(x.size(0), self.in_conv.out_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        mem_hidden = [
            torch.zeros_like(mem_in) for _ in self.hidden
        ]
        readout_acc = torch.zeros_like(mem_in)

        # Direct coding: the same analog residual input is injected at each time step.
        for _ in range(self.time_steps):
            h = self.input_gain.clamp(0.1, 10.0) * self.in_conv(x)
            s, mem_in = self.lif_in(h, mem_in)
            stats.add(s)

            for i, conv in enumerate(self.hidden):
                h = conv(s)
                s, mem_hidden[i] = self.lif_hidden[i](h, mem_hidden[i])
                stats.add(s)
            readout_acc = readout_acc + mem_hidden[-1]

        # Read membrane once for reconstruction precision; hidden convolutions
        # remain event-driven and are accounted for with spike operations.
        self.last_firing_rate = stats.firing_rate
        return x + self.res_scale * self.out_conv(readout_acc / self.time_steps)


class IstaProx(nn.Module):
    def __init__(self, channels=32, blocks=2):
        super().__init__()
        self.analysis = nn.Conv2d(1, channels, 3, padding=1)
        layers = []
        for _ in range(blocks):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        self.synthesis = nn.Conv2d(channels, 1, 3, padding=1)
        self.theta = nn.Parameter(torch.tensor(0.01))
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        z = self.analysis(x)
        z = self.body(z)
        theta = self.theta.abs()
        z = torch.sign(z) * torch.relu(torch.abs(z) - theta)
        return x + self.res_scale * self.synthesis(z)


class ReconNetCS(nn.Module):
    def __init__(self, cs_op, channels=16, blocks=4):
        super().__init__()
        self.cs = cs_op
        self.kind = "reconnet"
        self.block_size = cs_op.block_size
        self.fc = nn.Linear(cs_op.m, cs_op.n)
        layers = [nn.Conv2d(1, channels, 11, padding=5), nn.ReLU(inplace=True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 1, 7, padding=3)]
        self.refine = nn.Sequential(*layers)
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, y):
        b, m, hb, wb = y.shape
        patches = y.permute(0, 2, 3, 1).reshape(b * hb * wb, m)
        patches = self.fc(patches)
        patches = patches.view(b, hb, wb, 1, self.block_size, self.block_size)
        x = patches.permute(0, 3, 1, 4, 2, 5).reshape(b, 1, hb * self.block_size, wb * self.block_size)
        return x + self.res_scale * self.refine(x)

    def firing_rate(self):
        return 0.0


class CSNetStyle(nn.Module):
    def __init__(self, cs_op, channels=32, blocks=6):
        super().__init__()
        self.cs = cs_op
        self.kind = "csnet"
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.PReLU()]
        for _ in range(blocks):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.PReLU(),
            ]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.reconstruct = nn.Sequential(*layers)
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, y):
        x0 = self.cs.adjoint(y)
        return x0 + self.res_scale * self.reconstruct(x0)

    def firing_rate(self):
        return 0.0


class UnrolledCSNet(nn.Module):
    def __init__(self, cs_op, kind="ann", stages=5, channels=32, blocks_per_stage=2, time_steps=4, threshold=1.0, decay=0.5):
        super().__init__()
        self.cs = cs_op
        self.kind = kind
        self.stages = stages
        self.rhos = nn.Parameter(torch.ones(stages) * 0.1)
        prior_map = {
            "ann": AnnPrior,
            "spike": SpikingPrior,
            "ista": IstaProx,
        }
        if kind not in prior_map:
            raise ValueError(f"Unknown model kind: {kind}")
        prior_cls = prior_map[kind]
        self.priors = nn.ModuleList()
        for _ in range(stages):
            if kind in {"ann", "ista"}:
                self.priors.append(prior_cls(channels, blocks_per_stage))
            else:
                self.priors.append(prior_cls(channels, blocks_per_stage, time_steps, threshold, decay))

    def forward(self, y):
        x = self.cs.adjoint(y)
        for k, prior in enumerate(self.priors):
            r = x - self.rhos[k] * self.cs.normal_residual(x, y)
            x = prior(r)
        return x

    def firing_rate(self):
        if self.kind != "spike":
            return 0.0
        rates = [p.last_firing_rate for p in self.priors]
        return sum(rates) / max(len(rates), 1)


def build_model(config):
    cs_cfg = config["cs"]
    model_cfg = config["model"]
    cs_op = BlockCSOperator(
        block_size=cs_cfg["block_size"],
        measurement_rate=cs_cfg["measurement_rate"],
        seed=config.get("seed", 123),
    )
    if model_cfg["kind"] == "reconnet":
        return ReconNetCS(
            cs_op=cs_op,
            channels=model_cfg.get("channels", 16),
            blocks=model_cfg.get("blocks", model_cfg.get("blocks_per_stage", 4)),
        )
    if model_cfg["kind"] == "csnet":
        return CSNetStyle(
            cs_op=cs_op,
            channels=model_cfg.get("channels", 32),
            blocks=model_cfg.get("blocks", model_cfg.get("blocks_per_stage", 6)),
        )
    return UnrolledCSNet(
        cs_op=cs_op,
        kind=model_cfg["kind"],
        stages=model_cfg["stages"],
        channels=model_cfg["channels"],
        blocks_per_stage=model_cfg["blocks_per_stage"],
        time_steps=model_cfg.get("time_steps", 4),
        threshold=model_cfg.get("threshold", 1.0),
        decay=model_cfg.get("decay", 0.5),
    )
