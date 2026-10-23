import torch
import torch.nn as nn

from .cs import BlockCSOperator
from .spiking import LIFNode, SpikeStats, BurstIF


class BurstTransposeReadout(nn.Conv2d):
    """Stride-2 synthesis with the original [1,C,3,3] checkpoint layout.

    Every stored coefficient is used; transposing the first two axes exposes
    the layout expected by conv_transpose2d without introducing new parameters.
    """
    transposed_burst_readout=True

    def forward(self, x, output_size):
        base_h=(x.shape[-2]-1)*2-2*self.padding[0]+self.kernel_size[0]
        base_w=(x.shape[-1]-1)*2-2*self.padding[1]+self.kernel_size[1]
        extra=(output_size[0]-base_h,output_size[1]-base_w)
        if any(v not in (0,1) for v in extra):raise ValueError('Invalid synthesis output size')
        return torch.nn.functional.conv_transpose2d(x,self.weight.transpose(0,1),self.bias,
                                                   stride=2,padding=self.padding,output_padding=extra)


class MatchedBurstPrior(nn.Module):
    """Same active convolution parameters for ANN and burst-coded SNN.

    All hidden SNN communications are quantized signed event counts at eval;
    no parallel analog reconstruction branch is introduced.
    """
    def __init__(self,channels=64,blocks=2,time_steps=8,kind='spike',spatial_dilation=1):
        super().__init__()
        self.kind=kind
        self.in_conv=nn.Conv2d(1,channels,3,padding=1)
        if spatial_dilation<1:raise ValueError('spatial_dilation must be positive')
        self.hidden=nn.ModuleList([nn.Conv2d(channels,channels,3,
            padding=spatial_dilation if i%2 else 1,dilation=spatial_dilation if i%2 else 1) for i in range(blocks)])
        self.out_conv=nn.Conv2d(channels,1,3,padding=1)
        self.res_scale=nn.Parameter(torch.tensor(.1))
        # Same buffers exist in both variants; no unused parameter padding.
        self.nodes=nn.ModuleList([BurstIF(channels,time_steps) for _ in range(blocks+1)])
        if kind=='spike':
            for i,conv in enumerate(self.hidden):
                conv.burst_event_candidate=True
                object.__setattr__(conv,'burst_node',self.nodes[i])
            self.out_conv.burst_event_candidate=True
            object.__setattr__(self.out_conv,'burst_node',self.nodes[-1])
        self.spike_penalty=None;self.last_firing_rate=0.
        self.collect_activity=True
        self.spatial_stride=1

    def set_training_blend(self,blend):
        for node in self.nodes:node.training_blend=float(blend)

    @property
    def gain(self):
        return self.res_scale

    def forward(self,x,intervention=None):
        if intervention=='identity':return x
        if intervention not in (None,'identity','zero_spikes','reset_memory','shift_spikes'):
            raise ValueError(f'Unknown intervention: {intervention}')
        rates=[]
        h=self.in_conv(x)
        for i,node in enumerate(self.nodes):
            h=torch.nn.functional.leaky_relu(h,.1)
            if self.kind=='spike':
                h=node(h)
                if self.training or self.collect_activity:rates.append(node.counts.abs().mean()/node.levels)
                if intervention=='zero_spikes':h=torch.zeros_like(h)
                elif intervention=='shift_spikes':h=torch.roll(h,(h.shape[-2]//2,h.shape[-1]//2),(-2,-1))
            if i<len(self.hidden):h=self.hidden[i](h)
        self.spike_penalty=torch.stack(rates).mean() if rates else (x.new_zeros(()) if self.collect_activity or self.training else None)
        residual=self.out_conv(h,x.shape[-2:]) if self.spatial_stride==2 else self.out_conv(h)
        return x+self.res_scale*residual


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
    def __init__(self, channels=32, blocks=2, time_steps=4, threshold=1.0, decay=0.5,
                 learnable_neurons=False, membrane_mix=0.0, analog_refine=False):
        super().__init__()
        self.time_steps = time_steps
        self.in_conv = nn.Conv2d(1, channels, 3, padding=1)
        self.hidden = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(blocks)])
        self.out_conv = nn.Conv2d(channels, 1, 3, padding=1)
        self.lif_in = LIFNode(threshold, decay, learnable_neurons)
        self.lif_hidden = nn.ModuleList(
            [LIFNode(threshold, decay, learnable_neurons) for _ in range(blocks)])
        self.input_gain = nn.Parameter(torch.tensor(1.0))
        self.res_scale = nn.Parameter(torch.tensor(0.2))
        mix = min(max(float(membrane_mix), 1e-4), 1 - 1e-4)
        self.use_membrane_path = float(membrane_mix) > 0
        self.membrane_mix_raw = nn.Parameter(torch.logit(torch.tensor(mix)))
        self.temporal_logits = nn.Parameter(torch.zeros(time_steps))
        self.analog_refine = None
        self.analog_scale = None
        if analog_refine:
            self.analog_refine = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(channels, 1, 3, padding=1))
            self.analog_scale = nn.Parameter(torch.tensor(0.1))

        self.last_firing_rate = 0.0
        # Plain attributes: old checkpoints remain strictly loadable.
        # Opt in only from OUR hybrid; standalone/baseline priors keep old math.
        self.fast_inference = False
        self.collect_firing_rate = True

    def forward(self, x):
        if (self.fast_inference and not self.training and not torch.is_grad_enabled()
                and x.device.type in ('cpu', 'cuda')
                and not torch.is_autocast_enabled()
                and (x.device.type != 'cpu' or not torch.is_autocast_cpu_enabled())):
            return self._forward_inference(x)
        stats = SpikeStats()
        mem_in = torch.zeros(x.size(0), self.in_conv.out_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        mem_hidden = [
            torch.zeros_like(mem_in) for _ in self.hidden
        ]
        readouts = []
        membrane_mix = self.membrane_mix_raw.sigmoid() if self.use_membrane_path else 0.0

        # Direct coding: the same analog residual input is injected at each time step.
        for _ in range(self.time_steps):
            h = self.input_gain.clamp(0.1, 10.0) * self.in_conv(x)
            s, mem_in = self.lif_in(h, mem_in)
            stats.add(s)

            for i, conv in enumerate(self.hidden):
                h = conv(s + membrane_mix * torch.tanh(mem_hidden[i]))
                s, mem_hidden[i] = self.lif_hidden[i](h, mem_hidden[i])
                stats.add(s)
            readouts.append(self.out_conv(mem_hidden[-1]))

        # Membrane readout for reconstruction; this implementation uses dense
        # convolutions, even when their inputs are binary spikes.
        self.last_firing_rate = stats.firing_rate
        self.spike_penalty = stats.differentiable_rate
        weights = self.temporal_logits.softmax(dim=0)
        residual = sum(w * value for w, value in zip(weights, readouts))
        if self.analog_refine is not None:
            residual = residual + self.analog_scale * self.analog_refine(x)
        return x + self.res_scale * residual


    def _forward_inference(self, x):
        """Preserve T, weights, membrane dynamics and readout order.

        Cache only quantities that are constant during this forward pass.
        Disabled statistics report NaN, never a stale or fabricated firing rate.
        Hidden convolutions are still dense; this is not a sparse CUDA kernel.
        """
        drive = self.input_gain.clamp(0.1, 10.0) * self.in_conv(x)
        mem_in = torch.zeros_like(drive)
        mem_hidden = [torch.zeros_like(drive) for _ in self.hidden]
        nodes = [self.lif_in, *self.lif_hidden]
        thresholds = [node.threshold for node in nodes]
        decays = [node.decay for node in nodes]
        mix = self.membrane_mix_raw.sigmoid() if self.use_membrane_path else None
        spike_total = torch.zeros((), device=x.device, dtype=torch.float64)
        neuron_total = 0
        readouts = []
        for _ in range(self.time_steps):
            mem_in = decays[0] * mem_in + drive
            s = (mem_in - thresholds[0] > 0).to(x.dtype)
            mem_in = mem_in - s * thresholds[0]
            if self.collect_firing_rate:
                spike_total += s.sum().to(torch.float64)
                neuron_total += s.numel()
            for i, conv in enumerate(self.hidden):
                conv_input = s if mix is None else s + mix * torch.tanh(mem_hidden[i])
                h = conv(conv_input)
                mem_hidden[i] = decays[i + 1] * mem_hidden[i] + h
                s = (mem_hidden[i] - thresholds[i + 1] > 0).to(x.dtype)
                mem_hidden[i] = mem_hidden[i] - s * thresholds[i + 1]
                if self.collect_firing_rate:
                    spike_total += s.sum().to(torch.float64)
                    neuron_total += s.numel()
            readouts.append(self.out_conv(mem_hidden[-1]))
        self.last_firing_rate = (spike_total.item() / max(neuron_total, 1)
                                 if self.collect_firing_rate else float('nan'))
        weights = self.temporal_logits.softmax(dim=0)
        residual = sum(w * value for w, value in zip(weights, readouts))
        if self.analog_refine is not None:
            residual = residual + self.analog_scale * self.analog_refine(x)
        return x + self.res_scale * residual


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
    def __init__(self, cs_op, kind="ann", stages=5, channels=32, blocks_per_stage=2,
                 time_steps=4, threshold=1.0, decay=0.5, learnable_neurons=False,
                 membrane_mix=0.0, analog_refine=False, final_refine=False,
                 refine_blocks=6, prior_mode='legacy', spatial_dilation=1):
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
            if prior_mode=='burst':
                if kind not in ('ann','spike'):raise ValueError('Burst control supports ann/spike')
                self.priors.append(MatchedBurstPrior(channels,blocks_per_stage,time_steps,kind,spatial_dilation))
            elif kind in {"ann", "ista"}:
                self.priors.append(prior_cls(channels, blocks_per_stage))
            else:
                self.priors.append(prior_cls(
                    channels, blocks_per_stage, time_steps, threshold, decay,
                    learnable_neurons, membrane_mix, analog_refine,
                ))
        self.final_refine = None
        self.final_refine_scale = None
        if final_refine:
            layers = [nn.Conv2d(1, channels, 3, padding=1), nn.PReLU()]
            for _ in range(refine_blocks):
                layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU()]
            layers += [nn.Conv2d(channels, 1, 3, padding=1)]
            self.final_refine = nn.Sequential(*layers)
            nn.init.zeros_(self.final_refine[-1].weight)
            nn.init.zeros_(self.final_refine[-1].bias)
            self.final_refine_scale = nn.Parameter(torch.tensor(1.0))


    def forward(self, y, return_stages=False, intervention=None):
        x = self.cs.adjoint(y)
        outputs=[]
        for k, prior in enumerate(self.priors):
            r = x - self.rhos[k] * self.cs.normal_residual(x, y)
            if isinstance(prior,MatchedBurstPrior):x=prior(r,intervention)
            else:
                if intervention is not None:raise ValueError('Legacy prior has no intervention protocol')
                x=prior(r)
            outputs.append(x)
        if self.final_refine is not None:
            x = x + self.final_refine_scale * self.final_refine(x)
        if outputs:outputs[-1]=x
        return (x,outputs) if return_stages else x

    def firing_rate(self):
        if self.kind != "spike":
            return 0.0
        if self.priors and isinstance(self.priors[0],MatchedBurstPrior):
            if any(p.spike_penalty is None for p in self.priors):return float('nan')
            return float(torch.stack([p.spike_penalty for p in self.priors]).mean().detach())
        rates = [p.last_firing_rate for p in self.priors]
        return sum(rates) / max(len(rates), 1)

    def spike_regularization(self):
        return torch.stack([p.spike_penalty for p in self.priors]).mean()


def build_model(config):
    cs_cfg = config["cs"]
    model_cfg = config["model"]
    cs_op = BlockCSOperator(
        block_size=cs_cfg["block_size"],
        measurement_rate=cs_cfg["measurement_rate"],
        seed=config.get("seed", 123),
        learnable=cs_cfg.get("learnable", False),
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
        learnable_neurons=model_cfg.get("learnable_neurons", False),
        membrane_mix=model_cfg.get("membrane_mix", 0.0),
        analog_refine=model_cfg.get("analog_refine", False),
        final_refine=model_cfg.get("final_refine", False),
        refine_blocks=model_cfg.get("refine_blocks", 6),
        prior_mode=model_cfg.get('prior_mode','legacy'),
        spatial_dilation=model_cfg.get('spatial_dilation',1),
    )


def build_repository_reconstruction(kind='spike',channels=64,stages=10,time_steps=8,
                                    block_size=32,measurement_rate=.25,spatial_dilation=1,
                                    learnable_sensing=False,freeze_sensing=False,burst_calibration='rms',
                                    snn_spatial_stride=1,snn_fullres_tail=0):
    """Original UnrolledCSNet; orthonormal initialization, optional learned sensing.

    Learned sensing is frozen for both comparison branches after common training.
    It remains a counted parameter because it is trained and used in inference.
    """
    model=build_model(dict(seed=20261013,
        cs=dict(block_size=block_size,measurement_rate=measurement_rate,learnable=learnable_sensing),
        model=dict(kind=kind,stages=stages,channels=channels,blocks_per_stage=2,
                   time_steps=time_steps,prior_mode='burst',spatial_dilation=spatial_dilation)))
    generator=torch.Generator().manual_seed(20261013)
    q,_=torch.linalg.qr(torch.randn(model.cs.n,model.cs.m,generator=generator),mode='reduced')
    with torch.no_grad():
        model.cs.phi.copy_(q.T)
        model.rhos.fill_(1.)
    if learnable_sensing and freeze_sensing:model.cs.phi.requires_grad_(False)
    if burst_calibration not in ('rms','mse'):raise ValueError('Unknown burst calibration mode')
    if snn_spatial_stride not in (1,2):raise ValueError('SNN spatial stride must be 1 or 2')
    if snn_spatial_stride!=1 and kind!='spike':raise ValueError('Spatial synthesis changes are SNN-only')
    if not 0<=snn_fullres_tail<stages:raise ValueError('Full-resolution tail must be smaller than stage count')
    if snn_fullres_tail and (snn_spatial_stride!=2 or kind!='spike'):
        raise ValueError('Full-resolution tail requires stride-2 SNN')
    if snn_spatial_stride==2:
        for prior in model.priors[:stages-snn_fullres_tail]:
            prior.spatial_stride=2
            prior.in_conv.stride=(2,2)
            old=prior.out_conv
            readout=BurstTransposeReadout(old.in_channels,old.out_channels,old.kernel_size,padding=old.padding)
            readout.load_state_dict(old.state_dict(),strict=True)
            readout.burst_event_candidate=True
            object.__setattr__(readout,'burst_node',prior.nodes[-1])
            prior.out_conv=readout
    for module in model.modules():
        if isinstance(module,BurstIF):module.calibration_mode=burst_calibration
    return model
