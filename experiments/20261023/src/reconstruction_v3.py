"""Stability revision: bounded dynamics and residuals, matched analog control.

This is a new architecture, not a compatible continuation of round-1 weights.
Bounds control forward states; empirical gradient/validation guards are still
required. No claim of a globally bounded network Jacobian is made.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .reconstruction_v2 import StagewiseRecon


class StableSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad):
        voltage, = ctx.saved_tensors
        return grad * .3 * (1 - voltage.abs()).clamp_min(0)


class BoundedLIF(nn.Module):
    def __init__(self, threshold=.5, decay=.7, bound=2.):
        super().__init__()
        self.register_buffer('threshold', torch.tensor(threshold))
        self.register_buffer('decay', torch.tensor(decay))
        self.bound=bound

    def forward(self, current, membrane):
        pre=self.decay*membrane+current
        spikes=StableSpike.apply(pre-self.threshold)
        # Differentiate the reset as well; the surrogate slope is <=0.3.
        reset=pre-spikes*self.threshold
        membrane=self.bound*torch.tanh(reset/self.bound)
        return spikes,membrane


class StablePrior(nn.Module):
    def __init__(self, channels, time_steps, kind, residual_limit=.5):
        super().__init__()
        self.kind,self.time_steps=kind,time_steps
        self.residual_limit=residual_limit
        self.input=nn.Conv2d(1,channels,3,padding=1)
        self.hidden=nn.ModuleList([nn.Conv2d(channels,channels,3,padding=1) for _ in range(2)])
        self.norms=nn.ModuleList([nn.GroupNorm(1,channels,eps=1e-4) for _ in range(3)])
        self.output=nn.Conv2d(channels,1,3,padding=1)
        nn.init.normal_(self.output.weight,std=.01/math.sqrt(channels*9))
        nn.init.zeros_(self.output.bias)
        self.gain_raw=nn.Parameter(torch.tensor(math.log(.4/.6)))
        if kind=='spike':
            self.nodes=nn.ModuleList([BoundedLIF() for _ in range(3)])
            self.temporal_logits=nn.Parameter(torch.zeros(time_steps))
            for layer in self.hidden:layer.event_driven_candidate=True
        self.spike_penalty=None

    @property
    def gain(self):
        return .25+1.25*self.gain_raw.sigmoid()

    def current(self,value,index):
        return 2*torch.tanh(self.norms[index](value)/2)

    def analog_activation(self,value):
        return F.relu(value)

    def readout_state(self,state,spikes):
        return torch.tanh(state)

    def event_activity(self,spikes):
        return spikes.mean()

    def forward(self,x,intervention=None):
        if intervention not in (None,'identity','reset_memory','zero_spikes','shift_spikes'):
            raise ValueError(f'Unknown intervention: {intervention}')
        if intervention=='identity':return x
        drive=self.current(self.input(x),0)
        if self.kind=='ann':
            h=self.analog_activation(drive)
            for i,layer in enumerate(self.hidden):h=self.analog_activation(self.current(layer(h),i+1))
            readout=torch.tanh(h)
            self.spike_penalty=x.new_zeros(())
        else:
            states=[torch.zeros_like(drive) for _ in self.nodes]
            readouts,rates=[],[]
            for _ in range(self.time_steps):
                if intervention=='reset_memory':states=[torch.zeros_like(drive) for _ in self.nodes]
                s,states[0]=self.nodes[0](drive,states[0]);rates.append(self.event_activity(s))
                for i,layer in enumerate(self.hidden):
                    if intervention=='zero_spikes':
                        s=torch.zeros_like(s)
                    elif intervention=='shift_spikes':
                        # Deterministic permutation preserves this layer's spike count.
                        # Applied only at communication edges, not to the analog drive.
                        s=torch.roll(s,(s.shape[-2]//2,s.shape[-1]//2),(-2,-1))
                    s,states[i+1]=self.nodes[i+1](self.current(layer(s),i+1),states[i+1])
                    rates.append(self.event_activity(s))
                readouts.append(self.readout_state(states[-1],s))
            self.spike_penalty=torch.stack(rates).mean()
            readout=sum(w*r for w,r in zip(self.temporal_logits.softmax(0),readouts))
        residual=self.residual_limit*torch.tanh(self.output(readout)/self.residual_limit)
        return x+self.gain*residual


class StableStagewiseRecon(StagewiseRecon):
    def __init__(self,kind='spike',channels=48,stages=5,time_steps=4,**kwargs):
        super().__init__(kind=kind,channels=channels,stages=stages,time_steps=time_steps,**kwargs)
        self.priors=nn.ModuleList([StablePrior(channels,time_steps,kind) for _ in range(stages)])
