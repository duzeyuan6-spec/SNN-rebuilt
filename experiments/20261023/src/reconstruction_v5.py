"""Signed-event candidate: communicate both polarities and retain emitted charge.

This is a new from-scratch experiment, not a validated accuracy/energy claim.
Ordinary torch convolutions remain dense, including on signed event tensors.
"""
import torch
from torch import nn
from .reconstruction_v3 import StableSpike
from .reconstruction_v4 import PackedSpikePrior,EfficientStagewiseRecon


class SignedBoundedLIF(nn.Module):
    def __init__(self,threshold=.5,decay=.7,bound=2.):
        super().__init__()
        self.register_buffer('threshold',torch.tensor(threshold))
        self.register_buffer('decay',torch.tensor(decay))
        self.bound=bound

    def forward(self,current,membrane):
        voltage=self.bound*torch.tanh((self.decay*membrane+current)/self.bound)
        event=StableSpike.apply(voltage-self.threshold)-StableSpike.apply(-voltage-self.threshold)
        # Signed emitted charge is conserved at reset, after voltage bounding.
        return event,voltage-event*self.threshold


class SignedPackedPrior(PackedSpikePrior):
    def __init__(self,channels,time_steps,kind):
        super().__init__(channels,time_steps,kind)
        if kind=='spike':
            self.nodes=nn.ModuleList([SignedBoundedLIF() for _ in range(3)])
            for layer in self.hidden:layer.signed_event_candidate=True

    def analog_activation(self,value):
        # Give the analog control signed features too; do not weaken its range.
        return torch.tanh(value)

    def event_activity(self,spikes):
        return spikes.abs().mean()

    def readout_state(self,state,spikes):
        # V4 reads only the post-reset membrane, discarding the emitted charge.
        return torch.tanh(state+spikes*self.nodes[-1].threshold)


class SignedStagewiseRecon(EfficientStagewiseRecon):
    def __init__(self,kind='spike',channels=96,stages=5,time_steps=4,**kwargs):
        super().__init__(kind=kind,channels=channels,stages=stages,time_steps=time_steps,**kwargs)
        self.priors=nn.ModuleList([SignedPackedPrior(channels,time_steps,kind) for _ in range(stages)])
