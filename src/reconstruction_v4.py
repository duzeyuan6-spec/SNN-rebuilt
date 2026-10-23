"""Candidate efficient SNN prior; quality and hardware energy are unverified.

Pixel unshuffle retains all image samples. Binary communication occurs between
the grouped spatial convolution and dense pointwise convolution. The analog
readout predicts four subpixels. Exact final projection uses the fixed
row-orthonormal sensing matrix, before any display/metric clipping.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .reconstruction_v3 import StablePrior
from .reconstruction_v2 import StagewiseRecon


class PackedSpikePrior(StablePrior):
    def __init__(self,channels,time_steps,kind):
        if channels%4:raise ValueError('V4 channels must be divisible by four')
        super().__init__(channels,time_steps,kind)
        self.input=nn.Conv2d(4,channels,3,padding=1)
        self.hidden=nn.ModuleList([nn.Conv2d(channels,channels,3,padding=1,groups=4),
                                  nn.Conv2d(channels,channels,1)])
        self.output=nn.Conv2d(channels,4,3,padding=1)
        nn.init.normal_(self.output.weight,std=.01/math.sqrt(channels*9))
        nn.init.zeros_(self.output.bias)
        if kind=='spike':
            for layer in self.hidden:layer.event_driven_candidate=True

    def forward(self,x,intervention=None):
        if x.shape[-1]%2 or x.shape[-2]%2:raise ValueError('V4 prior needs even spatial dimensions')
        # The inherited residual connects packed image samples, not ANN features.
        return F.pixel_shuffle(super().forward(F.pixel_unshuffle(x,2),intervention),2)


class EfficientStagewiseRecon(StagewiseRecon):
    def __init__(self,kind='spike',channels=96,stages=5,time_steps=4,**kwargs):
        super().__init__(kind=kind,channels=channels,stages=stages,time_steps=time_steps,**kwargs)
        if self.cs.block_size%2:raise ValueError('V4 requires an even measurement block size')
        self.priors=nn.ModuleList([PackedSpikePrior(channels,time_steps,kind) for _ in range(stages)])

    def forward(self,y,return_stages=False,intervention=None):
        x,outputs=super().forward(y,return_stages=True,intervention=intervention)
        # Phi Phi^T = I for this fixed acquisition operator.
        x=x-self.cs.normal_residual(x,y)
        outputs[-1]=x
        return (x,outputs) if return_stages else x
