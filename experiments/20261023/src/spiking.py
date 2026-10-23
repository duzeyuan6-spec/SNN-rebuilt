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
        return grad_output * (1.0 - x.abs() / 2.0).clamp(min=0.0)


class LIFNode(nn.Module):
    def __init__(self, threshold=1.0, decay=0.5, learnable=False):
        super().__init__()
        threshold_raw = torch.tensor(float(threshold)).log()
        decay = min(max(float(decay), 1e-4), 1 - 1e-4)
        decay_raw = torch.logit(torch.tensor(decay))
        if learnable:
            self.threshold_raw = nn.Parameter(threshold_raw)
            self.decay_raw = nn.Parameter(decay_raw)
        else:
            self.register_buffer("threshold_raw", threshold_raw)
            self.register_buffer("decay_raw", decay_raw)

    @property
    def threshold(self):
        return self.threshold_raw.exp().clamp(0.05, 3.0)

    @property
    def decay(self):
        return self.decay_raw.sigmoid().clamp(0.05, 0.99)

    def forward(self, x, mem):
        mem = self.decay * mem + x
        spike = SpikeFn.apply(mem - self.threshold)
        mem = mem - spike.detach() * self.threshold
        return spike, mem


class SpikeStats:
    def __init__(self):
        self.spikes = 0.0
        self.neurons = 0.0
        self.training_spikes = None

    def add(self, spike):
        total = spike.float().sum()
        self.training_spikes = total if self.training_spikes is None else self.training_spikes + total
        self.spikes = self.spikes + total.detach()
        self.neurons += spike.numel()

    @property
    def differentiable_rate(self):
        if self.training_spikes is None:
            raise RuntimeError("No spikes collected in this forward pass")
        return self.training_spikes / max(self.neurons, 1)

    @property
    def firing_rate(self):
        if self.neurons == 0:
            return 0.0
        return float(self.spikes / self.neurons)


class BurstIF(nn.Module):
    """Static-input signed integrate-and-fire count, with a calibrated quantum.

    One charge injection emits up to levels positive or negative events. Counts
    are contracted across time before linear convolution, so torch still runs a
    dense convolution. This is NOT the same dynamics as the legacy leaky neuron.
    Calibration is channelwise and updated only from training data.
    """
    def __init__(self, channels, levels=8):
        super().__init__()
        if levels<1:raise ValueError('levels must be positive')
        self.levels=levels
        self.register_buffer('quantum',torch.ones(1,channels,1,1)/levels)
        self.register_buffer('calibrated',torch.tensor(False))
        self.training_blend=1.
        self.counts=None
        self.calibration_mode='rms'

    @torch.no_grad()
    def calibration_candidate(self, voltage):
        if self.calibration_mode=='rms':
            return (3*voltage.detach().square().mean((0,2,3),keepdim=True).sqrt()/self.levels).clamp_min(1e-4)
        if self.calibration_mode!='mse':raise ValueError('Unknown burst calibration mode')
        # Fixed spatial subsampling uses training activations only. Include the
        # original 3-RMS candidate, then minimize channelwise reconstruction MSE.
        x=voltage.detach()[...,::4,::4].permute(1,0,2,3).reshape(voltage.shape[1],-1)
        rms=x.square().mean(1).sqrt()
        factors=x.new_tensor([1.,1.5,2.,2.5,3.,4.,6.])
        candidates=(factors[:,None]*rms[None,:]/self.levels).clamp_min(1e-4)
        q=candidates[:,:,None]
        decoded=(x[None,:,:]/q).round().clamp(-self.levels,self.levels)*q
        indices=(decoded-x[None,:,:]).square().mean(2).argmin(0)
        return candidates.gather(0,indices[None,:]).reshape(1,-1,1,1)

    def forward(self, voltage):
        if self.training:
            with torch.no_grad():
                # RMS-based clipping range preserves typical values rather than
                # letting one outlier determine every neuron's resolution.
                candidate=self.calibration_candidate(voltage)
                if not bool(self.calibrated):self.quantum.copy_(candidate);self.calibrated.fill_(True)
                else:self.quantum.lerp_(candidate,.01)
        scaled=voltage/self.quantum
        counts=scaled.round().clamp(-self.levels,self.levels)
        if not self.training and not torch.is_grad_enabled():
            # Inference needs discrete counts only; omit the training surrogate
            # and its extra pointwise kernels. Keep counts for event accounting.
            self.counts=counts
            return counts*self.quantum
        # Straight-through approximation only inside the represented range.
        surrogate=scaled.clamp(-self.levels,self.levels)
        self.counts=surrogate+(counts-surrogate).detach()
        reconstructed=self.counts*self.quantum
        blend=self.training_blend if self.training else 1.
        if blend>=1.:return reconstructed
        return voltage+blend*(reconstructed-voltage)

