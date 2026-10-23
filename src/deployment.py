"""Opt-in inference deployment; never change the training model or checkpoint."""
import copy
import torch
from torch import nn


def prepare_deployment(model, *, channels_last=False, fuse_bn=False):
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if hasattr(module, 'collect_firing_rate'):
            module.collect_firing_rate = False
            module.last_firing_rate = float('nan')
    # Only adjacent Conv -> BN is legal. Never fuse across a PReLU or concat.
    if fuse_bn:
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                children = list(module.named_children())
                for (cn, conv), (bn, norm) in zip(children, children[1:]):
                    if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                        module._modules[cn] = torch.nn.utils.fuse_conv_bn_eval(conv, norm)
                        module._modules[bn] = nn.Identity()
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model


class FixedShapeRunner:
    """Single-stream, fixed-shape CUDA graph; returned tensors own their memory.

    Setup is intentionally explicit and is not included in steady-state timing.
    Do not mutate/train/load weights after capture. Construct a new runner instead.
    Statistics must be disabled: host scalar reads cannot be captured.
    """
    @torch.inference_mode()
    def __init__(self, model, example):
        if model.training or not example.is_cuda:
            raise ValueError('An eval model and CUDA input are required')
        if any(getattr(m, 'collect_firing_rate', False) for m in model.modules()):
            raise ValueError('Disable firing-rate collection before graph capture')
        self.model = model
        self.input = example.clone()
        self.shape, self.dtype, self.device = example.shape, example.dtype, example.device
        side = torch.cuda.Stream(device=example.device)
        side.wait_stream(torch.cuda.current_stream(example.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                model(self.input)
        torch.cuda.current_stream(example.device).wait_stream(side)
        torch.cuda.synchronize(example.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model(self.input)

    @torch.inference_mode()
    def __call__(self, x):
        if x.shape != self.shape or x.dtype != self.dtype or x.device != self.device:
            raise ValueError('Shape, dtype or device differs from capture; create another runner')
        self.input.copy_(x)
        self.graph.replay()
        if isinstance(self.output, tuple):
            return tuple(y.clone() for y in self.output)
        return self.output.clone()


class FP16Deployment(nn.Module):
    """Explicit approximate mode: FP32 IO, FP16 internal math; validate quality first."""
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model).half().eval()
        self.eval()

    def forward(self, x):
        output = self.model(x.to(dtype=torch.float16))
        if isinstance(output, tuple):
            return tuple(y.float() for y in output)
        return output.float()
