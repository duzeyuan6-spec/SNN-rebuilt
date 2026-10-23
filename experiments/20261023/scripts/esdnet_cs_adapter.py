"""Official ESDNet with a frozen CS front end; explicitly a task adaptation.

Upstream: https://github.com/MingTian99/ESDNet (IJCAI 2024).
Architecture is retained at the published training configuration. Differences:
grayscale I/O, Phi^T y input, portable torch backend and device-safe upsampling.
No claim of exact parameter matching or original-paper reproduction is made.
"""
import ast
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def build_esdnet(source, *, dim=48):
    from spikingjelly.activation_based import functional, layer
    source = Path(source)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    class TorchBackend(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == 'backend' and isinstance(node.value, ast.Constant) and node.value.value == 'cupy':
                node.value = ast.copy_location(ast.Constant('torch'), node.value)
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(TorchBackend().visit(tree))
    # Upstream instantiates a CUDA model at import time. Import definitions only.
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module == 'thop') and not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))]
    namespace = {'__name__': 'official_esdnet_definitions'}
    exec(compile(tree, str(source), 'exec'), namespace)
    # Equivalent bilinear interpolation; removes hard-coded default CUDA device.
    def portable_upsample(self, x):
        t,b,c,h,w = x.shape
        return self.up(F.interpolate(x.reshape(t*b,c,h,w), scale_factor=2,
                                    mode='bilinear', align_corners=False).reshape(t,b,c,2*h,2*w))
    namespace['UpSampling'].forward = portable_upsample
    model = namespace['ESDNet'](inp_channels=1,out_channels=1,dim=dim,
        en_num_blocks=[4,4,8,8],de_num_blocks=[2,2,2,2],T=4)
    # Released SpikingJelly tdBN omits the dimensionality checker inherited
    # from abstract _BatchNorm. Its forward flattens T,N into a 4-D batch.
    for module in model.modules():
        if isinstance(module, layer.ThresholdDependentBatchNorm2d):
            module._check_input_dim = types.MethodType(nn.BatchNorm2d._check_input_dim, module)
    functional.set_backend(model, 'torch')
    functional.set_step_mode(model, 'm')
    return model


class ESDNetCS(nn.Module):
    def __init__(self, source, phi, block_size=32, dim=48):
        super().__init__()
        self.register_buffer('phi', phi.detach().clone())
        self.block_size = block_size
        self.restorer = build_esdnet(source, dim=dim)

    def forward(self, x):
        from spikingjelly.activation_based import functional
        functional.reset_net(self.restorer)
        weight = self.phi.reshape(-1,1,self.block_size,self.block_size)
        y = F.conv2d(x,weight,stride=self.block_size)
        initial = F.conv_transpose2d(y,weight,stride=self.block_size)
        return self.restorer(initial)

    def reset(self):
        from spikingjelly.activation_based import functional
        functional.reset_net(self.restorer)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(123)
    phi=torch.linalg.qr(torch.randn(1024,256)).Q.T
    model=ESDNetCS(a.source,phi).to(a.device)
    x=torch.rand(1,1,32,32,device=a.device)
    model.train();y=model(x);loss=(y-x).square().mean();loss.backward()
    assert y.shape==x.shape and torch.isfinite(y).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.reset();model.eval()
    with torch.inference_mode():
        first=model(x).clone();second=model(x).clone()
    assert torch.equal(first,second), 'Neuron state leaked between images'
    print({'test':'forward_backward_and_reset_passed','decoder_parameters':sum(p.numel() for p in model.parameters()),
           'frozen_sensing_coefficients':model.phi.numel(),'total_capacity':sum(p.numel() for p in model.parameters())+model.phi.numel()})
