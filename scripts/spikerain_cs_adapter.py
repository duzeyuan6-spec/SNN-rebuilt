"""Author SpikeRain-M, grayscale CS adaptation; no capacity matching claim."""
import importlib.util
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def author_ssim(source):
    import ast
    from math import exp
    path=Path(source)/'utils/image_utils.py'
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('gaussian','create_window','_ssim','SSIM')]
    ns=dict(torch=torch,F=F,Variable=torch.autograd.Variable,exp=exp)
    exec(compile(tree,str(path),'exec'),ns)
    return ns['SSIM']()

class SpikeRainCS(nn.Module):
    def __init__(self, source, phi):
        super().__init__()
        source=Path(source)
        spec=importlib.util.spec_from_file_location('official_spikerain',source/'model/__init__.py',submodule_search_locations=[str(source/'model')])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        from official_spikerain.spikerain import SpikeRain
        from spikingjelly.activation_based import functional
        self.register_buffer('phi',phi.detach().clone())
        self.restorer=SpikeRain(inp_channels=1,out_channels=1,dim=48,en_num_blocks=[4,4,8,8],de_num_blocks=[2,2,2,2],T=4)
        functional.set_step_mode(self.restorer,'m')
        functional.set_backend(self.restorer,'torch')

    def reset(self):
        from spikingjelly.activation_based import functional
        functional.reset_net(self.restorer)

    def forward(self,x):
        self.reset();weight=self.phi.reshape(-1,1,32,32)
        initial=F.conv_transpose2d(F.conv2d(x,weight,stride=32),weight,stride=32)
        return self.restorer(initial)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123)
    m=SpikeRainCS(a.source,torch.linalg.qr(torch.randn(1024,256)).Q.T).cuda()
    x=torch.rand(1,1,32,32,device='cuda');m.train();y=m(x);(1-author_ssim(a.source)(y,x)).backward()
    assert y.shape==x.shape and torch.isfinite(y).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    with torch.no_grad(): assert torch.equal(m(x),m(x))
    print(dict(test='forward_backward_reset_passed',decoder_parameters=sum(p.numel() for p in m.parameters()),total_capacity=sum(p.numel() for p in m.parameters())+m.phi.numel()))
