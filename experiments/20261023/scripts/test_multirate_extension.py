"""Dimensional, gradient and MR=.25 compatibility checks before long runs."""
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models import build_repository_reconstruction
from ann_cs_adapters import OfficialCS as Original
from multirate_octuf_adapter import OfficialCS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    for rate in (.10,.25,.50):
        common=dict(channels=49,stages=11,time_steps=8,measurement_rate=rate,
                    spatial_dilation=2,learnable_sensing=True)
        ann=build_repository_reconstruction(kind='ann',**common)
        snn=build_repository_reconstruction(kind='spike',freeze_sensing=True,
                                          snn_spatial_stride=2,snn_fullres_tail=3,**common)
        snn.load_state_dict(ann.state_dict(),strict=True)
        assert torch.equal(ann.cs.phi,snn.cs.phi)
        assert sum(p.numel() for p in ann.parameters())==sum(p.numel() for p in snn.parameters())
        assert snn.cs.m==round(rate*1024)
        x=torch.rand(1,1,32,32)
        snn.train()
        pred=snn(snn.cs(x))
        pred.square().mean().backward()
        assert pred.shape==x.shape and torch.isfinite(pred).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in snn.parameters() if p.requires_grad)
        snn.eval()
        with torch.inference_mode():
            torch.testing.assert_close(snn(snn.cs(x)),snn(snn.cs(x)),rtol=0,atol=0)
        phi=ann.cs.phi.detach()
        model=OfficialCS(a.source,'octuf',phi)
        model.train()
        pred=model(x)
        model.loss(pred,x).backward()
        assert pred.shape==x.shape and torch.isfinite(pred).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert model.net.Phiweight.grad is None
        if rate==.25:
            old=Original(a.source,'octuf',phi).eval()
            old.load_state_dict(model.state_dict(),strict=True)
            model.eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(x),old(x),rtol=0,atol=0)
        print(dict(rate=rate,measurements=snn.cs.m,snn_capacity=sum(p.numel() for p in snn.parameters()),
                   decoder_trainable=sum(p.numel() for p in snn.parameters() if p.requires_grad),
                   octuf_decoder=sum(p.numel() for p in model.parameters() if p.requires_grad),passed=True),flush=True)


if __name__=='__main__':main()
