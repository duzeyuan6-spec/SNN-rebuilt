"""Rate-general OCTUF loader; original decoder and old MR=.25 adapter unchanged."""
import torch
from torch import nn
from ann_cs_adapters import definitions, FILES


class OfficialCS(nn.Module):
    def __init__(self, source, kind, phi):
        super().__init__()
        if kind != 'octuf' or phi.ndim != 2 or phi.shape[1] != 1024:
            raise ValueError('This extension only supports OCTUF and 32x32 sensing')
        self.kind = kind
        self.aux = None
        self.net = definitions(source, kind)['OCT'](10, phi.shape[0] / 1024)
        with torch.no_grad():
            self.net.Phiweight.copy_(phi.reshape(-1, 1, 32, 32))
        self.net.Phiweight.requires_grad_(False)

    def forward(self, x):
        return self.net(x)

    def reset(self):
        self.aux = None

    def fit_initializer(self, entries):
        pass

    def loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
