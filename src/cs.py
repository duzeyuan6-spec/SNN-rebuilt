import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockCSOperator(nn.Module):
    def __init__(self, block_size=33, measurement_rate=0.25, seed=123):
        super().__init__()
        self.block_size = block_size
        self.n = block_size * block_size
        self.m = int(round(self.n * measurement_rate))

        gen = torch.Generator()
        gen.manual_seed(seed)
        phi = torch.randn(self.m, self.n, generator=gen) / math.sqrt(self.m)
        # spectral norm normalization to avoid explosion from Phi^T Phi
        try:
            sv = torch.linalg.svdvals(phi)
            phi = phi / (sv[0] + 1e-12)
        except Exception:
            # fallback: l2 norm of matrix as coarse normalization
            norm = torch.linalg.norm(phi, ord=2)
            phi = phi / (norm + 1e-12)
        self.register_buffer("phi", phi)

    def forward(self, x):
        weight = self.phi.view(self.m, 1, self.block_size, self.block_size)
        return F.conv2d(x, weight, stride=self.block_size)

    def adjoint(self, y):
        weight = self.phi.view(self.m, 1, self.block_size, self.block_size)
        return F.conv_transpose2d(y, weight, stride=self.block_size)

    def normal_residual(self, x, y):
        return self.adjoint(self.forward(x) - y)

