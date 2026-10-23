import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_reconstruction_v2 import metric_structure_loss


class MetricLossTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123);torch.set_num_threads(2)

    def test_matches_skimage_on_same_arrays(self):
        for shape in [(1,1,16,19),(2,1,24,32)]:
            target=torch.rand(shape,dtype=torch.float64)
            pred=(target+.1*torch.randn(shape,dtype=torch.float64)).clamp(0,1)
            scores=[structural_similarity(t[0].numpy(),p[0].numpy(),data_range=1.) for t,p in zip(target,pred)]
            self.assertAlmostEqual(float(1-metric_structure_loss(pred,target)),float(np.mean(scores)),places=10)

    def test_identical_constant_and_gradient(self):
        x=torch.full((1,1,12,15),.4,dtype=torch.float64)
        self.assertAlmostEqual(float(metric_structure_loss(x,x)),0.,places=10)
        p=(x+.03*torch.randn_like(x)).requires_grad_()
        metric_structure_loss(p,x).backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        self.assertGreater(float(p.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main(verbosity=2)
