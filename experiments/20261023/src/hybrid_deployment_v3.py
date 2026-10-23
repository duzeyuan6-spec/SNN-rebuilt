"""Opt-in compilation for OUR hybrid only; no global backend changes.

Set TORCHDYNAMO_DISABLE=0 before importing the hybrid evaluation entry point.
Use FixedShapeRunner after a warmup call for fixed-shape deployment. Compilation
changes floating-point rounding; validate outputs for each deployment platform.
Training models, checkpoints, and external baseline instances are never mutated.
"""
import copy
import os
import torch


def compile_hybrid_transition(deployed_hybrid):
    if os.environ.get('TORCHDYNAMO_DISABLE') == '1':
        raise RuntimeError('Set TORCHDYNAMO_DISABLE=0 before importing evaluation modules')
    if deployed_hybrid.training:
        raise ValueError('Expected an eval-mode FP16Deployment of our hybrid')
    inner = getattr(deployed_hybrid, 'model', None)
    if inner is None or not all(hasattr(inner, n) for n in
                                ('backbone', 'spike_refiner', 'texture_refiner')):
        raise TypeError('Only our deployed hybrid is supported')
    result = copy.deepcopy(deployed_hybrid)
    result.model.backbone.transition = torch.compile(
        result.model.backbone.transition, fullgraph=True, dynamic=False)
    return result
