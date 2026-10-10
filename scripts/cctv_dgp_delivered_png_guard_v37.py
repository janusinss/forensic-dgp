"""Exact PNG-valued forward path with a declared, VM-only coarse derivative.

Flooring has a zero derivative almost everywhere. The backward rule below is
an identity surrogate on observed support, not the derivative of the PNG loss.
It is experimental diagnostic machinery, not an app processing change.
"""
import platform
import sys
import numpy as np
import torch
from torch.nn import functional as F


class ObservedPNGFloorSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, camera, mask):
        assert value.dtype == camera.dtype == torch.float32 and value.shape == camera.shape
        assert value.ndim == 4 and value.shape[1:] == (3, 256, 256)
        assert mask.shape == (value.shape[0], 1, 256, 256)
        raw = value.detach().cpu().numpy().copy()
        assert np.isfinite(raw).all() and (raw >= 0).all() and (raw <= 1).all()
        bytes8 = np.floor(raw * np.float32(255)).astype(np.uint8)
        # Match the actual application encoder's NumPy float32 division.
        canonical = bytes8.astype(np.float32) / np.float32(255)
        result = torch.where(mask.bool(), torch.from_numpy(canonical).to(value.device), camera.detach())
        ctx.save_for_backward(mask)
        return result

    @staticmethod
    def backward(ctx, gradient):
        assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing VM only; no local gradients'
        assert gradient.is_cuda, 'Coarse derivative diagnostic requires existing L4'
        mask, = ctx.saved_tensors
        return gradient * mask, None, None


def delivered_png(value, camera, mask):
    return ObservedPNGFloorSurrogate.apply(value, camera, mask)


def uniform7_float32(value):
    """Separable double accumulation with float32 intermediate filter outputs."""
    assert value.dtype == torch.float32 and value.shape[-2:] == (256, 256)
    indices = torch.cat((torch.tensor([2, 1, 0], device=value.device),
                         torch.arange(256, device=value.device),
                         torch.tensor([255, 254, 253], device=value.device)))
    vertical = F.avg_pool2d(value.index_select(2, indices).double(), (7, 1), 1).float()
    return F.avg_pool2d(vertical.index_select(3, indices).double(), (1, 7), 1).float()


def masked_mse(value, target, mask):
    squared = (value - target).square().double()
    return (squared * mask).sum((1, 2, 3)) / (mask.double().sum((1, 2, 3)) * 3)


def masked_ssim(value, target, valid7):
    """RGB seven-pixel uniform SSIM, sample covariance, observed valid interior."""
    u, v = uniform7_float32(value), uniform7_float32(target)
    vx = (uniform7_float32(value.square()) - u.square()) * (49 / 48)
    vy = (uniform7_float32(target.square()) - v.square()) * (49 / 48)
    covariance = (uniform7_float32(value * target) - u * v) * (49 / 48)
    numerator = (2 * u * v + .01 ** 2) * (2 * covariance + .03 ** 2)
    denominator = (u.square() + v.square() + .01 ** 2) * (vx + vy + .03 ** 2)
    score = numerator / denominator
    return (score.double() * valid7).sum((1, 2, 3)) / (valid7.double().sum((1, 2, 3)) * 3)
