"""Prospective pre-clamp residual supervision; no model or optimizer is defined.

The RGB Laplacian construction is informed by MPRNet's published edge loss.
This version adds strict observed support and three scales. It is not MPRNet.
Training targets must remain unavailable to the restoration forward path.
"""
import platform
import sys

import torch
from torch.nn import functional as F

EPSILON = 1e-3
DELTA_LIMIT = .5 - 1e-6
LEVEL_WEIGHTS = (1., .5, .25)
SUPPORT_RADII = (4, 10, 22)
TERMS = ('observed_correction', 'landmark_correction', 'RGB_pyramid_correction')


def _validate(value, support):
    assert value.ndim == 4 and value.shape[1:] == (3, 256, 256)
    assert value.dtype in (torch.float32, torch.float64)
    assert support.shape == (len(value), 1, 256, 256)
    assert support.dtype == torch.bool and value.device == support.device
    assert bool(torch.isfinite(value).all())
    assert bool(support.flatten(1).any(1).all())
    if value.requires_grad:
        assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', \
            'Gradient-bearing residual supervision requires the existing manual Linux VM'
        assert value.device.type == 'cuda' and 'L4' in torch.cuda.get_device_name(0)


def observed_mean(value, support):
    return (value * support).sum((2, 3), keepdim=True) / support.sum((2, 3), keepdim=True)


def projected_teacher(baseline, clean, support, clear):
    """Fixed bounded paired correction; clear controls explicitly teach zero.

    This projection is an oracle target using a clean TRAIN reference. It is
    neither a model prediction nor evidence of representability or quality.
    """
    _validate(baseline, support)
    assert clean.shape == baseline.shape and clean.dtype == baseline.dtype
    assert clean.device == baseline.device and clear.shape == (len(baseline),)
    assert clear.dtype == torch.bool and clear.device == baseline.device
    assert not baseline.requires_grad and not clean.requires_grad
    assert bool(torch.isfinite(clean).all())
    assert bool(((baseline >= 0) & (baseline <= 1)).all())
    assert bool(((clean >= 0) & (clean <= 1)).all())
    with torch.no_grad():
        residual = clean - baseline
        maximum = residual.masked_fill(~support, -torch.inf).amax((2, 3), keepdim=True)
        minimum = residual.masked_fill(~support, torch.inf).amin((2, 3), keepdim=True)
        delta = (residual - (minimum + maximum) / 2).clamp(-DELTA_LIMIT, DELTA_LIMIT)
        centered = delta - observed_mean(delta, support)
        centered = torch.where(support & ~clear[:, None, None, None], centered, 0)
        image = torch.where(support, (baseline + centered).clamp(0, 1), baseline)
        return centered.detach(), image.detach()


def _erode(support, radius):
    missing = F.pad((~support).to(torch.float64), (radius,) * 4, value=1)
    missing = F.max_pool2d(missing, (1, 2 * radius + 1), stride=1)
    missing = F.max_pool2d(missing, (2 * radius + 1, 1), stride=1)
    return missing == 0


def _blur(image):
    taps = image.new_tensor([.05, .25, .4, .25, .05])
    horizontal = taps.reshape(1, 1, 1, 5).repeat(3, 1, 1, 1)
    vertical = taps.reshape(1, 1, 5, 1).repeat(3, 1, 1, 1)
    image = F.conv2d(F.pad(image, (2, 2, 0, 0), mode='replicate'), horizontal, groups=3)
    return F.conv2d(F.pad(image, (0, 0, 2, 2), mode='replicate'), vertical, groups=3)


def _laplacian(image):
    low = _blur(image)[:, :, ::2, ::2]
    expanded = torch.zeros_like(image)
    expanded[:, :, ::2, ::2] = low * 4
    return image - _blur(expanded), low


def _robust_mean(error, support):
    assert bool(support.flatten(1).any(1).all()), 'Insufficient valid support for the declared scale'
    value = torch.sqrt(error.square() + EPSILON ** 2) - EPSILON
    return (value * support).sum((1, 2, 3)) / (3 * support.sum((1, 2, 3)))


def correction_terms(predicted_centered, teacher_centered, support, feature):
    """Per-case absolute residual terms; no baseline-relative regression hinge.

    These are candidate reconstruction terms only. Identity supervision and
    immutable raw/PNG preservation gates still need the separate VM protocol.
    """
    _validate(predicted_centered, support)
    assert teacher_centered.shape == predicted_centered.shape
    assert teacher_centered.dtype == predicted_centered.dtype
    assert teacher_centered.device == predicted_centered.device
    assert not teacher_centered.requires_grad and bool(torch.isfinite(teacher_centered).all())
    assert feature.shape == support.shape and feature.dtype == torch.bool
    assert feature.device == support.device and not bool((feature & ~support).any())
    assert bool(feature.flatten(1).any(1).all())
    error = predicted_centered - teacher_centered
    whole = _robust_mean(error, support)
    landmark = _robust_mean(error, feature)
    a, b = predicted_centered, teacher_centered
    detail = torch.zeros_like(whole)
    for level, (weight, radius) in enumerate(zip(LEVEL_WEIGHTS, SUPPORT_RADII)):
        aa, a = _laplacian(a)
        bb, b = _laplacian(b)
        valid = _erode(support, radius)[:, :, ::2 ** level, ::2 ** level]
        assert valid.shape == aa[:, :1].shape
        detail = detail + weight * _robust_mean(aa - bb, valid)
    return {'observed_correction': whole, 'landmark_correction': landmark,
            'RGB_pyramid_correction': detail / sum(LEVEL_WEIGHTS)}
