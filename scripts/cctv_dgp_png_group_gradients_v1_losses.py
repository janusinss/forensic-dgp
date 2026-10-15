"""Exact delivered pixels forward; explicitly approximate straight-through backward."""
import torch
from torch.nn import functional as F
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def delivered_surrogates(raw, delivered_u8):
    assert raw.dtype == torch.float32 and raw.shape[1:] == (3, 256, 256)
    assert delivered_u8.dtype == torch.uint8 and delivered_u8.shape == raw.shape
    exact32 = delivered_u8.to(torch.float32) / 255.
    exact64 = delivered_u8.to(torch.float64) / 255.
    rgb32 = raw + (exact32 - raw).detach()
    raw64 = raw.double()
    feature64 = raw64 + (exact64 - raw64).detach()
    assert torch.equal(rgb32.detach(), exact32) and torch.equal(feature64.detach(), exact64)
    return rgb32, feature64


def pixel_and_structure_losses(actual, target, mask, support, feature_rgb64):
    assert actual.dtype == target.dtype == torch.float32
    assert actual.shape == target.shape and actual.shape[1:] == (3, 256, 256)
    assert mask.shape == support.shape == (len(actual), 1, 256, 256)
    assert feature_rgb64.dtype == torch.float64 and feature_rgb64.shape == actual.shape
    observed = mask.bool().expand_as(actual)
    error = (actual - target).square().double()
    mse = (error * observed).sum((1, 2, 3)) / observed.sum((1, 2, 3))
    x, y = actual.double(), target.double()
    average = lambda image: F.avg_pool2d(image, 7, stride=1)
    ux, uy = average(x), average(y)
    vx = (average(x * x) - ux * ux) * (49. / 48.)
    vy = (average(y * y) - uy * uy) * (49. / 48.)
    cov = (average(x * y) - ux * uy) * (49. / 48.)
    ssmap = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / (
        (ux * ux + uy * uy + .01 ** 2) * (vx + vy + .03 ** 2))
    interior = average(mask.double()) > 1. - 1e-12
    assert (interior.sum((1, 2, 3)) > 0).all()
    ssim = (ssmap * interior).sum((1, 2, 3)) / (3 * interior.sum((1, 2, 3)))
    z = torch.arange(-6, 7, device=actual.device, dtype=torch.float64)
    gaussian = torch.exp(-.5 * (z / 2.) ** 2)
    gaussian = gaussian / gaussian.sum()
    luma = torch.tensor([.299, .587, .114], dtype=torch.float64,
                        device=actual.device)[None, :, None, None]

    def high(image):
        value = (image * luma).sum(1, keepdim=True)
        smooth = F.conv2d(value, gaussian[None, None, None, :], padding=(0, 6))
        smooth = F.conv2d(smooth, gaussian[None, None, :, None], padding=(6, 0))
        return value - smooth

    # PNG feature evaluation uses uint8/255 in float64, rather than expanded
    # float32 RGB. Keep this forward definition distinct from RGB/SSIM input.
    exact_target = (y * 255.).round() / 255.
    feature = ((high(feature_rgb64) - high(exact_target)).square() * support).sum((1, 2, 3)) / support.sum((1, 2, 3))
    assert torch.isfinite(mse).all() and torch.isfinite(ssim).all() and torch.isfinite(feature).all()
    # The scientific forward uses the retained float32 scikit-image evaluator.
    # Its numerical operations are not the analytic float64 pooling graph;
    # only the explicitly approximate backward graph uses that expression.
    values = actual.detach().permute(0, 2, 3, 1).cpu().numpy()
    targets = target.detach().permute(0, 2, 3, 1).cpu().numpy()
    masks = mask.detach().cpu().numpy()[:, 0] > 0
    exact_losses = []
    for actual_np, target_np, observed_np in zip(values, targets, masks):
        _, ssmap = structural_similarity(target_np, actual_np, data_range=1,
            channel_axis=-1, win_size=7, full=True)
        interior_np = cv2.erode(observed_np.astype(np.uint8), np.ones((7, 7), np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        assert interior_np.any()
        exact_losses.append(1. - float(ssmap[interior_np].astype(np.float64).mean()))
    exact = torch.tensor(exact_losses, dtype=torch.float64, device=actual.device)
    surrogate = 1. - ssim
    ssim_loss = surrogate + (exact - surrogate).detach()
    assert torch.equal(ssim_loss.detach(), exact)
    return {'MSE': mse, 'SSIM_loss': ssim_loss, 'landmark_structure': feature}
