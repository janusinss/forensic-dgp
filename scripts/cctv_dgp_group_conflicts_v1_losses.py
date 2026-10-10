"""Forward/differentiable surrogates; scientific evaluation uses retained metrics."""
import torch
from torch.nn import functional as F


def pixel_and_structure_losses(actual, target, mask, support):
    assert actual.dtype == target.dtype == torch.float32
    assert actual.shape == target.shape and actual.shape[1:] == (3, 256, 256)
    assert mask.shape == support.shape == (len(actual), 1, 256, 256)
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

    # The retained high-frequency metric uses uint8 target / 255 in float64.
    # RGB/SSIM instead use float32 targets; keep those definitions separate.
    exact_target = (y * 255.).round() / 255.
    feature = ((high(x) - high(exact_target)).square() * support).sum((1, 2, 3)) / support.sum((1, 2, 3))
    assert torch.isfinite(mse).all() and torch.isfinite(ssim).all() and torch.isfinite(feature).all()
    return {'MSE': mse, 'SSIM_loss': 1. - ssim, 'landmark_structure': feature}
