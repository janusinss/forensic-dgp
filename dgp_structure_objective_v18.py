"""Observed RGB and coarse structure losses; differentiable calls are VM-only."""
import torch
from torch.nn import functional as F


def reconstruction(generated, target, support, *, root):
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    if (target.requires_grad or support.requires_grad or generated.shape != target.shape
            or support.shape != (len(generated), 1, 256, 256)
            or target.dtype != torch.float32 or support.dtype != torch.float32
            or not torch.isfinite(generated).all() or not torch.isfinite(target).all()
            or not torch.isfinite(support).all() or not torch.all((support == 0) | (support == 1))):
        raise ValueError('Require matched finite RGB and detached binary observed support')
    denominator = (3 * support.sum((1, 2, 3))).clamp_min(1)
    mse = ((generated - target).square() * support).sum((1, 2, 3)) / denominator
    coarse = []
    for kernel in [8, 4, 2]:
        coverage = F.avg_pool2d(support, kernel)
        observed_generated = F.avg_pool2d(generated * support, kernel) / coverage.clamp_min(1e-6)
        observed_target = F.avg_pool2d(target * support, kernel) / coverage.clamp_min(1e-6)
        error = (observed_generated - observed_target).square() * coverage
        coarse.append(error.sum((1, 2, 3)) / (3 * coverage.sum((1, 2, 3))).clamp_min(1))
    return mse + .5 * torch.stack(coarse).mean(0), mse, torch.stack(coarse).mean(0)
