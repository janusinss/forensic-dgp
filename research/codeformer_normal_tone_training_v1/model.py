"""One training-objective intervention; frozen CodeFormer architecture/inference.

The inherited seven losses are freshly evaluated for every observation. Only
normal TRAIN slots replace coarse RGB L1 by a differentiated common luminance
offset removal on the same unclipped native512 RGB16x16 grid. No cached
gradients, inference correction, deployment role input or new architecture.
"""
from research.codeformer_adaptation_v2 import model as _base


LUMINANCE_WEIGHTS = (.2126, .7152, .0722)


def __getattr__(name):
    return getattr(_base, name)


def centered_error_l1(error):
    """Shared training/prerequisite math; remove one common luminance offset."""
    torch = _base.torch
    _base.require(error.ndim == 4 and error.shape[1] == 3 and len(error) > 0,
                  'Centered coarse error must be nonempty NCHW RGB')
    weights = error.new_tensor(LUMINANCE_WEIGHTS).view(1, 3, 1, 1)
    mean_luminance = (error * weights).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    _base.require(mean_luminance.shape == (len(error), 1, 1, 1),
                  'Normal coarse luminance mean must be per observation')
    _base.require(mean_luminance.requires_grad, 'Normal coarse luminance mean must remain differentiable')
    centered = (error - mean_luminance).abs().mean()
    _base.require(torch.isfinite(centered).item(), 'Nonfinite centered normal coarse term')
    return centered


def centered_coarse_l1(output, clean512):
    functional = _base.F
    rgb = (output.float() + 1) / 2
    error = functional.avg_pool2d(rgb, 32, 32) - functional.avg_pool2d(clean512.float(), 32, 32)
    _base.require(error.shape[1:] == (3, 16, 16), 'Training coarse grid must remain16x16')
    return centered_error_l1(error)


def coarse_supervision_metadata(output, clean512):
    """Detached reduction-only reporting; actual training mean stays in graph."""
    torch, functional = _base.torch, _base.F
    with torch.no_grad(), torch.autocast(device_type=output.device.type, enabled=False):
        rgb = (output.detach().float() + 1) / 2
        predicted_grid = functional.avg_pool2d(rgb, 32, 32)
        clean_grid = functional.avg_pool2d(clean512.detach().float(), 32, 32)
        error = predicted_grid - clean_grid
        weights = error.new_tensor(LUMINANCE_WEIGHTS).view(1, 3, 1, 1)
        mu = (error * weights).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
        ordinary = functional.l1_loss(predicted_grid, clean_grid)
        centered = (error - mu).abs().mean()
    return {'ordinary_coarse_l1': float(ordinary.cpu()), 'centered_coarse_l1': float(centered.cpu()),
            'mean_coarse_luminance_error': float(mu.cpu()), 'grid': [16, 16],
            'domain': 'unclipped native512 RGB', 'detached_reporting_only': True,
            'actual_training_mean_is_differentiated': True}


def forward_training_terms(net, teacher, perceptual, input256, clean512,
                           image_eligible=True, codes=None, behavior=None, *, normal_observation=False):
    _base.require(type(normal_observation) is bool, 'Explicit normal TRAIN observation flag required')
    total, terms, output = _base.forward_training_terms(net, teacher, perceptual,
        input256, clean512, image_eligible=image_eligible, codes=codes, behavior=behavior)
    if not normal_observation:
        return total, terms, output
    # Preserve the inherited insertion/reduction order and every other fresh
    # loss. The unused ordinary coarse value is replaced, never subtracted from
    # an already-rounded total or a cached gradient aggregate.
    with _base.torch.autocast(device_type=output.device.type, enabled=False):
        terms['coarse_rgb_l1'] = centered_coarse_l1(output, clean512)
        total = sum(_base.LOSS_WEIGHTS[name] * value for name, value in terms.items())
    _base.require(_base.torch.isfinite(total).item(), 'Nonfinite normal-tone training objective')
    return total, terms, output
