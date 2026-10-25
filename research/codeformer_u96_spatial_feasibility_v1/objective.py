"""Inert photographic RGB/adjacent components for a future manual diagnostic.

Importing this module performs no framework/model/gradient/optimizer work.
The caller must supply NumPy or Torch explicitly. No generated teacher is an
optimization target, and no pixel from DEV/native/wild/FINAL is admitted here.
"""
import math


ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
COMPONENTS = ('rgb', 'adjacent')
TASK_NAMES = tuple(role + '/' + name for role in ROLES for name in COMPONENTS)
LOCAL_ROI = (64, 64, 192, 192)


def split_scalar_terms(role, terms):
    """Keep RGB and adjacent separate; local regions each receive half weight."""
    expected = {'image', 'adjacent'} if role != ROLES[0] else {
        'image_inside', 'image_outside', 'adjacent_inside', 'adjacent_outside'}
    if (role not in ROLES or type(terms) is not dict or set(terms) != expected or
            any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
                for value in terms.values())):
        raise ValueError('Exact finite nonnegative photographic terms and declared role required')
    if role == ROLES[0]:
        return {'rgb': .5 * (terms['image_inside'] + terms['image_outside']),
                'adjacent': .5 * (terms['adjacent_inside'] + terms['adjacent_outside'])}
    return {'rgb': terms['image'], 'adjacent': terms['adjacent']}


def numpy_components(prediction, target, role, *, np):
    """Float64 saved-array reference with no gradients or model calls."""
    if (role not in ROLES or not isinstance(prediction, np.ndarray) or
            not isinstance(target, np.ndarray) or prediction.shape != (256, 256, 3) or
            target.shape != prediction.shape or prediction.dtype.kind != 'f' or
            target.dtype.kind != 'f' or not np.isfinite(prediction).all() or
            not np.isfinite(target).all() or prediction.min() < 0 or prediction.max() > 1 or
            target.min() < 0 or target.max() > 1):
        raise ValueError('Matching finite photographic RGB256 arrays in [0,1] required')
    residual = prediction.astype(np.float64) - target.astype(np.float64)

    def regional(mask):
        horizontal = np.abs(residual[:, 1:] - residual[:, :-1])[mask[:, 1:] & mask[:, :-1]]
        vertical = np.abs(residual[1:] - residual[:-1])[mask[1:] & mask[:-1]]
        return float(np.abs(residual)[mask].mean()), float(np.concatenate(
            (horizontal.reshape(-1), vertical.reshape(-1))).mean())

    inside = np.ones((256, 256), dtype=bool)
    if role == ROLES[0]:
        inside[:] = False
        inside[64:192, 64:192] = True
        di, ei = regional(inside)
        do, eo = regional(~inside)
        terms = {'image_inside': di, 'image_outside': do,
                 'adjacent_inside': ei, 'adjacent_outside': eo}
    else:
        image, adjacent = regional(inside)
        terms = {'image': image, 'adjacent': adjacent}
    return split_scalar_terms(role, terms), terms


def connected_components(prediction, target, role, *, torch):
    """Two connected scalars from one future graph, with no combined weighted loss.

The future worker must reserve two autograd.grad queries per case. A .25 edge
weight would cancel during unit-gradient normalization, so none is concealed
in either component. The actual delivered RAW/PNG checks remain separate.
"""
    if (role not in ROLES or not isinstance(prediction, torch.Tensor) or
            not isinstance(target, torch.Tensor) or prediction.shape != (1, 3, 256, 256) or
            target.shape != prediction.shape or prediction.dtype != torch.float32 or
            target.dtype != torch.float32 or prediction.device != target.device or
            not prediction.requires_grad or prediction.grad_fn is None or target.requires_grad or
            not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item() or
            prediction.min().item() < 0 or prediction.max().item() > 1 or
            target.min().item() < 0 or target.max().item() > 1):
        raise ValueError('Connected FP32 delivered RGB and detached photographic target required')
    residual = prediction - target

    def regional(mask):
        image = residual.abs().masked_select(mask.expand_as(residual)).mean()
        hmask = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        vmask = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        horizontal = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().masked_select(
            hmask.expand(1, 3, 256, 255))
        vertical = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().masked_select(
            vmask.expand(1, 3, 255, 256))
        return image, torch.cat((horizontal, vertical)).mean()

    mask = torch.ones((1, 1, 256, 256), dtype=torch.bool, device=prediction.device)
    if role == ROLES[0]:
        mask[:] = False
        mask[:, :, 64:192, 64:192] = True
        di, ei = regional(mask)
        do, eo = regional(~mask)
        components = {'rgb': .5 * (di + do), 'adjacent': .5 * (ei + eo)}
    else:
        image, adjacent = regional(mask)
        components = {'rgb': image, 'adjacent': adjacent}
    if any(not scalar.requires_grad or scalar.grad_fn is None or
           not torch.isfinite(scalar).item() for scalar in components.values()):
        raise ValueError('Both finite connected component scalars required')
    return components
