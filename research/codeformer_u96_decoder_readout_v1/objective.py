"""Prospective clean-target readout objective; imports no model or Torch.

No optimizer, data loading, model construction or runtime execution is provided.
The explicit tensor helper requires a future caller-supplied Torch module.
"""
import math


ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
LOCAL_ROI = (64, 64, 192, 192)


def objective_contract():
    return {
        'status': 'PROSPECTIVE_UNRUN_NOT_A_SEALED_TRAINING_PACKET',
        'target': 'Same-photograph aligned photographic clean256 for every role.',
        'roles': list(ROLES),
        'local_roi_half_open': list(LOCAL_ROI),
        'formula_global_clear_dark': 'RGB_L1 + 0.25 * adjacent_residual_L1',
        'formula_local': '0.5*(RGB_L1_inside+RGB_L1_outside) + 0.125*(adjacent_inside+adjacent_outside)',
        'adjacent_definition': 'Mean absolute horizontal and vertical prediction-minus-target residual differences, counting only edges wholly inside each region.',
        'U96_behavior_target_used_for_optimization': False,
        'clean_target_replaced_by_generated_teacher': False,
        'gradient_graph': 'Connected native512 RGB clamp then bilinear256; no PNG flooring in the optimizing graph.',
        'quality_or_preservation_thresholds_changed': False,
        'criterion_weights_proven_optimal': False,
    }


def combine_terms(role, terms):
    """Standard-library scalar reference for loss assembly, not a model call."""
    if role not in ROLES or type(terms) is not dict:
        raise ValueError('An explicit declared role and exact scalar terms are required.')
    expected = {'image', 'adjacent'} if role != ROLES[0] else {
        'image_inside', 'image_outside', 'adjacent_inside', 'adjacent_outside'}
    if set(terms) != expected or any(type(v) not in (int, float) or
                                   not math.isfinite(v) or v < 0 for v in terms.values()):
        raise ValueError('Exact nonnegative finite objective terms are required.')
    if role == ROLES[0]:
        return .5 * (terms['image_inside'] + terms['image_outside']) + .125 * (
            terms['adjacent_inside'] + terms['adjacent_outside'])
    return terms['image'] + .25 * terms['adjacent']


def numpy_clean_terms(prediction, target, role, *, np):
    """Independent saved-array scalar reference; not a gradient implementation."""
    if (role not in ROLES or not isinstance(prediction, np.ndarray) or
            not isinstance(target, np.ndarray) or prediction.shape != (256, 256, 3) or
            target.shape != prediction.shape or prediction.dtype.kind != 'f' or
            target.dtype.kind != 'f' or not np.isfinite(prediction).all() or
            not np.isfinite(target).all() or prediction.min() < 0 or prediction.max() > 1 or
            target.min() < 0 or target.max() > 1):
        raise ValueError('Matching finite RGB256 arrays in [0,1] and a declared role required.')
    residual = prediction.astype(np.float64) - target.astype(np.float64)

    def regional(mask):
        hmask = mask[:, 1:] & mask[:, :-1]
        vmask = mask[1:, :] & mask[:-1, :]
        horizontal = np.abs(residual[:, 1:] - residual[:, :-1])[hmask]
        vertical = np.abs(residual[1:, :] - residual[:-1, :])[vmask]
        return float(np.abs(residual)[mask].mean()), float(
            np.concatenate((horizontal.reshape(-1), vertical.reshape(-1))).mean())

    mask = np.ones((256, 256), dtype=bool)
    if role == ROLES[0]:
        mask[:] = False
        mask[64:192, 64:192] = True
        di, ei = regional(mask)
        do, eo = regional(~mask)
        return {'image_inside': di, 'image_outside': do,
                'adjacent_inside': ei, 'adjacent_outside': eo}
    image, adjacent = regional(mask)
    return {'image': image, 'adjacent': adjacent}


def connected_clean_objective(prediction, target, role, *, torch):
    """Future manual-runtime tensor objective; never imports or creates Torch."""
    if role not in ROLES:
        raise ValueError('Unknown role; no inferred corruption label is allowed.')
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor) or
            prediction.shape != (1, 3, 256, 256) or target.shape != prediction.shape or
            prediction.dtype != torch.float32 or target.dtype != torch.float32 or
            prediction.device != target.device or not prediction.requires_grad or
            prediction.grad_fn is None or target.requires_grad or
            not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item() or
            prediction.min().item() < 0 or prediction.max().item() > 1 or
            target.min().item() < 0 or target.max().item() > 1):
        raise ValueError('Connected finite FP32 delivered RGB and detached photographic target required.')
    residual = prediction - target

    def regional(region):
        values = residual.abs().masked_select(region.expand_as(residual))
        horizontal = region[:, :, :, 1:] & region[:, :, :, :-1]
        vertical = region[:, :, 1:, :] & region[:, :, :-1, :]
        h = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().masked_select(
            horizontal.expand(1, 3, 256, 255))
        v = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().masked_select(
            vertical.expand(1, 3, 255, 256))
        if not values.numel() or not h.numel() or not v.numel():
            raise ValueError('Every region needs both valid horizontal and vertical edges.')
        return values.mean(), torch.cat((h, v)).mean()

    whole = torch.ones((1, 1, 256, 256), dtype=torch.bool, device=prediction.device)
    if role == ROLES[0]:
        inside = torch.zeros_like(whole)
        inside[:, :, 64:192, 64:192] = True
        di, ei = regional(inside)
        do, eo = regional(~inside)
        terms = {'image_inside': di, 'image_outside': do,
                 'adjacent_inside': ei, 'adjacent_outside': eo}
        loss = .5 * (di + do) + .125 * (ei + eo)
    else:
        image, adjacent = regional(whole)
        terms = {'image': image, 'adjacent': adjacent}
        loss = image + .25 * adjacent
    if not loss.requires_grad or not torch.isfinite(loss).item():
        raise ValueError('Finite connected objective required before any optimizer action.')
    return loss, terms
