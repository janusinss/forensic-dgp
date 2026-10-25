"""One photographic RGB256 objective for both matched routes and all five roles.

The caller supplies NumPy or Torch explicitly. Importing this module loads no
framework and creates no model, target, optimizer, or training operation. Both
entry points return an UNDIVIDED role loss and three unweighted component terms.
The guarded manual worker must divide each role loss by five exactly once.
"""

ROLES = (
    'clear',
    'global_gaussian2p0',
    'local_gaussian2p0_soft',
    'compound_g1p2_lr64_jpeg70',
    'lowlight0p35_noise0p006',
)
ROLE_COUNT = 5
BLOCK_SIZE = 16
RGB_COEFFICIENT = 1.0
ADJACENT_COEFFICIENT = 0.25
COARSE_RGB_COEFFICIENT = 0.25
TERM_NAMES = ('RGB_L1', 'adjacent_residual_L1', 'block16_RGB_mean_residual_L1')


def formula_contract():
    """Describe the common, unscaled criterion without admitting optimization."""
    return {
        'status': 'INERT_UNRUN_MATCHED_PHOTOGRAPHIC_OBJECTIVE',
        'roles': list(ROLES),
        'same_formula_for_both_routes_and_every_role': True,
        'formula': 'RGB_L1 + 0.25 * adjacent_residual_L1 + 0.25 * block16_RGB_mean_residual_L1',
        'RGB_L1': 'mean(abs(prediction - photographic_clean_target)) over all RGB256 elements',
        'adjacent_residual_L1': 'mean absolute horizontal and vertical differences of RGB residual; every in-image edge once',
        'block16_RGB_mean_residual_L1': 'mean(abs(nonoverlapping_16x16_mean_RGB_residual)) over all 16x16 blocks and RGB channels',
        'coefficients': dict(zip(TERM_NAMES, (RGB_COEFFICIENT, ADJACENT_COEFFICIENT, COARSE_RGB_COEFFICIENT))),
        'region': 'Whole 256x256 RGB image for all five roles, including local degradation and clear controls',
        'block_size': BLOCK_SIZE,
        'target': 'Detached photographic clean TRAIN target supplied by caller; no generated teacher target',
        'returned_objective_is_unscaled': True,
        'microbatch_role_count': ROLE_COUNT,
        'microbatch_loss_division': 'External worker divides each returned role objective by five exactly once; no division in objective functions',
        'coarse_RGB_bound': 'block16_RGB_mean_residual_L1 <= RGB_L1 by Jensen; partially redundant low-frequency color emphasis, not independent evidence',
        'separate_luminance_training_penalty': False,
        'gradient_or_model_quality_verified': False,
        'optimization_admitted': False,
    }


def _role(role):
    if type(role) is not str or role not in ROLES:
        raise ValueError('Only the exact five matched photographic TRAIN roles are admitted')


def objective_numpy(prediction, target, role, *, np):
    """Float64 saved-array reference; return (undivided loss, raw components).

    This computes arithmetic only. It does not establish connected gradients or
    useful learning. The photographic target and prediction remain unchanged.
    """
    _role(role)
    if (not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray)
            or prediction.shape != (256, 256, 3) or target.shape != prediction.shape
            or prediction.dtype.kind != 'f' or target.dtype.kind != 'f'
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()
            or prediction.min() < 0 or prediction.max() > 1
            or target.min() < 0 or target.max() > 1):
        raise ValueError('Finite floating photographic HWC RGB256 arrays in [0,1] required')
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    horizontal = residual[:, 1:, :] - residual[:, :-1, :]
    vertical = residual[1:, :, :] - residual[:-1, :, :]
    edges = np.concatenate((horizontal.reshape(-1), vertical.reshape(-1)))
    block_means = residual.reshape(16, BLOCK_SIZE, 16, BLOCK_SIZE, 3).mean(axis=(1, 3))
    terms = {
        'RGB_L1': float(np.abs(residual).mean()),
        'adjacent_residual_L1': float(np.abs(edges).mean()),
        'block16_RGB_mean_residual_L1': float(np.abs(block_means).mean()),
    }
    objective = (RGB_COEFFICIENT * terms['RGB_L1']
                 + ADJACENT_COEFFICIENT * terms['adjacent_residual_L1']
                 + COARSE_RGB_COEFFICIENT * terms['block16_RGB_mean_residual_L1'])
    return float(objective), terms


def connected_objective(prediction, target, role, *, torch):
    """Connected FP32 operations reserved for the guarded manual runtime.

    The caller must supply a connected prediction and an already detached
    photographic target. This source check is not gradient verification. The
    manual worker must check NumPy parity and finite, positive trainable gradient
    norms before admitting any optimizer step, and apply the external /5 once.
    """
    _role(role)
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor)
            or prediction.shape != (1, 3, 256, 256) or target.shape != prediction.shape
            or prediction.dtype != torch.float32 or target.dtype != torch.float32
            or prediction.device != target.device or target.requires_grad or target.grad_fn is not None
            or not prediction.requires_grad or prediction.grad_fn is None
            or not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item()
            or prediction.min().item() < 0 or prediction.max().item() > 1
            or target.min().item() < 0 or target.max().item() > 1):
        raise ValueError('Connected FP32 NCHW RGB256 and detached photographic clean target in [0,1] required')
    residual = prediction - target
    horizontal = residual[:, :, :, 1:] - residual[:, :, :, :-1]
    vertical = residual[:, :, 1:, :] - residual[:, :, :-1, :]
    edges = torch.cat((horizontal.reshape(-1), vertical.reshape(-1)))
    block_means = residual.reshape(1, 3, 16, BLOCK_SIZE, 16, BLOCK_SIZE).mean(dim=(3, 5))
    terms = {
        'RGB_L1': residual.abs().mean(),
        'adjacent_residual_L1': edges.abs().mean(),
        'block16_RGB_mean_residual_L1': block_means.abs().mean(),
    }
    objective = (RGB_COEFFICIENT * terms['RGB_L1']
                 + ADJACENT_COEFFICIENT * terms['adjacent_residual_L1']
                 + COARSE_RGB_COEFFICIENT * terms['block16_RGB_mean_residual_L1'])
    if objective.grad_fn is None or not objective.requires_grad or not torch.isfinite(objective).item():
        raise ValueError('Finite connected undivided matched photographic objective required')
    return objective, terms
