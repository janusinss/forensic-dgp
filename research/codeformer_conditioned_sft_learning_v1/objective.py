"""Conditioned SFT V1 photographic & multi-site feature distillation objective.

Preserves the exact three unweighted photographic metric terms (`RGB_L1`,
`adjacent_residual_L1`, `block16_RGB_mean_residual_L1`) and `objective_numpy` /
`connected_objective` for full evaluation-gate compatibility with Matched V1,
while providing:
1. Input-conditioned photographic training objective (`conditioned_objective_numpy`,
   `connected_conditioned_objective`) that decouples 16x16 block color from
   zero-mean spatial detail (`pred - B16(pred)`) and multi-scale finite-difference
   gradients (stride 1 and stride 2), weighting coarse 16x16 color by `g_dark(x)`
   so low-light brightness recovery never shifts normal-lit global/coarse tone.
2. Multi-site Clean-SFT encoder feature distillation (`sft_feature_distillation_numpy`,
   `connected_sft_feature_distillation`) at resolutions `32, 64, 128, 256`
   weighted by `g_deg(x)` against detached clean-target `R2` encoder features.
3. Role-balanced conflict-projected gradient aggregation (`project_role_gradients_numpy`,
   `project_role_gradients_torch`) that removes any component of non-clear role
   gradients with negative inner product against the `clear` role gradient
   before 5-role averaging.
"""
import math

from . import conditioning

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

# Conditioned SFT V1 training weights
SPATIAL_DETAIL_COEFFICIENT = 0.35
STRIDE2_EDGE_COEFFICIENT = 0.15
NORMAL_COARSE_COEFFICIENT = 0.10
DARK_COARSE_COEFFICIENT = 0.35
LAMBDA_SFT = 0.08
SFT_SITES = ('32', '64', '128', '256')
SFT_SITE_WEIGHTS = {
    '32': 0.15,
    '64': 0.25,
    '128': 0.30,
    '256': 0.30,
}
CONDITIONED_TERM_NAMES = (
    'RGB_L1',
    'adjacent_residual_L1',
    'block16_RGB_mean_residual_L1',
    'spatial_zero_mean_L1',
    'stride2_residual_L1',
    'sft_feature_distillation_L1',
)


def formula_contract():
    """Return both the unchanged metric terms and the conditioned training contract."""
    return {
        'status': 'CONDITIONED_SFT_PHOTOGRAPHIC_AND_FEATURE_DISTILLATION_OBJECTIVE_V1',
        'roles': list(ROLES),
        'same_formula_for_every_role': True,
        'metric_formula': 'RGB_L1 + 0.25 * adjacent_residual_L1 + 0.25 * block16_RGB_mean_residual_L1',
        'conditioned_training_formula': (
            'RGB_L1 + 0.35 * spatial_zero_mean_L1 + 0.25 * adjacent_residual_L1 '
            '+ 0.15 * stride2_residual_L1 + (0.10 + 0.25 * g_dark(x)) * block16_RGB_mean_residual_L1 '
            '+ 0.08 * g_deg(x) * sft_feature_distillation_L1'
        ),
        'RGB_L1': 'mean(abs(prediction - photographic_clean_target)) over all RGB256 elements',
        'adjacent_residual_L1': 'mean absolute horizontal and vertical stride-1 differences of RGB residual',
        'block16_RGB_mean_residual_L1': 'mean(abs(nonoverlapping_16x16_mean_RGB_residual)) over all 16x16 blocks and RGB channels',
        'spatial_zero_mean_L1': 'mean(abs((prediction - B16(prediction)) - (target - B16(target))))',
        'stride2_residual_L1': 'mean absolute horizontal and vertical stride-2 differences of RGB residual',
        'sft_feature_distillation_L1': 'weighted sum across sites (32,64,128,256) of mean(abs(sft_feat(x) - stopgrad(sft_feat_R2(target))))',
        'sft_sites': list(SFT_SITES),
        'sft_site_weights': dict(SFT_SITE_WEIGHTS),
        'lambda_sft': LAMBDA_SFT,
        'coefficients': dict(zip(TERM_NAMES, (RGB_COEFFICIENT, ADJACENT_COEFFICIENT, COARSE_RGB_COEFFICIENT))),
        'conditioned_coefficients': {
            'RGB_COEFFICIENT': RGB_COEFFICIENT,
            'SPATIAL_DETAIL_COEFFICIENT': SPATIAL_DETAIL_COEFFICIENT,
            'ADJACENT_COEFFICIENT': ADJACENT_COEFFICIENT,
            'STRIDE2_EDGE_COEFFICIENT': STRIDE2_EDGE_COEFFICIENT,
            'NORMAL_COARSE_COEFFICIENT': NORMAL_COARSE_COEFFICIENT,
            'DARK_COARSE_COEFFICIENT': DARK_COARSE_COEFFICIENT,
            'LAMBDA_SFT': LAMBDA_SFT,
        },
        'role_gradient_conflict_projection': (
            'For each non-clear role r and trainable tensor p, if <g_r[p], g_clear[p]> < 0, '
            'replace g_r[p] with g_r[p] - (<g_r[p], g_clear[p]> / (||g_clear[p]||_2^2 + 1e-12)) * g_clear[p] '
            'before 5-role averaging.'
        ),
        'region': 'Whole 256x256 RGB image for all five roles',
        'block_size': BLOCK_SIZE,
        'returned_objective_is_unscaled': True,
        'microbatch_role_count': ROLE_COUNT,
        'target_or_role_label_used_in_gates': False,
    }


def _role(role):
    if type(role) is not str or role not in ROLES:
        raise ValueError('Only the exact five matched photographic TRAIN roles are admitted')


def objective_numpy(prediction, target, role, *, np):
    """Unchanged 3-term Float64 metric reference for 100% gate compatibility."""
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
    """Unchanged 3-term connected FP32 objective."""
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


def sft_feature_distillation_numpy(pred_features, target_features, *, np):
    """Float64 multi-site SFT feature distillation reference across sites 32/64/128/256."""
    conditioning.need(
        type(pred_features) is dict and type(target_features) is dict
        and set(pred_features) == set(SFT_SITES) and set(target_features) == set(SFT_SITES),
        'Exact 4-site SFT feature dicts (32, 64, 128, 256) required',
    )
    total = 0.0
    per_site = {}
    for site in SFT_SITES:
        pf = pred_features[site]
        tf = target_features[site]
        conditioning.need(
            isinstance(pf, np.ndarray) and isinstance(tf, np.ndarray)
            and pf.shape == tf.shape and pf.ndim == 4
            and pf.dtype.kind == 'f' and tf.dtype.kind == 'f'
            and np.isfinite(pf).all() and np.isfinite(tf).all(),
            'Finite matching 4D SFT feature arrays required at site ' + site,
        )
        site_l1 = float(np.abs(pf.astype(np.float64) - tf.astype(np.float64)).mean())
        per_site[site] = site_l1
        total += SFT_SITE_WEIGHTS[site] * site_l1
    return float(total), per_site


def connected_sft_feature_distillation(pred_features, target_features, *, torch):
    """Connected FP32 multi-site SFT feature distillation across sites 32/64/128/256."""
    conditioning.need(
        type(pred_features) is dict and type(target_features) is dict
        and set(pred_features) == set(SFT_SITES) and set(target_features) == set(SFT_SITES),
        'Exact 4-site SFT feature dicts (32, 64, 128, 256) required',
    )
    total = None
    per_site = {}
    for site in SFT_SITES:
        pf = pred_features[site]
        tf = target_features[site]
        conditioning.need(
            isinstance(pf, torch.Tensor) and isinstance(tf, torch.Tensor)
            and pf.shape == tf.shape and pf.ndim == 4
            and pf.dtype == torch.float32 and tf.dtype == torch.float32
            and not tf.requires_grad and tf.grad_fn is None
            and torch.isfinite(pf).all().item() and torch.isfinite(tf).all().item(),
            'Connected FP32 SFT feature and detached clean target feature required at site ' + site,
        )
        site_l1 = (pf - tf).abs().mean()
        per_site[site] = site_l1
        weighted = float(SFT_SITE_WEIGHTS[site]) * site_l1
        total = weighted if total is None else (total + weighted)
    return total, per_site


def conditioned_objective_numpy(prediction, target, input_rgb, role, *, np,
                                pred_sft_features=None, target_sft_features=None):
    """Float64 reference for the input-conditioned photographic + SFT objective."""
    _role(role)
    _, base_terms = objective_numpy(prediction, target, role, np=np)
    gates = conditioning.compute_input_conditioning_numpy(input_rgb, np=np)
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    coarse_res = conditioning.block16_mean_numpy(residual, np=np)
    spatial_res = residual - coarse_res
    h2 = residual[:, 2:, :] - residual[:, :-2, :]
    v2 = residual[2:, :, :] - residual[:-2, :, :]
    edges2 = np.concatenate((h2.reshape(-1), v2.reshape(-1)))

    sft_l1 = 0.0
    if pred_sft_features is not None or target_sft_features is not None:
        sft_l1, _ = sft_feature_distillation_numpy(
            pred_sft_features, target_sft_features, np=np
        )

    terms = {
        'RGB_L1': base_terms['RGB_L1'],
        'adjacent_residual_L1': base_terms['adjacent_residual_L1'],
        'block16_RGB_mean_residual_L1': base_terms['block16_RGB_mean_residual_L1'],
        'spatial_zero_mean_L1': float(np.abs(spatial_res).mean()),
        'stride2_residual_L1': float(np.abs(edges2).mean()),
        'sft_feature_distillation_L1': float(sft_l1),
    }
    coarse_weight = (
        NORMAL_COARSE_COEFFICIENT
        + (DARK_COARSE_COEFFICIENT - NORMAL_COARSE_COEFFICIENT) * gates['g_dark']
    )
    total = (
        RGB_COEFFICIENT * terms['RGB_L1']
        + SPATIAL_DETAIL_COEFFICIENT * terms['spatial_zero_mean_L1']
        + ADJACENT_COEFFICIENT * terms['adjacent_residual_L1']
        + STRIDE2_EDGE_COEFFICIENT * terms['stride2_residual_L1']
        + coarse_weight * terms['block16_RGB_mean_residual_L1']
        + LAMBDA_SFT * gates['g_deg'] * terms['sft_feature_distillation_L1']
    )
    return float(total), terms, gates


def connected_conditioned_objective(prediction, target, input_nchw, role, *, torch,
                                    pred_sft_features=None, target_sft_features=None):
    """Connected FP32 input-conditioned photographic + SFT distillation objective."""
    _role(role)
    _, base_terms = connected_objective(prediction, target, role, torch=torch)
    gates = conditioning.compute_input_conditioning_torch(input_nchw, torch=torch)
    residual = prediction - target
    coarse_res = conditioning.block16_mean_torch(residual, torch=torch)
    spatial_res = residual - coarse_res
    h2 = residual[:, :, :, 2:] - residual[:, :, :, :-2]
    v2 = residual[:, :, 2:, :] - residual[:, :, :-2, :]
    edges2 = torch.cat((h2.reshape(-1), v2.reshape(-1)))

    if pred_sft_features is not None or target_sft_features is not None:
        sft_l1, _ = connected_sft_feature_distillation(
            pred_sft_features, target_sft_features, torch=torch
        )
    else:
        sft_l1 = torch.zeros((), dtype=torch.float32, device=prediction.device)

    terms = {
        'RGB_L1': base_terms['RGB_L1'],
        'adjacent_residual_L1': base_terms['adjacent_residual_L1'],
        'block16_RGB_mean_residual_L1': base_terms['block16_RGB_mean_residual_L1'],
        'spatial_zero_mean_L1': spatial_res.abs().mean(),
        'stride2_residual_L1': edges2.abs().mean(),
        'sft_feature_distillation_L1': sft_l1,
    }
    coarse_weight = float(
        NORMAL_COARSE_COEFFICIENT
        + (DARK_COARSE_COEFFICIENT - NORMAL_COARSE_COEFFICIENT) * gates['g_dark']
    )
    sft_weight = float(LAMBDA_SFT * gates['g_deg'])
    total = (
        RGB_COEFFICIENT * terms['RGB_L1']
        + SPATIAL_DETAIL_COEFFICIENT * terms['spatial_zero_mean_L1']
        + ADJACENT_COEFFICIENT * terms['adjacent_residual_L1']
        + STRIDE2_EDGE_COEFFICIENT * terms['stride2_residual_L1']
        + coarse_weight * terms['block16_RGB_mean_residual_L1']
        + sft_weight * terms['sft_feature_distillation_L1']
    )
    if total.grad_fn is None or not total.requires_grad or not torch.isfinite(total).item():
        raise ValueError('Finite connected undivided conditioned SFT objective required')
    return total, terms, gates


def project_role_gradients_numpy(role_gradients, *, np, eps=1e-12):
    """Project out conflicting non-clear gradient components against clear role gradient.

    `role_gradients` maps each role in ROLES to a dict `{param_name: np.ndarray}`.
    Returns `(averaged_gradients, diagnostics)`.
    """
    conditioning.need(
        type(role_gradients) is dict and set(role_gradients) == set(ROLES),
        'Complete 5-role gradient dictionary required for conflict projection',
    )
    clear_grads = role_gradients['clear']
    param_names = tuple(clear_grads.keys())
    conditioning.need(len(param_names) > 0, 'Nonempty parameter gradient dict required')
    for role in ROLES:
        conditioning.need(
            type(role_gradients[role]) is dict and tuple(role_gradients[role].keys()) == param_names,
            'Identical ordered parameter keys required across all 5 roles',
        )

    projected = {role: {} for role in ROLES}
    conflict_count = 0
    total_pairs = 0
    for name in param_names:
        g_clear = clear_grads[name].astype(np.float64)
        conditioning.need(np.isfinite(g_clear).all(), 'Finite clear gradient required: ' + name)
        clear_norm_sq = float(np.sum(g_clear * g_clear))
        projected['clear'][name] = g_clear.copy()
        for role in ROLES:
            if role == 'clear':
                continue
            g_r = role_gradients[role][name].astype(np.float64)
            conditioning.need(np.isfinite(g_r).all() and g_r.shape == g_clear.shape,
                              f'Finite matching gradient required for {role}:{name}')
            total_pairs += 1
            dot = float(np.sum(g_r * g_clear))
            if dot < 0.0 and clear_norm_sq > eps:
                conflict_count += 1
                g_r = g_r - (dot / (clear_norm_sq + eps)) * g_clear
            projected[role][name] = g_r

    averaged = {}
    min_dot_after = 0.0
    for name in param_names:
        stack = sum(projected[role][name] for role in ROLES) / float(ROLE_COUNT)
        averaged[name] = stack.astype(np.float32)
        dot_after = float(np.sum(stack * projected['clear'][name]))
        min_dot_after = min(min_dot_after, dot_after)

    return averaged, {
        'total_non_clear_tensor_pairs': total_pairs,
        'projected_conflict_pairs': conflict_count,
        'min_inner_product_with_clear_after_projection': min_dot_after,
    }


def project_role_gradients_torch(role_gradients, *, torch, eps=1e-12):
    """Torch implementation of role-balanced conflict-projected gradient aggregation."""
    conditioning.need(
        type(role_gradients) is dict and set(role_gradients) == set(ROLES),
        'Complete 5-role gradient dictionary required for conflict projection',
    )
    clear_grads = role_gradients['clear']
    param_names = tuple(clear_grads.keys())
    conditioning.need(len(param_names) > 0, 'Nonempty parameter gradient dict required')
    for role in ROLES:
        conditioning.need(
            type(role_gradients[role]) is dict and tuple(role_gradients[role].keys()) == param_names,
            'Identical ordered parameter keys required across all 5 roles',
        )

    averaged = {}
    conflict_count = 0
    total_pairs = 0
    min_dot_after = 0.0
    for name in param_names:
        g_clear = clear_grads[name].detach().to(dtype=torch.float64)
        conditioning.need(torch.isfinite(g_clear).all().item(), 'Finite clear gradient required: ' + name)
        clear_norm_sq = float((g_clear * g_clear).sum().item())
        accum = g_clear.clone()
        for role in ROLES:
            if role == 'clear':
                continue
            g_r = role_gradients[role][name].detach().to(dtype=torch.float64)
            conditioning.need(torch.isfinite(g_r).all().item() and g_r.shape == g_clear.shape,
                              f'Finite matching gradient required for {role}:{name}')
            total_pairs += 1
            dot = float((g_r * g_clear).sum().item())
            if dot < 0.0 and clear_norm_sq > eps:
                conflict_count += 1
                g_r = g_r - (dot / (clear_norm_sq + eps)) * g_clear
            accum = accum + g_r
        avg_tensor = (accum / float(ROLE_COUNT)).to(dtype=torch.float32)
        averaged[name] = avg_tensor
        dot_after = float((accum / float(ROLE_COUNT) * g_clear).sum().item())
        min_dot_after = min(min_dot_after, dot_after)

    return averaged, {
        'total_non_clear_tensor_pairs': total_pairs,
        'projected_conflict_pairs': conflict_count,
        'min_inner_product_with_clear_after_projection': min_dot_after,
    }
