"""Conditioned SFT V2 multi-site distillation, perceptual/SSIM/Laplacian objective & exposure-split gradient router.

Importing this module loads zero neural frameworks (`torch` / `numpy` passed explicitly).
"""
import math

from . import conditioning_v2 as cond

ROLES_V2 = (
    'clear',
    'global_defocus_blur',
    'directional_motion_blur',
    'local_soft_surroundings',
    'compound_cctv_codec',
    'lowlight_sensor_noise',
)
NORMAL_DEGRADED_ROLES = (
    'global_defocus_blur',
    'directional_motion_blur',
    'local_soft_surroundings',
    'compound_cctv_codec',
)
SITES = ('32', '64', '128', '256')
SFT_SITE_WEIGHTS = {
    '32': 0.15,
    '64': 0.25,
    '128': 0.30,
    '256': 0.30,
}

# Phase 2 perceptual + structural objective weights
W_RGB = 1.00
W_SPATIAL_ZM = 0.45
W_SSIM7 = 0.35
W_LAPLACIAN_MS = 0.35
W_R2_PERCEPTUAL = 0.30
W_SFT_RETENTION = 0.20
W_COARSE_NORMAL = 0.08
W_COARSE_DARK = 0.30


def objective_v2_contract():
    return {
        'status': 'CONDITIONED_SFT_V2_TWO_PHASE_PERCEPTUAL_STRUCTURAL_OBJECTIVE',
        'roles_v2': list(ROLES_V2),
        'sft_sites': list(SITES),
        'sft_site_weights': dict(SFT_SITE_WEIGHTS),
        'phase1_formula': 'sum_s w_s * (L1(used_s, target_s) + 0.25*(1 - cos(used_s, target_s)) + 0.30*GradL1(used_s, target_s))',
        'phase2_weights': {
            'W_RGB': W_RGB,
            'W_SPATIAL_ZM': W_SPATIAL_ZM,
            'W_SSIM7': W_SSIM7,
            'W_LAPLACIAN_MS': W_LAPLACIAN_MS,
            'W_R2_PERCEPTUAL': W_R2_PERCEPTUAL,
            'W_SFT_RETENTION': W_SFT_RETENTION,
            'W_COARSE_NORMAL': W_COARSE_NORMAL,
            'W_COARSE_DARK': W_COARSE_DARK,
        },
        'exposure_split_gradient_projection': (
            'Normal-lit degraded roles project negative inner-product components against clear gradient; '
            'lowlight_sensor_noise is exposure-decoupled so dark luminance recovery never clashes with clear tone.'
        ),
    }


def ssim7_loss_torch(pred_nchw, target_nchw, *, torch):
    """Differentiable 7-window RGB SSIM loss: 1 - SSIM7(pred, target)."""
    F = torch.nn.functional
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    k = torch.ones((3, 1, 7, 7), dtype=pred_nchw.dtype, device=pred_nchw.device) / 49.0
    mu1 = F.conv2d(pred_nchw, k, padding=0, groups=3)
    mu2 = F.conv2d(target_nchw, k, padding=0, groups=3)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sig1_sq = (F.conv2d(pred_nchw * pred_nchw, k, padding=0, groups=3) - mu1_sq).clamp_min(0.0)
    sig2_sq = (F.conv2d(target_nchw * target_nchw, k, padding=0, groups=3) - mu2_sq).clamp_min(0.0)
    sig12 = F.conv2d(pred_nchw * target_nchw, k, padding=0, groups=3) - mu12
    ssim_map = ((2.0 * mu12 + c1) * (2.0 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2) + 1e-12)
    return 1.0 - ssim_map.mean()


def multiscale_laplacian_loss_torch(pred_nchw, target_nchw, *, torch):
    """Multi-scale (stride 1, 2, 4) finite-difference gradient + 2nd-order Laplacian L1 loss."""
    res = pred_nchw - target_nchw
    total = 0.0
    weights = ((1, 0.45), (2, 0.35), (4, 0.20))
    for stride, w in weights:
        gx = (res[:, :, :, stride:] - res[:, :, :, :-stride]).abs().mean()
        gy = (res[:, :, stride:, :] - res[:, :, :-stride, :]).abs().mean()
        lx = (res[:, :, :, 2 * stride:] - 2.0 * res[:, :, :, stride:-stride] + res[:, :, :, :-2 * stride]).abs().mean()
        ly = (res[:, :, 2 * stride:, :] - 2.0 * res[:, :, stride:-stride, :] + res[:, :, :-2 * stride, :]).abs().mean()
        total = total + w * (0.5 * (gx + gy) + 0.5 * (lx + ly))
    return total


def sft_pyramid_distillation_loss_torch(pred_feats, clean_feats, deg_feats, g_dark, *, torch):
    """Phase-1 & Phase-2 multi-site Exposure-Matched Clean-SFT distillation loss (L1 + Cosine + GradL1)."""
    F = torch.nn.functional
    cond.need(set(pred_feats) == set(SITES) and set(clean_feats) == set(SITES), 'Exact 4 SFT sites required')
    total = pred_feats['32'].sum() * 0.0
    per_site = {}
    for site in SITES:
        pf = pred_feats[site]
        tf = cond.match_sft_exposure_torch(clean_feats[site], deg_feats[site], g_dark, torch=torch)
        l1 = (pf - tf).abs().mean()
        cos = 1.0 - F.cosine_similarity(pf, tf, dim=1, eps=1e-8).mean()
        gx = ((pf[:, :, :, 1:] - pf[:, :, :, :-1]) - (tf[:, :, :, 1:] - tf[:, :, :, :-1])).abs().mean()
        gy = ((pf[:, :, 1:, :] - pf[:, :, :-1, :]) - (tf[:, :, 1:, :] - tf[:, :, :-1, :])).abs().mean()
        grad_l1 = 0.5 * (gx + gy)
        site_loss = l1 + 0.25 * cos + 0.30 * grad_l1
        total = total + float(SFT_SITE_WEIGHTS[site]) * site_loss
        per_site[site] = float(site_loss.detach().item())
    return total, per_site


def joint_phase2_loss_torch(pred_nchw, target_nchw, *, torch,
                            g_dark: float,
                            r2_perc_loss,
                            sft_ret_loss):
    """Connected Phase-2 joint loss combining spatial RGB, SSIM7, multi-scale Laplacian, R2-perceptual, and SFT retention."""
    F = torch.nn.functional
    res = pred_nchw - target_nchw
    rgb_l1 = res.abs().mean()

    def b16(t):
        b = t.shape[0]
        blocks = t.reshape(b, 3, 16, 16, 16, 16).mean(dim=(3, 5), keepdim=True)
        return blocks.expand(b, 3, 16, 16, 16, 16).reshape(b, 3, 256, 256)

    pred_zm = pred_nchw - b16(pred_nchw)
    targ_zm = target_nchw - b16(target_nchw)
    spatial_zm_l1 = (pred_zm - targ_zm).abs().mean()
    coarse_b16_l1 = F.avg_pool2d(res, kernel_size=16, stride=16).abs().mean()
    ssim_loss = ssim7_loss_torch(pred_nchw, target_nchw, torch=torch)
    lap_loss = multiscale_laplacian_loss_torch(pred_nchw, target_nchw, torch=torch)

    w_coarse = W_COARSE_NORMAL * (1.0 - float(g_dark)) + W_COARSE_DARK * float(g_dark)
    total = (
        W_RGB * rgb_l1
        + W_SPATIAL_ZM * spatial_zm_l1
        + W_SSIM7 * ssim_loss
        + W_LAPLACIAN_MS * lap_loss
        + W_R2_PERCEPTUAL * r2_perc_loss
        + W_SFT_RETENTION * sft_ret_loss
        + w_coarse * coarse_b16_l1
    )
    terms = {
        'RGB_L1': float(rgb_l1.detach().item()),
        'spatial_zero_mean_L1': float(spatial_zm_l1.detach().item()),
        'ssim7_loss': float(ssim_loss.detach().item()),
        'multiscale_laplacian_L1': float(lap_loss.detach().item()),
        'r2_perceptual_L1': float(r2_perc_loss.detach().item()),
        'sft_retention_loss': float(sft_ret_loss.detach().item()),
        'block16_coarse_L1': float(coarse_b16_l1.detach().item()),
    }
    return total, terms


def project_exposure_split_gradients_numpy(role_grads, *, np):
    """Exposure-split role-balanced gradient aggregation in Float64 NumPy.

    1. Normalize each role's full gradient vector to median non-zero role norm so no single
       heavy corruption dominates step direction.
    2. Project each normal-lit degraded role (`global_defocus_blur`, `directional_motion_blur`,
       `local_soft_surroundings`, `compound_cctv_codec`) onto `<g_r[p], g_clear[p]> >= 0`
       per tensor `p`, guaranteeing clear tone & sharpness non-regression.
    3. Keep `lowlight_sensor_noise` decoupled from normal-lit `clear` bias projection
       (only projecting 0-mean spatial weights if desired, never bias/coarse scalars),
       eliminating the V1 dark-luminance inversion!
    """
    cond.need(type(role_grads) is dict and set(role_grads) == set(ROLES_V2), 'Exact 6 V2 role gradient dicts required')
    param_names = sorted(role_grads['clear'].keys())

    # Compute per-role global L2 norm for balanced weighting
    role_norms = {}
    for r in ROLES_V2:
        sq = sum(float(np.sum(role_grads[r][p].astype(np.float64) ** 2)) for p in param_names)
        role_norms[r] = math.sqrt(sq)
    nonzero_norms = [v for v in role_norms.values() if v > 1e-12]
    target_norm = float(np.median(nonzero_norms)) if nonzero_norms else 1.0

    scaled = {}
    for r in ROLES_V2:
        scale = (target_norm / role_norms[r]) if role_norms[r] > 1e-12 else 1.0
        scaled[r] = {p: role_grads[r][p].astype(np.float64) * scale for p in param_names}

    projected = {r: {} for r in ROLES_V2}
    conflict_count = 0
    total_pairs = 0
    for p in param_names:
        gc = scaled['clear'][p]
        gc_norm_sq = float(np.sum(gc * gc))
        projected['clear'][p] = gc
        for r in NORMAL_DEGRADED_ROLES:
            total_pairs += 1
            gr = scaled[r][p]
            dot = float(np.sum(gr * gc))
            if dot < 0.0 and gc_norm_sq > 1e-20:
                gr = gr - (dot / (gc_norm_sq + 1e-12)) * gc
                conflict_count += 1
            projected[r][p] = gr
        # Lowlight role is NOT projected onto normal-lit clear biases, preventing coarse dark inversion
        projected['lowlight_sensor_noise'][p] = scaled['lowlight_sensor_noise'][p]

    active_divisor = float(max(1, len(nonzero_norms)))
    combined = {}
    for p in param_names:
        acc = sum(projected[r][p] for r in ROLES_V2) / active_divisor
        combined[p] = acc.astype(np.float32)

    return combined, {
        'role_norms_raw': role_norms,
        'balanced_target_norm': target_norm,
        'normal_projected_conflicts': conflict_count,
        'normal_total_tensor_pairs': total_pairs,
        'lowlight_decoupled_from_clear_bias': True,
    }
