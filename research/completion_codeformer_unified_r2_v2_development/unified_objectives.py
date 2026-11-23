"""Unified R2 v2 training objectives for CodeFormer face mask completion.
No top-level Torch or scientific imports so transfer verification remains import-free.
"""
from __future__ import annotations


def need(ok: bool, label: str) -> None:
    if not ok:
        raise ValueError(label)


def check_case_shapes(raw512, source256, mask256, clean256=None, clean512=None) -> None:
    need(tuple(raw512.shape) == (1, 3, 512, 512), 'Expected raw512 shape (1,3,512,512)')
    need(tuple(source256.shape) == (1, 3, 256, 256), 'Expected source256 shape (1,3,256,256)')
    need(tuple(mask256.shape) == (1, 1, 256, 256), 'Expected mask256 shape (1,1,256,256)')
    if clean256 is not None:
        need(tuple(clean256.shape) == (1, 3, 256, 256), 'Expected clean256 shape (1,3,256,256)')
    if clean512 is not None:
        need(tuple(clean512.shape) == (1, 3, 512, 512), 'Expected clean512 shape (1,3,512,512)')


def delivered_float(raw512, source256, mask256, *, interpolate):
    """Exact differentiable delivered-256 float composition matching the app contract."""
    check_case_shapes(raw512, source256, mask256)
    result = source256.clone()
    active = mask256.sum((1, 2, 3)) > 0
    if not bool(active.any().item()):
        return result
    need(bool(active.all().item()), 'Training/evaluation single-case active mask expected')
    generated512 = ((raw512 + 1.0) / 2.0).clamp(0.0, 1.0)
    generated256 = interpolate(
        generated512,
        size=source256.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )
    result[active] = source256[active] * (1.0 - mask256[active]) + generated256 * mask256[active]
    return result


def inner_boundary(mask, *, F):
    """7x7 morphological inner boundary band inside the mask."""
    return (mask - (1.0 - F.max_pool2d(1.0 - mask, 7, 1, 3))).clamp(0.0, 1.0)


def spatial_support_weights(
    mask256,
    *,
    F,
    inner_ring_weight: float = 1.0,
    outer_ring_weight: float = 0.50,
    core_weight: float = 0.10,
    visible_weight: float = 0.15,
):
    """Compute normalized 16x16 token weights decomposing inner ring, deep core, outer ring, and visible."""
    need(tuple(mask256.shape) == (1, 1, 256, 256), 'Expected mask256 shape (1,1,256,256)')
    m16 = F.max_pool2d(mask256, kernel_size=16, stride=16)
    need(tuple(m16.shape) == (1, 1, 16, 16), 'Expected m16 shape (1,1,16,16)')
    eroded16 = 1.0 - F.max_pool2d(1.0 - m16, kernel_size=3, stride=1, padding=1)
    inner_ring16 = (m16 - eroded16).clamp(0.0, 1.0)
    core16 = eroded16.clamp(0.0, 1.0)
    dilated16 = F.max_pool2d(m16, kernel_size=3, stride=1, padding=1)
    outer_ring16 = (dilated16 - m16).clamp(0.0, 1.0)
    v16 = (1.0 - dilated16).clamp(0.0, 1.0)
    w16_raw = (
        inner_ring_weight * inner_ring16
        + core_weight * core16
        + outer_ring_weight * outer_ring16
        + visible_weight * v16
    )
    mean_weight = w16_raw.mean().clamp_min(1e-8)
    w16_norm = w16_raw / mean_weight
    return w16_norm, m16, outer_ring16, v16


def spatial_code_latent_loss(
    logits,
    lq_feat,
    target_codes,
    target_latent,
    mask256,
    *,
    role_weight: float,
    latent_mse_weight: float = 0.10,
    F,
):
    """Ring-core support-weighted 16x16 spatial code Cross-Entropy + damped lq_feat Latent MSE."""
    need(tuple(logits.shape) == (1, 256, 512), 'Expected logits shape (1,256,512)')
    need(tuple(lq_feat.shape) == (1, 256, 16, 16), 'Expected lq_feat shape (1,256,16,16)')
    need(tuple(target_codes.shape) == (1, 256), 'Expected target_codes shape (1,256)')
    need(tuple(target_latent.shape) == (1, 256, 16, 16), 'Expected target_latent shape (1,256,16,16)')
    need(role_weight > 0.0 and latent_mse_weight >= 0.0, 'Valid weights required')
    w16_norm, m16, r16, _ = spatial_support_weights(mask256, F=F)
    per_token_ce = F.cross_entropy(
        logits.reshape(-1, 512),
        target_codes.reshape(-1),
        reduction='none',
    ).reshape(1, 1, 16, 16)
    weighted_ce = (per_token_ce * w16_norm).mean()
    mean_all_ce = per_token_ce.mean()
    hole_tokens = float(m16.sum().item())
    need(hole_tokens > 0.0, 'Non-empty 16x16 hole token support required')
    hole_ce = (per_token_ce * m16).sum() / m16.sum().clamp_min(1.0)
    per_token_mse = (lq_feat - target_latent).square().mean(dim=1, keepdim=True)
    weighted_latent_mse = (per_token_mse * w16_norm).mean()
    mean_all_latent_mse = per_token_mse.mean()
    hole_latent_mse = (per_token_mse * m16).sum() / m16.sum().clamp_min(1.0)
    combined = role_weight * (weighted_ce + latent_mse_weight * weighted_latent_mse)
    diagnostics = {
        'role_weight': float(role_weight),
        'latent_mse_weight': float(latent_mse_weight),
        'weighted_CE': float(weighted_ce.detach().item()),
        'mean_all_256_CE': float(mean_all_ce.detach().item()),
        'hole_16x16_CE': float(hole_ce.detach().item()),
        'weighted_latent_MSE': float(weighted_latent_mse.detach().item()),
        'mean_all_256_latent_MSE': float(mean_all_latent_mse.detach().item()),
        'hole_16x16_latent_MSE': float(hole_latent_mse.detach().item()),
        'hole_16x16_tokens': int(round(hole_tokens)),
        'ring_16x16_tokens': int(round(float(r16.sum().item()))),
    }
    return combined, diagnostics, w16_norm


def interchannel_chroma_loss(pred, target, mask):
    """Differentiable pairwise inter-channel difference consistency L1 inside mask."""
    need(pred.shape == target.shape and pred.shape[1] == 3, 'Expected 3-channel RGB tensors')
    need(mask.shape[1] == 1 and mask.shape[-2:] == pred.shape[-2:], 'Mask spatial mismatch')
    support = mask.sum()
    need(float(support.item()) > 0.0, 'Empty mask support in chroma loss')
    drg = ((pred[:, 0:1] - pred[:, 1:2]) - (target[:, 0:1] - target[:, 1:2])).abs()
    dgb = ((pred[:, 1:2] - pred[:, 2:3]) - (target[:, 1:2] - target[:, 2:3])).abs()
    dbr = ((pred[:, 2:3] - pred[:, 0:1]) - (target[:, 2:3] - target[:, 0:1])).abs()
    pairwise = (drg + dgb + dbr) / 3.0
    return (pairwise * mask).sum() / support


def spatial_gradient_l1(pred256, target256, mask256, *, F):
    """Finite-difference spatial gradient L1 across the hole and 1-pixel dilated seam."""
    support = F.max_pool2d(mask256, kernel_size=3, stride=1, padding=1)
    mx = (support[:, :, :, 1:] + support[:, :, :, :-1]).clamp(0.0, 1.0)
    my = (support[:, :, 1:, :] + support[:, :, :-1, :]).clamp(0.0, 1.0)
    need(float(mx.sum().item()) > 0.0 and float(my.sum().item()) > 0.0, 'Empty gradient support')
    gx_diff = (pred256[:, :, :, 1:] - pred256[:, :, :, :-1]) - (
        target256[:, :, :, 1:] - target256[:, :, :, :-1]
    )
    gy_diff = (pred256[:, :, 1:, :] - pred256[:, :, :-1, :]) - (
        target256[:, :, 1:, :] - target256[:, :, :-1, :]
    )
    grad_x = (gx_diff.abs() * mx).sum() / (mx.sum() * 3.0)
    grad_y = (gy_diff.abs() * my).sum() / (my.sum() * 3.0)
    return 0.5 * (grad_x + grad_y)


def unified_fusion_loss(
    raw512,
    source256,
    mask256,
    mask512,
    clean256,
    clean512,
    *,
    role_weight: float,
    style: str,
    F,
):
    """Dual-resolution (256 delivered + 512 raw) hole/core/seam/gradient + inter-channel chroma loss."""
    check_case_shapes(raw512, source256, mask256, clean256, clean512)
    need(tuple(mask512.shape) == (1, 1, 512, 512), 'Expected mask512 shape (1,1,512,512)')
    need(role_weight > 0.0, 'Positive role_weight required')
    need(float(mask256.sum().item()) > 0.0 and float(mask512.sum().item()) > 0.0, 'Empty mask support')

    out256 = delivered_float(raw512, source256, mask256, interpolate=F.interpolate)
    pred512 = ((raw512 + 1.0) / 2.0).clamp(0.0, 1.0)
    b256 = inner_boundary(mask256, F=F)
    core256 = (mask256 - b256).clamp_min(0.0)
    b512 = inner_boundary(mask512, F=F)
    need(float(b256.sum().item()) > 0.0 and float(b512.sum().item()) > 0.0, 'Empty boundary support')

    diff256 = (out256 - clean256).abs()
    diff512 = (pred512 - clean512).abs()
    hole256_l1 = (diff256 * mask256).sum() / (mask256.sum() * 3.0)
    core256_l1 = (
        (diff256 * core256).sum() / (core256.sum() * 3.0)
        if float(core256.sum().item()) > 0.0
        else hole256_l1
    )
    seam256_l1 = (diff256 * b256).sum() / (b256.sum() * 3.0)
    hole512_l1 = (diff512 * mask512).sum() / (mask512.sum() * 3.0)
    seam512_l1 = (diff512 * b512).sum() / (b512.sum() * 3.0)

    grad256_l1 = spatial_gradient_l1(out256, clean256, mask256, F=F)
    chroma256_l1 = interchannel_chroma_loss(out256, clean256, mask256)
    chroma512_l1 = interchannel_chroma_loss(pred512, clean512, mask512)
    chroma_weight = 0.70 if style == 'GRAY' else 0.35
    chroma_combined = 0.5 * chroma256_l1 + 0.5 * chroma512_l1

    pixel_objective = (
        1.50 * hole256_l1
        + 0.75 * core256_l1
        + 0.35 * seam256_l1
        + 0.25 * grad256_l1
        + 0.35 * hole512_l1
        + 0.20 * seam512_l1
        + chroma_weight * chroma_combined
    )
    weighted_total = role_weight * pixel_objective

    unclamped512 = (raw512.detach() + 1.0) / 2.0
    outside_clip = (unclamped512 < 0.0) | (unclamped512 > 1.0)
    mask512_3 = mask512.detach().expand_as(unclamped512) > 0.5

    diagnostics = {
        'role_weight': float(role_weight),
        'style': style,
        'chroma_weight': float(chroma_weight),
        'delivered256_hole_L1': float(hole256_l1.detach().item()),
        'delivered256_core_L1': float(core256_l1.detach().item()),
        'delivered256_seam_L1': float(seam256_l1.detach().item()),
        'raw512_hole_L1': float(hole512_l1.detach().item()),
        'raw512_seam_L1': float(seam512_l1.detach().item()),
        'delivered256_grad_L1': float(grad256_l1.detach().item()),
        'delivered256_chroma_L1': float(chroma256_l1.detach().item()),
        'raw512_chroma_L1': float(chroma512_l1.detach().item()),
        'pixel_objective_unweighted': float(pixel_objective.detach().item()),
        'weighted_pixel_objective': float(weighted_total.detach().item()),
        'raw512_clipped_fraction_all': float(outside_clip.float().mean().item()),
        'raw512_clipped_fraction_mask': float(outside_clip[mask512_3].float().mean().item()),
    }
    return weighted_total, out256, diagnostics


def anchor_trust_penalty(parameters, anchors, *, coef: float = 1e-4):
    """Differentiable L2 trust-region penalty anchoring trainable parameters to R1."""
    need(len(parameters) == len(anchors) and len(parameters) > 0, 'Parameter/anchor count mismatch')
    need(coef >= 0.0, 'Non-negative trust coefficient required')
    total = None
    for param, anchor in zip(parameters, anchors):
        term = (param - anchor).square().sum()
        total = term if total is None else (total + term)
    return 0.5 * coef * total
