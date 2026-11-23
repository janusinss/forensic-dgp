"""Unified R2 v3 training objectives, dual-192-FIT skip synthesis, casc3 ridge + soft-KL code refinement,
and Stage B Adam moment calibration for CodeFormer completion.
No top-level Torch or scientific imports so transfer verification remains import-free.
"""
from __future__ import annotations
import math


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


def decompose_16x16_regions(mask256, *, F):
    """Decompose 16x16 token grid into (inner_ring16, core16, outer_ring16, visible16)."""
    need(tuple(mask256.shape) == (1, 1, 256, 256), 'Expected mask256 shape (1,1,256,256)')
    m16 = F.max_pool2d(mask256, kernel_size=16, stride=16)
    need(tuple(m16.shape) == (1, 1, 16, 16), 'Expected m16 shape (1,1,16,16)')
    eroded16 = 1.0 - F.max_pool2d(1.0 - m16, kernel_size=3, stride=1, padding=1)
    inner_ring16 = (m16 - eroded16).clamp(0.0, 1.0)
    core16 = eroded16.clamp(0.0, 1.0)
    dilated16 = F.max_pool2d(m16, kernel_size=3, stride=1, padding=1)
    outer_ring16 = (dilated16 - m16).clamp(0.0, 1.0)
    visible16 = (1.0 - dilated16).clamp(0.0, 1.0)
    return inner_ring16, core16, outer_ring16, visible16


def spatial_support_weights_hole_ring(
    mask256,
    *,
    F,
    hole_weight: float = 1.0,
    ring_weight: float = 0.50,
    visible_weight: float = 0.15,
):
    """Compute normalized 16x16 token weights emphasizing hole and 1-cell context ring (Branch 1)."""
    need(tuple(mask256.shape) == (1, 1, 256, 256), 'Expected mask256 shape (1,1,256,256)')
    m16 = F.max_pool2d(mask256, kernel_size=16, stride=16)
    need(tuple(m16.shape) == (1, 1, 16, 16), 'Expected m16 shape (1,1,16,16)')
    dilated16 = F.max_pool2d(m16, kernel_size=3, stride=1, padding=1)
    r16 = (dilated16 - m16).clamp(0.0, 1.0)
    v16 = (1.0 - m16 - r16).clamp(0.0, 1.0)
    w16_raw = hole_weight * m16 + ring_weight * r16 + visible_weight * v16
    mean_weight = w16_raw.mean().clamp_min(1e-8)
    w16_norm = w16_raw / mean_weight
    return w16_norm, m16, r16, v16


def spatial_support_weights(
    mask256,
    *,
    F,
    inner_ring_weight: float = 1.0,
    outer_ring_weight: float = 0.50,
    core_weight: float = 0.10,
    visible_weight: float = 0.15,
):
    """Compute normalized 16x16 token weights decomposing inner ring, core, outer ring, and visible (Branch 2)."""
    inner_ring16, core16, outer_ring16, v16 = decompose_16x16_regions(mask256, F=F)
    m16 = (inner_ring16 + core16).clamp(0.0, 1.0)
    w16_raw = (
        inner_ring_weight * inner_ring16
        + core_weight * core16
        + outer_ring_weight * outer_ring16
        + visible_weight * v16
    )
    mean_weight = w16_raw.mean().clamp_min(1e-8)
    w16_norm = w16_raw / mean_weight
    return w16_norm, m16, outer_ring16, v16


def spatial_code_latent_loss_branch1(
    logits,
    lq_feat,
    target_codes,
    target_latent,
    mask256,
    *,
    role_weight: float,
    latent_mse_weight: float = 1.00,
    F,
):
    """Branch 1 (balanced128) hole+ring 16x16 code Cross-Entropy + 1.00 lq_feat Latent MSE."""
    need(tuple(logits.shape) == (1, 256, 512), 'Expected logits shape (1,256,512)')
    need(tuple(lq_feat.shape) == (1, 256, 16, 16), 'Expected lq_feat shape (1,256,16,16)')
    need(tuple(target_codes.shape) == (1, 256), 'Expected target_codes shape (1,256)')
    need(tuple(target_latent.shape) == (1, 256, 16, 16), 'Expected target_latent shape (1,256,16,16)')
    need(role_weight > 0.0 and latent_mse_weight >= 0.0, 'Valid weights required')
    w16_norm, m16, r16, _ = spatial_support_weights_hole_ring(mask256, F=F)
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
    """Branch 2 (orthogonal_balanced128) ring-core 16x16 code Cross-Entropy + 0.10 lq_feat Latent MSE."""
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


def _v3r2_hybrid_fusion_objective(
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
    w_h512: float,
    w_s512: float,
    w_h256: float,
    w_s256: float,
    w_grad: float,
    w_chr_rgb: float,
    w_chr_gray: float,
    area_tau256: float,
):
    out256 = delivered_float(raw512, source256, mask256, interpolate=F.interpolate)
    pred512_unclamped = (raw512 + 1.0) / 2.0
    pred512_clamped = pred512_unclamped.clamp(0.0, 1.0)

    b256 = inner_boundary(mask256, F=F)
    b512 = inner_boundary(mask512, F=F)
    need(float(b256.sum().item()) > 0.0 and float(b512.sum().item()) > 0.0, 'Empty inner boundary')

    m256_sum = mask256.sum()
    area_scale = float((m256_sum / (m256_sum + area_tau256)).item()) if area_tau256 > 0.0 else 1.0

    diff256 = (out256 - clean256).abs()
    diff512 = (pred512_unclamped - clean512).abs()
    hole256_l1 = (diff256 * mask256).sum() / (m256_sum * 3.0)
    seam256_l1 = (diff256 * b256).sum() / (b256.sum() * 3.0)
    hole512_l1 = (diff512 * mask512).sum() / (mask512.sum() * 3.0)
    seam512_l1 = (diff512 * b512).sum() / (b512.sum() * 3.0)

    grad256_l1 = spatial_gradient_l1(out256, clean256, mask256, F=F)
    chroma256_l1 = interchannel_chroma_loss(out256, clean256, mask256)
    chroma512_l1 = interchannel_chroma_loss(pred512_clamped, clean512, mask512)
    chroma_weight = w_chr_gray if style == 'GRAY' else w_chr_rgb
    chroma_combined = 0.5 * chroma256_l1 + 0.5 * chroma512_l1

    pixel_total = (
        w_h512 * hole512_l1
        + w_s512 * seam512_l1
        + w_h256 * hole256_l1
        + w_s256 * seam256_l1
        + w_grad * grad256_l1
        + chroma_weight * chroma_combined
    )
    weighted_total = (role_weight * area_scale) * pixel_total
    return (
        weighted_total,
        out256,
        {
            'role_weight': float(role_weight),
            'area_scale': float(area_scale),
            'chroma_weight': float(chroma_weight),
            'delivered256_hole_L1': float(hole256_l1.detach().item()),
            'delivered256_seam_L1': float(seam256_l1.detach().item()),
            'raw512_hole_L1': float(hole512_l1.detach().item()),
            'raw512_seam_L1': float(seam512_l1.detach().item()),
            'delivered256_grad_L1': float(grad256_l1.detach().item()),
            'delivered256_chroma_L1': float(chroma256_l1.detach().item()),
            'raw512_chroma_L1': float(chroma512_l1.detach().item()),
            'pixel_objective_unweighted': float(pixel_total.detach().item()),
            'weighted_pixel_objective': float(weighted_total.detach().item()),
        },
    )


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
    area_tau256: float = 300.0,
):
    """V3+R2v1 hybrid dual-resolution (512 native + 256 delivered) hole/seam/grad/chroma fusion loss
    with soft tiny-mask area damping and zero interior core256 gradient spike.
    """
    check_case_shapes(raw512, source256, mask256, clean256, clean512)
    need(tuple(mask512.shape) == (1, 1, 512, 512), 'Expected mask512 shape (1,1,512,512)')
    need(role_weight > 0.0, 'Positive role_weight required')
    need(float(mask256.sum().item()) > 0.0 and float(mask512.sum().item()) > 0.0, 'Empty mask support')

    weighted_total, out256, diagnostics = _v3r2_hybrid_fusion_objective(
        raw512,
        source256,
        mask256,
        mask512,
        clean256,
        clean512,
        role_weight=role_weight,
        style=style,
        F=F,
        w_h512=1.00,
        w_s512=1.00,
        w_h256=0.35,
        w_s256=0.20,
        w_grad=0.10,
        w_chr_rgb=0.10,
        w_chr_gray=0.25,
        area_tau256=area_tau256,
    )
    clipped = (raw512 < -1.0) | (raw512 > 1.0)
    diagnostics['raw512_clipped_fraction_all'] = float(clipped.float().mean().item())
    diagnostics['raw512_clipped_fraction_mask'] = float(
        (clipped.float() * mask512).sum().item() / (mask512.sum().item() * 3.0)
    )
    return weighted_total, out256, diagnostics


def anchor_trust_penalty(parameters, anchor_tensors, *, coef: float):
    """Quadratic L2 trust-region penalty keeping parameters close to R1 anchors."""
    need(len(parameters) == len(anchor_tensors) and len(parameters) > 0, 'Anchor count mismatch')
    need(coef >= 0.0, 'Non-negative trust coef required')
    sq_sum = None
    for param, anchor in zip(parameters, anchor_tensors):
        term = (param - anchor).square().sum()
        sq_sum = term if sq_sum is None else (sq_sum + term)
    return 0.5 * coef * sq_sum


def predict_logits_from_lq(net, lq_feat):
    """Run CodeFormer Transformer predictor head on 16x16 lq_feat without invoking encoder or generator."""
    pos_emb = net.position_emb.unsqueeze(1).repeat(1, lq_feat.shape[0], 1)
    query_emb = net.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
    for layer in net.ft_layers:
        query_emb = layer(query_emb, query_pos=pos_emb)
    return net.idx_pred_layer(query_emb).permute(1, 0, 2)


def encode_from_block15(net, f32):
    x = f32
    for i in range(15, len(net.encoder.blocks)):
        x = net.encoder.blocks[i](x)
    return x


def encode_16_to_penultimate(net, f16_in):
    x = f16_in
    for i in range(16, len(net.encoder.blocks) - 1):
        x = net.encoder.blocks[i](x)
    return x


def decode_batch_from_dec32(net, efd_b, dec32_b):
    """Run CodeFormer decoder tail (blocks 10..15 + fuse_convs_dict 32/64/128) without whole-model hook."""
    fuse_list = [net.fuse_generator_block[f_size] for f_size in net.connect_list]
    g = net.fuse_convs_dict['32'](efd_b['32'], dec32_b, 1.0)
    for i in range(10, len(net.generator.blocks)):
        g = net.generator.blocks[i](g)
        if i in fuse_list:
            f_size = str(g.shape[-1])
            g = net.fuse_convs_dict[f_size](efd_b[f_size], g, 1.0)
    return g


def solve_ridge_delta(X, dY, lam: float, *, torch):
    """Solve regularized ridge delta on CPU float32 for deterministic cross-device parity."""
    need(lam > 0.0, 'Positive ridge regularization required')
    x_cpu = X.detach().to(device='cpu', dtype=torch.float32)
    y_cpu = dY.detach().to(device='cpu', dtype=torch.float32)
    ones = torch.ones((x_cpu.shape[0], 1), dtype=torch.float32)
    x_aug = torch.cat([x_cpu, ones], dim=1)
    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ y_cpu
    reg = float(lam) * torch.eye(xtx.shape[0], dtype=torch.float32)
    theta = torch.linalg.solve(xtx + reg, xty)
    need(bool(torch.isfinite(theta).all().item()), 'Nonfinite ridge solution')
    return theta[:-1, :], theta[-1, :]


def _make_pred_adam_from_r1(pred_params, r1_code_opt_state_cpu, *, torch):
    """Construct Adam over 114 predictor parameters initialized from R1 code_optimizer slots 164..277."""
    need(len(pred_params) == 114, 'Expected 114 predictor parameter tensors')
    opt = torch.optim.Adam(pred_params, lr=1.0e-5, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    for idx, p in enumerate(pred_params):
        src = r1_code_opt_state_cpu['state'][164 + idx]
        opt.state[p] = {
            'step': src['step'].detach().clone().to(device='cpu', dtype=torch.float32),
            'exp_avg': src['exp_avg'].detach().clone().to(device=p.device, dtype=torch.float32),
            'exp_avg_sq': src['exp_avg_sq'].detach().clone().to(device=p.device, dtype=torch.float32),
        }
    return opt


def refine_code_backbone_from_r1(
    net,
    *,
    code_optimizer,
    branch1_skip_state: dict,
    branch2_skip_state: dict,
    r1_state_cpu: dict,
    r1_code_opt_state_cpu: dict,
    r1_f16_in: dict,
    r1_lq_feat: dict,
    r1_fit_probs: dict,
    fit_x512: dict,
    fit_regions: dict,
    fit_by_id: dict,
    teacher_fit: dict,
    schedule: list,
    guard,
    torch,
    F,
    w_branch1_skip: float = 0.75,
    w_branch2_skip: float = 0.75,
) -> dict:
    """Synthesize dual-192-FIT skip encoder (0.75*d_branch1 + 0.75*d_branch2), train pure-from-R1
    predictor specialization (ft5_100), solve 3-stage cascaded closed-form ridge compensation (casc3),
    and run geometry-routed soft-KL predictor alignment on FIT cases only.
    """
    device = net.position_emb.device
    fit_ids = sorted(fit_by_id.keys())
    need(len(fit_ids) == 128 and len(schedule) == 32, 'Expected 128 primary FIT cases and 32 cycles')

    state_keys = list(net.state_dict().keys())
    skip_keys = [k for k in state_keys if k.startswith('encoder.blocks.') and int(k.split('.')[2]) <= 14]
    bot_keys = [k for k in state_keys if k.startswith('encoder.blocks.') and int(k.split('.')[2]) >= 15]
    pred_keys = [
        k for k in state_keys
        if k == 'position_emb' or k.startswith(('feat_emb.', 'ft_layers.', 'idx_pred_layer.'))
    ]
    need(len(skip_keys) == 94 and len(bot_keys) == 70 and len(pred_keys) == 114, 'Code partition count mismatch')
    need(
        set(branch1_skip_state) == set(skip_keys) and set(branch2_skip_state) == set(skip_keys),
        'Skip branch key set mismatch',
    )

    synthesized_skip_state = {}
    for k in skip_keys:
        r1_w = r1_state_cpu[k].to(device=device, dtype=torch.float32)
        b1_w = branch1_skip_state[k].to(device=device, dtype=torch.float32)
        b2_w = branch2_skip_state[k].to(device=device, dtype=torch.float32)
        synthesized_skip_state[k] = r1_w + w_branch1_skip * (b1_w - r1_w) + w_branch2_skip * (b2_w - r1_w)

    pred_params = (
        [net.position_emb]
        + list(net.feat_emb.parameters())
        + list(net.ft_layers.parameters())
        + list(net.idx_pred_layer.parameters())
    )

    # 1. Train ring_ce_core_kl_v2 on r1_lq_feat from pure R1
    net.requires_grad_(False)
    for k in pred_keys:
        net.state_dict()[k].copy_(r1_state_cpu[k].to(device=device, dtype=torch.float32))
    r1_pred_anchors = [p.detach().clone() for p in pred_params]
    for p in pred_params:
        p.requires_grad_(True)
    opt_v2 = _make_pred_adam_from_r1(pred_params, r1_code_opt_state_cpu, torch=torch)
    for cycle, case_ids in enumerate(schedule, 1):
        guard()
        prog = float(cycle - 1) / 31.0
        lr_p = 4.0e-6 + (1.8e-5 - 4.0e-6) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt_v2.param_groups:
            g['lr'] = lr_p
        opt_v2.zero_grad(set_to_none=True)
        for cid in case_ids:
            case = fit_by_id[cid]
            t_c = teacher_fit[case['teacher_key']][0].reshape(256)
            in_r, core_r, out_r, vis_r = [t.reshape(256) for t in fit_regions[cid]]
            lg = predict_logits_from_lq(net, r1_lq_feat[cid]).reshape(256, 512)
            ce = F.cross_entropy(lg, t_c, reduction='none')
            kl_r1 = F.kl_div(F.log_softmax(lg, dim=-1), r1_fit_probs[cid], reduction='none').sum(dim=-1)
            w_ce = 1.0 * in_r + 0.40 * out_r
            w_kl = 2.0 * core_r + 0.80 * vis_r
            tok_loss = (w_ce * ce + w_kl * kl_r1).sum() / (w_ce + w_kl).sum().clamp_min(1e-8)
            rw = 1.10 if case['role'] == 'fit_anchor' else 0.90
            tr = anchor_trust_penalty(pred_params, r1_pred_anchors, coef=5e-5)
            loss_p = (rw * tok_loss + tr) / 4.0
            need(bool(torch.isfinite(loss_p).item()), 'Nonfinite ring_ce_core_kl_v2 loss')
            loss_p.backward()
        torch.nn.utils.clip_grad_norm_(pred_params, 1.0, error_if_nonfinite=True)
        opt_v2.step()
    opt_v2.zero_grad(set_to_none=True)
    v2_pred_st = {k: net.state_dict()[k].detach().clone() for k in pred_keys}

    # 2. Train pce_corekl_a on r1_lq_feat from pure R1
    for k in pred_keys:
        net.state_dict()[k].copy_(r1_state_cpu[k].to(device=device, dtype=torch.float32))
    opt_pca = _make_pred_adam_from_r1(pred_params, r1_code_opt_state_cpu, torch=torch)
    for g in opt_pca.param_groups:
        g['lr'] = 1.0e-5
    for _, case_ids in enumerate(schedule, 1):
        guard()
        opt_pca.zero_grad(set_to_none=True)
        for cid in case_ids:
            case = fit_by_id[cid]
            t_c = teacher_fit[case['teacher_key']][0].reshape(256)
            in_r, core_r, out_r, vis_r = [t.reshape(256) for t in fit_regions[cid]]
            mask_tokens = float((in_r + core_r).sum().item())
            lg = predict_logits_from_lq(net, r1_lq_feat[cid]).reshape(256, 512)
            ce = F.cross_entropy(lg, t_c, reduction='none')
            kl_r1 = F.kl_div(F.log_softmax(lg, dim=-1), r1_fit_probs[cid], reduction='none').sum(dim=-1)
            if mask_tokens > 32.0:
                w_ce = 1.0 * in_r + 1.0 * out_r + 1.0 * vis_r
                w_kl = 3.0 * core_r
            else:
                w_ce = 1.0 * in_r + 1.0 * out_r + 1.0 * core_r + 1.0 * vis_r
                w_kl = torch.zeros_like(core_r)
            tok_loss = (w_ce * ce + w_kl * kl_r1).mean()
            loss_pca = tok_loss / 4.0
            need(bool(torch.isfinite(loss_pca).item()), 'Nonfinite pce_corekl_a loss')
            loss_pca.backward()
        torch.nn.utils.clip_grad_norm_(pred_params, 1.0, error_if_nonfinite=True)
        opt_pca.step()
    opt_pca.zero_grad(set_to_none=True)
    net.requires_grad_(False)
    pca_pred_st = {k: net.state_dict()[k].detach().clone() for k in pred_keys}
    ft5_100 = {k: pca_pred_st[k] if k.startswith('ft_layers.5.') else v2_pred_st[k] for k in pred_keys}

    # 3. Assemble synthesized 192-FIT skip_encoder + R1 bottleneck + ft5_100 and extract fit_f32 + reference logits
    for k in skip_keys:
        net.state_dict()[k].copy_(synthesized_skip_state[k])
    for k in bot_keys:
        net.state_dict()[k].copy_(r1_state_cpu[k].to(device=device, dtype=torch.float32))
    for k in pred_keys:
        net.state_dict()[k].copy_(ft5_100[k])

    fit_f32 = {}
    uncomp_logits = {}
    target_fit_logits = {}
    with torch.no_grad():
        for i in range(0, len(fit_ids), 4):
            guard()
            chunk_ids = fit_ids[i : i + 4]
            h = torch.cat([fit_x512[cid] for cid in chunk_ids], dim=0)
            for bi in range(15):
                h = net.encoder.blocks[bi](h)
            f32_b = h.clone()
            lq_uncomp_b = encode_from_block15(net, f32_b)
            lg_uncomp_b = predict_logits_from_lq(net, lq_uncomp_b).reshape(4, 256, 512)
            r1_lq_b = torch.cat([r1_lq_feat[cid] for cid in chunk_ids], dim=0)
            lg_targ_b = predict_logits_from_lq(net, r1_lq_b).reshape(4, 256, 512)
            for j, cid in enumerate(chunk_ids):
                fit_f32[cid] = f32_b[j : j + 1].clone()
                uncomp_logits[cid] = lg_uncomp_b[j].clone()
                target_fit_logits[cid] = lg_targ_b[j].clone()

    # 4. Endpoint A lower-face reference logits (blocks[-1] ridge lam=300)
    last_idx = len(net.encoder.blocks) - 1
    w24_r1 = net.encoder.blocks[last_idx].weight.data.clone()
    b24_r1 = net.encoder.blocks[last_idx].bias.data.clone()
    h_list, y_list = [], []
    with torch.no_grad():
        for cid in fit_ids:
            f16 = net.encoder.blocks[15](fit_f32[cid])
            h_norm = encode_16_to_penultimate(net, f16)
            patches = F.unfold(h_norm, kernel_size=3, padding=1)[0].T
            curr_lq = net.encoder.blocks[last_idx](h_norm)[0].flatten(1).T
            targ_lq = r1_lq_feat[cid][0].flatten(1).T
            h_list.append(patches)
            y_list.append(targ_lq - curr_lq)
    w_a, b_a = solve_ridge_delta(torch.cat(h_list, dim=0), torch.cat(y_list, dim=0), 300.0, torch=torch)
    net.encoder.blocks[last_idx].weight.data.add_(w_a.T.reshape(256, 512, 3, 3).to(device=device, dtype=torch.float32))
    net.encoder.blocks[last_idx].bias.data.add_(b_a.to(device=device, dtype=torch.float32))
    lower_A_logits = {}
    with torch.no_grad():
        for cid in fit_ids:
            lower_A_logits[cid] = predict_logits_from_lq(net, encode_from_block15(net, fit_f32[cid])).reshape(256, 512).clone()

    # 5. Reset blocks[-1] to R1 and solve 3-stage casc3 closed-form ridge (lam = 10.0, 10.0, 3.0)
    net.encoder.blocks[last_idx].weight.data.copy_(w24_r1)
    net.encoder.blocks[last_idx].bias.data.copy_(b24_r1)

    x15_list, dy15_list = [], []
    with torch.no_grad():
        for cid in fit_ids:
            f32_pad = F.pad(fit_f32[cid], (0, 1, 0, 1), mode='constant', value=0)
            patches15 = F.unfold(f32_pad, kernel_size=3, stride=2)[0].T
            curr15 = net.encoder.blocks[15](fit_f32[cid])[0].flatten(1).T
            targ15 = r1_f16_in[cid][0].flatten(1).T
            x15_list.append(patches15)
            dy15_list.append(targ15 - curr15)
    w15_d, b15_d = solve_ridge_delta(torch.cat(x15_list, dim=0), torch.cat(dy15_list, dim=0), 10.0, torch=torch)
    net.encoder.blocks[15].conv.weight.data.add_(w15_d.T.reshape(256, 256, 3, 3).to(device=device, dtype=torch.float32))
    net.encoder.blocks[15].conv.bias.data.add_(b15_d.to(device=device, dtype=torch.float32))

    x24_list, dy24_list = [], []
    with torch.no_grad():
        for cid in fit_ids:
            f16 = net.encoder.blocks[15](fit_f32[cid])
            h_norm = encode_16_to_penultimate(net, f16)
            patches24 = F.unfold(h_norm, kernel_size=3, padding=1)[0].T
            curr24 = net.encoder.blocks[last_idx](h_norm)[0].flatten(1).T
            targ24 = r1_lq_feat[cid][0].flatten(1).T
            x24_list.append(patches24)
            dy24_list.append(targ24 - curr24)
    w24_d, b24_d = solve_ridge_delta(torch.cat(x24_list, dim=0), torch.cat(dy24_list, dim=0), 10.0, torch=torch)
    net.encoder.blocks[last_idx].weight.data.add_(w24_d.T.reshape(256, 512, 3, 3).to(device=device, dtype=torch.float32))
    net.encoder.blocks[last_idx].bias.data.add_(b24_d.to(device=device, dtype=torch.float32))

    wfe_0 = net.feat_emb.weight.data.clone()
    bfe_0 = net.feat_emb.bias.data.clone()
    xfe_list, dyfe_list = [], []
    with torch.no_grad():
        for cid in fit_ids:
            lq_pred = encode_from_block15(net, fit_f32[cid])[0].flatten(1).T
            lq_targ = r1_lq_feat[cid][0].flatten(1).T
            xfe_list.append(lq_pred)
            dyfe_list.append(F.linear(lq_targ, wfe_0, bfe_0) - F.linear(lq_pred, wfe_0, bfe_0))
    wfe_d, bfe_d = solve_ridge_delta(torch.cat(xfe_list, dim=0), torch.cat(dyfe_list, dim=0), 3.0, torch=torch)
    net.feat_emb.weight.data.add_(wfe_d.T.to(device=device, dtype=torch.float32))
    net.feat_emb.bias.data.add_(bfe_d.to(device=device, dtype=torch.float32))

    casc3_fit_lq = {}
    with torch.no_grad():
        for cid in fit_ids:
            casc3_fit_lq[cid] = encode_from_block15(net, fit_f32[cid]).clone()

    # 6. Step B1: 1-epoch predictor soft-KL alignment on casc3_fit_lq
    target_p1 = {cid: F.softmax(lg, dim=-1) for cid, lg in target_fit_logits.items()}
    target_p2 = {cid: F.softmax(lg / 2.0, dim=-1) for cid, lg in target_fit_logits.items()}
    for p in pred_params:
        p.requires_grad_(True)
    opt_b1 = _make_pred_adam_from_r1(pred_params, r1_code_opt_state_cpu, torch=torch)
    for step_idx, case_ids in enumerate(schedule, 1):
        guard()
        prog = float(step_idx - 1) / 31.0
        lr_b1 = (1.5e-5 * 0.2) + (1.5e-5 * 0.8) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt_b1.param_groups:
            g['lr'] = lr_b1
        opt_b1.zero_grad(set_to_none=True)
        lqs = torch.cat([casc3_fit_lq[cid] for cid in case_ids], dim=0)
        lg_b = predict_logits_from_lq(net, lqs).reshape(4, 256, 512)
        t_lg_b = torch.stack([target_fit_logits[cid] for cid in case_ids], dim=0)
        t_p1_b = torch.stack([target_p1[cid] for cid in case_ids], dim=0)
        t_p2_b = torch.stack([target_p2[cid] for cid in case_ids], dim=0)
        loss_b1 = (
            0.2 * F.mse_loss(lg_b, t_lg_b)
            + 1.0 * F.kl_div(F.log_softmax(lg_b, dim=-1), t_p1_b, reduction='batchmean')
            + 2.0 * F.kl_div(F.log_softmax(lg_b / 2.0, dim=-1), t_p2_b, reduction='batchmean') * 4.0
        )
        need(bool(torch.isfinite(loss_b1).item()), 'Nonfinite B1 alignment loss')
        loss_b1.backward()
        torch.nn.utils.clip_grad_norm_(pred_params, 1.0, error_if_nonfinite=True)
        opt_b1.step()
    opt_b1.zero_grad(set_to_none=True)
    net.requires_grad_(False)
    b1_pred_init = {k: net.state_dict()[k].detach().clone() for k in pred_keys}
    b1_fit_logits = {}
    with torch.no_grad():
        for cid in fit_ids:
            b1_fit_logits[cid] = predict_logits_from_lq(net, casc3_fit_lq[cid]).reshape(256, 512).clone()

    # 7. Geometry-routed soft-KL endpoints p58 and r3 -> linear mode blend 0.45*p58 + 0.55*r3
    routed_endpoints = {}
    for ep_name, w_A, w_U, lr_init, n_ep, lf_weight in [
        ('p58', 0.60, 0.80, 3.0e-5, 3, 3.0),
        ('r3', 1.00, 0.00, 3.5e-5, 3, 2.0),
    ]:
        w_B1 = 1.0 - w_A - w_U
        r_lg, r_p1, r_p2 = {}, {}, {}
        for cid in fit_ids:
            if fit_by_id[cid]['mask_kind'] == 'lower_face':
                lg = w_A * lower_A_logits[cid] + w_U * uncomp_logits[cid] + w_B1 * b1_fit_logits[cid]
            else:
                lg = b1_fit_logits[cid]
            r_lg[cid] = lg
            r_p1[cid] = F.softmax(lg, dim=-1)
            r_p2[cid] = F.softmax(lg / 2.0, dim=-1)

        for k in pred_keys:
            net.state_dict()[k].copy_(b1_pred_init[k])
        for p in pred_params:
            p.requires_grad_(True)
        opt_ep = _make_pred_adam_from_r1(pred_params, r1_code_opt_state_cpu, torch=torch)
        total_steps = len(schedule) * n_ep
        step_i = 0
        for _ in range(n_ep):
            for case_ids in schedule:
                guard()
                step_i += 1
                prog = float(step_i - 1) / max(1, total_steps - 1)
                lr_ep = (lr_init * 0.2) + (lr_init * 0.8) * 0.5 * (1.0 + math.cos(math.pi * prog))
                for g in opt_ep.param_groups:
                    g['lr'] = lr_ep
                opt_ep.zero_grad(set_to_none=True)
                lqs = torch.cat([casc3_fit_lq[cid] for cid in case_ids], dim=0)
                logits_b = predict_logits_from_lq(net, lqs).reshape(4, 256, 512)
                loss_ep = 0.0
                for bi, cid in enumerate(case_ids):
                    w_c = lf_weight if fit_by_id[cid]['mask_kind'] == 'lower_face' else 1.0
                    lg_i = logits_b[bi]
                    l_mse = F.mse_loss(lg_i, r_lg[cid])
                    l_kl1 = F.kl_div(F.log_softmax(lg_i, dim=-1), r_p1[cid], reduction='batchmean')
                    l_kl2 = F.kl_div(F.log_softmax(lg_i / 2.0, dim=-1), r_p2[cid], reduction='batchmean') * 4.0
                    loss_ep = loss_ep + w_c * (0.4 * l_mse + 1.0 * l_kl1 + 2.0 * l_kl2) / 4.0
                need(bool(torch.isfinite(loss_ep).item()), 'Nonfinite routed soft-KL loss')
                loss_ep.backward()
                torch.nn.utils.clip_grad_norm_(pred_params, 1.0, error_if_nonfinite=True)
                opt_ep.step()
        opt_ep.zero_grad(set_to_none=True)
        net.requires_grad_(False)
        routed_endpoints[ep_name] = {k: net.state_dict()[k].detach().clone() for k in pred_keys}

    net.zero_grad(set_to_none=True)
    for k in pred_keys:
        net.state_dict()[k].copy_(0.45 * routed_endpoints['p58'][k] + 0.55 * routed_endpoints['r3'][k])

    # Clear first-moment bias on bottleneck + predictor slots in code_optimizer while preserving step
    bot_and_pred_params = list(net.encoder.blocks[15:].parameters()) + pred_params
    for p in bot_and_pred_params:
        if p in code_optimizer.state:
            code_optimizer.state[p]['exp_avg'].zero_()

    skip_l2 = math.sqrt(
        sum(
            float((net.state_dict()[k].detach().cpu() - r1_state_cpu[k]).double().square().sum().item())
            for k in skip_keys
        )
    )
    bot_l2 = math.sqrt(
        sum(
            float((net.state_dict()[k].detach().cpu() - r1_state_cpu[k]).double().square().sum().item())
            for k in bot_keys
        )
    )
    pred_l2 = math.sqrt(
        sum(
            float((net.state_dict()[k].detach().cpu() - r1_state_cpu[k]).double().square().sum().item())
            for k in pred_keys
        )
    )
    return {
        'complete': True,
        'fit_only': True,
        'monitor_challenge_accessed': False,
        'skip_branch_weights': {'branch1': w_branch1_skip, 'branch2': w_branch2_skip},
        'skip_encoder_L2_from_R1': skip_l2,
        'casc3_bottleneck_L2_from_R1': bot_l2,
        'aligned_predictor_L2_from_R1': pred_l2,
        'casc3_lambdas': [10.0, 10.0, 3.0],
        'routed_blend': {'p58': 0.45, 'r3': 0.55},
    }


def calibrate_fusion_moments_from_r1(
    net,
    *,
    fusion_optimizer,
    r1_state_cpu: dict,
    fit_x512: dict,
    fit_io: dict,
    fit_by_id: dict,
    schedule: list,
    guard,
    torch,
    F,
) -> dict:
    """Calibrate fusion_optimizer first and second Adam moments (step 32 -> 96) on the dual-resolution
    V3+R2v1 hybrid objective using frozen Stage A FIT features, then reset fuse_convs_dict weights to R1
    before the 32 Stage B optimization cycles (step 96 -> 128).
    """
    device = net.position_emb.device
    fit_ids = sorted(fit_by_id.keys())
    need(len(fit_ids) == 128 and len(schedule) == 32, 'Expected 128 FIT cases and 32 cycles')

    fuse_keys = [k for k in net.state_dict().keys() if k.startswith('fuse_convs_dict.')]
    need(len(fuse_keys) == 54, 'Expected 54 fusion state tensors')
    fusion_params = [param for size in ('32', '64', '128') for param in net.fuse_convs_dict[size].parameters()]
    need(len(fusion_params) == 54, 'Expected 54 fusion parameter tensors')

    # Precompute frozen Stage A skip features + dec32 on FIT cases without invoking whole-model hook
    net.requires_grad_(False)
    out_list = [net.fuse_encoder_block[f_size] for f_size in net.connect_list]
    fit_efd = {}
    fit_dec32 = {}
    with torch.no_grad():
        for i in range(0, len(fit_ids), 4):
            guard()
            chunk_ids = fit_ids[i : i + 4]
            h = torch.cat([fit_x512[cid] for cid in chunk_ids], dim=0)
            efd_b = {}
            for bi in range(15):
                h = net.encoder.blocks[bi](h)
                if bi in out_list:
                    efd_b[str(h.shape[-1])] = h.clone()
            lq_b = encode_from_block15(net, h)
            lg_b = predict_logits_from_lq(net, lq_b)
            _, top_b = torch.topk(F.softmax(lg_b, dim=2), 1, dim=2)
            for j, cid in enumerate(chunk_ids):
                fit_efd[cid] = {s: efd_b[s][j : j + 1].clone() for s in ('32', '64', '128')}
                g = net.quantize.get_codebook_feat(top_b[j : j + 1], shape=[1, 16, 16, 256])
                for gi in range(10):
                    g = net.generator.blocks[gi](g)
                fit_dec32[cid] = g.clone()

    calibration_passes = [
        (1.2e-5, 3.0e-6, 1.15, 0.85, 0.90, 0.90, 0.40, 0.20, 0.10, 0.12, 0.28, 200.0, 5.0e-5),
        (1.5e-5, 4.0e-6, 1.15, 0.85, 0.90, 0.90, 0.40, 0.20, 0.10, 0.12, 0.28, 300.0, 5.0e-5),
    ]
    for (
        lr_max,
        lr_min,
        rw_anc,
        rw_aug,
        w_h512,
        w_s512,
        w_h256,
        w_s256,
        w_grad,
        w_chr_rgb,
        w_chr_gray,
        area_tau,
        trust_c,
    ) in calibration_passes:
        for k in fuse_keys:
            net.state_dict()[k].copy_(r1_state_cpu[k].to(device=device, dtype=torch.float32))
        net.requires_grad_(False)
        for p in fusion_params:
            p.requires_grad_(True)
        r1_fuse_anchors = [p.detach().clone() for p in fusion_params]
        for cycle, case_ids in enumerate(schedule, 1):
            guard()
            prog = float(cycle - 1) / 31.0
            lr = lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))
            for g in fusion_optimizer.param_groups:
                g['lr'] = float(lr)
            fusion_optimizer.zero_grad(set_to_none=True)

            efd_b = {s: torch.cat([fit_efd[cid][s] for cid in case_ids], dim=0) for s in ('32', '64', '128')}
            dec_b = torch.cat([fit_dec32[cid] for cid in case_ids], dim=0)
            raw_b = decode_batch_from_dec32(net, efd_b, dec_b)

            batch_obj = 0.0
            for j, cid in enumerate(case_ids):
                case = fit_by_id[cid]
                rgb, m, m512, t256, t512 = fit_io[cid]
                rw = rw_anc if case['role'] == 'fit_anchor' else rw_aug
                f_obj, _, _ = _v3r2_hybrid_fusion_objective(
                    raw_b[j : j + 1],
                    rgb,
                    m,
                    m512,
                    t256,
                    t512,
                    role_weight=rw,
                    style=case['style'],
                    F=F,
                    w_h512=w_h512,
                    w_s512=w_s512,
                    w_h256=w_h256,
                    w_s256=w_s256,
                    w_grad=w_grad,
                    w_chr_rgb=w_chr_rgb,
                    w_chr_gray=w_chr_gray,
                    area_tau256=area_tau,
                )
                batch_obj = batch_obj + f_obj / 4.0
            tr = anchor_trust_penalty(fusion_params, r1_fuse_anchors, coef=trust_c)
            loss = batch_obj + tr
            need(bool(torch.isfinite(loss).item()), 'Nonfinite fusion moment calibration loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_params, 1.0, error_if_nonfinite=True)
            fusion_optimizer.step()

    # Reset fuse_convs_dict weights back to pure R1 while retaining calibrated Adam moments (step == 96)
    for k in fuse_keys:
        net.state_dict()[k].copy_(r1_state_cpu[k].to(device=device, dtype=torch.float32))
    fusion_optimizer.zero_grad(set_to_none=True)
    net.requires_grad_(False)
    net.zero_grad(set_to_none=True)
    steps_after = [float(fusion_optimizer.state[p]['step'].item()) for p in fusion_params]
    need(all(s == 96.0 for s in steps_after), 'Expected calibrated fusion Adam step == 96')
    return {
        'complete': True,
        'fit_only': True,
        'calibration_passes': 2,
        'calibration_steps': 64,
        'fusion_step_after_calibration': 96,
        'weights_reset_to_R1_before_stage_b': True,
    }
