"""Photographic-reference anchor substitution for a future manual u64 pilot.

This module imports no framework and constructs no model or optimizer. NumPy
readback and connected Torch computation require explicit caller-supplied modules.
Clear and low-light anchor references change; criterion, coefficients, regions
and local/global blur formulas retain their audited historical definitions.
"""

ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
BETA = 0.01
LUMINANCE = (0.2126, 0.7152, 0.0722)
ROI = (64, 64, 192, 192)


def formula_contract():
    return {
        'status': 'INERT_UNRUN_PHOTOGRAPHIC_ANCHOR_REFERENCE_SUBSTITUTION',
        'anchor_reference': 'Aligned photographic clean TRAIN target, never generated U96/u64 output',
        'clear': '2 * (SmoothL1_RGB + 0.25 * SmoothL1_adjacent_residual)',
        'global_gaussian_1p2': 'L1_RGB + 0.25 * L1_adjacent_residual',
        'local_gaussian_1p2': '0.5 * (L1_RGB_inside + L1_RGB_outside) + 0.125 * (L1_adjacent_inside + L1_adjacent_outside)',
        'lowlight_linear_0p35': 'L1_RGB + 0.25 * L1_adjacent_residual + 2 * SmoothL1_block16_luminance',
        'SmoothL1_beta': BETA, 'luminance_weights': list(LUMINANCE),
        'block_size': 16, 'local_half_open_roi': list(ROI),
        'regional_edges': 'Both endpoints inside the selected region; exclude cross-boundary edges',
        'microbatch_loss_division': 'External worker divides each role objective by four exactly once',
        'changes_architecture_or_quality_gates': False,
        'gradient_and_model_quality_verified': False,
    }


def _role(role):
    if role not in ROLES:
        raise ValueError('Only the four retained paired TRAIN roles are admitted')


def numpy_objective(prediction, target, role, *, np):
    """Saved-array float64 reference only; no parameter application or gradients."""
    _role(role)
    if (not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray)
            or prediction.shape != (256, 256, 3) or target.shape != prediction.shape
            or prediction.dtype.kind != 'f' or target.dtype.kind != 'f'
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()
            or prediction.min() < 0 or prediction.max() > 1
            or target.min() < 0 or target.max() > 1):
        raise ValueError('Finite floating photographic RGB256 arrays in [0,1] required')
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    residual = p - t

    def smooth(values):
        a = np.abs(values)
        return np.where(a < BETA, 0.5 * a * a / BETA, a - 0.5 * BETA)

    def regional(mask, *, use_smooth=False):
        horizontal = (residual[:, 1:] - residual[:, :-1])[mask[:, 1:] & mask[:, :-1]]
        vertical = (residual[1:] - residual[:-1])[mask[1:] & mask[:-1]]
        edges = np.concatenate((horizontal.reshape(-1), vertical.reshape(-1)))
        penalty = smooth if use_smooth else np.abs
        return float(penalty(residual[mask]).mean()), float(penalty(edges).mean())

    whole = np.ones((256, 256), dtype=bool)
    terms = {}
    if role == ROLES[0]:
        inside = np.zeros_like(whole)
        inside[64:192, 64:192] = True
        di, ei = regional(inside)
        do, eo = regional(~inside)
        terms = {'inside_image_L1': di, 'outside_image_L1': do,
                 'inside_adjacent_L1': ei, 'outside_adjacent_L1': eo}
        objective = 0.5 * (di + do) + 0.125 * (ei + eo)
    elif role == 'clear':
        si, se = regional(whole, use_smooth=True)
        di, ei = regional(whole)
        terms = {'photographic_image_anchor_L1': di, 'photographic_adjacent_anchor_L1': ei,
                 'photographic_image_anchor_SmoothL1': si,
                 'photographic_adjacent_anchor_SmoothL1': se}
        objective = 2 * (si + 0.25 * se)
    else:
        di, ei = regional(whole)
        terms = {'target_image_L1': di, 'target_adjacent_L1': ei}
        objective = di + 0.25 * ei
        if role == 'lowlight_linear_0p35':
            def pooled(value):
                return (value * np.asarray(LUMINANCE)).sum(axis=2).reshape(16, 16, 16, 16).mean(axis=(1, 3))
            difference = pooled(p) - pooled(t)
            anchor = float(smooth(difference).mean())
            terms.update(photographic_block16_luminance_anchor_L1=float(np.abs(difference).mean()),
                         photographic_block16_luminance_anchor_SmoothL1=anchor)
            objective += 2 * anchor
    return float(objective), terms


def connected_objective(prediction, target, role, *, torch):
    """Ordinary connected operations for the later guarded manual L4 worker.

    No generated teacher argument exists. Checks are not neural/gradient proof;
    actual finite gradient and parity readbacks remain mandatory in that worker.
    """
    _role(role)
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor)
            or prediction.shape != (1, 3, 256, 256) or target.shape != prediction.shape
            or prediction.dtype != torch.float32 or target.dtype != torch.float32
            or prediction.device != target.device or target.requires_grad
            or not prediction.requires_grad or prediction.grad_fn is None
            or not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item()
            or prediction.min().item() < 0 or prediction.max().item() > 1
            or target.min().item() < 0 or target.max().item() > 1):
        raise ValueError('Connected delivered FP32 RGB256 and detached photographic clean target required')
    residual = prediction - target
    F = torch.nn.functional

    def regional(mask, *, use_smooth=False):
        values = residual.masked_select(mask.expand_as(residual))
        hm = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        vm = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        dh = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).masked_select(hm.expand(1, 3, 256, 255))
        dv = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).masked_select(vm.expand(1, 3, 255, 256))
        edges = torch.cat((dh, dv))
        def penalty(value):
            return (F.smooth_l1_loss(value, torch.zeros_like(value), beta=BETA, reduction='mean')
                    if use_smooth else value.abs().mean())
        return penalty(values), penalty(edges)

    whole = torch.ones((1, 1, 256, 256), dtype=torch.bool, device=prediction.device)
    if role == ROLES[0]:
        inside = torch.zeros_like(whole)
        inside[:, :, 64:192, 64:192] = True
        di, ei = regional(inside)
        do, eo = regional(~inside)
        terms = {'inside_image_L1': di, 'outside_image_L1': do,
                 'inside_adjacent_L1': ei, 'outside_adjacent_L1': eo}
        objective = 0.5 * (di + do) + 0.125 * (ei + eo)
    elif role == 'clear':
        si, se = regional(whole, use_smooth=True)
        di, ei = regional(whole)
        terms = {'photographic_image_anchor_L1': di, 'photographic_adjacent_anchor_L1': ei,
                 'photographic_image_anchor_SmoothL1': si,
                 'photographic_adjacent_anchor_SmoothL1': se}
        objective = 2 * (si + 0.25 * se)
    else:
        di, ei = regional(whole)
        terms = {'target_image_L1': di, 'target_adjacent_L1': ei}
        objective = di + 0.25 * ei
        if role == 'lowlight_linear_0p35':
            weights = torch.tensor(LUMINANCE, device=prediction.device, dtype=torch.float32).view(1, 3, 1, 1)
            def pooled(value):
                return (value * weights).sum(dim=1, keepdim=True).reshape(1, 1, 16, 16, 16, 16).mean(dim=(3, 5))
            difference = pooled(prediction) - pooled(target)
            anchor = F.smooth_l1_loss(difference, torch.zeros_like(difference), beta=BETA, reduction='mean')
            terms.update(photographic_block16_luminance_anchor_L1=difference.abs().mean(),
                         photographic_block16_luminance_anchor_SmoothL1=anchor)
            objective = objective + 2 * anchor
    if objective.grad_fn is None or not objective.requires_grad or not torch.isfinite(objective).item():
        raise ValueError('Finite connected photographic role objective required')
    return objective, terms
