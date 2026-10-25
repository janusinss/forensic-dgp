"""Pure input-only exposure/degradation conditioning and block16 decoupling.

Importing this module loads no numerical framework, image decoder, model or
worker. Callers pass NumPy or Torch explicitly. All conditioning gates depend
solely on the 256x256 RGB input observation x in [0, 1], never on any clean
target, role label or external metadata.
"""
import math

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
BLOCK_SIZE = 16
GRID_SIZE = 16
IMAGE_SIZE = 256
TOP1PCT_PIXELS = 655
DARK_HIGH_LOW = 0.640
DARK_HIGH_HIGH = 0.680
DEG_LAP_LOW = 0.0060
DEG_LAP_HIGH = 0.0220
DEFAULT_ANCHOR_BETA = 0.005
DEFAULT_DISPLACEMENT_TAU = 1e-6


def need(condition, message):
    if not condition:
        raise ValueError(message)


def displacement_gamma(max_linf_displacement, *, tau=DEFAULT_DISPLACEMENT_TAU):
    """Map parameter Linf displacement from u64 initializer to gamma in [0, 1]."""
    need(type(max_linf_displacement) in (int, float)
         and math.isfinite(max_linf_displacement)
         and float(max_linf_displacement) >= 0.0,
         'Finite nonnegative parameter displacement required')
    need(type(tau) in (int, float) and math.isfinite(tau) and float(tau) > 0.0,
         'Positive displacement threshold tau required')
    return float(min(1.0, float(max_linf_displacement) / float(tau)))


def conditioning_contract():
    """Return the static mathematical specification of input-only conditioning."""
    return {
        'status': 'PURE_INPUT_ONLY_EXPOSURE_DEGRADATION_CONDITIONING_V1',
        'image_geometry': [IMAGE_SIZE, IMAGE_SIZE, 3],
        'block_size': BLOCK_SIZE,
        'luma_weights_rec709': list(LUMA_WEIGHTS),
        'top1pct_pixel_count': TOP1PCT_PIXELS,
        'dark_gate_ramp': [DARK_HIGH_LOW, DARK_HIGH_HIGH],
        'degradation_laplacian_ramp': [DEG_LAP_LOW, DEG_LAP_HIGH],
        'default_anchor_beta': DEFAULT_ANCHOR_BETA,
        'default_displacement_tau': DEFAULT_DISPLACEMENT_TAU,
        'target_or_role_label_used_in_gates': False,
        'zero_update_exact_u64_parity_preserved': True,
    }


def _check_hwc_numpy(array, *, np, label):
    need(isinstance(array, np.ndarray) and array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and array.dtype.kind == 'f' and np.isfinite(array).all()
         and float(array.min()) >= 0.0 and float(array.max()) <= 1.0,
         'Finite floating HWC RGB256 ' + label + ' in [0,1] required')
    return array.astype(np.float64)


def block16_mean_numpy(array, *, np):
    """Orthogonal 16x16 block-mean projection expanded to (256, 256, 3)."""
    need(isinstance(array, np.ndarray) and array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and array.dtype.kind == 'f' and np.isfinite(array).all(),
         'Finite floating HWC RGB256 array required for block16 projection')
    blocks = array.astype(np.float64).reshape(
        GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE, 3
    ).mean(axis=(1, 3), keepdims=True)
    return np.repeat(np.repeat(blocks, BLOCK_SIZE, axis=1), BLOCK_SIZE, axis=3).reshape(
        IMAGE_SIZE, IMAGE_SIZE, 3
    )


def compute_input_conditioning_numpy(input_rgb, *, np):
    """Compute input-only g_dark(x) and g_deg(x) from HWC float RGB256 in [0,1]."""
    x = _check_hwc_numpy(input_rgb, np=np, label='input observation')
    weights = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
    luma = x @ weights
    mean_luma = float(luma.mean())
    sorted_luma = np.sort(luma.reshape(-1))
    top1pct_luma = float(sorted_luma[-TOP1PCT_PIXELS:].mean())
    g_dark = float(np.clip(
        (DARK_HIGH_HIGH - top1pct_luma) / (DARK_HIGH_HIGH - DARK_HIGH_LOW),
        0.0, 1.0,
    ))
    lap_x = np.abs(luma[:, 2:] - 2.0 * luma[:, 1:-1] + luma[:, :-2])
    lap_y = np.abs(luma[2:, :] - 2.0 * luma[1:-1, :] + luma[:-2, :])
    lap_energy = float(0.5 * (lap_x.mean() + lap_y.mean()))
    central = luma[64:192, 64:192]
    c_lap_x = np.abs(central[:, 2:] - 2.0 * central[:, 1:-1] + central[:, :-2])
    c_lap_y = np.abs(central[2:, :] - 2.0 * central[1:-1, :] + central[:-2, :])
    central_lap = float(0.5 * (c_lap_x.mean() + c_lap_y.mean()))
    effective_lap = min(lap_energy, central_lap)
    g_deg = float(np.clip(
        (DEG_LAP_HIGH - effective_lap) / (DEG_LAP_HIGH - DEG_LAP_LOW),
        0.25, 1.0,
    ))
    return {
        'g_dark': g_dark,
        'g_deg': g_deg,
        'mean_luma': mean_luma,
        'top1pct_luma': top1pct_luma,
        'laplacian_energy': lap_energy,
        'central_laplacian_energy': central_lap,
        'effective_laplacian_energy': effective_lap,
    }


def condition_output_numpy(prediction_rgb, input_rgb, *, np, baseline_rgb=None,
                           beta=DEFAULT_ANCHOR_BETA, gamma=1.0):
    """Apply input-conditioned block16 decoupling in HWC RGB256 float space.

    When gamma == 0.0, returns prediction_rgb unchanged (exact 0-update parity).
    When baseline_rgb is supplied, also removes (1 - g_dark) * B16(pred - base)
    before applying the input-anchored coarse correction.
    """
    pred = _check_hwc_numpy(prediction_rgb, np=np, label='prediction')
    inp = _check_hwc_numpy(input_rgb, np=np, label='input observation')
    need(type(beta) in (int, float) and math.isfinite(beta) and 0.0 <= float(beta) <= 0.25,
         'Finite anchor beta in [0, 0.25] required')
    need(type(gamma) in (int, float) and math.isfinite(gamma) and 0.0 <= float(gamma) <= 1.0,
         'Finite activation gamma in [0, 1] required')
    gates = compute_input_conditioning_numpy(inp, np=np)
    if float(gamma) == 0.0:
        return prediction_rgb.astype(np.float32).copy(), gates
    g_dark = gates['g_dark']
    normal_weight = float(gamma) * (1.0 - g_dark)
    adjusted = pred.copy()
    if baseline_rgb is not None:
        base = _check_hwc_numpy(baseline_rgb, np=np, label='baseline prediction')
        coarse_drift = block16_mean_numpy(pred - base, np=np)
        adjusted = adjusted - normal_weight * coarse_drift
    if float(beta) > 0.0:
        anchor_ref = _check_hwc_numpy(baseline_rgb, np=np, label='baseline prediction') if baseline_rgb is not None else pred
        anchor_shift = block16_mean_numpy(inp - anchor_ref, np=np)
        adjusted = adjusted + normal_weight * float(beta) * anchor_shift
    clipped = np.clip(adjusted, 0.0, 1.0).astype(np.float32)
    return clipped, gates


def block16_mean_torch(tensor, *, torch):
    """Orthogonal 16x16 block-mean projection on NCHW (1, 3, 256, 256) tensor."""
    need(isinstance(tensor, torch.Tensor) and tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
         and tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(),
         'Finite FP32 NCHW RGB256 tensor required for block16 projection')
    blocks = tensor.reshape(1, 3, GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(
        dim=(3, 5), keepdim=True
    )
    return blocks.expand(1, 3, GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).reshape(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    )


def compute_input_conditioning_torch(input_nchw, *, torch):
    """Compute detached input-only g_dark(x) and g_deg(x) from NCHW RGB256 in [0,1]."""
    need(isinstance(input_nchw, torch.Tensor) and input_nchw.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
         and input_nchw.dtype == torch.float32 and not input_nchw.requires_grad
         and torch.isfinite(input_nchw).all().item()
         and input_nchw.min().item() >= 0.0 and input_nchw.max().item() <= 1.0,
         'Detached finite FP32 NCHW RGB256 input in [0,1] required')
    x = input_nchw.detach().to(dtype=torch.float64)
    w = torch.tensor(LUMA_WEIGHTS, dtype=torch.float64, device=x.device).reshape(1, 3, 1, 1)
    luma = (x * w).sum(dim=1)  # (1, 256, 256)
    mean_luma = float(luma.mean().item())
    sorted_luma, _ = torch.sort(luma.reshape(-1))
    top1pct_luma = float(sorted_luma[-TOP1PCT_PIXELS:].mean().item())
    g_dark = max(0.0, min(1.0, (DARK_HIGH_HIGH - top1pct_luma) / (DARK_HIGH_HIGH - DARK_HIGH_LOW)))
    lap_x = (luma[:, :, 2:] - 2.0 * luma[:, :, 1:-1] + luma[:, :, :-2]).abs()
    lap_y = (luma[:, 2:, :] - 2.0 * luma[:, 1:-1, :] + luma[:, :-2, :]).abs()
    lap_energy = float(0.5 * (lap_x.mean().item() + lap_y.mean().item()))
    central = luma[:, 64:192, 64:192]
    c_lap_x = (central[:, :, 2:] - 2.0 * central[:, :, 1:-1] + central[:, :, :-2]).abs()
    c_lap_y = (central[:, 2:, :] - 2.0 * central[:, 1:-1, :] + central[:, :-2, :]).abs()
    central_lap = float(0.5 * (c_lap_x.mean().item() + c_lap_y.mean().item()))
    effective_lap = min(lap_energy, central_lap)
    g_deg = max(0.25, min(1.0, (DEG_LAP_HIGH - effective_lap) / (DEG_LAP_HIGH - DEG_LAP_LOW)))
    return {
        'g_dark': float(g_dark),
        'g_deg': float(g_deg),
        'mean_luma': mean_luma,
        'top1pct_luma': top1pct_luma,
        'laplacian_energy': lap_energy,
        'central_laplacian_energy': central_lap,
        'effective_laplacian_energy': effective_lap,
    }


def condition_output_torch(prediction_nchw, input_nchw, *, torch,
                           beta=DEFAULT_ANCHOR_BETA, gamma=1.0):
    """Differentiable single-pass NCHW RGB256 conditioning preserving 0-update parity."""
    need(isinstance(prediction_nchw, torch.Tensor)
         and prediction_nchw.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
         and prediction_nchw.dtype == torch.float32
         and torch.isfinite(prediction_nchw).all().item(),
         'Finite FP32 NCHW RGB256 prediction required')
    gates = compute_input_conditioning_torch(input_nchw, torch=torch)
    need(type(beta) in (int, float) and math.isfinite(beta) and 0.0 <= float(beta) <= 0.25,
         'Finite anchor beta in [0, 0.25] required')
    need(type(gamma) in (int, float) and math.isfinite(gamma) and 0.0 <= float(gamma) <= 1.0,
         'Finite activation gamma in [0, 1] required')
    if float(gamma) == 0.0 or float(beta) == 0.0:
        return prediction_nchw, gates
    normal_weight = float(gamma) * (1.0 - gates['g_dark'])
    if normal_weight == 0.0:
        return prediction_nchw, gates
    anchor_shift = block16_mean_torch(input_nchw.detach() - prediction_nchw, torch=torch)
    conditioned = prediction_nchw + float(normal_weight * float(beta)) * anchor_shift
    return conditioned, gates
