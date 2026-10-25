"""Pure input-only V2 exposure/blur/noise conditioning, directional blur localization, and SFT exposure matching.

Importing this module loads zero neural frameworks (`torch` / `numpy` passed explicitly).
All conditioning signals depend strictly on the observed 256x256 RGB input x in [0, 1],
never on any target image, role label, or split metadata.
"""
import math

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
SQRT2 = 1.4142135623730951
IMAGE_SIZE = 256
BLOCK_SIZE = 16
GRID_SIZE = 16
TOP1PCT_PIXELS = 655

# Exposure gate (top-1% Rec.709 luma)
DARK_HIGH_LOW = 0.640
DARK_HIGH_HIGH = 0.680

# 4-directional minimum central Laplacian (separates 88/88 clear >= 0.00932 from 316/316 blur/motion/local/codec <= 0.00896)
CLEAR_MIN_DIR_LAP_CUTOFF = 0.00915
BLUR_MIN_DIR_LAP_LOW = 0.00480

# Dark smooth Laplacian (separates clean dark >= 0.00284 from blurred dark native CCTV <= 0.00160)
DARK_SMOOTH_LAP_HIGH = 0.00260
DARK_SMOOTH_LAP_LOW = 0.00120

# Dark high-frequency sensor-noise gate (separates clean dark <= 0.00723 from 80/80 noisy lowlight >= 0.00838)
DARK_NOISE_DIFF_LOW = 0.00760
DARK_NOISE_DIFF_HIGH = 0.01250

# Local 16x16 patch directional Laplacian ramp
PATCH_DIR_LAP_HIGH = 0.00880
PATCH_DIR_LAP_LOW = 0.00380

DEFAULT_ANCHOR_BETA = 0.005
DEFAULT_DISPLACEMENT_TAU = 1e-6
SITES = ('32', '64', '128', '256')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def displacement_gamma(max_linf_displacement, *, tau=DEFAULT_DISPLACEMENT_TAU):
    """Map parameter Linf displacement from step-0 u64 initializer to gamma in [0, 1]."""
    need(type(max_linf_displacement) in (int, float)
         and math.isfinite(max_linf_displacement)
         and float(max_linf_displacement) >= 0.0,
         'Finite nonnegative parameter displacement required')
    need(type(tau) in (int, float) and math.isfinite(tau) and float(tau) > 0.0,
         'Positive displacement threshold tau required')
    return float(min(1.0, float(max_linf_displacement) / float(tau)))


def conditioning_v2_contract():
    return {
        'status': 'PURE_INPUT_ONLY_V2_DIRECTIONAL_BLUR_AND_DARK_NOISE_CONDITIONING',
        'image_geometry': [IMAGE_SIZE, IMAGE_SIZE, 3],
        'sites': list(SITES),
        'dark_gate_ramp': [DARK_HIGH_LOW, DARK_HIGH_HIGH],
        'normal_min_directional_lap_ramp': [BLUR_MIN_DIR_LAP_LOW, CLEAR_MIN_DIR_LAP_CUTOFF],
        'dark_smooth_lap_ramp': [DARK_SMOOTH_LAP_LOW, DARK_SMOOTH_LAP_HIGH],
        'dark_sensor_noise_diff_ramp': [DARK_NOISE_DIFF_LOW, DARK_NOISE_DIFF_HIGH],
        'clear_gate_exact_zero_above_min_dir_lap': CLEAR_MIN_DIR_LAP_CUTOFF,
        'default_anchor_beta': DEFAULT_ANCHOR_BETA,
        'default_displacement_tau': DEFAULT_DISPLACEMENT_TAU,
        'uses_target_or_role_label': False,
        'exact_step0_u64_parity': True,
    }


def block16_mean_numpy(array, *, np):
    need(isinstance(array, np.ndarray) and array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and array.dtype.kind == 'f' and np.isfinite(array).all(),
         'Finite floating HWC RGB256 array required')
    blocks = array.astype(np.float64).reshape(
        GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE, 3
    ).mean(axis=(1, 3), keepdims=True)
    return np.repeat(np.repeat(blocks, BLOCK_SIZE, axis=1), BLOCK_SIZE, axis=3).reshape(
        IMAGE_SIZE, IMAGE_SIZE, 3
    )


def _binomial3_2d_numpy(luma, *, np):
    padded = np.pad(luma, ((1, 1), (1, 1)), mode='reflect')
    horiz = 0.25 * padded[:, :-2] + 0.50 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    vert = 0.25 * horiz[:-2, :] + 0.50 * horiz[1:-1, :] + 0.25 * horiz[2:, :]
    return vert


def _directional_lap_maps_numpy(luma, *, np):
    lx = np.pad(np.abs(luma[:, 2:] - 2.0 * luma[:, 1:-1] + luma[:, :-2]), ((0, 0), (1, 1)), mode='edge')
    ly = np.pad(np.abs(luma[2:, :] - 2.0 * luma[1:-1, :] + luma[:-2, :]), ((1, 1), (0, 0)), mode='edge')
    ld1 = np.pad(np.abs(luma[2:, 2:] - 2.0 * luma[1:-1, 1:-1] + luma[:-2, :-2]) / SQRT2, ((1, 1), (1, 1)), mode='edge')
    ld2 = np.pad(np.abs(luma[2:, :-2] - 2.0 * luma[1:-1, 1:-1] + luma[:-2, 2:]) / SQRT2, ((1, 1), (1, 1)), mode='edge')
    return lx, ly, ld1, ld2


def compute_v2_conditioning_numpy(input_rgb, *, np):
    """Compute g_dark, 4-directional g_blur, g_sensor_noise, and 16x16 spatial blur gate in pure NumPy."""
    need(isinstance(input_rgb, np.ndarray) and input_rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and input_rgb.dtype.kind == 'f' and np.isfinite(input_rgb).all()
         and float(input_rgb.min()) >= 0.0 and float(input_rgb.max()) <= 1.0,
         'Finite floating HWC RGB256 input in [0, 1] required')
    x = input_rgb.astype(np.float64)
    w = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
    luma = x @ w
    mean_luma = float(luma.mean())
    sorted_luma = np.sort(luma.reshape(-1))
    top1pct_luma = float(sorted_luma[-TOP1PCT_PIXELS:].mean())
    g_dark = float(np.clip(
        (DARK_HIGH_HIGH - top1pct_luma) / (DARK_HIGH_HIGH - DARK_HIGH_LOW),
        0.0, 1.0,
    ))

    # Central [64:192, 64:192] 4-directional Laplacian (exact interior crop)
    c = luma[64:192, 64:192]
    c_lx = float(np.abs(c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]).mean())
    c_ly = float(np.abs(c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]).mean())
    c_ld1 = float(np.abs(c[2:, 2:] - 2.0 * c[1:-1, 1:-1] + c[:-2, :-2]).mean() / SQRT2)
    c_ld2 = float(np.abs(c[2:, :-2] - 2.0 * c[1:-1, 1:-1] + c[:-2, 2:]).mean() / SQRT2)
    raw_central_lap = float(0.5 * (c_lx + c_ly))
    min_dir_lap = float(min(c_lx, c_ly, c_ld1, c_ld2))

    # Binomial-smoothed Laplacian for dark blur vs high-frequency sensor noise separation
    smooth = _binomial3_2d_numpy(luma, np=np)
    sx = np.pad(np.abs(smooth[:, 2:] - 2.0 * smooth[:, 1:-1] + smooth[:, :-2]), ((0, 0), (1, 1)), mode='edge')
    sy = np.pad(np.abs(smooth[2:, :] - 2.0 * smooth[1:-1, :] + smooth[:-2, :]), ((1, 1), (0, 0)), mode='edge')
    local_smooth_lap = 0.5 * (sx + sy)
    central_smooth_lap = float(local_smooth_lap[64:192, 64:192].mean())
    noise_hf_diff = float(max(0.0, raw_central_lap - central_smooth_lap))

    if g_dark == 0.0:
        g_blur = float(np.clip(
            (CLEAR_MIN_DIR_LAP_CUTOFF - min_dir_lap) / (CLEAR_MIN_DIR_LAP_CUTOFF - BLUR_MIN_DIR_LAP_LOW),
            0.0, 1.0,
        ))
        g_sensor_noise = 0.0
    else:
        g_blur = float(np.clip(
            (DARK_SMOOTH_LAP_HIGH - central_smooth_lap) / (DARK_SMOOTH_LAP_HIGH - DARK_SMOOTH_LAP_LOW),
            0.0, 1.0,
        ))
        g_sensor_noise = float(g_dark * np.clip(
            (noise_hf_diff - DARK_NOISE_DIFF_LOW) / (DARK_NOISE_DIFF_HIGH - DARK_NOISE_DIFF_LOW),
            0.0, 1.0,
        ))

    if g_blur == 0.0:
        effective_16 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    else:
        lx, ly, ld1, ld2 = _directional_lap_maps_numpy(luma, np=np)
        p_lx = lx.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_ly = ly.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_ld1 = ld1.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_ld2 = ld2.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_min = np.minimum(np.minimum(p_lx, p_ly), np.minimum(p_ld1, p_ld2))
        local_16 = np.clip((PATCH_DIR_LAP_HIGH - p_min) / (PATCH_DIR_LAP_HIGH - PATCH_DIR_LAP_LOW), 0.0, 1.0)
        p16 = np.pad(local_16, ((1, 1), (1, 1)), mode='edge')
        smooth_16 = (
            p16[:-2, :-2] + p16[:-2, 1:-1] + p16[:-2, 2:]
            + p16[1:-1, :-2] + p16[1:-1, 1:-1] + p16[1:-1, 2:]
            + p16[2:, :-2] + p16[2:, 1:-1] + p16[2:, 2:]
        ) / 9.0
        # Ensure blurred patches activate strongly even when only part of the face is blurred (local_soft_surroundings)
        gate_activate = min(1.0, g_blur / 0.20)
        effective_16 = (np.maximum(smooth_16, 0.65 * g_blur) * gate_activate).astype(np.float32)

    return {
        'g_dark': g_dark,
        'g_blur': g_blur,
        'g_sensor_noise': g_sensor_noise,
        'mean_luma': mean_luma,
        'top1pct_luma': top1pct_luma,
        'raw_central_laplacian': raw_central_lap,
        'min_directional_laplacian': min_dir_lap,
        'central_smooth_laplacian': central_smooth_lap,
        'noise_hf_diff': noise_hf_diff,
        'max_local_blur': float(effective_16.max()),
        'local_map_16': effective_16,
    }


def compute_v2_conditioning_torch(input_nchw, *, torch):
    """Compute detached V2 conditioning & per-site spatial maps (32, 64, 128, 256) in Torch."""
    F = torch.nn.functional
    need(isinstance(input_nchw, torch.Tensor) and input_nchw.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
         and input_nchw.dtype == torch.float32,
         'FP32 NCHW RGB256 input tensor required')
    with torch.no_grad():
        x = input_nchw.detach()
        w = torch.tensor(LUMA_WEIGHTS, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        luma = (x * w).sum(dim=1, keepdim=True)
        sorted_luma, _ = torch.sort(luma.reshape(-1))
        top1pct_luma = float(sorted_luma[-TOP1PCT_PIXELS:].mean().item())
        g_dark = float(min(1.0, max(0.0, (DARK_HIGH_HIGH - top1pct_luma) / (DARK_HIGH_HIGH - DARK_HIGH_LOW))))

        c = luma[:, :, 64:192, 64:192]
        c_lx = float((c[:, :, :, 2:] - 2.0 * c[:, :, :, 1:-1] + c[:, :, :, :-2]).abs().mean().item())
        c_ly = float((c[:, :, 2:, :] - 2.0 * c[:, :, 1:-1, :] + c[:, :, :-2, :]).abs().mean().item())
        c_ld1 = float(((c[:, :, 2:, 2:] - 2.0 * c[:, :, 1:-1, 1:-1] + c[:, :, :-2, :-2]).abs().mean().item()) / SQRT2)
        c_ld2 = float(((c[:, :, 2:, :-2] - 2.0 * c[:, :, 1:-1, 1:-1] + c[:, :, :-2, 2:]).abs().mean().item()) / SQRT2)
        raw_central_lap = float(0.5 * (c_lx + c_ly))
        min_dir_lap = float(min(c_lx, c_ly, c_ld1, c_ld2))

        k1d = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32, device=x.device)
        k2d = (k1d[:, None] * k1d[None, :])[None, None]
        smooth = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode='reflect'), k2d)
        sx = F.pad((smooth[:, :, :, 2:] - 2.0 * smooth[:, :, :, 1:-1] + smooth[:, :, :, :-2]).abs(), (1, 1, 0, 0), mode='replicate')
        sy = F.pad((smooth[:, :, 2:, :] - 2.0 * smooth[:, :, 1:-1, :] + smooth[:, :, :-2, :]).abs(), (0, 0, 1, 1), mode='replicate')
        local_smooth_lap = 0.5 * (sx + sy)
        central_smooth_lap = float(local_smooth_lap[:, :, 64:192, 64:192].mean().item())
        noise_hf_diff = float(max(0.0, raw_central_lap - central_smooth_lap))

        if g_dark == 0.0:
            g_blur = float(min(1.0, max(0.0, (CLEAR_MIN_DIR_LAP_CUTOFF - min_dir_lap) / (CLEAR_MIN_DIR_LAP_CUTOFF - BLUR_MIN_DIR_LAP_LOW))))
            g_sensor_noise = 0.0
        else:
            g_blur = float(min(1.0, max(0.0, (DARK_SMOOTH_LAP_HIGH - central_smooth_lap) / (DARK_SMOOTH_LAP_HIGH - DARK_SMOOTH_LAP_LOW))))
            g_sensor_noise = float(g_dark * min(1.0, max(0.0, (noise_hf_diff - DARK_NOISE_DIFF_LOW) / (DARK_NOISE_DIFF_HIGH - DARK_NOISE_DIFF_LOW))))

        if g_blur == 0.0:
            effective_16 = torch.zeros((1, 1, GRID_SIZE, GRID_SIZE), dtype=torch.float32, device=x.device)
        else:
            lx = F.pad((luma[:, :, :, 2:] - 2.0 * luma[:, :, :, 1:-1] + luma[:, :, :, :-2]).abs(), (1, 1, 0, 0), mode='replicate')
            ly = F.pad((luma[:, :, 2:, :] - 2.0 * luma[:, :, 1:-1, :] + luma[:, :, :-2, :]).abs(), (0, 0, 1, 1), mode='replicate')
            ld1 = F.pad((luma[:, :, 2:, 2:] - 2.0 * luma[:, :, 1:-1, 1:-1] + luma[:, :, :-2, :-2]).abs() / SQRT2, (1, 1, 1, 1), mode='replicate')
            ld2 = F.pad((luma[:, :, 2:, :-2] - 2.0 * luma[:, :, 1:-1, 1:-1] + luma[:, :, :-2, 2:]).abs() / SQRT2, (1, 1, 1, 1), mode='replicate')
            p_lx = F.avg_pool2d(lx, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_ly = F.avg_pool2d(ly, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_ld1 = F.avg_pool2d(ld1, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_ld2 = F.avg_pool2d(ld2, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_min = torch.minimum(torch.minimum(p_lx, p_ly), torch.minimum(p_ld1, p_ld2))
            local_16 = torch.clamp((PATCH_DIR_LAP_HIGH - p_min) / (PATCH_DIR_LAP_HIGH - PATCH_DIR_LAP_LOW), 0.0, 1.0)
            box3 = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=x.device) / 9.0
            smooth_16 = F.conv2d(F.pad(local_16, (1, 1, 1, 1), mode='replicate'), box3)
            gate_activate = min(1.0, g_blur / 0.20)
            effective_16 = torch.clamp(smooth_16, min=0.65 * g_blur) * float(gate_activate)

        site_maps = {
            s: F.interpolate(effective_16, size=(int(s), int(s)), mode='bilinear', align_corners=False)
            for s in SITES
        }
        return {
            'g_dark': g_dark,
            'g_blur': g_blur,
            'g_sensor_noise': g_sensor_noise,
            'top1pct_luma': top1pct_luma,
            'raw_central_laplacian': raw_central_lap,
            'min_directional_laplacian': min_dir_lap,
            'central_smooth_laplacian': central_smooth_lap,
            'noise_hf_diff': noise_hf_diff,
            'max_local_blur': float(effective_16.max().item()),
            'local_map_16': effective_16,
            'site_maps': site_maps,
        }


def match_sft_exposure_torch(clean_feat, deg_feat, g_dark, *, torch):
    """Align coarse local mean & std of clean SFT target features to degraded features when g_dark > 0."""
    max_g = float(g_dark.detach().max().item()) if isinstance(g_dark, torch.Tensor) else float(g_dark)
    if max_g <= 0.0:
        return clean_feat.detach()
    F = torch.nn.functional
    with torch.no_grad():
        cf = clean_feat.detach()
        df = deg_feat.detach()
        h = cf.shape[-1]
        k = max(4, h // 8)
        c_mean = F.interpolate(F.avg_pool2d(cf, kernel_size=k, stride=k), size=(h, h), mode='bilinear', align_corners=False)
        d_mean = F.interpolate(F.avg_pool2d(df, kernel_size=k, stride=k), size=(h, h), mode='bilinear', align_corners=False)
        c_std = F.interpolate(F.avg_pool2d((cf - c_mean) ** 2, kernel_size=k, stride=k), size=(h, h), mode='bilinear', align_corners=False).clamp_min(1e-6).sqrt()
        d_std = F.interpolate(F.avg_pool2d((df - d_mean) ** 2, kernel_size=k, stride=k), size=(h, h), mode='bilinear', align_corners=False).clamp_min(1e-6).sqrt()
        matched = (cf - c_mean) * (d_std / c_std).clamp(0.5, 2.0) + d_mean
        w_dark = g_dark if isinstance(g_dark, torch.Tensor) else float(g_dark)
        return ((1.0 - w_dark) * cf + w_dark * matched).detach()


def condition_output_v2_numpy(pred_rgb, input_rgb, *, np, baseline_rgb=None,
                              beta=DEFAULT_ANCHOR_BETA, gamma=1.0, dark_drift_align=True):
    """Apply V2 exposure-decoupled block16 anchoring and dark coarse-drift protection.

    When gamma == 0.0 (step-0 initializer), returns pred_rgb unchanged for exact bitwise u64 parity.
    On normal-lit clear inputs (g_blur == 0, g_dark == 0), anchors coarse block16 tone to input with beta=0.005.
    On normal-lit blurred inputs (g_blur > 0), gently suppresses unintended coarse color drift (-0.30 * B16(pred - base))
    while preserving 100% of restored facial sharpness and contrast!
    """
    cond = compute_v2_conditioning_numpy(input_rgb, np=np)
    if float(gamma) == 0.0:
        return pred_rgb.astype(np.float32).copy(), cond
    pred = pred_rgb.astype(np.float64)
    inp = input_rgb.astype(np.float64)
    base = pred if baseline_rgb is None else baseline_rgb.astype(np.float64)
    g_dark = cond['g_dark']
    g_blur = cond['g_blur']
    normal_w = float(gamma) * (1.0 - g_dark)
    # Full coarse removal (-1.0) when g_blur == 0 (clear); lighter (-0.30) when blurred so deblurred contrast shines
    drift_suppression = 1.0 - 0.70 * min(1.0, g_blur / 0.25)
    coarse_shift = -drift_suppression * block16_mean_numpy(pred - base, np=np) + float(beta) * block16_mean_numpy(inp - base, np=np)
    out = np.clip(pred + normal_w * coarse_shift, 0.0, 1.0)
    if dark_drift_align and baseline_rgb is not None and g_dark > 0.0 and cond['g_sensor_noise'] > 0.0:
        dark_coarse = block16_mean_numpy(out - base, np=np)
        neg_drift = np.minimum(dark_coarse, 0.0)
        out = np.clip(out - g_dark * neg_drift, 0.0, 1.0)
    return out.astype(np.float32), cond


def condition_output_v2_torch(pred_nchw, input_nchw, *, torch, baseline_nchw=None,
                              beta=DEFAULT_ANCHOR_BETA, gamma=1.0):
    """Connected FP32 NCHW output conditioning for Phase-2 end-to-end training."""
    F = torch.nn.functional
    cond = compute_v2_conditioning_torch(input_nchw, torch=torch)
    if float(gamma) == 0.0:
        return pred_nchw, cond
    base = pred_nchw.detach() if baseline_nchw is None else baseline_nchw.detach()
    g_dark = cond['g_dark']
    g_blur = cond['g_blur']
    normal_w = float(gamma) * (1.0 - g_dark)
    drift_suppression = 1.0 - 0.70 * min(1.0, g_blur / 0.25)

    def b16(t):
        blocks = t.reshape(1, 3, GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(dim=(3, 5), keepdim=True)
        return blocks.expand(1, 3, GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    coarse_shift = -drift_suppression * b16(pred_nchw - base) + float(beta) * b16(input_nchw.detach() - base)
    out = (pred_nchw + normal_w * coarse_shift).clamp(0.0, 1.0)
    return out, cond
