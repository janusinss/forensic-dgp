"""Conservative Gibbs ringing (overshoot/undershoot halo) detection and suppression."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label,
    map_coordinates,
)

from .preparation import png

_NORMAL_OFFSETS = np.arange(-4, 5, dtype=np.float32)
_LAPLACIAN_HP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4.0
_BILATERAL_RADIUS = 3
_BILATERAL_SIGMA_SPACE = 2.2


def _luma_and_gradient(rgb_float):
    """Compute BT.601 float32 luma, Scharr gradient magnitude, and unit edge normal."""
    y_luma = (
        0.299 * rgb_float[..., 0]
        + 0.587 * rgb_float[..., 1]
        + 0.114 * rgb_float[..., 2]
    ).astype(np.float32)
    gx = cv2.Scharr(y_luma, cv2.CV_32F, 1, 0, scale=1.0 / 32.0, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Scharr(y_luma, cv2.CV_32F, 0, 1, scale=1.0 / 32.0, borderType=cv2.BORDER_REPLICATE)
    grad = np.hypot(gx, gy)
    safe = np.maximum(grad, 1e-6)
    return y_luma, grad, gx / safe, gy / safe


def _sample_normal_triplet(field, rows, cols, nx_c, ny_c, tangent_step=1.25):
    """Sample normal profiles at t = -tangent_step, 0, +tangent_step along edge tangent."""
    rows_f = rows.astype(np.float32)[:, None]
    cols_f = cols.astype(np.float32)[:, None]
    dn_y = ny_c[:, None] * _NORMAL_OFFSETS[None, :]
    dn_x = nx_c[:, None] * _NORMAL_OFFSETS[None, :]
    ty = -nx_c[:, None]
    tx = ny_c[:, None]
    # Note: unit tangent orthogonal to (nx, ny) is (-ny, nx) -> row shift +nx, col shift -ny.
    # Either sign (-ny, nx) or (ny, -nx) is symmetric for the +/- pair.
    samples = []
    for t_off in (-tangent_step, 0.0, tangent_step):
        sy = rows_f + t_off * ty + dn_y
        sx = cols_f + t_off * tx + dn_x
        samples.append(map_coordinates(field, [sy, sx], order=1, mode='nearest'))
    return samples[0], samples[1], samples[2]


def _has_alternating_lobes(residuals, tolerance):
    """Return True where residuals exhibit both significant positive and negative lobes."""
    signs = np.where(
        residuals > tolerance[:, None],
        1,
        np.where(residuals < -tolerance[:, None], -1, 0),
    )
    return (signs.max(axis=1) == 1) & (signs.min(axis=1) == -1)


def _analyze_ringing(source):
    """Locate crisp ringing edge cores and their 2..4 px flanking halo band."""
    if (
        not isinstance(source, np.ndarray)
        or source.ndim != 3
        or source.shape[2] != 3
        or source.dtype != np.uint8
    ):
        return None
    height, width = source.shape[:2]
    if min(height, width) < 32:
        return None

    rgb_float = source.astype(np.float32)
    y_luma, grad, nx, ny = _luma_and_gradient(rgb_float)
    if float(y_luma.max() - y_luma.min()) < 16.0 or float(y_luma.std()) < 2.0:
        return None

    hp_resp = cv2.filter2D(y_luma, cv2.CV_32F, _LAPLACIAN_HP, borderType=cv2.BORDER_REPLICATE)
    noise_sigma = float(np.median(np.abs(hp_resp)) / (6.0 * 0.67448975))
    grad_thresh = max(9.0, 4.5 * noise_sigma, 2.2 * float(np.median(grad)))

    yy, xx = np.indices((height, width), dtype=np.float32)
    grad_plus = map_coordinates(grad, [yy + ny, xx + nx], order=1, mode='nearest')
    grad_minus = map_coordinates(grad, [yy - ny, xx - nx], order=1, mode='nearest')
    core_all = (grad >= grad_thresh) & (grad >= grad_plus - 1e-3) & (grad > grad_minus + 1e-3)
    core_interior = core_all.copy()
    core_interior[:4] = False
    core_interior[-4:] = False
    core_interior[:, :4] = False
    core_interior[:, -4:] = False

    cy, cx = np.nonzero(core_interior)
    if len(cy) < 10:
        return None

    nxc, nyc = nx[cy, cx], ny[cy, cx]
    p_m, p_0, p_p = _sample_normal_triplet(y_luma, cy, cx, nxc, nyc)
    profiles = 0.25 * p_m + 0.50 * p_0 + 0.25 * p_p
    _, g_prof, _ = _sample_normal_triplet(grad, cy, cx, nxc, nyc, tangent_step=0.0)

    b_low = 0.25 * profiles[:, 0] + 0.50 * profiles[:, 1] + 0.25 * profiles[:, 2]
    b_high = 0.25 * profiles[:, 6] + 0.50 * profiles[:, 7] + 0.25 * profiles[:, 8]
    jump = b_high - b_low
    g_core = grad[cy, cx]
    g_flank_max = np.maximum(g_prof[:, 0:3].max(axis=1), g_prof[:, 6:9].max(axis=1))
    isolated = (jump >= 18.0) & (g_flank_max < 0.55 * g_core)
    if int(isolated.sum()) < 10:
        return None

    d_dark = np.diff(profiles[:, 0:4], axis=1)
    d_bright = np.diff(profiles[:, 5:9], axis=1)
    rev_dark = np.maximum(0.0, -d_dark).sum(axis=1)
    rev_bright = np.maximum(0.0, -d_bright).sum(axis=1)
    osc2_dark = (
        np.abs(profiles[:, 0] - 2.0 * profiles[:, 1] + profiles[:, 2])
        + np.abs(profiles[:, 1] - 2.0 * profiles[:, 2] + profiles[:, 3])
    )
    osc2_bright = (
        np.abs(profiles[:, 5] - 2.0 * profiles[:, 6] + profiles[:, 7])
        + np.abs(profiles[:, 6] - 2.0 * profiles[:, 7] + profiles[:, 8])
    )

    tol = np.maximum(1.0, 0.012 * jump)
    both_dark_wing = _has_alternating_lobes(profiles[:, 0:4] - b_low[:, None], tol)
    both_bright_wing = _has_alternating_lobes(profiles[:, 5:9] - b_high[:, None], tol)
    both_dark_24 = _has_alternating_lobes(profiles[:, 0:3] - b_low[:, None], tol)
    both_bright_24 = _has_alternating_lobes(profiles[:, 6:9] - b_high[:, None], tol)

    res_m = np.concatenate([p_m[:, 0:3] - b_low[:, None], p_m[:, 6:9] - b_high[:, None]], axis=1)
    res_p = np.concatenate([p_p[:, 0:3] - b_low[:, None], p_p[:, 6:9] - b_high[:, None]], axis=1)
    denom = np.linalg.norm(res_m, axis=1) * np.linalg.norm(res_p, axis=1)
    safe_denom = np.where(denom > 1e-3, denom, 1.0)
    coherence = np.where(denom > 1e-3, (res_m * res_p).sum(axis=1) / safe_denom, 0.0)

    dark_ok = both_dark_wing & (rev_dark >= 0.030 * jump) & (osc2_dark >= 0.045 * jump)
    bright_ok = both_bright_wing & (rev_bright >= 0.030 * jump) & (osc2_bright >= 0.045 * jump)
    osc_ok = (
        (dark_ok & bright_ok)
        | (both_dark_24 & (rev_dark >= 0.038 * jump))
        | (both_bright_24 & (rev_bright >= 0.038 * jump))
    )
    ring_flag = (
        isolated
        & osc_ok
        & (np.maximum(rev_dark, rev_bright) >= 2.5)
        & (coherence >= 0.38)
    )
    if not np.any(ring_flag):
        return None

    ring_raw = np.zeros((height, width), dtype=bool)
    ring_raw[cy[ring_flag], cx[ring_flag]] = True
    local_hits = cv2.boxFilter(
        ring_raw.astype(np.float32),
        -1,
        (7, 7),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    ring_bridged = (
        ring_raw | (core_interior & binary_dilation(ring_raw, structure=np.ones((5, 5), dtype=bool)))
    ) & (local_hits >= 3.5)

    comp, num_comp = label(ring_bridged, structure=np.ones((3, 3), dtype=int))
    counts = np.bincount(comp.ravel(), minlength=num_comp + 1)
    counts[0] = 0
    min_comp = max(10, min(16, min(height, width) // 8))
    ring_core = (counts >= min_comp)[comp]

    ring_core_px = int(ring_core.sum())
    min_total = max(12, round(0.18 * min(height, width)))
    kept_frac = float(ring_core_px) / max(1, int(isolated.sum()))
    raw_in_kept = int((ring_core & ring_raw).sum())
    if ring_core_px < min_total or kept_frac < 0.05 or raw_in_kept < max(8, min_total // 2):
        return None

    dist_ring, (near_y, near_x) = distance_transform_edt(~ring_core, return_indices=True)
    dist_any = distance_transform_edt(~core_all)
    side_dot = (xx - near_x) * nx[near_y, near_x] + (yy - near_y) * ny[near_y, near_x]
    halo_mask = (
        (dist_ring >= 1.8)
        & (dist_ring <= 4.5)
        & (dist_any >= 1.8)
        & (np.abs(side_dot) >= 0.35 * dist_ring)
    )
    halo_px = int(halo_mask.sum())
    if halo_px < max(36, round(0.6 * min(height, width))):
        return None

    # F2 tightened flank/area gates: genuine ringing sits on calm isolated step edges
    # (median flank/core gradient <= 0.39) with meaningful spatial halo/core support.
    ry, rx = np.nonzero(ring_core)
    if len(ry) == 0:
        return None
    _, g_ring_prof, _ = _sample_normal_triplet(
        grad, ry, rx, nx[ry, rx], ny[ry, rx], tangent_step=0.0
    )
    g_ring_core = grad[ry, rx]
    g_ring_flank = np.maximum(g_ring_prof[:, 0:3].max(axis=1), g_ring_prof[:, 6:9].max(axis=1))
    flank_ratio_med = float(np.median(g_ring_flank / np.maximum(g_ring_core, 1e-6)))
    halo_band_fraction = round(float(halo_px) / float(height * width), 4)
    if not math.isfinite(flank_ratio_med) or flank_ratio_med > 0.39:
        return None
    if halo_band_fraction < 0.0023 and ring_core_px < max(48, round(0.00045 * height * width)):
        return None

    kept_idx = ring_core[cy, cx] & ring_flag
    rel_rev = np.maximum(rev_dark[kept_idx], rev_bright[kept_idx]) / np.maximum(jump[kept_idx], 1.0)
    report = {
        'policy': 'gibbs-halo-deringing-v1',
        'classification_is_approximate': True,
        'native_size': [int(width), int(height)],
        'primary_core_pixels': int(len(cy)),
        'ringing_core_pixels': ring_core_px,
        'halo_band_pixels': halo_px,
        'halo_band_fraction': halo_band_fraction,
        'median_edge_contrast': round(float(np.median(jump[kept_idx])), 2),
        'median_reversal_ratio': round(float(np.median(rel_rev)), 4),
        'median_tangential_coherence': round(float(np.median(coherence[kept_idx])), 4),
        'noise_sigma_255': round(noise_sigma, 3),
    }
    if not all(
        math.isfinite(val)
        for val in (
            report['halo_band_fraction'],
            report['median_edge_contrast'],
            report['median_reversal_ratio'],
            report['median_tangential_coherence'],
            report['noise_sigma_255'],
        )
    ):
        return None

    context = {
        'rgb_float': rgb_float,
        'y_luma': y_luma,
        'grad': grad,
        'nx': nx,
        'ny': ny,
        'core_all': core_all,
        'ring_core': ring_core,
        'halo_mask': halo_mask,
        'dist_ring': dist_ring,
        'near_y': near_y,
        'near_x': near_x,
        'side_dot': side_dot,
    }
    return report, context


def _estimate_rgb_plateaus(rgb_float, ring_core, nx, ny):
    """Estimate dark-side and bright-side RGB plateau baselines at ringing core pixels."""
    height, width = ring_core.shape
    ry, rx = np.nonzero(ring_core)
    nxr, nyr = nx[ry, rx], ny[ry, rx]
    c_low = np.zeros((height, width, 3), dtype=np.float32)
    c_high = np.zeros((height, width, 3), dtype=np.float32)
    for ch in range(3):
        pm, p0, pp = _sample_normal_triplet(rgb_float[..., ch], ry, rx, nxr, nyr)
        pc = 0.25 * pm + 0.50 * p0 + 0.25 * pp
        c_low[ry, rx, ch] = 0.25 * pc[:, 0] + 0.50 * pc[:, 1] + 0.25 * pc[:, 2]
        c_high[ry, rx, ch] = 0.25 * pc[:, 6] + 0.50 * pc[:, 7] + 0.25 * pc[:, 8]
    luma_jump = np.abs(
        0.299 * (c_high[ry, rx, 0] - c_low[ry, rx, 0])
        + 0.587 * (c_high[ry, rx, 1] - c_low[ry, rx, 1])
        + 0.114 * (c_high[ry, rx, 2] - c_low[ry, rx, 2])
    )
    jump_map = np.zeros((height, width), dtype=np.float32)
    jump_map[ry, rx] = luma_jump
    return c_low, c_high, jump_map


def _bilateral_halo(clamped, core_all, side_dot, halo_rows, halo_cols, sigma_color):
    """Apply vectorized same-side edge-preserving bilateral filtering on halo pixels only."""
    pad = _BILATERAL_RADIUS
    c_pad = np.pad(clamped, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    core_pad = np.pad(core_all, ((pad, pad), (pad, pad)), mode='constant', constant_values=True)
    side_sign = np.where(core_all, 0, np.where(side_dot > 0.0, 1, -1))
    side_pad = np.pad(side_sign, ((pad, pad), (pad, pad)), mode='edge')

    center_rgb = clamped[halo_rows, halo_cols]
    center_side = side_sign[halo_rows, halo_cols]
    inv_2sc2 = 1.0 / (2.0 * sigma_color[:, None] ** 2)
    inv_2ss2 = 1.0 / (2.0 * (_BILATERAL_SIGMA_SPACE ** 2))

    acc = np.zeros_like(center_rgb)
    wsum = np.zeros((len(halo_rows), 1), dtype=np.float32)
    hy_p = halo_rows + pad
    hx_p = halo_cols + pad
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            ws = math.exp(-(dy * dy + dx * dx) * inv_2ss2)
            ny_p = hy_p + dy
            nx_p = hx_p + dx
            valid = (~core_pad[ny_p, nx_p]) & (side_pad[ny_p, nx_p] == center_side)
            nbr = c_pad[ny_p, nx_p]
            diff2 = np.sum((nbr - center_rgb) ** 2, axis=1, keepdims=True) / 3.0
            weight = ws * np.exp(-diff2 * inv_2sc2) * valid[:, None]
            acc += weight * nbr
            wsum += weight
    return np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-6), center_rgb)


def detect_ringing(source):
    """Detect genuine Gibbs ringing halos flanking crisp high-contrast edges."""
    analysis = _analyze_ringing(source)
    return None if analysis is None else analysis[0]


def prepare_deringing(source):
    """Suppress Gibbs overshoot/undershoot strictly inside the 2..4 px flanking halo band."""
    analysis = _analyze_ringing(source)
    if analysis is None:
        return None, None, {}
    report, ctx = analysis

    rgb_float = ctx['rgb_float']
    y_before = ctx['y_luma']
    grad_before = ctx['grad']
    core_all = ctx['core_all']
    ring_core = ctx['ring_core']
    halo_mask = ctx['halo_mask']
    dist_ring = ctx['dist_ring']
    near_y = ctx['near_y']
    near_x = ctx['near_x']
    side_dot = ctx['side_dot']

    c_low_map, c_high_map, jump_map = _estimate_rgb_plateaus(
        rgb_float, ring_core, ctx['nx'], ctx['ny']
    )

    hy, hx = np.nonzero(halo_mask)
    cy_n, cx_n = near_y[hy, hx], near_x[hy, hx]
    c_low_h = c_low_map[cy_n, cx_n]
    c_high_h = c_high_map[cy_n, cx_n]
    c_min = np.minimum(c_low_h, c_high_h)
    c_max = np.maximum(c_low_h, c_high_h)
    is_pos = side_dot[hy, hx] > 0.0
    p_side = np.where(is_pos[:, None], c_high_h, c_low_h)
    slack = np.maximum(2.5, 0.03 * (c_max - c_min))

    clamped = rgb_float.copy()
    clamped_h = np.clip(rgb_float[hy, hx], c_min, c_max)
    clamped[hy, hx] = np.clip(clamped_h, p_side - slack, p_side + slack)

    sigma_color = np.clip(0.28 * jump_map[cy_n, cx_n], 12.0, 44.0).astype(np.float32)
    bilat_h = _bilateral_halo(clamped, core_all, side_dot, hy, hx, sigma_color)

    alpha_h = np.clip((4.7 - dist_ring[hy, hx]) / 1.4, 0.3, 1.0).astype(np.float32)[:, None]
    output = source.copy()
    output[hy, hx] = np.clip(
        np.rint(rgb_float[hy, hx] * (1.0 - alpha_h) + bilat_h * alpha_h), 0, 255
    ).astype(np.uint8)

    changed = int(np.any(output != source, axis=2).sum())
    if not changed or not np.array_equal(output[~halo_mask], source[~halo_mask]):
        return None, None, {}

    y_after, grad_after, _, _ = _luma_and_gradient(output.astype(np.float32))
    y_side = 0.299 * p_side[:, 0] + 0.587 * p_side[:, 1] + 0.114 * p_side[:, 2]
    osc_before = float(np.mean((y_before[hy, hx] - y_side) ** 2))
    osc_after = float(np.mean((y_after[hy, hx] - y_side) ** 2))
    hp_before = float(
        np.mean(cv2.filter2D(y_before, cv2.CV_32F, _LAPLACIAN_4, borderType=cv2.BORDER_REPLICATE)[hy, hx] ** 2)
    )
    hp_after = float(
        np.mean(cv2.filter2D(y_after, cv2.CV_32F, _LAPLACIAN_4, borderType=cv2.BORDER_REPLICATE)[hy, hx] ** 2)
    )
    core_grad_before = float(grad_before[ring_core].mean())
    core_grad_after = float(grad_after[ring_core].mean())

    if osc_before <= 1e-6 or hp_before <= 1e-6 or core_grad_before <= 1e-6:
        return None, None, {}

    osc_drop = 1.0 - osc_after / osc_before
    hp_drop = 1.0 - hp_after / hp_before
    core_retention = core_grad_after / core_grad_before
    if (
        not all(math.isfinite(v) for v in (osc_drop, hp_drop, core_retention))
        or osc_drop < 0.25
        or hp_drop < 0.25
        or core_retention < 0.97
    ):
        return None, None, {}

    halo_mask_u8 = halo_mask.astype(np.uint8) * 255
    report.update(
        changed_native_pixels=changed,
        halo_oscillation_energy_before=round(osc_before, 4),
        halo_oscillation_energy_after=round(osc_after, 4),
        halo_oscillation_drop=round(osc_drop, 4),
        halo_highpass_energy_before=round(hp_before, 4),
        halo_highpass_energy_after=round(hp_after, 4),
        halo_highpass_drop=round(hp_drop, 4),
        core_gradient_retention=round(core_retention, 6),
        bilateral_sigma_space=float(_BILATERAL_SIGMA_SPACE),
        bilateral_sigma_color_median=round(float(np.median(sigma_color)), 2),
        edge_core_untouched=True,
        outside_halo_exact=True,
    )
    return output, report, {'deringing-halo-mask.png': png(halo_mask_u8)}
