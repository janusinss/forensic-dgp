"""Structure-tensor tangent-aligned directional anti-aliasing for jagged diagonal edges."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, label, map_coordinates

from .preparation import png, sha

POLICY = 'structure-tensor-tangent-antialiasing-v1'
MIN_SIDE = 32
COHERENCE_MIN = 0.55
DIAGONAL_TANGENT_MIN = 0.20
GRADIENT_MIN = 12.0
TANGENT_OFFSETS = np.array([-1.5, -0.75, 0.0, 0.75, 1.5], dtype=np.float32)
TANGENT_WEIGHTS = np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32)
SAMPLE_STEPS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)[:, None]
SAW_STEPS = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32)[:, None]
RIM_ALPHA = 0.25


def _as_rgb_uint8(source):
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        return None
    height, width = source.shape[:2]
    if min(height, width) < MIN_SIDE:
        return None
    return source


def _bt601_luma(rgb_uint8):
    return rgb_uint8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _is_hv_block_grid(luma, coherence, grad, tx, ty, y0, y1, x0, x1):
    sub_strong = (coherence[y0:y1, x0:x1] >= COHERENCE_MIN) & (grad[y0:y1, x0:x1] >= GRADIENT_MIN)
    if not sub_strong.any():
        return False
    hv_frac = float((np.minimum(np.abs(tx[y0:y1, x0:x1][sub_strong]),
                                np.abs(ty[y0:y1, x0:x1][sub_strong])) < 0.12).mean())
    if hv_frac < 0.25:
        return False
    sub = luma[y0:y1, x0:x1]
    if min(sub.shape) < 16:
        return False
    scores = []
    periods = []
    for axis in (0, 1):
        profile = np.abs(np.diff(sub, axis=axis)).mean(axis=1 - axis)
        mean_p = float(profile.mean())
        if mean_p < 1.0:
            return False
        best_s = 0.0
        best_p = 0
        max_p = min(16, profile.size // 3)
        for period in range(3, max_p + 1):
            score = max(float(profile[phase::period].mean()) for phase in range(period)) / mean_p
            if score > best_s:
                best_s = score
                best_p = period
        scores.append(best_s)
        periods.append(best_p)
    return min(scores) >= 2.0 and min(periods) > 0 and max(periods) <= 1.5 * min(periods)


def _tangent_metrics(luma, ry, rx, nx, ny, tx, ty, grad):
    vals = map_coordinates(
        luma,
        [ry[None, :].astype(np.float32) + SAMPLE_STEPS * ty[ry, rx][None, :],
         rx[None, :].astype(np.float32) + SAMPLE_STEPS * tx[ry, rx][None, :]],
        order=1,
        mode='nearest',
        prefilter=False,
    )
    d2 = np.array([
        vals[0] - 2.0 * vals[1] + vals[2],
        vals[1] - 2.0 * vals[2] + vals[3],
        vals[2] - 2.0 * vals[3] + vals[4],
    ], dtype=np.float32)
    d2_mag = np.abs(d2).mean(axis=0)
    d3_mag = np.abs(np.diff(d2, axis=0)).mean(axis=0)
    vp = map_coordinates(luma, [ry + ny[ry, rx], rx + nx[ry, rx]], order=1, mode='nearest', prefilter=False)
    vm = map_coordinates(luma, [ry - ny[ry, rx], rx - nx[ry, rx]], order=1, mode='nearest', prefilter=False)
    cross_g = np.abs(vp - vm) * 0.5
    denom = np.maximum(np.maximum(cross_g, grad[ry, rx]), 1.0)
    return d2_mag, d3_mag, cross_g, d2_mag / denom, d3_mag / denom


def _opposite_flank_fraction(d_left, d_right, main_step, ratio):
    step_abs = np.maximum(np.abs(main_step), np.float32(1.0))
    thresh = -ratio * step_abs * step_abs
    opp = (
        ((d_left * main_step) < thresh)
        & ((d_right * main_step) < thresh)
        & (np.minimum(np.abs(d_left), np.abs(d_right)) >= 0.4 * np.maximum(np.abs(d_left), np.abs(d_right)))
    )
    return float(opp.mean())


def _is_sawtooth_wrap_ramp(luma, ry, rx, nx, ny):
    """Reject periodic modulo wrap ramps whose flanking slopes uniformly oppose the main jump."""
    vals = map_coordinates(
        luma,
        [ry[None, :].astype(np.float32) + SAW_STEPS * ny[ry, rx][None, :],
         rx[None, :].astype(np.float32) + SAW_STEPS * nx[ry, rx][None, :]],
        order=1,
        mode='nearest',
        prefilter=False,
    )
    main_step = vals[3] - vals[2]
    return (
        _opposite_flank_fraction(vals[1] - vals[0], vals[5] - vals[4], main_step, np.float32(0.008)) >= 0.55
        or _opposite_flank_fraction(vals[2] - vals[1], vals[4] - vals[3], main_step, np.float32(0.05)) >= 0.55
    )


def _analyze_aliasing(source):
    rgb = _as_rgb_uint8(source)
    if rgb is None:
        return None
    height, width = rgb.shape[:2]
    luma = _bt601_luma(rgb)

    gx = cv2.Scharr(luma, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(luma, cv2.CV_32F, 0, 1) / 32.0
    grad = np.hypot(gx, gy)

    jxx = cv2.GaussianBlur(gx * gx, (5, 5), 1.0)
    jxy = cv2.GaussianBlur(gx * gy, (5, 5), 1.0)
    jyy = cv2.GaussianBlur(gy * gy, (5, 5), 1.0)
    trace = jxx + jyy
    diff = jxx - jyy
    coherence = np.hypot(diff, 2.0 * jxy) / np.maximum(trace, np.float32(1e-6))

    # Dominant eigenvector (nx, ny) is normal to the edge; (-ny, nx) is the local edge tangent.
    theta_n = 0.5 * np.arctan2(2.0 * jxy, diff)
    nx = np.cos(theta_n).astype(np.float32)
    ny = np.sin(theta_n).astype(np.float32)
    tx = -ny
    ty = nx

    yy_idx, xx_idx = np.indices((height, width), dtype=np.float32)
    gp = map_coordinates(grad, [yy_idx + ny, xx_idx + nx], order=1, mode='nearest', prefilter=False)
    gm = map_coordinates(grad, [yy_idx - ny, xx_idx - nx], order=1, mode='nearest', prefilter=False)
    diag_ridge = (
        (coherence >= COHERENCE_MIN)
        & (np.minimum(np.abs(tx), np.abs(ty)) >= DIAGONAL_TANGENT_MIN)
        & (grad >= GRADIENT_MIN)
        & (grad >= gp - 1e-3)
        & (grad >= gm - 1e-3)
    )
    diag_ridge[:3] = diag_ridge[-3:] = diag_ridge[:, :3] = diag_ridge[:, -3:] = False
    if not diag_ridge.any():
        return None

    components, count = label(diag_ridge, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    sizes[0] = 0
    min_comp = max(8, min(16, min(height, width) // 3))

    aliased_ridge = np.zeros((height, width), dtype=bool)
    for cid in range(1, count + 1):
        if sizes[cid] < min_comp:
            continue
        comp_mask = components == cid
        ry, rx = np.nonzero(comp_mask)
        y0 = max(0, int(ry.min()) - 12)
        y1 = min(height, int(ry.max()) + 13)
        x0 = max(0, int(rx.min()) - 12)
        x1 = min(width, int(rx.max()) + 13)
        if _is_hv_block_grid(luma, coherence, grad, tx, ty, y0, y1, x0, x1):
            continue
        if _is_sawtooth_wrap_ramp(luma, ry, rx, nx, ny):
            continue
        _, _, _, r2, r3 = _tangent_metrics(luma, ry, rx, nx, ny, tx, ty, grad)
        if float(np.median(r2)) >= 0.38 and float(np.median(r3)) >= 0.85 and float((r2 >= 0.38).mean()) >= 0.55:
            aliased_ridge |= comp_mask

    n_aliased = int(aliased_ridge.sum())
    diag_count = int(diag_ridge.sum())
    aliased_ratio = float(n_aliased / max(1, diag_count))
    if min(height, width) >= 64:
        if n_aliased < 24 or aliased_ratio < 0.04:
            return None
    elif n_aliased < min_comp:
        return None
    med_coherence = float(np.median(coherence[aliased_ridge]))
    if aliased_ratio < 0.01 and (n_aliased < 48 or med_coherence < 0.72):
        return None

    ry, rx = np.nonzero(aliased_ridge)
    d2_before, _, cross_before, r2_all, r3_all = _tangent_metrics(luma, ry, rx, nx, ny, tx, ty, grad)
    rough_before = float(d2_before.mean())
    cross_before_mean = float(cross_before.mean())
    if rough_before <= 1e-4 or cross_before_mean < 4.0:
        return None

    report = {
        'suspected': True,
        'classification_is_approximate': True,
        'native_size': [int(width), int(height)],
        'diagonal_ridge_pixels': diag_count,
        'aliased_ridge_pixels': n_aliased,
        'median_coherence': round(med_coherence, 4),
        'median_normalized_roughness': round(float(np.median(r2_all)), 4),
        'median_oscillation_index': round(float(np.median(r3_all)), 4),
        'tangent_roughness_before': round(rough_before, 4),
        'cross_edge_grad_before': round(cross_before_mean, 4),
    }
    return {
        'report': report,
        'luma': luma,
        'grad': grad,
        'nx': nx,
        'ny': ny,
        'tx': tx,
        'ty': ty,
        'aliased_ridge': aliased_ridge,
        'ry': ry,
        'rx': rx,
        'rough_before': rough_before,
        'cross_before': cross_before_mean,
    }


def detect_aliasing(source):
    """Detect jagged staircase aliasing along coherent diagonal/slanted contours."""
    analysis = _analyze_aliasing(source)
    return None if analysis is None else analysis['report']


def prepare_antialiasing(source):
    """Apply structure-tensor tangent-aligned 1D directional smoothing to aliased ridges."""
    analysis = _analyze_aliasing(source)
    if analysis is None:
        return None, None, {}

    rgb = source
    aliased_ridge = analysis['aliased_ridge']
    tx = analysis['tx']
    ty = analysis['ty']
    nx = analysis['nx']
    ny = analysis['ny']
    grad = analysis['grad']
    ry = analysis['ry']
    rx = analysis['rx']

    dilated = binary_dilation(aliased_ridge, structure=np.ones((3, 3), dtype=bool))
    alpha = np.where(aliased_ridge, 1.0, np.where(dilated, RIM_ALPHA, 0.0)).astype(np.float32)
    edge_mask = alpha > 0.0
    my, mx = np.nonzero(edge_mask)
    if my.size == 0:
        return None, None, {}

    mtx = tx[my, mx]
    mty = ty[my, mx]
    my_f = my.astype(np.float32)
    mx_f = mx.astype(np.float32)
    rgb_f = rgb.astype(np.float32)
    smoothed = np.zeros((my.size, 3), dtype=np.float32)
    for offset, weight in zip(TANGENT_OFFSETS, TANGENT_WEIGHTS):
        cy = my_f + float(offset) * mty
        cx = mx_f + float(offset) * mtx
        for ch in range(3):
            smoothed[:, ch] += float(weight) * map_coordinates(
                rgb_f[..., ch], [cy, cx], order=1, mode='nearest', prefilter=False
            )

    mix = alpha[my, mx][:, None]
    output = rgb.copy()
    output[my, mx] = np.clip(np.rint(rgb_f[my, mx] * (1.0 - mix) + smoothed * mix), 0.0, 255.0).astype(np.uint8)
    output[~edge_mask] = rgb[~edge_mask]

    changed = int(np.any(output != rgb, axis=2).sum())
    if changed == 0:
        return None, None, {}

    luma_after = _bt601_luma(output)
    d2_after, _, cross_after, _, _ = _tangent_metrics(luma_after, ry, rx, nx, ny, tx, ty, grad)
    rough_before = float(analysis['rough_before'])
    rough_after = float(d2_after.mean())
    cross_before = float(analysis['cross_before'])
    cross_after_mean = float(cross_after.mean())

    roughness_reduction = 1.0 - (rough_after / max(rough_before, 1e-6))
    cross_edge_retention = cross_after_mean / max(cross_before, 1e-6)

    if (
        not math.isfinite(roughness_reduction)
        or not math.isfinite(cross_edge_retention)
        or roughness_reduction < 0.25
        or cross_edge_retention < 0.94
    ):
        return None, None, {}

    mask_u8 = edge_mask.astype(np.uint8) * 255
    full_png = png(output)
    mask_png = png(mask_u8)
    report = dict(analysis['report'])
    report.update(
        policy=POLICY,
        tangent_offsets=[float(v) for v in TANGENT_OFFSETS],
        tangent_weights=[float(v) for v in TANGENT_WEIGHTS],
        tangent_roughness_after=round(rough_after, 4),
        tangent_roughness_reduction=round(float(roughness_reduction), 4),
        cross_edge_grad_after=round(cross_after_mean, 4),
        cross_edge_gradient_retention=round(float(cross_edge_retention), 4),
        masked_native_pixels=int(edge_mask.sum()),
        changed_native_pixels=changed,
        full_image_sha256=sha(full_png),
        mask_sha256=sha(mask_png),
    )
    return output, report, {'antialiasing-edge-mask.png': mask_png}
