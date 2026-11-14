"""Conservative detector and constrained multi-scale solver for color banding."""

import cv2
import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, find_objects,
                           gaussian_filter, label, uniform_filter)

from .irregular_mosaic import irregular_mosaic_mask
from .preparation import png

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Deterministic 8x8 Bayer ordered dithering matrix centered in [-0.492, +0.492] LSB.
_BAYER8 = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    + 0.5
) / 64.0 - 0.5


def _axis_has_periodic_grid(frac_1d):
    """Return True when a 1D boundary-fraction profile has uniform periodic spikes."""
    if frac_1d.size < 16:
        return False
    left = np.r_[True, frac_1d[:-1] <= frac_1d[1:]]
    right = np.r_[frac_1d[1:] < frac_1d[:-1], True]
    peaks = np.flatnonzero((frac_1d > 0.40) & (left | right))
    if peaks.size < 4:
        return False
    gaps = np.diff(peaks)
    gaps = gaps[gaps >= 4]
    if gaps.size < 3:
        return False
    median_gap = float(np.median(gaps))
    if not (4.0 <= median_gap <= 48.0):
        return False
    regular = float((np.abs(gaps - median_gap) <= 1.5).mean()) >= 0.70
    quiet_interior = float(np.median(frac_1d)) < 0.35 * float(np.median(frac_1d[peaks]))
    return regular and quiet_interior


def _has_orthogonal_mosaic_grid(luma):
    """Detect uniform 2D orthogonal privacy mosaic grids without rejecting 1D ramps."""
    dh_frac = (np.abs(np.diff(luma, axis=1)) > 1.0).mean(axis=0)
    dv_frac = (np.abs(np.diff(luma, axis=0)) > 1.0).mean(axis=1)
    return _axis_has_periodic_grid(dh_frac) and _axis_has_periodic_grid(dv_frac)


def _compute_step_energy(img_f32, edge_h_mask, edge_v_mask):
    """Measure discrete contour-step energy (>= 1.5 LSB transitions) inside smooth mask."""
    dh = np.abs(np.diff(img_f32, axis=1)).max(-1)[edge_h_mask]
    dv = np.abs(np.diff(img_f32, axis=0)).max(-1)[edge_v_mask]
    diffs = np.concatenate([dh, dv])
    if not diffs.size:
        return 0.0, 0.0
    step_energy = float(np.mean(np.where(diffs >= 1.5, diffs, 0.0)))
    step_rate = float(np.mean(diffs >= 1.5))
    return step_energy, step_rate


def _analyze_banding(source):
    """Analyze uint8 RGB image for quantized color banding and build smooth-region state."""
    arr = np.asarray(source)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        return None
    height, width = arr.shape[:2]
    if min(height, width) < 32:
        return None

    f32 = arr.astype(np.float32)
    s16 = arr.astype(np.int16)
    luma = f32 @ _LUMA_WEIGHTS

    if _has_orthogonal_mosaic_grid(luma):
        return None

    dh_rgb = np.abs(np.diff(s16, axis=1)).max(-1)
    dv_rgb = np.abs(np.diff(s16, axis=0)).max(-1)
    grad_rgb = np.zeros((height, width), dtype=np.int16)
    grad_rgb[:, :-1] = np.maximum(grad_rgb[:, :-1], dh_rgb)
    grad_rgb[:, 1:] = np.maximum(grad_rgb[:, 1:], dh_rgb)
    grad_rgb[:-1, :] = np.maximum(grad_rgb[:-1, :], dv_rgb)
    grad_rgb[1:, :] = np.maximum(grad_rgb[1:, :], dv_rgb)

    dh_y = np.abs(np.diff(luma, axis=1))
    dv_y = np.abs(np.diff(luma, axis=0))
    grad_y = np.zeros((height, width), dtype=np.float32)
    grad_y[:, :-1] = np.maximum(grad_y[:, :-1], dh_y)
    grad_y[:, 1:] = np.maximum(grad_y[:, 1:], dh_y)
    grad_y[:-1, :] = np.maximum(grad_y[:-1, :], dv_y)
    grad_y[1:, :] = np.maximum(grad_y[1:, :], dv_y)

    # Support both RGB quantization plateaus and Y-only quantized plateaus (where chroma rounds by 1-2 LSB).
    plateau_px = (grad_rgb <= 1) | ((grad_y <= 1.05) & (grad_rgb <= 2))
    strong_edge = grad_rgb > 38
    plateau_ratio_7 = uniform_filter(plateau_px.astype(np.float32), size=7, mode="nearest")
    smooth_base = plateau_px | ((grad_rgb <= 40) & (plateau_ratio_7 >= 0.45))

    if float(plateau_px.mean()) < 0.08 or float(smooth_base.mean()) < 0.12:
        return None

    # Exclude irregular polygonal/Voronoi privacy mosaics cleanly.
    if float(irregular_mosaic_mask(arr).mean()) >= 0.04:
        return None

    sm_h = smooth_base[:, :-1] & smooth_base[:, 1:] & (~strong_edge[:, :-1]) & (~strong_edge[:, 1:])
    sm_v = smooth_base[:-1, :] & smooth_base[1:, :] & (~strong_edge[:-1, :]) & (~strong_edge[1:, :])
    if int(sm_h.sum()) + int(sm_v.sum()) < 256:
        return None

    all_diffs = np.concatenate([dh_rgb[sm_h], dv_rgb[sm_v]])
    all_ly = np.concatenate([dh_y[sm_h], dv_y[sm_v]])
    rgb_zero_frac = float((all_diffs == 0).mean())
    y_zero_frac = float((all_ly < 0.55).mean())
    plateau_flat_frac = max(rgb_zero_frac, y_zero_frac)

    step_mask = (all_diffs >= 2) & (all_diffs <= 40)
    steps = all_diffs[step_mask]
    step_frac = float(step_mask.mean())
    if steps.size < max(64, int(all_diffs.size * 0.004)) or step_frac < 0.005 or plateau_flat_frac < 0.45:
        return None

    # Reject smooth continuous 8-bit gradients (diff==1 dominates and discrete plateau-flanked jumps are absent).
    diff1_count = int((all_diffs == 1).sum())
    step3_count = int(((all_diffs >= 3) & (all_diffs <= 40)).sum())
    step3_frac = float(step3_count / all_diffs.size)

    # Check 2..40 LSB luma jumps flanked on both sides by flat luminance (< 0.55 LSB).
    left_flat_h = np.ones_like(sm_h)
    left_flat_h[:, 1:] = dh_y[:, :-1] < 0.55
    right_flat_h = np.ones_like(sm_h)
    right_flat_h[:, :-1] = dh_y[:, 1:] < 0.55
    both_h = sm_h & (dh_rgb >= 2) & (dh_rgb <= 40) & (dh_y >= 1.5) & left_flat_h & right_flat_h

    up_flat_v = np.ones_like(sm_v)
    up_flat_v[1:, :] = dv_y[:-1, :] < 0.55
    down_flat_v = np.ones_like(sm_v)
    down_flat_v[:-1, :] = dv_y[1:, :] < 0.55
    both_v = sm_v & (dv_rgb >= 2) & (dv_rgb <= 40) & (dv_y >= 1.5) & up_flat_v & down_flat_v

    both_count = int(both_h.sum()) + int(both_v.sum())
    has_quantized_jumps = (step3_frac >= 0.005 and step3_count >= 0.22 * diff1_count) or (
        both_count >= max(64, int(steps.size * 0.25))
    )
    if not has_quantized_jumps:
        return None

    # Protect strong structural edges (dilated 3px) and fine natural texture islands.
    edge_protect = binary_dilation(strong_edge, iterations=3)
    tex_protect = binary_dilation(plateau_ratio_7 < 0.40, iterations=1)
    step_px = ((grad_rgb >= 2) & (grad_rgb <= 40)) & (~edge_protect) & (~tex_protect)
    cand_bin = (plateau_px | step_px) & (~edge_protect) & (~tex_protect)
    cand_bin &= binary_dilation(step_px, iterations=20)

    components, _ = label(cand_bin)
    keep = np.zeros_like(cand_bin)
    for idx, bounds in enumerate(find_objects(components), 1):
        if bounds is None:
            continue
        local = components[bounds] == idx
        if int(local.sum()) < 64 or int((local & step_px[bounds]).sum()) < 10:
            continue
        keep[bounds] |= local

    if float(keep.mean()) < 0.02:
        return None

    # Ensure multiple distinct tone bands (reject single crisp 2-tone or 3-tone flat graphics).
    plateau_luma_levels = np.unique(np.round(luma[keep & plateau_px] / 2.0))
    if plateau_luma_levels.size < 4 or int((keep & step_px).sum()) < 24:
        return None

    hard_forbidden = edge_protect | tex_protect | (~keep)
    soft = gaussian_filter(keep.astype(np.float32), sigma=1.5, mode="nearest")
    soft[hard_forbidden] = 0.0
    soft = np.clip(
        soft * gaussian_filter((~edge_protect).astype(np.float32), sigma=1.5, mode="nearest"),
        0.0,
        1.0,
    )
    interior = binary_erosion(keep & (~edge_protect) & (~tex_protect), iterations=2)
    soft[interior] = np.maximum(soft[interior], 0.999)
    soft[edge_protect] = 0.0

    eval_mask = soft > 0.5
    if int(eval_mask.sum()) < 64:
        return None
    eval_h = eval_mask[:, :-1] & eval_mask[:, 1:]
    eval_v = eval_mask[:-1, :] & eval_mask[1:, :]
    before_energy, before_rate = _compute_step_energy(f32, eval_h, eval_v)
    if before_energy <= 1e-6 or before_rate < 0.004:
        return None

    # F2 tightened step gate: reject subtle <=3-LSB studio vignette rounding when >=4-LSB steps are absent.
    all_eval_diffs = np.concatenate([dh_rgb[eval_h], dv_rgb[eval_v]])
    steps3 = all_eval_diffs[(all_eval_diffs >= 3) & (all_eval_diffs <= 40)]
    if steps3.size == 0:
        return None
    p90_step3 = float(np.percentile(steps3, 90))
    step4_frac = float(((all_eval_diffs >= 4) & (all_eval_diffs <= 40)).mean())
    if not (np.isfinite(p90_step3) and np.isfinite(step4_frac)):
        return None
    if p90_step3 < 4.0 and step4_frac < 0.002:
        return None

    report = {
        "suspected": True,
        "method": "quantized-plateau-contour-v1",
        "smooth_mask_fraction": float(round(float((soft > 0.1).mean()), 6)),
        "plateau_flat_fraction": float(round(plateau_flat_frac, 6)),
        "contour_step_fraction": float(round(step_frac, 6)),
        "median_step_lsb": float(round(float(np.median(steps)), 4)),
        "before_step_energy": float(round(before_energy, 6)),
        "plateau_tone_levels": int(plateau_luma_levels.size),
    }
    return {
        "arr": arr,
        "f32": f32,
        "luma": luma,
        "strong_edge": strong_edge,
        "soft": soft,
        "eval_h": eval_h,
        "eval_v": eval_v,
        "before_energy": before_energy,
        "report": report,
    }


def detect_banding(source):
    """Detect quantized tonal color banding without triggering on textures or mosaics."""
    analysis = _analyze_banding(source)
    return None if analysis is None else dict(analysis["report"])


def _normalized_convolution_multiscale(f32, soft):
    """Interpolate flat plateaus across bands strictly inside the smooth mask."""
    mask3 = soft[..., None]
    accum = np.zeros_like(f32)
    sigmas = (3.5, 8.0, 16.0)
    for sigma in sigmas:
        num = gaussian_filter(f32 * mask3, sigma=(sigma, sigma, 0.0), mode="reflect")
        den = gaussian_filter(mask3, sigma=(sigma, sigma, 0.0), mode="reflect")
        accum += num / (den + 1e-6)
    return accum / float(len(sigmas))


def _edge_gradient_retention(luma_in, luma_out, strong_edge):
    """Compute retained strong-edge gradient ratio on grad > 38 boundaries."""
    if not bool(strong_edge.any()):
        return 1.0
    eh = strong_edge[:, :-1] | strong_edge[:, 1:]
    ev = strong_edge[:-1, :] | strong_edge[1:, :]
    gx0 = np.abs(np.diff(luma_in, axis=1))
    gy0 = np.abs(np.diff(luma_in, axis=0))
    orig = float(gx0[eh].sum() + gy0[ev].sum())
    if orig <= 1e-6:
        return 1.0
    gx1 = np.abs(np.diff(luma_out, axis=1))
    gy1 = np.abs(np.diff(luma_out, axis=0))
    return float((gx1[eh].sum() + gy1[ev].sum()) / orig)


def prepare_debanding(source):
    """Suppress color banding via mask-constrained normalized convolution and Bayer dither."""
    analysis = _analyze_banding(source)
    if analysis is None:
        return None, None, {}

    arr = analysis["arr"]
    f32 = analysis["f32"]
    luma_in = analysis["luma"]
    soft = analysis["soft"]
    height, width = arr.shape[:2]

    smooth_interp = _normalized_convolution_multiscale(f32, soft)
    mask3 = soft[..., None]
    dither_tile = np.tile(_BAYER8, ((height + 7) // 8, (width + 7) // 8))[:height, :width][..., None]
    dithered = np.where(mask3 > 0.1, f32 * (1.0 - mask3) + (smooth_interp + dither_tile) * mask3, f32)
    out_u8 = np.clip(np.round(dithered), 0, 255).astype(np.uint8)
    if np.array_equal(out_u8, arr):
        return None, None, {}

    out_f32 = out_u8.astype(np.float32)
    after_energy, _ = _compute_step_energy(out_f32, analysis["eval_h"], analysis["eval_v"])
    before_energy = analysis["before_energy"]
    step_reduction = float((before_energy - after_energy) / max(before_energy, 1e-6))

    luma_out = out_f32 @ _LUMA_WEIGHTS
    edge_retention = _edge_gradient_retention(luma_in, luma_out, analysis["strong_edge"])

    active = soft > 0.1
    if bool(active.any()):
        lap_in = np.abs(cv2.Laplacian(luma_in, cv2.CV_32F)) / 4.0
        lap_out = np.abs(cv2.Laplacian(luma_out, cv2.CV_32F)) / 4.0
        lap_delta = float(lap_out[active].mean() - lap_in[active].mean())
    else:
        lap_delta = 0.0

    # Quantitative acceptance gate: >=30% step reduction, >=95% edge retention, <=0.6 LSB noise delta.
    if step_reduction < 0.30 or edge_retention < 0.95 or lap_delta > 0.60:
        return None, None, {}

    report = dict(analysis["report"])
    report.update(
        {
            "after_step_energy": float(round(after_energy, 6)),
            "step_reduction": float(round(step_reduction, 6)),
            "edge_retention": float(round(edge_retention, 6)),
            "laplacian_noise_delta": float(round(lap_delta, 6)),
            "dither_pattern": "bayer-8x8-0.5lsb",
            "applied": True,
        }
    )
    if not all(np.isfinite(float(v)) for v in report.values() if isinstance(v, (int, float)) and not isinstance(v, bool)):
        return None, None, {}

    mask_u8 = np.clip(np.round(soft * 255.0), 0, 255).astype(np.uint8)
    return out_u8, report, {"debanding-mask.png": png(mask_u8)}
