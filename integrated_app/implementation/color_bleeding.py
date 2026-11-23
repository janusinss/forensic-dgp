"""Conservative removal of 4:2:0/chroma-smeared color bleeding along sharp luma edges."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label, map_coordinates

from .preparation import png, sha

POLICY = 'luma-guided-chroma-debleed-v1'
MIN_SIDE = 32
JBF_RADIUS = 3
JBF_SPATIAL_SIGMA = 2.0
JBF_LUMA_SIGMA = 10.0
JBF_PASSES = 2


def _as_rgb_uint8(source):
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        return None
    height, width = source.shape[:2]
    if min(height, width) < MIN_SIDE:
        return None
    return source


def _to_ycrcb_f32(rgb_uint8):
    ycrcb = cv2.cvtColor(rgb_uint8.astype(np.float32) / 255.0, cv2.COLOR_RGB2YCrCb) * 255.0
    return ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]


def _scharr_grad(channel):
    gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1) / 32.0
    return gx, gy, np.hypot(gx, gy)


def _joint_bilateral_roi(y_roi, cr_roi, cb_roi, radius=JBF_RADIUS,
                         sigma_s=JBF_SPATIAL_SIGMA, sigma_y=JBF_LUMA_SIGMA):
    height, width = y_roi.shape
    y_pad = np.pad(y_roi, radius, mode='reflect')
    cr_pad = np.pad(cr_roi, radius, mode='reflect')
    cb_pad = np.pad(cb_roi, radius, mode='reflect')
    w_sum = np.zeros_like(y_roi, dtype=np.float32)
    cr_sum = np.zeros_like(cr_roi, dtype=np.float32)
    cb_sum = np.zeros_like(cb_roi, dtype=np.float32)
    inv_2ss = -0.5 / (sigma_s * sigma_s)
    inv_2sy = -0.5 / (sigma_y * sigma_y)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            w_spatial = np.float32(math.exp((dy * dy + dx * dx) * inv_2ss))
            ys = slice(radius + dy, radius + dy + height)
            xs = slice(radius + dx, radius + dx + width)
            diff_y = y_pad[ys, xs] - y_roi
            w = w_spatial * np.exp((diff_y * diff_y) * inv_2sy)
            w_sum += w
            cr_sum += w * cr_pad[ys, xs]
            cb_sum += w * cb_pad[ys, xs]
    denom = np.maximum(w_sum, np.float32(1e-6))
    return cr_sum / denom, cb_sum / denom


def _analyze_color_bleeding(source):
    rgb = _as_rgb_uint8(source)
    if rgb is None:
        return None
    height, width = rgb.shape[:2]
    y_ch, cr_ch, cb_ch = _to_ycrcb_f32(rgb)
    # Reject monochrome / nearly achromatic images early.
    if float(np.hypot(cr_ch - 127.5, cb_ch - 127.5).max()) < 6.0:
        return None

    gx_y, gy_y, g_y = _scharr_grad(y_ch)
    _, _, g_cr = _scharr_grad(cr_ch)
    _, _, g_cb = _scharr_grad(cb_ch)
    g_c = np.hypot(g_cr, g_cb)

    inv_gy = 1.0 / np.maximum(g_y, np.float32(1e-4))
    nx = gx_y * inv_gy
    ny = gy_y * inv_gy
    yy_idx, xx_idx = np.indices((height, width), dtype=np.float32)
    gp = map_coordinates(g_y, [yy_idx + ny, xx_idx + nx], order=1, mode='nearest', prefilter=False)
    gm = map_coordinates(g_y, [yy_idx - ny, xx_idx - nx], order=1, mode='nearest', prefilter=False)
    ridge = (g_y >= 12.0) & (g_y >= gp - 1e-3) & (g_y >= gm - 1e-3)
    ridge[:4] = ridge[-4:] = ridge[:, :4] = ridge[:, -4:] = False
    ry, rx = np.nonzero(ridge)
    if ry.size < 10:
        return None

    ds = np.arange(-4, 5, dtype=np.float32)[:, None]
    rny = ny[ry, rx][None, :]
    rnx = nx[ry, rx][None, :]
    sy = ry[None, :].astype(np.float32) + ds * rny
    sx = rx[None, :].astype(np.float32) + ds * rnx
    py = map_coordinates(y_ch, [sy, sx], order=1, mode='nearest', prefilter=False)
    pcr = map_coordinates(cr_ch, [sy, sx], order=1, mode='nearest', prefilter=False)
    pcb = map_coordinates(cb_ch, [sy, sx], order=1, mode='nearest', prefilter=False)

    dcr_step = pcr[-1] - pcr[0]
    dcb_step = pcb[-1] - pcb[0]
    c_step = np.hypot(dcr_step, dcb_step)
    dy_step = py[-1] - py[0]
    y_step = np.abs(dy_step)
    dy_all = np.diff(py, axis=0)
    dc_all = np.hypot(np.diff(pcr, axis=0), np.diff(pcb, axis=0))

    # Require single monotonic transitions so nearby third regions do not mimic smear.
    mono_y = np.abs(dy_all).sum(axis=0) <= 1.25 * y_step + 4.0
    mono_c = dc_all.sum(axis=0) <= 1.45 * c_step + 6.0
    crisp_luma = np.abs(dy_all).max(axis=0) >= 0.28 * y_step
    valid = (c_step >= 10.0) & (y_step >= 16.0) & crisp_luma & mono_y & mono_c
    if int(valid.sum()) < 10:
        return None

    ry, rx = ry[valid], rx[valid]
    py, pcr, pcb = py[:, valid], pcr[:, valid], pcb[:, valid]
    c_step = c_step[valid]
    uy = np.sign(dy_step[valid])[None, :]
    ucr = (dcr_step[valid] / c_step)[None, :]
    ucb = (dcb_step[valid] / c_step)[None, :]
    dy_dir = np.maximum(0.0, np.diff(py, axis=0) * uy)
    dc_dir = np.maximum(0.0, np.diff(pcr, axis=0) * ucr + np.diff(pcb, axis=0) * ucb)

    offs = np.abs(np.arange(-3.5, 4.0, 1.0, dtype=np.float32))[:, None]
    w_y = (offs * dy_dir).sum(axis=0) / np.maximum(dy_dir.sum(axis=0), 1e-4)
    w_c = (offs * dc_dir).sum(axis=0) / np.maximum(dc_dir.sum(axis=0), 1e-4)
    flank_idx = np.array([0, 1, 2, 5, 6, 7])
    luma_flat = dy_dir[flank_idx] <= np.maximum(2.5, 0.15 * dy_dir.max(axis=0)[None, :])
    spill = (dc_dir[flank_idx] * luma_flat).sum(axis=0) / np.maximum(dc_dir.sum(axis=0), 1e-4)
    is_bleed = (w_c - w_y >= 0.20) & (spill >= 0.23)

    all_map = np.zeros((height, width), dtype=bool)
    bleed_map = np.zeros((height, width), dtype=bool)
    all_map[ry, rx] = True
    bleed_map[ry[is_bleed], rx[is_bleed]] = True

    dilated = binary_dilation(bleed_map, structure=np.ones((5, 5), dtype=bool))
    components, count = label(dilated)
    kept_ridge = np.zeros((height, width), dtype=bool)
    min_pts = max(8, min(14, round(0.25 * min(height, width))))
    for cid in range(1, count + 1):
        comp_mask = components == cid
        n_bleed = int((comp_mask & bleed_map).sum())
        n_all = int((comp_mask & all_map).sum())
        if n_bleed >= min_pts and n_bleed >= 0.50 * max(1, n_all):
            kept_ridge |= (comp_mask & bleed_map)

    n_kept = int(kept_ridge.sum())
    bleed_edge_fraction = float(n_kept / max(1, int(ry.size)))
    min_bleed_edge_px = 14 if min(height, width) < 64 else 24
    if n_kept < min_bleed_edge_px or bleed_edge_fraction < 0.25:
        return None

    kept_sel = kept_ridge[ry, rx]
    med_wy = float(np.median(w_y[kept_sel]))
    med_wc = float(np.median(w_c[kept_sel]))
    med_spill = float(np.median(spill[kept_sel]))
    if med_wc - med_wy < 0.22 or med_spill < 0.24:
        return None

    dist = distance_transform_edt(~kept_ridge)
    alpha = np.clip((4.5 - dist) / 1.5, 0.0, 1.0).astype(np.float32)
    bleed_mask = alpha > 0.0
    if not bleed_mask.any() or float(bleed_mask.mean()) < 0.005:
        return None

    flank_eval = bleed_mask & (dist >= 1.0) & (dist <= 3.2) & (g_y < 8.0)
    if int(flank_eval.sum()) < 8:
        flank_eval = bleed_mask & (g_y < 8.0) & (~kept_ridge)
    if int(flank_eval.sum()) < 8:
        return None
    flank_gc_before = float(g_c[flank_eval].mean())
    if flank_gc_before < 0.5:
        return None

    luma_eval = binary_dilation(kept_ridge, structure=np.ones((3, 3), dtype=bool)) & (g_y >= 10.0)
    if int(luma_eval.sum()) < 8:
        luma_eval = kept_ridge

    report = {
        'suspected': True,
        'classification_is_approximate': True,
        'native_size': [int(width), int(height)],
        'chromatic_edge_pixels': int(ry.size),
        'bleeding_edge_pixels': int(n_kept),
        'bleeding_edge_fraction': round(float(n_kept / max(1, ry.size)), 4),
        'luma_edge_width': round(med_wy, 4),
        'chroma_edge_width': round(med_wc, 4),
        'width_excess': round(float(med_wc - med_wy), 4),
        'off_edge_spill_ratio': round(med_spill, 4),
        'off_edge_chroma_grad_before': round(flank_gc_before, 4),
    }
    return {
        'report': report,
        'y': y_ch,
        'cr': cr_ch,
        'cb': cb_ch,
        'g_y': g_y,
        'alpha': alpha,
        'bleed_mask': bleed_mask,
        'flank_eval': flank_eval,
        'luma_eval': luma_eval,
    }


def detect_color_bleeding(source):
    """Detect chroma gradient spill into luma-flat flanks around crisp chromatic edges."""
    analysis = _analyze_color_bleeding(source)
    return None if analysis is None else analysis['report']


def prepare_color_bleeding(source):
    """Apply soft luma-guided joint bilateral filtering to Cr/Cb in bleeding bands."""
    analysis = _analyze_color_bleeding(source)
    if analysis is None:
        return None, None, {}

    rgb = source
    height, width = rgb.shape[:2]
    y_ch = analysis['y']
    cr_ch = analysis['cr']
    cb_ch = analysis['cb']
    alpha = analysis['alpha']
    bleed_mask = analysis['bleed_mask']
    flank_eval = analysis['flank_eval']
    luma_eval = analysis['luma_eval']

    my, mx = np.nonzero(bleed_mask)
    pad = JBF_RADIUS * JBF_PASSES + 2
    y0 = max(0, int(my.min()) - pad)
    y1 = min(height, int(my.max()) + pad + 1)
    x0 = max(0, int(mx.min()) - pad)
    x1 = min(width, int(mx.max()) + pad + 1)

    y_roi = y_ch[y0:y1, x0:x1]
    cr_roi = cr_ch[y0:y1, x0:x1].copy()
    cb_roi = cb_ch[y0:y1, x0:x1].copy()
    for _ in range(JBF_PASSES):
        cr_roi, cb_roi = _joint_bilateral_roi(y_roi, cr_roi, cb_roi)

    a_roi = alpha[y0:y1, x0:x1]
    cr_out = cr_ch.copy()
    cb_out = cb_ch.copy()
    cr_out[y0:y1, x0:x1] = cr_ch[y0:y1, x0:x1] * (1.0 - a_roi) + cr_roi * a_roi
    cb_out[y0:y1, x0:x1] = cb_ch[y0:y1, x0:x1] * (1.0 - a_roi) + cb_roi * a_roi

    ycrcb_out = np.stack([y_ch, cr_out, cb_out], axis=-1) / 255.0
    rgb_clean = cv2.cvtColor(ycrcb_out.astype(np.float32), cv2.COLOR_YCrCb2RGB) * 255.0
    output = np.clip(np.rint(rgb_clean), 0.0, 255.0).astype(np.uint8)
    output[~bleed_mask] = rgb[~bleed_mask]

    changed = int(np.any(output != rgb, axis=2).sum())
    if changed == 0:
        return None, None, {}

    y_after, cr_after, cb_after = _to_ycrcb_f32(output)
    _, _, g_y_after = _scharr_grad(y_after)
    _, _, g_cr_after = _scharr_grad(cr_after)
    _, _, g_cb_after = _scharr_grad(cb_after)
    g_c_after = np.hypot(g_cr_after, g_cb_after)

    flank_gc_before = float(analysis['report']['off_edge_chroma_grad_before'])
    flank_gc_after = float(g_c_after[flank_eval].mean())
    bleed_reduction = 1.0 - (flank_gc_after / max(flank_gc_before, 1e-6))

    gy_before_mean = float(analysis['g_y'][luma_eval].mean())
    gy_after_mean = float(g_y_after[luma_eval].mean())
    luma_grad_retention = gy_after_mean / max(gy_before_mean, 1e-6)
    mean_y_diff = float(np.abs(y_after - y_ch).mean())
    mask_y_diff = float(np.abs(y_after[bleed_mask] - y_ch[bleed_mask]).mean())

    if (
        not math.isfinite(bleed_reduction)
        or not math.isfinite(luma_grad_retention)
        or not math.isfinite(mean_y_diff)
        or not math.isfinite(mask_y_diff)
        or bleed_reduction < 0.25
        or luma_grad_retention < 0.98
        or mean_y_diff > 1.0
        or mask_y_diff > 1.0
    ):
        return None, None, {}

    mask_u8 = bleed_mask.astype(np.uint8) * 255
    full_png = png(output)
    mask_png = png(mask_u8)
    report = dict(analysis['report'])
    report.update(
        policy=POLICY,
        jbf_window=2 * JBF_RADIUS + 1,
        jbf_spatial_sigma=float(JBF_SPATIAL_SIGMA),
        jbf_luma_sigma=float(JBF_LUMA_SIGMA),
        jbf_passes=int(JBF_PASSES),
        off_edge_chroma_grad_after=round(flank_gc_after, 4),
        off_edge_bleed_reduction=round(float(bleed_reduction), 4),
        luma_gradient_retention=round(float(luma_grad_retention), 4),
        mean_luma_abs_diff=round(mean_y_diff, 4),
        mask_luma_abs_diff=round(mask_y_diff, 4),
        masked_native_pixels=int(bleed_mask.sum()),
        changed_native_pixels=changed,
        full_image_sha256=sha(full_png),
        mask_sha256=sha(mask_png),
    )
    return output, report, {'color-bleeding-mask.png': mask_png}
