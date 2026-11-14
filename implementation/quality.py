"""Retained input-only restoration suggestion and native-scale noise estimation."""
from __future__ import annotations

import math

import cv2
import numpy as np

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
NATIVE_Y_NOISE_THRESHOLD = 4.0
NATIVE_CHROMA_NOISE_THRESHOLD = 4.5


def native_ycrcb_noise_sigma(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate native-resolution Immerkær MAD noise sigma on (Y, Cr, Cb) without upsampling.

    Args:
        rgb: Native uint8 (or float [0, 255]) RGB array of shape (H, W, 3).
        mask: Optional exclusion mask matching ``quality_signals`` (non-zero = excluded).

    Returns:
        Finite ``(sigma_y, sigma_cr, sigma_cb)`` tuple in 0..255 luma/chroma units.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return (0.0, 0.0, 0.0)
    height, width = arr.shape[:2]
    if min(height, width) < 8:
        return (0.0, 0.0, 0.0)

    if arr.dtype != np.uint8:
        arr_u8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    else:
        arr_u8 = arr
    ycrcb = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)

    if mask is None:
        support = np.ones((height, width), dtype=bool)
    else:
        excl = (np.asarray(mask) > 0).astype(np.uint8)
        if excl.shape[:2] != (height, width):
            return (0.0, 0.0, 0.0)
        support = cv2.erode(1 - excl, np.ones((7, 7), np.uint8)).astype(bool)

    support[:3] = False
    support[-3:] = False
    support[:, :3] = False
    support[:, -3:] = False
    if int(support.sum()) < 256:
        support = np.ones((height, width), dtype=bool)
        if min(height, width) > 8:
            support[:3] = False
            support[-3:] = False
            support[:, :3] = False
            support[:, -3:] = False

    sigmas: list[float] = []
    denom = 6.0 * 0.67448975
    for ch_idx in range(3):
        ch = ycrcb[..., ch_idx]
        low = cv2.GaussianBlur(ch, (7, 7), 1.5)
        grad = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
        thresh = float(np.percentile(grad[support], 35))
        flat = support & (grad <= thresh)
        if not bool(flat.any()):
            flat = support
        resp = cv2.filter2D(ch, cv2.CV_32F, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REPLICATE)
        val = float(np.median(np.abs(resp[flat])) / denom)
        sigmas.append(round(val, 3) if math.isfinite(val) and val >= 0.0 else 0.0)
    return (sigmas[0], sigmas[1], sigmas[2])


def quality_signals(square_rgb, square_mask, native_rgb=None):
    """Exact retained face_workflow.quality_signals math, without its Torch import."""
    rgb = cv2.resize(square_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(square_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    support = cv2.erode(1-mask, np.ones((9, 9), np.uint8)).astype(bool)
    support[:4] = False
    support[-4:] = False
    support[:, :4] = False
    support[:, -4:] = False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    enough = int(support.sum()) >= 512
    if not enough:
        return {'blur_variance': None, 'noise_sigma_255': None, 'suggest_restoration': False,
                'reason': 'Too little visible area for the automatic quality signal; choose an override.',
                'visible_signal_pixels': int(support.sum()), 'blur_threshold': 24, 'noise_threshold': 8}
    smoothed = cv2.GaussianBlur(gray, (3, 3), .6)
    blur = float(cv2.Laplacian(smoothed, cv2.CV_32F)[support].var())
    low = cv2.GaussianBlur(gray, (7, 7), 1.5)
    gradient = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
    flat = support & (gradient <= np.percentile(gradient[support], 35))
    response = cv2.filter2D(gray, cv2.CV_32F, _IMMERKAER_KERNEL)
    noise = float(np.median(np.abs(response[flat])) / (6 * .67448975))
    reasons = []
    if blur < 24:
        reasons.append('blur signal')
    if noise >= 8:
        reasons.append('noise signal')
    result = {'blur_variance': blur, 'noise_sigma_255': noise, 'suggest_restoration': bool(reasons),
              'reason': ', '.join(reasons) if reasons else 'Visible detail does not trigger restoration.',
              'visible_signal_pixels': int(support.sum()), 'blur_threshold': 24, 'noise_threshold': 8}
    if native_rgb is not None:
        sig_y, sig_cr, sig_cb = native_ycrcb_noise_sigma(native_rgb)
        gate_fired = (sig_y >= NATIVE_Y_NOISE_THRESHOLD) or (
            max(sig_cr, sig_cb) >= NATIVE_CHROMA_NOISE_THRESHOLD
        )
        if gate_fired and 'noise signal' not in reasons:
            reasons.append('noise signal')
            result['suggest_restoration'] = True
            result['reason'] = ', '.join(reasons)
        result['sigma_y_cr_cb'] = [sig_y, sig_cr, sig_cb]
        result['noise_sigma_native'] = round(max(sig_y, 0.85 * max(sig_cr, sig_cb)), 3)
        result['noise_gate_triggered'] = bool(gate_fired)
    return result

