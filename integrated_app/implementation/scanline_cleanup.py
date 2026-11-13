"""Conservative suppression of strongly periodic horizontal scanlines and spatial interlacing comb."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import fft
from scipy.ndimage import binary_dilation, gaussian_filter1d

from .preparation import png, sha


def _spectrum(samples, axis):
    length = samples.shape[axis]
    window = np.hanning(length).astype(np.float32)
    shape = [1, 1]
    shape[axis] = length
    high = samples - gaussian_filter1d(samples, 2, axis=axis)
    power = np.abs(fft.rfft(high * window.reshape(shape), axis=axis)) ** 2
    return fft.rfftfreq(length), power / np.sum(window ** 2)


def detect_scanlines(source):
    """Require a narrow vertical-frequency peak across several image strips."""
    gray = np.asarray(Image.fromarray(source).convert('L'), dtype=np.float32)
    height, width = gray.shape
    if min(height, width) < 64:
        return None
    # Subsample across the stripes without resampling their vertical spacing.
    fy, columns = _spectrum(gray[:, ::max(1, width // 192)], axis=0)
    fx, rows = _spectrum(gray[::max(1, height // 192)], axis=1)
    vertical = columns.mean(axis=1)
    horizontal = rows.mean(axis=0)
    band = fy >= .12
    peak_index = np.flatnonzero(band)[np.argmax(vertical[band])]
    frequency = fy[peak_index]
    if frequency < .135 or vertical[peak_index] < 2:
        return None
    peak_band = np.abs(fy - frequency) <= .014
    perpendicular_band = np.abs(fx - frequency) <= .014
    peak_ratio = vertical[peak_index] / max(.001, np.median(vertical[band]))
    concentration = vertical[peak_band & band].sum() / max(.001, vertical[band].sum())
    directional_ratio = vertical[peak_band].mean() / max(.001, horizontal[perpendicular_band].mean())
    if peak_ratio < 30 or concentration < .48 or directional_ratio < 6:
        return None
    support = 0
    for strip in np.array_split(columns, 6, axis=1):
        profile = strip.mean(axis=1)
        if (profile[peak_index] >= 12 * max(.001, np.median(profile[band])) and
                profile[peak_band & band].sum() >= .30 * profile[band].sum()):
            support += 1
    if support < 3:
        return None
    centers = [float(frequency)]
    harmonic = 2 * frequency
    if harmonic <= .5:
        near = np.flatnonzero(np.abs(fy - harmonic) <= .01)
        if near.size:
            best = near[np.argmax(vertical[near])]
            if vertical[best] >= max(1., 12 * np.median(vertical[band])):
                centers.append(float(fy[best]))
    return {'frequencies_cycles_per_pixel': centers, 'spacing_native_pixels': round(float(1 / frequency), 2),
            'peak_ratio': round(float(peak_ratio), 2), 'band_energy_fraction': round(float(concentration), 4),
            'directional_ratio': round(float(directional_ratio), 2), 'supporting_strips': support}


def prepare_scanlines(source):
    """Suppress the detected bands while retaining the source dimensions."""
    report = detect_scanlines(source)
    if report is None:
        return None, None
    height, width = source.shape[:2]
    pad = min(64, height - 1)
    frequencies = fft.rfftfreq(height + 2 * pad)
    bandwidth = .012
    response = np.ones(frequencies.shape, np.float32)
    for center in report['frequencies_cycles_per_pixel']:
        response *= 1 - .98 * np.exp(-.5 * ((frequencies - center) / bandwidth) ** 2)
    output = source.copy()
    for left in range(0, width, 128):
        block = source[:, left:left + 128].astype(np.float32)
        padded = np.pad(block, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        filtered = fft.irfft(fft.rfft(padded, axis=0) * response[:, None, None], n=len(padded), axis=0)
        output[:, left:left + 128] = np.clip(np.rint(filtered[pad:pad + height]), 0, 255).astype(np.uint8)
    changed = int(np.any(output != source, axis=2).sum())
    if not changed:
        return None, None
    report.update(policy='horizontal-scanline-notch-v1', classification_is_approximate=True,
                  native_size=[width, height], notch_sigma_cycles_per_pixel=bandwidth, changed_native_pixels=changed)
    return output, report


def _bt601_luma(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H, W, 3) to float32 BT.601 luma."""
    arr = rgb.astype(np.float32)
    return arr[..., 0] * np.float32(0.299) + arr[..., 1] * np.float32(0.587) + arr[..., 2] * np.float32(0.114)


def _spatial_comb_energy(luma: np.ndarray) -> np.ndarray:
    """Compute intra-field comb metric M_comb(x, y) suppressing smooth slopes via vertical span."""
    comb = np.zeros_like(luma, dtype=np.float32)
    if luma.shape[0] < 3:
        return comb
    cross = (luma[1:-1, :] - luma[:-2, :]) * (luma[1:-1, :] - luma[2:, :])
    v_diff = ((luma[2:, :] - luma[:-2, :]) ** 2) * np.float32(0.25)
    comb[1:-1, :] = np.maximum(np.float32(0.0), cross - v_diff)
    return comb


def _ela_deinterlace(rgb: np.ndarray, mask_alpha: np.ndarray) -> np.ndarray:
    """Apply 5-tap directional Edge-Directed Line Average (d in {-2..2}) weighted by `mask_alpha`."""
    height, width = rgb.shape[:2]
    if height < 5 or width < 7 or not np.any(mask_alpha > 0.0):
        return rgb.copy()
    rgb_f = rgb.astype(np.float32)
    luma = _bt601_luma(rgb)
    padded_y = np.pad(luma, ((1, 1), (3, 3)), mode='edge')
    padded_rgb = np.pad(rgb_f, ((1, 1), (3, 3), (0, 0)), mode='edge')

    best_cost = np.full((height, width), np.inf, dtype=np.float32)
    best_interp = np.zeros_like(rgb_f)
    for d in (-2, -1, 0, 1, 2):
        cost = np.zeros((height, width), dtype=np.float32)
        for k in (-1, 0, 1):
            top = padded_y[0:height, 3 + k - d : 3 + k - d + width]
            bot = padded_y[2:height + 2, 3 + k + d : 3 + k + d + width]
            cost += np.abs(top - bot)
        cost += abs(d) * np.float32(0.5)
        better = cost < best_cost
        interp = np.float32(0.5) * (
            padded_rgb[0:height, 3 - d : 3 - d + width] + padded_rgb[2:height + 2, 3 + d : 3 + d + width]
        )
        best_cost = np.where(better, cost, best_cost)
        best_interp = np.where(better[..., None], interp, best_interp)

    ela_3tap = np.float32(0.5) * rgb_f + np.float32(0.5) * best_interp
    alpha = np.clip(mask_alpha.astype(np.float32), 0.0, 1.0)[..., None]
    out = rgb_f * (np.float32(1.0) - alpha) + ela_3tap * alpha
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _chroma_comb_cleanup(rgb: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
    """Apply YCbCr Cr/Cb vertical [0.25, 0.5, 0.25] + bilateral filtering along `edge_weight`."""
    if min(rgb.shape[:2]) < 3 or not np.any(edge_weight > 0.0):
        return rgb.copy()
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    weight = np.clip(edge_weight.astype(np.float32), 0.0, 1.0)
    for idx in (1, 2):
        ch = ycrcb[..., idx]
        padded = np.pad(ch, ((1, 1), (0, 0)), mode='edge')
        v3 = np.float32(0.25) * padded[:-2, :] + np.float32(0.50) * padded[1:-1, :] + np.float32(0.25) * padded[2:, :]
        bi = cv2.bilateralFilter(v3, d=5, sigmaColor=14.0, sigmaSpace=2.5)
        ycrcb[..., idx] = ch * (np.float32(1.0) - weight) + bi * weight
    return cv2.cvtColor(np.clip(np.rint(ycrcb), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def clean_residual_scanline_comb(
    cleaned: np.ndarray,
    source: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Suppress residual scanline comb and chroma fringes with masked 5-tap ELA + YCbCr cleanup.

    Replaces unconditional isotropic Gaussian post-blurring after `prepare_scanlines`
    so sharp facial structure is preserved while residual comb edges and color
    fringes are suppressed.
    """
    if not isinstance(cleaned, np.ndarray) or cleaned.ndim != 3 or cleaned.shape[2] != 3 or cleaned.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image array of shape (H, W, 3).')
    if min(cleaned.shape[:2]) < 5 or cleaned.shape[1] < 7:
        return cleaned.copy(), {'comb_ela_masked_pixels': 0, 'chroma_cleanup_pixels': 0, 'changed_native_pixels': 0}

    m_comb = _spatial_comb_energy(_bt601_luma(cleaned))
    comb_alpha = np.clip((m_comb - np.float32(12.0)) / np.float32(48.0), 0.0, 1.0).astype(np.float32)
    comb_alpha = cv2.dilate(comb_alpha, np.ones((3, 1), np.uint8))

    out = _ela_deinterlace(cleaned, comb_alpha)
    if source is not None and isinstance(source, np.ndarray) and source.shape == cleaned.shape:
        notch_diff = np.abs(cleaned.astype(np.float32) - source.astype(np.float32)).max(axis=2)
        chroma_w = np.clip(np.maximum(comb_alpha, notch_diff / np.float32(18.0)), 0.0, 1.0).astype(np.float32)
        ref = source
    else:
        chroma_w = comb_alpha
        ref = cleaned

    out = _chroma_comb_cleanup(out, chroma_w)
    info = {
        'comb_ela_masked_pixels': int((comb_alpha > 0.05).sum()),
        'chroma_cleanup_pixels': int((chroma_w > 0.05).sum()),
        'changed_native_pixels': int(np.any(out != ref, axis=2).sum()),
    }
    return out, info


def detect_and_clean_spatial_comb(
    source: np.ndarray,
    force: bool = False,
) -> tuple[np.ndarray | None, dict | None, dict]:
    """Detect localized spatial intra-field combing and clean via masked 5-tap ELA + chroma suppression."""
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        return None, None, {}
    if min(source.shape[:2]) < 32:
        return None, None, {}

    luma = _bt601_luma(source)
    m_comb = _spatial_comb_energy(luma)
    residual = np.zeros_like(luma)
    residual[1:-1, :] = luma[1:-1, :] - np.float32(0.5) * (luma[:-2, :] + luma[2:, :])
    alt = np.zeros_like(luma)
    alt[1:-2, :] = np.maximum(-(residual[1:-2, :] * residual[2:-1, :]), -(residual[1:-2, :] * residual[:-3, :]))

    comb_mask = (m_comb > 20.0) & (alt > 8.0)
    frac = float(comb_mask.mean())
    run3_mask = comb_mask[:, :-2] & comb_mask[:, 1:-1] & comb_mask[:, 2:]
    run3_frac = float(run3_mask.mean())
    m_before_mean = float(m_comb[1:-1].mean())
    if not force and (m_before_mean < 10.0 or run3_frac < 0.018):
        return None, None, {}

    alpha = np.clip((m_comb - np.float32(12.0)) / np.float32(48.0), 0.0, 1.0).astype(np.float32)
    alpha *= binary_dilation(comb_mask, iterations=1).astype(np.float32)
    out = _ela_deinterlace(source, alpha)
    out = _chroma_comb_cleanup(out, np.clip(alpha * np.float32(1.2), 0.0, 1.0))
    if np.array_equal(out, source):
        return None, None, {}

    m_after_mean = float(_spatial_comb_energy(_bt601_luma(out))[1:-1].mean())
    if not force and m_after_mean > 0.75 * m_before_mean:
        return None, None, {}

    out_png = png(out)
    report = {
        'policy': 'spatial-comb-ela-v1',
        'spacing_native_pixels': 2.0,
        'comb_fraction': round(frac, 4),
        'run3_fraction': round(run3_frac, 4),
        'm_comb_before': round(m_before_mean, 3),
        'm_comb_after': round(m_after_mean, 3),
        'changed_native_pixels': int(np.any(out != source, axis=2).sum()),
        'full_image_sha256': sha(out_png),
    }
    return out, report, {'spatial-comb-cleaned.png': out_png}
