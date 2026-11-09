"""Conservative suppression of strongly periodic horizontal scanlines."""

import numpy as np
from PIL import Image
from scipy import fft
from scipy.ndimage import gaussian_filter1d


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
