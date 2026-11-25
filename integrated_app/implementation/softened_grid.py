"""Search for weak mosaic seams in overlapping luminance windows."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def spectral_grid_with_confidence(gray):
    periods = []
    acs = []
    for axis in (0, 1):
        profile = np.abs(np.diff(gray, axis=axis)).mean(axis=1-axis)
        slope = profile.mean()
        curvature = np.abs(np.diff(gray, n=2, axis=axis)).mean()
        if slope < .4 or curvature > .7 * slope:
            return None, 0.0
        signal = profile - gaussian_filter1d(profile, 5)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=512))
        band = spectrum[26:103]
        index = int(np.argmax(band)) + 26
        peak = spectrum[index]
        if peak < max(3.5 * np.median(band), .28 * slope * len(profile) / 4):
            return None, 0.0
        period = 512 / index
        lag = round(period)
        overlap = np.linalg.norm(signal[:-lag]) * np.linalg.norm(signal[lag:])
        if overlap == 0:
            return None, 0.0
        ac = float(np.dot(signal[:-lag], signal[lag:]) / overlap)
        if ac < .32:
            return None, 0.0
        periods.append(period)
        acs.append(ac)
    if max(periods) <= 1.2 * min(periods):
        return periods, min(acs)
    return None, 0.0


def spectral_grid(gray):
    periods, _ = spectral_grid_with_confidence(gray)
    return periods


def softened_mask_with_confidence(gray):
    """Require agreement among overlapping windows in a single grid band and return confidence."""
    side = 96
    step = 16
    height, width = gray.shape
    votes = np.zeros((height, width), dtype=np.uint16)
    support = {}
    for y in range(0, height - side + 1, step):
        for x in range(0, width - side + 1, step):
            periods, ac = spectral_grid_with_confidence(gray[y:y + side, x:x + side])
            if periods is not None:
                key = round(sum(periods) / 2)
                support.setdefault(key, []).append((x, y, sum(periods) / 2.0, ac))
    if not support:
        return votes > 0, None, 0.0
    period_key = max(support, key=lambda p: sum(len(support.get(q, ())) for q in (p-1, p, p+1)))
    matched_periods = []
    matched_acs = []
    for p in (period_key-1, period_key, period_key+1):
        for x, y, per, ac in support.get(p, ()):
            votes[y + 16:y + side - 16, x + 16:x + side - 16] += 1
            matched_periods.append(per)
            matched_acs.append(ac)
    mask = votes >= 2
    if not mask.any():
        return mask, None, 0.0
    return mask, float(np.median(matched_periods)), float(np.median(matched_acs))


def softened_mask(gray):
    """Require agreement among overlapping windows in a single grid band."""
    mask, _, _ = softened_mask_with_confidence(gray)
    return mask
