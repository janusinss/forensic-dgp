"""Search for weak mosaic seams in overlapping luminance windows."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def spectral_grid(gray):
    periods = []
    for axis in (0, 1):
        profile = np.abs(np.diff(gray, axis=axis)).mean(axis=1-axis)
        slope = profile.mean()
        curvature = np.abs(np.diff(gray, n=2, axis=axis)).mean()
        if slope < .4 or curvature > .7 * slope:
            return None
        signal = profile - gaussian_filter1d(profile, 5)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=512))
        band = spectrum[26:103]
        index = int(np.argmax(band)) + 26
        peak = spectrum[index]
        if peak < max(3.5 * np.median(band), .28 * slope * len(profile) / 4):
            return None
        period = 512 / index
        lag = round(period)
        overlap = np.linalg.norm(signal[:-lag]) * np.linalg.norm(signal[lag:])
        if overlap == 0 or np.dot(signal[:-lag], signal[lag:]) < .32 * overlap:
            return None
        periods.append(period)
    if max(periods) <= 1.2 * min(periods):
        return periods
    return None


def softened_mask(gray):
    """Require agreement among overlapping windows in a single grid band."""
    side = 96
    step = 16
    height, width = gray.shape
    votes = np.zeros((height, width), dtype=np.uint16)
    support = {}
    for y in range(0, height - side + 1, step):
        for x in range(0, width - side + 1, step):
            periods = spectral_grid(gray[y:y + side, x:x + side])
            if periods is not None:
                key = round(sum(periods) / 2)
                support.setdefault(key, []).append((x, y))
    if not support:
        return votes > 0
    period = max(support, key=lambda p: sum(len(support.get(q, ())) for q in (p-1, p, p+1)))
    for p in (period-1, period, period+1):
        for x, y in support.get(p, ()):
            votes[y + 16:y + side - 16, x + 16:x + side - 16] += 1
    return votes >= 2
