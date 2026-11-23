"""Experimental local detection for skewed, nearly periodic mosaic seams."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import binary_closing, label, map_coordinates, sobel


@lru_cache(maxsize=4)
def line_coordinates(side):
    """Map bands parallel to each candidate seam into the local window."""
    half = side // 2
    along = np.arange(-half + 3, half - 2, 2, dtype=np.float32)
    across = np.arange(-half + 4, half - 3, dtype=np.float32)
    slopes = np.arange(-.60, .61, .05, dtype=np.float32)
    a = np.broadcast_to(along, (len(slopes), len(across), len(along)))
    b = across[None, :, None] + slopes[:, None, None] * along[None, None, :]
    return slopes, a + half, b + half


def family(gradient, side):
    slopes, a, b = line_coordinates(side)
    middle = a.shape[-1] // 2
    stripes = map_coordinates(gradient, [b, a], order=1, mode="nearest", prefilter=False)
    left, right = stripes[..., :middle].mean(-1), stripes[..., middle:].mean(-1)
    length = left.shape[-1]
    fft_size = 4 * length
    window = np.hanning(length)
    f1 = np.fft.rfft((left - left.mean(-1, keepdims=True)) * window, n=fft_size)
    f2 = np.fft.rfft((right - right.mean(-1, keepdims=True)) * window, n=fft_size)
    # A seam should continue across the window with the same phase in each half.
    cross = np.maximum(0., (f1 * f2.conj()).real)
    first = int(np.ceil(fft_size / (length / 4)))
    last = int(fft_size / 3.8)
    band = cross[:, first:last + 1]
    if not band.size:
        return None
    power = band / np.maximum(.01, (left.mean(-1) * right.mean(-1))[:, None])
    winners = np.argsort(power.ravel())[-8:][::-1]
    for winner in winners:
        s, k = np.unravel_index(winner, power.shape)
        frequency = k + first
        period = fft_size / frequency
        phase = -np.angle(f1[s, frequency] + f2[s, frequency]) * period / (2 * np.pi)
        centers = (phase % period) + period * np.arange(-1, 1 + int(length / period))
        centers = centers[(centers >= .5) & (centers < length - .5)]
        if centers.size < 4:
            continue
        halfway = .5 * (centers[1:] + centers[:-1])
        samples = np.arange(length)
        ratios = []
        for profile in (left[s], right[s]):
            seam = np.median(np.interp(centers, samples, profile))
            interior = np.median(np.interp(halfway, samples, profile))
            ratios.append((seam + .5) / (interior + .5))
        score = min(ratios)
        if score >= 2.2:
            return float(slopes[s]), period, score, phase % period + 4
    return None


def cell_evidence(rgb, rows, columns):
    """Describe flat interiors and the diversity of their colours."""
    half = rgb.shape[0] // 2
    y, x = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    dy = (y - rows[3] - rows[0] * (x - half)) % rows[1]
    dx = (x - columns[3] - columns[0] * (y - half)) % columns[1]
    interior = ((dy > .30 * rows[1]) & (dy < .70 * rows[1]) &
                (dx > .30 * columns[1]) & (dx < .70 * columns[1]))
    cropped = interior[:-1, :-1] & interior[1:, :-1] & interior[:-1, 1:]
    if cropped.sum() < 20:
        return 0., 1.
    gradients = np.maximum(np.abs(rgb[1:, :-1] - rgb[:-1, :-1]).max(-1),
                           np.abs(rgb[:-1, 1:] - rgb[:-1, :-1]).max(-1))
    flat = float((gradients[cropped] < 1.5).mean())
    samples = np.floor(rgb[interior] / 12).astype(np.int16)
    colours, counts = np.unique(samples, axis=0, return_counts=True)
    dominant = float(np.sort(counts)[-2:].sum() / counts.sum())
    return flat, dominant


def perspective_mosaic_mask(source):
    """Return supported mosaic pixels for skewed grids in an RGB analysis image."""
    gray = source.astype(np.float32) @ np.array([.299, .587, .114], dtype=np.float32)
    short = min(gray.shape)
    if short < 72 or gray.std() < 5:
        return np.zeros(gray.shape, dtype=bool)
    height, width = gray.shape
    votes = np.zeros(gray.shape, dtype=np.uint8)
    rgb = source.astype(np.float32)
    gy = np.abs(sobel(gray, axis=0)) / 8
    gx = np.abs(sobel(gray, axis=1)) / 8
    for side in sorted({max(48, round(short * fraction / 2) * 2) for fraction in (.22, .31)}):
        stride = side // 5
        half = side // 2
        for y in range(half, height - half + 1, stride):
            for x in range(half, width - half + 1, stride):
                region = np.s_[y-half:y+half, x-half:x+half]
                rows = family(gy[region], side)
                if rows is None:
                    continue
                columns = family(gx[region].T, side)
                if columns is None:
                    continue
                # A rotated orthogonal grid has equal and opposite slopes.
                skew = abs(rows[0] + columns[0])
                if skew < .095 or max(rows[1], columns[1]) > 1.75 * min(rows[1], columns[1]):
                    continue
                flat, dominant = cell_evidence(rgb[region], rows, columns)
                if flat < .70 or dominant > .65:
                    continue
                pad = side // 5
                votes[y-pad:y+pad, x-pad:x+pad] += 1
    supported = binary_closing(votes >= 2, iterations=3)
    components, count = label(supported)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    sizes[0] = 0
    return (sizes >= max(100, round(gray.size * .008)))[components]
