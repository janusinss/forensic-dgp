"""Find strong, repeated block seams on a small native crop."""
import numpy as np

WINDOW = 24
FREQUENCIES = np.linspace(1 / 6, 1 / 2.9, 90)
PHASES = np.exp(-2j * np.pi * FREQUENCIES[:, None] * np.arange(WINDOW - 1)[None, :])


def tiny_grid_mask(gray):
    height, width = gray.shape
    if min(height, width) < 32 or max(height, width) >= 48:
        return None, None
    # A low-color grid can be part of the scene, such as a checkerboard.
    if np.unique(gray).size < 8:
        return None, None
    votes = np.zeros((height, width), dtype=np.uint8)
    candidates = []
    for y in range(0, height - WINDOW + 1, 4):
        for x in range(0, width - WINDOW + 1, 4):
            tile = gray[y:y + WINDOW, x:x + WINDOW]
            periods, strengths = [], []
            for axis in (0, 1):
                profile = np.abs(np.diff(tile, axis=axis)).mean(axis=1-axis)
                contrast = np.abs(PHASES @ (profile - profile.mean())) / max(12., profile.sum())
                peak = int(np.argmax(contrast))
                periods.append(1 / FREQUENCIES[peak])
                strengths.append(float(contrast[peak]))
            if min(strengths) >= .65 and max(periods) <= 1.2 * min(periods):
                candidates.append((min(strengths), periods, x, y))
    if len(candidates) < 4:
        return None, None
    _, reference, _, _ = max(candidates, key=lambda item: item[0])
    for _, periods, x, y in candidates:
        if max(abs(periods[axis] - reference[axis]) for axis in (0, 1)) <= .5:
            votes[y + 3:y + WINDOW - 3, x + 3:x + WINDOW - 3] += 1
    mask = votes >= 3
    if np.count_nonzero(mask) < 144:
        return None, None
    return mask, [round(float(period), 2) for period in reversed(reference)]
