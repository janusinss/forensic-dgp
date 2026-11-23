"""Coordinate separate four-pixel mosaic and JPEG evidence in Picture Auto."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation

from .jpeg_deblocking import jpeg_luma_dc, run_deblock, CHECKPOINT_SHA256
from .preparation import png, sha


def mosaic4_mask(gray):
    """Require seams at both four-pixel phases of each eight-pixel cycle."""
    height, width = gray.shape
    groups = defaultdict(list)
    for y in range(0, height - 63, 16):
        for x in range(0, width - 63, 16):
            tile = gray[y:y + 64, x:x + 64]
            phases = []
            for axis, offset in ((0, y), (1, x)):
                edges = np.abs(np.diff(tile, axis=axis)).mean(axis=1-axis)
                positions = np.arange(edges.size) + 1 + offset
                means = np.array([edges[positions % 4 == p].mean() for p in range(4)])
                phase = int(np.argmax(means))
                interior = float(np.median(means))
                if means[phase] - interior < max(1.5, .65 * interior):
                    break
                alternates = [edges[positions % 8 == p].mean() - interior for p in (phase, phase + 4)]
                if min(alternates) < max(.85, .18 * max(alternates)):
                    break
                phases.append(phase)
            if len(phases) == 2:
                groups[tuple(phases)].append((x, y))
    windows = max(groups.values(), key=len, default=())
    if len(windows) < 8:
        return None
    votes = np.zeros(gray.shape, dtype=np.uint16)
    for x, y in windows:
        votes[y + 12:y + 52, x + 12:x + 52] += 1
    return votes >= 2


def phase_signals(gray, region):
    """Measure 8 px and midpoint seams from adjacent pairs within a region."""
    profiles = []
    for axis in (0, 1):
        allowed = region[:-1] & region[1:] if axis == 0 else region[:, :-1] & region[:, 1:]
        edges = np.abs(np.diff(gray, axis=axis))
        positions = (np.arange(edges.shape[axis]) + 1) % 8
        phases = positions[:, None] if axis == 0 else positions[None, :]
        groups = [edges[allowed & (phases == phase)] for phase in range(8)]
        if min(map(len, groups)) < 256:
            return None
        means = [float(group.mean()) for group in groups]
        interior = float(np.mean([means[index] for index in (1, 2, 3, 5, 6, 7)]))
        profiles.append((means[0] - interior, means[4] - interior, interior))
    return profiles


def prepare_mixed_jpeg(contents, source, rectangle):
    """Apply a bounded hybrid only if each artifact has separate evidence."""
    height, width = source.shape[:2]
    if min(width, height) < 192 or max(width, height) > 320:
        return None
    dc = jpeg_luma_dc(contents)
    if dc is None or not 35 <= dc < 45:
        return None
    gray = np.asarray(Image.fromarray(source).convert('L'), dtype=np.float32)
    mask = mosaic4_mask(gray)
    if mask is None or not .30 <= mask.mean() <= .70:
        return None
    if rectangle is not None:
        x, y, crop_width, crop_height = rectangle
        if not mask[y:y + crop_height, x:x + crop_width].any():
            return None
    outside = ~binary_dilation(mask, iterations=12)
    before = phase_signals(gray, outside)
    if before is None or any(seam < max(1., .65 * detail) or midpoint > .5 * seam
                             for seam, midpoint, detail in before):
        return None
    inner_before = phase_signals(gray, mask)
    if inner_before is None or any(min(axis[:2]) < 1. for axis in inner_before):
        return None
    smoothed = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(1)))
    inner_after = phase_signals(np.asarray(Image.fromarray(smoothed).convert('L'), dtype=np.float32), mask)
    if any(abs(remaining) > .8 * initial for original, filtered in zip(inner_before, inner_after)
           for initial, remaining in zip(original[:2], filtered[:2])):
        return None
    deblocked = run_deblock(source)
    if deblocked is None:
        return None
    prepared = np.rint(.65 * source + .35 * deblocked).astype(np.uint8)
    prepared[mask] = smoothed[mask]
    after = phase_signals(np.asarray(Image.fromarray(prepared).convert('L'), dtype=np.float32), outside)
    if any(abs(result[0]) > .8 * initial[0] or result[2] < .9 * initial[2]
           for initial, result in zip(before, after)):
        return None
    prepared_png = png(prepared)
    mask_png = png(mask.astype(np.uint8) * 255)
    report = {'policy': 'coordinated-jpeg-mosaic-v1', 'mixed_jpeg_mosaic': True,
              'classification_is_approximate': True, 'native_size': [width, height],
              'mosaic_grid_period_native': 4, 'native_Gaussian_radius': 1.,
              'jpeg_luma_dc_quantization': dc, 'outside_deblock_fraction': .35,
              'mosaic_seam_excess_before': [[round(value, 4) for value in axis[:2]] for axis in inner_before],
              'mosaic_seam_excess_after': [[round(value, 4) for value in axis[:2]] for axis in inner_after],
              'outside_jpeg_seam_excess_before': [round(item[0], 4) for item in before],
              'outside_jpeg_seam_excess_after': [round(item[0], 4) for item in after],
              'masked_native_pixels': int(mask.sum()),
              'changed_native_pixels': int(np.any(prepared != source, axis=2).sum()),
              'preparation_model': 'FBCNN RGB', 'preparation_model_sha256': CHECKPOINT_SHA256,
              'preparation_forward_returns': 1, 'full_image_sha256': sha(prepared_png), 'mask_sha256': sha(mask_png)}
    return prepared, report, {'full-image-smoothed.png': prepared_png, 'full-image-pixelation-mask.png': mask_png}
