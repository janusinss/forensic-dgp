"""Bounded single-image defocus matching using Gaussian edge responses."""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


def estimate_tiles(luma):
    # For a Gaussian-blurred step edge, peak gradient is inversely proportional
    # to its width. Compare at known scales, as in DOI 10.1016/j.patcog.2011.03.009.
    scales = (.8, 2.0, 3.5)
    gradients = [tuple(gaussian_filter(luma, s, order=axis) for axis in ((1, 0), (0, 1)))
                 for s in scales]
    peaks = [np.hypot(*gradient) for gradient in gradients]
    y, x = np.indices(luma.shape, dtype=np.float32)
    ny, nx = (g / np.maximum(peaks[0], .01) for g in gradients[0])
    ridge = np.ones(luma.shape, bool)
    for direction in (-1, 1):
        neighbor = map_coordinates(peaks[0], [y + direction * ny, x + direction * nx], order=1, mode='nearest')
        ridge &= peaks[0] >= neighbor - .02
    ridge[:5] = ridge[-5:] = ridge[:, :5] = ridge[:, -5:] = False
    estimates = []
    for scale, peak in zip(scales[1:], peaks[1:]):
        ratio = (peaks[0] / np.maximum(peak, .01)) ** 2
        estimates.append(np.sqrt(np.maximum(0, (scale ** 2 - ratio * scales[0] ** 2) /
                                                np.maximum(ratio - 1, .001))))
        ridge &= (ratio > 1.04) & (peak >= 1.5)
    low, high = estimates
    valid = ridge & (peaks[0] >= 3) & (abs(low - high) <= .65 + .3 * high) & (high <= 12)
    rows = np.array_split(np.arange(luma.shape[0]), max(1, round(luma.shape[0] / 48)))
    columns = np.array_split(np.arange(luma.shape[1]), max(1, round(luma.shape[1] / 48)))
    tiles = np.full((len(rows), len(columns)), np.nan, np.float32)
    for row, ys in enumerate(rows):
        for column, xs in enumerate(columns):
            block = np.ix_(ys, xs)
            samples = high[block][valid[block]]
            if len(samples) >= 12:
                tiles[row, column] = np.median(samples)
    return tiles


def estimate_tiles_normalized(source, max_longest=256):
    """Return tile blur estimates normalized to longest-side=256 pixel units."""
    image = source if isinstance(source, Image.Image) else Image.fromarray(source)
    longest = max(image.size)
    scale = min(1., max_longest / longest)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    luma = np.asarray(image.convert('L').resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    return estimate_tiles(luma) / (scale * longest / 256), size


def match_surroundings(rgb, source, rectangle, effective):
    """Estimate from the full upload and blend local blur across its surroundings."""
    original = Image.fromarray(source)
    longest = max(original.size)
    factor = min(1., 512 / longest)
    sample_size = tuple(max(1, round(side * factor)) for side in original.size)
    luma = np.asarray(original.convert('L').resize(sample_size, Image.Resampling.LANCZOS), dtype=np.float32)
    tiles = estimate_tiles(luma)
    reliable = np.isfinite(tiles)
    summary = {'status': 'insufficient_edges', 'scope': 'full_upload_including_protect', 'analysis_size': list(sample_size),
               'reliable_tiles': int(reliable.sum()), 'target_radius_at256': None,
               'maximum_added_radius_at256': 0.0}
    if reliable.sum() < 2:
        return rgb, np.zeros(effective.shape, bool), summary

    # Require two similar tiles before promoting the strongest tier.
    values = tiles[reliable] / (factor * longest / 256)
    ordered = np.sort(values)
    supporting = ordered[1:] - ordered[:-1] <= .5 + .15 * ordered[:-1]
    if not supporting.any():
        return rgb, np.zeros(effective.shape, bool), summary
    base_target = min(12., float(ordered[1:][supporting][-1]))
    target = base_target
    if longest > 256 and (~reliable).any():
        # Heavy blur drops edges past the 512-px ridge gate; check 256-scale safely.
        tiles_256, _ = estimate_tiles_normalized(original, max_longest=256)
        reliable_256 = np.isfinite(tiles_256)
        if reliable_256.sum() >= 2:
            ordered_256 = np.sort(tiles_256[reliable_256])
            supp_256 = ordered_256[1:] - ordered_256[:-1] <= .5 + .15 * ordered_256[:-1]
            if supp_256.any():
                target_256 = min(12., float(ordered_256[1:][supp_256][-1]))
                if target_256 > target + .45:
                    scale_gap = max(0., min(1., factor * longest / 256 - 1.))
                    target = min(12., target_256, target * (1. + .32 * scale_gap))
    summary.update(status='estimated', target_radius_at256=round(target, 2))
    local = tiles / (factor * longest / 256)
    # Missing edges do not imply a matched region. Fill those tiles from nearby
    # measurements, leaving directly measured tiles at their estimated level.
    weights = gaussian_filter(reliable.astype(np.float32), 1)
    nearby = gaussian_filter(np.where(reliable, local, 0), 1)
    nearest = distance_transform_edt(~reliable, return_distances=False, return_indices=True)
    filled = np.divide(nearby, weights, out=local[tuple(nearest)], where=weights > 0)
    local = np.where(reliable, local, filled)
    smooth = np.asarray(Image.fromarray(local).resize(original.size, Image.Resampling.BILINEAR))
    matched_tiles = reliable & (local >= base_target - .35)
    if matched_tiles.any():
        holdout = np.asarray(Image.fromarray(matched_tiles).resize(original.size, Image.Resampling.NEAREST))
        taper = np.clip(distance_transform_edt(~holdout) / max(1, 8 * longest / 256), 0, 1)
    else:
        taper = 1.
    if rectangle is not None:
        x, y, w, h = rectangle
        smooth = smooth[y:y + h, x:x + w]
        if matched_tiles.any():
            taper = taper[y:y + h, x:x + w]
    if (~effective).any():
        # Outward smoothstep taper from protected pixels eliminates boundary step seams.
        d_out = distance_transform_edt(effective)
        t_px = max(6.0, 10.0 * longest / 256.0)
        u = np.clip(d_out / t_px, 0.0, 1.0).astype(np.float32)
        protect_taper = u * u * (3.0 - 2.0 * u)
    else:
        protect_taper = 1.0
    difference = target - smooth
    radius = np.sqrt(np.maximum(0, target ** 2 - smooth ** 2))
    # Taper small residual differences near the target rather than cutting off
    # at a tile boundary. Blend between adjacent half-pixel filters.
    steps = np.where(effective, radius * np.clip(difference / .35, 0, 1) * taper * protect_taper * 2, 0).astype(np.float32)
    maximum_step = float(steps.max())
    summary['maximum_added_radius_at256'] = round(maximum_step / 2, 2)
    summary['inferred_tiles'] = int((~reliable).sum())
    if maximum_step == 0:
        return rgb, steps > 0, summary
    matched = rgb.astype(np.float32)
    for band in range(1, int(np.ceil(maximum_step)) + 1):
        weight = 1 - np.abs(steps - band)
        selection = weight > 0
        if not selection.any():
            continue
        blurred = np.asarray(original.filter(ImageFilter.GaussianBlur(band * longest / 512)))
        if rectangle is not None:
            blurred = blurred[y:y + h, x:x + w]
        matched[selection] += weight[selection, None] * (blurred[selection].astype(np.float32) - rgb[selection])
    return np.rint(matched).clip(0, 255).astype(np.uint8), steps > 0, summary
