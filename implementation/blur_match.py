"""Bounded single-image defocus matching using Gaussian edge responses."""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter, gaussian_laplace, map_coordinates


_EMPIRICAL_SIGMA_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)


def _empirical_surround_sigma_at256(source, protect_full):
    """Estimate target Gaussian sigma (256-scale) matching sharp exterior LoG energy to protected core.

    Returns `(sigma_at256, core_hf)` or `(None, None)` when exterior is not sharper than core.
    """
    if protect_full is None or not protect_full.any() or protect_full.all():
        return None, None
    image = source if isinstance(source, Image.Image) else Image.fromarray(source)
    longest = max(image.size)
    if longest < 24:
        return None, None
    scale = min(1.0, 256.0 / longest)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    luma = np.asarray(image.convert('L').resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    prot_256 = np.asarray(
        Image.fromarray(protect_full.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    ) > 0
    if min(luma.shape) < 16 or not prot_256.any() or prot_256.all():
        return None, None

    border = np.ones(luma.shape, dtype=bool)
    border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = False

    core_eroded = binary_erosion(prot_256, iterations=2) & border
    core_eval = core_eroded if int(core_eroded.sum()) >= 24 else (prot_256 & border)
    ext_eval = (~binary_dilation(prot_256, iterations=2)) & border
    if int(core_eval.sum()) < 16 or int(ext_eval.sum()) < 32:
        return None, None

    log0 = np.abs(gaussian_laplace(luma, sigma=1.2))
    ext_p65 = float(np.percentile(log0[ext_eval], 65))
    ext_sharp = ext_eval & (log0 >= ext_p65)
    if int(ext_sharp.sum()) < 16:
        ext_sharp = ext_eval

    core_hf = float(np.percentile(log0[core_eval], 75))
    core_p90 = float(np.percentile(log0[core_eval], 90))
    ext_all_p75 = float(np.percentile(log0[ext_eval], 75))
    ext_hf = float(np.percentile(log0[ext_sharp], 75))
    if ext_all_p75 < max(2.0, core_hf * 1.4) or ext_hf < max(2.5, core_hf * 1.8):
        # Handle narrow sharp exterior strips (e.g. mugshot height-chart ticks on a
        # smooth wall) where 75% of the exterior is plain wall, top 10% is sharp,
        # AND the protected core itself is genuinely blurred (low 90th percentile).
        ext_p90 = float(np.percentile(log0[ext_eval], 90))
        ext_p95 = float(np.percentile(log0[ext_eval], 95))
        if (
            core_p90 <= 2.0
            and ext_p90 >= max(1.8, core_p90 * 1.6)
            and ext_p95 >= max(3.8, core_p90 * 3.2)
        ):
            ext_sharp = ext_eval & (log0 >= ext_p95)
            if int(ext_sharp.sum()) < 16:
                ext_sharp = ext_eval & (log0 >= ext_p90)
            if int(ext_sharp.sum()) < 16:
                return None, None
            ext_hf = float(np.percentile(log0[ext_sharp], 75))
        else:
            return None, None

    norm = max(1e-6, scale * longest / 256.0)
    target_val = max(0.1, core_hf)
    prev_sigma = 0.0
    prev_val = ext_hf
    for sigma in _EMPIRICAL_SIGMA_GRID:
        blurred = gaussian_filter(luma, sigma=sigma)
        cur_val = float(np.percentile(np.abs(gaussian_laplace(blurred, sigma=1.2))[ext_sharp], 75))
        if cur_val <= target_val:
            log_prev = np.log(max(0.05, prev_val))
            log_cur = np.log(max(0.05, cur_val))
            log_tgt = np.log(max(0.05, target_val))
            denom = max(1e-6, log_prev - log_cur)
            t = float(np.clip((log_prev - log_tgt) / denom, 0.0, 1.0))
            est = prev_sigma + t * (sigma - prev_sigma)
            return float(min(12.0, max(1.5, est / norm))), core_hf
        prev_sigma = sigma
        prev_val = cur_val
    return float(min(12.0, max(1.5, _EMPIRICAL_SIGMA_GRID[-1] / norm))), core_hf


def _tile_log_hf_p75(image, tile_shape, protect_full=None):
    """Compute per-tile exterior |LoG(sigma=1.2)| energy at 256-scale."""
    longest = max(image.size)
    scale = min(1.0, 256.0 / longest)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    luma_256 = np.asarray(image.convert('L').resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    log_abs = np.abs(gaussian_laplace(luma_256, sigma=1.2))
    prot_256 = None
    if protect_full is not None and protect_full.any():
        prot_256 = np.asarray(
            Image.fromarray(protect_full.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
        ) > 0
    rows = np.array_split(np.arange(luma_256.shape[0]), tile_shape[0])
    cols = np.array_split(np.arange(luma_256.shape[1]), tile_shape[1])
    out = np.zeros(tile_shape, dtype=np.float32)
    for r, ys in enumerate(rows):
        for c, xs in enumerate(cols):
            if ys.size and xs.size:
                block = np.ix_(ys, xs)
                if prot_256 is not None:
                    ext_pts = log_abs[block][~prot_256[block]]
                    out[r, c] = float(np.percentile(ext_pts, 95)) if ext_pts.size else 0.0
                else:
                    out[r, c] = float(np.percentile(log_abs[block], 75))
    return out


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

    protect_full = np.zeros(source.shape[:2], dtype=bool)
    if rectangle is not None:
        rx, ry, rw, rh = rectangle
        protect_full[ry:ry + rh, rx:rx + rw] = ~effective
    else:
        protect_full = ~effective

    full_emp, core_hf = _empirical_surround_sigma_at256(original, protect_full) if protect_full.any() else (None, None)
    crop_emp_scaled = None
    if rectangle is not None and (~effective).any():
        crop_emp, crop_core_hf = _empirical_surround_sigma_at256(rgb, ~effective)
        if core_hf is None:
            core_hf = crop_core_hf
        if crop_emp is not None:
            crop_longest = max(rgb.shape[:2])
            crop_emp_scaled = min(12.0, crop_emp * (crop_longest / max(1.0, float(longest))))
    emp_candidates = [v for v in (full_emp, crop_emp_scaled) if v is not None]
    empirical_sigma = max(emp_candidates) if emp_candidates else None

    base_target = None
    if reliable.sum() >= 2:
        # Require two similar tiles before promoting the strongest tier.
        values = tiles[reliable] / (factor * longest / 256)
        ordered = np.sort(values)
        supporting = ordered[1:] - ordered[:-1] <= .5 + .15 * ordered[:-1]
        if supporting.any():
            base_target = min(12., float(ordered[1:][supporting][-1]))

    target = base_target
    if target is not None and longest > 256 and (~reliable).any():
        # Heavy blur drops edges past the 512-px ridge gate; check 256-scale without 1.32x clamp.
        tiles_256, _ = estimate_tiles_normalized(original, max_longest=256)
        reliable_256 = np.isfinite(tiles_256)
        if reliable_256.sum() >= 2:
            ordered_256 = np.sort(tiles_256[reliable_256])
            supp_256 = ordered_256[1:] - ordered_256[:-1] <= .5 + .15 * ordered_256[:-1]
            if supp_256.any():
                target_256 = min(12., float(ordered_256[1:][supp_256][-1]))
                if target_256 > target + .45:
                    target = target_256

    if empirical_sigma is not None and empirical_sigma >= 1.5:
        if target is None:
            target = min(12., empirical_sigma)
            summary['estimation_method'] = 'empirical_hf_match'
        elif empirical_sigma > target + .35:
            target = min(12., empirical_sigma)
            summary['estimation_method'] = 'ridge_and_empirical_hf'
        else:
            summary['estimation_method'] = 'ridge_tiles'
    elif target is None:
        return rgb, np.zeros(effective.shape, bool), summary
    else:
        summary['estimation_method'] = 'ridge_tiles'

    # Prevent severe over-smoothing (sharp_prep < 0.5) on tiny cropped faces (< 96 px)
    # where empirical surround sigma can saturate against tiny noisy exterior patches.
    crop_longest_native = max(rgb.shape[:2])
    if crop_longest_native < 96 and target > 5.0:
        target = 5.0
    summary.update(status='estimated', target_radius_at256=round(target, 2))
    if empirical_sigma is not None:
        summary['empirical_surround_sigma_at256'] = round(empirical_sigma, 2)

    if reliable.any():
        local = tiles / (factor * longest / 256)
        seed_mask = reliable
        if protect_full.any():
            prot_sample = np.asarray(
                Image.fromarray(protect_full.astype(np.uint8) * 255).resize(sample_size, Image.Resampling.NEAREST)
            ) > 0
            t_rows = np.array_split(np.arange(luma.shape[0]), tiles.shape[0])
            t_cols = np.array_split(np.arange(luma.shape[1]), tiles.shape[1])
            prot_frac = np.zeros(tiles.shape, dtype=np.float32)
            for r_i, ys in enumerate(t_rows):
                for c_i, xs in enumerate(t_cols):
                    if ys.size and xs.size:
                        prot_frac[r_i, c_i] = float(prot_sample[np.ix_(ys, xs)].mean())
            ext_reliable = reliable & (prot_frac <= 0.30)
            if ext_reliable.any():
                seed_mask = ext_reliable
        # Missing edges do not imply a matched region. Fill those tiles from nearby
        # exterior measurements so blurred-face boundary tiles never bleed into sharp strips.
        weights = gaussian_filter(seed_mask.astype(np.float32), 1)
        nearby = gaussian_filter(np.where(seed_mask, local, 0), 1)
        nearest = distance_transform_edt(~seed_mask, return_distances=False, return_indices=True)
        filled = np.divide(nearby, weights, out=local[tuple(nearest)], where=weights > 0)
        local = np.where(seed_mask, local, filled)
        tile_hf = None
        if empirical_sigma is not None and core_hf is not None and protect_full.any():
            tile_hf = _tile_log_hf_p75(original, tiles.shape, protect_full=protect_full)
            # Tiles containing genuinely sharp exterior features (e.g. narrow ruler ticks
            # beside a blurred head) can have inflated ridge medians from the adjacent blur;
            # cap their local blur estimate so they receive full surround matching.
            if core_hf <= 2.0 and target >= 2.0:
                sharp_ext = tile_hf > max(core_hf * 2.0, 2.5)
                if sharp_ext.any():
                    local = np.where(sharp_ext, np.minimum(local, 1.5), local)
        smooth = np.asarray(Image.fromarray(local).resize(original.size, Image.Resampling.BILINEAR))
        matched_tiles = seed_mask & (local >= target - .35)
        if matched_tiles.any() and empirical_sigma is not None and tile_hf is not None:
            # Never hold out sharp ruler-tick/chart tiles that fooled estimate_tiles().
            matched_tiles &= tile_hf <= max(core_hf * 1.6, 2.0)
    else:
        smooth = np.zeros(source.shape[:2], dtype=np.float32)
        matched_tiles = np.zeros_like(reliable, dtype=bool)

    if matched_tiles.any():
        holdout = np.asarray(Image.fromarray(matched_tiles).resize(original.size, Image.Resampling.NEAREST))
        taper = np.clip(distance_transform_edt(~holdout) / max(1.0, 3.0 * longest / 256.0), 0.0, 1.0)
    else:
        taper = 1.0
    if rectangle is not None:
        x, y, w, h = rectangle
        smooth = smooth[y:y + h, x:x + w]
        if matched_tiles.any():
            taper = taper[y:y + h, x:x + w]
    if (~effective).any():
        # Crisp outward smoothstep taper (~2-4 px at crop 256-scale) prevents sharp halo rings
        # while avoiding hard pixel step seams at the detected blur boundary.
        d_out = distance_transform_edt(effective)
        crop_longest = max(rgb.shape[:2])
        t_px = max(2.0, 3.0 * crop_longest / 256.0)
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
