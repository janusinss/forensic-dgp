"""Conservative optional Protect + Auto-match routing before restoration."""

import io

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label,
    uniform_filter,
    uniform_filter1d,
)

from .antialiasing import prepare_antialiasing
from .blur_match import estimate_tiles, estimate_tiles_normalized
from .color_bleeding import prepare_color_bleeding
from .debanding import prepare_debanding
from .deringing import prepare_deringing
from .irregular_mosaic import irregular_mosaic_mask
from .macroblock_deblocking import detect_macroblocking, prepare_macroblock_deblocking
from .perspective_mosaic import perspective_mosaic_mask
from .preparation import build_preview, json_bytes, png, sha
from .scanline_cleanup import detect_scanlines, prepare_scanlines
from .softened_grid import softened_mask
from .tiny_grid import tiny_grid_mask

PIXELATION_RADIUS_AT256 = 3.0
PIXELATION_CANDIDATES_AT256 = (2.5, 3.0, 3.5)
SCANLINE_RADIUS_AT256 = 1.5
COMPOSABLE_STAGE_ORDER = (
    'edge_deringing',
    'gradient_debanding',
    'tangent_antialiasing',
    'chroma_debleeding',
)


def coarse_grid(profile):
    """Find repeated boundaries with fractional spacing after image resizing."""
    count = profile.size
    coordinates = np.arange(count)
    baseline = max(.35, float(profile.mean()))
    # Use spectral peaks to propose spacings, then verify the edge strength
    # and the contrast with the middles of blocks in the spatial profile.
    # The median prevents a single prominent facial edge from deciding.
    fft_size = 16 * count
    spectrum = np.abs(np.fft.rfft((profile - profile.mean()) * np.hanning(count), n=fft_size))
    peaks = np.flatnonzero((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:])) + 1
    max_period = min(128, count / 4)
    peaks = peaks[(peaks >= fft_size / max_period) & (peaks <= fft_size / 3)]
    candidates = []
    for peak in sorted(peaks, key=lambda index: spectrum[index], reverse=True)[:12]:
        for frequency in (peak - .5, peak, peak + .5):
            period = fft_size / frequency
            centers = np.arange(0, period, .5)[:, None] + period * np.arange(int(count / period))
            if centers.shape[1] < 4:
                continue
            centers = centers[centers[:, -1] <= count - 1]
            if not centers.size:
                continue
            edges = np.interp(centers, coordinates, profile)
            middles = np.interp((centers[:, 1:] + centers[:, :-1]) / 2, coordinates, profile)
            strength = np.median(edges, axis=1) / baseline
            contrast = 1 + (np.median(edges, axis=1) - np.median(middles, axis=1)) / baseline
            score = float(np.minimum(strength, contrast).max())
            candidates.append((score, period))
    best = max((score for score, _ in candidates), default=0.)
    # A strong fundamental can also produce peaks at multiples of its period.
    return min(((score, period) for score, period in candidates if score >= max(3., best * .75)),
               key=lambda candidate: candidate[1], default=(0., None))


def pixelation_signal(source, face):
    """Conservative two-axis periodic-edge heuristic, not a blur classifier."""
    x, y, width, height = face
    image = Image.fromarray(source[y:y + height, x:x + width]).convert('L')
    scale = min(1., 768 / max(image.size)) if max(image.size) else 1.
    if scale < 1.:
        image = image.resize(tuple(max(1, round(side * scale)) for side in image.size), Image.Resampling.BOX)
    gray = np.asarray(image, dtype=np.float32)
    report = {'suspected': False, 'method': 'two-axis-periodic-edges-v2',
              'score': 0., 'period_native': None, 'classification_is_approximate': True}
    if min(gray.shape) < 24 or float(gray.std()) < 5:
        return report
    profiles = [np.abs(np.diff(gray, axis=axis)).mean(axis=1-axis) for axis in (0, 1)]
    candidates = []
    regular_limit = min(33, min(gray.shape) // 6) if min(gray.shape) >= 48 else 0
    for period in range(3, regular_limit + 1):
        scores = [max(float(profile[phase::period].mean()) for phase in range(period)) /
                  max(float(profile.mean()), .01) for profile in profiles]
        candidates.append((min(scores), period))
    best = max((score for score, _ in candidates), default=0.)
    report['score'] = round(best, 3)
    # Prefer the smallest strong supported period over a larger harmonic.
    regular_period = min((period for score, period in candidates if score >= max(1.9, best * .85)), default=None)
    (vertical_score, vertical), (horizontal_score, horizontal) = map(coarse_grid, profiles)
    coarse_score = min(vertical_score, horizontal_score)
    # Require a strong match in both directions and comparable spacings.
    # The wider search admits rectangular blocks from non-uniform resizing.
    if coarse_score >= 3. and max(vertical, horizontal) <= 1.8 * min(vertical, horizontal):
        spacing = (vertical + horizontal) / 2
        regular = regular_period is not None and max(abs(regular_period - vertical), abs(regular_period - horizontal)) <= .25
        report.update(suspected=True, score=round(coarse_score, 3),
                      pattern='regular_grid' if regular else 'coarse_or_resized_grid',
                      period_native=round((regular_period if regular else spacing) / scale, 2),
                      block_width_native=round(horizontal / scale, 2),
                      block_height_native=round(vertical / scale, 2))
    elif regular_period is not None:
        report.update(suspected=True, period_native=round(regular_period / scale, 2), pattern='regular_grid')
    return report


def compression_grid_phases(gray):
    """Find a strong shifted 8-pixel grid using cross-difference peaks."""
    cross = np.abs(np.diff(np.diff(gray, axis=0), axis=1))
    phases = []
    for axis in (0, 1):
        view = np.moveaxis(cross, axis, 0)
        center = view[1:-1]
        peaks = (center >= 2) & (center > view[:-2]) & (center > view[2:])
        counts = np.bincount(np.arange(1, len(view) - 1) % 8, weights=peaks.sum(axis=1), minlength=8)
        order = np.argsort(counts)
        peak, runner_up = counts[order[-1]], counts[order[-2]]
        if peak < max(20, 1.9 * counts.mean(), 1.55 * runner_up):
            return None
        phases.append(int(order[-1]))
    return phases


def grid_phase(profile, period):
    """Estimate where a proposed grid's first boundary falls in the window."""
    positions = np.arange(0, period, .25)
    seams = positions[:, None] + period * np.arange(int(profile.size / period))
    valid = seams[:, -1] <= profile.size - 1
    if not valid.any():
        return None
    values = np.interp(seams[valid], np.arange(profile.size), profile)
    return positions[valid][np.argmax(np.median(values, axis=1))]


def supported_seams(gradient, period, axis, phase=None):
    """Compare local energy along predicted seams with energy inside blocks."""
    across = np.moveaxis(gradient, axis, 0)
    if phase is None:
        phase = grid_phase(across.mean(axis=1), period)
    if phase is None:
        return np.zeros_like(gradient, dtype=bool)
    distance = np.abs((np.arange(len(across)) - phase + period / 2) % period - period / 2)
    seam = (distance <= min(2., max(.55, period * .15))).astype(np.float32)
    interior = (distance >= max(1., period * .25)).astype(np.float32)
    size = max(9, round(period * 2.25)) | 1

    def mean_at(weights):
        total = uniform_filter(across * weights[:, None], size=size, mode='constant')
        count = uniform_filter1d(weights, size=size, mode='constant')
        return total / np.maximum(.005, count[:, None])
    supported = mean_at(seam) >= 1.55 * mean_at(interior) + .65
    return np.moveaxis(supported, 0, axis)


def regional_block_support(source, *, native_grid=True):
    """Return coarse scan support and independently verified local seams."""
    height, width = source.shape[:2]
    short = min(height, width)
    sides = sorted({min(short, 512, max(48, round(short * fraction))) for fraction in (.15, .23, .36, .60)})
    fine, coarse = np.zeros((2, height, width), bool)
    votes, row_votes, column_votes = np.zeros((3, height, width), np.uint16)
    if short < 48:
        return fine, votes, row_votes, column_votes
    if len(sides) == 1 and sides[0] < short:
        sides.append(short)
    rgb = source.astype(np.float32)
    vertical = np.abs(np.diff(rgb, axis=0)).mean(axis=2)
    horizontal = np.abs(np.diff(rgb, axis=1)).mean(axis=2)
    gray = np.asarray(Image.fromarray(source).convert('L'), dtype=np.float32)
    for index, side in enumerate(sides):
        support = fine if index < len(sides) // 2 else coarse
        inset = round(side * .30)
        rows = np.linspace(0, height - side, min(12, 1 + int(np.ceil((height - side) / (.4 * side)))), dtype=int)
        columns = np.linspace(0, width - side, min(12, 1 + int(np.ceil((width - side) / (.4 * side)))), dtype=int)
        for y in rows:
            for x in columns:
                signal = pixelation_signal(source, (x, y, side, side))
                if signal['suspected']:
                    support[y + inset:y + side - inset, x + inset:x + side - inset] = True
                    period_y = signal.get('block_height_native', signal['period_native'])
                    period_x = signal.get('block_width_native', signal['period_native'])
                    phase_y = phase_x = None
                else:
                    phases = compression_grid_phases(gray[y:y + side, x:x + side]) if native_grid else None
                    if phases is None:
                        continue
                    phase_y, phase_x = phases
                    period_y = period_x = 8

                across_rows = supported_seams(vertical[y:y + side - 1, x:x + side], period_y, 0, phase_y)
                across_columns = supported_seams(horizontal[y:y + side, x:x + side - 1], period_x, 1, phase_x)
                region = np.s_[y:y + side - 1, x:x + side - 1]
                row_votes[region] += across_rows[:, :-1]
                column_votes[region] += across_columns[:-1, :]
                votes[region] += across_rows[:, :-1] & across_columns[:-1, :]
    return fine & coarse, votes, row_votes, column_votes


def choose_pixelation_radius(image, mask):
    """Compare cheap smoothing candidates on the same 256-pixel analysis canvas."""
    from .runtime import prepare_crop

    if max(image.size) > 1600:
        size = tuple(max(1, round(side * 1600 / max(image.size))) for side in image.size)
        image = image.resize(size, Image.Resampling.BOX)
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
    source = np.asarray(image)
    original, observed, _, geometry = prepare_crop(source)
    side = max(image.size)
    mask_canvas = Image.new('L', (side, side))
    mask_canvas.paste(Image.fromarray(mask.astype(np.uint8) * 255), tuple(geometry['native_center_offset']))
    scope = (np.asarray(mask_canvas.resize((256, 256), Image.Resampling.NEAREST)) > 0) & observed
    report = {'policy': 'block-score-detail-tradeoff-v1', 'basis': 'default_weak_evidence',
              'selected_radius_at256': PIXELATION_RADIUS_AT256}
    if np.count_nonzero(scope) < 1024:
        return report
    yy, xx = np.nonzero(scope)
    box = (int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))
    if not pixelation_signal(original, box)['suspected']:
        return report

    def detail(canvas):
        gray = np.asarray(Image.fromarray(canvas).convert('L'), dtype=np.float32)
        dx = gaussian_filter(gray, 2.5, order=(0, 1))
        dy = gaussian_filter(gray, 2.5, order=(1, 0))
        return float(np.hypot(dx, dy)[scope].mean())

    original_detail = detail(original)
    if original_detail < .5:
        return report
    candidates = []
    for radius in PIXELATION_CANDIDATES_AT256:
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius * side / 256)))
        prepared = source.copy()
        prepared[mask] = blurred[mask]
        canvas, _, _, _ = prepare_crop(prepared)
        residual = pixelation_signal(canvas, box)['score']
        if residual <= 0:
            return report
        retained = min(1., detail(canvas) / original_detail)
        # Calibrated on the exploratory sweep. Both measurements are input
        # diagnostics; neither predicts ground-truth or identity accuracy.
        cost = residual + 2.8 * (1 - retained)
        candidates.append({'radius_at256': radius, 'block_score': residual,
                           'detail_retention': round(retained, 4), 'cost': round(cost, 5)})
    default = next(item for item in candidates if item['radius_at256'] == PIXELATION_RADIUS_AT256)
    best = min(candidates, key=lambda item: item['cost'])
    if default['cost'] - best['cost'] < .005:
        best = default
    report.update(basis='measured_artifact_detail_tradeoff', selected_radius_at256=best['radius_at256'],
                  candidates=candidates)
    return report


def _padded_box_for_macroblock(shape, y0, y1, x0, x1):
    """Expand a crop bounds tuple minimally to >=32x32 inside image bounds."""
    height, width = shape[:2]
    y0, y1 = max(0, int(y0)), min(height, int(y1))
    x0, x1 = max(0, int(x0)), min(width, int(x1))
    if y1 - y0 < 32:
        pad = 32 - (y1 - y0)
        y0 = max(0, y0 - pad // 2)
        y1 = min(height, y0 + 32)
        y0 = max(0, y1 - 32)
    if x1 - x0 < 32:
        pad = 32 - (x1 - x0)
        x0 = max(0, x0 - pad // 2)
        x1 = min(width, x0 + 32)
        x0 = max(0, x1 - 32)
    return y0, y1, x0, x1


def _det_passes_macroblock_gate(source, det, origin=(0, 0)):
    """Verify non-flat, non-tiny-fragment DCT detection actually deblocks cleanly."""
    if det is None or not det.get('suspected') or det.get('is_flat_privacy_mosaic'):
        return False
    if det.get('regional') and float(det.get('mask_fraction', 0.0)) < 0.25:
        return False
    cleaned, _, _ = prepare_macroblock_deblocking(source, origin=origin, allow_fbcnn=True)
    return cleaned is not None


def _is_genuine_dct_macroblocking(source, origin=(0, 0)):
    """Return True when source shows non-flat 8x8/16x16 DCT macroblocking."""
    if min(source.shape[:2]) < 32:
        return False
    det = detect_macroblocking(source, origin=origin)
    if det is None or not det.get('suspected') or det.get('is_flat_privacy_mosaic'):
        return False
    cleaned, _, _ = prepare_macroblock_deblocking(source, origin=origin, allow_fbcnn=True)
    return cleaned is not None


def _strip_dct_macroblocking_components(source, mask):
    """Remove genuine DCT macroblocking regions from a privacy-mosaic mask."""
    if min(source.shape[:2]) < 32 or not mask.any():
        return mask
    comps, num = label(mask)
    if num == 0:
        return mask
    det_full = detect_macroblocking(source)
    full_mask = None
    full_checked = False
    cleaned_mask = mask.copy()
    for cid, sl in enumerate(find_objects(comps), start=1):
        if sl is None:
            continue
        y0, y1, x0, x1 = _padded_box_for_macroblock(
            source.shape, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        )
        crop = source[y0:y1, x0:x1]
        sub_det = detect_macroblocking(crop, origin=(x0, y0))
        if sub_det is not None:
            if _det_passes_macroblock_gate(crop, sub_det, origin=(x0, y0)):
                cleaned_mask[comps == cid] = False
        else:
            if not full_checked:
                full_checked = True
                if _det_passes_macroblock_gate(source, det_full):
                    full_mask = det_full.get('mask')
            if full_mask is not None and float(full_mask[comps == cid].mean()) >= 0.5:
                cleaned_mask[comps == cid] = False
    return cleaned_mask


def _bracket_cluster_spans(sub):
    """Mark pixels strictly between first and last True entries per row/column in a cluster."""
    row_any = np.any(sub, axis=1)
    h_span = np.zeros_like(sub)
    if row_any.any():
        left = np.argmax(sub[row_any], axis=1)
        right = sub.shape[1] - 1 - np.argmax(sub[row_any, ::-1], axis=1)
        xs = np.arange(sub.shape[1])[None, :]
        h_span[row_any] = (xs >= left[:, None]) & (xs <= right[:, None])
    col_any = np.any(sub, axis=0)
    v_span = np.zeros_like(sub)
    if col_any.any():
        top = np.argmax(sub[:, col_any], axis=0)
        bottom = sub.shape[0] - 1 - np.argmax(sub[::-1, col_any], axis=0)
        ys = np.arange(sub.shape[0])[:, None]
        v_span[:, col_any] = (ys >= top[None, :]) & (ys <= bottom[None, :])
    return h_span | v_span


def _recover_mosaic_mask_holes(mask, votes, row_votes, column_votes):
    """Close seam-supported pockets and bracketed 1D seam tears inside dilated support."""
    r_close = max(6, min(18, round(max(mask.shape) * .028)))
    yy, xx = np.ogrid[-r_close:r_close + 1, -r_close:r_close + 1]
    disk = (xx * xx + yy * yy) <= (r_close * r_close)
    seam_support = (column_votes >= 1) | (row_votes >= 1) | (votes >= 1)
    merged = mask | (binary_closing(mask, structure=disk) & seam_support)
    dilated = binary_dilation(merged, structure=disk)
    strong_1d = (row_votes >= 2) | (column_votes >= 2) | (votes >= 1)
    dil_labels, _ = label(dilated)
    for cid, sl in enumerate(find_objects(dil_labels), start=1):
        if sl is None:
            continue
        cluster = dil_labels[sl] == cid
        spans = _bracket_cluster_spans(merged[sl] & cluster)
        merged[sl] |= spans & cluster & strong_1d[sl]
    return binary_fill_holes(merged)


def _blur_stats_from_tiles(tiles, image_size, face):
    """Measure median face vs outside blur radii and decision clarity from tile grid."""
    x, y, width, height = face
    img_width, img_height = image_size
    rows, columns = tiles.shape
    yy, xx = np.meshgrid((np.arange(rows) + .5) * img_height / rows,
                         (np.arange(columns) + .5) * img_width / columns, indexing='ij')
    inside = ((xx >= x + width * .15) & (xx < x + width * .85) &
              (yy >= y + height * .15) & (yy < y + height * .85))
    outside = ~((xx >= x) & (xx < x + width) & (yy >= y) & (yy < y + height))
    face_samples = tiles[inside & np.isfinite(tiles)]
    outside_samples = tiles[outside & np.isfinite(tiles)]
    if min(face_samples.size, outside_samples.size) < 3:
        return int(face_samples.size), int(outside_samples.size), None, None, False
    face_blur = float(np.median(face_samples))
    background_blur = float(np.median(outside_samples))
    clear = face_blur >= max(background_blur * 1.5, background_blur + .75)
    return int(face_samples.size), int(outside_samples.size), face_blur, background_blur, clear


def picture_pixelation(source, check_macroblocking=True):
    """Localize supported block patterns without depending on face boxes."""
    image = Image.fromarray(source)
    analysis = source
    if max(image.size) > 1600:
        size = tuple(max(1, round(side * 1600 / max(image.size))) for side in image.size)
        analysis = np.asarray(image.resize(size, Image.Resampling.BOX))
    window_support, votes, row_votes, column_votes = regional_block_support(analysis, native_grid=analysis is source)
    seams = votes >= 2

    # Keep the coarse mask as a fallback when a grid is too irregular to
    # track locally. Where seams explain most of a region, use their extent.
    coarse_regions, _ = label(window_support)
    region_sizes = np.bincount(coarse_regions.ravel())
    explained = np.bincount(coarse_regions[seams], minlength=region_sizes.size)
    unresolved = explained < .5 * region_sizes
    unresolved[0] = False
    components, _ = label(seams | unresolved[coarse_regions])

    sizes = np.bincount(components.ravel())
    # Isolated agreement can come from ordinary textures or a single edge.
    # Require spatial support before changing the original-resolution pixels.
    keep = sizes >= max(64, components.size * .002)
    keep[0] = False
    mask = keep[components]
    if not mask.any():
        return None, None, {}
    initial_mask = mask.copy()
    # Recover narrow gaps bracketed by the mask only when at least four
    # overlapping scan windows still support block seams in that direction.
    span = 2 * min(24, round(max(mask.shape) * .035)) + 1
    horizontal_gaps = binary_closing(mask, structure=np.ones((1, span), bool)) & (column_votes >= 4)
    vertical_gaps = binary_closing(mask, structure=np.ones((span, 1), bool)) & (row_votes >= 4)
    gaps = (horizontal_gaps | vertical_gaps) & ~mask
    mask |= gaps
    mask = _recover_mosaic_mask_holes(mask, votes, row_votes, column_votes)
    # Require multi-window two-axis seam corroboration per component so isolated
    # staircase aliasing cannot trigger a false regional blur mask.
    mask_comps, _ = label(mask)
    for cid, sl in enumerate(find_objects(mask_comps), start=1):
        if sl is None:
            continue
        comp_slice = mask_comps[sl] == cid
        if int(votes[sl][comp_slice].max()) < 4:
            mask[sl][comp_slice] = False
    if not mask.any():
        return None, None, {}
    recovered_gaps = int((mask & ~initial_mask).sum())
    if analysis is not source:
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
    if check_macroblocking:
        mask = _strip_dct_macroblocking_components(source, mask)
        if not mask.any():
            return None, None, {}
    return apply_pixelation_mask(source, mask, analysis.shape[:2][::-1], recovered_gaps)


def apply_pixelation_mask(source, mask, analysis_size, recovered_gaps, angle=None, tiny_period=None,
                          geometric_pattern=None):
    image = Image.fromarray(source)
    if geometric_pattern or angle is not None or tiny_period is not None:
        basis = geometric_pattern or ('rotated_mosaic_pilot' if angle is not None else 'tiny_mosaic_pilot')
        strength = {'basis': basis, 'selected_radius_at256': PIXELATION_RADIUS_AT256}
    else:
        strength = choose_pixelation_radius(image, mask)
    radius_at256 = strength['selected_radius_at256']
    radius = radius_at256 * max(image.size) / 256
    if geometric_pattern:
        filtered = gaussian_filter(source.astype(np.float32), sigma=(radius, radius, 0)).round().astype(np.uint8)
    else:
        filtered = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)))
    prepared = source.copy()
    dist_in = distance_transform_edt(mask)
    max_dist = float(dist_in.max())
    if tiny_period is not None or max_dist < 2.:
        prepared[mask] = filtered[mask]
    else:
        # Strictly inward smoothstep feathering preserves ~mask 100% bit-exact.
        feather_px = min(max(1., 3.5 * max(image.size) / 256), max(1., max_dist * .5))
        alpha = np.clip(dist_in[mask] / feather_px, 0., 1.).astype(np.float32)
        alpha = alpha * alpha * (3. - 2. * alpha)
        blended = source[mask].astype(np.float32) * (1. - alpha[:, None]) + filtered[mask].astype(np.float32) * alpha[:, None]
        prepared[mask] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    changed = np.any(prepared != source, axis=2)
    if not changed.any():
        return None, None, {}

    regions, _ = label(mask)
    bounds = [[cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start]
              for rows, cols in find_objects(regions)]
    full_png = png(prepared)
    mask_png = png(mask.astype(np.uint8) * 255)
    report = {'policy': 'multiscale-picture-block-regions-v3', 'classification_is_approximate': True,
              'native_size': list(image.size), 'analysis_size': list(analysis_size), 'region_bounds': bounds,
              'Gaussian_radius_at256': radius_at256, 'native_Gaussian_radius': radius,
              'strength_selection': strength,
              'recovered_gap_analysis_pixels': recovered_gaps,
              'masked_native_pixels': int(mask.sum()), 'changed_native_pixels': int(changed.sum()),
              'full_image_sha256': sha(full_png), 'mask_sha256': sha(mask_png)}
    if angle is not None:
        report['rotation_correction_degrees'] = angle
    if tiny_period is not None:
        report['tiny_grid_period_native'] = tiny_period
    if geometric_pattern:
        report['geometric_mosaic_pattern'] = geometric_pattern
    return prepared, report, {'full-image-smoothed.png': full_png, 'full-image-pixelation-mask.png': mask_png}


def tiny_crop_pixelation(source):
    height, width = source.shape[:2]
    if min(height, width) < 32 or max(height, width) >= 48:
        return None, None, {}
    gray = np.asarray(Image.fromarray(source).convert('L'), dtype=np.float32)
    mask, periods = tiny_grid_mask(gray)
    if mask is None:
        return None, None, {}
    if all(3.65 <= period <= 4.35 for period in periods) and _is_genuine_dct_macroblocking(source):
        return None, None, {}
    return apply_pixelation_mask(source, mask, (width, height), 0, tiny_period=periods)


def rotated_pixelation(source):
    """Rank angle corrections by grid evidence and map a supported mask back."""
    image = Image.fromarray(source)
    if min(image.size) < 128:
        return None, None, {}
    scale = min(1., 256 / max(image.size))
    size = tuple(round(side * scale) for side in image.size)
    if min(size) < 96:
        return None, None, {}
    analysis = image.resize(size, Image.Resampling.BOX) if scale < 1 else image
    # Only inspect the central region to exclude interpolated corners.
    left, top = (round(side * .15625) for side in size)
    core = (left, top, size[0] - left, size[1] - top)
    old_angles = (-15, 15, -30, 30)
    ranked = []
    valid_image = Image.new('L', size, 255)

    def probe(angle):
        valid = valid_image.rotate(angle, resample=Image.Resampling.NEAREST).crop(core)
        if valid.getextrema()[0] == 0:
            return
        window = analysis.rotate(angle, resample=Image.Resampling.BICUBIC).crop(core)
        signal = pixelation_signal(np.asarray(window), (0, 0, *window.size))
        ranked.append((signal['suspected'], signal['score'], angle, window))

    for angle in range(-33, 34, 3):
        if angle:
            probe(angle)
    promising = sorted(ranked, key=lambda item: item[1], reverse=True)[:2]
    for _, score, angle, _ in promising:
        if 1.65 <= score < 2.6:
            for finer in (angle - 1, angle + 1):
                if abs(finer) <= 33:
                    probe(finer)
    # Stable tie-breaking keeps the strongest aligned grid first.
    ranked.sort(key=lambda item: (item[0], item[1], -abs(item[2])), reverse=True)
    preferred = [item for item in ranked if item[0]][:3]
    tried = {item[2] for item in preferred}
    fallback = [item for item in ranked if item[2] in old_angles and item[2] not in tried]
    for _, _, angle, window in preferred + fallback:
        _, report, evidence = picture_pixelation(np.asarray(window), check_macroblocking=False)
        if report is None or report['masked_native_pixels'] < max(256, .04 * window.width * window.height):
            continue
        with Image.open(io.BytesIO(evidence['full-image-pixelation-mask.png'])) as detected:
            mask_image = Image.new('L', size)
            mask_image.paste(detected, (left, top))
        mask_image = mask_image.rotate(-angle, resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask_image.resize(image.size, Image.Resampling.NEAREST)) > 0
        return apply_pixelation_mask(source, mask, size, 0, angle)
    return None, None, {}


def prepare_mosaics(source):
    """Select geometric masks and coordinate spatially distinct treatments."""
    regular = picture_pixelation(source)
    if regular[0] is None and not _is_genuine_dct_macroblocking(source):
        regular = rotated_pixelation(source)
    height, width = source.shape[:2]
    existing_area = regular[1].get('masked_native_pixels', width * height) if regular[0] is not None else 0
    size = tuple(round(side * min(1., 256 / max(width, height))) for side in (width, height))
    if min(width, height) < 128 or min(size) < 96:
        return regular
    analysis = np.asarray(Image.fromarray(source).resize(size, Image.Resampling.BOX))
    irregular = irregular_mosaic_mask(analysis)
    fallback = regular
    if existing_area < .08 * width * height:
        candidates = (('perspective', perspective_mosaic_mask(analysis)), ('irregular', irregular))
        pattern, mask = max(candidates, key=lambda item: item[1].sum())
        coverage = float(mask.mean())
        if coverage >= .04 and coverage * width * height > 2 * existing_area:
            native_mask = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
            fallback = apply_pixelation_mask(source, native_mask, size, 0, geometric_pattern=pattern)
    if regular[0] is None or irregular.mean() < .04 or 'full-image-pixelation-mask.png' not in regular[2]:
        return fallback
    with Image.open(io.BytesIO(regular[2]['full-image-pixelation-mask.png'])) as base_image:
        base = np.asarray(base_image) > 0
        boundary = np.asarray(base_image.resize(size, Image.Resampling.NEAREST)) > 0
    boundary = binary_dilation(boundary, iterations=2)
    components, count = label(irregular)
    areas = np.bincount(components.ravel(), minlength=count + 1)
    overlaps = np.bincount(components[boundary], minlength=count + 1)
    accepted = (areas >= .04 * irregular.size) & (overlaps <= .30 * areas)
    accepted[0] = False
    extra = accepted[components] & ~boundary
    if extra.mean() < .03:
        return fallback
    extra = np.asarray(Image.fromarray(extra).resize((width, height), Image.Resampling.NEAREST)) & ~base
    prepared_extra, extra_report, extra_evidence = apply_pixelation_mask(source, extra, size, 0,
                                                                        geometric_pattern='irregular')
    if prepared_extra is None:
        return fallback
    prepared = regular[0].copy()
    prepared[extra] = prepared_extra[extra]
    mask = base | extra
    image_png = png(prepared)
    mask_png = png(mask.astype(np.uint8) * 255)
    report = {**regular[1], 'policy': 'coordinated-mosaic-regions-v1', 'coordinated_mosaic_regions': True,
              'region_bounds': regular[1]['region_bounds'] + extra_report['region_bounds'],
              'irregular_mask_sha256': extra_report['mask_sha256'], 'masked_native_pixels': int(mask.sum()),
              'changed_native_pixels': int(np.any(prepared != source, axis=2).sum()),
              'full_image_sha256': sha(image_png), 'mask_sha256': sha(mask_png)}
    return prepared, report, {'full-image-smoothed.png': image_png, 'full-image-pixelation-mask.png': mask_png,
                              'irregular-region-mask.png': extra_evidence['full-image-pixelation-mask.png']}


def prepare_scanline_mosaic(source, notched):
    """Remove a coherent row offset before testing for an independent mosaic."""
    height, width = source.shape[:2]
    if min(height, width) < 96 or max(height, width) > 640:
        return None
    residual = source.astype(np.float32) - notched
    row_offset = np.median(residual, axis=1, keepdims=True)
    removed_energy = float(np.mean(residual ** 2))
    if removed_energy < 4:
        return None
    explained = 1 - float(np.mean((residual - row_offset) ** 2)) / removed_energy
    if explained < .72:
        return None
    corrected = np.clip(np.rint(source - row_offset), 0, 255).astype(np.uint8)
    if detect_scanlines(corrected) is not None:
        return None

    _, mosaic, artifacts = picture_pixelation(corrected)
    if mosaic is None:
        return None
    mask_png = artifacts['full-image-pixelation-mask.png']
    with Image.open(io.BytesIO(mask_png)) as image:
        mask = np.asarray(image) > 0
    if not .04 <= mask.mean() <= .65:
        return None
    y, x = np.nonzero(mask)
    rectangle = (int(x.min()), int(y.min()), int(x.max() - x.min() + 1), int(y.max() - y.min() + 1))
    if min(rectangle[2:]) < 48 or mask.sum() < .40 * rectangle[2] * rectangle[3]:
        return None
    detected = pixelation_signal(corrected, rectangle)
    if not detected['suspected'] or detected['score'] < 2.4:
        return None
    scale = max(width, height) / 256
    corrected_image = Image.fromarray(corrected)
    floor = np.asarray(corrected_image.filter(ImageFilter.GaussianBlur(.6 * scale)))
    baseline_signal = pixelation_signal(floor, rectangle)
    if not baseline_signal['suspected']:
        return None

    def detail(rgb):
        gray = np.asarray(Image.fromarray(rgb).convert('L'), dtype=np.float32)
        dx = gaussian_filter(gray, 2.5 * scale, order=(0, 1))
        dy = gaussian_filter(gray, 2.5 * scale, order=(1, 0))
        return float(np.hypot(dx, dy)[mask].mean())

    baseline_detail = detail(floor)
    if baseline_detail < .5:
        return None
    for radius in (2., 2.5, 3.):
        filtered = np.asarray(corrected_image.filter(ImageFilter.GaussianBlur(radius * scale)))
        combined = floor.copy()
        combined[mask] = filtered[mask]
        remaining = pixelation_signal(combined, rectangle)['score']
        retained = detail(combined) / baseline_detail
        if remaining <= .80 * baseline_signal['score'] and retained >= .85:
            return combined, mask_png, {'radius_at256': radius, 'masked_native_pixels': int(mask.sum()),
                                        'grid_score_before': baseline_signal['score'], 'grid_score_after': remaining,
                                        'detail_retention': round(retained, 4), 'row_energy_fraction': round(explained, 4),
                                        'background_radius_at256': .6, 'mask_sha256': sha(mask_png)}
    return None


def prepare_source(source):
    """Choose full-source preparation before falling back to face analysis."""
    cleaned, report = prepare_scanlines(source)
    if cleaned is None:
        mosaics = prepare_mosaics(source)
        if mosaics[0] is not None:
            return mosaics
        if _is_genuine_dct_macroblocking(source):
            return None, None, {}
        return softened_pixelation_evidence(source)
    joint = prepare_scanline_mosaic(source, cleaned)
    if joint is not None:
        combined, mask_png, mosaic = joint
        contents = png(combined)
        report.update(policy='coordinated-scanline-mosaic-v1', mosaic=mosaic, Gaussian_radius_at256=.6,
                      changed_native_pixels=int(np.any(combined != source, axis=2).sum()), full_image_sha256=sha(contents))
        return combined, report, {'scanline-cleaned.png': contents, 'scanline-mosaic-mask.png': mask_png}
    # A light finishing pass helps V2 with the residual comb edges that a
    # narrow notch leaves behind, as observed in the wild2 comparison.
    image = Image.fromarray(cleaned)
    cleaned = np.asarray(image.filter(ImageFilter.GaussianBlur(SCANLINE_RADIUS_AT256 * max(image.size) / 256)))
    contents = png(cleaned)
    report['Gaussian_radius_at256'] = SCANLINE_RADIUS_AT256
    report['changed_native_pixels'] = int(np.any(cleaned != source, axis=2).sum())
    report['full_image_sha256'] = sha(contents)
    return cleaned, report, {'scanline-cleaned.png': contents}


def softened_pixelation_evidence(source):
    """Record a suspected softened grid for review without changing the input."""
    image = Image.fromarray(source)
    scale = min(1., 384 / max(image.size))
    size = tuple(round(side * scale) for side in image.size)
    if min(size) < 112:
        return None, None, {}
    analysis = image.convert('L')
    if scale < 1:
        analysis = analysis.resize(size, Image.Resampling.BOX)
    mask = softened_mask(np.asarray(analysis, dtype=np.float32))
    if not mask.any():
        return None, None, {}
    mask_image = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
    contents = png(np.asarray(mask_image, dtype=np.uint8) * 255)
    report = {'policy': 'softened-mosaic-spectrum-v1', 'classification_is_approximate': True,
              'native_size': list(image.size), 'analysis_size': list(size),
              'masked_native_pixels': int(np.count_nonzero(mask_image)), 'mask_sha256': sha(contents)}
    return None, report, {'softened-mosaic-mask.png': contents}


def decide(source, face):
    block = pixelation_signal(source, face)
    if block['suspected']:
        x, y, width, height = face
        if min(source.shape[:2]) >= 32:
            y0, y1, x0, x1 = _padded_box_for_macroblock(source.shape, y, y + height, x, x + width)
            is_dct = _is_genuine_dct_macroblocking(source[y0:y1, x0:x1], origin=(x0, y0))
        else:
            is_dct = False
        if not is_dct:
            return {'policy': 'face-versus-surroundings-v1', 'route': 'face_gaussian',
                    'reason': 'suspected_pixelation', 'degradation': block}
    image = Image.fromarray(source)
    longest = max(image.size)
    scale = min(1., 512 / longest)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    luma = np.asarray(image.convert('L').resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    tiles = estimate_tiles(luma) / (scale * longest / 256)
    in_n, out_n, face_blur, background_blur, clear = _blur_stats_from_tiles(tiles, image.size, face)
    if longest > 256 and (in_n < 8 or not clear):
        tiles_256, _ = estimate_tiles_normalized(image, max_longest=256)
        in_256, out_256, face_256, bg_256, clear_256 = _blur_stats_from_tiles(tiles_256, image.size, face)
        if clear_256:
            in_n, out_n, face_blur, background_blur, clear = in_256, out_256, face_256, bg_256, True
        elif not clear and in_256 >= 3 and out_256 >= 3 and in_n < 3:
            in_n, out_n, face_blur, background_blur = in_256, out_256, face_256, bg_256
    report = {'policy': 'face-versus-surroundings-v1', 'route': 'direct',
              'reason': 'insufficient_edges', 'face_tiles': in_n,
              'outside_tiles': out_n, 'degradation': block}
    if face_blur is None or background_blur is None:
        return report
    report.update(face_radius_at256=face_blur, outside_radius_at256=background_blur)
    if clear:
        report.update(route='protect_auto', reason='face_blur_exceeds_surroundings')
    else:
        report['reason'] = 'no_clear_blur_gap'
    return report


def prepare(native, source, face, rectangle):
    """Return prepared bytes and evidence, or a recorded direct-restore decision."""
    decision = decide(source, face)
    decision['analysis_RGB_sha256'] = sha(source.tobytes())
    decision['detected_face_rectangle'] = list(face)
    if decision['route'] == 'direct':
        return None, None, {}, decision
    x, y, width, height = face
    offset_x, offset_y = rectangle[:2] if rectangle is not None else (0, 0)
    if decision['route'] == 'face_gaussian':
        # Only the detected face is written. Keep original crop geometry and
        # all surrounding pixels exact; never treat smoothing as ground truth.
        apply = np.zeros(native.shape[:2], np.uint8)
        left, top = max(0, x - offset_x), max(0, y - offset_y)
        right = min(native.shape[1], x + width - offset_x)
        bottom = min(native.shape[0], y + height - offset_y)
        if right <= left or bottom <= top:
            decision.update(route='direct', reason='face_outside_selected_crop')
            return None, None, {}, decision
        apply[top:bottom, left:right] = 255
        strength = choose_pixelation_radius(Image.fromarray(native), apply > 0)
        radius = strength['selected_radius_at256']
        response, evidence = build_preview(png(native), png(np.zeros_like(apply)), png(apply),
                                          nominal_radius=radius, exact_marks=True, match_blur=False)
        decision.update(Gaussian_radius_at256=radius, strength_selection=strength,
                        strength_basis=strength['basis'], smoothing_is_experimental=True)
        return finish(response, evidence, decision, automatic_protection=False)
    # A detector box is a search region, not a protection mask. Within a
    # bounded facial region, retain the low-detail component supported by the
    # blurred face. Sharp hair/clothing and disconnected smooth background
    # stay outside Protect. This is a heuristic, not semantic segmentation.
    # Work on the full source so selecting a crop cannot change the mask.
    gray = np.asarray(Image.fromarray(source).convert('L'), dtype=np.float32)
    scale = max(source.shape[:2]) / 512
    residual = gray - gaussian_filter(gray, max(.5, scale))
    detail = np.sqrt(gaussian_filter(residual ** 2, max(1, 3 * scale)))
    yy, xx = np.indices(gray.shape)
    region = ((xx >= x) & (xx < x + width) &
              (yy >= y - height * .09) & (yy < y + height * .88))
    components, _ = label((detail < 1.2) & region)
    core = ((xx >= x + width * .2) & (xx < x + width * .8) &
            (yy >= y + height * .2) & (yy < y + height * .65))
    counts = np.bincount(components[core].ravel(), minlength=1)
    counts[0] = 0
    if not counts.any():
        decision.update(route='direct', reason='no_supported_blurred_protection_region')
        return None, None, {}, decision
    protected = components == int(counts.argmax())
    mask = protected[offset_y:offset_y + native.shape[0], offset_x:offset_x + native.shape[1]].astype(np.uint8) * 255
    decision['protection_shape'] = 'blur-supported-component-v1'
    decision['protection_search_bounds'] = [x, y - height * .09, width, height * .97]
    decision['protection_detail_threshold'] = 1.2
    decision['protection_is_approximate'] = True
    if not mask.any() or mask.all():
        decision.update(route='direct', reason='no_surroundings_in_selected_crop')
        return None, None, {}, decision
    response, evidence = build_preview(png(native), png(mask), png(np.zeros_like(mask)),
                                      nominal_radius=1, exact_marks=True, match_blur=True,
                                      full_picture=source, rectangle=rectangle)
    return finish(response, evidence, decision, automatic_protection=True)


def finish(response, evidence, decision, automatic_protection):
    metadata = response['metadata']
    if metadata['no_effect']:
        decision.update(route='direct', reason='no_additional_blur_needed')
        return None, None, {}, decision
    decision['changed_native_pixels'] = metadata['changed_native_pixels']
    metadata['automatic_protection'] = automatic_protection
    metadata['automatic_detection_or_adaptive_matching'] = True
    metadata['manual_assisted_masks'] = False
    # The endpoint owns the upload provenance. Preparation evidence describes
    # its selected crop and must not replace the retained full upload.
    evidence['automatic-crop-input.png'] = evidence.pop('uploaded-original.bin')
    metadata['payload_pins']['automatic-crop-input.png'] = metadata['payload_pins'].pop('uploaded-original.bin')
    evidence['processing.json'] = json_bytes(metadata)
    return evidence['derived-native.png'], metadata, evidence, decision


def prepare_macroblocking(source, origin=(0, 0), mask=None, allow_fbcnn=True, dc=None):
    """Run FBCNN or HEVC macroblock deblocking and return canonical stage metadata."""
    cleaned, report, raw_evidence = prepare_macroblock_deblocking(
        source, origin=origin, mask=mask, allow_fbcnn=allow_fbcnn, dc=dc
    )
    if cleaned is None or report is None:
        return None, None, None, {}
    method = report.get('method', '')
    stage = 'macroblock_fbcnn' if method in ('fbcnn_rgb', 'fbcnn_plus_hevc_16x16') else 'macroblock_hevc'
    mask_png = raw_evidence.get('macroblock-mask.png')
    if mask_png is None:
        mask_png = png(np.full(cleaned.shape[:2], 255, dtype=np.uint8))
    return cleaned, stage, report, {'macroblock-mask.png': mask_png}


def prepare_composables(source):
    """Run quantitatively gated refiners in fixed artifact-safe order."""
    current = source
    stages = []
    stage_reports = {}
    evidence = {}
    pipeline = (
        ('edge_deringing', prepare_deringing),
        ('gradient_debanding', prepare_debanding),
        ('tangent_antialiasing', prepare_antialiasing),
        ('chroma_debleeding', prepare_color_bleeding),
    )
    for stage_name, fn in pipeline:
        output, report, stage_evidence = fn(current)
        if output is None or report is None:
            continue
        current = output
        stages.append(stage_name)
        stage_reports[stage_name] = report
        evidence.update(stage_evidence)
    if not stages:
        return None, [], {}, {}
    return current, stages, stage_reports, evidence
