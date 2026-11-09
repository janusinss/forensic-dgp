"""Conservative optional Protect + Auto-match routing before restoration."""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_closing, find_objects, gaussian_filter, label, uniform_filter, uniform_filter1d

from .blur_match import estimate_tiles
from .preparation import build_preview, json_bytes, png, sha
from .scanline_cleanup import prepare_scanlines

PIXELATION_RADIUS_AT256 = 3.0
PIXELATION_CANDIDATES_AT256 = (2.5, 3.0, 3.5)
SCANLINE_RADIUS_AT256 = 1.5


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


def picture_pixelation(source):
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
    # Recover narrow gaps bracketed by the mask only when at least four
    # overlapping scan windows still support block seams in that direction.
    span = 2 * min(24, round(max(mask.shape) * .035)) + 1
    horizontal_gaps = binary_closing(mask, structure=np.ones((1, span), bool)) & (column_votes >= 4)
    vertical_gaps = binary_closing(mask, structure=np.ones((span, 1), bool)) & (row_votes >= 4)
    gaps = (horizontal_gaps | vertical_gaps) & ~mask
    mask |= gaps
    if analysis is not source:
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
    strength = choose_pixelation_radius(image, mask)
    radius_at256 = strength['selected_radius_at256']
    radius = radius_at256 * max(image.size) / 256
    filtered = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)))
    prepared = source.copy()
    prepared[mask] = filtered[mask]
    changed = np.any(prepared != source, axis=2)
    if not changed.any():
        return None, None, {}

    regions, _ = label(mask)
    bounds = [[cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start]
              for rows, cols in find_objects(regions)]
    full_png = png(prepared)
    mask_png = png(mask.astype(np.uint8) * 255)
    report = {'policy': 'multiscale-picture-block-regions-v3', 'classification_is_approximate': True,
              'native_size': list(image.size), 'analysis_size': list(analysis.shape[:2][::-1]), 'region_bounds': bounds,
              'Gaussian_radius_at256': radius_at256, 'native_Gaussian_radius': radius,
              'strength_selection': strength,
              'recovered_gap_analysis_pixels': int(gaps.sum()),
              'masked_native_pixels': int(mask.sum()), 'changed_native_pixels': int(changed.sum()),
              'full_image_sha256': sha(full_png), 'mask_sha256': sha(mask_png)}
    return prepared, report, {'full-image-smoothed.png': full_png, 'full-image-pixelation-mask.png': mask_png}


def prepare_source(source):
    """Choose full-source preparation before falling back to face analysis."""
    cleaned, report = prepare_scanlines(source)
    if cleaned is None:
        return picture_pixelation(source)
    # A light finishing pass helps V2 with the residual comb edges that a
    # narrow notch leaves behind, as observed in the wild2 comparison.
    image = Image.fromarray(cleaned)
    cleaned = np.asarray(image.filter(ImageFilter.GaussianBlur(SCANLINE_RADIUS_AT256 * max(image.size) / 256)))
    contents = png(cleaned)
    report['Gaussian_radius_at256'] = SCANLINE_RADIUS_AT256
    report['changed_native_pixels'] = int(np.any(cleaned != source, axis=2).sum())
    report['full_image_sha256'] = sha(contents)
    return cleaned, report, {'scanline-cleaned.png': contents}


def decide(source, face):
    block = pixelation_signal(source, face)
    if block['suspected']:
        return {'policy': 'face-versus-surroundings-v1', 'route': 'face_gaussian',
                'reason': 'suspected_pixelation', 'degradation': block}
    x, y, width, height = face
    image = Image.fromarray(source)
    longest = max(image.size)
    scale = min(1., 512 / longest)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    luma = np.asarray(image.convert('L').resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    tiles = estimate_tiles(luma) / (scale * longest / 256)
    rows, columns = tiles.shape
    yy, xx = np.meshgrid((np.arange(rows) + .5) * image.height / rows,
                         (np.arange(columns) + .5) * image.width / columns, indexing='ij')
    inside = ((xx >= x + width * .15) & (xx < x + width * .85) &
              (yy >= y + height * .15) & (yy < y + height * .85))
    outside = ~((xx >= x) & (xx < x + width) & (yy >= y) & (yy < y + height))
    face_samples = tiles[inside & np.isfinite(tiles)]
    outside_samples = tiles[outside & np.isfinite(tiles)]
    report = {'policy': 'face-versus-surroundings-v1', 'route': 'direct',
              'reason': 'insufficient_edges', 'face_tiles': int(face_samples.size),
              'outside_tiles': int(outside_samples.size), 'degradation': block}
    if min(face_samples.size, outside_samples.size) < 3:
        return report
    face_blur, background_blur = float(np.median(face_samples)), float(np.median(outside_samples))
    report.update(face_radius_at256=face_blur, outside_radius_at256=background_blur)
    if face_blur >= max(background_blur * 1.5, background_blur + .75):
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
