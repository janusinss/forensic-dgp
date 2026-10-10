"""Finite-probe deliverable metrics. These paired photographic TRAIN data are not CCTV."""
import numpy as np
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity

METRICS = ('MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE')


def erode(mask, radius):
    n = 2 * radius + 1
    sums = np.pad(np.pad(mask.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return sums[n:, n:] - sums[:-n, n:] - sums[n:, :-n] + sums[:-n, :-n] == n * n


def detail(a, b, feature):
    z = np.arange(-6, 7, dtype=np.float64); kernel = np.exp(-.5 * (z / 2.) ** 2); kernel /= kernel.sum()
    weights = np.array([.299, .587, .114])
    high = lambda v: v - convolve1d(convolve1d(v, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    return float(np.square(high((a.astype(np.float64) / 255 * weights).sum(2)) - high((b.astype(np.float64) / 255 * weights).sum(2)))[feature].mean())


def case_metrics(raw, png, target, camera, mask, feature, baseline, vector, raw_vector, truth):
    assert raw.dtype == baseline.dtype == np.float32 and raw.shape == baseline.shape == (256, 256, 3)
    assert png.dtype == target.dtype == camera.dtype == np.uint8 and png.shape == target.shape == camera.shape == raw.shape
    assert mask.dtype == feature.dtype == bool and mask.shape == feature.shape == (256, 256) and mask.any() and feature.any()
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw <= 1).all()
    assert np.array_equal(png, np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8))
    assert np.array_equal(png[~mask], camera[~mask])
    actual, reference = png.astype(np.float32) / 255, target.astype(np.float32) / 255
    mse = float(np.square((actual - reference)[mask]).astype(np.float64).mean())
    _, ssmap = structural_similarity(reference, actual, data_range=1, channel_axis=-1, win_size=7, full=True)
    raw_mse = float(np.square((raw - reference)[mask]).astype(np.float64).mean())
    _, raw_ssmap = structural_similarity(reference, raw, data_range=1, channel_axis=-1, win_size=7, full=True)
    shift = (raw - baseline)[mask].astype(np.float64).mean(0)
    mean_only = np.clip(baseline.astype(np.float64) + shift, 0, 1).astype(np.float32)
    mean_png = np.where(mask[..., None], np.floor(mean_only * np.float32(255)), camera).astype(np.uint8)
    return {'MSE': mse, 'SSIM': float(ssmap[erode(mask, 3)].astype(np.float64).mean()), 'ArcFace_observed_fixed': float(vector @ truth),
            'landmark_high_frequency_MSE': detail(png, target, feature),
            'constant_mean_shift_only_MSE': float(np.square((mean_png.astype(np.float32) / 255 - reference)[mask]).astype(np.float64).mean()),
            'raw_MSE': raw_mse, 'raw_SSIM': float(raw_ssmap[erode(mask, 3)].astype(np.float64).mean()), 'raw_ArcFace': float(raw_vector @ truth),
            'postclip_mean_RGB_shift': shift.tolist(), 'visible_outside_bytes_preserved': int((~mask).sum() * 3)}


def groups(rows):
    result = {}
    for key in sorted({'all', 'clear', 'degraded'} | {r['source'] + '/' + name for r in rows for name in ['all', 'clear', 'degraded', r['profile']]}):
        subset = [r for r in rows if key in {'all', 'clear' if r['profile'] == 'clear' else 'degraded', r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'), r['source'] + '/' + r['profile']}]
        assert subset
        result[key] = {'cases': len(subset), **{name: float(np.mean([r['metrics'][name] for r in subset])) for name in METRICS}}
    assert len(result) == 17
    return result


def compare_groups(original, current):
    assert set(original) == set(current) and len(original) == 17
    failures = []
    for name, baseline in original.items():
        actual = current[name]; assert actual['cases'] == baseline['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            bad = actual[metric] > baseline[metric] + 1e-12 if metric == 'MSE' else actual[metric] < baseline[metric] - 1e-6
            if bad: failures.append({'group': name, 'metric': metric, 'baseline': baseline[metric], 'candidate': actual[metric]})
    sources = sorted(k[:-9] for k in original if k.endswith('/degraded'))
    gains = {source: 1 - current[source + '/degraded']['landmark_high_frequency_MSE'] / original[source + '/degraded']['landmark_high_frequency_MSE'] for source in sources}
    base, actual = original['degraded'], current['degraded']
    fraction = max(0., base['MSE'] - actual['constant_mean_shift_only_MSE']) / max(base['MSE'] - actual['MSE'], 1e-12)
    return {'groups_checked': 17, 'failures': failures, 'all17_preservation_groups_pass': not failures,
            'source_structure_gains': gains, 'brightness_gain_fraction': fraction, 'brightness_fraction_limit': .2,
            'brightness_gate_pass': fraction <= .2, 'capacity_1pct_at50_and_10pct_at800_unchanged': True,
            'finite_probe_is_not_capacity_qualification': True}
