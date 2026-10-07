"""Independent saved-array/PNG/vector/metric readback of every520 V29 DEV case."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v29_paired_development_v1'
BASE = ROOT / 'outputs/cctv_dgp_generalization_vm_v15'
ARMS = ['resize', 'original_dgp', 'v29']


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def close(a, b):
    if a is None or b is None or isinstance(a, bool) or isinstance(b, bool): assert a == b
    else: assert abs(a - b) <= 1e-10, (a, b)


def main():
    start = time.monotonic(); assert not (OUT / 'saved_output_audit.json').exists()
    p, r, e = [read(OUT / n) for n in ['plan.json', 'results.json', 'execution.json']]
    assert r['complete'] and r['plan_sha256'] == e['plan_sha256'] == sha(OUT / 'plan.json')
    assert r['seconds'] <= p['budget']['wall_seconds_including_loading'] == 1200
    assert r['model_forwards'] == {'original_DGP': 520, 'candidate_DGP': 520, 'recognizer_target': 104, 'recognizer_output': 1560}
    assert r['states_before_after'] == e['states_before'] and r['optimizer_updates'] == r['local_gradient_calls'] == 0
    assert not r['reserved_final_used'] and not r['app_changed'] and r['paired_photographic_development_not_native']
    for name, h in p['sources_sha256'].items(): assert sha(ROOT / name) == h, name
    for name, h in r['artifacts_sha256'].items(): assert sha(OUT / name) == h, name
    assert len(p['references']) == 104 and len(p['cases']) == len(r['rows']) == 520
    refs = {}; z = np.arange(-6, 7, dtype=np.float64); kernel = np.exp(-.5 * (z/2)**2); kernel /= kernel.sum()
    def high(rgb):
        y = (rgb.astype(np.float64) / 255 * np.asarray([.299, .587, .114])).sum(2)
        return y - convolve1d(convolve1d(y, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    for ref in p['references']:
        target = pixels(BASE / ref['target']); support = pixels(BASE / ref['observed'], 'L') > 0
        feature = np.zeros((256, 256), bool)
        for x, y in np.floor(np.asarray(ref['landmarks5']).reshape(5, 2)).astype(int):
            feature[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
        # Independent integral-image erosion, rather than the runner's OpenCV13.
        padded = np.pad(support.astype(np.int64), 6)
        integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        inner = integral[13:, 13:] - integral[:-13, 13:] - integral[13:, :-13] + integral[:-13, :-13] == 169
        feature &= inner; assert feature.any()
        truth = np.load(OUT / ('embeddings/' + ref['id'] + '_target.npy'), allow_pickle=False)
        assert truth.dtype == np.float32 and truth.shape == (512,) and np.isfinite(truth).all() and abs(float(truth @ truth) - 1) < 1e-5
        refs[ref['id']] = {'target': target, 'mask': support, 'feature': feature, 'truth': truth}
    all_rows = []; projection_error = 0.
    def metric(png, ref):
        actual = png.astype(np.float32) / np.float32(255); target = ref['target'].astype(np.float32) / np.float32(255)
        err = actual - target; mse = float(np.square(err[ref['mask']]).astype(np.float64).mean())
        _, smap = structural_similarity(target, actual, data_range=1, channel_axis=-1, win_size=7, full=True)
        interior = cv2.erode(ref['mask'].astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        return {'MSE': mse, 'PSNR': float(-10*np.log10(mse)) if mse else None, 'perfect_match': mse == 0,
                'SSIM': float(smap[interior].astype(np.float64).mean()),
                'MAE': float(np.abs(err[ref['mask']]).astype(np.float64).mean()),
                'landmark_high_frequency_MSE': float(np.square(high(png) - high(ref['target']))[ref['feature']].mean())}
    for c, row in zip(p['cases'], r['rows']):
        assert time.monotonic() - start <= 600
        assert c['id'] == row['id'] and c['source'] == row['source'] and c['profile'] == row['profile']
        ref = refs[c['reference_id']]; support = ref['mask']; image = pixels(BASE / c['input'])
        base, candidate, final = [np.load(OUT / ('raw/' + c['id'] + '_' + k + '.npy'), allow_pickle=False)
                                  for k in ['original_raw', 'candidate_unprojected', 'result']]
        for a in [base, candidate, final]:
            assert a.dtype == np.float32 and a.shape == (256, 256, 3) and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
        assert np.array_equal(candidate[~support], image[~support].astype(np.float32) / np.float32(255))
        d = (candidate - base).astype(np.float64)
        expected = np.clip(base.astype(np.float64) + d - d[support].mean(0), 0, 1)
        expected[~support] = image[~support].astype(np.float64) / 255
        error = float(np.abs(expected - final).max()); assert error <= 2e-6; projection_error = max(error, projection_error)
        scores = {}
        for arm in ARMS:
            png = pixels(OUT / row['outputs'][arm]); assert np.array_equal(png[~support], image[~support])
            if arm == 'resize': assert np.array_equal(png, image)
            else:
                expected_png = np.floor((base if arm == 'original_dgp' else final) * np.float32(255)).astype(np.uint8)
                expected_png[~support] = image[~support]; assert np.array_equal(png, expected_png)
            vector = np.load(OUT / ('embeddings/' + c['id'] + '_' + arm + '.npy'), allow_pickle=False)
            assert vector.dtype == np.float32 and vector.shape == (512,) and np.isfinite(vector).all() and abs(float(vector @ vector) - 1) < 1e-5
            scores[arm] = metric(png, ref); scores[arm]['ArcFace_observed_fixed'] = float(vector @ ref['truth'])
            for key, value in scores[arm].items(): close(value, row['metrics'][arm][key])
        shift = (final - base)[support].astype(np.float64).mean(0)
        assert np.array_equal(shift, np.asarray(row['postclip_mean_RGB_shift']))
        mean = np.clip(base.astype(np.float64) + shift, 0, 1).astype(np.float32)
        mean_png = np.floor(mean * np.float32(255)).astype(np.uint8); mean_png[~support] = image[~support]
        mean_mse = metric(mean_png, ref)['MSE']; close(mean_mse, row['constant_mean_shift_only_MSE'])
        all_rows.append({'source': c['source'], 'profile': c['profile'], 'scores': scores, 'mean_only': mean_mse})
    failures, gains = [], {}
    for group, reported in r['groups'].items():
        chosen = [x for x in all_rows if group in ['all', 'clear' if x['profile'] == 'clear' else 'degraded', x['source'] + '/all',
                  x['source'] + ('/clear' if x['profile'] == 'clear' else '/degraded'), x['source'] + '/' + x['profile']]]
        assert chosen
        recalculated = {arm: {k: float(np.mean([x['scores'][arm][k] for x in chosen]))
                              for k in ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE']} for arm in ARMS}
        for arm in ARMS:
            assert reported[arm]['cases'] == len(chosen)
            for k, v in recalculated[arm].items(): close(v, reported[arm][k])
        b, a = recalculated['original_dgp'], recalculated['v29']
        for k in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            bad = a[k] > b[k] + 1e-12 if k == 'MSE' else a[k] < b[k] - 1e-6
            if bad: failures.append({'group': group, 'metric': k, 'original': b[k], 'v29': a[k]})
        if group == 'degraded' or group.endswith('/degraded'):
            gains[group] = 1 - a['landmark_high_frequency_MSE'] / b['landmark_high_frequency_MSE']
    assert len(r['groups']) == 17 and failures == r['diagnostic_preservation_failures']
    for k, v in gains.items(): close(v, r['degraded_structure_gain_fraction'][k])
    degraded = [x for x in all_rows if x['profile'] != 'clear']
    b = float(np.mean([x['scores']['original_dgp']['MSE'] for x in degraded]))
    a = float(np.mean([x['scores']['v29']['MSE'] for x in degraded]))
    brightness = max(0, b - float(np.mean([x['mean_only'] for x in degraded]))) / max(b-a, 1e-12)
    close(brightness, r['brightness_gain_fraction'])
    cells = 0
    for sheet in r['sheets']:
        with Image.open(OUT / sheet['file']) as im:
            assert im.size == (1072, 1516)
            for cell in sheet['cells']:
                x, y = cell['xy']; value = np.asarray(im.crop((x, y, x+256, y+256))).copy()
                assert np.array_equal(value, pixels(ROOT / cell['path']))
                assert hashlib.sha256(value.tobytes()).hexdigest() == cell['pixel_sha256']; cells += 1
    assert cells == 200 and len(r['sheets']) == 10 and time.monotonic() - start <= 600
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'results_sha256': sha(OUT / 'results.json'),
               'source_bindings_verified': len(p['sources_sha256']), 'artifact_bindings_verified': len(r['artifacts_sha256']),
               'cases_recomputed': 520, 'exact_PNG_compositions': 1040, 'exact_resize_controls': 520,
               'saved_vectors_checked': 1664, 'all17_groups_recomputed': True, 'all200_preview_cells_exact': True,
               'projection_float64_maximum_error': projection_error,
               'diagnostic_preservation_failures': failures, 'degraded_structure_gain_fraction': gains,
               'brightness_gain_fraction': brightness, 'neural_forwards': 0, 'local_gradient_calls': 0,
               'optimizer_updates': 0, 'native_PSNR_SSIM_claim': False, 'reserved_final_used': False,
               'app_changed': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'saved_output_audit.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'cases_recomputed': 520, 'failures': len(failures), 'seconds': time.monotonic() - start}))


if __name__ == '__main__': main()
