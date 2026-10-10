"""Pure saved-output contract for the manual spatial-decoder capacity pilot."""
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity

PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
METRICS = ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_schedule(cases, batches):
    assert len(cases) == 3905 and len({c['id'] for c in cases}) == 3905
    assert all(c['role'] == 'train' for c in cases)
    assert len(batches) == 800
    for ids in batches:
        assert len(ids) == len(set(ids)) == 5 and all(type(i) is int and 0 <= i < 3905 for i in ids)
        group = [cases[i] for i in ids]
        assert len({c['source_person_or_reference'] for c in group}) == 1
        assert [c['profile'] for c in group] == PROFILES
    assert sorted(i for b in batches[:781] for i in b) == list(range(3905))
    assert len({i for b in batches[781:] for i in b}) == 95


def erode(mask, radius):
    size = 2 * radius + 1
    integral = np.pad(np.pad(mask.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size] == size**2


def feature_support(mask, landmarks):
    patch = np.zeros((256, 256), bool)
    for xx, yy in np.floor(np.asarray(landmarks)).astype(int):
        patch[max(0, yy-12):min(256, yy+12), max(0, xx-12):min(256, xx+12)] = True
    patch &= erode(mask, 6)
    assert patch.any()
    return patch


def exported_pixel_metrics(prediction, target, mask):
    """Exact retained PNG metrics; independently AST-compared to the original."""
    actual=prediction.astype(np.float32)/255;ref=target.astype(np.float32)/255
    error=actual-ref;mse=float(np.square(error[mask]).astype(np.float64).mean())
    _,ssmap=structural_similarity(ref,actual,data_range=1,channel_axis=-1,win_size=7,full=True)
    interior=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return {"MSE":mse,"PSNR":float(-10*np.log10(mse)) if mse else None,"perfect_match":mse==0,
            "SSIM":float(ssmap[interior].astype(np.float64).mean()),"MAE":float(np.abs(error[mask]).astype(np.float64).mean())}


def detail_metric(png, target, support):
    z = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-.5 * (z/2)**2); kernel /= kernel.sum()
    luma = np.asarray([.299, .587, .114])
    high = lambda v: v - convolve1d(convolve1d(v, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    a = (png.astype(np.float64)/255*luma).sum(2)
    b = (target.astype(np.float64)/255*luma).sum(2)
    return float(np.square(high(a)-high(b))[support].mean())


def groups(rows):
    keys = {'all', 'clear', 'degraded'} | {r['source']+'/'+s for r in rows for s in ['all', 'clear', 'degraded', r['profile']]}
    result = {}
    for key in sorted(keys):
        selected = [r for r in rows if key in {'all', 'clear' if r['profile']=='clear' else 'degraded',
                    r['source']+'/all', r['source']+('/clear' if r['profile']=='clear' else '/degraded'), r['source']+'/'+r['profile']}]
        assert selected
        result[key] = {'cases': len(selected), **{m: float(np.mean([r['metrics'][m] for r in selected])) for m in METRICS}}
    assert len(result) == 17
    return result


def capacity(baseline, candidate, minimum_gain):
    assert set(baseline) == set(candidate) and len(baseline) == 17
    failures = []
    for key, before in baseline.items():
        after = candidate[key]; assert before['cases'] == after['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            bad = after[metric] > before[metric]+1e-12 if metric == 'MSE' else after[metric] < before[metric]-1e-6
            if bad: failures.append({'group': key, 'metric': metric, 'baseline': before[metric], 'candidate': after[metric]})
    gain = 1-candidate['degraded']['landmark_high_frequency_MSE']/baseline['degraded']['landmark_high_frequency_MSE']
    source_gains = {key: 1-candidate[key]['landmark_high_frequency_MSE']/baseline[key]['landmark_high_frequency_MSE']
                    for key in baseline if key.endswith('/degraded')}
    b, a = baseline['degraded'], candidate['degraded']
    brightness = max(0, b['MSE']-a['constant_mean_shift_only_MSE'])/max(b['MSE']-a['MSE'], 1e-12)
    if brightness > .2: failures.append({'group': 'degraded', 'metric': 'brightness_gain_fraction', 'candidate': brightness, 'maximum': .2})
    return {'relative_feature_gain': gain, 'minimum_gain': minimum_gain, 'source_feature_gains': source_gains,
            'brightness_gain_fraction': brightness, 'preservation_failures': failures,
            'pass': not failures and gain >= minimum_gain and all(v >= 0 for v in source_gains.values())}
