"""Independent OpenCV/NumPy check of saved-TRAIN residual-supervision arithmetic."""
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cctv_dgp_residual_supervision_v1 as primitive

OUT = ROOT / 'outputs/cctv_dgp_residual_supervision_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eroded(mask, radius):
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def blur(value):
    taps = np.array([.05, .25, .4, .25, .05], np.float64)
    return cv2.sepFilter2D(value, cv2.CV_64F, taps, taps, borderType=cv2.BORDER_REPLICATE)


def pyramid_band(value):
    low = blur(value)[::2, ::2]
    up = np.zeros_like(value)
    up[::2, ::2] = 4 * low
    return value - blur(up), low


def teacher(base, target, mask, clear):
    if clear:
        return np.zeros_like(base)
    residual = target - base
    limits = (residual[mask].min(0) + residual[mask].max(0)) / 2
    bounded = np.clip(residual - limits, -.499999, .499999)
    result = bounded - bounded[mask].mean(0)
    result[~mask] = 0
    return result


def means(predicted, wanted, mask, patch):
    # Independently stated constants/formula, not imported producer math.
    robust = lambda error, valid: float((np.sqrt(error[valid] ** 2 + 1e-6) - .001).mean())
    values = {'observed_correction': robust(predicted - wanted, mask),
              'landmark_correction': robust(predicted - wanted, patch)}
    total = 0
    a, b = predicted, wanted
    for level, weight in enumerate([1., .5, .25]):
        aa, a = pyramid_band(a)
        bb, b = pyramid_band(b)
        radius = 6 * (2 ** level) - 2
        valid = eroded(mask, radius)[::2 ** level, ::2 ** level]
        assert valid.any()
        total += weight * robust(aa - bb, valid)
    values['RGB_pyramid_correction'] = total / 1.75
    return values


def tensor(value, dtype=torch.float64):
    return torch.from_numpy(value.transpose(2, 0, 1).copy())[None].to(dtype)


def regressions():
    mask = np.zeros((256, 256), bool)
    mask[16:240, 20:236] = True
    mask[120:132, 120:132] = False
    patch = np.zeros_like(mask)
    patch[80:104, 80:104] = True
    support = torch.from_numpy(mask)[None, None]
    feature = torch.from_numpy(patch)[None, None]
    rng = np.random.default_rng(991204)
    a = rng.normal(0, .05, (256, 256, 3))
    b = rng.normal(0, .05, (256, 256, 3))
    values = primitive.correction_terms(tensor(a), tensor(b), support, feature)
    changed = a.copy()
    changed[~mask] += 100
    alternate = primitive.correction_terms(tensor(changed), tensor(b), support, feature)
    padding_error = max(abs(float(values[k][0]) - float(alternate[k][0])) for k in values)
    assert padding_error <= 1e-12, 'Padding/hole contamination must not affect valid supervision'
    empty = torch.zeros_like(support)
    narrow = torch.zeros_like(support)
    narrow[:, :, 100:105, 100:105] = True
    cases = {
        'empty_support_rejected': lambda: primitive.correction_terms(tensor(a), tensor(b), empty, feature),
        'insufficient_scale_support_rejected': lambda: primitive.correction_terms(tensor(a), tensor(b), narrow, narrow),
        'nonfinite_residual_rejected': lambda: primitive.correction_terms(tensor(a * np.nan), tensor(b), support, feature),
        'feature_outside_observed_rejected': lambda: primitive.correction_terms(tensor(a), tensor(b), support, ~support),
        'local_gradient_path_rejected': lambda: primitive.correction_terms(tensor(a).requires_grad_(True), tensor(b), support, feature),
    }
    rejected = {}
    for name, call in cases.items():
        try:
            call()
        except AssertionError:
            rejected[name] = True
        else:
            raise AssertionError(name)
    pattern = ((np.indices((256, 256)).sum(0) % 2) * 2 - 1).astype(np.float64)
    chromatic = np.zeros((256, 256, 3))
    chromatic[:, :, 0] = .1 * pattern
    chromatic[:, :, 1] = -.1 * .299 / .587 * pattern
    weighted_luma = chromatic @ np.array([.299, .587, .114])
    assert np.abs(weighted_luma).max() < 1e-16
    color_loss = means(chromatic, np.zeros_like(chromatic), mask, patch)['RGB_pyramid_correction']
    assert color_loss > .01
    return {**rejected, 'padding_and_hole_invariance_error': padding_error,
            'RGB_detail_detects_luma_cancelling_structure': True,
            'luma_cancelling_RGB_detail_loss': color_loss}


def main():
    start = time.monotonic()
    destination = OUT / 'independent_audit.json'
    assert not destination.exists(), 'Preserve the earlier audit'
    report = read(OUT / 'analysis.json')
    plan = read(OUT / 'plan.json')
    assert report['complete'] and report['plan_sha256'] == sha(OUT / 'plan.json')
    assert plan['cases'] == 145 and plan['references'] == 29 and plan['cap_seconds'] == 300
    assert plan['support_radii'] == [4, 10, 22] and plan['level_weights'] == [1., .5, .25]
    assert plan['epsilon'] == .001 and plan['delta_limit'] == .499999
    for name, pin in report['source_bindings'].items():
        assert sha(ROOT / name) == pin, name
    p = read(BUNDLE / 'protocol.json')
    rows = {r['id']: r for r in report['rows']}
    assert len(rows) == len(report['rows']) == 145 and set(rows) == {c['id'] for c in p['cases']}
    torch.set_num_threads(1)
    maximum_loss_error = maximum_f32_error = 0
    recomputed = []
    with torch.inference_mode():
        for index, case in enumerate(p['cases']):
            assert time.monotonic() - start < 300
            assert case['role'] == 'train'
            row = rows[case['id']]
            assert row['source'] == case['source'] and row['profile'] == case['profile']
            with np.load(ROOT / row['saved_original_npz'], allow_pickle=False) as arrays:
                base = arrays['original_rgb'].astype(np.float64)
            with Image.open(BUNDLE / case['target']) as im:
                target = np.asarray(im.convert('RGB')).astype(np.float64) / 255
            with Image.open(BUNDLE / case['observed']) as im:
                mask = np.asarray(im.convert('L')) > 0
            with Image.open(BUNDLE / case['input']) as im:
                camera = np.asarray(im.convert('RGB')).astype(np.float32) / np.float32(255)
            base[~mask] = camera[~mask].astype(np.float64)
            wanted = teacher(base, target, mask, case['profile'] == 'clear')
            assert np.abs(wanted[mask].mean(0)).max() < 1e-12
            assert np.array_equal(wanted[~mask], np.zeros_like(wanted[~mask]))
            patch = np.zeros((256, 256), bool)
            for x, y in np.floor(case['landmarks5_canvas_xy']).astype(int):
                patch[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
            patch &= eroded(mask, 6)
            for fraction in plan['fractions']:
                actual = means(wanted * fraction, wanted, mask, patch)
                for key, value in actual.items():
                    maximum_loss_error = max(maximum_loss_error, abs(value - row['losses'][str(fraction)][key]))
                    assert abs(value - row['losses'][str(fraction)][key]) < 2e-12
            initial = means(np.zeros_like(wanted), wanted, mask, patch)
            support = torch.from_numpy(mask.copy())[None, None]
            feature = torch.from_numpy(patch.copy())[None, None]
            clear = torch.tensor([case['profile'] == 'clear'])
            fixed, _ = primitive.projected_teacher(tensor(base, torch.float32), tensor(target, torch.float32), support, clear)
            terms32 = primitive.correction_terms(torch.zeros_like(fixed), fixed, support, feature)
            for key in initial:
                error = abs(initial[key] - float(terms32[key][0]))
                maximum_f32_error = max(maximum_f32_error, error)
                assert error < 5e-7
                values = [row['losses'][str(f)][key] for f in plan['fractions']]
                assert all(a >= b - 2e-12 for a, b in zip(values, values[1:]))
            assert bool(np.abs(wanted).max() > 0) == row['target_has_nonzero_correction']
            recomputed.append({**row, 'initial': initial})
            if (index + 1) % 25 == 0:
                print({'independent_cases': index + 1, 'of': 145}, flush=True)
        checks = regressions()
    fixed_rows = [r for r in recomputed if r['id'] in plan['calibration'] and r['profile'] != 'clear']
    assert len(fixed_rows) == 40
    for key, value in report['normalizers_initial40_degraded'].items():
        expected = max(.001, float(np.mean([r['initial'][key] for r in fixed_rows])))
        assert abs(expected - value) < 2e-12
    for group in report['groups']:
        subset = [r for r in recomputed if (group['source'] == 'all' or r['source'] == group['source']) and
                  (group['profile'] == 'all' or r['profile'] == group['profile'] or
                   (group['profile'] == 'degraded' and r['profile'] != 'clear'))]
        assert len(subset) == group['cases']
        for key, value in group['initial_raw_terms'].items():
            assert abs(float(np.mean([r['initial'][key] for r in subset])) - value) < 2e-12
    receipt = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
               'plan_sha256': sha(OUT / 'plan.json'), 'checker_sha256': sha(Path(__file__).resolve()),
               'cases_checked': 145, 'references': 29, 'term_values_checked': 145 * 4 * 3,
               'group_rows_checked': len(report['groups']), 'source_bindings_checked': len(report['source_bindings']),
               'maximum_OpenCV_vs_Torch_f64_term_error': maximum_loss_error,
               'maximum_Torch_f32_vs_independent_f64_initial_error': maximum_f32_error,
               'regressions': checks, 'neural_model_forwards': 0, 'gradient_queries': 0,
               'backward_calls': 0, 'optimizer_updates': 0, 'learned_capacity_proven': False,
               'full_preservation_or_native_usefulness_proven': False,
               'new_training_packet_prepared': False, 'goal_complete': False,
               'seconds': time.monotonic() - start, 'cap_seconds': 300}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
