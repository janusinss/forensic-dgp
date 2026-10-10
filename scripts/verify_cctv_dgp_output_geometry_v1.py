"""Independent saved-input/oracle-formula arithmetic; never evaluates a network."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_output_geometry_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm'


def main():
    start = time.monotonic()
    target = OUT / 'independent_audit.json'
    assert not target.exists()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))

    def sha(path):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert time.monotonic() - start < 300
        return digest

    plan, analysis = read(OUT / 'plan.json'), read(OUT / 'analysis.json')
    p = read(BUNDLE / 'protocol.json')
    assert analysis['plan_sha256'] == sha(OUT / 'plan.json')
    for name, digest in analysis['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert plan['parent_protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    assert len(plan['selected']) == len(analysis['rows']) == len(p['cases']) == 145
    assert [row['id'] for row in analysis['rows']] == [case['id'] for case in p['cases']]
    assert {row['id'] for row in plan['selected']} == {case['id'] for case in p['cases']}
    assert len({case['reference_id'] for case in p['cases']}) == analysis['references'] == 29
    assert plan['variants'] == ['original', 'target_mean_centered_oracle', 'bounded_target_mean_centered_oracle']
    assert plan['delta_limit'] == .5 - 1e-6 and plan['cap_seconds'] == 300
    selected = {row['id']: row for row in plan['selected']}
    limit = plan['delta_limit']
    k = cv2.getGaussianKernel(13, 2, cv2.CV_64F).ravel()
    maximum_MSE_error, maximum_HF_error, maximum_SSIM_error = 0., 0., 0.
    count, bounded_count = 0, 0
    for case, row in zip(p['cases'], analysis['rows']):
        assert row['saved_original'] == selected[case['id']]
        with np.load(ROOT / selected[case['id']]['npz'], allow_pickle=False) as arrays:
            base = arrays['original_rgb'].copy()
        with Image.open(BUNDLE / case['input']) as image:
            camera = np.asarray(image).copy()
        with Image.open(BUNDLE / case['target']) as image:
            target_pixels = np.asarray(image).copy()
        with Image.open(BUNDLE / case['observed']) as image:
            mask = np.asarray(image.convert('L')) > 0
        base[~mask] = camera[~mask].astype(np.float32) / np.float32(255)
        desired = target_pixels[mask].astype(np.float64) / 255 - base[mask].astype(np.float64)
        avg = desired.sum(0) / len(desired)
        low, high = desired.min(0), desired.max(0)
        centered_values = desired - avg
        bounded_values = np.minimum(np.maximum(desired - (high + low) / 2, -limit), limit)
        bounded_centered = bounded_values - bounded_values.sum(0) / len(bounded_values)
        assert np.max(np.abs(centered_values.mean(0))) < 1e-12
        assert np.max(np.abs(bounded_centered.mean(0))) < 1e-12
        assert np.allclose(avg, row['required_observed_RGB_mean_shift'], rtol=0, atol=1e-14)
        assert np.array_equal(high - low, row['residual_span_per_channel'])
        constructible = bool((high - low < 2 * limit).all())
        assert constructible == row['uncapped_centered_target_has_bounded_delta_construction']
        if constructible:
            assert np.max(np.abs(centered_values - bounded_centered)) < 1e-12
            bounded_count += 1
        variants = {'original': base}
        for label, correction in zip(plan['variants'][1:], [centered_values, bounded_centered]):
            value = base.copy()
            value[mask] = np.minimum(np.maximum(base[mask].astype(np.float64) + correction, 0), 1).astype(np.float32)
            assert np.array_equal(value[~mask], base[~mask])
            variants[label] = value
        difference = float(np.abs(variants[plan['variants'][1]] - variants[plan['variants'][2]]).max())
        assert difference == row['bounded_and_uncapped_maximum_raw_difference']
        patch = np.zeros((256, 256), bool)
        for xx, yy in np.floor(case['landmarks5_canvas_xy']).astype(int):
            patch[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
        padded = np.pad(mask.astype(np.int64), 6)
        integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        counts = integral[13:, 13:] - integral[:-13, 13:] - integral[13:, :-13] + integral[:-13, :-13]
        support = patch & (counts == 169)
        interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        ref = target_pixels.astype(np.float32) / np.float32(255)
        ref_luma = (target_pixels.astype(np.float64) / 255 * [.299, .587, .114]).sum(2)
        ref_high = ref_luma - cv2.sepFilter2D(ref_luma, cv2.CV_64F, k, k, borderType=cv2.BORDER_REFLECT)
        for label, value in variants.items():
            encoded = np.floor(value * np.float32(255)).astype(np.uint8)
            encoded[~mask] = camera[~mask]
            png = encoded.astype(np.float32) / np.float32(255)
            for stage, array in [('raw', value), ('PNG', png)]:
                recorded = row['metrics'][label][stage]
                error = array - ref
                mse = float(np.square(error[mask]).astype(np.float64).mean())
                _, smap = structural_similarity(ref, array, data_range=1, channel_axis=-1, win_size=7, full=True)
                ssim = float(smap[interior].astype(np.float64).mean())
                luma = (array.astype(np.float64) * [.299, .587, .114]).sum(2)
                detail = luma - cv2.sepFilter2D(luma, cv2.CV_64F, k, k, borderType=cv2.BORDER_REFLECT)
                hf = float(np.square(detail - ref_high)[support].mean())
                maximum_MSE_error = max(maximum_MSE_error, abs(mse - recorded['MSE']))
                maximum_HF_error = max(maximum_HF_error, abs(hf - recorded['landmark_high_frequency_MSE']))
                maximum_SSIM_error = max(maximum_SSIM_error, abs(ssim - recorded['SSIM']))
                assert abs(mse - recorded['MSE']) <= 1e-15
                assert abs(hf - recorded['landmark_high_frequency_MSE']) <= 1e-15
                assert abs(ssim - recorded['SSIM']) <= 1e-12
                assert recorded['MAE'] == float(np.abs(error[mask]).astype(np.float64).mean())
                count += 1
    assert count == 870 and bounded_count == analysis['cases_with_bounded_uncapped_construction']
    assert len(analysis['groups']) == 54
    for group in analysis['groups']:
        subset = [row for row in analysis['rows'] if (group['source'] == 'all' or row['source'] == group['source']) and
            (group['profile'] == 'all' or (row['profile'] == 'clear') == (group['profile'] == 'clear'))]
        assert len(subset) == group['cases']
        for key in plan['metrics']:
            assert group[key] == float(np.mean([row['metrics'][group['variant']][group['stage']][key] for row in subset]))
        baseline = float(np.mean([row['metrics']['original'][group['stage']]['landmark_high_frequency_MSE'] for row in subset]))
        assert group['oracle_relative_landmark_high_frequency_gain'] == 1 - group['landmark_high_frequency_MSE'] / baseline
    assert analysis['oracle_target_access_is_unavailable_in_app'] and analysis['ArcFace_not_evaluated']
    assert analysis['full_preservation_gate_not_evaluated'] and analysis['network_representability_not_tested']
    assert analysis['neural_calls'] == analysis['optimizer_updates'] == analysis['gradient_queries'] == 0
    assert not analysis['native_or_DEV_or_reserved_final_used'] and not analysis['app_promotion']
    receipt = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'), 'plan_sha256': sha(OUT / 'plan.json'),
        'cases_checked': 145, 'metrics_rows_checked': count, 'group_rows_checked': 54,
        'mean_centering_and_amplitude_constructions_checked': True,
        'independent_CV2_convolution_checked_against_Scipy': True,
        'maximum_MSE_error': maximum_MSE_error, 'maximum_detail_error': maximum_HF_error,
        'maximum_SSIM_error': maximum_SSIM_error, 'model_calls': 0, 'optimizer_updates': 0,
        'not_network_or_complete_preservation_qualification': True, 'goal_complete': False,
        'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - start, 'cap_seconds': 300}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
