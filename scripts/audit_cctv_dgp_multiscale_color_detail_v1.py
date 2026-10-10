"""Independent fixed counterfactual/filter replay; no models or gradients."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1_return/outputs'
OUT = ROOT / 'outputs/cctv_dgp_multiscale_color_detail_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert(mode)).copy()


def returned_raw(p, arms):
    cache = {arm: {} for arm in ['baseline'] + arms}
    rows = {arm: {r['id']: r for r in read(RETURNED / arm / 'metrics.json')['rows']}
            for arm in cache}
    for begin in range(0, 100, 5):
        ids = [c['id'] for c in p['cases'][begin:begin+5]]
        base_bits = None
        for arm in cache:
            with np.load(RETURNED / arm / 'packs' / f'b{begin//5:02d}.npz', allow_pickle=False) as pack:
                assert pack['ids'].tolist() == ids and pack['shape'].tolist() == [5,256,256,3]
                assert bool(pack['xor']) == (arm != 'baseline')
                planes = pack['planes']
                assert planes.dtype == np.uint8 and planes.shape == (4, 983040)
                flat_bytes = np.stack([planes[k] for k in range(4)], axis=1).ravel()
                bits = flat_bytes.view('<u4').reshape(5,256,256,3)
                if base_bits is None:
                    assert arm == 'baseline'
                    base_bits = bits.copy()
                else:
                    bits = bits ^ base_bits
                values = bits.view('<f4').copy()
            for cid, raw in zip(ids, values):
                assert hashlib.sha256(raw.tobytes()).hexdigest() == rows[arm][cid]['raw_RGB_float32_sha256']
                cache[arm][cid] = raw
    return cache


def main():
    start = time.monotonic()
    plan = read(OUT / 'plan.json')
    result = read(OUT / 'results.json')
    assert result['complete'] and result['plan_sha256'] == sha(OUT / 'plan.json')
    assert not (OUT / 'independent_audit.json').exists()
    for name, digest in {**plan['bindings'], **result['output_bindings']}.items():
        assert sha(ROOT / name) == digest, name
    p = read(PACKET / 'protocol.json')
    raw_cache = returned_raw(p, plan['arms'])
    baseline = read(RETURNED / 'baseline/metrics.json')['groups']
    x = np.linspace(-6, 6, 13, dtype=np.float64)
    kernel = np.exp(-x*x/8)
    kernel /= kernel.sum()
    luma = np.array([.299, .587, .114])
    worst_transform = 0.
    worst_HF = 0.
    checked = 0
    for summary in result['controls']:
        folder = OUT / summary['arm'] / summary['variant']
        metrics = read(folder / 'metrics.json')
        assert [r['id'] for r in metrics['rows']] == plan['exact_case_ids']
        for case, row in zip(p['cases'], metrics['rows']):
            assert time.monotonic()-start < 420
            assert row['source'] == case['source'] and row['profile'] == case['profile']
            base = raw_cache['baseline'][case['id']]
            candidate = raw_cache[summary['arm']][case['id']]
            raw = np.load(folder / 'raw' / (case['id']+'.npy'), allow_pickle=False)
            mask = pixels(PACKET / case['observed'], 'L') > 0
            camera = pixels(PACKET / case['input'])
            target = pixels(PACKET / case['target'])
            reference = target.astype(np.float32)/np.float32(255)
            change = candidate.astype(np.float64)-base.astype(np.float64)
            if summary['variant'] == 'remove_constant_RGB_delta':
                retained = change-np.sum(change[mask], axis=0)/np.count_nonzero(mask)
            else:
                assert summary['variant'] == 'retain_base_low_band_sigma2'
                retained = change-cv2.sepFilter2D(change, cv2.CV_64F, kernel, kernel,
                                                borderType=cv2.BORDER_REFLECT)
            expected = np.minimum(1, np.maximum(0, base.astype(np.float64)+retained)).astype(np.float32)
            expected[~mask] = camera[~mask].astype(np.float32)/np.float32(255)
            worst_transform = max(worst_transform, float(np.abs(raw-expected).max()))
            assert raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all()
            assert raw.min() >= 0 and raw.max() <= 1
            assert np.array_equal(raw[~mask], expected[~mask])
            png = pixels(folder / 'previews' / (case['id']+'.png'))
            delivered = (raw*np.float32(255)).astype(np.uint8)
            delivered[~mask] = camera[~mask]
            assert np.array_equal(png, delivered)
            yy, xx = np.mgrid[:256, :256]
            patches = np.zeros((256,256), bool)
            for lx, ly in np.floor(case['landmarks5_canvas_xy']).astype(int):
                patches |= (xx >= lx-12) & (xx < lx+12) & (yy >= ly-12) & (yy < ly+12)
            feature = patches & (cv2.erode(mask.astype(np.uint8), np.ones((13,13),np.uint8),
                borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
            interior = cv2.erode(mask.astype(np.uint8), np.ones((7,7),np.uint8),
                borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            target_luma = (target.astype(np.float64)/255*luma).sum(2)
            target_HF = target_luma-cv2.sepFilter2D(target_luma, cv2.CV_64F, kernel, kernel,
                                                  borderType=cv2.BORDER_REFLECT)
            for stage, values in [('raw',raw), ('png',png.astype(np.float32)/np.float32(255))]:
                mse = float(np.square(values-reference)[mask].astype(np.float64).mean())
                assert mse == row[stage]['MSE']
                _, ss = structural_similarity(reference, values, data_range=1, channel_axis=-1,
                                               win_size=7, full=True)
                assert float(ss[interior].astype(np.float64).mean()) == row[stage]['SSIM']
                a = (values.astype(np.float64)*luma).sum(2)
                hf = a-cv2.sepFilter2D(a, cv2.CV_64F, kernel, kernel, borderType=cv2.BORDER_REFLECT)
                error = float(np.square(hf-target_HF)[feature].mean())
                worst_HF = max(worst_HF, abs(error-row[stage]['landmark_high_frequency_MSE']))
            checked += 1
        for stage in ['raw','png']:
            for key, group in metrics['groups'][stage].items():
                chosen = [r for r in metrics['rows'] if key == 'all'
                    or key == ('clear' if r['profile']=='clear' else 'degraded')
                    or key == r['source']+'/all'
                    or key == r['source']+('/clear' if r['profile']=='clear' else '/degraded')
                    or key == r['source']+'/'+r['profile']]
                assert group['cases'] == len(chosen)
                for name in ['MSE','SSIM','landmark_high_frequency_MSE']:
                    assert group[name] == float(np.mean([r[stage][name] for r in chosen]))
            before = baseline[stage]
            after = metrics['groups'][stage]
            bad = []
            for key in sorted(before):
                for name in ['MSE','SSIM']:
                    fails = (after[key][name] > before[key][name]+1e-12 if name == 'MSE'
                             else after[key][name] < before[key][name]-1e-6)
                    if fails:
                        bad.append((key,name))
            gate = summary['comparisons'][stage]
            assert bad == [(r['group'],r['metric']) for r in gate['preservation_failures']]
            gain = 1-after['degraded']['landmark_high_frequency_MSE']/before['degraded']['landmark_high_frequency_MSE']
            assert gain == gate['relative_feature_gain']
            assert gate['pixel_only_requirements_pass'] == (gain >= .01 and not bad)
    assert checked == 400 and worst_transform <= 2e-7 and worst_HF <= 1e-12
    for name in ['local_neural_calls','local_gradient_queries','local_optimizer_updates','mathematical_optimizer_calls',
                 'native_pixels_decoded','final_pixels_decoded']:
        assert result[name] == 0
    assert result['ArcFace_not_recomputed'] and result['visual_usefulness_not_assessed']
    assert not result['model_qualification'] and not result['app_changed']
    for name, digest in {**plan['bindings'], **result['output_bindings']}.items():
        assert sha(ROOT / name) == digest, name
    receipt = dict(complete=True, plan_sha256=sha(OUT/'plan.json'), results_sha256=sha(OUT/'results.json'),
        checker_sha256=sha(Path(__file__)), transformed_TRAIN_records=400, raw_and_PNG_records=800,
        exact_delivered_PNGs=400, source_bindings_verified=len(plan['bindings']),
        maximum_independent_transform_error=worst_transform, maximum_independent_filter_error=worst_HF,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        all_pixel_gate_decisions_unchanged=True, ArcFace_and_visual_qualification_not_tested=True,
        model_qualification=False, goal_complete=False, seconds=time.monotonic()-start)
    with (OUT/'independent_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True, cases=400, maximum_transform_error=worst_transform,
               maximum_filter_error=worst_HF, seconds=time.monotonic()-start),flush=True)


if __name__ == '__main__':
    main()
