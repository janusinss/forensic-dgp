"""Independent saved-pixel recount and metric fixtures; no models or gradients."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1'
RETURN = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    import numpy as np
    from PIL import Image
    import torch
    from scipy.ndimage import minimum_filter, uniform_filter
    from skimage.metrics import structural_similarity
    torch.set_num_threads(4)
    analysis = read(OUT / 'analysis.json')
    assert analysis['complete'] and analysis['forward_cases'] == 500
    for name, digest in analysis['bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    p = read(RETURN / 'protocol.json')
    count, failures = 0, []
    for cohort in p['cohorts']:
        label = cohort['name']
        before = read(RETURN / f'outputs/state0_{label}/before/receipt.json')
        for variant in ['before'] + [v['name'] for v in p['variants']]:
            folder = RETURN / f'outputs/state0_{label}/{variant}'
            current = read(folder / 'receipt.json')
            for case in cohort['cases']:
                assert time.monotonic() - started < 300
                cid = case['id']
                raw = np.load(folder / (cid + '.npy'), allow_pickle=False)
                with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                with Image.open(folder / (cid + '.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                expected = np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8)
                assert np.array_equal(expected, png)
                count += 1
            if variant == 'before': continue
            comparison = read(folder / 'comparison.json')
            for failure in comparison['preservation_against_original']['failures']:
                key = failure['group']
                selected = [i for i, row in enumerate(before['rows']) if key in {
                    'all', 'clear' if row['profile'] == 'clear' else 'degraded',
                    row['source'] + '/all', row['source'] + ('/clear' if row['profile'] == 'clear' else '/degraded'),
                    row['source'] + '/' + row['profile']}]
                metric = {'MSE': 'raw_MSE', 'SSIM': 'raw_SSIM', 'ArcFace_observed_fixed': 'raw_ArcFace'}[failure['metric']]
                a = np.mean([before['rows'][i]['metrics'][metric] for i in selected])
                b = np.mean([current['rows'][i]['metrics'][metric] for i in selected])
                bad = b > a + 1e-12 if metric == 'raw_MSE' else b < a - 1e-6
                failures.append({'cohort': label, 'variant': variant, **failure, 'raw_metric': metric,
                                 'raw_baseline': float(a), 'raw_candidate': float(b), 'same_group_raw_metric_also_fails': bool(bad)})
    assert count == 500 and failures == analysis['PNG_failures']
    assert len(failures) == 11 and sum(not r['same_group_raw_metric_also_fails'] for r in failures) == 9
    path = ROOT / 'scripts/cctv_dgp_delivered_png_guard_v37.py'
    spec = importlib.util.spec_from_file_location('forward_only_fixture', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    rng = np.random.default_rng(370036)
    cases = []
    with torch.inference_mode():
        for kind in ['random', 'integer_edges', 'near_constant', 'boundary_impulse']:
            camera = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            target = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8).astype(np.float32) / np.float32(255)
            raw = rng.random((256, 256, 3), dtype=np.float32)
            if kind == 'integer_edges':
                raw = camera.astype(np.float32) / np.float32(255)
                raw[::2] = np.nextafter(raw[::2], np.float32(0))
            if kind == 'near_constant':
                raw[:] = np.float32(.50001); target[:] = np.float32(.50002)
            if kind == 'boundary_impulse':
                raw[:] = 0; raw[0] = 1; raw[:, -1] = 1
            mask = np.ones((256, 256), bool); mask[70:96, 100:160] = False
            interior = minimum_filter(mask, size=7, mode='constant', cval=0).astype(bool)
            def tensor(a): return torch.from_numpy(np.asarray(a).copy()).permute(2, 0, 1)[None]
            x, y = tensor(raw), tensor(target)
            m, valid = torch.from_numpy(mask)[None, None], torch.from_numpy(interior)[None, None]
            png = np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8)
            canonical = png.astype(np.float32) / np.float32(255)
            actual = helper.delivered_png(x, tensor(camera.astype(np.float32) / np.float32(255)), m)
            assert torch.equal(actual, tensor(canonical)) and not actual.requires_grad
            reference_filter = np.stack([uniform_filter(raw[..., c], size=7) for c in range(3)], 2)
            filter_error = float(np.abs(helper.uniform7_float32(x).numpy()[0].transpose(1, 2, 0) - reference_filter).max())
            _, ssmap = structural_similarity(canonical, target, channel_axis=2, data_range=1., full=True)
            reference_ssim = float(ssmap[interior].astype(np.float64).mean())
            reference_mse = float(np.square((canonical - target)[mask]).astype(np.float64).mean())
            mse_error = abs(float(helper.masked_mse(actual, y, m)[0]) - reference_mse)
            ssim_error = abs(float(helper.masked_ssim(actual, y, valid)[0]) - reference_ssim)
            assert filter_error <= 1e-7 and mse_error <= 1e-12 and ssim_error <= 1e-7
            cases.append({'fixture': kind, 'filter_error': filter_error, 'MSE_error': mse_error, 'SSIM_error': ssim_error})
    result = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'), 'checker_sha256': sha(Path(__file__)),
              'bound_source_files': len(analysis['bindings_sha256']), 'all500_saved_PNG_encodings_exact': True,
              'all11_failure_classifications_recounted': True, 'PNG_only_failures': 9, 'forward_fixtures': cases,
              'coarse_derivative_tested': False, 'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
              'VM_calls': 0, 'app_changed': False, 'goal_complete': False,
              'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert result['seconds'] < 300
    with (OUT / 'independent_forward_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'bound_source_files', 'PNG_only_failures', 'seconds']}), flush=True)


if __name__ == '__main__': main()
