"""Saved-output forward arithmetic only: no DGP, recognizer or autograd calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'
OUT = ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    import numpy as np
    from PIL import Image
    import torch
    import skimage
    torch.set_num_threads(4)
    helper_path = ROOT / 'scripts/cctv_dgp_delivered_png_guard_v37.py'
    spec = importlib.util.spec_from_file_location('new_delivered_forward_only', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    audited = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json'
    reviewed = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/independent_analysis_page_audit.json'
    assert read(audited)['complete'] and read(reviewed)['all700_cells_exact']
    p = read(RETURN / 'protocol.json')
    assert not OUT.exists()
    OUT.mkdir()
    count, max_mse_error, max_ssim_error = 0, 0., 0.
    png_only, rows = [], []
    bindings = {q.relative_to(ROOT).as_posix(): sha(q) for q in [audited, reviewed, helper_path, Path(__file__)]}

    def load_png(path):
        with Image.open(path) as image: return np.asarray(image.convert('RGB')).copy()

    def tensor(array):
        return torch.from_numpy(array.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]

    def erode(mask):
        summed = np.pad(np.pad(mask.astype(np.int64), 3).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return summed[7:, 7:] - summed[:-7, 7:] - summed[7:, :-7] + summed[:-7, :-7] == 49

    def indices(receipt, key):
        return [i for i, r in enumerate(receipt['rows']) if key in {'all', 'clear' if r['profile'] == 'clear' else 'degraded',
            r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'), r['source'] + '/' + r['profile']}]

    try:
        with torch.inference_mode():
            for cohort in p['cohorts']:
                label = cohort['name']
                parent = RETURN / f'outputs/state0_{label}'
                before = read(parent / 'before/receipt.json')
                for variant in ['before'] + [r['name'] for r in p['variants']]:
                    folder = parent / variant
                    receipt = read(folder / 'receipt.json')
                    bindings[(folder / 'receipt.json').relative_to(ROOT).as_posix()] = sha(folder / 'receipt.json')
                    errors = []
                    for begin in range(0, 50, 5):
                        assert time.monotonic() - started < 300
                        cases = cohort['cases'][begin:begin + 5]
                        cameras, targets, masks, raws, delivered = [], [], [], [], []
                        for case in cases:
                            cid = case['id']
                            paths = [MIXED / case['input'], MIXED / case['target'], MIXED / case['observed'], folder / (cid + '.npy'), folder / (cid + '.png')]
                            camera, target = load_png(paths[0]), load_png(paths[1])
                            with Image.open(paths[2]) as image: mask = np.asarray(image).copy() > 0
                            cameras.append(tensor(camera)); targets.append(tensor(target))
                            masks.append(torch.from_numpy(mask.copy())[None, None])
                            raw = np.load(paths[3], allow_pickle=False)
                            raws.append(torch.from_numpy(raw.copy()).permute(2, 0, 1)[None])
                            delivered.append(tensor(load_png(paths[4])))
                            for q in paths: bindings[q.relative_to(ROOT).as_posix()] = sha(q)
                        raw, camera, target, mask = [torch.cat(group) for group in [raws, cameras, targets, masks]]
                        expected = torch.cat(delivered)
                        values = helper.delivered_png(raw, camera, mask)
                        assert torch.equal(values, expected) and not values.requires_grad
                        valid = torch.cat([torch.from_numpy(erode(m[0, 0].numpy()))[None, None] for m in masks])
                        mse = helper.masked_mse(values, target, mask).numpy()
                        ssim = helper.masked_ssim(values, target, valid).numpy()
                        for case, row, m, s in zip(cases, receipt['rows'][begin:begin + 5], mse, ssim, strict=True):
                            assert case['id'] == row['id']
                            me, se = abs(float(m) - row['metrics']['MSE']), abs(float(s) - row['metrics']['SSIM'])
                            max_mse_error, max_ssim_error = max(max_mse_error, me), max(max_ssim_error, se)
                            errors.append({'id': case['id'], 'MSE_forward_error': me, 'SSIM_forward_error': se})
                            count += 1
                    raw_failures = []
                    for key in before['groups']:
                        selection = indices(before, key)
                        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
                            mapped = {'MSE': 'raw_MSE', 'SSIM': 'raw_SSIM', 'ArcFace_observed_fixed': 'raw_ArcFace'}[metric]
                            baseline = float(np.mean([before['rows'][i]['metrics'][mapped] for i in selection]))
                            actual = float(np.mean([receipt['rows'][i]['metrics'][mapped] for i in selection]))
                            bad = actual > baseline + 1e-12 if metric == 'MSE' else actual < baseline - 1e-6
                            if variant != 'before' and bad: raw_failures.append({'group': key, 'metric': mapped, 'baseline': baseline, 'candidate': actual})
                    if variant != 'before':
                        comparison = read(folder / 'comparison.json')
                        for failure in comparison['preservation_against_original']['failures']:
                            selected = indices(before, failure['group'])
                            raw_key = {'MSE': 'raw_MSE', 'SSIM': 'raw_SSIM', 'ArcFace_observed_fixed': 'raw_ArcFace'}[failure['metric']]
                            old_raw = float(np.mean([before['rows'][i]['metrics'][raw_key] for i in selected]))
                            new_raw = float(np.mean([receipt['rows'][i]['metrics'][raw_key] for i in selected]))
                            raw_bad = new_raw > old_raw + 1e-12 if raw_key == 'raw_MSE' else new_raw < old_raw - 1e-6
                            png_only.append({'cohort': label, 'variant': variant, **failure,
                                'raw_metric': raw_key, 'raw_baseline': old_raw, 'raw_candidate': new_raw,
                                'same_group_raw_metric_also_fails': raw_bad})
                    rows.append({'cohort': label, 'variant': variant, 'forward_checks': errors,
                                 'descriptive_all17_raw_MSE_SSIM_ArcFace_failures': raw_failures})
                    print(json.dumps({'forward_only_cases': count, 'of': 500, 'cohort': label, 'variant': variant}), flush=True)
        assert count == 500 and len(png_only) == 11
        assert max_mse_error <= 1e-12 and max_ssim_error <= 1e-7, 'Forward definition mismatch; retain diagnostic, do not prepare gradients'
        assert not any(name in __import__('sys').modules for name in ['dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
        result = {'complete': True, 'forward_cases': count, 'all500_canonical_PNG_tensors_exact': True,
            'maximum_MSE_forward_error': max_mse_error, 'maximum_SSIM_forward_error': max_ssim_error,
            'MSE_forward_tolerance': 1e-12, 'SSIM_forward_tolerance': 1e-7,
            'all17_raw_metric_checks': rows, 'PNG_failures': png_only,
            'PNG_failures_without_same_group_raw_failure': sum(not r['same_group_raw_metric_also_fails'] for r in png_only),
            'PNG_failures_with_same_group_raw_failure': sum(r['same_group_raw_metric_also_fails'] for r in png_only),
            'runtime': {'torch': torch.__version__, 'skimage': skimage.__version__, 'device': 'cpu', 'threads': 4},
            'no_models_loaded': True, 'neural_calls': 0, 'local_gradient_calls': 0, 'backwards': 0,
            'optimizer_updates': 0, 'VM_calls': 0, 'coarse_derivative_tested': False,
            'finite_preservation_implied': False, 'app_processing_changed': False, 'goal_complete': False,
            'bindings_sha256': bindings, 'seconds': time.monotonic() - started, 'cap_seconds': 300}
        write(OUT / 'analysis.json', result)
        print(json.dumps({k: result[k] for k in ['complete', 'forward_cases', 'maximum_MSE_forward_error', 'maximum_SSIM_forward_error',
            'PNG_failures_without_same_group_raw_failure', 'seconds']}), flush=True)
    except BaseException as error:
        write(OUT / 'failure.json', {'complete': False, 'cause': str(error), 'traceback': traceback.format_exc(),
            'checked_outputs': count, 'maximum_MSE_forward_error': max_mse_error, 'maximum_SSIM_forward_error': max_ssim_error,
            'local_gradients': 0, 'optimizer_updates': 0, 'automatic_repeat_permitted': False})
        raise


if __name__ == '__main__':
    main()
