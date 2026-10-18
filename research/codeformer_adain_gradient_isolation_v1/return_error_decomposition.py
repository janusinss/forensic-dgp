"""CPU paired error decomposition; never correct or enhance any output.

Global per-channel mean error plus residual squared error is an exact MSE
identity. It neither explains the MAE gate nor proves identity/structure or an
illumination cause. Residual includes contrast, texture, geometry and noise.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def split_error(prediction, target, observed=None):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError('Aligned HWC RGB arrays required')
    error = prediction.astype(np.float64) - target.astype(np.float64)
    if not np.isfinite(error).all():
        raise ValueError('Finite paired arrays required')
    if observed is None:
        observed = np.ones(error.shape[:2], dtype=bool)
    if observed.shape != error.shape[:2] or observed.dtype != np.bool_ or not observed.any():
        raise ValueError('Exact nonempty observed support required')
    error = error[observed]
    means = error.mean(axis=0, dtype=np.float64)
    bias_mse = float(np.square(means).mean())
    residual_mse = float(np.square(error - means).mean(dtype=np.float64))
    mse = float(np.square(error).mean(dtype=np.float64))
    if not np.isclose(mse, bias_mse + residual_mse, rtol=1e-12, atol=1e-15):
        raise ValueError('MSE decomposition identity differs')
    return {'mae_unchanged': float(np.abs(error).mean(dtype=np.float64)),
        'observed_pixels': int(observed.sum()),
        'mse': mse, 'per_channel_mean_error': means.tolist(),
        'global_channel_bias_MSE_component': bias_mse,
        'residual_MSE_component': residual_mse,
        'signed_mean_sRGB_luminance_error': float(means @ np.array([.2126, .7152, .0722])),
        'components_sum_error': abs(mse - bias_mse - residual_mse)}


def pixels(path):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (256, 256):
            raise ValueError('Exact RGB256 required')
        return np.asarray(image).astype(np.float32) / np.float32(255)


def audit(review):
    review = Path(review).resolve(strict=True)
    metricroot = review / 'metrics_v1'
    meta = read(metricroot / 'metrics_audit.json')
    manifest = read(metricroot / 'visual_manifest.json')
    guards = read(metricroot / 'guards.json')
    assert meta['complete'] and manifest['complete']
    failures = {row['id'] for row in guards['per_case_failures'] if row['state'] == 'isolated_update50' and row['control'] == 'clear'}
    rows, sources = [], {}
    root = review.parents[1]
    packet = root / 'outputs/cctv_codeformer_adain_gradient_isolation_v1_vm'
    p = read(packet/'protocol.json')
    assert sha(packet/'protocol.json') == '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
    def bound(relative):
        for directory in (packet, root/'outputs/cctv_codeformer_fp32_comparison_v1_vm',
                          root/'outputs/cctv_codeformer_adaptation_v2_vm'):
            path = directory / relative
            if path.is_file():
                assert sha(path) == p['files_sha256'][relative]
                sources[str(path)] = sha(path)
                return path
        raise ValueError('Missing declared dependency')
    cases = {'comparison_'+row['id']: row for row in read(bound(p['comparison_cases']))['cases']}
    for case in manifest['cases']:
        if case['evidence'] != 'paired_synthetic' or case['profile'] != 'clear':
            continue
        targetpath = Path(case['target'])
        assert sha(targetpath) == case['target_sha256']
        target = pixels(targetpath)
        sources[str(targetpath)] = sha(targetpath)
        observed = np.ones((256,256), dtype=bool)
        if case['id'].startswith('comparison_'):
            entry = cases[case['id']]
            path = bound(entry['observed_path']); assert sha(path) == entry['observed_sha256']
            with Image.open(path) as image:
                assert image.size == (256,256)
                observed = np.asarray(image.convert('L')) > 0
        for state, record in case['states'].items():
            for view, key in [('raw', 'raw'), ('native_png', 'native_png')]:
                path = Path(record[key]); assert sha(path) == record[key + '_sha256']
                sources[str(path)] = sha(path)
                if view == 'raw':
                    with path.open('rb') as stream:
                        value = np.load(stream, allow_pickle=False)
                        assert stream.read(1) == b''
                    assert value.shape == (256, 256, 3) and value.dtype == np.float32
                else:
                    value = pixels(path)
                rows.append({'id': case['id'], 'state': state, 'view': view,
                    'is_failed_clear_control': case['id'] in failures, **split_error(value, target, observed)})
    assert len(rows) == 61 * 3 * 2
    index = {(row['id'], row['state'], row['view']): row for row in rows}
    deltas = []
    for case in sorted({row['id'] for row in rows}):
        for view in ('raw', 'native_png'):
            for before in ('original_update0', 'retained_FP32_update50'):
                left, right = index[case, 'isolated_update50', view], index[case, before, view]
                deltas.append({'id': case, 'view': view, 'comparison': 'isolated_update50_minus_' + before,
                    'is_failed_clear_control': case in failures,
                    **{key: left[key] - right[key] for key in ('mse', 'mae_unchanged',
                        'global_channel_bias_MSE_component', 'residual_MSE_component',
                        'signed_mean_sRGB_luminance_error')}})
    result = {'complete': True, 'paired_clear_controls': 61, 'rows': rows, 'deltas': deltas,
        'all_control_cases_included_not_only_failures': True,
        'original_outputs_not_changed': True, 'quality_thresholds_not_changed': True,
        'not_an_oracle_output_or_gate_replacement': True,
        'interpretation': 'MSE identity of global per-channel error mean plus residual; residual is not a pure structure/identity measure; this does not explain the MAE gate or prove illumination cause.',
        'input_receipt_sha256': {p.name: sha(p) for p in (metricroot/'metrics_audit.json', metricroot/'visual_manifest.json', metricroot/'guards.json')},
        'read_sources_sha256': sources, 'source_sha256': sha(__file__),
        'neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False}
    path = review / 'clear_error_decomposition_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--review', required=True)
    args = parser.parse_args();result = audit(args.review)
    print(json.dumps({'complete': result['complete'], 'clear_cases': result['paired_clear_controls'],
                      'rows': len(result['rows']), 'outputs_changed': False}))
