"""Independent saved scalar/vector/array arithmetic; no learned-model calls."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve1d
    started = time.monotonic()
    folder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1'
    result = read(folder / 'results.json')
    plan = read(folder / 'plan.json')
    bundle = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
    returned = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    p = read(bundle / 'protocol.json')
    assert result['plan_sha256'] == sha(folder / 'plan.json')
    assert sha(bundle / 'protocol.json') == plan['protocol_sha256']
    assert plan['states'] == [0, 50] and plan['maximum_recognizer_forwards'] == 30
    assert 0 < result['seconds'] < plan['seconds_cap'] == 300
    for name, value in result['source_bindings_sha256'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == value, name
    for name, value in result['artifacts_sha256'].items():
        assert sha(folder / name) == value
    assert len(result['artifacts_sha256']) == 30
    assert result['counts'] == {'recognizer_forwards': 30, 'saved_prediction_replays': 20}
    assert all(result[key] == 0 for key in ['head_forwards', 'DGP_forwards', 'gradient_calls', 'backward_calls', 'optimizer_updates'])
    runner = ROOT / 'scripts/audit_cctv_dgp_batchmatched_identity_v26_loss_v1.py'
    assert result['runner_sha256'] == plan['runner_sha256'] == sha(runner)
    tree = ast.parse(runner.read_text(encoding='utf-8'), feature_version=(3, 10))
    assert any(isinstance(n, ast.With) and any(isinstance(i.context_expr, ast.Call)
        and isinstance(i.context_expr.func, ast.Attribute) and i.context_expr.func.attr == 'inference_mode'
        for i in n.items) for n in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ['backward', 'autograd', 'grad', 'step', 'AdamW', 'Adam', 'SGD', 'requires_grad_']
    assert [r['id'] for r in result['rows']] == plan['case_order'] == [c['id'] for c in p['cases']]
    assert plan['fixed_batches'] == [list(range(i, i + 5)) for i in range(0, 50, 5)]
    assert len(result['batches']) == 20
    normalizers = result['normalizers']
    cohort = read(returned / 'outputs/cohort_loss_setup.json')
    assert normalizers == {key: cohort[key] for key in normalizers}
    maximum_term = maximum_pixel = maximum_feature = maximum_cosine = 0.
    kernel = np.exp(-.5 * (np.arange(-6, 7, dtype=np.float64) / 2) ** 2)
    kernel /= kernel.sum()
    def high(value):
        return value - convolve1d(convolve1d(value, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    def rgb(path):
        with Image.open(path) as im:
            return np.asarray(im).copy()
    def erode(mask, radius):
        size = 2 * radius + 1
        padded = np.pad(mask.astype(np.int64), radius)
        cumulative = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        sums = cumulative[size:, size:] - cumulative[:-size, size:] - cumulative[size:, :-size] + cumulative[:-size, :-size]
        return sums == size * size
    f32 = np.float32
    term_keys = ['degraded_landmark_detail', 'degraded_observed_detail', 'degraded_pixel',
                 'clear_baseline_anchor', 'pixel_regression', 'SSIM_regression', 'ArcFace_regression']
    for row, case in zip(result['rows'], p['cases']):
        assert row['source'] == case['source'] and row['profile'] == case['profile']
        mask = rgb(bundle / case['observed']) > 0
        target = rgb(bundle / case['target']).astype(np.float32) / f32(255)
        base = np.load(bundle / case['raw_dgp'], allow_pickle=False)
        interior = erode(mask, 6)
        feature = np.zeros(mask.shape, bool)
        for point in case['landmarks5_canvas_xy']:
            x, y = np.floor(point).astype(int)
            feature[max(0, y - 12):min(256, y + 12), max(0, x - 12):min(256, x + 12)] = True
        feature &= interior
        truth = np.load(folder / ('truth_' + row['reference'] + '.npy'), allow_pickle=False)
        assert truth.shape == (512,) and truth.dtype == np.float32
        for update in ['0', '50']:
            state = row['states'][update]
            values = {key: f32(state[key]) for key in [
                'raw_feature_MSE', 'raw_interior_MSE', 'raw_pixel_MSE', 'base_raw_pixel_MSE',
                'raw_SSIM', 'base_raw_SSIM', 'raw_ArcFace', 'same_call_base_raw_ArcFace', 'baseline_anchor_MSE']}
            w, clear = (f32(0), f32(1)) if case['profile'] == 'clear' else (f32(1.25), f32(0))
            denominator = max(values['base_raw_pixel_MSE'], f32(1e-5))
            terms = [w * values['raw_feature_MSE'] / f32(normalizers['feature_normalizer']),
                w * f32(.25) * values['raw_interior_MSE'] / f32(normalizers['interior_normalizer']),
                w * f32(.05) * values['raw_pixel_MSE'] / denominator,
                clear * f32(.05) * values['baseline_anchor_MSE'] / denominator,
                f32(2) * max(f32(0), (values['raw_pixel_MSE'] - values['base_raw_pixel_MSE']) / denominator),
                f32(5) * max(f32(0), values['base_raw_SSIM'] - values['raw_SSIM']),
                f32(5) * max(f32(0), values['same_call_base_raw_ArcFace'] - values['raw_ArcFace'])]
            assert list(state['terms']) == term_keys
            for key, value in zip(term_keys, terms):
                difference = abs(float(value) - state['terms'][key])
                maximum_term = max(maximum_term, difference)
                assert difference <= 1e-7, (case['id'], update, key)
            assert sum(state['terms'].values()) == state['objective']
            raw = np.load(returned / ('outputs/update' + update) / (case['id'] + '.npy'), allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
            raw_pixel = float(np.square((raw - target)[mask]).astype(np.float64).mean())
            base_pixel = float(np.square((base - target)[mask]).astype(np.float64).mean())
            anchor = float(np.square((raw - base)[mask]).astype(np.float64).mean())
            for key, expected in [('raw_pixel_MSE', raw_pixel), ('base_raw_pixel_MSE', base_pixel), ('baseline_anchor_MSE', anchor)]:
                difference = abs(state[key] - expected)
                maximum_pixel = max(maximum_pixel, difference)
                assert difference <= 2e-8, (case['id'], update, key)
            delta = ((raw.astype(np.float64) - target.astype(np.float64)) * np.array([.299, .587, .114])).sum(2)
            high_delta = high(delta)
            for key, measurement in [('raw_feature_MSE', feature), ('raw_interior_MSE', interior)]:
                difference = abs(float(np.square(high_delta)[measurement].mean()) - state[key])
                maximum_feature = max(maximum_feature, difference)
                assert difference <= 2e-9, (case['id'], update, key)
            vectors = np.load(folder / state['identity_vectors'], allow_pickle=False)
            assert vectors.shape == (10, 512) and vectors.dtype == np.float32 and np.isfinite(vectors).all()
            j = state['identity_row']
            for key, vector in [('same_call_base_raw_ArcFace', vectors[j]), ('raw_ArcFace', vectors[5 + j])]:
                difference = abs(float((vector * truth).sum(dtype=np.float32)) - state[key])
                maximum_cosine = max(maximum_cosine, difference)
                assert difference <= 2e-7, (case['id'], update, key)
            if update == '0':
                assert np.array_equal(raw, base) and np.array_equal(vectors[j], vectors[j + 5])
                assert all(state['terms'][key] == 0 for key in term_keys[3:])
            if case['profile'] == 'clear':
                assert all(state['terms'][key] == 0 for key in term_keys[:3])
            else:
                assert state['terms']['clear_baseline_anchor'] == 0
    for batch in result['batches']:
        chosen = [r for r in result['rows'] if r['id'] in batch['ids']]
        assert len(chosen) == 5
        state = str(batch['update'])
        assert [r['id'] for r in chosen] == [p['cases'][i]['id'] for i in plan['fixed_batches'][batch['batch']]]
        difference = abs(sum(r['states'][state]['objective'] for r in chosen) / 5 - batch['compiled_objective'])
        assert difference <= plan['compiled_loss_absolute_tolerance'] == 1e-6
    for label, group in result['groups'].items():
        chosen = [r for r in result['rows'] if label in {'all', 'clear' if r['profile'] == 'clear' else 'degraded',
                  r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'), r['source'] + '/' + r['profile']}]
        assert len(chosen) == group['cases']
        for update in ['0', '50']:
            assert abs(sum(r['states'][update]['objective'] for r in chosen) / len(chosen) - group['states'][update]['objective']) < 1e-12
            for key in term_keys:
                assert abs(sum(r['states'][update]['terms'][key] for r in chosen) / len(chosen) - group['states'][update]['terms'][key]) < 1e-12
            for key in term_keys[4:]:
                assert sum(r['states'][update]['terms'][key] > 0 for r in chosen) == group['states'][update]['active_penalty_cases'][key]
        assert group['objective_reduction'] == group['states']['0']['objective'] - group['states']['50']['objective']
    assert len(result['groups']) == 17
    assert abs(result['groups']['all']['states']['0']['objective'] - 1.3) <= 1e-6
    assert not result['GPU_gradient_or_optimizer_causality_proven'] and not result['VM_actions']
    receipt = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'results_sha256': sha(folder / 'results.json'), 'source_bindings_verified': len(result['source_bindings_sha256']),
        'saved_vectors_verified': 30, 'case_states_verified': 100, 'group_states_verified': 34,
        'maximum_term_assembly_difference': maximum_term, 'maximum_raw_pixel_difference': maximum_pixel,
        'maximum_fixed_filter_difference': maximum_feature, 'maximum_saved_vector_cosine_difference': maximum_cosine,
        'compiled_CPU_original_source_and_same_batch_context_checked': True,
        'limit': 'Independent scalar/vector/pixel/high-pass arithmetic and source checks, not independent recognizer forward or SSIM field replay, L4 derivative rerun, or optimizer-trajectory causality.',
        'neural_calls': 0, 'gradient_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with (folder / 'independent_saved_loss_audit.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
