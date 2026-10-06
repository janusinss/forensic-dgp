"""Independent saved-array and exact review-cell verification; no neural calls."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_detail_skip_v23 as a


def verify():
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    folder = ROOT / 'outputs/cctv_dgp_detail_skip_v23_diagnostic'
    returned = ROOT / 'outputs/cctv_dgp_detail_skip_v23_return'
    results = a.read(folder / 'results.json'); p = a.read(a.BUNDLE / 'protocol.json')
    for name, digest in results['source_bindings_sha256'].items():
        a.require(a.sha(a.safe(ROOT, name)) == digest, 'Diagnostic source changed')
    for name, digest in results['artifacts_sha256'].items():
        a.require(a.sha(a.safe(folder, name)) == digest, 'Diagnostic review sheet changed')
    a.require([r['id'] for r in results['rows']] == [c['id'] for c in p['cases']], 'All50 diagnostic cases required')
    for c, row in zip(p['cases'], results['rows']):
        mask = a.observed(a.BUNDLE / c['observed'])
        raw = a.raw_rgb(returned / 'outputs/update50' / (c['id'] + '.npy'))
        base = a.raw_rgb(a.BUNDLE / c['raw_dgp'])
        actual = a.rgb(returned / 'outputs/update50' / (c['id'] + '.png'))
        original = a.rgb(returned / 'outputs/update0' / (c['id'] + '.png'))
        delta = (raw - base)[mask].astype(np.float64)
        byte = actual.astype(np.int16) - original.astype(np.int16)
        expected = {'saved_postclip_correction_RMS': float(np.sqrt(np.square(delta).mean())),
            'saved_postclip_correction_maximum': float(np.abs(delta).max()),
            'changed_PNG_pixels': int(np.any(byte != 0, axis=2).sum()), 'changed_PNG_values': int((byte != 0).sum()),
            'maximum_PNG_byte_change': int(np.abs(byte).max()), 'PNG_identical': bool(np.array_equal(actual, original))}
        a.numeric_tree({k: row[k] for k in expected}, expected, 1e-15, c['id'])
        target = a.rgb(a.BUNDLE / c['target']); feature = a.feature_mask(c, mask)
        def raw_error(value):
            from scipy.ndimage import convolve1d
            kernel = np.exp(-.5*(np.arange(-6,7,dtype=np.float64)/2)**2);kernel/=kernel.sum()
            y = ((value.astype(np.float64)-target.astype(np.float64)/255)*[.299,.587,.114]).sum(2)
            high = y-convolve1d(convolve1d(y,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
            return float(np.square(high)[feature].mean())
        expected_errors = {'raw_baseline_feature_MSE': raw_error(base), 'raw_update50_feature_MSE': raw_error(raw),
            'baseline_PNG_feature_MSE': a.detail_metric(original,target,feature), 'update50_PNG_feature_MSE': a.detail_metric(actual,target,feature)}
        a.numeric_tree({k:row[k] for k in expected_errors},expected_errors,1e-15,c['id']+'/raw_PNG_errors')
        a.require(row['CPU_raw_replay_maximum'] <= 2e-6 and row['source'] == c['source'] and row['profile'] == c['profile'], 'Diagnostic trace scope differs')
    cases = {c['id']: c for c in p['cases']}
    cells = 0
    for sheet in results['sheets']:
        with Image.open(folder / sheet['file']) as image:
            image = np.asarray(image).copy()
        a.require(image.shape == (1516, 1072, 3), 'Original review sheet dimensions differ')
        for entry in sheet['cells']:
            c = cases[entry['case']]; col = entry['column']; x, y = entry['xy']
            if col == 0: source = a.BUNDLE / c['input']
            elif col == 3: source = a.BUNDLE / c['target']
            else: source = returned / ('outputs/update' + ('0' if col == 1 else '50')) / (c['id'] + '.png')
            expected = a.rgb(source); cell = image[y:y + 256, x:x + 256]
            a.require(np.array_equal(expected, cell) and hashlib.sha256(cell.tobytes()).hexdigest() == entry['pixel_sha256'], 'Review cell is not exact source pixels')
            cells += 1
    a.require(len(results['sheets']) == 10 and cells == 200 and results['head_exact_layer_traces'] == 50 and
              results['backward_calls'] == results['optimizer_updates'] == 0, 'Diagnostic operation budget differs')
    receipt = {'complete': True, 'seconds': time.monotonic() - started, 'checker_sha256': a.sha(Path(__file__)),
        'results_sha256': a.sha(folder / 'results.json'), 'source_bindings_verified': len(results['source_bindings_sha256']),
        'sheet_bindings_verified': 10, 'raw_PNG_change_rows': 50, 'exact_review_cells': cells,
        'internal_stage_measurements': 'Known-source exact-layer trace; final output matches independently replayed VM raws. Intermediate field arrays are not independently recomputed by this saved-array check.',
        'neural_calls': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'app_promotion': False, 'goal_complete': False}
    a.write(folder / 'independent_saved_diagnostic_audit.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    verify()
