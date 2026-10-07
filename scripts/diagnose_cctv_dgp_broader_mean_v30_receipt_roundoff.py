"""Reproduce the failed exact equality using all saved PNGs; no neural calls."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RETURNED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
OUT = ROOT / 'outputs/cctv_dgp_broader_mean_v30_audit_execution_v1/receipt_roundoff_diagnostic.json'


def main():
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve1d
    start = time.monotonic(); assert not OUT.exists()
    p = json.loads((RETURNED / 'protocol.json').read_text())
    receipt = json.loads((RETURNED / 'outputs/early_structure_stop.json').read_text())
    z = np.arange(-6, 7, dtype=np.float64); kernel = np.exp(-.5 * (z / 2)**2); kernel /= kernel.sum()
    high = lambda a: a - convolve1d(convolve1d(a, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    references = {}; values = {}; errors = {}; luma = np.array([.299, .587, .114])
    for state in [0, 50]:
        saved = json.loads((RETURNED / f'outputs/update{state}/metrics.json').read_text())
        selected = []; maximum = 0.
        for case, row in zip(p['case_rows'], saved['rows']):
            assert time.monotonic() - start < 600
            cid = case['id']; assert cid == row['id']
            rid = case['source_person_or_reference']
            if rid not in references:
                with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                padded = np.pad(mask.astype(np.int64), 6); summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                interior = summed[13:, 13:] - summed[:-13, 13:] - summed[13:, :-13] + summed[:-13, :-13] == 169
                feature = np.zeros((256, 256), bool)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior
                references[rid] = (high((target.astype(np.float64) / 255 * luma).sum(2)), feature)
            with Image.open(RETURNED / f'outputs/update{state}/{cid}.png') as im: png = np.asarray(im.convert('RGB')).copy()
            target_high, feature = references[rid]
            metric = float(np.square(high((png.astype(np.float64) / 255 * luma).sum(2)) - target_high)[feature].mean())
            maximum = max(maximum, abs(metric - row['metrics']['landmark_high_frequency_MSE']))
            if case['profile'] != 'clear': selected.append(metric)
        assert len(selected) == 3124 and maximum <= 1e-10
        values[state] = float(np.mean(selected)); errors[state] = maximum
    gain = 1 - values[50] / values[0]
    difference = abs(gain - receipt['relative_feature_error_gain'])
    assert receipt['update'] == 50 and receipt['minimum'] == .01 and receipt['pass'] == (gain >= .01) == False
    assert gain != receipt['relative_feature_error_gain'] and difference < 1e-12
    result = {'complete': True, 'saved_early_gain': receipt['relative_feature_error_gain'], 'CPU_recomputed_gain': gain,
              'absolute_receipt_difference': difference, 'exact_equality_reproduced_failure': True,
              'all_other_gate_predicates_pass': True, 'early_minimum_unchanged': .01, 'gate_pass_unchanged': False,
              'per_snapshot_maximum_feature_metric_error': errors, 'PNGs_read': 7810,
              'proposed_receipt_only_tolerance': 1e-12, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'seconds': time.monotonic() - start}
    with OUT.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
