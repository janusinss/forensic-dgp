"""Check prospective finite-output arithmetic on audited V32 arrays; no neural work."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.ndimage import convolve1d
from PIL import Image
from cctv_dgp_loss_cone_probe_v33_metrics import case_metrics, erode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation/metric_arithmetic_audit_v1'
PRIOR = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(); OUT.mkdir(); started = time.monotonic()
    p = json.loads((ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm/protocol.json').read_text())
    write(OUT / 'plan.json', {'complete': True, 'saved_raw_cases': 200, 'terms_verified': 'First six components; no new raw recognizer vectors requested.',
          'purpose': 'Verify NumPy/scikit-image reassembly against audited actual VM loss receipts before accepting the prospective finite-step reviewer.',
          'cap_seconds': 180, 'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py']},
          'loss_value_tolerance': 1e-4, 'neural_calls': 0, 'optimizer_updates': 0})
    z = np.arange(-6, 7, dtype=np.float64); k = np.exp(-.5 * (z / 2) ** 2); k /= k.sum(); luma = np.array([.299, .587, .114])
    high = lambda a: a - convolve1d(convolve1d(a, k, axis=0, mode='reflect'), k, axis=1, mode='reflect')
    count, worst, rows = 0, 0., []
    for state in [0, 50]:
        for cohort in p['cohorts']:
            label = cohort['name']; folder = PRIOR / f'outputs/state{state}_{label}'
            expected = np.asarray(json.loads((folder / 'receipt.json').read_text())['component_values'])[:6]
            values = []
            for case in cohort['cases']:
                assert time.monotonic() - started < 180
                cid = case['id']; raw = np.load(folder / (cid + '.npy'), allow_pickle=False)
                with Image.open(folder / (cid + '.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                feature = np.zeros((256, 256), bool); interior = erode(mask, 6)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior
                initial = PRIOR / f'outputs/state0_{label}'; base = np.load(initial / (cid + '.npy'), allow_pickle=False)
                vector = np.load(folder / (cid + '_embedding.npy'), allow_pickle=False); truth = np.load(initial / (cid + '_target_embedding.npy'), allow_pickle=False)
                a = case_metrics(raw, png, target, camera, mask, feature, base, vector, vector, truth)
                base_png = np.where(mask[..., None], np.floor(base * np.float32(255)), camera).astype(np.uint8)
                b = case_metrics(base, base_png, target, camera, mask, feature, base, vector, vector, truth)
                delta = high((raw.astype(np.float64) * luma).sum(2)) - high((target.astype(np.float64) / 255 * luma).sum(2))
                f, inside = float(np.square(delta)[feature].mean()), float(np.square(delta)[interior].mean())
                degraded = case['profile'] != 'clear'; denom = max(b['raw_MSE'], 1e-5)
                anchor = float(np.square((raw - base)[mask]).astype(np.float64).mean())
                values.append([1.25 * f / p['normalizers'][0] if degraded else 0, .3125 * inside / p['normalizers'][1] if degraded else 0,
                               .0625 * a['raw_MSE'] / denom if degraded else 0, .05 * anchor / denom if not degraded else 0,
                               2 * max(0., (a['raw_MSE'] - b['raw_MSE']) / denom), 5 * max(0., b['raw_SSIM'] - a['raw_SSIM'])])
                count += 1
            actual = np.asarray(values).mean(0); error = float(np.abs(actual - expected).max()); assert error <= 1e-4
            worst = max(worst, error); rows.append({'state': state, 'cohort': label, 'component_maximum_error': error, 'actual': actual.tolist(), 'expected': expected.tolist()})
    assert count == 200
    write(OUT / 'audit.json', {'complete': True, 'cases': count, 'rows': rows, 'maximum_component_error': worst,
          'saved_PNG_outside_bytes_checked': True, 'neural_calls': 0, 'gradient_queries': 0, 'parameter_assignments': 0, 'optimizer_updates': 0,
          'raw_ArcFace_reassembly_requires_returned_raw_vectors': True, 'seconds': time.monotonic() - started})
    print(json.dumps({'complete': True, 'cases': count, 'maximum_component_error': worst, 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__': main()
