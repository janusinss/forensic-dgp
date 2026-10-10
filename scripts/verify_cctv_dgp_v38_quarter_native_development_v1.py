"""Independently recompute V38-quarter native saved-output composition, routing and cells."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v38_quarter_native_development_v1'
BASE = ROOT / 'outputs/cctv_chokepoint_native_development_v1'
CACHE = ROOT / 'outputs/cctv_chokepoint_native_comparison_v1'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def main():
    start = time.monotonic(); assert not (OUT / 'saved_output_audit.json').exists()
    p, r, ex = [read(OUT / name) for name in ['plan.json', 'results.json', 'execution.json']]
    assert r['complete'] and r['plan_sha256'] == ex['plan_sha256'] == sha(OUT / 'plan.json')
    for name, digest in p['sources_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in r['artifacts_sha256'].items(): assert sha(OUT / name) == digest, name
    assert r['fresh_neural_forwards'] == {'original_DGP': 24, 'candidate_DGP': 24}
    assert r['states_before_after'] == ex['states_before'] and r['seconds'] <= p['budget']['wall_seconds'] == 180
    assert r['local_gradient_calls'] == r['optimizer_updates'] == 0 and r['native_evidence_unpaired']
    assert r['PSNR'] is r['SSIM'] is r['identity_accuracy'] is None and not r['reserved_final_used']
    old = {row['id']: row for row in read(CACHE / 'results.json')['records']}; choices = Counter()
    projection_max = 0.; checked = 0
    for c, row in zip(p['cases'], r['rows']):
        assert c['id'] == row['id'] and c['role'] == 'development' and row['input_only_usable']
        image = pixels(BASE / c['input']); support = pixels(BASE / c['observed'], 'L') > 0
        baseline, candidate, saved = [np.load(OUT / ('raw/' + c['id'] + '_' + key + '.npy'), allow_pickle=False)
                                       for key in ['original_raw', 'candidate_unprojected', 'result']]
        for value in [baseline, candidate, saved]:
            assert value.dtype == np.float32 and value.shape == (256, 256, 3) and np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        assert np.array_equal(candidate[~support], image[~support].astype(np.float32) / np.float32(255))
        delta = (candidate - baseline).astype(np.float64)
        ref = baseline.astype(np.float64) + delta - delta[support].mean(0)
        ref = np.where(support[..., None], ref.clip(0, 1), image.astype(np.float64) / 255)
        error = float(np.abs(ref - saved).max()); assert error <= 2e-6
        projection_max = max(error, projection_max)
        delivered = np.floor(saved * np.float32(255)).astype(np.uint8); delivered[~support] = image[~support]
        assert np.array_equal(delivered, pixels(OUT / row['outputs']['v38_quarter']))
        assert np.array_equal(image, pixels(OUT / row['outputs']['resize']))
        cache_raw = np.load(CACHE / ('raw/' + c['id'] + '_identity_v2.npy'), allow_pickle=False)
        assert float(np.abs(cache_raw - baseline).max()) == row['fresh_original_cache_maximum_error'] <= 1e-5
        for arm, oa in [('phase3', 'phase3'), ('original_dgp', 'identity_v2'), ('codeformer_w1', 'codeformer_w1')]:
            cached = np.load(CACHE / ('raw/' + c['id'] + '_' + oa + '.npy'), allow_pickle=False)
            png = np.floor(cached * np.float32(255)).astype(np.uint8); png[~support] = image[~support]
            assert np.array_equal(png, pixels(CACHE / old[c['id']]['outputs'][oa]))
            assert np.array_equal(png, pixels(OUT / row['outputs'][arm])); checked += 1
        region = cv2.erode(support.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        region[:4] = region[-4:] = False; region[:, :4] = region[:, -4:] = False
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), .6), cv2.CV_32F)[region].var())
        low = cv2.GaussianBlur(gray, (7, 7), 1.5)
        grad = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
        flat = region & (grad <= np.percentile(grad[region], 35))
        response = cv2.filter2D(gray, cv2.CV_32F, np.asarray([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32))
        noise = float(np.median(np.abs(response[flat])) / (6 * .67448975))
        signals = row['quality_signals']
        assert blur == signals['blur_variance'] and noise == signals['noise_sigma_255']
        chosen = 'v38_quarter' if blur < 24 or noise >= 8 else 'resize'; assert chosen == row['auto_selected']
        assert signals['blur_threshold'] == 24 and signals['noise_threshold'] == 8
        assert np.array_equal(pixels(OUT / row['outputs']['v38_quarter_auto']), pixels(OUT / row['outputs'][chosen]))
        for arm in p['arms']: assert np.array_equal(pixels(OUT / row['outputs'][arm])[~support], image[~support])
        choices[chosen] += 1; checked += 1
    cells = 0
    for sheet in r['sheets']:
        with Image.open(OUT / sheet['file']) as im:
            assert im.size == (1604, 1220)
            for cell in sheet['cells']:
                x, y = cell['xy']; value = np.asarray(im.crop((x, y, x + 256, y + 256))).copy()
                assert np.array_equal(value, pixels(OUT / cell['path']))
                assert hashlib.sha256(value.tobytes()).hexdigest() == cell['pixel_sha256']; cells += 1
    assert len(r['rows']) == 24 and checked == 96 and cells == 144 and time.monotonic() - start <= 120
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'results_sha256': sha(OUT / 'results.json'),
               'source_bindings_verified': len(p['sources_sha256']), 'artifact_bindings_verified': len(r['artifacts_sha256']),
               'projection_float64_maximum_error': projection_max, 'raw_PNG_compositions': 96,
               'exact_Auto_aliases': 24, 'all144_sheet_cells_exact': True, 'auto_choices': dict(choices),
               'padding_exact': True, 'native_evidence_unpaired': True,
               'model_states_independently_replayed': False, 'state_scope': 'Execution reports frozen state parity; saved-output arithmetic and all source/checkpoint bindings independently verified',
               'neural_forwards': 0, 'local_gradient_calls': 0, 'optimizer_updates': 0,
               'PSNR': None, 'SSIM': None, 'identity_accuracy': None, 'reserved_final_used': False,
               'app_changed': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'saved_output_audit.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
