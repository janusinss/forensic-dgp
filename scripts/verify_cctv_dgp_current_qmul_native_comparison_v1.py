"""Independent saved-pixel, input geometry and current-DGP CPU replay audit."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_current_qmul_native_comparison_v1'
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def main():
    start = time.monotonic(); p = read(OUT/'plan.json'); r = read(OUT/'results.json')
    assert r['complete'] and r['plan_sha256'] == sha(OUT/'plan.json') and p['frozen_before_outputs']
    assert r['before'] == r['after'] and r['same_exact_inputs']
    assert r['model_forwards'] == {'phase3': 24, 'current_dgp': 24, 'codeformer_w1': 24}
    assert r['seconds'] <= p['budget']['worker_seconds_including_loading']
    assert r['PSNR'] is None and r['SSIM'] is None and r['identity_accuracy'] is None
    assert r['optimizer_updates'] == r['gradient_queries'] == 0 and not r['reserved_pixels_used']
    for n, h in p['sources_sha256'].items(): assert sha(ROOT/n) == h, n
    for n, h in r['artifacts_sha256'].items(): assert sha(OUT/n) == h, n
    assert [c['id'] for c in p['cases']] == [c['id'] for c in r['records']]
    geometry_cases = 0; raw_arrays = 0; pngs = 0; cells = 0
    for c, record in zip(p['cases'], r['records']):
        assert c['role'] == 'development' and c['source_country'] is None
        native = pixels(ROOT/'outputs/cctv_native_development_v2'/c['source_file'])
        h, w = native.shape[:2]; side = max(h, w); left, top = (side-w)//2, (side-h)//2
        canvas = Image.new('RGB', (side, side), (128, 128, 128)); canvas.paste(Image.fromarray(native), (left, top))
        common = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR)).copy()
        mask = Image.new('L', (side, side), 0); mask.paste(255, (left, top, left+w, top+h))
        observed = np.asarray(mask.resize((256, 256), Image.Resampling.NEAREST)) > 0
        assert np.array_equal(observed, pixels(OUT/record['observed'], 'L') > 0)
        assert np.array_equal(common, pixels(OUT/record['outputs']['resize']))
        normalized = common.astype(np.float32)/np.float32(255)
        assert hashlib.sha256(normalized.tobytes()).hexdigest() == record['input_tensor_float32_SHA256']
        assert record['input_review'] == next(x for x in p['input_reviews'] if x['id'] == c['id'])
        geometry_cases += 1; pngs += 1
        for arm in p['arms'][1:]:
            raw = np.load(OUT/record['raw'][arm], allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            expected = np.floor(raw*np.float32(255)).astype(np.uint8); expected[~observed] = common[~observed]
            actual = pixels(OUT/record['outputs'][arm]); assert np.array_equal(expected, actual)
            change = float(np.abs(actual.astype(np.float64)-common.astype(np.float64))[observed].mean())
            assert change == record['observed_MAE_change_255_diagnostic_only'][arm]
            raw_arrays += 1; pngs += 1
    for page_index, name in enumerate(r['pages']):
        sheet = pixels(OUT/name); assert sheet.shape == (24+6*292, 4*268, 3)
        for i, record in enumerate(r['records'][page_index*6:(page_index+1)*6]):
            for j, arm in enumerate(p['arms']):
                assert np.array_equal(sheet[48+292*i:48+292*i+256, 5+268*j:5+268*j+256], pixels(OUT/record['outputs'][arm])); cells += 1
    import torch
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    torch.set_num_threads(4)
    checkpoint, fingerprint = p['checkpoints']['current_dgp']
    model, _ = load_frozen_dgp_restorer(ROOT/checkpoint, expected_sha256=fingerprint)
    before = state_hash(model); error = 0.; replay_cases = 0
    with torch.inference_mode():
        for record in r['records']:
            assert time.monotonic()-start <= 180
            common = pixels(OUT/record['outputs']['resize'])
            x = torch.from_numpy(common.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
            raw = model(x)[0].permute(1, 2, 0).numpy().copy()
            error = max(error, float(np.abs(raw-np.load(OUT/record['raw']['current_dgp'], allow_pickle=False)).max())); replay_cases += 1
    assert error <= 3e-6 and state_hash(model) == before == r['before']['current_dgp']
    for n, h in p['sources_sha256'].items(): assert sha(ROOT/n) == h
    result = {'complete': True, 'plan_sha256': sha(OUT/'plan.json'), 'results_sha256': sha(OUT/'results.json'),
        'checker_sha256': sha(Path(__file__)), 'source_bindings_verified': len(p['sources_sha256']),
        'artifacts_verified': len(r['artifacts_sha256']), 'geometry_cases': geometry_cases,
        'raw_arrays_verified': raw_arrays, 'PNG_compositions_verified': pngs, 'unscaled_gallery_cells_verified': cells,
        'fresh_current_DGP_replay_cases': replay_cases, 'maximum_current_DGP_raw_error': error,
        'Phase3_CodeFormer_fresh_replay_performed': False, 'model_state_unchanged': True,
        'local_gradient_queries': 0, 'optimizer_updates': 0, 'reserved_pixels_used': False,
        'visual_review_pending': True, 'independent_final_review': False, 'model_qualification': False,
        'seconds': time.monotonic()-start}
    with (OUT/'independent_saved_output_audit.json').open('x', encoding='utf-8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(result)


if __name__ == '__main__': main()
