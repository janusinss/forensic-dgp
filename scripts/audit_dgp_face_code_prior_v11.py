"""Independent no-forward audit of V11 interface/capacity artifacts."""
import ast
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import cctv_dgp_face_prior_v10 as v
import check_dgp_face_code_prior_v11_r2 as check


def close(a, b):
    if isinstance(a, bool) or a is None or isinstance(a, str):
        v.require(a == b, 'Recorded scalar differs')
    else:
        v.require(np.isclose(a, b, rtol=1e-7, atol=1e-9), 'Recorded metric differs')


def audit():
    start = time.monotonic(); p = check.verify(); out = check.OUT / 'run'
    v.require(not (out / 'independent_audit.json').exists(), 'Preserve earlier audit')
    r = v.read(out / 'results.json'); terminal = v.read(out / 'neural_execution_receipt.json')
    v.require(r['complete'] and terminal['complete'] and r['plan_sha256'] == v.sha(check.OUT / 'frozen_plan.json'), 'Run/plan binding differs')
    expected = {key: value for key, value in p['budget'].items() if key.endswith('_forwards')}
    v.require(r['counts'] == terminal['counts'] == expected, 'Forward count contract differs')
    v.require(terminal['before_state_hashes'] == terminal['after_state_hashes'] and r['state_hashes_unchanged'], 'Frozen state changed')
    for flag in ['native_used', 'validation_used', 'native_reserved_used', 'production_promoted']:
        v.require(r[flag] is False, 'Unsupported use/promotion claimed')
    v.require(r['local_backward_calls'] == r['local_optimizer_updates'] == 0 and 0 < r['seconds'] <= check.CAP, 'Training/timing contract differs')
    v.require(r['zero_conditioner_parity'] == terminal['zero_conditioner_parity'] and len(r['zero_conditioner_parity']) == 2, 'Parity receipt differs')
    for row in r['zero_conditioner_parity']:
        v.require(row['max_abs_float_difference'] <= 2e-6 and row['floor_png_identical'], 'Parity failed')
    for name, pin in r['artifacts_sha256'].items():
        v.require(v.sha(v.safe(out, name)) == pin, 'Changed result artifact: ' + name)
    # Extract the earlier independently pinned numeric routine without NN imports.
    source = ROOT / 'cctv_dgp_pilot.py'; tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exported_pixel_metrics']
    v.require(len(nodes) == 1, 'Numeric routine missing')
    ns = {'np': np, 'cv2': cv2, 'structural_similarity': structural_similarity}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    v.require([r['id'] for r in r['rows']] == [ref['id'] for ref in p['references']], 'Training cohort differs')
    # Load only numeric embeddings from weights; no network or neural forward.
    import torch
    teacher_state = torch.load(ROOT / check.TEACHER, map_location='cpu', weights_only=True)['params_ema']
    embedding = teacher_state['quantize.embedding.weight'].numpy()
    pngs = cosines = raws = codes_checked = cells = 0
    summaries = {}
    for ref, row in zip(p['references'], r['rows']):
        v.require(row['role'] == ref['role'] == 'train' and row['source'] == ref['source'] and row['target'] == ref['files']['target'], 'Source/role differs')
        target = v.rgb(ROOT / ref['files']['target']); support = np.asarray(Image.open(ROOT / ref['files']['observed'])) > 0
        latent = np.load(out / row['latents'], allow_pickle=False)
        codes = latent['codes']; quantized = latent['quantized']; gt = latent['target_embedding']
        v.require(codes.dtype == np.int64 and codes.shape == (1, 256) and codes.min() >= 0 and codes.max() < 1024, 'Invalid teacher code labels')
        expected_q = embedding[codes].reshape(1, 16, 16, 256).transpose(0, 3, 1, 2)
        v.require(quantized.dtype == np.float32 and quantized.shape == (1, 256, 16, 16) and np.isfinite(quantized).all(), 'Invalid quantized teacher features')
        np.testing.assert_allclose(quantized, expected_q, rtol=1e-6, atol=1e-6)
        v.require(gt.dtype == np.float32 and gt.shape == (512,) and np.isfinite(gt).all() and np.isclose(np.linalg.norm(gt), 1, atol=1e-5), 'Invalid clean embedding')
        codes_checked += 256
        v.require(set(row['arms']) == set(p['capacity_arms']), 'Capacity arm missing')
        for arm, metrics in row['arms'].items():
            raw = np.load(out / metrics['raw'], allow_pickle=False)
            v.require(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1, 'Invalid raw image')
            image = v.rgb(out / metrics['prediction']); expected_png = np.clip(raw * 255, 0, 255).astype(np.uint8); expected_png[~support] = target[~support]
            np.testing.assert_array_equal(image, expected_png); raws += 1; pngs += 1
            for key, value in ns['exported_pixel_metrics'](image, target, support).items(): close(metrics[key], value)
            vec = np.load(out / metrics['embedding'], allow_pickle=False)
            v.require(vec.dtype == np.float32 and vec.shape == (512,) and np.isfinite(vec).all() and np.isclose(np.linalg.norm(vec), 1, atol=1e-5), 'Invalid prediction embedding')
            close(metrics['ArcFace_observed_fixed'], float(np.clip(vec @ gt, -1, 1))); cosines += 1
            summaries.setdefault(row['source'] + '/' + arm, []).append(metrics)
        images = [v.rgb(out / row['arms'][a]['prediction']) for a in ['teacher_vq_reconstruction', 'clean_code_oracle_w0']]
        close(row['teacher_prior_png_max_abs'], int(np.abs(images[0].astype(np.int16) - images[1].astype(np.int16)).max()))
    v.require(r['grids'] == ['asian_capacity_5_rows.png', 'ffhq_capacity_5_rows.png'], 'Capacity grids differ')
    for source, name in zip(['dataset/asian_faces', 'dataset/thumbnails128x128'], r['grids']):
        selected = [row for row in r['rows'] if row['source'] == source]
        with Image.open(out / name) as im: sheet = np.asarray(im)
        v.require(sheet.shape == (1464, 1040, 3), 'Original grid geometry differs')
        for i, row in enumerate(selected):
            images = [v.rgb(ROOT / row['target'])] + [v.rgb(out / row['arms'][a]['prediction']) for a in p['capacity_arms']]
            for j, image in enumerate(images):
                y = 52 + i * 288
                np.testing.assert_array_equal(sheet[y:y+256, 2+j*260:258+j*260], image); cells += 1
    v.require((pngs, raws, cosines, codes_checked, cells) == (30, 30, 30, 2560, 40), 'Audit counts differ')
    numeric = {}
    for group, rows in summaries.items():
        mse = float(np.mean([row['MSE'] for row in rows]))
        numeric[group] = {'cases': len(rows), 'MSE': mse, 'PSNR': float(-10 * np.log10(mse)) if mse else None,
                          'SSIM': float(np.mean([row['SSIM'] for row in rows])),
                          'ArcFace_observed_fixed': float(np.mean([row['ArcFace_observed_fixed'] for row in rows]))}
    receipt = {'complete': True, 'plan_sha256': r['plan_sha256'], 'results_sha256': v.sha(out / 'results.json'),
               'terminal_receipt_sha256': v.sha(out / 'neural_execution_receipt.json'), 'auditor_sha256': v.sha(Path(__file__)),
               'pngs_checked': pngs, 'raw_floats_checked': raws, 'cosines_rebuilt': cosines,
               'clean_code_labels_checked': codes_checked, 'grid_cells_checked': cells,
               'source_summaries': numeric, 'seconds': time.monotonic() - start,
               'neural_forwards_in_audit': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
               'native_used': False, 'validation_used': False, 'production_promoted': False,
               'limitation': 'Checks bytes, codebook lookup, recorded state/counts and independent arithmetic. Does not replay clean encoder, decoder, recognizer or establish restoration usefulness.'}
    v.write(out / 'independent_audit.json', receipt); print(receipt)


if __name__ == '__main__': audit()
