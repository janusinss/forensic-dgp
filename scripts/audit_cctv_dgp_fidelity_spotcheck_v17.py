"""Independent saved-output audit; imports no neural model and performs no training.

Recomputes compositions, paired training-preview metrics and exact grid cells.
Neural counts, frozen-state equality and CPU/CUDA parity are execution receipts,
not independent replays of neural inference.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_fidelity_spotcheck_v17'
BUNDLE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
PARENT = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
RETURN = ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16_r2/outputs/broader_codes_v16_r2'
R2_PIN = '4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0'
R2_RESULTS = 'ccb9c94416a08672a37700fdf3c460170a7f7b9f1beecae88205e71820f2a20b'
RUNNER_PIN = '20712c35d7fd209dfb71e6950c3bd9db92a0320e311bb96902eb1956913128b5'
PROFILES = ['clear', 'blur_lr24', 'compound_lr24']
SOURCES = ['dataset/asian_faces', 'dataset/thumbnails128x128']
ARMS = ['starting_codes_w1_none', 'own_epoch8_codes_w1_none']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe(root, name):
    part = PurePosixPath(name)
    require(bool(name) and not part.is_absolute() and '..' not in part.parts
            and '\\' not in name and ':' not in name, 'Unsafe path')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escapes root')
    return path


def rgb(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'RGB256 required')
        return np.asarray(image).copy()


def metrics(image, target, support):
    error = image.astype(np.float32) / 255 - target.astype(np.float32) / 255
    mse = float(np.square(error[support]).astype(np.float64).mean())
    _, pixel_ssim = structural_similarity(target.astype(np.float32) / 255,
        image.astype(np.float32) / 255, channel_axis=-1, data_range=1,
        win_size=7, full=True)
    inner = cv2.erode(support.astype(np.uint8), np.ones((7, 7), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    require(support.any() and inner.any(), 'Missing observation support')
    return {'MSE': mse, 'PSNR': -10 * math.log10(mse) if mse else None,
        'SSIM': float(pixel_ssim[inner].astype(np.float64).mean()),
        'MAE': float(np.abs(error[support]).astype(np.float64).mean()),
        'perfect_match': mse == 0}


def audit():
    started = time.monotonic()
    receipt = OUT / 'independent_saved_output_audit.json'
    require(not receipt.exists(), 'Preserve existing audit; do not overwrite')
    plan, result = read(OUT / 'protocol.json'), read(OUT / 'results.json')
    require(result['complete'] and not (OUT / 'failure.json').exists(), 'Completed inference required')
    require(sha(OUT / 'runner.py') == sha(ROOT / 'scripts/run_cctv_dgp_fidelity_spotcheck_v17.py')
            == plan['runner_sha256'] == RUNNER_PIN, 'Runner differs')
    require(sha(OUT / 'protocol.json') == result['protocol_sha256'], 'Control protocol differs')
    require(sha(BUNDLE / 'broader_codes_protocol_v16_r2.json') == R2_PIN
            == plan['parent_protocol_sha256'], 'R2 protocol differs')
    require(sha(RETURN / 'results.json') == R2_RESULTS == plan['parent_results_sha256'], 'R2 results differ')
    full = read(ROOT / 'outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json')
    require(full['complete'] and full['results_sha256'] == R2_RESULTS, 'Full R2 audit required')
    p, r2 = read(BUNDLE / 'broader_codes_protocol_v16_r2.json'), read(RETURN / 'results.json')
    ids = p['train_preview_reference_ids'][:2] + p['train_preview_reference_ids'][5:7]
    refs = {r['id']: r for r in p['references']}
    cases = [next(c for c in p['training_cases'] if c['reference_id'] == rid and c['profile'] == profile)
        for profile in PROFILES for rid in ids]
    require(plan['reference_ids'] == ids and plan['case_ids'] == [c['id'] for c in cases]
            and plan['profiles'] == PROFILES and plan['role'] == 'train'
            and all(refs[rid]['role'] == 'train' for rid in ids)
            and [sum(refs[rid]['source'] == s for rid in ids) for s in SOURCES] == [2, 2],
            'Training-only control selection differs')
    require(plan['arms'] == ARMS and plan['fidelity'] == 1.0 and plan['statistics'] == 'none'
            and plan['device'] == 'cpu' and plan['cross_device_w0_float_cap'] == 2e-5
            and plan['neural_cap_seconds'] == 600 and plan['external_cap_seconds'] == 720,
            'Rendering policy/finite bounds differ')
    require(plan['timing_sample_cases'] == 4 and plan['timing_steady_case_indices'] == [1, 3]
            and plan['timing_safety_factor'] == 1.25 and plan['timing_reserve_seconds'] == 30,
            'Timing policy differs')
    for key in ['teacher_used', 'validation_used', 'native_used', 'native_reserved_used',
                'checkpoint_selected', 'production_promoted']:
        require(plan[key] is False and result[key] is False, 'Scope differs:' + key)
    for key in ['neural_dgp_forwards', 'conditioner_forwards', 'recognizer_forwards',
                'backward_calls', 'optimizer_updates']:
        require(result[key] == 0, 'Forbidden scope receipt:' + key)
    require(result['counts'] == {'prior_encoder': 12, 'prior_classifier': 0, 'prior_generator': 26}
            and result['frozen_before'] == result['frozen_after'], 'Inference state/count receipt differs')
    timing = read(OUT / 'timing.json')
    require(timing['cases'] == 4 and timing['cap_seconds'] == 600
            and 0 < timing['seconds'] <= timing['projected_seconds'] <= 600
            and 0 < result['seconds'] <= 600, 'Finite timing receipt differs')
    pp = read(PARENT / 'face_code_fit_protocol_v12.json')
    for name in [pp['weights']['prior'], 'dgp_face_code_conditioner_v11.py',
                 'pretrained_face_restoration_portable_v12.py']:
        require(sha(safe(PARENT, name)) == p['parent_assets_sha256'][name], 'Frozen prior/renderer differs')
    require(result['prior_provenance']['weights_sha256'] == p['parent_assets_sha256'][pp['weights']['prior']],
            'Prior provenance differs')
    # Loader provenance describes its default adapter (AdaIN=True). The actual
    # bound runner calls decode without original_features, so effective AdaIN=False.
    for index, item in zip([0, 2], result['parity']):
        require(item == read(OUT / ('parity_' + str(index) + '.json'))
                and item['id'] == cases[index]['id'] and item['passed'] is True
                and math.isfinite(item['maximum_float_difference'])
                and 0 <= item['maximum_float_difference'] <= 2e-5, 'Parity receipt differs')
    require(len(result['parity']) == 2 and len(result['rows']) == 12, 'Control coverage differs')
    for name, digest in result['artifacts_sha256'].items():
        require(sha(safe(OUT, name)) == digest, 'New artifact differs:' + name)
    require(len(result['artifacts_sha256']) == 51, 'Require24 raws,24 PNGs and3 grids')
    def old(name):
        path = safe(RETURN, name)
        require(sha(path) == r2['artifacts_sha256'][name], 'Parent returned artifact differs')
        return path

    snapshots = {s['epoch']: {row['id']: row for row in read(old(s['metrics']))['rows']}
        for s in r2['snapshots'] if s['epoch'] in [0, 8]}

    images, comparisons = {}, []
    for case, row in zip(cases, result['rows']):
        cid, ref = case['id'], refs[case['reference_id']]
        require(row['role'] == 'train' and all(row[k] == case[k] for k in
            ['id', 'reference_id', 'source', 'profile']) and set(row['arms']) == set(ARMS), 'Row differs')
        for name in [case['input'], ref['target'], ref['observed']]:
            require(sha(safe(MIXED, name)) == p['data_assets_sha256'][name], 'Selected data differs')
        camera, target = rgb(MIXED / case['input']), rgb(MIXED / ref['target'])
        with Image.open(MIXED / ref['observed']) as image:
            require(image.size == (256, 256) and image.mode == 'L', 'Observation mask differs')
            support = np.asarray(image) > 0
        columns = [camera, rgb(old(snapshots[0][cid]['dgp_preview'])),
                   rgb(old(snapshots[8][cid]['prediction']))]
        measured = {'camera': metrics(camera, target, support),
            'retained_dgp_v2': metrics(columns[1], target, support),
            'own_epoch8_w0_none': metrics(columns[2], target, support),
            'starting_w0_none': metrics(rgb(old(snapshots[0][cid]['prediction'])), target, support)}
        for epoch, arm in [(0, ARMS[0]), (8, ARMS[1])]:
            probe = np.load(old(snapshots[epoch][cid]['logits']), allow_pickle=False)
            require(probe.dtype == np.float32 and probe.shape == (256, 1024)
                    and np.isfinite(probe).all(), 'Parent logits invalid')
            item = row['arms'][arm]
            require(item['raw'] == arm + '/' + cid + '.npy'
                    and item['prediction'] == arm + '/' + cid + '.png', 'Output path differs')
            raw = np.load(safe(OUT, item['raw']), allow_pickle=False)
            require(raw.dtype == np.float32 and raw.shape == (256, 256, 3)
                    and np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1, 'Raw invalid')
            actual = rgb(safe(OUT, item['prediction']))
            composed = np.floor(raw * 255).astype(np.uint8)
            composed[~support] = camera[~support]
            require(np.array_equal(actual, composed), 'Raw/PNG observation composition differs')
            fresh = metrics(actual, target, support)
            for key, value in fresh.items():
                require(item[key] == value if value is None or isinstance(value, bool)
                    else math.isclose(item[key], value, rel_tol=1e-9, abs_tol=1e-9), 'Metric differs:' + key)
            measured[arm] = fresh
            columns.append(actual)
        images[cid] = columns + [target]
        comparisons.append({'id': cid, 'source': case['source'], 'profile': case['profile'],
                            'role': 'train', 'arms': measured})
    expected_grids = ['grids/' + profile + '_4_rows.png' for profile in PROFILES]
    require(result['grids'] == expected_grids, 'Grid inventory differs')
    for profile, name in zip(PROFILES, expected_grids):
        with Image.open(safe(OUT, name)) as grid:
            require(grid.mode == 'RGB' and grid.size == (1560, 1176), 'Original-cell grid size differs')
            for i, rid in enumerate(ids):
                case = next(c for c in cases if c['reference_id'] == rid and c['profile'] == profile)
                for j, expected in enumerate(images[case['id']]):
                    x, y = j * 260 + 2, 24 + i * 288 + 28
                    require(np.array_equal(np.asarray(grid.crop((x, y, x + 256, y + 256))), expected),
                            'Grid cell changed/resized')
    groups = {'clear': [r for r in comparisons if r['profile'] == 'clear'],
              'degraded': [r for r in comparisons if r['profile'] != 'clear']}
    for source in SOURCES:
        for profile in PROFILES:
            groups[source + '/' + profile] = [r for r in comparisons if r['source'] == source and r['profile'] == profile]
    summary = {}
    for key, items in groups.items():
        summary[key] = {}
        for arm in comparisons[0]['arms']:
            mse = float(np.mean([r['arms'][arm]['MSE'] for r in items]))
            summary[key][arm] = {'cases': len(items), 'MSE': mse,
                'PSNR': -10 * math.log10(mse) if mse else None,
                'SSIM': float(np.mean([r['arms'][arm]['SSIM'] for r in items])),
                'MAE': float(np.mean([r['arms'][arm]['MAE'] for r in items]))}
    result_hash = sha(OUT / 'results.json')
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'complete': True, 'date': '2026-10-05', 'protocol_sha256': result['protocol_sha256'],
            'results_sha256': result_hash, 'runner_sha256': RUNNER_PIN, 'auditor_sha256': sha(Path(__file__)),
            'cases': 12, 'training_references': 4, 'raws': 24, 'PNGs': 24, 'grids': 3, 'original_grid_cells': 72,
            'recomputed_output_metrics': 24, 'comparisons': comparisons, 'summary': summary,
            'effective_renderer_override': {'statistics': 'none', 'adain': False, 'fidelity': 1.0},
            'serialized_inference_receipts_verified': True, 'neural_inference_replayed': False,
            'parity_raws_retained': False, 'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
            'validation_used': False, 'native_used': False, 'production_promoted': False,
            'scope': 'Paired training-only photographic processing control; no generalization, recognition or native/local-performance claim.',
            'seconds': time.monotonic() - started}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'complete': True, 'cases': 12, 'PNGs': 24, 'grid_cells': 72,
           'results_sha256': result_hash, 'seconds': time.monotonic() - started})


if __name__ == '__main__':
    audit()
