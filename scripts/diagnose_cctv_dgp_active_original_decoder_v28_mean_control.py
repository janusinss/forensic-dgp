"""One fixed inference-only mean-colour control on audited V28 final800 outputs."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
BUNDLE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
RETURNED = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_return'
OUT = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def control(baseline, candidate, support, camera):
    """No target or case label. One mean subtraction, followed by required clipping."""
    assert baseline.dtype == candidate.dtype == np.float32
    assert baseline.shape == candidate.shape == camera.shape == (256, 256, 3)
    assert support.shape == (256, 256) and support.dtype == bool and support.any()
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    shift = delta[support].mean(0)
    centered = np.clip(candidate.astype(np.float64) - shift, 0, 1).astype(np.float32)
    raw = np.where(support[..., None], centered, camera.astype(np.float32) / np.float32(255)).astype(np.float32)
    png = np.where(support[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8)
    return raw, png, shift


def main():
    start = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert not OUT.exists(), 'Preserve prior diagnostic evidence'
    p = read(BUNDLE / 'protocol.json');base = read(PARENT / 'protocol.json')
    assert sha(BUNDLE / 'protocol.json') == audit['protocol_sha256']
    bindings = {audit_path.relative_to(ROOT).as_posix(): sha(audit_path),
                (BUNDLE / 'protocol.json').relative_to(ROOT).as_posix(): sha(BUNDLE / 'protocol.json')}
    for name, digest in base['assets_sha256'].items():
        path = PARENT / name
        assert sha(path) == digest, name
        bindings[path.relative_to(ROOT).as_posix()] = digest
    imported = read(ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_return_import.json')
    for name, digest in imported['files_sha256'].items():
        assert sha(RETURNED / name) == digest, name
    for name in ['scripts/audit_cctv_dgp_active_original_decoder_v28_return.py',
                 'outputs/cctv_dgp_active_original_decoder_v28_return_import.json']:
        bindings[name] = sha(ROOT / name)
    assert bindings['scripts/audit_cctv_dgp_active_original_decoder_v28_return.py'] == audit['checker_sha256']
    for case in p['case_rows']:
        for suffix in ['.npy', '.png', '_embedding.npy']:
            path = RETURNED / 'outputs/update800' / (case['id'] + suffix)
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        for name in ['outputs/update0/' + case['id'] + '.npy',
                     'outputs/initial_baseline/' + case['id'] + '_target_embedding.npy']:
            path = RETURNED / name
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    OUT.mkdir()
    plan = {'format': 'V28-fixed-mean-colour-control-v1', 'date': '2026-10-06',
            'question': 'Does one target-independent RGB-mean subtraction alone repair V28 final800 failed gates?',
            'cases': [row['id'] for row in p['case_rows']], 'checkpoint': 'Final800 only',
            'transform': 'float64(candidate-baseline) mean per RGB over observed support; subtract once from candidate, clip0..1, float32; retain camera outside support; floor255 PNG',
            'searches_or_tuning': 0, 'target_or_profile_used_to_generate': False,
            'metrics': 'Unchanged17-group deliveredPNG MSE/SSIM/ArcFace and landmark-HF/source/brightness requirements',
            'local_cap_seconds': 300, 'source_bindings_sha256': bindings,
            'runner_sha256': sha(Path(__file__)), 'CPU_fixed_recognizer_calls_bound': 50,
            'training_or_derivative_calls': 0, 'native_or_reserved_used': False,
            'interpretation': 'A development processing control, not a model acceptance, new VM recipe or app promotion. All original raw outputs, checkpoints and failed gates remain.',
            'goal_complete': False}
    write(OUT / 'plan.json', plan)
    sys.path.insert(0, str(PARENT));sys.path.insert(0, str(BUNDLE))
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, exported_pixel_metrics
    from cctv_dgp_app_input_v28 import canonical_tensor
    from audit_cctv_dgp_active_original_decoder_v28_return import groups, capacity
    from PIL import Image
    from scipy.ndimage import convolve1d
    import torch
    torch.set_num_threads(4)
    recognizer = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu')
    before = state_hash(recognizer);assert before == p['frozen_recognizer_state']
    references = {row['id']: row for row in base['references']}
    positions = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-.5 * (positions / 2)**2);kernel /= kernel.sum()
    high = lambda a: a - convolve1d(convolve1d(a, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    luma = np.array([.299, .587, .114])
    rows = [];calls = 0
    def rgb(path):
        with Image.open(path) as im:
            return np.asarray(im.convert('RGB')).copy()
    with torch.no_grad():
        for case in p['case_rows']:
            assert time.monotonic() - start < 300
            cid = case['id'];camera = rgb(PARENT / case['input'])
            with Image.open(PARENT / case['observed']) as im:
                support = np.asarray(im).copy() > 0
            baseline = np.load(RETURNED / 'outputs/update0' / (cid + '.npy'), allow_pickle=False)
            candidate = np.load(RETURNED / 'outputs/update800' / (cid + '.npy'), allow_pickle=False)
            raw, png, subtracted = control(baseline, candidate, support, camera)
            # Targets are loaded only after the complete control output exists.
            target = rgb(PARENT / case['target'])
            truth = np.load(RETURNED / 'outputs/initial_baseline' / (cid + '_target_embedding.npy'), allow_pickle=False)
            grid = torch.from_numpy(grid112(references[case['source_person_or_reference']]['matrix112']))[None]
            vector = recognizer.embedding(canonical_tensor(png, 'cpu'), torch.from_numpy(support)[None, None].float(), grid)[0].numpy().copy();calls += 1
            metrics = exported_pixel_metrics(png, target, support)
            metrics['ArcFace_observed_fixed'] = float(vector @ truth)
            padded = np.pad(support.astype(np.int64), 6);summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            interior = summed[13:, 13:] - summed[:-13, 13:] - summed[13:, :-13] + summed[:-13, :-13] == 169
            feature = np.zeros((256, 256), bool)
            for point in case['landmarks5_canvas_xy']:
                x, y = np.floor(point).astype(int)
                feature[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
            feature &= interior
            metrics['landmark_high_frequency_MSE'] = float(np.square(high((png.astype(np.float64)/255*luma).sum(2)) - high((target.astype(np.float64)/255*luma).sum(2)))[feature].mean())
            shift = (raw - baseline)[support].astype(np.float64).mean(0)
            mean_only = np.clip(baseline.astype(np.float64) + shift, 0, 1).astype(np.float32)
            mean_png = np.where(support[..., None], np.floor(mean_only * np.float32(255)), camera).astype(np.uint8)
            metrics['constant_mean_shift_only_MSE'] = exported_pixel_metrics(mean_png, target, support)['MSE']
            np.save(OUT / (cid + '.npy'), raw, allow_pickle=False);Image.fromarray(png).save(OUT / (cid + '.png'))
            np.save(OUT / (cid + '_embedding.npy'), vector, allow_pickle=False)
            rows.append({'id': cid, 'source': case['source'], 'profile': case['profile'], 'metrics': metrics,
                         'subtracted_RGB_mean': subtracted.tolist(), 'remaining_postclip_RGB_mean': shift.tolist()})
    assert calls == 50 and state_hash(recognizer) == before
    assert all(not value.requires_grad and value.grad is None for value in recognizer.parameters())
    grouped = groups(rows)
    failures, gain, source_gains, fraction, passes = capacity(audit['snapshots_audited'][0]['groups'], grouped)
    record = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'runner_sha256': sha(Path(__file__)),
              'cases': 50, 'rows': rows, 'groups': grouped, 'preservation_failures': failures,
              'degraded_feature_MSE_relative_gain': gain, 'source_feature_gains': source_gains,
              'brightness_gain_fraction': fraction, 'necessary_control_gate_pass': passes,
              'original_V28_necessary_capacity_pass': False, 'V28_gate_failures_retained': True,
              'CPU_fixed_recognizer_forwards': calls, 'recognizer_state_unchanged': before,
              'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
              'model_changes': False, 'app_promotion': False, 'new_VM_recipe_created': False,
              'native_or_reserved_used': False, 'independent_final_review': False, 'goal_complete': False,
              'output_sha256': {path.name: sha(path) for path in OUT.iterdir() if path.suffix in ['.npy', '.png']},
              'visual_review_pending': True, 'seconds': time.monotonic() - start}
    write(OUT / 'results.json', record)
    print(json.dumps({key: record[key] for key in ['complete', 'preservation_failures', 'degraded_feature_MSE_relative_gain', 'brightness_gain_fraction', 'necessary_control_gate_pass', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
