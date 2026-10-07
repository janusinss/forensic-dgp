"""Independent array, delivered metric, frozen recognizer and fixed-gate readback."""
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np
from diagnose_cctv_dgp_active_original_decoder_v28_mean_control import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
BUNDLE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
RETURNED = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_return'
OUT = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1'


def main():
    start = time.monotonic();r = read(OUT / 'results.json');plan = read(OUT / 'plan.json')
    assert r['complete'] and r['plan_sha256'] == sha(OUT / 'plan.json')
    assert r['runner_sha256'] == plan['runner_sha256'] == sha(ROOT / 'scripts/diagnose_cctv_dgp_active_original_decoder_v28_mean_control.py')
    for name, digest in plan['source_bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in r['output_sha256'].items():
        assert sha(OUT / name) == digest, name
    tree = ast.parse((ROOT / 'scripts/diagnose_cctv_dgp_active_original_decoder_v28_mean_control.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'control')
    assert [a.arg for a in function.args.args] == ['baseline', 'candidate', 'support', 'camera']
    assert not any(isinstance(n, ast.Name) and n.id in ['target', 'profile', 'case', 'truth', 'landmarks'] for n in ast.walk(function))
    assert not plan['target_or_profile_used_to_generate'] and plan['searches_or_tuning'] == 0
    sys.path.insert(0, str(PARENT));sys.path.insert(0, str(BUNDLE))
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, exported_pixel_metrics
    from cctv_dgp_app_input_v28 import canonical_tensor
    from audit_cctv_dgp_active_original_decoder_v28_return import groups, capacity
    from scipy.ndimage import convolve1d
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    p = read(BUNDLE / 'protocol.json');base = read(PARENT / 'protocol.json')
    net = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu')
    before = state_hash(net);assert before == p['frozen_recognizer_state']
    references = {item['id']: item for item in base['references']}
    z = np.arange(-6, 7, dtype=np.float64);k = np.exp(-.5*(z/2)**2);k /= k.sum()
    high = lambda a: a - convolve1d(convolve1d(a, k, axis=0, mode='reflect'), k, axis=1, mode='reflect')
    luma = np.array([.299, .587, .114]);rows = [];calls = 0
    def rgb(path):
        with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()
    with torch.no_grad():
        for c, reported in zip(p['case_rows'], r['rows']):
            assert time.monotonic() - start < 300
            cid = c['id'];assert reported['id'] == cid
            camera = rgb(PARENT / c['input']);target = rgb(PARENT / c['target'])
            with Image.open(PARENT / c['observed']) as im:mask = np.asarray(im).copy() > 0
            baseline = np.load(RETURNED / 'outputs/update0' / (cid + '.npy'), allow_pickle=False)
            candidate = np.load(RETURNED / 'outputs/update800' / (cid + '.npy'), allow_pickle=False)
            dc = (candidate.astype(np.float64) - baseline.astype(np.float64))[mask].mean(0)
            centered = np.clip(candidate.astype(np.float64) - dc, 0, 1).astype(np.float32)
            expected = np.where(mask[..., None], centered, camera.astype(np.float32)/np.float32(255)).astype(np.float32)
            raw = np.load(OUT / (cid + '.npy'), allow_pickle=False);assert np.array_equal(raw, expected)
            png = rgb(OUT / (cid + '.png'))
            assert np.array_equal(png, np.where(mask[..., None], np.floor(raw*np.float32(255)), camera).astype(np.uint8))
            assert np.array_equal(dc, np.asarray(reported['subtracted_RGB_mean']))
            shift = (raw - baseline)[mask].astype(np.float64).mean(0)
            assert np.array_equal(shift, np.asarray(reported['remaining_postclip_RGB_mean']))
            vector = np.load(OUT / (cid + '_embedding.npy'), allow_pickle=False)
            grid = torch.from_numpy(grid112(references[c['source_person_or_reference']]['matrix112']))[None]
            actual = net.embedding(canonical_tensor(png, 'cpu'), torch.from_numpy(mask)[None, None].float(), grid)[0].numpy().copy();calls += 1
            assert np.array_equal(vector, actual)
            truth = np.load(RETURNED / 'outputs/initial_baseline' / (cid + '_target_embedding.npy'), allow_pickle=False)
            metrics = exported_pixel_metrics(png, target, mask);metrics['ArcFace_observed_fixed'] = float(vector @ truth)
            padded = np.pad(mask.astype(np.int64), 6);summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            interior = summed[13:, 13:] - summed[:-13, 13:] - summed[13:, :-13] + summed[:-13, :-13] == 169
            feature = np.zeros((256, 256), bool)
            for point in c['landmarks5_canvas_xy']:
                x, y = np.floor(point).astype(int);feature[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
            feature &= interior
            metrics['landmark_high_frequency_MSE'] = float(np.square(high((png.astype(np.float64)/255*luma).sum(2))-high((target.astype(np.float64)/255*luma).sum(2)))[feature].mean())
            mean_only = np.clip(baseline.astype(np.float64) + shift, 0, 1).astype(np.float32)
            mean_png = np.where(mask[..., None], np.floor(mean_only*np.float32(255)), camera).astype(np.uint8)
            metrics['constant_mean_shift_only_MSE'] = exported_pixel_metrics(mean_png, target, mask)['MSE']
            assert metrics == reported['metrics']
            rows.append({'id': cid, 'source': c['source'], 'profile': c['profile'], 'metrics': metrics})
    grouped = groups(rows);assert grouped == r['groups']
    audit = read(ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json')
    failures, gain, source_gains, fraction, passes = capacity(audit['snapshots_audited'][0]['groups'], grouped)
    assert failures == r['preservation_failures'] and gain == r['degraded_feature_MSE_relative_gain']
    assert source_gains == r['source_feature_gains'] and fraction == r['brightness_gain_fraction']
    assert passes == r['necessary_control_gate_pass'] is False and len(failures) == 5
    assert r['V28_gate_failures_retained'] and not r['original_V28_necessary_capacity_pass']
    assert calls == r['CPU_fixed_recognizer_forwards'] == 50 and state_hash(net) == before
    assert all(not v.requires_grad and v.grad is None for v in net.parameters())
    assert r['local_gradient_calls'] == r['local_backward_calls'] == r['local_optimizer_updates'] == 0
    record = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'results_sha256': sha(OUT / 'results.json'),
              'source_bindings_verified': len(plan['source_bindings_sha256']), 'output_arrays_and_PNGs_verified': 150,
              'CPU_fixed_recognizer_forwards': calls, 'metrics_and_all17_groups_verified': True,
              'gate_failures_retained': len(failures), 'fixed_control_is_insufficient': True,
              'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'app_promotion': False,
              'new_VM_recipe_created': False, 'goal_complete': False, 'seconds': time.monotonic()-start}
    write(OUT / 'independent_readback.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':main()
