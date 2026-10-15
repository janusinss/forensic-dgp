"""Independent saved-array/NumPy math audit; no model/gradient/optimizer calls."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis'
RECEIPT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_math_audit.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1024**2), b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    assert not RECEIPT.exists()
    torch.set_num_threads(4)
    p = read(B/'protocol.json')
    assert p['reserved_final_pixels'] == 0 and p['scientific_thresholds']['brightness_fraction_maximum'] == .2
    bindings = {}
    manifest = read(B/'export_manifest.json')['files']

    def bind(path):
        value = sha(path)
        if path.is_relative_to(B): assert manifest[path.relative_to(B).as_posix()] == value
        bindings[path.relative_to(ROOT).as_posix()] = value

    for name in ['independent_audit.json', 'matched_AB_comparison.json', 'visual_review.json']:
        path = ANALYSIS/name; assert read(path)['complete']; bind(path)
    sys.path.insert(0, str(B))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_raw_metrics import mean_only, pixel_metrics
    bind(B/'cctv_dgp_head4_lossless_v1.py'); bind(B/'frozen_raw_metrics.py')
    helper = ROOT/'scripts/cctv_dgp_bank_photometric_guard_v1_loss.py'; bind(helper)
    spec = importlib.util.spec_from_file_location('new_guard_loss', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    refs = {row['id']: row for row in p['references']}
    saved = read(B/'outputs/bank_comparison_v1/B50/metrics.json')
    bind(B/'outputs/bank_comparison_v1/B50/metrics.json')
    by_id = {row['id']: row for row in saved['rows']}
    rows = []; max_difference = 0.; zero_checked = 0

    def pixels(path, mode='RGB'):
        bind(path)
        with Image.open(path) as image:
            assert image.size == (256, 256)
            return np.asarray(image.convert(mode)).copy()

    def tensor(value): return torch.from_numpy(np.asarray(value, dtype=np.float32)).permute(0, 3, 1, 2)

    with torch.inference_mode():
        for begin in range(0, len(p['cases']), 5):
            cs = p['cases'][begin:begin+5]
            if cs[0]['id'] not in p['preview_case_ids']: continue
            assert [c['profile'] for c in cs] == ['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
            ids = [c['id'] for c in cs]
            paths = [B/'outputs/bank_comparison_v1'/label/'packs'/f'b{begin//5:03d}.npz' for label in ['baseline','B50']]
            for path in paths: bind(path)
            baseline, _ = unpack(paths[0], ids)
            candidate, _ = unpack(paths[1], ids, baseline=baseline)
            target8 = [pixels(B/refs[c['source_person_or_reference']]['target']) for c in cs]
            camera = [pixels(B/c['input']) for c in cs]
            masks = np.stack([pixels(B/refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cs])
            target = np.stack(target8).astype(np.float32)/np.float32(255)
            pred_t, base_t, target_t = [tensor(v) for v in [candidate, baseline, target]]
            mask_t = torch.from_numpy(masks.astype(np.float32))[:, None]
            value, detail = module.brightness_guard(pred_t, base_t, target_t, mask_t, p['normalizers'][0])
            gold = []
            for i, case in enumerate(cs):
                control, _, _ = mean_only(candidate[i], baseline[i], camera[i], masks[i])
                old_mse = pixel_metrics(baseline[i], target8[i], masks[i])['MSE']
                new_mse = pixel_metrics(candidate[i], target8[i], masks[i])['MSE']
                mean_mse = pixel_metrics(control, target8[i], masks[i])['MSE']
                assert abs(mean_mse-by_id[case['id']]['raw']['constant_mean_shift_only_MSE']) < 1e-13
                for name, expected in [('old_MSE',old_mse),('actual_MSE',new_mse),('mean_only_MSE',mean_mse)]:
                    difference = abs(detail[name][i]-expected); max_difference = max(max_difference,difference)
                    assert difference < 1e-12, (case['id'],name,difference)
                gold.append(max(0., max(0.,old_mse-mean_mse)-.2*max(0.,old_mse-new_mse)))
            expected = 10*np.mean(gold[1:])/p['normalizers'][0]
            assert abs(float(value)-expected) < 1e-10
            zero, zero_detail = module.brightness_guard(base_t, base_t, target_t, mask_t, p['normalizers'][0])
            assert float(zero) == 0. and zero_detail['degraded_excess'] == 0.
            zero_checked += 5
            rows.append(dict(reference=cs[0]['reference_id'], profiles=ids, penalty=float(value), independent_penalty=expected))
            assert time.monotonic()-start < 300, 'Finite forward-only math audit stop'
        # Pure observed brightness improvement must be penalized, and an outside-
        # observation-only modification must not change the penalty.
        base = torch.full((5,3,256,256), .25)
        target = torch.full_like(base,.30); mask = torch.zeros((5,1,256,256)); mask[:,:,32:224,32:224] = 1
        pred = torch.where(mask.bool(),target,base)
        pure, _ = module.brightness_guard(pred,base,target,mask,p['normalizers'][0])
        changed = torch.where(mask.bool(),pred,torch.full_like(pred,.9))
        outside, _ = module.brightness_guard(changed,base,target,mask,p['normalizers'][0])
        assert float(pure) > 0 and float(pure) == float(outside)
    assert len(rows) == 20 and zero_checked == 100
    assert all(sha(ROOT/path) == digest for path,digest in bindings.items())
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        helper_sha256=sha(helper), evidence_bindings=bindings, saved_TRAIN_cases=100,
        exact_baseline_zero_penalty_cases=zero_checked, saved_gate_mean_only_math_rechecked_cases=100,
        maximum_MSE_difference=max_difference, independently_recomputed_reference_penalties=rows,
        pure_brightness_gain_penalized=True, outside_observed_pixels_do_not_affect_guard=True,
        math_cap_seconds=300, seconds=time.monotonic()-start,
        model_forward_calls=0, recognizer_forward_calls=0, gradient_queries=0, optimizer_updates=0, backwards=0,
        CUDA_gradient_training_validation_pending=True, useful_quality_improvement_unverified=True,
        model_qualified=False, app_promotion=False, goal_complete=False)
    with RECEIPT.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,cases=100,zero_penalty_cases=100,gradient_queries=0,optimizer_updates=0,seconds=result['seconds']),flush=True)


if __name__ == '__main__': main()
