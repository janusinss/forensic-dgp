"""Independent byte/metric readback of the fixed R2 TRAIN comparison; no neural calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
PRIOR = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def main():
    prepared, plan, p = map(read, [OUT / 'preparation.json', OUT / 'plan.json', BUNDLE / 'protocol.json'])
    audit_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['necessary_capacity_pass']
    assert prepared['audit_sha256'] == plan['independent_return_audit_sha256'] == sha(audit_path)
    assert prepared['plan_sha256'] == sha(OUT / 'plan.json')
    assert prepared['generator_sha256'] == sha(ROOT / 'scripts/prepare_cctv_dgp_feature_fusion_v32_r2_failure_review_v1.py')
    assert plan['case_ids'] == p['preview_case_ids'] and len(set(plan['case_ids'])) == 50
    assert plan['columns'] == ['Input', 'Retained DGP', 'V31 stopped50', 'V32 r2 stopped50', 'Paired TRAIN target']
    assert plan['regions'] == ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']
    assert not plan['image_resizing_or_display_processing'] and plan['cell_size'] == [256, 256]
    for name, digest in prepared['source_bindings_sha256'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest
    schedule = read(BUNDLE / 'schedule.json')['batches'][:50]
    selected = [p['case_rows'][i] for batch in schedule for i in batch]
    exposed = {c['id'] for c in selected}
    assert len(selected) == len(exposed) == prepared['unique_cases_optimized_by50'] == 250
    assert len({c['source_person_or_reference'] for c in selected}) == prepared['unique_references_touched_by50'] == 50
    for batch in schedule:
        cases = [p['case_rows'][i] for i in batch]
        assert len({c['source_person_or_reference'] for c in cases}) == 1
        assert {c['profile'] for c in cases} == {'clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'}
    baseline, stopped = [read(RETURNED / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    oldbase, oldstop = [read(PRIOR / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    assert baseline['groups'] == oldbase['groups']
    before, after = [{row['id']: row for row in value['rows']} for value in [baseline, stopped]]
    for key, bucket in prepared['exposure_buckets'].items():
        ids = [cid for cid in before if before[cid]['profile'] != 'clear' and
               (key == 'all_TRAIN' or key == 'exposed_by50_TRAIN' and cid in exposed or
                key == 'not_yet_exposed_TRAIN' and cid not in exposed or key == 'fixed50_TRAIN' and cid in plan['case_ids'])]
        b, a = [float(np.mean([data[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids])) for data in [before, after]]
        assert len(ids) == bucket['degraded_cases'] and bucket['TRAIN_only_not_evaluation']
        assert b == bucket['baseline_feature_MSE'] and a == bucket['stopped50_feature_MSE']
        assert abs(1 - a / b - bucket['relative_feature_gain']) <= 1e-12
    early = read(RETURNED / 'outputs/early_structure_stop.json')
    assert early == prepared['early_stop'] and early['minimum'] == .01 and not early['pass']
    failures = []
    for group, row in prepared['groups'].items():
        assert row['baseline'] == baseline['groups'][group] and row['V31_stopped50'] == oldstop['groups'][group]
        assert row['V32_r2_stopped50'] == stopped['groups'][group]
        b, a = row['baseline'], row['V32_r2_stopped50']
        bad = [k for k in ['MSE', 'SSIM', 'ArcFace_observed_fixed']
               if (a[k] > b[k] + 1e-12 if k == 'MSE' else a[k] < b[k] - 1e-6)]
        assert row['V32_r2_preservation_regressions'] == bad
        assert abs(1 - a['landmark_high_frequency_MSE'] / b['landmark_high_frequency_MSE'] - row['V32_r2_feature_gain']) <= 1e-12
        failures.extend({'group': group, 'metric': metric} for metric in bad)
    assert failures == prepared['preservation_regressions_at50']
    approved = {c['id']: c for c in p['initial_proof_case_rows']}
    seen, count = [], 0
    for sheet in prepared['sheets']:
        canvas = rgb(OUT / sheet['file'])
        assert canvas.shape == (1516, 1336, 3) and sha(OUT / sheet['file']) == prepared['sheet_sha256'][sheet['file']]
        assert len(sheet['cells']) == 25 and len(sheet['cases']) == 5
        for cell in sheet['cells']:
            cid, column = cell['case'], cell['column']
            case = approved[cid]
            expected = [PARENT / case['input'], RETURNED / f'outputs/update0/{cid}.png',
                        PRIOR / f'outputs/update50/{cid}.png', RETURNED / f'outputs/update50/{cid}.png', PARENT / case['target']]
            assert cid in sheet['cases'] and 0 <= column < 5
            assert cell['source'] == expected[column].relative_to(ROOT).as_posix()
            original, (x, y) = rgb(expected[column]), cell['xy']
            assert original.shape == (256, 256, 3) and hashlib.sha256(original.tobytes()).hexdigest() == cell['pixel_sha256']
            assert np.array_equal(canvas[y:y + 256, x:x + 256], original)
            assert sum(c['case'] == cid and c['column'] == column for c in sheet['cells']) == 1
            count += 1
        seen.extend(sheet['cases'])
    assert seen == plan['case_ids'] and count == prepared['exact_cells_checked'] == 250
    assert [row['id'] for row in prepared['rows']] == seen
    for row in prepared['rows']:
        assert row['exposed_by50'] == (row['id'] in exposed)
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)),
               'preparation_sha256': sha(OUT / 'preparation.json'), 'plan_sha256': sha(OUT / 'plan.json'),
               'independent_return_audit_sha256': sha(audit_path),
               'source_bindings_verified': len(prepared['source_bindings_sha256']),
               'exact_256_cells_verified': count, 'fixed_cases_verified': len(seen), 'sheets_verified': 10,
               'same_V31_V32_original_baseline_verified': True, 'paired_schedule_and_exposure_buckets_verified': True,
               'preservation_regressions_at50': failures, 'early_one_percent_failure_preserved': True,
               'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'app_promotion': False,
               'independent_final_review': False, 'goal_complete': False}
    with (OUT / 'independent_preparation_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
