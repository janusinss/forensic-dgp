"""Independent saved-sheet/exposure readback; no neural calls or model imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
RETURNED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
V30 = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def main():
    prepared = read(OUT / 'preparation.json')
    plan = read(OUT / 'plan.json')
    audit_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['necessary_capacity_pass']
    assert prepared['complete'] and prepared['audit_sha256'] == plan['audit_sha256'] == sha(audit_path)
    assert prepared['plan_sha256'] == sha(OUT / 'plan.json')
    assert prepared['generator_sha256'] == sha(ROOT / 'scripts/prepare_cctv_dgp_profile_batches_v31_failure_review.py')
    for name, digest in prepared['source_bindings_sha256'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest
    for name, digest in prepared['sheet_sha256'].items():
        assert sha(OUT / name) == digest
    p = read(BUNDLE / 'protocol.json')
    assert plan['case_ids'] == p['preview_case_ids'] and len(set(plan['case_ids'])) == 50
    assert plan['columns'] == ['Input', 'Retained DGP', 'V30 stopped50', 'V31 stopped50', 'Paired TRAIN target']
    assert plan['regions'] == ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']
    assert not plan['image_processing_or_resize'] and plan['cell_size'] == [256, 256]
    assert all(not plan[k] for k in ['native_or_reserved_used', 'app_promotion', 'independent_final_review', 'goal_complete'])
    schedule = read(BUNDLE / 'schedule.json')['batches'][:50]
    approved = {r['id']: r for r in p['case_rows']}
    selected_rows = [p['case_rows'][i] for batch in schedule for i in batch]
    exposed = {c['id'] for c in selected_rows}
    assert len(selected_rows) == len(exposed) == 250
    assert len({c['source_person_or_reference'] for c in selected_rows}) == 50
    assert sum(c['profile'] == 'clear' for c in selected_rows) == 50
    for batch in schedule:
        cases = [p['case_rows'][i] for i in batch]
        assert len({c['source_person_or_reference'] for c in cases}) == 1
        assert {c['profile'] for c in cases} == {'clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'}
    base, stopped = [read(RETURNED / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    previous_base, previous_stop = [read(V30 / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    assert base['groups'] == previous_base['groups']
    before, after = [{r['id']: r for r in values['rows']} for values in [base, stopped]]
    for key, bucket in prepared['exposure_buckets'].items():
        eligible = [cid for cid in before if before[cid]['profile'] != 'clear' and
                    (key == 'all_TRAIN' or key == 'exposed_by50_TRAIN' and cid in exposed or
                     key == 'not_yet_exposed_TRAIN' and cid not in exposed or
                     key == 'fixed50_TRAIN' and cid in plan['case_ids'])]
        assert len(eligible) == bucket['degraded_cases']
        initial, current = [float(np.mean([values[cid]['metrics']['landmark_high_frequency_MSE'] for cid in eligible]))
                            for values in [before, after]]
        assert initial == bucket['baseline_feature_MSE'] and current == bucket['stopped50_feature_MSE']
        assert abs(1 - current / initial - bucket['relative_feature_gain']) <= 1e-12
        assert bucket['not_held_out_evaluation']
    early = read(RETURNED / 'outputs/early_structure_stop.json')
    assert early == prepared['early_stop'] and early['minimum'] == .01 and not early['pass']
    assert abs(early['relative_feature_error_gain'] - prepared['exposure_buckets']['all_TRAIN']['relative_feature_gain']) <= 1e-12
    failures = []
    for group, row in prepared['groups'].items():
        assert row['baseline'] == base['groups'][group] and row['V30_stopped50'] == previous_stop['groups'][group]
        assert row['V31_stopped50'] == stopped['groups'][group] and row['cases'] == base['groups'][group]['cases']
        initial, current = row['baseline'], row['V31_stopped50']
        bad = [metric for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']
               if (current[metric] > initial[metric] + 1e-12 if metric == 'MSE' else current[metric] < initial[metric] - 1e-6)]
        assert row['V31_preservation_regressions'] == bad
        failures.extend({'group': group, 'metric': metric} for metric in bad)
    assert failures == prepared['preservation_regressions_at50']
    row_lookup = {row['id']: row for row in prepared['rows']}
    seen, cells = [], 0
    for sheet in prepared['sheets']:
        canvas = pixels(OUT / sheet['file'])
        assert canvas.shape == (1516, 1336, 3) and len(sheet['cells']) == 25
        assert len(sheet['cases']) == 5 and len(set(sheet['cases'])) == 5
        assert {approved[cid]['source_person_or_reference'] for cid in sheet['cases']} == {sheet['reference']}
        for cid in sheet['cases']:
            assert row_lookup[cid]['V31_exposed_by50'] == (cid in exposed)
        for cell in sheet['cells']:
            original = pixels(ROOT / cell['source'])
            assert original.shape == (256, 256, 3)
            assert hashlib.sha256(original.tobytes()).hexdigest() == cell['pixel_sha256']
            x, y = cell['xy']
            assert np.array_equal(canvas[y:y+256, x:x+256], original)
            assert cell['case'] in sheet['cases'] and 0 <= cell['column'] < 5
            assert sum(c['case'] == cell['case'] and c['column'] == cell['column'] for c in sheet['cells']) == 1
            cells += 1
        seen.extend(sheet['cases'])
    assert seen == plan['case_ids'] and cells == prepared['exact_cells_checked'] == 250
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)),
               'preparation_sha256': sha(OUT / 'preparation.json'), 'plan_sha256': sha(OUT / 'plan.json'),
               'independent_return_audit_sha256': sha(audit_path),
               'source_bindings_verified': len(prepared['source_bindings_sha256']),
               'exact_256_cells_verified': cells, 'fixed_cases_verified': len(seen), 'sheets_verified': 10,
               'paired_schedule_and_exposure_buckets_verified': True, 'same_V30_V31_original_baseline_verified': True,
               'preservation_regressions_at50': failures, 'early_one_percent_failure_preserved': True,
               'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'app_promotion': False,
               'independent_final_review': False, 'goal_complete': False}
    with (OUT / 'independent_preparation_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
