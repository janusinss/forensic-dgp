"""Independent combined arithmetic, logical coverage and original-pixel sheet audit."""
from pathlib import Path
import hashlib
import time
import numpy as np
from PIL import Image
from cctv_dgp_actual_step_review_v1_contract import NAME, PROPOSALS, ROLES, read, write, sha, output_prefix

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
ORIGINAL_RETURN = ROOT / 'outputs' / (NAME + '_return')
TAIL_RETURN = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_vm_return'
OUT = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_analysis'


def pixels(path):
    with Image.open(path) as image:
        assert image.mode == 'RGB' and image.size == (256, 256)
        return np.asarray(image).copy()


def main():
    started = time.monotonic()
    target = OUT / 'independent_arithmetic_and_sheets_audit.json'
    assert not target.exists()
    plan = read(OUT / 'prospective_review.json')
    analysis = read(OUT / 'analysis.json')
    p = read(BUNDLE / 'protocol.json')
    original_plan = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json')
    partial_audit = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json')
    tail_audit = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json')
    partial = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis/analysis.json')
    for name, digest in analysis['source_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert plan['visual_rows'] == original_plan['visual_rows'][-25:] and plan['columns'] == original_plan['columns']
    assert plan['no_new_case_or_probe_selection'] and plan['sheets'] == 5 and plan['exact256_pixel_cells'] == 150
    assert partial_audit['complete'] and tail_audit['complete']
    assert partial_audit['all3045_raw_PNG_and_mean_only_outputs_checked'] and tail_audit['all315_outputs_checked']
    old_conditions = {(row['update'], row['proposal']) for row in partial_audit['conditions']}
    final_conditions = {(row['update'], row['proposal']) for row in tail_audit['conditions']}
    expected = {(probe['update'], proposal) for probe in p['probes'] for proposal in PROPOSALS}
    assert len(old_conditions) == 29 and len(final_conditions) == 3 and len(expected) == 30
    assert old_conditions | final_conditions == expected
    assert old_conditions & final_conditions == {(45, 'zero'), (45, 'recorded')}
    assert len(expected) * 105 == 3150 == analysis['unique_completed_diagnostic_slots']
    assert 3045 + 315 - 210 == 3150
    assert analysis['original_run_remains_storage_stopped'] and analysis['tail_completed_separately']
    assert analysis['original_full_checker_not_relabelled_as_pass'] and not analysis['full_TRAIN_capacity_pass']
    receipts, rows, terms, logical = {}, [], [], []
    for probe in p['probes']:
        for proposal in PROPOSALS:
            source = TAIL_RETURN if (probe['update'], proposal) == (45, 'cone') else ORIGINAL_RETURN
            path = source / output_prefix(probe['update'], proposal) / 'metrics.json'
            receipts[(probe['update'], proposal)] = read(path)
            logical.append({'update': probe['update'], 'proposal': proposal, 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        before = receipts[(probe['update'], 'zero')]
        for proposal in PROPOSALS[1:]:
            receipt = receipts[(probe['update'], proposal)]
            delta = np.array(receipt['objective_term_means']['current_batch']) - np.array(before['objective_term_means']['current_batch'])
            terms.append({'update': probe['update'], 'proposal': proposal,
                          'current_batch_raw_objective_term_changes': delta.tolist(),
                          'sum_of_seven_term_changes': float(delta.sum()), 'landmark_term_nonincrease': bool(delta[0] <= 0)})
            for role in ROLES:
                for stage in ('raw', 'PNG'):
                    rows.append({'update': probe['update'], 'proposal': proposal, 'role': role, 'stage': stage,
                                 'cases': receipt['groups'][role][stage]['all']['cases'],
                                 **receipt['comparison_to_same_before_state'][role][stage]})
    assert analysis['logical_conditions'] == logical
    assert len(rows) == 120 and analysis['all120_role_stage_state_proposal_rows'] == rows
    assert len(terms) == 20 and analysis['all20_current_batch_raw_term_changes'] == terms
    assert [row for row in rows if not (row['update'] == 45 and row['proposal'] == 'cone')] == partial['all114_role_stage_state_proposal_rows']
    for summary in analysis['summaries']:
        subset = [row for row in rows if (row['proposal'], row['role'], row['stage']) == (summary['proposal'], summary['role'], summary['stage'])]
        gain = np.array([row['relative_feature_gain'] for row in subset])
        assert len(subset) == summary['states'] == 10
        assert summary['finite_preservation_pass_updates'] == [row['update'] for row in subset if row['finite_preservation_pass']]
        assert summary['finite_preservation_failure_updates'] == [row['update'] for row in subset if not row['finite_preservation_pass']]
        assert summary['positive_feature_gain_updates'] == [row['update'] for row in subset if row['relative_feature_gain'] > 0]
        assert summary['feature_gain_mean'] == float(gain.mean()) and summary['feature_gain_minimum'] == float(gain.min())
        assert summary['feature_gain_maximum'] == float(gain.max())
    assert len(analysis['summaries']) == 12
    cases = {case['id']: case for case in p['cases']}
    checked = 0
    for sheet_index, sheet in enumerate(analysis['sheets']):
        assert sha(ROOT / sheet['path']) == sheet['sha256'] and not sheet['actually_viewed']
        with Image.open(ROOT / sheet['path']) as image:
            assert image.mode == 'RGB' and image.size == (1816, 1504)
            sheet_pixels = np.asarray(image).copy()
        selected = plan['visual_rows'][5 * sheet_index:5 * (sheet_index + 1)]
        for slot, row in enumerate(selected):
            case = cases[row['id']]
            camera, target_pixels = pixels(BUNDLE / case['input']), pixels(BUNDLE / case['target'])
            with Image.open(BUNDLE / case['observed']) as image:
                mask = np.asarray(image.convert('L')) > 0
            with np.load(TAIL_RETURN / output_prefix(45, 'zero') / row['role'] / (row['id'] + '.npz'), allow_pickle=False) as data:
                original_pixels = np.floor(data['original_rgb'] * np.float32(255)).astype(np.uint8)
            original_pixels[~mask] = camera[~mask]
            expected_pixels = [camera, target_pixels, original_pixels]
            expected_pixels += [pixels(TAIL_RETURN / output_prefix(45, proposal) / row['role'] / (row['id'] + '.png')) for proposal in PROPOSALS]
            for column, array in enumerate(expected_pixels):
                cell = sheet['cells'][6 * slot + column]
                x, y = 280 + 256 * column, 64 + 288 * slot
                assert cell['row'] == row and cell['column'] == plan['columns'][column]
                assert cell['box'] == [x, y, x + 256, y + 256]
                assert cell['pixel_sha256'] == hashlib.sha256(array.tobytes()).hexdigest()
                assert np.array_equal(sheet_pixels[y:y + 256, x:x + 256], array)
                checked += 1
    assert checked == 150 and len(analysis['sheets']) == 5
    prior = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1/milestone.json')
    for name, digest in prior['app_bindings_sha256'].items():
        assert sha(ROOT / name) == digest
    report = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
              'prospective_plan_sha256': sha(OUT / 'prospective_review.json'),
              'logical_condition_union_verified': 30, 'duplicate_control_conditions_not_double_counted': 2,
              'unique_completed_diagnostic_slots': 3150, 'numeric_summary_rows_checked': 120,
              'current_batch_term_changes_checked': 20, 'summaries_checked': 12,
              'all5_new_sheets_checked': True, 'all150_new_cells_exact': True, 'visual_rows': 25,
              'actually_viewed_by_this_pixel_checker': False, 'app_bindings_verified': len(prior['app_bindings_sha256']),
              'original_failed_run_status_preserved': True, 'model_forwards': 0, 'optimizer_updates': 0,
              'gradient_queries': 0, 'VM_calls': 0, 'app_promotion': False, 'goal_complete': False,
              'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert report['seconds'] < 300
    write(target, report)
    print({'complete': True, 'unique_slots': 3150, 'summary_rows': 120,
           'sheets': 5, 'exact_cells': 150, 'app_bindings_verified': 14}, flush=True)


if __name__ == '__main__':
    main()
