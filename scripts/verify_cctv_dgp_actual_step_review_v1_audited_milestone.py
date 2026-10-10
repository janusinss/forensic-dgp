"""Read back milestone bindings, scope, gates, visual coverage and manual packet."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_audited_milestone'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    target = OUT / 'independent_closure_audit.json'
    assert not target.exists(), 'Do not overwrite a prior closure receipt'
    milestone = read('outputs/cctv_dgp_actual_step_review_v1_audited_milestone/milestone.json')
    for name, expected in milestone['new_evidence_sha256'].items():
        assert sha(ROOT / name) == expected, name
    prior_path = 'outputs/cctv_dgp_actual_step_review_v1_prepared_milestone/milestone.json'
    prior = read(prior_path)
    assert sha(ROOT / prior_path) == milestone['previous_milestone_sha256']
    assert len(prior['new_evidence_sha256']) == milestone['previous_binding_count'] == 703
    backup = ROOT / milestone['previous_handoff_path']
    for name, expected in prior['new_evidence_sha256'].items():
        actual = backup if name == 'PROJECT_HANDOFF.md' else ROOT / name
        assert sha(actual) == expected, name
    section = (OUT / 'new_handoff_section.md').read_bytes()
    assert (ROOT / 'PROJECT_HANDOFF.md').read_bytes() == section + backup.read_bytes()
    assert sha(backup) == milestone['previous_handoff_sha256']
    assert len(milestone['app_bindings_sha256']) == 14
    for name, expected in milestone['app_bindings_sha256'].items():
        assert sha(ROOT / name) == expected, name
    base = 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis/'
    analysis = read(base + 'analysis.json')
    plan = read(base + 'prospective_review.json')
    pixels = read(base + 'independent_arithmetic_and_sheets_audit.json')
    visual = read(base + 'visual_review.json')
    assert pixels['analysis_sha256'] == visual['analysis_sha256'] == sha(ROOT / (base + 'analysis.json'))
    assert visual['prospective_selection_sha256'] == sha(ROOT / (base + 'prospective_review.json'))
    assert visual['sheets_viewed'] == 45 and visual['rows_viewed'] == 225
    assert visual['exact256_pixel_cells_viewed'] == 1350
    assert [{k: r[k] for k in ['update', 'role', 'id', 'source', 'profile']} for r in visual['rows']] == plan['visual_rows']
    assert all(r['actually_viewed'] and r['columns_viewed'] == plan['columns'] for r in visual['rows'])
    assert all(not r['model_qualification'] and not r['independent_final_review'] for r in visual['rows'])
    assert len(visual['sheets']) == 45
    for old_sheet, reviewed in zip(analysis['sheets'], visual['sheets']):
        assert old_sheet['path'] == reviewed['path'] and old_sheet['sha256'] == reviewed['sha256']
        assert reviewed['actually_viewed'] and sha(ROOT / reviewed['path']) == reviewed['sha256']
    assert not analysis['visual_review_complete'], 'Original generator receipt remains unchanged'
    assert not pixels['actually_viewed'], 'Arithmetic verification must not claim visual inspection'
    summaries = analysis['summaries']
    for row in summaries:
        if row['role'] != 'current_batch' or row['stage'] == 'PNG':
            assert not row['finite_preservation_pass_updates']
        else:
            assert row['finite_preservation_pass_updates'] == [27]
    terms = analysis['all19_current_batch_raw_term_changes']
    assert sum(not r['landmark_term_nonincrease'] for r in terms if r['proposal'] == 'recorded') == 9
    assert sum(not r['landmark_term_nonincrease'] for r in terms if r['proposal'] == 'cone') == 5
    sci = read('outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json')
    assert sci['complete'] and sci['conditions_count'] == 29 and sci['CPU_replay']['cases'] == 145
    assert sci['all3045_raw_PNG_and_mean_only_outputs_checked'] and sci['categorical_decisions_exact']
    assert sci['archive_sha256'] == milestone['returned_archive_sha256']
    assert not sci['full_3150_review_complete']
    assert len(sci['derived_floored_ratio_recomputation_exception']) == 3
    assert all(r['both_degraded_MSE_gates_failed'] and r['all_categorical_decisions_exact']
               and r['absolute_ratio_error'] <= r['roundoff_bound']
               for r in sci['derived_floored_ratio_recomputation_exception'])
    assert (ROOT / sci['original_failure_path']).exists()
    assert sha(ROOT / sci['original_failure_path']) == sci['original_failure_sha256']
    old_import = read('outputs/cctv_dgp_actual_step_review_v1_independent_audit.json')
    assert old_import['members_verified'] == 9369 and 'import only' in old_import['audit_scope']
    original_failure = read('outputs/cctv_dgp_actual_step_review_v1_vm_return/outputs/failure.json')
    assert original_failure['error'] == 'Return-storage limit; retain partial review'
    assert original_failure['optimizer_updates'] == 0
    tail_path = 'outputs/cctv_dgp_actual_step_tail_v1_vm/protocol.json'
    tail = read(tail_path)
    assert sha(ROOT / tail_path) == milestone['tail_protocol_sha256']
    assert tail['updates'] == [45] and tail['forward_slots'] == 315
    assert len(tail['overlap_files_sha256']) == 828
    assert tail['budgets']['minimum_free_bytes'] == 2 * 1024**3
    assert tail['budgets']['maximum_output_bytes'] == 512 * 1024**2
    assert tail['manual_VM_execution_required'] and tail['prepared_only']
    assert tail['no_new_restoration_recipe'] and not tail['automatic_follow_on']
    assert not list((ROOT / 'outputs').glob('cctv-dgp-actual-step-tail-v1-results*'))
    guide = (ROOT / 'CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md').read_text(encoding='utf-8')
    assert milestone['tail_protocol_sha256'] in guide and milestone['tail_execution_archive_sha256'] in guide
    assert guide.count('gcloud compute scp') == 5, 'Two uploads and three separate PuTTY-compatible downloads'
    assert 'tmux new-session -A -s dgp_actual_step_tail_v1' in guide
    assert not milestone['goal_complete'] and not milestone['app_promotion']
    assert not milestone['new_training_recipe_prepared'] and not milestone['new_trained_checkpoint']
    elapsed = time.monotonic() - start
    assert elapsed <= 300
    result = {
        'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'),
        'checker_sha256': sha(Path(__file__)),
        'new_evidence_bindings_checked': len(milestone['new_evidence_sha256']),
        'all703_previous_bindings_retained': True, 'entire_previous_handoff_preserved': True,
        'all14_app_bindings_unchanged': True, 'all225_visual_records_match_prospective_selection': True,
        'all45_reviewed_sheets_hash_exact': True,
        'visual_review_is_assistant_diagnostic_not_independent_final': True,
        'preservation_failures_not_reclassified': True,
        'original_storage_and_partial_auditor_failures_retained': True,
        'full_3150_review_complete': False, 'manual_tail_packet_prepared_unrun': True,
        'optimizer_updates': 0, 'gradient_queries': 0, 'model_forwards': 0, 'VM_calls': 0,
        'file_deletions': 0, 'app_promotion': False, 'independent_final_review': False,
        'goal_complete': False, 'seconds': elapsed, 'cap_seconds': 300,
    }
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print({'complete': True, 'bindings': result['new_evidence_bindings_checked'],
           'previous_bindings': 703, 'app_bindings': 14, 'goal_complete': False, 'seconds': elapsed}, flush=True)


if __name__ == '__main__':
    main()
