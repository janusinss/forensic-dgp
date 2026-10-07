"""Independent readback of the R2 return, retained history and chosen design review."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1'
PREVIOUS = ROOT / 'outputs/completion_mat_mirror_comparison_v1_milestone'
REVIEW = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), name
        assert sha(path) == digest, name
    return len(mapping)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json')
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    current = verify(m['new_evidence_sha256'])
    prior = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256']
    old = verify(prior['new_evidence_sha256'], m['previous_document_locations'])
    closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert closure['complete'] and closure['milestone_sha256'] == m['previous_milestone_sha256']
    assert closure['previous_actual_Windows_backup_receipt_preserved'] == 4431
    final_count = verify(read(PREVIOUS / 'final_readback.json')['evidence_sha256'], m['previous_document_locations'])
    doc = m['document']
    before, after = (ROOT / doc['before_path']).read_bytes(), (ROOT / doc['name']).read_bytes()
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest() == doc['before_sha256']
    assert sha(ROOT / doc['name']) == doc['after_sha256']

    decision = read(ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json')
    assert decision['complete'] and decision['discussion_satisfied']
    assert decision['answer'] == m['user_design_answer'] == 'apply the best approach and do research also if needed'
    assert decision['selected_direction'] == m['selected_direction'] == 'Diagnose the current DGP learning design first'
    assert not m['architecture_discussion_pending'] and not decision['failed_gates_waived']
    assert not decision['agent_training_authorized'] and decision['manual_existing_L4_training_required']
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json')
    assert audit['complete'] and audit['VM_failure_retained']
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r2_return.py')
    assert audit['checker_sha256'] == 'ddd86ab2a36a3b9e6d92615b5246df500784f1ec2e5e10e3104272c528733831'
    assert audit['protocol_sha256'] == m['return_protocol_sha256'] == '87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c'
    assert audit['archive_sha256'] == m['return_archive_sha256'] == '576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9'
    assert audit['archive_bytes'] == 1324635165 and audit['members_verified'] == m['returned_files'] == 27625
    assert audit['TRAIN_assets_verified'] == 5467 and audit['saved_gradient_values_checked'] == 75324711
    assert audit['completed_snapshot_PNG_metrics_vectors_and_mean_controls_audited'] == m['snapshots_rows'] == 7810
    assert audit['selected23_preflight_complete'] and [s['update'] for s in audit['snapshots_audited']] == [0, 50]
    assert not audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert not audit['all_final3905_neural_replayed'] and not audit['native_or_reserved_used']
    assert audit['local_gradient_calls'] == audit['local_backward_calls'] == audit['local_optimizer_updates'] == 0
    imported = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_import.json')
    returned_count = verify({'outputs/cctv_dgp_feature_fusion_v32_r2_return/' + name: digest
                             for name, digest in imported['files_sha256'].items()})
    assert imported['complete'] and returned_count == 27625
    early, failure = map(read, [RETURNED / 'outputs/early_structure_stop.json', RETURNED / 'outputs/failure.json'])
    assert not early['pass'] and early['minimum'] == m['minimum_gain'] == .01
    assert early['relative_feature_error_gain'] == m['structure_gain'] < .01 and not m['early_gate_pass']
    assert failure['optimizer_updates'] == m['VM_optimizer_updates'] == 50
    assert not (RETURNED / 'outputs/results.json').exists()
    assert (RETURNED / 'outputs/stopped_dgp_candidate_v32.pth').is_file()
    prepared, review_audit, visual = map(read, [REVIEW / 'preparation.json', REVIEW / 'independent_preparation_audit.json', REVIEW / 'visual_review.json'])
    assert prepared['complete'] and review_audit['complete'] and visual['complete']
    assert review_audit['preparation_sha256'] == sha(REVIEW / 'preparation.json')
    assert review_audit['independent_return_audit_sha256'] == sha(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json')
    assert review_audit['exact_256_cells_verified'] == visual['comparison_cells_reviewed'] == 250
    assert visual['cases_reviewed'] == len(visual['rows']) == 50 and len(visual['sheets']) == 10
    assert prepared['fixed50_cases_exposed_by50'] == 0
    assert len(prepared['groups']) == m['preservation_groups_passed'] == 17
    assert not prepared['preservation_regressions_at50'] and not visual['useful_incremental_whole_face_gain_established']
    assert not visual['early_numerical_failure_waived'] and not visual['independent_final_review']
    displacement = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_displacement_v1/analysis.json')
    assert displacement['complete'] and displacement['frozen_partition_unchanged']
    assert not displacement['model_constructed'] and displacement['gradient_calls'] == displacement['optimizer_updates'] == 0
    assert displacement['partitions']['fusion']['parameters'] == 479616 and displacement['partitions']['decoder']['parameters'] == 498627

    mat = read(ROOT / 'outputs/completion_mat_mirror_comparison_v1/protocol.json')
    app_count = verify(mat['app_preservation_sha256'])
    assert app_count == 14 and len(mat['cases']) == 32 and len(mat['excluded_before_generation']) == 4
    assert closure['all32_assisted_outputs_and8_development_sheets_preserved'] and not closure['MAT_quality_qualification']
    r1 = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_independent_audit.json')
    assert r1['complete'] and r1['members_verified'] == 180 and r1['saved_gradient_values_checked'] == 75324711
    assert r1['snapshots_audited'] == [] and r1['local_optimizer_updates'] == 0
    assert sha(ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md') == '2aad6635473dad349c85ab177ec0d1fc09d1089c350bf620298f4a06f67d2299'
    assert sha(ROOT / 'PRACTICAL_OUTPUT_SCOPE.md') == '47a02049970719379727aca9f5a6707177e5894a35c211814076aa7c628bdb2b'
    assert m['local_gradient_calls'] == m['local_optimizer_updates'] == m['VM_writes'] == 0
    assert not m['training_started_by_agent'] and not m['app_promotion'] and not m['new_model_code_or_trainer_released']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': current, 'previous_MAT_bindings_preserved': old,
               'previous_final_readback_bindings_preserved': final_count, 'full_previous_handoff_preserved': True,
               'returned_files_verified': returned_count, 'snapshot_rows_previously_independently_audited': 7810,
               'fixed50_review_complete': True, 'unchanged_app_bindings_verified': app_count,
               'R1_failure_and_MAT_quality_limits_preserved': True, 'prior_actual_Windows_backup_receipt_preserved': 4431,
               'early_one_percent_failure_preserved': True, 'user_design_discussion_satisfied': True,
               'selected_direction': decision['selected_direction'], 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
               'VM_writes': 0, 'training_started_by_agent': False, 'manual_VM_training_required': True,
               'app_promotion': False, 'independent_final_review': False, 'goal_complete': False,
               'seconds': time.monotonic() - started}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
