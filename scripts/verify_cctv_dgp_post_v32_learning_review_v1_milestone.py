"""Independent closure of measured losses and the finite, unlaunched diagnostic."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_post_v32_learning_review_v1_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1'
LOSSES = ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1'
PACKET = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'
PREP = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_preparation'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(mapping)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json')
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    new_count = verify(m['new_evidence_sha256'])
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256']
    old_count = verify(previous['new_evidence_sha256'], m['previous_document_locations'])
    old_closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert old_closure['complete'] and old_closure['milestone_sha256'] == m['previous_milestone_sha256']
    assert old_closure['returned_files_verified'] == 27625 and old_closure['prior_actual_Windows_backup_receipt_preserved'] == 4431
    old_final = verify(read(PREVIOUS / 'final_readback.json')['evidence_sha256'], m['previous_document_locations'])
    doc = m['document']
    before, after = (ROOT / doc['before_path']).read_bytes(), (ROOT / doc['name']).read_bytes()
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest() == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    loss, loss_audit, plan = map(read, [LOSSES / 'analysis.json', LOSSES / 'independent_audit.json', LOSSES / 'plan.json'])
    assert loss['complete'] and loss_audit['complete']
    assert loss_audit['analysis_sha256'] == sha(LOSSES / 'analysis.json') and loss_audit['plan_sha256'] == sha(LOSSES / 'plan.json')
    assert loss_audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_post_v32_saved_losses_v1.py')
    source_count = verify(plan['source_bindings_sha256'])
    assert source_count == loss_audit['bindings_verified'] == 387
    assert len(loss['rows']) == loss_audit['cases_verified'] == 50 and loss_audit['independent_raw_feature_region_quantities_verified'] == 400
    assert loss['CPU_recognizer_images'] == loss_audit['independent_raw_recognizer_images'] == 100
    assert loss['DGP_forwards'] == loss['local_gradient_calls'] == loss['local_backward_calls'] == loss['optimizer_updates'] == 0
    assert loss_audit['gradient_calls'] == loss_audit['optimizer_updates'] == 0
    assert loss['all50_fixed_PREVIEW_TRAIN_not_optimized_by50'] and loss_audit['fixed50_not_optimized_by50']
    degraded = loss['groups']['degraded']
    assert abs(degraded['raw_feature_gain'] - .009686835344338562) < 1e-12
    assert abs(degraded['PNG_feature_gain'] - .009915577984984658) < 1e-12
    assert degraded['active_preservation_terms_at50']['ArcFace_regression'] == 8
    assert loss['groups']['clear']['active_preservation_terms_at50']['pixel_regression'] == 1
    assert loss_audit['first_preparation_schema_failure_preserved'] and loss_audit['PNG_gate_thresholds_unchanged']
    assert not loss_audit['causal_gradient_or_optimizer_history_proof']
    research = read(ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/primary_research.json')
    decision = read(ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json')
    assert research['complete'] and len(research['sources']) == 6 and research['no_third_party_tutorials_used']
    assert decision['discussion_satisfied'] and decision['answer'] == 'apply the best approach and do research also if needed'
    assert not decision['failed_gates_waived'] and not research['new_loss_or_model_selected_from_paper_alone']
    packet, preparation, audit = map(read, [PACKET / 'protocol.json', PREP / 'preparation.json', PREP / 'independent_packet_audit.json'])
    assert audit['complete'] and audit['protocol_sha256'] == preparation['protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert audit['archive_sha256'] == preparation['archive_sha256'] == '22d76f3637a3ac580bc18fbac79ff5e3d24a61ec0a00ce082fde78c7069c6052'
    assert preparation['archive_bytes'] == 54322 and preparation['packet_files'] == audit['packet_files_verified'] == 3
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_v32_loss_gradient_v1_packet_audit_r1.py')
    assert audit['adversarial_boundary_regressions_passed'] == 6 and audit['Windows_pre_neural_gradient_rejection']
    assert audit['Python3_10_source_syntax_pass'] and audit['read_only_local_Bash_syntax_pass'] and audit['PuTTY_downloads_separate']
    assert packet['optimizer_updates'] == packet['backwards'] == packet['epochs'] == 0 and packet['gradient_queries'] == 280
    assert packet['selected_parameters'] == 978243 and packet['selected_tensors'] == 23
    original_p = read(ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2/protocol.json')
    assert packet['retained_gates'] == original_p['retained_capacity_gates']
    assert packet['terms'] == original_p['terms'] and packet['parameter_layout'] == original_p['parameter_layout']
    assert packet['budgets']['worker_seconds'] == 600 and packet['budgets']['export_uncompressed_bytes'] == 1536 * 1024**2
    assert not (PACKET / 'outputs').exists() and not audit['VM_execution_started'] and not preparation['actual_VM_run_started']
    assert not packet['new_training_recipe_created'] and packet['human_manual_VM_execution_required']
    assert not packet['native_or_reserved_used'] and not packet['app_promotion']
    assert read(PREP / 'packet_audit_failure_v1.json')['same_read_only_Bash_recheck']['exit_code'] == 3221225794
    corrected, parsed = map(read, [PREP / 'packet_audit_r1_correction.json', PREP / 'bash_syntax_approved.json'])
    assert corrected['original_checker_packet_protocol_worker_and_caps_unchanged']
    assert parsed['complete'] and parsed['exit_code'] == 0 and parsed['read_only_syntax_check'] and not parsed['VM_or_training_executed']
    assert parsed['script_sha256'] == sha(PACKET / 'scripts/run_loss_gradient.sh')
    app_count = verify(read(ROOT / 'outputs/completion_mat_mirror_comparison_v1/protocol.json')['app_preservation_sha256'])
    assert app_count == 14
    assert m['local_gradient_calls'] == m['local_optimizer_updates'] == m['diagnostic_optimizer_update_cap'] == 0
    assert not m['training_started_by_agent'] and not m['VM_execution_started'] and not m['app_changes'] and not m['app_promotion']
    assert m['original_one_percent_gate_failure_preserved'] and not m['preservation_terms_and_gates_removed']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': new_count, 'previous_R2_return_bindings_preserved': old_count,
               'previous_final_readback_bindings_preserved': old_final, 'full_previous_handoff_preserved': True,
               'independently_audited_loss_source_bindings_preserved': source_count, 'fixed50_loss_findings_verified': True,
               'primary_research_limits_and_user_decision_preserved': True, 'distinct_source_balanced100_case_packet_verified': True,
               'unchanged_app_bindings_verified': app_count, 'prior_actual_Windows_backup_receipt_preserved': 4431,
               'R2_failures_and_MAT_quality_limits_preserved': True, 'first_schema_and_Bash_sandbox_failures_preserved': True,
               'diagnostic_gradient_query_cap': 280, 'diagnostic_optimizer_update_cap': 0,
               'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_execution_started': False,
               'training_started_by_agent': False, 'manual_VM_training_required': True,
               'app_promotion': False, 'independent_final_review': False, 'goal_complete': False,
               'seconds': time.monotonic() - started}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
