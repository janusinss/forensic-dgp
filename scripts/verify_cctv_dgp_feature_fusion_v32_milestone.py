"""Independent readback of diagnostic result, finite V32 and preserved history."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32'
RETURN = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json'); count = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    assert m['new_training_recipe_prepared'] and m['manual_VM_required']
    assert not any(m[k] for k in ['actual_V32_training_started_by_agent', 'app_changes', 'native_or_reserved_used', 'quality_qualification', 'app_promotion'])
    assert m['local_gradient_queries'] == m['local_optimizer_updates'] == m['diagnostic_optimizer_updates'] == 0
    doc = m['document']; before = (ROOT / doc['before_path']).read_bytes(); after = (ROOT / doc['name']).read_bytes()
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest() == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == 'e35f196a8ee9de12c89422f0a5d18523ab830ce3e07bda28b783697336adc8b1'
    prior = verify(previous['new_evidence_sha256'], m['previous_document_locations'])
    final = read(PREVIOUS / 'final_readback.json'); assert final['complete'] and final['independent_milestone_pass']
    final_count = verify(final['evidence_sha256'], m['previous_document_locations'])
    previous_closed = read(PREVIOUS / 'independent_closure_audit.json')
    assert previous_closed['complete'] and previous_closed['milestone_sha256'] == m['previous_milestone_sha256']
    assert previous_closed['previous_independent_backup_stamps_receipt_preserved'] == 4431
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json')
    imported = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return_import.json')
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['VM_failure_retained']
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py') == 'cefd32a14d5b5453b8acf656d9be6daa14096d91fb36956eb954632794b1a5e5'
    assert audit['members_verified'] == imported['members'] == 756
    assert audit['archive_sha256'] == imported['archive_sha256'] == m['diagnostic_return_archive_sha256']
    for name, digest in imported['files_sha256'].items(): assert sha(RETURN / name) == digest, name
    result = read(RETURN / 'outputs/results.json')
    assert result['complete'] and result['component_gradient_calls'] == 280
    assert result['optimizer_updates'] == result['backwards'] == result['epochs'] == 0
    assert result['DGP_and_recognizer_unmodified'] and not result['new_checkpoint_created']
    assert sum(r['values_checked'] for r in audit['cohort_gradient_analysis']) == m['saved_gradient_values_verified'] == 301298844
    assert audit['CPU_replay']['cases_at_both_states'] == 200 and audit['local_gradient_calls'] == audit['local_backward_calls'] == audit['local_optimizer_updates'] == 0
    analysis = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_analysis/analysis.json')
    analysis_count = verify(analysis['source_bindings_sha256'])
    assert analysis['complete'] and analysis['new_finite_fusion_partition_pilot_justified_for_preparation']
    assert len(analysis['rows']) == 4 and not analysis['disconnected_improvement_tensors']
    for interpreted, measured in zip(analysis['rows'], audit['cohort_gradient_analysis']):
        assert (interpreted['state'], interpreted['cohort']) == (measured['state'], measured['cohort'])
        assert interpreted['all23_connected'] == all(n > 0 for n in measured['selected23_improvement_norms'].values())
        for label in ['fusion', 'decoder']:
            partition = measured['partition_analysis']['feature_fusion' if label == 'fusion' else 'decoder']
            assert interpreted[label + '_improvement_norm'] == partition['improvement_norm']
            assert interpreted[label + '_preservation_norm'] == partition['preservation_norm']
        if interpreted['state'] == 0:
            assert all(v < 0 for v in list(interpreted['fusion_component_directional_derivatives'].values())[:3])
    prep = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation'
    checked, prepared = read(prep / 'independent_packet_audit.json'), read(prep / 'preparation.json')
    p = read(PACKET / 'protocol.json')
    assert checked['complete'] and checked['regressions_passed'] == 11 and checked['packet_files_verified'] == 11
    assert checked['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_packet.py')
    assert checked['protocol_sha256'] == prepared['protocol_sha256'] == sha(PACKET / 'protocol.json') == m['protocol_sha256']
    assert checked['archive_sha256'] == prepared['archive_sha256'] == m['execution_archive_sha256']
    assert checked['losses_optimizer_gates_data_schedule_forward_and_normalization_unchanged']
    assert checked['Windows_pre_neural_training_rejection'] and checked['read_only_Bash_syntax_pass']
    assert checked['five_manual_steps_verified'] and checked['PuTTY_downloads_separate']
    assert p['selected_parameters'] == 978243 and p['selected_tensors'] == 23
    assert p['optimizer_updates'] == 800 and p['budgets']['minimum_free_disk_bytes'] == 8 * 1024**3
    assert not (PACKET / 'outputs').exists() and not checked['actual_VM_training_started']
    forward = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_forward_review/review.json')
    forward_count = verify(forward['source_bindings_sha256'])
    assert forward['complete'] and forward['reviewer_sha256'] == sha(ROOT / 'scripts/review_cctv_dgp_feature_fusion_v32_forward.py')
    assert forward['protocol_sha256'] == m['protocol_sha256'] and forward['actual_named_parameter_order_matches_packet']
    assert forward['all_buffers_unchanged'] and forward['train_true_retains_evaluation_normalization']
    assert forward['gradient_queries'] == forward['backwards'] == forward['optimizer_updates'] == 0
    assert forward['original_DGP_forwards'] == forward['candidate_DGP_forwards'] == 4
    assert len(forward['rows']) == 4 and all(r['exact_raw_parity'] and r['exact_PNG_parity'] and r['grad_fn'] is None for r in forward['rows'])
    early = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/early_structure_stop.json')
    failed = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/failure.json')
    assert early['minimum'] == .01 and not early['pass'] and early['relative_feature_error_gain'] == .006945252687208803
    assert failed['optimizer_updates'] == 50 and not failed['resume_permitted']
    app = read(ROOT / 'outputs/dgp_app_v3_integration_record.json')
    app_count = verify({n: d for n, d in app['sources_sha256'].items() if n != 'static/face_workflow.js'})
    assert sha(ROOT / 'static/face_workflow.js') == '9be6cfab57e0041f87571cd38ba6d0d1e0850742b8fafd498f1b1d126137a060'
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == app['checkpoint_sha256']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': count, 'previous_bindings_preserved': prior, 'prior_final_readback_preserved': final_count,
               'returned_files_preserved': len(imported['files_sha256']), 'saved_gradient_values_checked_by_pinned_return_audit': 301298844,
               'CPU_return_cases_verified': 200, 'analysis_bindings_verified': analysis_count,
               'V32_packet_and_11_regressions_verified': True, 'V32_forward_bindings_verified': forward_count,
               'V31_one_percent_failure_preserved': True, 'previous_actual_Windows_backup_receipt_preserved': 4431,
               'unchanged_app_sources': app_count, 'original_app_checkpoint_preserved': True,
               'full_previous_handoff_preserved': True, 'actual_V32_training_started_by_agent': False,
               'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'manual_VM_required': True,
               'native_or_reserved_used': False, 'quality_qualification': False, 'goal_complete': False,
               'seconds': time.monotonic() - started}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
