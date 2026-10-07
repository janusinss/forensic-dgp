"""Independent readback of released diagnostic and retained V31 closure."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'


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
    start = time.monotonic()
    m = read(OUT / 'milestone.json')
    count = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    assert not any(m[key] for key in ['actual_VM_execution_started', 'app_changes', 'new_training_recipe',
                                     'native_or_reserved_used', 'quality_qualification', 'app_promotion'])
    assert m['manual_VM_required'] and m['local_gradient_queries'] == m['optimizer_updates'] == 0
    assert m['local_original_DGP_forwards'] == m['local_candidate_DGP_forwards'] == 4
    doc = m['document']; before = (ROOT / doc['before_path']).read_bytes(); after = (ROOT / doc['name']).read_bytes()
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == '9a2d845b4398ade0b49f42f8ca8cefbc7bff0e7f8d161fb89cf750cceaedcb1f'
    prior = verify(previous['new_evidence_sha256'], m['previous_document_locations'])
    assert prior == 55
    closed = read(PREVIOUS / 'independent_closure_audit.json')
    assert closed['complete'] and closed['milestone_sha256'] == m['previous_milestone_sha256']
    assert closed['returned_files_preserved'] == 27623 and closed['actual_Windows_backup_stamps_preserved'] == 4431
    assert closed['one_percent_structure_failure_and_stopped_checkpoint_preserved']
    final = read(PREVIOUS / 'final_readback.json')
    assert final['complete'] and final['independent_milestone_pass']
    final_count = verify(final['evidence_sha256'], m['previous_document_locations'])
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']:
        assert sha(ROOT / name) == previous['new_evidence_sha256'][name]
    p = read(PACKET / 'protocol.json')
    assert sha(PACKET / 'protocol.json') == m['protocol_sha256'] == '5926dea8c477a9d51ff97186e6fec2b0d8eab8e67a7cf7a0ed1a6c716756e0af'
    basis_count = verify(p['local_basis_sha256'])
    assert p['selected_tensors'] == 23 and len(p['fusion_parameter_names']) == 11 and len(p['decoder_parameter_names']) == 12
    assert p['fusion_parameters'] == 479616 and p['decoder_parameters'] == 498627
    assert p['gradient_queries'] == m['gradient_queries_bound'] == 280
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == 0
    assert p['budgets']['minimum_free_disk_bytes'] == 6 * 1024**3 and p['budgets']['worker_seconds'] == 600
    assert not p['new_training_recipe_created'] and not (PACKET / 'outputs').exists()
    reviewed_root = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation_r1'
    audit = read(reviewed_root / 'independent_packet_audit.json')
    assert audit['complete'] and audit['protocol_sha256'] == m['protocol_sha256'] and audit['archive_sha256'] == m['archive_sha256']
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_feature_fusion_gradient_v1_packet.py')
    assert audit['packet_files_verified'] == 3 and audit['regressions_passed'] == 7
    assert audit['two_paired_ten_reference_cohorts_verified'] and audit['unexposed_excludes_all50_touched']
    assert audit['Windows_pre_neural_gradient_rejection'] and audit['read_only_local_Bash_syntax_pass']
    assert audit['five_manual_steps_verified'] and audit['PuTTY_downloads_separate']
    assert audit['local_neural_or_gradient_calls'] == audit['optimizer_updates'] == 0 and not audit['VM_execution_started']
    original_fail = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation/bash_syntax.json')
    successful = read(reviewed_root / 'bash_syntax.json')
    assert not original_fail['complete'] and "couldn't create signal pipe" in original_fail['stderr']
    assert successful['complete'] and successful['exit_code'] == 0
    assert original_fail['script_sha256'] == successful['script_sha256'] == sha(PACKET / 'scripts/run_fusion.sh')
    proof = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_forward_review/review.json')
    proof_count = verify(proof['source_bindings_sha256'])
    assert proof['complete'] and proof['protocol_sha256'] == m['protocol_sha256']
    assert proof['reviewer_sha256'] == sha(ROOT / 'scripts/review_cctv_dgp_feature_fusion_gradient_v1_forward.py')
    assert proof['actual_named_parameter_order_matches_packet'] and proof['FPN_gradient_connectivity_still_unmeasured']
    assert proof['original_DGP_forwards'] == proof['candidate_DGP_forwards'] == 4
    assert proof['gradient_queries'] == proof['backwards'] == proof['optimizer_updates'] == 0
    assert len(proof['rows']) == 4 and all(row['grad_fn'] is None and row['exact_raw_parity'] and row['exact_PNG_parity'] for row in proof['rows'])
    failure = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/failure.json')
    early = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/early_structure_stop.json')
    assert failure['optimizer_updates'] == 50 and failure['resume_permitted'] is False
    assert not early['pass'] and early['minimum'] == .01 and early['relative_feature_error_gain'] == .006945252687208803
    app = read(ROOT / 'outputs/dgp_app_v3_integration_record.json')
    app_count = verify({name: digest for name, digest in app['sources_sha256'].items() if name != 'static/face_workflow.js'})
    assert sha(ROOT / 'static/face_workflow.js') == '9be6cfab57e0041f87571cd38ba6d0d1e0850742b8fafd498f1b1d126137a060'
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == app['checkpoint_sha256']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': count, 'prior55_bindings_preserved': prior,
               'prior_final_readback_bindings_preserved': final_count, 'packet_local_basis_verified': basis_count,
               'CPU_parity_source_bindings_verified': proof_count, 'exact_initial_four_case_CPU_parity': True,
               'actual23_parameter_order_verified': True, 'manual_gradient_bound': 280, 'optimizer_updates': 0,
               'local_gradient_queries': 0, 'VM_execution_started': False,
               'Bash_startup_failure_and_unchanged_success_preserved': True, 'V31_early_failure_retained': True,
               'previous_independent_backup_stamps_receipt_preserved': 4431, 'unchanged_app_sources': app_count,
               'app_model_unchanged': True, 'full_previous_handoff_preserved': True,
               'native_or_reserved_used': False, 'quality_qualification': False,
               'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
