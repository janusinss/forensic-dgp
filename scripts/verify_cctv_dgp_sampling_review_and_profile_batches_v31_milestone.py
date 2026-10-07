"""Independent closure readback, retained failure history and local-cache stamps."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_sampling_review_and_profile_batches_v31_milestone'
PREVIOUS = ROOT / 'outputs/dgp_v30_return_and_sampling_gradient_v1_milestone'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, alternate=None):
    for name, digest in mapping.items():
        path = (ROOT / (alternate or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    start = time.monotonic(); m = read(OUT / 'milestone.json'); count = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['diagnostic_gradient_queries'] == 280 and m['diagnostic_optimizer_updates'] == 0
    assert m['local_gradient_calls'] == m['local_backward_calls'] == m['local_optimizer_updates'] == 0
    assert not m['V31_actual_VM_training_started'] and not m['native_or_reserved_used'] and not m['app_promotion']
    assert m['goal_status'] == 'active' and not m['goal_complete'] and not m['new_VM_actions']
    for row in m['documents']:
        before = (ROOT / row['before_path']).read_bytes(); after = (ROOT / row['name']).read_bytes()
        assert sha(ROOT / row['before_path']) == row['before_sha256'] and sha(ROOT / row['name']) == row['after_sha256']
        split = before.index(b'\n') + 1; assert after[:split] + after[split + row['addition_bytes']:] == before
        addition = after[split:split + row['addition_bytes']].decode('ascii')
        for text in ['280 saved', 'zero optimizer', '16.2719%', 'batch formation only', 'Actual V31 VM training remains unstarted', 'Goal active/incomplete']:
            assert text in addition
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == '2dddd0b8bd9f4c443e16a103afdecbf9e940266aaed554a68156bd5211594a09'
    old = read(PREVIOUS / 'milestone.json'); old_count = verify(old['new_evidence_sha256'], m['previous56_original_locations'])
    assert old_count == 56 and not old['V30_capacity_pass'] and old['V30_optimizer_updates'] == 50
    assert sha(PREVIOUS / 'final_readback.json') == m['previous_final_readback_sha256']
    corrections = read(PREVIOUS / 'final_readback.json'); correction_count = verify(corrections['correction_evidence_sha256'])
    assert corrections['complete'] and corrections['independent_closure_pass']
    closure = read(PREVIOUS / 'independent_closure_audit_r2.json')
    assert closure['complete'] and closure['V30_failed_early_gate_retained'] and closure['R1_schema_failure_retained']
    cleanup = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/closure_manifest.json')
    assert sha(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/closure_manifest.json') == old['previous_cleanup_closure_sha256']
    cleanup_count = verify(cleanup['new_evidence_sha256'], old['previous_cleanup110_original_locations']); assert cleanup_count == 110
    earlier = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/closure_manifest.json')
    earlier_docs = {r['name']: 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1/recovery_metadata/project_instructions/' + r['name'] for r in m['documents']}
    earlier_count = verify(earlier['new_evidence_sha256'], earlier_docs); assert earlier_count == 71
    from verify_cctv_dgp_vm_storage_cleanup_20261007_v1 import history
    historic_docs = {r['name']: 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/before_docs/' + r['name'] for r in m['documents']}
    history_counts = history(historic_docs)
    from verify_cctv_dgp_vm_storage_cleanup_20261007_v2 import verify as verify_cache_stamps
    cache = verify_cache_stamps()
    assert cache['actual_Windows_cache_files_preserved'] == 4431 and cache['actual_Windows_cache_bytes_preserved'] == 39448585279
    imported = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_return_import.json')
    assert imported['complete'] and imported['members'] == 756 and imported['local_gradient_calls'] == 0
    assert not imported['returned_code_executed']
    returned_count = verify({'outputs/cctv_dgp_v30_sampling_gradient_v1_return/' + name: digest for name, digest in imported['files_sha256'].items()})
    assert returned_count == 756 and imported['archive_sha256'] == m['diagnostic_archive_sha256']
    audit = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['optimizer_updates'] == 0
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_v30_sampling_gradient_v1_return.py')
    assert audit['members_verified'] == 756 and audit['CPU_replay']['cases_at_both_states'] == 200
    result = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_return/outputs/results.json')
    assert result['component_gradient_calls'] == 280 and result['optimizer_updates'] == result['backwards'] == result['epochs'] == 0
    assert result['V30_failure_retained'] and result['DGP_and_recognizer_unmodified']
    analysis = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_analysis/analysis.json')
    assert analysis['complete'] and analysis['independent_audit_sha256'] == sha(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json')
    assert analysis['all_selected12_improvement_gradients_nonzero'] and len(analysis['pixel_regressions']) == 5
    assert sum(r['profile'] == 'clear' for r in analysis['pixel_regressions']) == 4
    packet = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_preparation/independent_packet_audit.json')
    assert packet['complete'] and packet['protocol_sha256'] == m['V31_protocol_sha256'] and packet['archive_sha256'] == m['V31_archive_sha256']
    assert packet['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_profile_batches_v31_packet.py')
    assert packet['regressions_passed'] == 10 and packet['original_initialization_architecture_losses_optimizer_and_gates_unchanged']
    assert packet['first_epoch_covers_all3905_once'] and packet['every_batch_has_paired_clear_and_four_degradations']
    assert packet['Windows_pre_neural_training_rejection'] and packet['read_only_Bash_syntax_pass'] and not packet['actual_VM_training_started']
    assert not (ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31/outputs').exists()
    v30 = read(ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json')
    assert v30['complete'] and v30['VM_failure_retained'] and not v30['necessary_capacity_pass'] and not v30['training_completed800']
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == old['app_record_sha256'] == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(app_path); assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': count, 'previous_bindings_verified': old_count, 'previous_correction_bindings_verified': correction_count,
               'cleanup_bindings_verified': cleanup_count, 'earlier_cleanup_bindings_verified': earlier_count,
               'research_history_bindings_verified': history_counts, 'diagnostic_return_files_reverified': 756,
               'complete_previous_document_bodies_preserved': 3, 'actual_Windows_cache_stamps_preserved': 4431,
               'app22_bindings_preserved': True, 'all_prior_failure_records_retained': True,
               'V31_actual_VM_training_started': False, 'local_gradient_calls': 0, 'app_promotion': False,
               'reserved_final_used': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
