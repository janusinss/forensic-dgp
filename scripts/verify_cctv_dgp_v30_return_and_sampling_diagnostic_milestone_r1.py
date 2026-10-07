"""Independent readback of the new milestone and retained research/maintenance history."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v30_return_and_sampling_gradient_v1_milestone'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, alternate=None):
    for name, digest in mapping.items():
        path = (ROOT / (alternate or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    start = time.monotonic(); m = read(OUT / 'milestone.json'); current = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['V30_optimizer_updates'] == 50 and not m['V30_capacity_pass']
    assert m['unchanged_minimum'] == .01 and m['diagnostic_gradient_queries_bound'] == 280 and m['diagnostic_optimizer_updates'] == 0
    assert m['local_gradient_calls'] == m['local_backward_calls'] == m['local_optimizer_updates'] == 0
    assert not m['app_promotion'] and not m['native_or_reserved_used'] and not m['goal_complete'] and m['goal_status'] == 'active'
    assert not m['new_training_recipe_created'] and not m['diagnostic_actual_VM_run_started'] and not m['new_VM_actions']
    for row in m['documents']:
        before = (ROOT / row['before_path']).read_bytes(); after = (ROOT / row['name']).read_bytes()
        assert sha(ROOT / row['before_path']) == row['before_sha256'] and sha(ROOT / row['name']) == row['after_sha256']
        split = before.index(b'\n') + 1
        assert after[:split] + after[split + row['addition_bytes']:] == before
        addition = after[split:split + row['addition_bytes']].decode('ascii')
        for literal in ['0.805717%', '280 gradient queries', 'zero optimizer', 'Goal active/incomplete', 'V30-not-started']:
            assert literal in addition
    closure_path = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/closure_manifest.json'
    assert sha(closure_path) == m['previous_cleanup_closure_sha256'] == 'b409d886495f3e9375b54bef1fc0ad6dd32aabfe8e73ea26ce1e280e44b8507c'
    cleanup = read(closure_path); cleanup_count = verify(cleanup['new_evidence_sha256'], m['previous_cleanup110_original_locations'])
    assert cleanup_count == 110 and cleanup['removed_inactive_cache_files'] == 4429
    from verify_cctv_dgp_vm_storage_cleanup_20261007_v1 import history
    older_documents = {row['name']: 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/before_docs/' + row['name'] for row in m['documents']}
    history_counts = history(older_documents)
    # Verify the complete previous cleanup evidence too, with its original root documents.
    old_cleanup = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/closure_manifest.json'
    old = read(old_cleanup)
    alternate = {r['name']: 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1/recovery_metadata/project_instructions/' + r['name'] for r in m['documents']}
    older_cleanup_count = verify(old['new_evidence_sha256'], alternate)
    history_record = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/local_history_pre_cleanup_audit.json')
    assert sha(old_cleanup) == history_record['previous_cleanup_closure_sha256'] and older_cleanup_count == 71
    from verify_cctv_dgp_vm_storage_cleanup_20261007_v2 import verify as verify_cache_stamps
    backup = verify_cache_stamps()
    assert backup['actual_Windows_cache_files_preserved'] == 4431 and backup['actual_Windows_cache_bytes_preserved'] == 39448585279
    imported_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return_import.json'; imported = read(imported_path)
    assert imported['complete'] and imported['members'] == 27622 and not imported['returned_code_executed']
    return_count = verify({'outputs/cctv_dgp_broader_mean_v30_return/' + name: digest for name, digest in imported['files_sha256'].items()})
    assert return_count == 27622
    audit = read(ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json')
    assert audit['complete'] and audit['VM_failure_retained'] and audit['original_checker_failure_retained']
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_broader_mean_v30_return_r1.py')
    assert [r['update'] for r in audit['snapshots_audited']] == [0, 50] and not audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert audit['early_structure_minimum_unchanged'] == .01 and audit['receipt_arithmetic_absolute_tolerance'] == 1e-12
    assert audit['archive_sha256'] == m['V30_archive_sha256'] == imported['archive_sha256']
    visual = read(ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1/visual_review.json')
    assert visual['complete'] and visual['cases_reviewed'] == 50 and len(visual['rows']) == 50
    assert all(len(row['regions_inspected']) == 5 and not row['V30_useful_whole_face_gain_established'] for row in visual['rows'])
    measured = read(ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1/preparation.json')
    assert measured['exact_cells_checked'] == 250 and not measured['preservation_regressions_at50']
    assert measured['unique_cases_optimized_by50'] == 250 and measured['unique_references_touched_by50'] == 218
    for snapshot in audit['snapshots_audited']:
        assert len(snapshot['groups']) == 17
    gain = 1 - audit['snapshots_audited'][1]['groups']['degraded']['landmark_high_frequency_MSE'] / audit['snapshots_audited'][0]['groups']['degraded']['landmark_high_frequency_MSE']
    assert abs(gain - m['V30_structure_gain_fraction']) <= 1e-12 and gain < .01
    execution = read(ROOT / 'outputs/cctv_dgp_broader_mean_v30_return/outputs/execution_receipt.json')
    assert execution['optimizer_updates'] == execution['backwards'] == 50
    assert execution['worker_seconds'] <= 4500 and execution['fit_seconds'] <= 3600
    assert execution['peak_allocated_VRAM_bytes'] <= 20 * 1024**3
    packet = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_preparation/independent_packet_audit.json')
    assert packet['complete'] and packet['protocol_sha256'] == m['diagnostic_protocol_sha256']
    assert packet['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_v30_sampling_gradient_v1_packet.py')
    assert packet['regressions_passed'] == 11 and packet['gradient_queries_bound'] == 280 and packet['optimizer_updates'] == 0
    assert packet['reference_repetition_matched']
    assert packet['distinct_references_in_each_cohort'] == 48
    assert packet['read_only_Bash_syntax_pass'] and packet['Windows_pre_neural_gradient_rejection']
    assert not packet['VM_execution_started'] and not (ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_vm/outputs').exists()
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == m['app_record_sha256'] == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(app_path); assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    report = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
              'new_bindings_verified': current, 'previous_cleanup_bindings': cleanup_count,
              'older_cleanup_bindings': older_cleanup_count, 'previous_research_history_bindings': history_counts,
              'V30_returned_files_reverified': return_count, 'full_previous_document_bodies_preserved': True,
              'actual_Windows_cache_files_unchanged_stamps': 4431, 'app22_bindings_unchanged': True,
              'V30_failed_early_gate_retained': True, 'original_local_checker_failure_retained': True,
              'diagnostic_packet_unrun_and_no_optimizer': True, 'reserved_final_used': False,
              'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    failure_path = OUT / 'original_closure_checker_failure_bound.json'
    failure = read(failure_path)
    assert failure['checker_exit_code'] == 1 and failure['original_milestone_sha256'] == sha(OUT / 'milestone.json')
    assert failure['original_checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_v30_return_and_sampling_diagnostic_milestone.py')
    for name, row in failure['verified_original_archives'].items():
        assert sha(ROOT / row['path']) == old['new_evidence_sha256'][name] == row['sha256']
    report['original_bound_checker_failure_retained'] = True
    report['original_closure_checker_failure_sha256'] = sha(failure_path)
    report['corrected_previous_cleanup_document_archives'] = failure['verified_original_archives']
    with (OUT / 'independent_closure_audit_r1.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
