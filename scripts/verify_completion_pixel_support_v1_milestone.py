"""Independent milestone readback; no model calls or historical pilot execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_pixel_support_v1_r1'
PREVIOUS = ROOT / 'outputs/dgp_mask_review_race_fix_v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    m = read(OUT / 'milestone.json'); new_count = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    assert m['new_model_forwards'] == m['local_gradient_calls'] == m['optimizer_updates'] == 0
    assert not m['mask_edits'] and not m['app_or_model_changes'] and not m['app_promotion']
    assert not m['quality_qualification'] and not m['reserved_final_used']
    doc = m['document']; before = (ROOT / doc['before_path']).read_bytes(); after = (ROOT / doc['name']).read_bytes()
    assert sha(ROOT / doc['before_path']) == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    split = before.index(b'\n') + 1; assert after[:split] + after[split + doc['addition_bytes']:] == before
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == 'e26d5f973722184a612c86db7a1e2d2987db4eee0476c78c0baf7a147d2c60df'
    prior_count = verify(previous['new_evidence_sha256'], m['previous_document_locations']); assert prior_count == 72
    final = read(PREVIOUS / 'final_readback.json'); prior_final_count = verify(final['evidence_sha256'], m['previous_document_locations'])
    assert final['complete'] and final['independent_closure_pass']
    prior_closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert prior_closure['complete'] and prior_closure['milestone_sha256'] == m['previous_milestone_sha256']
    assert prior_closure['actual_Windows_backup_stamps_preserved'] == 4431
    assert prior_closure['other12_app_sources_unchanged'] and prior_closure['V31_packet_unchanged']
    for row in previous['documents']:
        if row['name'] != 'PROJECT_HANDOFF.md': assert sha(ROOT / row['name']) == row['after_sha256']
    plan, results = read(OUT / 'plan.json'), read(OUT / 'results.json')
    source_count = verify(plan['sources_sha256']); assert source_count == 300
    assert results['complete'] and results['plan_sha256'] == sha(OUT / 'plan.json')
    assert results['raw_completion_compositions'] == 28 and results['empty_mask_exact_bypasses'] == 4 and results['ROI_recounts'] == 8
    audit = read(OUT / 'independent_audit.json')
    assert audit['complete'] and audit['results_sha256'] == sha(OUT / 'results.json')
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_completion_pixel_support_v1_r1.py')
    assert audit['eligible_Off_compositions_independently_verified'] == 32 and audit['support_ROIs_independently_recounted'] == 8
    assert audit['original_analysis_failure_preserved'] and audit['original_preparation_failure_and_operator_review_precedence_verified']
    visual = read(OUT / 'visual_review.json')
    assert visual['complete'] and visual['results_sha256'] == sha(OUT / 'results.json')
    assert visual['independent_saved_pixel_audit_sha256'] == sha(OUT / 'independent_audit.json')
    assert len(visual['sheets']) == 2 and not visual['automatic_or_assisted_quality_qualification']
    for row in visual['sheets']:
        assert row['actually_viewed'] and row['requested_detail'] == 'original' and row['source_pixels'] == [1320, 1192]
        assert sha(OUT / row['path']) == row['sha256']
    status_root = ROOT / 'outputs/cctv_dgp_v31_progress_status_v1'
    transport = read(status_root / 'launch_status_transport.json')
    assert transport['exit_code'] == 0 and transport['stdout_sha256'] == sha(status_root / 'launch_status.log')
    assert transport['stderr_sha256'] == sha(status_root / 'launch_status_stderr.log')
    live = read(status_root / 'launch_status.log')
    assert live['complete'] and live['read_only'] and not live['training_started_by_agent']
    assert live['snapshot_utc'] == m['V31_snapshot_utc']
    process = next(p for p in live['V31_processes'] if p['pid'] == m['live_V31_PID_at_snapshot'])
    assert '--run' in process['args'] and m['V31_protocol_sha256'] in process['args']
    assert live['GPU']['exit_code'] == 0 and live['GPU']['stdout'].startswith('4942, ')
    assert any("'V31_snapshot': 50" in line for line in live['trainer_log']['last_lines'])
    assert 'results.json' not in live['outputs_present'] and 'failure.json' not in live['outputs_present']
    assert 'export_receipt' not in live and not m['V31_terminal_receipt_observed'] and not m['V31_early_gate_pass_observed']
    assert not m['V31_training_started_by_agent']
    assert sha(ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json') == m['V31_protocol_sha256']
    assert sha(ROOT / 'outputs/cctv-dgp-profile-batches-v31-execution.tar.gz') == m['V31_archive_sha256']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': new_count, 'prior72_bindings_preserved': prior_count,
               'prior_final_readback_bindings_preserved': prior_final_count, 'diagnostic_sources_verified': source_count,
               'complete_previous_handoff_preserved': True, 'scope_documents_unchanged': 2,
               'both_failed_diagnostic_attempts_preserved': True, '32_compositions_or_bypasses_and8_ROIs_audited': True,
               'timestamped_human_VM_run_verified': True, 'live_PID_at_recorded_snapshot': 4942,
               'VM_workload_started_stopped_or_modified_by_agent': False, 'V31_packet_and_gates_unchanged': True,
               'app_or_model_changes': False, 'new_inference_or_training': False,
               'quality_qualification': False, 'goal_complete': False}
    with (OUT / 'independent_milestone_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
