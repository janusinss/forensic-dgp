"""Independent closure of returned zero-update diagnostic and manual V33 preparation."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_gradient_return_v33_probe_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_post_v32_learning_review_v1_milestone'
PREP = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation'
PACKET = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(mapping)


def main():
    started = time.monotonic(); m = read(OUT / 'milestone.json'); assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    count = verify(m['new_evidence_sha256']); old = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256']
    previous_count = verify(old['new_evidence_sha256'], m['previous_document_locations'])
    old_closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert old_closure['complete'] and old_closure['milestone_sha256'] == m['previous_milestone_sha256']
    old_final = verify(read(PREVIOUS / 'final_readback.json')['evidence_sha256'], m['previous_document_locations'])
    doc = m['document']; before = (ROOT / doc['before_path']).read_bytes(); after = (ROOT / doc['name']).read_bytes(); split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest() == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    returned = read(ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json')
    execution = read(ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_audit_execution/execution.json')
    imported = read(ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return_import.json')
    import_count = verify({'outputs/cctv_dgp_v32_loss_gradient_v1_return/' + k: v for k, v in imported['files_sha256'].items()})
    assert returned['complete'] and returned['diagnostic_complete'] and import_count == returned['members_verified'] == 756
    assert execution['complete'] and returned['optimizer_updates'] == returned['local_gradient_calls'] == returned['local_backward_calls'] == returned['local_optimizer_updates'] == 0
    assert returned['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py')
    assert returned['archive_sha256'] == '529acd5750c64e7320260c09f55cdd3a115c2f465afd02abc4101009155143e3' and returned['archive_bytes'] == 721569318
    assert returned['CPU_replay']['cases_at_both_states'] == 200 and returned['CPU_replay']['raw_maximum_error'] < 1e-5
    terminal = read(ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/results.json')
    assert terminal['component_gradient_calls'] == 280 and terminal['optimizer_updates'] == terminal['backwards'] == terminal['epochs'] == 0
    arrays = sum(row['values_checked'] for row in returned['cohort_gradient_analysis']); assert arrays == 301298844
    stopped = [r for r in returned['cohort_gradient_analysis'] if r['state'] == 50]
    assert len(stopped) == 2 and all(r['improvement_preservation_cosine'] < 0 and r['directional_derivatives_along_negative_original_objective'][0] > 0 for r in stopped)
    sidecars = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_sidecar_download'
    assert not read(sidecars / 'failure_record.json')['complete'] and read(sidecars / 'recovery_readback.json')['complete']
    for name in ['cctv-dgp-v32-loss-gradient-v1-results.tar.gz.sha256', 'cctv-dgp-v32-loss-gradient-v1-export.json']:
        assert (ROOT / 'outputs' / name).read_bytes() == (sidecars / name).read_bytes()
    analysis = read(ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1/analysis.json')
    arithmetic = read(ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1/independent_arithmetic_audit.json')
    assert analysis['complete'] and len(analysis['rows']) == 44 and analysis['all_nonzero_component_halfspaces_verified']
    assert arithmetic['complete'] and arithmetic['independent_primal_SLSQP_checks'] == 88 and arithmetic['analytic_boundary_fixtures'] == 5
    assert arithmetic['invalid_inputs_rejected'] == 3 and arithmetic['random_independent_primal_checks'] == 24
    assert analysis['neural_calls'] == analysis['gradient_queries'] == analysis['optimizer_updates'] == 0
    assert arithmetic['neural_calls'] == arithmetic['gradient_queries'] == arithmetic['parameter_changes'] == arithmetic['optimizer_updates'] == 0
    research = read(ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1/primary_research.json'); assert research['complete'] and research['primary_sources_only'] and len(research['sources']) == 4
    assert not research['curvature_measured'] and not research['AdamW_history_reconstructed']
    p, prepared, checked = map(read, [PACKET / 'protocol.json', PREP / 'preparation_receipt.json', PREP / 'independent_packet_audit_r1.json'])
    assert checked['complete'] and checked['protocol_sha256'] == prepared['protocol_sha256'] == sha(PACKET / 'protocol.json') == 'ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee'
    assert checked['archive_sha256'] == prepared['archive_sha256'] == '282931a3b265c834a5736ac7f70c0816e08b74a15948d4c94645679a309eeace'
    assert checked['archive_members'] == 5 and prepared['archive_bytes'] == 42130 and checked['basis_VM_files_verified'] == 414
    assert checked['unsafe_return_boundaries_rejected'] == 7 and checked['independent_primal_displacement_checks'] == 8 and checked['maximum_primal_L2_error'] < 1e-12
    assert checked['Windows_rejected_before_neural_import_or_writes'] and checked['Python310_syntax_verified']
    assert checked['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_loss_cone_probe_v33_packet_r1.py')
    assert checked['actual_optimizer_updates'] == checked['parameter_assignments'] == checked['neural_calls'] == checked['VM_launches'] == 0
    syntax = read(PREP / 'Bash_syntax_audit.json'); assert syntax['complete'] and syntax['exit_code'] == 0 and syntax['readonly_parse_only']
    assert syntax['script_sha256'] == sha(PACKET / 'scripts/run_v33_probe.sh')
    corrected = read(PREP / 'reviewer_correction_r1.json'); assert corrected['VM_packet_and_protocol_unchanged'] and corrected['protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert corrected['new_return_auditor_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r1.py')
    assert corrected['new_checker_sha256'] == checked['checker_sha256']
    assert corrected['failed_auditor_retained_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return.py')
    assert corrected['failed_checker_retained_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_loss_cone_probe_v33_packet.py')
    metrics = read(PREP / 'metric_arithmetic_audit_v1/audit.json'); assert metrics['complete'] and metrics['cases'] == 200 and metrics['maximum_component_error'] < 1e-4
    assert metrics['neural_calls'] == metrics['gradient_queries'] == metrics['parameter_assignments'] == metrics['optimizer_updates'] == 0
    assert p['optimizer_updates'] == 8 and p['committed_trajectory_updates'] == p['new_gradient_queries'] == p['backwards'] == p['epochs'] == 0
    assert p['trial_outputs'] + p['before_outputs'] == 1400 and p['CPU_replay_outputs'] == 280 and len(p['variants']) == 6
    assert p['budgets']['worker_seconds'] == 900 and p['budgets']['export_uncompressed_bytes'] == 2 * 1024**3
    assert not (PACKET / 'outputs').exists() and not (ROOT / 'outputs/cctv-dgp-loss-cone-probe-v33-results.tar.gz').exists()
    guide = (ROOT / 'CCTV_DGP_LOSS_CONE_PROBE_V33_VM.md').read_text(); lines = [line for line in guide.splitlines() if line.startswith('gcloud compute scp')]
    assert len(lines) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in lines)
    assert p['retained_capacity_gates']['early_at50_minimum'] == .01 and p['retained_capacity_gates']['final_at800_minimum'] == .1
    assert not p['loss_weights_or_normalizers_changed'] and p['retained_capacity_gates']['preservation_groups'] == 17
    closed = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
    assert not read(closed / 'outputs/early_structure_stop.json')['pass'] and read(closed / 'outputs/failure.json')['optimizer_updates'] == 50
    assert not (closed / 'outputs/results.json').exists()
    app_count = verify(read(ROOT / 'outputs/completion_mat_mirror_comparison_v1/protocol.json')['app_preservation_sha256']); assert app_count == 14
    assert not m['V33_VM_execution_started'] and m['agent_VM_launches'] == m['local_gradient_calls'] == m['local_optimizer_updates'] == 0
    assert not m['app_changes'] and not m['app_promotion'] and not m['gates_or_protection_removed'] and not m['native_or_reserved_used']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': count, 'previous_milestone_bindings_preserved': previous_count, 'previous_final_readback_bindings_preserved': old_final,
               'full_previous_handoff_preserved': True, 'returned_gradient_files_verified': import_count, 'saved_gradient_values_verified': arrays,
               'CPU_restoration_replays_verified': 200, 'independent_saved_matrix_primal_checks_preserved': 88,
               'V33_packet_files_verified': 5, 'V33_basis_files_verified': 414, 'V33_primal_displacement_checks_verified': 8,
               'V33_PNG_and_loss_arithmetic_cases_verified': 200, 'V33_readonly_Bash_syntax_verified': True,
               'V33_human_manual_execution_required': True, 'all_reported_preparation_failures_retained': True,
               'research_limits_and_user_discussion_preserved': True, 'unchanged_app_bindings_verified': app_count,
               'previous_actual_Windows_backup_receipt_preserved': old_closure['prior_actual_Windows_backup_receipt_preserved'],
               'original_gate_failure_preserved': True, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
               'agent_VM_launches': 0, 'V33_VM_execution_started': False, 'app_promotion': False, 'independent_final_review': False,
               'goal_complete': False, 'seconds': time.monotonic() - started}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8') as stream: stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
