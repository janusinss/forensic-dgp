"""Independent milestone closure with archived document and failure lineage."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_bindings(bindings, started, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT / handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT / (overrides or {}).get(name, name)
        assert sha(path) == digest, name
        assert time.monotonic() - started < 300


def main():
    started = time.monotonic(); m = read(OUT / 'milestone.json')
    verify_bindings(m['new_evidence_sha256'], started)
    before = (ROOT / m['previous_handoff_path']).read_bytes(); current = (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    at, amount = before.index(b'\n') + 1, m['document']['addition_bytes']
    assert current[:at] == before[:at] and current[at + amount:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    paths = ['outputs/completion_conditioning_union_v1_milestone', 'outputs/cctv_dgp_v35_return_v36_probe_milestone',
             'outputs/cctv_dgp_v34_return_v35_probe_milestone', 'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    descendant = m; counts = []
    for folder in paths:
        old = read(ROOT / folder / 'milestone.json')
        assert descendant['previous_milestone_sha256'] == sha(ROOT / folder / 'milestone.json')
        verify_bindings(old['new_evidence_sha256'], started, descendant['previous_handoff_path'], descendant.get('previous_file_overrides'))
        assert read(ROOT / folder / 'independent_closure_audit.json')['complete']
        counts.append(len(old['new_evidence_sha256'])); descendant = old
    assert counts == [539, 2230, 316, 5961]
    print(json.dumps({'historical_bindings_preserved': counts, 'seconds': time.monotonic() - started}), flush=True)
    imported = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return_import.json')
    audit = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json')
    outer = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_audit_run_v1/external_receipt.json')
    analysis = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json')
    visual = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/visual_review.json')
    pages = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/independent_analysis_page_audit.json')
    result = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return/outputs/results.json')
    assert imported['complete'] and imported['members'] == audit['members_verified'] == 2135
    assert not imported['returned_code_executed'] and audit['complete'] and audit['finite_probe_complete']
    assert outer['complete'] and outer['worker_exit_code'] == 0 and not outer['timeout']
    assert audit['CPU_replay']['outputs'] == 100 and audit['CPU_replay']['all_states_restored']
    assert len(audit['finite_output_arithmetic']) == 10 and all(r['cases'] == 50 and r['groups'] == 17 for r in audit['finite_output_arithmetic'])
    assert result['raw_outputs'] == 500 and result['candidate_displacement_trials'] == 4
    assert result['optimizer_updates'] == result['new_gradient_queries'] == result['committed_trajectory_updates'] == 0
    assert result['all_trial_states_reset'] and not result['new_checkpoint_created']
    assert pages['complete'] and pages['all700_cells_exact'] and pages['pages_verified'] == 20
    assert pages['V36_PNG_failures'] == 11 and pages['V35_PNG_failures'] == 41
    assert pages['all100_V35_original_PNG_and_raw_files_exact'] and not analysis['jointly_eligible_subset_variants']
    assert visual['actually_viewed_cases'] == 100 and visual['model_output_cells_reviewed'] == 500
    assert not visual['independent_final_review'] and not visual['app_adoption'] and not visual['all_scales_visible_structure_qualified']
    path = ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1'
    forward, forward_audit = read(path / 'analysis.json'), read(path / 'independent_forward_audit.json')
    assert forward['complete'] and forward_audit['complete'] and forward['all500_canonical_PNG_tensors_exact']
    assert forward_audit['PNG_only_failures'] == forward['PNG_failures_without_same_group_raw_failure'] == 9
    assert forward['PNG_failures_with_same_group_raw_failure'] == 2 and len(forward_audit['forward_fixtures']) == 4
    assert forward['maximum_MSE_forward_error'] <= 1e-12 and forward['maximum_SSIM_forward_error'] <= 1e-7
    assert not forward_audit['coarse_derivative_tested'] and not forward['coarse_derivative_tested']
    v37 = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_vm/protocol.json')
    packet = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_preparation/independent_packet_audit_r1.json')
    assert packet['complete'] and packet['archive_members_verified'] == 4 and packet['V36_original_bindings_verified'] == 412
    assert packet['VM_launches'] == packet['actual_gradient_queries'] == packet['actual_neural_calls'] == packet['actual_optimizer_updates'] == 0
    assert packet['original_checker_sandbox_failure_retained'] and not packet['packet_changed_for_repair']
    assert not (ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return').exists()
    assert not (ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_vm/outputs').exists()
    prior = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json')
    assert v37['retained_capacity_gates'] == prior['retained_capacity_gates'] and v37['cohorts'] == prior['cohorts']
    assert v37['gradient_queries'] == 300 and v37['optimizer_updates'] == v37['parameter_updates'] == 0
    assert not v37['true_PNG_derivative_claimed'] and not v37['VM_execution_started'] and not v37['goal_complete']
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14; verify_bindings(app, started)
    completion = read(ROOT / 'outputs/completion_conditioning_union_v1/visual_review.json')
    assert not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification'] and not completion['app_adoption']
    for key in ['VM_calls_here', 'local_gradient_calls', 'local_optimizer_updates', 'V37_VM_launches']: assert m[key] == 0
    assert not m['app_adoption'] and not m['app_changes'] and not m['goal_complete']
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    receipt = {'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'), 'checker_sha256': sha(Path(__file__)),
               'new_bindings_verified': len(m['new_evidence_sha256']), 'historical_bindings_preserved': counts,
               'entire_previous_handoff_preserved': True, 'V36_members_verified': 2135, 'all500_outputs_and20_pages_reviewed': True,
               'all700_cells_independently_verified': True, 'V36_11_PNG_failures_retained': True,
               'all100_original_V35_V36_outputs_byte_exact': True, 'all500_PNG_forward_values_exact': True,
               'all11_failure_classifications_and4_fixtures_verified': True, 'V37_packet_verified_but_unrun': True,
               'V37_zero_update_coarse_diagnostic_prepared': True, 'original_checker_sandbox_failure_retained': True,
               'all14_app_bindings_unchanged': True, 'completion_failures_and_automatic_assisted_status_retained': True,
               'VM_calls_here': 0, 'neural_calls_in_closure': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
               'independent_final_review': False, 'automatic_quality_qualification': False,
               'assisted_quality_qualification': False, 'app_adoption': False, 'goal_complete': False,
               'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert receipt['seconds'] < 300
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ['complete', 'new_bindings_verified', 'historical_bindings_preserved', 'seconds']}), flush=True)


if __name__ == '__main__': main()
