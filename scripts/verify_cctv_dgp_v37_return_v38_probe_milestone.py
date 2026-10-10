"""Independent handoff closure preserving failure history and unopened final scope."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v37_return_v38_probe_milestone'


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
    paths = ['outputs/cctv_dgp_v36_return_v37_diagnostic_milestone', 'outputs/completion_conditioning_union_v1_milestone',
             'outputs/cctv_dgp_v35_return_v36_probe_milestone', 'outputs/cctv_dgp_v34_return_v35_probe_milestone',
             'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    descendant = m; counts = []
    for folder in paths:
        old = read(ROOT / folder / 'milestone.json')
        assert descendant['previous_milestone_sha256'] == sha(ROOT / folder / 'milestone.json')
        verify_bindings(old['new_evidence_sha256'], started, descendant['previous_handoff_path'], descendant.get('previous_file_overrides'))
        assert read(ROOT / folder / 'independent_closure_audit.json')['complete']
        counts.append(len(old['new_evidence_sha256'])); descendant = old
    assert counts == [2562, 539, 2230, 316, 5961]
    imported = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return_import.json')
    audit = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json')
    outer = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_audit_run_v1/external_receipt.json')
    result = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return/outputs/results.json')
    analysis = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/analysis.json')
    checked = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/independent_analysis_audit.json')
    assert imported['complete'] and imported['members'] == audit['members_verified'] == 431 and not imported['returned_code_executed']
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['failure_retained']
    assert outer['complete'] and outer['worker_exit_code'] == 0 and not outer['timeout']
    assert result['complete'] and result['gradient_queries'] == 300 and result['optimizer_updates'] == result['parameter_updates'] == result['epochs'] == 0
    assert not result['new_checkpoint_created'] and not result['goal_complete']
    assert checked['complete'] and analysis['all100_original_raw_and_PNG_files_byte_exact']
    assert checked['V36_failures_flagged_by_coarse_predictions'] == analysis['V36_failures_flagged_by_new_coarse_prediction'] == 0
    assert len(analysis['V36_actual_PNG_failures']) == 11
    math = read(ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1/analysis.json')
    geometry = read(ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1/independent_geometry_audit.json')
    packet = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_preparation/independent_packet_audit.json')
    p = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_vm/protocol.json')
    assert math['complete'] and geometry['complete'] and math['geometry_qualified_for_disposable_probe']
    assert geometry['all210_rows_and408_calibration_observations_verified'] and geometry['all65_V36_clearances_unchanged']
    assert not geometry['finite_preservation_implied'] and geometry['new97_PNG_margins_recalibrated']
    assert packet['complete'] and packet['archive_files'] == 9 and packet['all210_projection_rows_verified']
    assert packet['direction_differs_from_failed_V36'] and packet['all_quality_gates_retained'] and packet['frozen_snapshot_and_replay_AST_preserved']
    assert packet['VM_calls'] == packet['neural_calls'] == packet['gradient_calls'] == packet['optimizer_updates'] == 0
    assert not (ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return').exists()
    assert not (ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_vm/outputs').exists()
    oldp = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json')
    assert p['retained_capacity_gates'] == oldp['retained_capacity_gates'] and p['cohorts'] == oldp['cohorts']
    assert p['constraint_labels'][:108] == oldp['constraint_labels'] and p['clearance_targets'][:108] == oldp['clearance_targets']
    assert p['new_gradient_queries'] == p['optimizer_updates'] == p['committed_trajectory_updates'] == 0
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14; verify_bindings(app, started)
    completion = read(ROOT / 'outputs/completion_conditioning_union_v1/visual_review.json')
    assert not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification'] and not completion['app_adoption']
    for key in ['VM_calls_here', 'local_gradient_calls', 'local_optimizer_updates', 'V38_VM_launches']: assert m[key] == 0
    assert m['V37_diagnostic_independently_audited'] and m['V36_11_PNG_failures_retained'] and m['V38_prepared_only']
    assert not m['app_changes'] and not m['app_adoption'] and not m['goal_complete']
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    receipt = {'complete':True, 'milestone_sha256':sha(OUT / 'milestone.json'), 'checker_sha256':sha(Path(__file__)),
               'new_bindings_verified':len(m['new_evidence_sha256']), 'historical_bindings_preserved':counts,
               'entire_previous_handoff_preserved':True, 'V37_all431_files_and300_saved_gradients_audited':True,
               'V37_no_training_updates_or_quality_claim':True, 'V36_11_failures_and0_coarse_predictions_retained':True,
               'all210_rows_and408_observations_verified':True, 'V38_packet_verified_but_unrun':True,
               'all14_app_bindings_unchanged':True, 'completion_automatic_assisted_failures_retained':True,
               'VM_calls_here':0, 'neural_calls_in_closure':0, 'local_gradient_calls':0, 'local_optimizer_updates':0,
               'independent_final_review':False, 'automatic_quality_qualification':False, 'assisted_quality_qualification':False,
               'app_adoption':False, 'goal_complete':False, 'seconds':time.monotonic() - started, 'cap_seconds':300}
    assert receipt['seconds'] < 300
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key:receipt[key] for key in ['complete', 'new_bindings_verified', 'historical_bindings_preserved', 'seconds']}), flush=True)


if __name__ == '__main__': main()
