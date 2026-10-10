"""Independently close the new milestone and retain preceding evidence and app states."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v35_return_v36_probe_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_v34_return_v35_probe_milestone'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    m, old = read(OUT / 'milestone.json'), read(PREVIOUS / 'milestone.json')
    assert m['complete'] and sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256']
    for name, digest in m['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
    before = (ROOT / m['previous_handoff_path']).read_bytes()
    current = (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    at, amount = before.index(b'\n') + 1, m['document']['addition_bytes']
    assert current[:at] == before[:at] and current[at + amount:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    for name, digest in old['new_evidence_sha256'].items():
        path = ROOT / m['previous_handoff_path'] if name == 'PROJECT_HANDOFF.md' else ROOT / name
        assert sha(path) == digest, name
    historical = ROOT / 'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone/milestone.json'
    prior = read(historical)
    assert sha(historical) == old['previous_milestone_sha256']
    for name, digest in prior['new_evidence_sha256'].items():
        path = ROOT / old['previous_handoff_path'] if name == 'PROJECT_HANDOFF.md' else ROOT / old['previous_file_overrides'].get(name, name)
        assert sha(path) == digest, name
    audit = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2.json')
    imported = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return_import.json')
    external = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_audit_run_r2/external_receipt.json')
    assert audit['complete'] and audit['finite_probe_complete'] and audit['members_verified'] == 2135
    assert external['complete'] and external['worker_exit_code'] == 0 and not external['timeout']
    assert audit['CPU_replay']['outputs'] == 100 and audit['CPU_replay']['all_states_restored']
    assert audit['local_gradient_calls'] == audit['local_optimizer_updates'] == 0
    assert imported['archive_sha256'] == audit['archive_sha256']
    for name, digest in imported['files_sha256'].items():
        assert sha(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return' / name) == digest, name
    result = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return/outputs/results.json')
    assert result['candidate_displacement_trials'] == 4 and result['raw_outputs'] == 500
    assert result['optimizer_updates'] == result['committed_trajectory_updates'] == result['new_gradient_queries'] == result['epochs'] == 0
    assert result['all_trial_states_reset'] and not result['new_checkpoint_created']
    analysis = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/analysis.json')
    visual = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/visual_review.json')
    pages = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/independent_analysis_page_audit.json')
    assert analysis['complete'] and analysis['jointly_eligible_subset_variants'] == []
    assert visual['complete'] and visual['actually_viewed_cases'] == 100 and visual['model_output_cells_reviewed'] == 500
    assert pages['complete'] and pages['all700_cells_exact'] and not visual['independent_final_review']
    failures = [f for row in analysis['variants'] for f in row['preservation_against_original']['failures']]
    assert len(failures) == 41 and sum(f['metric'] == 'ArcFace_observed_fixed' for f in failures) == 40
    assert all(row['descriptive_raw_MSE_ArcFace_failures'] for row in analysis['variants'])
    correction = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_audit_correction_r2/failure_and_correction.json')
    assert correction['complete'] and correction['unsafe_member_rejections'] == 9 and not correction['quality_gates_changed']
    failed = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_audit_run_v1/external_receipt.json')
    assert not failed['complete'] and failed['worker_exit_code'] == 1
    assert read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1/failure_preservation.json')['complete']
    bundle = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm'
    p = read(bundle / 'protocol.json')
    prep = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_preparation/independent_packet_audit.json')
    assert prep['complete'] and prep['all108_projection_rows_verified'] and prep['all65_empirical_clearances_independently_recalibrated']
    assert prep['exporter_and_prospective_checker_prefix_match'] and prep['frozen_snapshot_AST_preserved']
    assert prep['VM_launches'] == prep['neural_calls'] == prep['gradient_calls'] == prep['optimizer_updates'] == 0
    assert not (bundle / 'outputs').exists() and not (ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return').exists()
    assert p['candidate_displacement_trials'] == 4 and p['before_outputs'] + p['trial_outputs'] == 500
    assert p['retained_capacity_gates'] == read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_vm/protocol.json')['retained_capacity_gates']
    for name, digest in p['local_basis_sha256'].items():
        assert sha(ROOT / name) == digest, name
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items():
        assert sha(ROOT / name) == digest, name
    covering = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/visual_review.json')
    assert not covering['automatic_quality_qualification'] and not covering['assisted_quality_qualification'] and not covering['app_adoption']
    receipt = {'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'), 'checker_sha256': sha(Path(__file__)),
        'new_bindings_verified': len(m['new_evidence_sha256']), 'previous_bindings_preserved': len(old['new_evidence_sha256']),
        'older_bindings_preserved': len(prior['new_evidence_sha256']), 'entire_previous_handoff_preserved': True,
        'V35_members_verified': 2135, 'V35_CPU_replays': 100, 'all500_metrics_and100_case_reviews_retained': True,
        'all41_PNG_preservation_failures_retained': True, 'prefix_and_renderer_failures_retained': True,
        'V36_all108_constraints_and65_calibrations_verified': True, 'V36_packet_verified': True,
        'V36_VM_execution_started_here': False, 'all14_app_bindings_unchanged': True,
        'covering_family_failures_remain_binding': True, 'neural_or_gradient_calls_in_closure': 0,
        'optimizer_updates': 0, 'new_VM_calls': 0, 'native_or_reserved_final_used': False,
        'independent_final_quality_review': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    assert receipt['seconds'] <= 300
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ['complete', 'new_bindings_verified', 'previous_bindings_preserved', 'older_bindings_preserved', 'seconds']}))


if __name__ == '__main__':
    main()
