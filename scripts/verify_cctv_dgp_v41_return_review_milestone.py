"""Independent return/recovery/visual and full historical retention closure."""
from pathlib import Path
import hashlib
import sys
import time
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v41_return_review_milestone'


def verify(mapping, start, handoff=None, overrides=None):
    for name, digest in mapping.items():
        path = ROOT/handoff if name == 'PROJECT_HANDOFF.md' and handoff else ROOT/(overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic()-start < 600


def main():
    start = time.monotonic(); milestone = read(OUT/'milestone.json'); assert milestone['complete']
    verify(milestone['new_evidence_sha256'], start)
    before = (ROOT/milestone['previous_handoff_path']).read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at = before.index(b'\n')+1; assert hashlib.sha256(before).hexdigest() == milestone['document']['before_sha256']
    assert current[:at] == before[:at] and current[at+milestone['document']['addition_bytes']:] == before[at:]
    folders = ['cctv_dgp_v40_signal_return_v41_prepared_milestone',
        'cctv_dgp_v40_learning_signal_v1_vm_availability_milestone', 'cctv_dgp_v40_learning_signal_v1_milestone',
        'cctv_dgp_v40_return_completion_milestone', 'cctv_dgp_v39_return_v40_prepared_milestone',
        'cctv_dgp_v39_prepared_milestone_v1', 'cctv_dgp_v38_return_development_milestone_v1',
        'cctv_dgp_v37_return_v38_probe_milestone', 'cctv_dgp_v36_return_v37_diagnostic_milestone',
        'completion_conditioning_union_v1_milestone', 'cctv_dgp_v35_return_v36_probe_milestone',
        'cctv_dgp_v34_return_v35_probe_milestone', 'cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = milestone; counts = []
    for folder in folders:
        path = ROOT/'outputs'/folder; parent = read(path/'milestone.json')
        assert child['previous_milestone_sha256'] == sha(path/'milestone.json')
        verify(parent['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        closed = read(path/'independent_closure_audit.json')
        assert closed['complete'] and closed['milestone_sha256'] == sha(path/'milestone.json')
        counts.append(len(parent['new_evidence_sha256'])); child = parent
        print({'historical_folder': folder, 'bindings_checked': counts[-1]}, flush=True)
    assert counts == [7462, 14, 578, 28136, 11283, 536, 9157, 734, 2562, 539, 2230, 316, 5961]
    audit = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json')
    imported = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json')
    assert audit['complete'] and audit['failure_retained'] and not audit['training_finished800'] and not audit['necessary_capacity_pass']
    assert audit['members_verified'] == imported['members'] == 27551 and imported['returned_code_executed'] is False
    assert imported['archive_sha256'] == audit['archive_sha256'] == '01a6e9198e731d81895c0b64cde21f45ed7eb01be2bb48b4c78a9186726c94dc'
    assert imported['archive_bytes'] == 1178179150 and imported['uncompressed_bytes'] == 1204622796
    verify({('outputs/cctv_dgp_pcgrad_fit_v41_return/'+name): digest for name, digest in imported['files_sha256'].items()}, start)
    assert audit['per_update_learning_evidence']['all_logged_updates_verified'] == 50
    assert audit['per_update_learning_evidence']['saved_component_queries'] == 350
    assert audit['per_update_learning_evidence']['parameter_chain_and_snapshot_tensors_exact']
    assert [s['update'] for s in audit['snapshots']] == [0, 50] and [r['cases'] for r in audit['CPU_replays']] == [50, 50]
    assert all(r['all_states_unchanged'] for r in audit['CPU_replays'])
    assert sum(s['PNG_cases'] for s in audit['snapshots']) == sum(s['mean_only_cases'] for s in audit['snapshots']) == 7810
    gate = audit['gates'][0]; assert not gate['pass'] and gate['minimum_gain'] == .01 and len(gate['preservation_failures']) == 1
    assert gate['relative_feature_gain'] == 0.000516710942854326
    failure = gate['preservation_failures'][0]
    assert failure['group'] == 'dataset/thumbnails128x128/compound_lr24' and failure['metric'] == 'ArcFace_observed_fixed'
    recovery = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_audit_recovery_r1'
    original = ROOT/'scripts/audit_cctv_dgp_pcgrad_fit_v41_return.py'
    amended = ROOT/'scripts/audit_cctv_dgp_pcgrad_fit_v41_return_r1.py'
    plan = read(recovery/'source_recovery.json'); assert plan['inverse_source_exact'] and plan['only_export_prefix_and_receipt_name_changed']
    assert sha(original) == plan['original_auditor_sha256'] and sha(amended) == plan['recovery_auditor_sha256'] == audit['checker_sha256']
    value = amended.read_text().replace("\nEXPORT_PREFIX = 'cctv_dgp_spatial_fit_v40_return'  # Exact literal in the immutable V41 exporter.\n", '')
    for old, new in reversed(plan['changes']): value = value.replace(new, old)
    assert value == original.read_text()
    assert read(recovery/'original_prefix_test_failure.json')['original_test_expected_failure_confirmed']
    assert read(recovery/'scope_regression_pass.json')['unsafe_or_wrong_scope_members_rejected'] == 11
    first = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_audit_execution.json')
    passed = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_audit_execution_r1.json')
    assert first['exit_code'] == 1 and first['checker_unchanged'] and passed['complete'] and passed['checker_unchanged']
    worker = (ROOT/'scripts/cctv_dgp_pcgrad_fit_v41_vm.py').read_text()
    assert "arcname='cctv_dgp_spatial_fit_v40_return/'" in worker
    analysis = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/analysis.json')
    checked = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/independent_analysis_audit.json')
    visual = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/visual_review.json')
    assert checked['complete'] and checked['analysis_sha256'] == sha(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/analysis.json')
    assert checked['saved_updates'] == 50 and checked['exact_unscaled_sheet_cells'] == 500 and checked['initial_PNG_case_parity_exact'] == 3905
    assert analysis['no_proposal_loaded_into_model_or_training_checkpoint'] and analysis['original_gate_unchanged'] == gate
    assert len(analysis['direction_summary'][0]['actual_ascent_despite_projected_nonincrease']) == 9
    assert visual['complete'] and visual['all100_TRAIN_cases_and500_cells_actually_viewed'] and len(visual['pages']) == 20
    assert visual['analysis_sha256'] == sha(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/analysis.json')
    assert visual['all_visible_features_considered'] == ['eyes', 'nose', 'mouth', 'face_outline', 'visible_appearance']
    assert not visual['quality_qualification'] and not visual['app_promotion'] and not visual['input_only_labels_or_gates_changed']
    p = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41/protocol.json'); verify(p['sources_sha256'], start)
    verify({('outputs/cctv_dgp_pcgrad_fit_vm_v41/'+name): digest for name, digest in p['assets_sha256'].items()}, start)
    completion = read(ROOT/'outputs/completion_feature_fusion_off_v1/visual_review.json')
    assert not completion['app_adoption'] and not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification']
    development = read(ROOT/'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json')
    assert len(development['diagnostic_preservation_failures']) == 1
    app = read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14; verify(app, start)
    assert milestone['V41_updates'] == 50 and milestone['failed_capacity_and_preservation_retained'] and not milestone['new_training_recipe_prepared']
    assert milestone['VM_calls_here'] == milestone['local_gradient_calls'] == milestone['local_optimizer_updates'] == 0
    assert not milestone['app_promotion'] and not milestone['independent_final_review'] and not milestone['goal_complete'] and 'torch' not in sys.modules
    assert time.monotonic()-start < 600
    write(OUT/'independent_closure_audit.json', {'complete': True, 'milestone_sha256': sha(OUT/'milestone.json'),
          'checker_sha256': sha(Path(__file__)), 'new_bindings_verified': len(milestone['new_evidence_sha256']),
          'historical_bindings_preserved': counts, 'entire_previous_handoff_preserved': True,
          'human_V41_return_and_all50_steps_and350_queries_bound': True, 'all100_TRAIN_visual_cases_bound': True,
          'original_archive_prefix_failure_and_narrow_recovery_verified': True,
          'all_original_gate_split_checkpoint_and_app_bindings_retained': True, 'new_training_recipe_prepared': False,
          'VM_calls_here': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'app_promotion': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'new_bindings': len(milestone['new_evidence_sha256']), 'historical': counts,
           'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
