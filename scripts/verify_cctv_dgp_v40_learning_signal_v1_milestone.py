"""Independently close prepared diagnostic and all retained milestone history."""
from pathlib import Path
import hashlib
import sys
import time
from completion_feature_fusion_off_v1_common import ROOT, sha, read, write

OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_milestone'


def verify(bindings, start, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT/handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT/(overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic()-start < 600


def main():
    start = time.monotonic(); m = read(OUT/'milestone.json'); assert m['complete']; verify(m['new_evidence_sha256'], start)
    before = (ROOT/m['previous_handoff_path']).read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); at = before.index(b'\n')+1
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    assert current[:at] == before[:at] and current[at+m['document']['addition_bytes']:] == before[at:]
    folders = ['outputs/cctv_dgp_v40_return_completion_milestone', 'outputs/cctv_dgp_v39_return_v40_prepared_milestone',
        'outputs/cctv_dgp_v39_prepared_milestone_v1', 'outputs/cctv_dgp_v38_return_development_milestone_v1',
        'outputs/cctv_dgp_v37_return_v38_probe_milestone', 'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone',
        'outputs/completion_conditioning_union_v1_milestone', 'outputs/cctv_dgp_v35_return_v36_probe_milestone',
        'outputs/cctv_dgp_v34_return_v35_probe_milestone', 'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = m; counts = []
    for folder in folders:
        parent = read(ROOT/folder/'milestone.json'); assert child['previous_milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        verify(parent['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        checked = read(ROOT/folder/'independent_closure_audit.json'); assert checked['complete'] and checked['milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        counts.append(len(parent['new_evidence_sha256'])); child = parent
    assert counts == [28136, 11283, 536, 9157, 734, 2562, 539, 2230, 316, 5961]
    analysis = read(ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis/results.json')
    checked = read(ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis/independent_analysis_audit.json')
    assert analysis['complete'] and analysis['tensors_changed'] == 57 and analysis['parameter_values'] == 17952 and analysis['stop_and_snapshot50_tensors_exact']
    assert checked['complete'] and checked['results_sha256'] == sha(ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis/results.json')
    assert checked['all57_stop_snapshot_tensors_and17952_delta_values_verified'] and checked['not_an_optimizer_trajectory_reconstruction']
    p = read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json')
    prepared = read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_preparation/prepared.json')
    packet = read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_preparation/independent_packet_audit.json')
    assert packet['complete'] and prepared['complete'] and packet['protocol_sha256'] == prepared['protocol_sha256'] == sha(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json')
    assert packet['V40_decoder_inverse_source_and_AST_exact'] and packet['V40_losses_filters_normalizers_and_gates_exact']
    assert packet['100_TRAIN_cases20_references_metadata_selection_exact'] and packet['Windows_pre_neural_rejection_passed']
    assert packet['scope_rejections'] == 3 and packet['unsafe_archive_rejections'] == len(packet['saved_gradient_rejections']) == 9
    assert packet['single_remote_source_gcloud_commands_verified'] == 5 and not packet['diagnostic_VM_execution_started']
    assert p['component_gradient_calls'] == 280 and p['optimizer_updates'] == p['parameter_updates'] == p['backwards'] == p['epochs'] == 0
    assert p['retained_capacity_gates'] == read(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json')['retained_capacity_gates']
    assert not (ROOT/'outputs/cctv-dgp-v40-learning-signal-v1-results.tar.gz').exists()
    assert not (ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return').exists() and not (ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/outputs').exists()
    original = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json'); assert original['failure_retained'] and not original['necessary_capacity_pass']
    assert not original['training_finished800'] and original['gates'][0]['minimum_gain'] == .01 and not original['gates'][0]['pass']
    app = read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256']; assert len(app) == 14; verify(app, start)
    completion = read(ROOT/'outputs/completion_feature_fusion_off_v1/visual_review.json')
    assert not completion['app_adoption'] and not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification']
    development = read(ROOT/'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json'); assert len(development['diagnostic_preservation_failures']) == 1
    assert m['diagnostic_prepared_not_run'] and m['V40_failure_preserved'] and m['DGP_primary_app_bindings_unchanged']
    assert m['VM_calls'] == m['local_neural_calls_here'] == m['local_gradient_calls'] == m['local_optimizer_updates'] == 0
    assert not m['training_started_here'] and not m['app_promotion'] and not m['goal_complete']
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'pretrained_completion'])
    assert time.monotonic()-start < 600
    write(OUT/'independent_closure_audit.json', {'complete': True, 'milestone_sha256': sha(OUT/'milestone.json'), 'checker_sha256': sha(Path(__file__)),
        'new_bindings_verified': len(m['new_evidence_sha256']), 'historical_bindings_preserved': counts, 'entire_previous_handoff_preserved': True,
        'saved57_tensor17952_value_analysis_bound': True, 'prepared_only_diagnostic_and_all_rejections_bound': True,
        'V40_failed_gate_and_original_stopped_checkpoints_retained': True, 'V38_development_failure_and_completion_rejection_preserved': True,
        'all14_app_bindings_unchanged': True, 'VM_calls': 0, 'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
        'app_promotion': False, 'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(m['new_evidence_sha256']), 'historical': counts, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
