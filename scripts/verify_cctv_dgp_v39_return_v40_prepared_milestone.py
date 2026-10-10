"""Independent closure: audited zero-update proof, verified unrun pilot, full goal open."""
import hashlib
from pathlib import Path
import time
import sys
from cctv_dgp_spatial_fit_v40_contract import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v39_return_v40_prepared_milestone'


def verify(bindings, start, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT/handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT/(overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic()-start < 600


def main():
    start = time.monotonic(); m = read(OUT/'milestone.json'); assert m['complete']
    verify(m['new_evidence_sha256'], start)
    before = (ROOT/m['previous_handoff_path']).read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at = before.index(b'\n')+1
    assert current[:at] == before[:at] and current[at+m['document']['addition_bytes']:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    folders = ['outputs/cctv_dgp_v39_prepared_milestone_v1', 'outputs/cctv_dgp_v38_return_development_milestone_v1',
               'outputs/cctv_dgp_v37_return_v38_probe_milestone', 'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone',
               'outputs/completion_conditioning_union_v1_milestone', 'outputs/cctv_dgp_v35_return_v36_probe_milestone',
               'outputs/cctv_dgp_v34_return_v35_probe_milestone', 'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = m; historical = []
    for folder in folders:
        parent = read(ROOT/folder/'milestone.json')
        assert child['previous_milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        verify(parent['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        audit = read(ROOT/folder/'independent_closure_audit.json')
        assert audit['complete'] and audit['milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        historical.append(len(parent['new_evidence_sha256'])); child = parent
    assert historical == [536, 9157, 734, 2562, 539, 2230, 316, 5961]
    a = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json')
    r = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_return/outputs/results.json')
    assert a['complete'] and a['diagnostic_complete'] and a['members_verified'] == 273
    assert a['saved_gradient_readback']['saved_component_queries'] == 70 and not a['saved_gradient_readback']['zero_improvement_tensor_names']
    assert a['saved_initial_readback']['spatial_initial_pairs_exact'] == 50 and a['CPU_replay']['cases'] == 10
    assert r['raw_and_PNG_parity_exact_cases'] == 50 and r['all57_improvement_gradients_nonzero']
    assert r['optimizer_updates'] == r['parameter_updates'] == r['backwards'] == r['epochs'] == 0
    packet = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation/independent_packet_audit.json')
    p = read(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json')
    assert packet['complete'] and packet['TRAIN_inputs'] == 3905 and packet['references'] == 781 and packet['packet_assets'] == 5490
    assert packet['all_archive_members_checked'] == 5491 and packet['all_numeric_quality_gates_retained']
    assert packet['V39_architecture_identical_except_scope_and_name'] and packet['original_filters_and_losses_exact']
    assert packet['unsafe_return_archive_rejections'] == 9 and packet['scope_rejections'] == 3 and packet['bad_schedule_rejections'] == 2
    assert packet['actual_Windows_pre_neural_rejection'] and packet['Bash_readonly_syntax_pass'] and packet['single_remote_source_commands'] == 5
    assert p['updates'] == 800 and p['snapshots'] == [0, 50, 800] and p['decoder_parameters'] == 17952 and p['decoder_tensors'] == 57
    assert p['retained_capacity_gates'] == read(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39/protocol.json')['retained_capacity_gates']
    assert not (ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/outputs').exists() and not (ROOT/'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz').exists()
    draft = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation_draft_v1/preparation_review.json')
    assert draft['all_recipe_and_gate_and_optimizer_and_scope_logic_unchanged'] and draft['VM_calls'] == 0
    assert (ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation/Bash_sandbox_syntax_failure.json').is_file()
    paired = read(ROOT/'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json')
    native = read(ROOT/'outputs/cctv_dgp_v38_quarter_native_development_v1/visual_review.json')
    assert len(paired['diagnostic_preservation_failures']) == 1 and native['cases_reviewed'] == 24 and not native['restoration_qualified']
    completion = read(ROOT/'outputs/completion_margin_feather_v1/visual_review.json')
    assert not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification'] and not completion['app_adoption']
    app = read(ROOT/'outputs/completion_conditioning_union_v1/protocol.json')['app_preservation_sha256']; assert len(app) == 14; verify(app, start)
    assert m['V39_independent_return_pass'] and m['V40_packet_verified_and_unrun'] and not m['training_started_here']
    assert m['VM_calls_here'] == m['local_gradient_calls'] == m['local_optimizer_updates'] == 0
    assert not m['goal_complete'] and not m['app_promotion'] and not any(name in sys.modules for name in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    write(OUT/'independent_closure_audit.json', {'complete': True, 'milestone_sha256': sha(OUT/'milestone.json'), 'checker_sha256': sha(Path(__file__)),
          'new_bindings_verified': len(m['new_evidence_sha256']), 'historical_bindings_preserved': historical,
          'entire_previous_handoff_preserved': True, 'V39_saved_gradients_and_ten_frozen_replays_audited': True,
          'V39_actual_optimizer_updates': 0, 'V40_full_existing_TRAIN_packet_verified_and_unrun': True,
          'all14_app_bindings_unchanged': True, 'V38_development_failure_and_completion_defects_retained': True,
          'VM_calls': 0, 'local_neural_calls_in_closure': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
          'app_promotion': False, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(m['new_evidence_sha256']), 'historical': historical, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
