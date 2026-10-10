"""Independent closure: V39 prepared-only, history preserved, full goal unfinished."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v39_prepared_milestone_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(bindings, start, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT / handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT / (overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic() - start < 300


def main():
    start = time.monotonic(); m = read(OUT / 'milestone.json')
    assert m['complete']; verify(m['new_evidence_sha256'], start)
    before = (ROOT / m['previous_handoff_path']).read_bytes(); current = (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    at = before.index(b'\n') + 1; amount = m['document']['addition_bytes']
    assert current[:at] == before[:at] and current[at + amount:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    folders = ['outputs/cctv_dgp_v38_return_development_milestone_v1',
               'outputs/cctv_dgp_v37_return_v38_probe_milestone',
               'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone',
               'outputs/completion_conditioning_union_v1_milestone',
               'outputs/cctv_dgp_v35_return_v36_probe_milestone',
               'outputs/cctv_dgp_v34_return_v35_probe_milestone',
               'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = m; counts = []
    for folder in folders:
        old = read(ROOT / folder / 'milestone.json')
        assert child['previous_milestone_sha256'] == sha(ROOT / folder / 'milestone.json')
        verify(old['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        previous = read(ROOT / folder / 'independent_closure_audit.json')
        assert previous['complete'] and previous['milestone_sha256'] == sha(ROOT / folder / 'milestone.json')
        counts.append(len(old['new_evidence_sha256'])); child = old
    assert counts == [9157, 734, 2562, 539, 2230, 316, 5961]
    initial = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1'
    r, c = read(initial / 'results.json'), read(initial / 'independent_initial_audit.json')
    assert r['complete'] and c['complete'] and r['cases'] == 50
    assert r['exact_initial_raw_and_PNG_parity'] and c['all50_raw_pairs_exact'] and c['all100_PNG_compositions_exact']
    assert c['seed_tensors'] == r['decoder_tensors'] == 57 and c['seed_parameter_values'] == 17952
    assert c['seed_is_untrained_asset'] and not r['trained_checkpoint_created'] and not r['gradient_connectivity_established']
    assert len(c['first_reference_per_source_replay_cases']) == 10 and c['frozen_DGP_replay_forwards'] == 10
    assert r['gradient_calls'] == r['optimizer_updates'] == c['local_gradient_calls'] == c['optimizer_updates'] == 0
    assert r['partial_support_exact'] and r['invalid_inputs_rejected_before_model_calls'] == 5
    prep = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_preparation'
    packet = read(prep / 'independent_packet_audit.json')
    runner = read(prep / 'independent_packet_audit_r1_execution.json')
    p = read(ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39/protocol.json')
    assert packet['complete'] and runner['complete'] and packet['archive_files'] == 142
    assert packet['archive_bytes'] == 213203706 and packet['uncompressed_packet_bytes'] == 235212632
    assert packet['assets_verified'] == 141 and packet['local_basis_verified'] == 152
    assert packet['all_original_loss_and_filter_definition_AST_exact'] and packet['all_quality_gates_retained']
    assert packet['scope_rejections'] == 3 and packet['unsafe_archive_rejections'] == 9
    assert packet['saved_gradient_rejections'] == ['nonfinite', 'wrong_dtype', 'wrong_shape', 'preservation_nonzero', 'wrong_order', 'wrong_sum', 'zero_improvement']
    assert packet['Windows_pre_neural_rejection_passed'] and packet['Bash_readonly_syntax_verified']
    assert packet['single_remote_source_gcloud_commands_verified'] == 5 and packet['Python310_sources_checked'] == 19
    assert runner['all_scientific_and_archive_and_command_check_AST_unchanged'] and runner['packet_worker_protocol_and_gates_unchanged']
    assert runner['independent_packet_audit_sha256'] == sha(prep / 'independent_packet_audit.json')
    assert sha(ROOT / 'scripts/verify_cctv_dgp_spatial_decoder_v39_packet.py') == runner['frozen_checker_sha256'] == packet['checker_sha256']
    assert sha(ROOT / 'scripts/verify_cctv_dgp_spatial_decoder_v39_packet_r1.py') == runner['runner_sha256']
    assert (prep / 'packet_audit_sandbox_failure.json').is_file() and (prep / 'Bash_sandbox_syntax_failure.json').is_file()
    assert p['retained_capacity_gates'] == read(ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27/protocol.json')['prospective_gates']
    assert p['component_gradient_calls'] == 70 and p['optimizer_updates'] == p['parameter_updates'] == p['epochs'] == 0
    assert p['budgets']['worker_seconds'] == 600 and p['budgets']['external_seconds'] == 630
    assert not (ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39/outputs').exists()
    assert not (ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_return').exists()
    assert not (ROOT / 'outputs/cctv-dgp-spatial-decoder-v39-results.tar.gz').exists()
    paired = read(ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json')
    native = read(ROOT / 'outputs/cctv_dgp_v38_quarter_native_development_v1/visual_review.json')
    assert len(paired['diagnostic_preservation_failures']) == 1 and native['cases_reviewed'] == 24
    assert not native['restoration_qualified']
    completion = read(ROOT / 'outputs/completion_margin_feather_v1/visual_review.json')
    assert not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification'] and not completion['app_adoption']
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14; verify(app, start)
    for key in ['VM_calls_here', 'local_gradient_calls', 'local_optimizer_updates', 'V39_VM_launches']:
        assert m[key] == 0
    assert m['V38_development_failure_retained'] and m['V39_packet_prepared_and_verified'] and m['V39_prepared_only']
    assert not m['V39_gradient_connectivity_established'] and not m['app_changes'] and not m['app_adoption'] and not m['goal_complete']
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    result = {'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'), 'checker_sha256': sha(Path(__file__)),
              'new_bindings_verified': len(m['new_evidence_sha256']), 'historical_bindings_preserved': counts,
              'entire_previous_handoff_preserved': True, 'V38_development_failure_and_completion_defects_retained': True,
              'V39_all50_initial_raw_and_PNG_pairs_exact': True, 'V39_untrained_seed_and_ten_frozen_replays_verified': True,
              'V39_all142_packet_files_and_original_definitions_verified': True,
              'local_audit_environment_failures_retained_and_scientific_checks_unchanged': True,
              'V39_prepared_only_not_VM_executed': True, 'all14_app_bindings_unchanged': True,
              'VM_calls_here': 0, 'neural_calls_in_closure': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
              'app_adoption': False, 'training_capacity_pass': False, 'automatic_quality_qualification': False,
              'assisted_quality_qualification': False, 'independent_final_review': False, 'goal_complete': False,
              'seconds': time.monotonic() - start, 'cap_seconds': 300}
    assert result['seconds'] < 300
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ['complete', 'new_bindings_verified', 'historical_bindings_preserved', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
