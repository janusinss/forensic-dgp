"""Independent complete milestone/readback, exact commands and history checks."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone'
PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(bindings, mapping=None):
    for name, digest in bindings.items():
        path = (ROOT / (mapping or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    started = time.monotonic(); m = read(OUT / 'milestone.json')
    assert not (OUT / 'independent_readback.json').exists()
    current = verify(m['new_evidence_sha256'])
    path = ROOT / 'outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json'
    assert sha(path) == m['previous_milestone_sha256'] == 'fab3256737334bed2eeb984df4d6deb545654e05dada1d127bbe8716b4382103'
    previous = read(path); counts = [verify(previous['new_evidence_sha256'], m['previous66_original_locations'])]
    for name, key in [
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json', 'previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json', 'previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json', 'previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json', 'previous513_original_locations')]:
        path = ROOT / name; assert sha(path) == previous['previous_milestone_sha256']
        old = read(path); counts.append(verify(old['new_evidence_sha256'], previous[key])); previous = old
    assert counts == m['preserved_history_binding_counts'] == [66, 692, 299, 697, 513]
    for name, backup in m['previous66_original_locations'].items():
        original = (ROOT / backup).read_bytes(); updated = (ROOT / name).read_bytes()
        split = original.index(b'\n') + 1
        assert updated.startswith(original[:split]) and updated.endswith(original[split:])
        prefix = updated[split:len(updated) - len(original[split:])].decode('utf-8')
        for text in ['corrected V26 gradient return audited', '98.4251552%', '51,049-byte',
            'zero local derivatives/backwards/optimizer updates', 'Actual L4 proof', 'Goal active/incomplete']:
            assert text in prefix, (name, text)
    assert len(m['previous66_original_locations']) == 7
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'; assert sha(app_path) == m['app_record_sha256']
    app = read(app_path); assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    returned = read(ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json')
    assert returned['complete'] and returned['corrected_initial_identity_exact_zero_verified']
    assert returned['VM_gradient_calls_verified'] == 140 and returned['VM_optimizer_updates_verified'] == 0
    assert returned['original_V26_quality_failure_retained'] and not returned['gradient_trajectory_causality_proven']
    assert returned['archive_bytes'] == 1665230 and returned['fixed_state_gradient_values_verified'] == 752934
    verify(returned['source_bindings_sha256'])
    analysis_path = ROOT / 'outputs/cctv_dgp_v26_gradient_paths_v1/analysis.json'
    analysis = read(analysis_path); arithmetic = read(analysis_path.with_name('independent_readback.json'))
    assert arithmetic['complete'] and arithmetic['analysis_sha256'] == sha(analysis_path)
    verify(analysis['source_bindings_sha256'])
    assert abs(analysis['snapshots'][1]['paths']['direct']['component_squared_gradient_fraction'][0] - .9842515515403157) < 1e-12
    assert abs(analysis['snapshots'][1]['preservation_to_reward_norm_ratio'] - .3701499123801087) < 1e-12
    prep = ROOT / 'outputs/cctv_dgp_feature_skips_v27_preparation'
    packet = read(prep / 'preparation.json')
    assert packet['protocol_sha256'] == PIN == m['new_protocol_sha256']
    archive = ROOT / 'outputs/cctv-dgp-feature-skips-v27-execution.tar.gz'
    assert sha(archive) == m['new_archive_sha256'] == packet['archive_sha256']
    assert archive.stat().st_size == m['new_archive_bytes'] == packet['archive_bytes'] == 51049
    expected_checkers = {
        'independent_packet_audit.json': 'scripts/verify_cctv_dgp_feature_skips_v27.py',
        'independent_interface_readback.json': 'scripts/verify_cctv_dgp_feature_skips_v27_interface.py',
        'windows_execution_guard.json': 'scripts/verify_cctv_dgp_feature_skips_v27_host_guard.py',
        'independent_return_audit_source_check.json': 'scripts/verify_cctv_dgp_feature_skips_v27_return_audit.py'}
    for name, checker in expected_checkers.items():
        r = read(prep / name)
        assert r['complete'] and r['checker_sha256'] == sha(ROOT / checker)
    interface = read(prep / 'interface_inference_audit.json')
    assert len(interface['initial_cases']) == 50 and len(interface['probes']) == 5 and interface['head_forwards'] == 61
    assert all(row['raw_initial_exact_baseline'] and row['PNG_initial_exact_baseline'] for row in interface['initial_cases'])
    assert interface['local_gradient_calls'] == interface['local_backward_calls'] == interface['local_optimizer_updates'] == 0
    assert interface['no_checkpoint_or_probe_images_created'] and interface['preset_coefficients_use_no_target_no_fit_no_derivative_estimate']
    verify(interface['source_bindings_sha256'])
    bundle = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'; p = read(bundle / 'protocol.json')
    assert sha(bundle / 'protocol.json') == PIN and p['design']['trainable_parameters'] == 55524
    assert p['design']['direct_feature_skip_parameters'] == 1743
    verify({(bundle / name).relative_to(ROOT).as_posix(): digest for name, digest in p['assets_sha256'].items()})
    verify(p['local_basis_sha256'])
    return_preparation = read(prep / 'return_audit_preparation.json')
    verify({**return_preparation['basis_sha256'], **return_preparation['generated_source_sha256']})
    test = read(prep / 'test_receipt.json')
    assert test['complete'] and test['exit_code'] == 0 and test['tests_passed'] == m['new_tests_passed'] == 14
    assert test['tests_sha256'] == sha(ROOT / 'tests/test_cctv_dgp_feature_skips_v27.py')
    assert test['synthetic_receipts_are_not_pilot_results'] and test['local_model_or_gradient_calls'] == 0
    runbook = (ROOT / 'CCTV_DGP_FEATURE_SKIPS_V27_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ', runbook, flags=re.MULTILINE) == ['1', '2', '3', '4', '5']
    lines = [line for line in runbook.splitlines() if line.startswith('gcloud compute scp ')]
    prefix = 'gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a '
    expected = [prefix + '"' + name + '" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"' for name in [
        'cctv-dgp-feature-skips-v27-execution.tar.gz', 'cctv-dgp-feature-skips-v27-execution.tar.gz.sha256']]
    expected += [prefix + '"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name + '" "."' for name in [
        'cctv-dgp-feature-skips-v27-results.tar.gz', 'cctv-dgp-feature-skips-v27-results.tar.gz.sha256',
        'cctv-dgp-feature-skips-v27-export.json']]
    assert lines == expected
    for token in ['cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"',
        'tmux new-session -A -s dgp_feature_skips_v27',
        'test ! -e ~/forensic-dgp/cctv_dgp_feature_skips_vm_v27',
        'python3 ~/forensic-dgp/cctv_dgp_feature_skips_vm_v27/scripts/install_v27.py',
        '--protocol-sha ' + PIN + ' --install', '--protocol-sha ' + PIN + ' --verify-transfer',
        'bash scripts/run_v27.sh ' + PIN]:
        assert token in runbook, token
    assert ' --preflight' not in runbook, 'No extra neural preflight run may be recommended'
    assert m['local_gradient_calls'] == m['local_backward_calls'] == m['local_optimizer_updates'] == 0
    assert m['actual_V27_L4_pilot_pending'] and m['prospective_full_return_audit_verified']
    assert m['human_manual_VM_execution_required'] and not m['VM_actions'] and not m['app_promotion']
    assert not m['native_or_reserved_or_new_covering_pixels_used'] and not m['entire_optimizer_trajectory_cause_proven']
    assert m['goal_status'] == 'active' and not m['goal_complete'] and not m['independent_final_review_complete']
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'milestone_sha256': sha(OUT / 'milestone.json'), 'new_bindings_verified': current,
        'previous66_deeper692_299_697_513_bindings_verified': counts,
        'full_historical_document_bodies_preserved': 7, 'app22_bindings_verified': 22,
        'completed_concurrent_VM_maintenance_preserved': True,
        'actual_corrected_gradient_return_and_failed_V26_quality_separated': True,
        'single_feature_path_packet_exact_five_manual_steps_and_separate_PuTTY_downloads_verified': True,
        'all_original_scientific_and_finite_limits_retained': True,
        'actual_V27_L4_pilot_pending': True, 'new_tests_passed': 14,
        'local_neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'app_promotion': False, 'goal_status': 'active', 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with (OUT / 'independent_readback.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
