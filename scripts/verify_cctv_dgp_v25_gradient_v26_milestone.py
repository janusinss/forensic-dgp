"""Independent history, diagnostic, finite-packet and exact manual-command readback."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_bindings(bindings, locations=None):
    for name, digest in bindings.items():
        path = (ROOT / (locations or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json')
    newcount = verify_bindings(m['new_evidence_sha256'])
    oldpath = ROOT / 'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json'
    assert sha(oldpath) == m['previous_milestone_sha256']
    old = read(oldpath)
    priorcount = verify_bindings(old['new_evidence_sha256'], m['previous697_original_locations'])
    deep_path = ROOT / 'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
    assert sha(deep_path) == old['previous_milestone_sha256']
    deep = read(deep_path)
    deepcount = verify_bindings(deep['new_evidence_sha256'], old['previous513_original_locations'])
    assert priorcount == 697 and deepcount == 513
    history = 0
    for name, before in m['previous697_original_locations'].items():
        original = (ROOT / before).read_bytes()
        current = (ROOT / name).read_bytes()
        split = original.index(b'\n') + 1
        assert current.startswith(original[:split]) and current.endswith(original[split:]), name
        prefix = current[split:len(current) - len(original[split:])].decode('utf-8')
        for token in ['V25 gradient return audited', '140 gradient queries and zero optimizer',
                      '49,537 bytes', 'all 26 matched parameter gradients', '1% early structure gate at 50',
                      'Goal active/incomplete', 'historical pilots', 'Actual V26 L4 proof/training']:
            assert token in prefix, (name, token)
        history += 1
    assert history == 10
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == m['app_record_sha256']
    app = read(apppath)
    appcount = verify_bindings({**app['sources_sha256'], **app['evidence_sha256']})
    assert appcount == 22
    audit = read(ROOT / 'outputs/cctv_dgp_v25_gradient_diagnostic_v1_independent_audit.json')
    assert audit['complete'] and audit['VM_optimizer_updates_verified'] == 0
    assert audit['initial_raw_equals_cached_baseline_cases'] == 50 and audit['VM_gradient_calls_verified'] == 140
    assert audit['fixed_state_gradient_values_verified'] == 752934 and audit['returned_files_verified'] == 10
    assert audit['identity_baseline_numerical_discrepancy_confirmed'] and not audit['gradient_trajectory_causality_proven']
    verify_bindings(audit['source_bindings_sha256'])
    v25 = read(ROOT / 'outputs/cctv_dgp_spatial_features_v25_independent_audit.json')
    assert v25['complete'] and v25['VM_failure_present'] and not v25['VM_training_result_present']
    assert not v25['early_structure_stop']['pass'] and v25['early_structure_stop']['minimum'] == .01
    prep = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_preparation'
    packet = read(prep / 'preparation.json')
    check = read(prep / 'independent_packet_source_audit.json')
    rb = read(prep / 'independent_preparation_readback.json')
    assert all(item['complete'] for item in [packet, check, rb])
    assert rb['new_tests_passed'] == 11 and rb['Bash_n_exit_code'] == 0
    verify_bindings(rb['source_bindings_sha256'])
    pin = m['V26_protocol_sha256']
    bundle = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
    assert sha(bundle / 'protocol.json') == pin == packet['protocol_sha256'] == check['protocol_sha256'] == rb['protocol_sha256']
    protocol = read(bundle / 'protocol.json')
    assert len(protocol['assets_sha256']) == 240
    for name, digest in protocol['assets_sha256'].items():
        assert sha(bundle / name) == digest, name
    old_protocol_path = ROOT / 'outputs/cctv_dgp_spatial_features_vm_v25/protocol.json'
    assert sha(old_protocol_path) == protocol['closed_V25_protocol_sha256']
    old_protocol = read(old_protocol_path)
    for key in ['cases', 'references', 'design', 'budgets', 'prospective_gates']:
        assert protocol[key] == old_protocol[key], key
    proof = protocol['identity_preflight_policy']
    assert proof['matched_identity_penalty_exactly_zero'] and proof['all26_matched_parameter_gradients_exactly_zero']
    assert proof['component_weight'] == 5 and proof['margin'] == 0 and proof['gradient_calls'] == 20
    assert proof['seconds_cap'] == 180 and proof['run_preflight_total_seconds_cap'] == 300
    assert not proof['optimizer_constructed'] and proof['legacy_discrepancy_must_reproduce']
    assert protocol['actual_VM_processing_correction_and_learning_pending'] and not protocol['goal_complete']
    archive = ROOT / 'outputs/cctv-dgp-batchmatched-identity-v26-execution.tar.gz'
    assert archive.stat().st_size == 49537 and sha(archive) == m['V26_archive_sha256'] == packet['archive_sha256']
    runbook = (ROOT / 'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md').read_text(encoding='utf-8')
    assert [int(value) for value in re.findall(r'^([1-5])\. ', runbook, re.M)] == list(range(1, 6))
    commands = re.findall(r'^gcloud compute scp (.+)$', runbook, re.M)
    assert len(commands) == 5
    for command in commands:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in command
        assert len(re.findall(r'"(janusdominic0@forensic-dgp-thesis:[^"]+)"', command)) == 1
    name = 'cctv-dgp-batchmatched-identity-v26'
    assert name + '-execution.tar.gz' in commands[0] and name + '-execution.tar.gz.sha256' in commands[1]
    for command, suffix in zip(commands[2:], ['results.tar.gz', 'results.tar.gz.sha256', 'export.json']):
        assert '/home/janusdominic0/' + name + '-' + suffix in command and command.endswith('"."')
    assert runbook.count(pin) == 4
    for token in ['tmux new-session -A -s dgp_batchmatched_identity_v26',
                  'test ! -e ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26',
                  'python3 ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26/scripts/install_v26.py',
                  '--install', '--verify-transfer', 'bash scripts/run_v26.sh ' + pin,
                  'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate']:
        assert token in runbook, token
    assert m['actual_V26_L4_proof_and_training_pending'] and m['V26_return_and_whole_face_review_pending']
    assert m['goal_status'] == 'active' and not m['goal_complete'] and not m['app_promotion']
    assert m['local_model_calls'] == m['local_gradient_calls'] == m['local_backward_calls'] == m['local_optimizer_updates'] == 0
    assert not m['VM_actions'] and not m['native_or_reserved_used']
    result = {
        'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'milestone_sha256': sha(OUT / 'milestone.json'), 'new_bindings_verified': newcount,
        'previous697_bindings_verified': priorcount, 'deeper513_bindings_verified': deepcount,
        'original_history_bodies_preserved': history, 'app22_bindings_verified': appcount,
        'concurrent_completed_maintenance_preserved': True, 'gradient_return_and_original_V25_failure_verified': True,
        'single_correction_and_original_finite_gates_retained': True, 'thin_V26_packet_verified': True,
        'five_manual_steps_and_separate_PuTTY_downloads_verified': True,
        'actual_V26_L4_proof_and_training_pending': True, 'local_model_calls': 0,
        'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'VM_actions': False, 'app_promotion': False, 'goal_status': 'active', 'goal_complete': False,
        'seconds': time.monotonic() - started,
    }
    with (OUT / 'independent_readback.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
