"""Independent history, scalar evidence, finite packet and exact-command readback."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(bindings, locations=None):
    for name, digest in bindings.items():
        path = (ROOT / (locations or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json')
    current = verify(m['new_evidence_sha256'])
    path = ROOT / 'outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json'
    assert sha(path) == m['previous_milestone_sha256']
    previous = read(path)
    previous_count = verify(previous['new_evidence_sha256'], m['previous692_original_locations'])
    priorpath = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json'
    assert sha(priorpath) == previous['previous_milestone_sha256']
    prior = read(priorpath)
    prior_count = verify(prior['new_evidence_sha256'], previous['previous299_original_locations'])
    oldpath = ROOT / 'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json'
    assert sha(oldpath) == prior['previous_milestone_sha256']
    old = read(oldpath)
    old_count = verify(old['new_evidence_sha256'], prior['previous697_original_locations'])
    deeppath = ROOT / 'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
    assert sha(deeppath) == old['previous_milestone_sha256']
    deep_count = verify(read(deeppath)['new_evidence_sha256'], old['previous513_original_locations'])
    assert (previous_count, prior_count, old_count, deep_count) == (692, 299, 697, 513)
    history = 0
    for name, backup in m['previous692_original_locations'].items():
        original, updated = (ROOT / backup).read_bytes(), (ROOT / name).read_bytes()
        split = original.index(b'\n') + 1
        assert updated.startswith(original[:split]) and updated.endswith(original[split:])
        prefix = updated[split:len(updated) - len(original[split:])].decode('utf-8')
        for token in ['corrected V26 scalar objective audited', '1.2997388024', '16,047-byte',
                      'zero optimizer updates/epochs', 'Actual L4', 'Goal active/incomplete']:
            assert token in prefix, (name, token)
        history += 1
    assert history == 4
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == m['app_record_sha256']
    app = read(apppath)
    app_count = verify({**app['sources_sha256'], **app['evidence_sha256']})
    assert app_count == 22
    lossfolder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1'
    loss, losscheck = read(lossfolder / 'results.json'), read(lossfolder / 'independent_saved_loss_audit.json')
    assert loss['complete'] and losscheck['complete']
    assert losscheck['results_sha256'] == sha(lossfolder / 'results.json')
    assert losscheck['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_batchmatched_identity_v26_loss_v1.py')
    assert loss['runner_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_batchmatched_identity_v26_loss_v1.py')
    assert losscheck['case_states_verified'] == 100 and losscheck['group_states_verified'] == 34
    assert losscheck['maximum_term_assembly_difference'] == 0
    verify(loss['source_bindings_sha256'])
    for name, digest in loss['artifacts_sha256'].items():
        assert sha(lossfolder / name) == digest
    assert len(loss['artifacts_sha256']) == 30 and loss['counts']['recognizer_forwards'] == 30
    assert loss['groups']['all']['states']['0']['objective'] == m['corrected_loss_start']
    assert loss['groups']['all']['states']['50']['objective'] == m['corrected_loss_stopped50']
    assert m['active_penalty_cases_stopped50'] == {'pixel_regression': 0, 'SSIM_regression': 2, 'ArcFace_regression': 1}
    assert m['corrected_loss_start'] > m['corrected_loss_stopped50']
    audit = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json')
    assert audit['complete'] and audit['VM_failure_present'] and not audit['VM_training_result_present']
    assert not audit['early_structure_stop']['pass'] and audit['early_structure_stop']['minimum'] == .01
    prep = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_preparation'
    packet, checked = read(prep / 'preparation.json'), read(prep / 'independent_packet_and_return_audit.json')
    assert packet['complete'] and checked['complete']
    assert packet['protocol_sha256'] == checked['protocol_sha256'] == m['diagnostic_protocol_sha256']
    assert checked['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_v26_gradient_diagnostic_v1.py')
    archive = ROOT / 'outputs/cctv-dgp-v26-gradient-diagnostic-v1-execution.tar.gz'
    assert sha(archive) == m['diagnostic_archive_sha256'] == packet['archive_sha256']
    assert archive.stat().st_size == m['diagnostic_archive_bytes'] == 16047
    protocol = read(ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm/protocol.json')
    assert protocol['head_batches'] == 20 and protocol['component_gradient_calls'] == 140
    assert protocol['recognizer_forwards'] == 70 and protocol['optimizer_updates'] == 0
    assert protocol['snapshots'] == [0, 50] and not protocol['new_training_recipe']
    assert protocol['initial_corrected_identity_exact_zero_each_batch_and_all26_gradients']
    assert protocol['timing_limits'] == {'worker_seconds': 420, 'external_seconds': 480, 'external_kill_grace_seconds': 30,
        'export_seconds': 30, 'external_export_seconds': 60, 'export_kill_grace_seconds': 10,
        'free_disk_bytes': 1024 ** 3, 'allocated_VRAM_bytes': 20 * 1024 ** 3}
    verify(protocol['local_basis_sha256'])
    runbook = (ROOT / 'CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ', runbook, flags=re.MULTILINE) == ['1', '2', '3', '4', '5']
    command_lines = [line for line in runbook.splitlines() if line.startswith('gcloud compute scp ')]
    prefix = 'gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a '
    expected = [prefix + '"' + name + '" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"' for name in [
        'cctv-dgp-v26-gradient-diagnostic-v1-execution.tar.gz', 'cctv-dgp-v26-gradient-diagnostic-v1-execution.tar.gz.sha256']]
    expected += [prefix + '"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name + '" "."' for name in [
        'cctv-dgp-v26-gradient-diagnostic-v1-results.tar.gz', 'cctv-dgp-v26-gradient-diagnostic-v1-results.tar.gz.sha256',
        'cctv-dgp-v26-gradient-diagnostic-v1-export.json']]
    assert command_lines == expected
    for token in ['cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"',
                  'tmux new-session -A -s dgp_v26_gradient_diagnostic_v1',
                  '--protocol-sha ' + packet['protocol_sha256'] + ' --verify-transfer',
                  'bash scripts/run_gradient.sh ' + packet['protocol_sha256'],
                  'test ! -e ~/forensic-dgp/cctv_dgp_v26_gradient_diagnostic_v1_vm',
                  'test -d ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26/outputs/frozen_DGP_features']:
        assert token in runbook, token
    assert m['new_tests_passed'] == 11 and m['Bash_n_exit_code'] == 0
    assert m['local_gradient_calls'] == m['local_backward_calls'] == m['local_optimizer_updates'] == 0
    assert m['actual_L4_diagnostic_pending'] and m['independent_return_audit_pending']
    assert not m['new_training_recipe'] and not m['corrected_optimizer_trajectory_cause_proven']
    assert not m['VM_actions'] and not m['native_or_reserved_used'] and not m['app_promotion']
    assert m['goal_status'] == 'active' and not m['goal_complete']
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'milestone_sha256': sha(OUT / 'milestone.json'), 'new_bindings_verified': current,
        'previous692_bindings_verified': previous_count, 'deeper299_bindings_verified': prior_count,
        'deeper697_bindings_verified': old_count, 'deeper513_bindings_verified': deep_count,
        'full_historical_bodies_preserved': history, 'app22_bindings_verified': app_count,
        'concurrent_completed_maintenance_preserved': True, 'corrected_scalar_evidence_and_original_failure_verified': True,
        'finite_zero_update_packet_and_exact_five_manual_steps_verified': True,
        'separate_PuTTY_download_calls_verified': True, 'new_tests_passed': 11,
        'actual_L4_diagnostic_pending': True, 'neural_calls': 0, 'local_gradient_calls': 0,
        'optimizer_updates': 0, 'VM_actions': False, 'app_promotion': False,
        'goal_status': 'active', 'goal_complete': False, 'seconds': time.monotonic() - started}
    with (OUT / 'independent_readback.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
