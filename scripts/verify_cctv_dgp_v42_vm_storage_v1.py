"""Independent readback of a read-only maintenance observation; no network calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v42_vm_storage_preflight_v1'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    availability = read(OUT/'availability.json'); guest = read(OUT/'guest_inventory.json')
    instance = read(OUT/'instance_status.json')
    assert availability['complete'] and availability['guest_inventory_available']
    assert instance['name'] == 'forensic-dgp-thesis' and instance['id'] == '4410777042005672095'
    assert instance['status'] == 'RUNNING' and instance['machineType'].endswith('/g2-standard-4')
    assert instance['zone'].endswith('/us-central1-a')
    assert guest == read(OUT/'transport/guest_inventory_stdout.log')
    for label in ['instance_status', 'guest_inventory']:
        r = read(OUT/'transport'/(label+'_transport.json'))
        assert r['complete'] and r['exit_code'] == 0 and not r['observation_timeout']
        assert r['read_only'] and r['TLS_validation_enabled']
        assert r['stdout_sha256'] == sha(OUT/'transport'/(label+'_stdout.log'))
        assert r['stderr_sha256'] == sha(OUT/'transport'/(label+'_stderr.log'))
    assert availability['inspector_sha256'] == sha(ROOT/'scripts/inspect_cctv_dgp_v42_vm_storage_v1.py')
    assert availability['remote_inventory_source_sha256'] == sha(OUT/'remote_inventory_source.py')
    assert availability['transport_source_sha256'] == sha(ROOT/'scripts/storage_gcloud_20261009_v1.py')
    assert guest['required_after_install_bytes'] == 8*1024**3
    assert guest['free_after_install_requirement_met_now'] == (guest['free_bytes'] >= 8*1024**3)
    assert guest['free_GiB'] == guest['free_bytes']/1024**3
    assert guest['GPU_idle'] is False and '1324,' in guest['GPU_processes']['stdout']
    assert any('1324' in line and '--run' in line for line in guest['training_or_diagnostic_process_lines'])
    assert 'dgp_residual_epochs_v42' in guest['tmux']['stdout']
    pin = sha(ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json')
    assert guest['V42_protocol_sha256'] == pin
    assert guest['files_removed'] == 0 and guest['VM_started'] is False
    assert guest['training_or_diagnostic_launched'] is False and guest['model_or_gradient_calls'] == 0
    old = read(ROOT/'outputs/cctv_dgp_current_training_review_v1/training_history.json')
    for name, value in old['source_bindings'].items(): assert sha(ROOT/name) == value
    result = {'complete': True, 'free_bytes_at_observation': guest['free_bytes'],
        'free_GiB_at_observation': guest['free_GiB'], 'manual_V42_worker_PID_observed': 1324,
        'GPU_idle_at_observation': False, 'protocol_sha256': pin,
        'storage_requirement_met_at_observation': True, 'future_free_space_not_guaranteed': True,
        'files_removed': 0, 'active_work_untouched': True,
        'local_model_and_source_bindings_rechecked': len(old['source_bindings']),
        'inspector_sha256': availability['inspector_sha256'],
        'guest_inventory_sha256': sha(OUT/'guest_inventory.json'),
        'availability_sha256': sha(OUT/'availability.json'),
        'report_sha256': sha(ROOT/'CCTV_DGP_V42_VM_STORAGE_PREFLIGHT_V1.md'),
        'checker_sha256': sha(Path(__file__)), 'local_gradient_or_optimizer_calls': 0,
        'training_success_or_quality_implied': False, 'goal_complete': False}
    with (OUT/'independent_audit.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(result)


if __name__ == '__main__': main()
