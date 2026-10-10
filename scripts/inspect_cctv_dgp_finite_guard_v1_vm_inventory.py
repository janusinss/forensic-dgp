"""Read-only maintenance inventory with retained TLS validation and pinned host key."""
import base64
from pathlib import Path
import os
import shlex
import subprocess
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_finite_guard_v1_vm_inventory'


def main():
    assert not OUT.exists(), 'Preserve every inventory attempt'
    trust = transport.read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json')
    assert trust['complete'] and trust['current_offered_key_exact_match']
    assert trust['known_historical_hostkey'] == transport.HOSTKEY
    ca = ROOT / 'scratch/gcloud_windows_trust.pem'
    cloud = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    assert ca.is_file() and cloud.is_file()
    environment = dict(os.environ)
    environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ca)
    OUT.mkdir()
    transport.OUT = OUT
    source = transport.SOURCE.replace('cctv_dgp_actual_step_review_v1_vm', 'cctv_dgp_group_conflicts_v1_vm')
    (OUT / 'remote_inventory_source.py').write_text(source, encoding='utf-8', newline='\n')
    transport.write(OUT / 'prior_connection_failures.json', {
        'preserved_before_corrected_inventory': True,
        'sandbox_attempt': 'gcloud credential/log directories inaccessible',
        'elevated_attempt': 'default CA bundle could not verify compute.googleapis.com',
        'fix': 'Use existing Windows trust bundle in this process environment, keep TLS validation and the pinned SSH host key',
        'files_removed': 0, 'training_launched': False})
    base = ['--project=forensic-dgp-thesis', '--zone=us-central1-a']
    api = transport.invoke([str(cloud), 'compute', 'instances', 'describe', 'forensic-dgp-thesis'] + base +
        ['--format=json(name,id,status,machineType,zone)'], environment, 'api', 60)
    assert api['complete'], 'Keep failed API inventory; no VM start'
    status = transport.read(OUT / 'api_stdout.log')
    assert status['id'] == '4410777042005672095' and status['name'] == 'forensic-dgp-thesis'
    assert status['machineType'].endswith('/g2-standard-4') and status['zone'].endswith('/us-central1-a')
    transport.write(OUT / 'instance_status.json', status)
    if status['status'] != 'RUNNING':
        transport.write(OUT / 'availability.json', {'complete': True, 'status': status['status'],
            'guest_inventory_available': False, 'VM_started': False, 'files_removed': 0, 'training_launched': False})
        print({'status': status['status'], 'guest_inventory_available': False, 'VM_started': False}, flush=True)
        return
    encoded = base64.b64encode(source.encode()).decode()
    command = 'python3 -B -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')
    args = [str(cloud), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + base + [
        '--quiet', '--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + transport.HOSTKEY,
        '--command=' + command]
    assert len(subprocess.list2cmdline(args)) < 7900
    receipt = transport.invoke(args, environment, 'guest', 120)
    assert receipt['complete'], 'Keep failed guest inventory; no automatic retry'
    guest = transport.read(OUT / 'guest_stdout.log')
    assert guest['complete'] and guest['files_removed'] == 0 and not guest['diagnostic_or_training_launched']
    assert guest['gpu']['exit_code'] == guest['gpu_processes']['exit_code'] == 0
    assert 'NVIDIA L4' in guest['gpu']['stdout']
    transport.write(OUT / 'guest_inventory.json', guest)
    free = guest['disk']['free_bytes']
    transport.write(OUT / 'availability.json', {'complete': True, 'guest_inventory_available': True,
        'free_bytes': free, 'free_GiB': free / 1024**3,
        'GPU_idle_at_snapshot': not guest['gpu_processes']['stdout'].strip(),
        'snapshot_not_future_launch_guarantee': True, 'files_removed': 0, 'training_launched': False,
        'TLS_validation_enabled': True, 'hostkey_pinned': transport.HOSTKEY,
        'guest_inventory_sha256': transport.sha(OUT / 'guest_inventory.json'),
        'inspector_sha256': transport.sha(Path(__file__))})
    print({'complete': True, 'free_GiB': free / 1024**3,
        'GPU_idle': not guest['gpu_processes']['stdout'].strip(), 'files_removed': 0,
        'training_launched': False}, flush=True)


if __name__ == '__main__':
    main()
