"""Fresh pinned maintenance inventory; never start, train, or delete."""
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1/inventory_initial'
HOSTKEY = original.HOSTKEY


def main():
    assert not OUT.exists(), 'Retain all inventory attempts'
    trust = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    recorded = original.read(trust)
    assert recorded['complete'] and recorded['known_historical_hostkey'] == HOSTKEY
    assert recorded['current_offered_key_exact_match']
    ca = ROOT / 'scratch/gcloud_windows_trust.pem'
    cloud = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    assert ca.is_file() and cloud.is_file()
    OUT.mkdir(parents=True)
    original.OUT = OUT
    source = original.SOURCE.replace(
        "disk=shutil.disk_usage(root);pilot=", "tail=root/'cctv_dgp_actual_step_tail_v1_vm'\ndisk=shutil.disk_usage(root);pilot=")
    source = source.replace("'current_diagnostic_metadata':current,", "'current_diagnostic_metadata':current,'tail_root_present':tail.is_dir(),'tail_protocol_sha256':hashlib.sha256((tail/'protocol.json').read_bytes()).hexdigest() if (tail/'protocol.json').is_file() else None,")
    (OUT / 'remote_inventory_source.py').write_text(source, encoding='utf-8', newline='\n')
    environment = dict(os.environ)
    environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ca)
    started = time.monotonic()
    base = ['--project=forensic-dgp-thesis', '--zone=us-central1-a']
    args = [str(cloud), 'compute', 'instances', 'describe', 'forensic-dgp-thesis'] + base + [
        '--format=json(name,id,status,machineType,zone,disks.deviceName,disks.source)']
    print({'API_inventory_started': True, 'cap_seconds': 60}, flush=True)
    api = original.invoke(args, environment, 'api', 60)
    assert api['complete'], 'Read the preserved API transport receipt before any retry'
    instance = original.read(OUT / 'api_stdout.log')
    assert instance['name'] == 'forensic-dgp-thesis' and instance['id'] == '4410777042005672095'
    assert instance['machineType'].endswith('/g2-standard-4') and instance['zone'].endswith('/us-central1-a')
    original.write(OUT / 'instance_status.json', instance)
    assert instance['status'] == 'RUNNING', 'Do not start a stopped VM for maintenance'
    encoded = base64.b64encode(source.encode()).decode()
    remote = 'python3 -B -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')
    args = [str(cloud), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + base + [
        '--quiet', '--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + HOSTKEY, '--command=' + remote]
    assert len(subprocess.list2cmdline(args)) < 8000
    print({'guest_inventory_started': True, 'cap_seconds': 120}, flush=True)
    transport = original.invoke(args, environment, 'guest', 120)
    assert transport['complete'], 'Read the preserved guest transport receipt before any retry'
    guest = original.read(OUT / 'guest_stdout.log')
    assert guest['complete'] and guest['gpu']['exit_code'] == guest['gpu_processes']['exit_code'] == 0
    assert 'NVIDIA L4' in guest['gpu']['stdout'] and guest['files_removed'] == 0
    assert not guest['diagnostic_or_training_launched']
    original.write(OUT / 'guest_inventory.json', guest)
    free = guest['disk']['free_bytes']
    value = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
             'instance_id': instance['id'], 'free_bytes': free, 'free_GiB': free / 1024**3,
             'manual_tail_free_requirement_bytes': 2 * 1024**3,
             'meets_requirement_before_install': free >= 2 * 1024**3,
             'GPU_compute_idle_at_snapshot': not guest['gpu_processes']['stdout'].strip(),
             'hostkey_pinned': HOSTKEY, 'TLS_validation_enabled': True,
             'trust_receipt_sha256': original.sha(trust), 'trust_bundle_sha256': original.sha(ca),
             'inspector_sha256': original.sha(__file__),
             'original_inspector_sha256': original.sha(original.__file__),
             'guest_inventory_sha256': original.sha(OUT / 'guest_inventory.json'),
             'remote_inventory_source_sha256': original.sha(OUT / 'remote_inventory_source.py'),
             'seconds': time.monotonic() - started, 'files_removed': 0,
             'VM_started': False, 'diagnostic_or_training_launched': False,
             'model_or_gradient_calls': 0, 'goal_complete': False}
    original.write(OUT / 'availability.json', value)
    print({key: value[key] for key in ['complete', 'free_GiB', 'GPU_compute_idle_at_snapshot', 'files_removed']}, flush=True)


if __name__ == '__main__':
    main()
