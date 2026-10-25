"""Fresh read-only API/SSH inventory after completed local pass1; never apply.

Default prepares local connection/source pins only. --run is root controlled and
requires independently closed local offload evidence. No upload, VM start/stop,
training, process kill or removal command exists in the guest source/driver.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/vm_offload_connection_20261018_v1'
RESEARCH = ROOT / 'outputs/vm_offload_research_20261018_v1/research_receipt.json'
SOURCE = ROOT / 'outputs/vm_offload_research_20261018_v1/inventory_guest_source.py'
SOURCE_SHA = 'db58983f792d1b049d767d3e7ef7ea55a7868c63f10269e34c31903f07387bda'
TRUST = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
TRUST_SHA = 'b941bf482dcf2dca2f7e77ff7df23deee4b10c414d0e022b11c275192f62a7d2'
HOSTKEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
CLOUD = Path('C:/Users/janus/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd')
CA = ROOT / 'scratch/gcloud-windows-roots-v1.pem'
INSTANCE = 'forensic-dgp-thesis'
INSTANCE_ID = '4410777042005672095'
MAX_LOG = 20 * 1024**2


def require(value, reason):
    if not value:
        raise ValueError(reason)


def file_pin(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        state = p.lstat()
        require(not stat.S_ISLNK(state.st_mode) and not getattr(state, 'st_file_attributes', 0) & 0x400,
                'Linked/reparse connection source forbidden')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Regular single-link source required')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Connection source changed while hashing')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': digest}


def read(path):
    p = Path(path)
    require(p.stat().st_size < MAX_LOG, 'Bounded connection JSON required')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(payload) <= MAX_LOG, 'Bounded receipt required')
    with Path(path).open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return file_pin(path)


def prepare():
    research = read(RESEARCH)
    require(research.get('complete') is True and research.get('VM_connected') is False
            and research.get('cloud_API_called') is False, 'Inert historical research receipt required')
    source_pin = file_pin(SOURCE)
    require(source_pin['sha256'] == SOURCE_SHA and source_pin['bytes'] == 4547, 'Pinned read-only guest source changed')
    ast.parse(SOURCE.read_text(encoding='utf-8'), feature_version=(3, 10))
    require(file_pin(TRUST)['sha256'] == TRUST_SHA, 'Historical exact host-key receipt changed')
    trust = read(TRUST)
    require(trust.get('complete') is True and trust.get('known_historical_hostkey') == HOSTKEY
            and trust.get('current_VM_ID') == INSTANCE_ID, 'Pinned historical identity differs')
    cloudpin, capin = file_pin(CLOUD), file_pin(CA)
    require(CA.read_bytes().count(b'-----BEGIN CERTIFICATE-----') > 0, 'Existing PEM trust bundle required')
    source = base64.b64encode(SOURCE.read_bytes()).decode('ascii')
    remote = 'python3 -B -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(source) + '))')
    api = ['compute', 'instances', 'describe', INSTANCE, '--project=' + INSTANCE,
           '--zone=us-central1-a', '--quiet', '--format=json(id,name,status,machineType,zone,guestAccelerators)']
    ssh = ['compute', 'ssh', 'janusdominic0@' + INSTANCE, '--project=' + INSTANCE,
           '--zone=us-central1-a', '--quiet', '--ssh-flag=-batch', '--ssh-flag=-hostkey',
           '--ssh-flag=' + HOSTKEY, '--command=' + remote]
    require(len(subprocess.list2cmdline([str(CLOUD), *ssh])) < 7900,
            'Windows batch command line cap exceeded')
    return {'format': 'fresh-read-only-vm-storage-connection-plan-20261018-v1', 'complete': True,
            'research': file_pin(RESEARCH), 'guest_source': source_pin, 'historical_trust': file_pin(TRUST),
            'gcloud_executable': cloudpin, 'TLS_CA_bundle': capin,
            'TLS_validation_enabled': True, 'pinned_hostkey': HOSTKEY,
            'identity': {'project': INSTANCE, 'instance': INSTANCE, 'id': INSTANCE_ID,
                         'zone': 'us-central1-a', 'machine': 'g2-standard-4', 'GPU': 'NVIDIA L4',
                         'user': 'janusdominic0'},
            'api_argv': api, 'ssh_argv': ssh, 'caps_seconds': {'API': 60, 'SSH': 240, 'guest_alarm': 180},
            'max_stdout_stderr_bytes': MAX_LOG, 'no_payload_hashes_or_cleanup_authorized': True,
            'VM_connected': False, 'API_called': False, 'uploads': 0, 'files_removed': 0,
            'actual_training_or_neural_calls': 0, 'driver': file_pin(Path(__file__))}


def environment(plan):
    result = dict(os.environ)
    for name in ('CLOUDSDK_CORE_DISABLE_SSL_VALIDATION', 'CLOUDSDK_AUTH_DISABLE_SSL_VALIDATION'):
        require(result.get(name, '').lower() not in ('true', '1', 'yes'), 'TLS validation may not be disabled')
    result['CLOUDSDK_CORE_DISABLE_SSL_VALIDATION'] = 'false'
    result['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = plan['TLS_CA_bundle']['path']
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    return result


def stop_own_transport(child):
    """Bound only our local transport process tree; never a guest process."""
    if child.poll() is not None:
        return
    if os.name == 'nt':
        stopped = subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        require(stopped.returncode in (0, 128) or child.poll() is not None,
                'Own local gcloud transport tree did not terminate')
    else:
        child.kill()
    child.wait(timeout=15)


def invoke(folder, label, argv, cap, env):
    stdout, stderr = folder / (label + '_stdout.json'), folder / (label + '_stderr.log')
    start, child, stop = time.monotonic(), None, None
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            child = subprocess.Popen([str(CLOUD), *argv], stdout=out, stderr=err, env=env,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(folder / (label + '_started.json'), {'pid': child.pid, 'cap_seconds': cap,
                 'read_only_stage': label, 'TLS_validation_enabled': True, 'hostkey_pinned': HOSTKEY})
            while child.poll() is None:
                if time.monotonic() - start > cap or stdout.stat().st_size + stderr.stat().st_size > MAX_LOG:
                    stop = 'time_or_log_cap'; stop_own_transport(child); break
                time.sleep(.25)
            code = child.wait(timeout=15)
        require(stdout.stat().st_size + stderr.stat().st_size <= MAX_LOG, 'Transport log cap exceeded')
        receipt = {'complete': code == 0 and stop is None, 'read_only': True, 'pid': child.pid,
                   'child_terminal': True, 'exit_code': code, 'external_stop': stop,
                   'seconds': time.monotonic() - start, 'cap_seconds': cap,
                   'stdout': file_pin(stdout), 'stderr': file_pin(stderr), 'automatic_retry': False,
                   'TLS_validation_enabled': True, 'hostkey_pinned': HOSTKEY,
                   'uploads': 0, 'files_removed': 0, 'training_calls': 0}
        save(folder / (label + '_transport.json'), receipt)
        require(receipt['complete'], 'Read-only transport failed; preserve terminal evidence, no automatic retry')
        return read(stdout)
    except BaseException as error:
        if child is not None and child.poll() is None:
            stop_own_transport(child)
        save(folder / (label + '_failure.json'), {'complete': False, 'read_only': True,
             'exception_type': type(error).__name__, 'reason': str(error)[:2048],
             'pid': child.pid if child else None, 'child_terminal': child is None or child.poll() is not None,
             'uploads': 0, 'files_removed': 0, 'training_calls': 0})
        raise


def require_local_closure(local):
    require(local.get('complete') is True and local.get('independently_verified') is True and
            local.get('files_removed') == 3 and local.get('logical_bytes_removed') == 15463424100,
            'Independent exact local15.463GB pass1 closure required before VM connection')
    apply_pin = local.get('apply_receipt')
    require(type(apply_pin) is dict and all(k in apply_pin for k in ('path', 'bytes', 'sha256')),
            'Closure must bind exact actual local apply receipt')
    return apply_pin


def run(plan, closure_name, closure_sha):
    closure = Path(closure_name)
    closure = closure if closure.is_absolute() else ROOT / closure
    require(closure.absolute().is_relative_to(ROOT) and file_pin(closure)['sha256'] == closure_sha,
            'Exact local pass1 closure under workspace required')
    local = read(closure)
    recorded_apply = require_local_closure(local)
    apply_path = Path(recorded_apply['path'])
    apply_path = apply_path if apply_path.is_absolute() else ROOT / apply_path
    require(apply_path.absolute().is_relative_to(ROOT), 'Apply receipt must remain in exact workspace')
    apply_pin = file_pin(apply_path)
    require(all(apply_pin[k] == recorded_apply[k] for k in ('bytes', 'sha256')), 'Bound apply receipt changed')
    applied = read(apply_path)
    require(applied.get('complete') is True and applied.get('files_removed') == 3 and
            applied.get('logical_bytes_removed') == 15463424100 and
            applied.get('protected_postapply', {}).get('complete') is True and
            applied.get('namespace_after', {}).get('complete') is True,
            'Actual local removal/protected/namespace closure is not complete')
    for key in ('gcloud_executable', 'TLS_CA_bundle', 'guest_source', 'historical_trust', 'driver'):
        require(file_pin(plan[key]['path']) == plan[key], 'Prepared connection source changed')
    folder = OUT / ('inventory_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    folder.mkdir()
    save(folder / 'intent.json', {'complete_intent': True, 'plan': plan,
         'local_closure': file_pin(closure), 'API_SSH_readonly_only': True,
         'no_VM_start_stop_upload_training_or_cleanup': True})
    env = environment(plan)
    api = invoke(folder, 'api', plan['api_argv'], 60, env)
    require(str(api.get('id')) == INSTANCE_ID and api.get('name') == INSTANCE and
            api.get('zone', '').endswith('/us-central1-a') and
            api.get('machineType', '').endswith('/g2-standard-4'), 'Live API identity mismatch; no SSH')
    if api.get('status') != 'RUNNING':
        save(folder / 'availability.json', {'complete': True, 'VM_status': api.get('status'),
             'guest_available': False, 'VM_started': False, 'files_removed': 0, 'training_calls': 0})
        print(json.dumps({'read_only_inventory': True, 'status': api.get('status'), 'guest_connected': False})); return
    accelerators = api.get('guestAccelerators', [])
    require(len(accelerators) == 1 and accelerators[0].get('acceleratorCount') == 1 and
            accelerators[0].get('acceleratorType', '').endswith('/nvidia-l4'), 'Live accelerator identity mismatch')
    guest = invoke(folder, 'guest', plan['ssh_argv'], 240, env)
    require(guest.get('complete') is True and guest.get('files_removed') == 0 and guest.get('uploads') == 0
            and guest.get('model_gradient_optimizer_training_calls') == 0,
            'Complete read-only guest inventory required')
    identity = guest['identity']
    require(identity['project/project-id'] == INSTANCE and identity['instance/id'] == INSTANCE_ID
            and identity['instance/name'] == INSTANCE and identity['instance/zone'].endswith('/us-central1-a')
            and identity['instance/machine-type'].endswith('/g2-standard-4') and guest['uid'] == 1001
            and 'NVIDIA L4' in guest['commands']['gpu']['stdout'], 'Guest/API identity mismatch')
    save(folder / 'inventory_receipt.json', {'complete': True, 'local_pass1_closed': True,
         'guest_inventory': file_pin(folder / 'guest_stdout.json'), 'API_identity': file_pin(folder / 'api_stdout.json'),
         'free_bytes_at_snapshot': guest['disk']['free_bytes'],
         'GPU_idle_at_snapshot': guest['gpu_compute_idle_observed'],
         'no_candidate_or_cleanup_authorized': True, 'uploads': 0, 'files_removed': 0,
         'training_calls': 0, 'TLS_validation_enabled': True, 'hostkey_pinned': HOSTKEY})
    print(json.dumps({'complete': True, 'read_only_inventory': True,
                      'receipt_directory': str(folder.relative_to(ROOT)), 'files_removed': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--local-closure')
    parser.add_argument('--local-closure-sha256')
    args = parser.parse_args()
    plan = prepare()
    OUT.mkdir(exist_ok=True)
    if not args.run:
        target = OUT / 'connection_plan_v1.json'
        save(target, plan)
        print(json.dumps({'complete': True, 'prepared_only': True, 'plan': file_pin(target),
                          'API_SSH_calls': 0, 'files_removed': 0})); return
    require(args.local_closure and args.local_closure_sha256, 'Explicit closed local pass1 required before API/SSH')
    run(plan, args.local_closure, args.local_closure_sha256)


if __name__ == '__main__':
    main()
