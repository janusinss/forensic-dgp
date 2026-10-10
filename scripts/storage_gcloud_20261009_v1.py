"""Pinned, bounded transport for this authorized archive-only VM cleanup."""
import argparse
import ast
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1'
LOGS = OUT / 'transport'
HOME = '/home/janusdominic0'
REPO = HOME + '/forensic-dgp'
REMOTE = REPO + '/maintenance_storage_20261009_v1/archive-duplicates'
PLAN = OUT / 'archive-duplicates_plan_20261009_v1.json'
BACKEND = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261009_v1.py'
AUDITOR = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261009_v1.py'
HOSTKEY = transport.HOSTKEY
BASE = ['--project=forensic-dgp-thesis', '--zone=us-central1-a', '--quiet']
FILES = (BACKEND, AUDITOR, PLAN)


def setup():
    import os
    cloud = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    ca = ROOT / 'scratch/gcloud_windows_trust.pem'
    assert cloud.is_file() and ca.is_file()
    trust = transport.read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json')
    assert trust['complete'] and trust['known_historical_hostkey'] == HOSTKEY and trust['current_offered_key_exact_match']
    environment = dict(os.environ)
    environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ca)
    LOGS.mkdir(exist_ok=True)
    transport.OUT = LOGS
    return cloud, environment


def invoke(args, label, cap):
    cloud, environment = setup()
    print({'stage': label, 'started': True, 'cap_seconds': cap}, flush=True)
    started = time.monotonic()
    expired = False
    with (LOGS / (label + '_stdout.log')).open('xb') as stdout, (LOGS / (label + '_stderr.log')).open('xb') as stderr:
        try:
            result = subprocess.run([str(cloud)] + args, env=environment, stdout=stdout, stderr=stderr, timeout=cap)
            code = result.returncode
        except subprocess.TimeoutExpired:
            expired = True
            code = None
    receipt = {'complete': code == 0, 'exit_code': code, 'stage': label,
               'seconds': time.monotonic() - started, 'cap_seconds': cap, 'observation_timeout': expired,
               'stdout_sha256': transport.sha(LOGS / (label + '_stdout.log')),
               'stderr_sha256': transport.sha(LOGS / (label + '_stderr.log')),
               'read_only': label != 'apply' and not label.startswith('upload_'),
               'exact_plan_deletion_authorized_for_this_stage': label == 'apply',
               'TLS_validation_enabled': True, 'hostkey_pinned': HOSTKEY,
               'training_or_diagnostic_launched': False, 'automatic_retry': False}
    transport.write(LOGS / (label + '_transport.json'), receipt)
    print({'stage': label, **{key: receipt[key] for key in ('complete', 'exit_code', 'seconds', 'observation_timeout')}}, flush=True)
    if not receipt['complete']:
        print((LOGS / (label + '_stderr.log')).read_text(encoding='utf-8', errors='replace')[-4000:], flush=True)
    assert receipt['complete'], 'Preserve failed transport; no automatic repeat'


def ssh(source, label, cap):
    encoded = base64.b64encode(source.encode()).decode()
    command = 'python3 -B -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')
    args = ['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + BASE + [
        '--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + HOSTKEY, '--command=' + command]
    assert len(subprocess.list2cmdline(args)) < 7900
    invoke(args, label, cap)


def pins():
    return {HOME + '/' + path.name: transport.sha(path) for path in FILES}


def remote_check():
    return ("import hashlib,json,os,socket,subprocess;from pathlib import Path\n"
            "assert socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and Path.home().resolve()==Path('/home/janusdominic0')\n"
            + 'pins=' + repr(pins()) + '\n'
            + "for name,digest in pins.items():\n p=Path(name);assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest\n")


def validate_preapply_backups():
    started = time.monotonic()
    plan = transport.read(PLAN)
    verified = transport.read(OUT / 'remote_receipts/verification.json')
    protected_path = OUT / 'remote_receipts/protected_before.json'
    assert verified['complete'] and verified['files_removed'] == 0
    assert verified['plan_sha256'] == transport.sha(PLAN) and verified['script_sha256'] == transport.sha(BACKEND)
    assert verified['protected_snapshot_sha256'] == transport.sha(protected_path)
    assert verified['pip_cache_files'] == [] and len(verified['archives']) == len(plan['files'])
    assert {row['path']: row['sha256'] for row in verified['archives']} == {row['path']: row['sha256'] for row in plan['files']}
    protected = transport.read(protected_path)
    assert all(protected['sha256'][name] == digest for name, digest in plan['protected_assets_sha256'].items())
    for index, row in enumerate(plan['files'], 1):
        backup = Path(row['local_backup'])
        info = backup.stat()
        assert info.st_size == row['bytes'] and info.st_mtime_ns == row['local_backup_mtime_ns']
        assert not backup.is_symlink() and transport.sha(backup) == row['sha256']
        assert transport.sha(row['local_checksum']) == row['local_checksum_sha256']
        if row['local_export_receipt']:
            assert transport.sha(row['local_export_receipt']) == row['local_export_receipt_sha256']
        print({'Windows_preapply_backup_rechecked': index, 'of': len(plan['files'])}, flush=True)
    for name, digest in transport.read(OUT / 'local_protected_bindings.json').items():
        assert transport.sha(name) == digest
    value = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
             'plan_sha256': transport.sha(PLAN), 'backup_archives': len(plan['files']),
             'backup_bytes': sum(row['bytes'] for row in plan['files']),
             'verification_sha256': transport.sha(OUT / 'remote_receipts/verification.json'),
             'protected_snapshot_sha256': transport.sha(protected_path),
             'all_local_backups_retained': True, 'files_removed': 0, 'seconds': time.monotonic() - started}
    transport.write(OUT / 'local_preapply_backup_recheck.json', value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('upload', 'verify', 'return', 'apply', 'audit'))
    parser.add_argument('--file', choices=('verification.json', 'protected_before.json', 'cleanup_receipt.json', 'deletions.jsonl'))
    args = parser.parse_args()
    assert transport.read(OUT / 'local_backup_verification.json')['complete']
    if args.mode == 'upload':
        for path in (BACKEND, AUDITOR):
            ast.parse(path.read_text(encoding='utf-8'))
        source = ("import os,socket;from pathlib import Path\n"
                  "assert socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and Path.home().resolve()==Path('/home/janusdominic0')\n"
                  + 'names=' + repr(list(pins())) + '\n'
                  + "assert all(not os.path.lexists(name) for name in names)\n"
                  + "assert not os.path.lexists('/home/janusdominic0/forensic-dgp/maintenance_storage_20261009_v1')\n"
                  + "print('Fresh maintenance destinations confirmed; no deletion or launch')\n")
        ssh(source, 'fresh_upload_destinations', 90)
        for path in FILES:
            invoke(['compute', 'scp'] + BASE + ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag=' + HOSTKEY,
                   str(path), 'janusdominic0@forensic-dgp-thesis:' + HOME + '/' + path.name],
                   'upload_' + path.name.replace('.', '_').replace('-', '_'), 120)
        transport.write(OUT / 'upload_manifest.json', {'complete': True, 'remote_files_sha256': pins(),
                                                       'training_or_diagnostic_launched': False})
    elif args.mode == 'return':
        assert args.file
        target = OUT / 'remote_receipts' / args.file
        target.parent.mkdir(exist_ok=True)
        assert not target.exists()
        invoke(['compute', 'scp'] + BASE + ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag=' + HOSTKEY,
               'janusdominic0@forensic-dgp-thesis:' + REMOTE + '/' + args.file, str(target)],
               'return_' + args.file.replace('.', '_'), 360)
    else:
        assert transport.read(OUT / 'upload_manifest.json')['remote_files_sha256'] == pins()
        if args.mode == 'apply':
            validate_preapply_backups()
        if args.mode in ('verify', 'apply'):
            operation = '--verify-only' if args.mode == 'verify' else '--apply'
            arguments = ['python3', '-B', HOME + '/' + BACKEND.name, '--plan', HOME + '/' + PLAN.name,
                         '--expected-plan-sha', transport.sha(PLAN), operation]
        else:
            arguments = ['python3', '-B', HOME + '/' + AUDITOR.name, '--phase', 'archive-duplicates',
                         '--plan-sha', transport.sha(PLAN), '--backend-sha', transport.sha(BACKEND)]
        source = remote_check() + 'subprocess.run(' + repr(arguments) + ',check=True,timeout=1830)\n'
        ssh(source, args.mode, 1980)
        if args.mode == 'audit':
            value = json.loads((LOGS / 'audit_stdout.log').read_text(encoding='utf-8'))
            assert value['complete'] and value['files_removed_by_this_audit'] == 0
            assert value['plan_sha256'] == transport.sha(PLAN) and value['backend_sha256'] == transport.sha(BACKEND)
            transport.write(OUT / 'independent_live_audit.json', value)
            print({key: value[key] for key in ('complete', 'deleted_files_verified_absent', 'free_bytes_live',
                  'protected_hashed_files_live_verified', 'manual_tail_free_requirement_met', 'seconds')}, flush=True)


if __name__ == '__main__':
    main()
