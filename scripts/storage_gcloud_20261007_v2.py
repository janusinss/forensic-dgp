"""Finite maintenance transport for exact backed caches; never launch training."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2'
HOME = '/home/janusdominic0'
PHASE = 'inactive-feature-caches'
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
BACKEND = 'cleanup_cctv_dgp_vm_storage_20261007_v2.py'
AUDITOR = 'independent_cctv_dgp_vm_storage_cleanup_20261007_v2.py'
PLAN = PHASE + '_plan_20261007_v2.json'
MANIFEST = 'cache_manifest_for_cleanup_20261007_v2.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ssh_source(source, label, seconds):
    encoded = base64.b64encode(source.encode()).decode()
    bootstrap = 'import base64;exec(base64.b64decode(' + repr(encoded) + '))'
    command = 'python3 -B -c ' + shlex.quote(bootstrap)
    transport.run(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + transport.BASE +
                  ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + KEY,
                   '--command=' + command], label, seconds)


def scp(source, destination, label):
    transport.run(['compute', 'scp'] + transport.BASE +
                  ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag=' + KEY,
                   str(source), str(destination)], label, 240)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guard-upload', 'upload', 'verify', 'apply', 'audit', 'return'))
    parser.add_argument('--file')
    args = parser.parse_args()
    transport.OUT = OUT
    trust = read(OUT / 'hostkey_verification.json')
    assert trust['complete'] and trust['known_historical_hostkey'] == KEY and trust['current_offered_key_exact_match']
    preparation = read(OUT / 'preparation.json')
    assert preparation['complete'] and preparation['backup_files_verified'] == 4431
    assert preparation['backend_sha256'] == sha(ROOT / 'scripts' / BACKEND)
    assert preparation['plan_sha256'] == sha(OUT / PLAN)
    assert preparation['manifest_sha256'] == sha(OUT / MANIFEST)
    if args.mode == 'guard-upload':
        source = "import os,json;from pathlib import Path;assert os.uname().nodename.split('.')[0]=='forensic-dgp-thesis';assert str(Path.home())==" + repr(HOME) + ';'
        source += 'names=' + repr([BACKEND, AUDITOR, PLAN, MANIFEST]) + ';'
        source += "assert all(not os.path.lexists(Path.home()/name) for name in names);assert not os.path.lexists(Path.home()/'forensic-dgp/maintenance_storage_20261007_v2');"
        source += "print(json.dumps({'complete':True,'fresh_control_destinations':len(names),'files_removed':0,'training_started':False}))"
        ssh_source(source, 'fresh_control_upload_guard', 120)
    elif args.mode == 'upload':
        assert read(OUT / 'fresh_control_upload_guard.log')['complete']
        assert args.file in (BACKEND, AUDITOR, PLAN, MANIFEST)
        path = ROOT / 'scripts' / args.file if args.file.endswith('.py') else OUT / args.file
        scp(path, 'janusdominic0@forensic-dgp-thesis:' + HOME + '/' + args.file,
            'upload_' + args.file.replace('.', '_'))
    elif args.mode in ('verify', 'apply', 'audit'):
        pins = {name: sha(ROOT / 'scripts' / name if name.endswith('.py') else OUT / name)
                for name in (BACKEND, AUDITOR, PLAN, MANIFEST)}
        source = 'import hashlib,subprocess;from pathlib import Path;\n'
        source += 'pins=' + repr(pins) + '\n'
        source += "for name,pin in pins.items():\n assert hashlib.sha256((Path.home()/name).read_bytes()).hexdigest()==pin,name\n"
        if args.mode == 'audit':
            command = ['python3', '-B', HOME + '/' + AUDITOR, '--phase', PHASE,
                       '--plan-sha', pins[PLAN], '--backend-sha', pins[BACKEND]]
        else:
            if args.mode == 'apply':
                pre = read(OUT / 'remote_receipts/verification.json')
                assert pre['complete'] and pre['plan_sha256'] == pins[PLAN] and pre['files_removed'] == 0
                local = read(OUT / 'local_backup_pre_apply_audit.json')
                assert local['complete'] and local['verified_cache_files'] == 4431 and local['manifest_sha256'] == pins[MANIFEST]
            command = ['python3', '-B', HOME + '/' + BACKEND, '--plan', HOME + '/' + PLAN,
                       '--expected-plan-sha', pins[PLAN], '--verify-only' if args.mode == 'verify' else '--apply']
        source += 'subprocess.run(' + repr(command) + ',check=True,timeout=1830)\n'
        ssh_source(source, args.mode, 1980)
    else:
        assert args.file in ('verification.json', 'protected_before.json', 'cleanup_receipt.json', 'deletions.jsonl')
        destination = OUT / 'remote_receipts' / args.file
        destination.parent.mkdir(exist_ok=True)
        assert not destination.exists(), 'Preserve existing returned evidence'
        remote = HOME + '/forensic-dgp/maintenance_storage_20261007_v2/' + PHASE + '/' + args.file
        scp('janusdominic0@forensic-dgp-thesis:' + remote, destination,
            'return_' + args.file.replace('.', '_'))


if __name__ == '__main__':
    main()
