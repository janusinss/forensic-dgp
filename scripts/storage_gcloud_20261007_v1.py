"""Fresh bounded transport for explicitly authorized VM storage maintenance."""
import argparse
import base64
import json
from pathlib import Path
import shlex

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1'
transport.OUT = OUT
HOME = transport.HOME
REPO = transport.REPO
BACKEND = 'cleanup_cctv_dgp_vm_storage_20261007_v1.py'
AUDITOR = 'independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py'
PLAN = 'archive-duplicates_plan_20261007_v1.json'
REMOTE = REPO + '/maintenance_storage_20261007_v1/archive-duplicates'


def source_command(source):
    encoded = base64.b64encode(source.read_bytes()).decode('ascii')
    return 'python3 -B -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inventory', 'upload', 'verify', 'apply', 'audit', 'return', 'runtime'))
    parser.add_argument('--file')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.mode in ('inventory', 'runtime'):
        source = ROOT / 'scripts' / ('inspect_cctv_dgp_vm_storage_20261005.py' if args.mode == 'inventory'
                                    else 'verify_cctv_dgp_vm_runtime_after_storage_cleanup_20261007_v1.py')
        label = 'inventory_before' if args.mode == 'inventory' else 'final_runtime'
        transport.ssh(source_command(source), label, 420 if args.mode == 'inventory' else 180)
        result = json.loads((OUT / (label + '.log')).read_text(encoding='utf-8-sig'))
        assert result['complete'] and result['hostname'] == 'forensic-dgp-thesis'
        with (OUT / (label + '.json')).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('disk', 'GPU', 'tmux', 'related_processes') if key in result}), flush=True)
    elif args.mode == 'upload':
        assert args.file in (BACKEND, AUDITOR, PLAN)
        path = OUT / PLAN if args.file == PLAN else ROOT / 'scripts' / args.file
        assert path.is_file()
        transport.run(['compute', 'scp'] + transport.BASE + [str(path),
                      'janusdominic0@forensic-dgp-thesis:' + HOME + '/' + args.file],
                      'upload_' + args.file.replace('.', '_'), 120)
    elif args.mode in ('verify', 'apply', 'audit'):
        pin = transport.sha(OUT / PLAN)
        backend_pin = transport.sha(ROOT / 'scripts' / BACKEND)
        helper = AUDITOR if args.mode == 'audit' else BACKEND
        helper_pin = transport.sha(ROOT / 'scripts' / helper)
        check = 'import hashlib;from pathlib import Path;assert hashlib.sha256(Path(' + repr(HOME + '/' + helper) + ').read_bytes()).hexdigest()==' + repr(helper_pin)
        command = 'python3 -B -c ' + shlex.quote(check) + ' && python3 -B ' + shlex.quote(HOME + '/' + helper)
        if args.mode == 'audit':
            command += ' --phase archive-duplicates --plan-sha ' + pin + ' --backend-sha ' + backend_pin
        else:
            command += ' --plan ' + shlex.quote(HOME + '/' + PLAN) + ' --expected-plan-sha ' + pin
            command += ' --verify-only' if args.mode == 'verify' else ' --apply'
        transport.ssh(command, 'independent_VM_audit' if args.mode == 'audit' else args.mode, 960, True)
    elif args.mode == 'return':
        assert args.file in ('verification.json', 'protected_before.json', 'cleanup_receipt.json', 'deletions.jsonl')
        destination = OUT / 'remote_receipts' / args.file
        destination.parent.mkdir(exist_ok=True)
        assert not destination.exists()
        transport.run(['compute', 'scp'] + transport.BASE + ['janusdominic0@forensic-dgp-thesis:' + REMOTE + '/' + args.file,
                      str(destination)], 'return_' + args.file.replace('.', '_'), 240)


if __name__ == '__main__':
    main()
