"""Bounded gcloud transport for authorized storage maintenance; no training launcher."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1'
HOME = '/home/janusdominic0'
REPO = HOME + '/forensic-dgp'
GCLOUD = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
BASE = ['--project=forensic-dgp-thesis', '--zone=us-central1-a', '--quiet']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(args, label, seconds=120, echo=False):
    environment = dict(os.environ)
    environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ROOT / 'scratch/gcloud_windows_trust.pem')
    assert environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] and OUT.is_dir()
    assert label and all(c.isalnum() or c in '_-' for c in label)
    started = time.monotonic()
    print(json.dumps({'stage': label, 'started': True, 'timeout_seconds': seconds}), flush=True)
    with (OUT / (label + '.log')).open('xb') as stdout, (OUT / (label + '_stderr.log')).open('xb') as stderr:
        result = subprocess.run([str(GCLOUD)] + args, env=environment, stdout=stdout, stderr=stderr, timeout=seconds)
    receipt = {'exit_code': result.returncode, 'seconds': time.monotonic() - started, 'stage': label,
               'stdout_sha256': sha(OUT / (label + '.log')), 'stderr_sha256': sha(OUT / (label + '_stderr.log'))}
    with (OUT / (label + '_transport.json')).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    if result.returncode or echo:
        print((OUT / (label + ('_stderr.log' if result.returncode else '.log'))).read_text(encoding='utf-8', errors='replace')[-4500:], flush=True)
    assert result.returncode == 0, 'Transport failed; preserve logs, no automatic retry'


def ssh(command, label, seconds=120, echo=False):
    run(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + BASE +
        ['--ssh-flag=-batch', '--command=' + command], label, seconds, echo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('cache-inventory', 'upload', 'verify-archives', 'apply-archives', 'audit-archives', 'audit-caches',
                                         'verify-caches', 'apply-caches', 'return-receipt', 'download-v23', 'final-disk'))
    parser.add_argument('--file')
    args = parser.parse_args()
    helper = 'cleanup_cctv_dgp_vm_storage_20261006_v1.py'
    if args.mode == 'cache-inventory':
        source = ROOT / 'scripts/inspect_cctv_dgp_vm_inactive_caches_20261005_r2.py'
        encoded = base64.b64encode(source.read_bytes()).decode('ascii')
        program = 'import base64;exec(base64.b64decode(' + repr(encoded) + '))'
        ssh('python3 -c ' + shlex.quote(program), 'inactive_cache_manifest', 1260)
        d = json.loads((OUT / 'inactive_cache_manifest.log').read_text(encoding='utf-8-sig'))
        assert d['complete'] and d['GPU_idle'] and not d['files_removed']
        with (OUT / 'inactive_cache_manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(d, indent=2) + '\n')
        print(json.dumps({'files': len(d['files']), 'backup_bytes': d['all_backup_bytes'], 'seconds': d['seconds']}))
    elif args.mode.startswith('audit-'):
        phase = 'archive-duplicates' if args.mode.endswith('archives') else 'inactive-feature-caches'
        source = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261006_v1.py'
        command = 'python3 ' + shlex.quote(HOME + '/' + source.name) + ' --phase ' + phase + ' --plan-sha ' + sha(OUT / (phase + '_plan.json')) + ' --backend-sha ' + sha(ROOT / 'scripts' / helper)
        ssh(command, args.mode + '_file_r1', 1980, True)
    elif args.mode == 'upload':
        assert args.file in (helper, 'independent_cctv_dgp_vm_storage_cleanup_20261006_v1.py', 'archive-duplicates_plan.json', 'inactive-feature-caches_plan.json')
        path = ROOT / 'scripts' / args.file if args.file.endswith('.py') else OUT / args.file
        assert path.is_file()
        # One controlled local file, one destination. Existing pilot files are not targets.
        run(['compute', 'scp'] + BASE + [str(path), 'janusdominic0@forensic-dgp-thesis:' + HOME + '/' + args.file],
            'upload_' + args.file.replace('.', '_'), 120)
    elif args.mode.startswith(('verify-', 'apply-')):
        phase = 'archive-duplicates' if args.mode.endswith('archives') else 'inactive-feature-caches'
        plan = OUT / (phase + '_plan.json')
        pin = sha(plan)
        expected_helper = sha(ROOT / 'scripts' / helper)
        operation = '--verify-only' if args.mode.startswith('verify-') else '--apply'
        check = 'import hashlib;from pathlib import Path;assert hashlib.sha256(Path(' + repr(HOME + '/' + helper) + ').read_bytes()).hexdigest()==' + repr(expected_helper)
        command = 'python3 -c ' + shlex.quote(check) + ' && python3 ' + shlex.quote(HOME + '/' + helper) + ' --plan ' + shlex.quote(HOME + '/' + plan.name) + ' --expected-plan-sha ' + pin + ' ' + operation
        ssh(command, args.mode, 1980, True)
    elif args.mode == 'return-receipt':
        assert args.file and args.file.count('/') == 1
        phase, name = args.file.split('/')
        assert phase in ('archive-duplicates', 'inactive-feature-caches') and name in ('verification.json', 'protected_before.json', 'cleanup_receipt.json', 'deletions.jsonl')
        destination = OUT / 'remote_receipts' / phase / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        remote = REPO + '/maintenance_storage_20261006_v1/' + phase + '/' + name
        run(['compute', 'scp'] + BASE + ['janusdominic0@forensic-dgp-thesis:' + remote, str(destination)],
            'return_' + phase + '_' + name.replace('.', '_'), 180)
    elif args.mode == 'download-v23':
        name = args.file
        assert name in ('cctv-dgp-detail-skip-v23-results.tar.gz', 'cctv-dgp-detail-skip-v23-results.tar.gz.sha256', 'cctv-dgp-detail-skip-v23-export.json')
        destination = ROOT / 'outputs' / name
        assert not destination.exists(), 'Preserve existing downloaded evidence'
        run(['compute', 'scp'] + BASE + ['janusdominic0@forensic-dgp-thesis:' + HOME + '/' + name, str(destination)],
            'backup_v23_' + name.replace('.', '_'), 300)
    elif args.mode == 'final-disk':
        code = "import json,os,shutil,subprocess;d=shutil.disk_usage('/');print(json.dumps({'hostname':os.uname().nodename,'total_bytes':d.total,'used_bytes':d.used,'free_bytes':d.free,'GPU':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),'tmux':subprocess.check_output(['tmux','list-panes','-a','-F','#{session_name} #{pane_current_command}'],text=True)}));print(subprocess.check_output(['df','-h','/'],text=True))"
        ssh('python3 -c ' + shlex.quote(code), 'final_disk', 90, True)


if __name__ == '__main__':
    main()
