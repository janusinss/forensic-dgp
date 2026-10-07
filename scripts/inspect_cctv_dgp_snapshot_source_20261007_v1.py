"""Read-only, finite research-cache evidence before a migration disk snapshot."""
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import time


REPO = Path('/home/janusdominic0/forensic-dgp')
ROOTS = [
    REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/anatomical_cache',
    REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/fixed_cache',
    REPO / 'cctv_dgp_broader_codes_vm_v16_r2/outputs/broader_codes_v16_r2/cache',
]


def inspect():
    started = time.monotonic()
    assert os.uname().nodename == 'forensic-dgp-thesis'
    assert pwd.getpwuid(os.getuid()).pw_name == 'janusdominic0'
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
        text=True, timeout=30).strip()
    assert not gpu, 'GPU work active; keep source untouched and defer backup'
    panes = subprocess.run(
        ['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_current_command}'],
        capture_output=True, text=True, timeout=30)
    if panes.returncode:
        assert 'no server running' in panes.stderr or 'no sessions' in panes.stderr
    pane_rows = [row.split('\t', 1) for row in panes.stdout.splitlines() if row.strip()]
    assert all(row[1] in ('bash', 'sh', 'zsh', 'fish', 'tail', 'less') for row in pane_rows), (
        'Non-idle tmux pane; defer snapshot until the workload is understood')
    root_device = os.stat('/').st_dev
    files = []
    for root in ROOTS:
        assert root.is_dir() and root.resolve() == root
        assert root.stat().st_dev == root_device, 'Cache is on another disk; include that disk'
        for path in sorted(root.rglob('*')):
            assert time.monotonic() - started < 600, 'Finite source audit stop: 600 seconds'
            if path.is_dir():
                continue
            assert not path.is_symlink() and path.resolve() == path
            before = path.stat()
            assert stat.S_ISREG(before.st_mode) and before.st_dev == root_device
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
                    assert time.monotonic() - started < 600, 'Finite source audit stop: 600 seconds'
                    digest.update(block)
            after = path.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), (
                'Cache changed during read; retain failure and defer snapshot')
            files.append({'path': str(path), 'bytes': after.st_size,
                          'sha256': digest.hexdigest(), 'inode': after.st_ino,
                          'mtime_ns': after.st_mtime_ns})
    assert len(files) == 4431 and sum(row['bytes'] for row in files) == 39448585279
    subprocess.run(['sync'], check=True, timeout=120)
    mount = subprocess.check_output(
        ['findmnt', '--json', '--target', '/', '--output', 'SOURCE,TARGET,FSTYPE'],
        text=True, timeout=30)
    print(json.dumps({'complete': True, 'hostname': os.uname().nodename,
                      'GPU_idle': not gpu, 'tmux': pane_rows,
                      'mount': json.loads(mount), 'roots': [str(path) for path in ROOTS],
                      'files': files, 'file_count': len(files),
                      'bytes': sum(row['bytes'] for row in files),
                      'seconds': time.monotonic() - started,
                      'files_removed': 0, 'training_started': False}), flush=True)


if __name__ == '__main__':
    inspect()
