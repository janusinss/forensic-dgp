"""Hash-bound maintenance of one locally backed-up home archive; no model work."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import time
import urllib.request

HOME = Path('/home/janusdominic0')
ROOT = HOME / 'forensic-dgp'
TARGET = HOME / 'cctv-dgp-bank-photometric-guard-v1-results.tar.gz'
OUT = HOME / 'dgp_bank_group_gradients_v1_storage_receipts'
PACKETS = ['cctv_dgp_bank_photometric_guard_v1_vm', 'cctv_dgp_bank_group_gradients_v1_vm']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def idle():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=15, check=True)
    assert not gpu.stdout.strip(), 'Preserve active GPU workload'
    panes = subprocess.run(['tmux', 'list-panes', '-a', '-F',
                            '#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}'],
                           capture_output=True, text=True, timeout=15)
    assert panes.returncode in (0, 1)
    assert all(line.split('|')[2] in ('bash', 'sh', 'zsh', 'fish') for line in panes.stdout.splitlines()), 'Preserve active tmux task'
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            command = (entry / 'comm').read_text().strip()
            cwd = (entry / 'cwd').resolve(strict=True)
            args = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if cwd.is_relative_to(ROOT) and command not in ('bash', 'sh', 'zsh', 'fish', 'tmux: client', 'tmux: server'):
                raise AssertionError('Preserve active research process: ' + str(entry.name) + ' ' + args)
            for descriptor in (entry / 'fd').iterdir():
                try:
                    assert descriptor.resolve(strict=True) != TARGET, 'Archive is in use; retain it'
                except (FileNotFoundError, PermissionError):
                    pass
            processes.append(dict(pid=int(entry.name), command=command, cwd=str(cwd), args=args))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return dict(GPU_compute_idle=True, gpu=gpu.stdout, tmux=panes.stdout, user_processes=processes)


def candidate(row):
    path = Path(row['path'])
    assert path == TARGET and path.parent.resolve() == HOME and not path.is_symlink()
    assert not any(p.is_symlink() for p in path.parents)
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid == 1001 and info.st_nlink == 1
    actual = dict(path=str(path), bytes=info.st_size, allocated_bytes=info.st_blocks * 512,
                  uid=info.st_uid, nlink=info.st_nlink, inode=info.st_ino, mtime_ns=info.st_mtime_ns)
    assert all(row[k] == v for k, v in actual.items()), 'Exact candidate metadata changed'
    assert sha(path) == row['sha256'], 'VM archive differs from full retained local backup'
    return actual


def protected(plan):
    rows, hashes = [], {}
    for folder, dirs, files in os.walk(ROOT, followlinks=False):
        dirs.sort()
        files.sort()
        for name in dirs + files:
            path = Path(folder) / name
            info = path.lstat()
            relative = path.relative_to(ROOT).as_posix()
            rows.append([relative, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                         info.st_ino, info.st_nlink, info.st_uid,
                         os.readlink(path) if stat.S_ISLNK(info.st_mode) else None])
            if path.parts[len(ROOT.parts)] in PACKETS and stat.S_ISREG(info.st_mode):
                hashes[relative] = sha(path)
    for name, expected in plan['critical_sha256'].items():
        assert hashes[name] == expected, 'Protected packet binding differs: ' + name
    return dict(metadata=sorted(rows), packet_bytes_sha256=hashes,
                all_scientific_cache_bytes_rehashed=False)


def gzwrite(name, value):
    with gzip.open(OUT / name, 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--phase', choices=['verify', 'apply', 'audit'], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    def stop(*_):
        raise TimeoutError('Finite 900-second maintenance stop; retain evidence')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(900)
    assert Path.home().resolve() == HOME and os.getuid() == 1001
    assert socket.gethostname().split('.')[0] == 'forensic-dgp-thesis'
    req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',
                                 headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.read().decode() == '4410777042005672095'
    assert sha(args.plan) == args.plan_sha
    plan = read(args.plan)
    assert plan['complete'] and plan['worker_sha256'] == sha(__file__)
    assert plan['authorization_scope'] == 'one exact completed home archive; full hash-verified local backup; all research tree assets retained'
    assert len(plan['candidates']) == 1 and plan['candidates'][0]['path'] == str(TARGET)
    runtime = idle()
    if args.phase == 'verify':
        assert not OUT.exists(), 'Preserve previous maintenance evidence'
        candidate(plan['candidates'][0])
        before = protected(plan)
        OUT.mkdir()
        gzwrite('protected_before.json.gz', before)
        write(OUT / 'verify_receipt.json', dict(complete=True, plan_sha256=args.plan_sha,
            worker_sha256=sha(__file__), candidate_count=1, files_removed=0,
            protected_sha256=sha(OUT / 'protected_before.json.gz'), protected_canonical_sha256=canonical(before),
            research_metadata_entries=len(before['metadata']), packet_hashes=len(before['packet_bytes_sha256']),
            runtime=runtime, free_bytes=shutil.disk_usage(ROOT).free, seconds=time.monotonic()-started))
        print(json.dumps(read(OUT / 'verify_receipt.json')), flush=True)
        return
    verified = read(OUT / 'verify_receipt.json')
    assert verified['complete'] and verified['plan_sha256'] == args.plan_sha
    assert sha(OUT / 'protected_before.json.gz') == verified['protected_sha256']
    with gzip.open(OUT / 'protected_before.json.gz', 'rb') as stream:
        before = json.load(stream)
    assert canonical(before) == verified['protected_canonical_sha256']
    assert protected(plan) == before, 'Research tree changed; preserve candidate'
    if args.phase == 'audit':
        assert not TARGET.exists() and not os.path.lexists(TARGET)
        result = dict(complete=True, plan_sha256=args.plan_sha, worker_sha256=sha(__file__),
            deleted_exact_archive_absent=True, protected_canonical_sha256=canonical(before),
            free_bytes=shutil.disk_usage(ROOT).free, packet_hashes=len(before['packet_bytes_sha256']),
            fresh_runtime=runtime, files_removed_by_audit=0, model_gradient_or_training_calls=0,
            seconds=time.monotonic()-started)
        write(OUT / 'live_audit.json', result)
        print(json.dumps(result), flush=True)
        return
    assert not (OUT / 'cleanup_receipt.json').exists() and not (OUT / 'deletion_ledger.jsonl').exists()
    row = plan['candidates'][0]
    candidate(row)
    runtime = idle()
    free_before = shutil.disk_usage(ROOT).free
    with (OUT / 'deletion_ledger.jsonl').open('x', encoding='utf-8') as ledger:
        # Only this explicit home file is ever unlinked. No recursive delete exists.
        TARGET.unlink()
        ledger.write(json.dumps(dict(path=str(TARGET), sha256=row['sha256'], bytes=row['bytes'],
            inode=row['inode'], uid=row['uid'], nlink=row['nlink'], allocated_bytes=row['allocated_bytes'],
            local_backup=row['local_backup'], backup_verified=True))+'\n')
        ledger.flush()
        os.fsync(ledger.fileno())
    after = protected(plan)
    gzwrite('protected_after.json.gz', after)
    assert before == after, 'Retained research tree differs; preserve removal ledger'
    free_after = shutil.disk_usage(ROOT).free
    write(OUT / 'cleanup_receipt.json', dict(complete=True, plan_sha256=args.plan_sha,
        worker_sha256=sha(__file__), files_removed=1, removed_paths=[str(TARGET)],
        protected_metadata_entries=len(after['metadata']), packet_byte_hashes=len(after['packet_bytes_sha256']),
        protected_canonical_before=canonical(before), protected_canonical_after=canonical(after),
        free_before_bytes=free_before, free_after_bytes=free_after, measured_free_change_bytes=free_after-free_before,
        allocated_reclaim_bytes=row['allocated_bytes'], archive_logical_bytes=row['bytes'],
        all_scientific_cache_bytes_rehashed=False, research_checkpoints_caches_splits_logs_and_failures_retained=True,
        runtime_before_removal=runtime, VM_started=False, model_gradient_or_training_calls=0,
        goal_complete=False, seconds=time.monotonic()-started))
    write(OUT / 'manifest.json', dict(complete=True, plan_sha256=args.plan_sha,
        files_sha256={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    import tarfile
    archive = HOME / 'dgp-bank-group-gradients-v1-storage-receipts.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for path in sorted(OUT.iterdir()):
            tar.add(path, arcname='receipts/'+path.name, recursive=False)
    write(HOME / 'dgp-bank-group-gradients-v1-storage-export.json', dict(complete=True,
        archive_sha256=sha(archive), bytes=archive.stat().st_size, model_gradient_or_training_calls=0))
    print(json.dumps(dict(complete=True, files_removed=1, free_GiB=free_after/1024**3,
                         seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
