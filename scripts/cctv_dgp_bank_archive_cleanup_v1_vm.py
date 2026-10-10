"""Remove only four verified obsolete home archives; no research-tree mutation."""
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
OUT = HOME / 'dgp_bank_archive_cleanup_v1_receipts'
TARGET_NAMES = {'cctv-dgp-multiscale-calibration-v1-execution.tar.gz', 'cctv-dgp-multiscale-calibration-v1-results.tar.gz', 'dgp-head4-archive-cleanup-v1-receipts.tar.gz', 'dgp-multiscale-archive-cleanup-v1-receipts.tar.gz'}
EXTENSIONS = {'.py', '.md', '.json', '.jsonl', '.pth', '.pt', '.onnx', '.sha256', '.log', '.csv', '.sh'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def workload(targets):
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=15)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), 'Active GPU work; retain archives'
    tmux = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}|#{pane_dead}'],
                          capture_output=True, text=True, timeout=15)
    assert tmux.returncode == 0 or (tmux.returncode == 1 and 'No such file or directory' in tmux.stderr), 'tmux observation failed'
    for pane in tmux.stdout.splitlines():
        fields = pane.split('|')
        assert len(fields) == 6 and fields[2] in {'bash', 'zsh', 'sh', 'fish'} and fields[5] == '0', 'Active tmux task; retain archives'
    research_processes = []
    open_targets = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            comm = (folder / 'comm').read_text().strip()
            command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = (folder / 'cwd').resolve()
            if (str(ROOT) + '/' in command and comm not in {'sshd', 'bash', 'sh', 'zsh'}) or (
                cwd.is_relative_to(ROOT) and comm in {'python', 'python3', 'python3.10', 'tar', 'gzip', 'rsync', 'cp', 'mv'}):
                research_processes.append({'pid': int(folder.name), 'comm': comm, 'cwd': str(cwd)})
            for fd in (folder / 'fd').iterdir():
                try:
                    if str(fd.resolve()) in targets:
                        open_targets.append({'pid': int(folder.name), 'path': str(fd.resolve())})
                except (FileNotFoundError, PermissionError, OSError):
                    pass
        except (FileNotFoundError, PermissionError, OSError):
            pass
    assert not research_processes, 'Active research process; retain archives'
    assert not open_targets, 'Transfer archive has an open reader; retain archives'
    return dict(UTC=datetime.now(timezone.utc).isoformat(), GPU_compute_idle=True,
                tmux_stdout=tmux.stdout, tmux_stderr=tmux.stderr,
                research_processes=research_processes, open_archive_readers=open_targets)


def archive_check(row):
    path = Path(row['path'])
    assert path.parent == HOME and path.resolve() == path and path.name in TARGET_NAMES
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and not path.is_symlink() and s.st_uid == os.getuid() and s.st_nlink == 1
    for key, value in [('bytes', s.st_size), ('allocated_bytes', s.st_blocks*512), ('uid', s.st_uid),
                       ('nlink', s.st_nlink), ('inode', s.st_ino), ('mtime_ns', s.st_mtime_ns)]:
        assert row[key] == value, (str(path), key)
    assert sha(path) == row['sha256'], 'Remote archive does not match retained local backup'


def protected():
    metadata = []
    hashes = {}
    inode_hashes = {}
    for folder, dirs, files in os.walk(ROOT, followlinks=False):
        dirs.sort()
        files.sort()
        for name in dirs + files:
            path = Path(folder) / name
            s = path.lstat()
            rel = path.relative_to(ROOT).as_posix()
            metadata.append([rel, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_nlink, s.st_uid,
                             os.readlink(path) if stat.S_ISLNK(s.st_mode) else None])
            if stat.S_ISREG(s.st_mode) and '.venv' not in path.parts and path.suffix in EXTENSIONS:
                key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
                if key not in inode_hashes:
                    inode_hashes[key] = sha(path)
                hashes[rel] = inode_hashes[key]
            if len(metadata) % 50000 == 0:
                print(json.dumps({'protected_metadata_entries': len(metadata), 'protected_byte_hashes': len(hashes)}), flush=True)
    return dict(metadata=sorted(metadata), critical_and_evidence_hashes=hashes,
                all_scientific_cache_bytes_rehashed=False,
                preservation_proof='Exact nlink1 deletion outside research tree; all research metadata and critical byte hashes compared')


def protected_write(name, value):
    with gzip.open(OUT / name, 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--phase', choices=['verify', 'apply'], required=True)
    args = parser.parse_args()
    start = time.monotonic()
    assert Path.home().resolve() == HOME and ROOT.is_dir() and socket.gethostname().split('.')[0] == 'forensic-dgp-thesis' and os.getuid() == 1001
    meta = urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id', headers={'Metadata-Flavor': 'Google'}), timeout=5).read().decode()
    assert meta == '4410777042005672095' and sha(args.plan) == args.plan_sha
    plan = read(args.plan)
    assert plan['complete'] and plan['instance_id'] == meta and plan['remote_script_sha256'] == sha(Path(__file__))
    assert plan['research_root'] == str(ROOT) and plan['authorization_scope'] == 'inventory-first hash-bound inactive archive copies only'
    assert len(plan['candidates']) == 4 and {Path(row['path']).name for row in plan['candidates']} == TARGET_NAMES
    assert all(row['full_gzip_CRC_verified'] for row in plan['candidates'])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Maintenance900s stop; retain ledger')))
    signal.alarm(900)
    targets = {row['path'] for row in plan['candidates']}
    idle = workload(targets)
    for row in plan['candidates']:
        archive_check(row)
    if args.phase == 'verify':
        assert not OUT.exists(), 'Retain previous maintenance evidence'
        OUT.mkdir()
        before = protected()
        protected_write('protected_before.json.gz', before)
        write(OUT / 'verify_receipt.json', dict(complete=True, plan_sha256=args.plan_sha, candidate_count=4,
              protected_canonical_sha256=canonical(before), protected_file_sha256=sha(OUT / 'protected_before.json.gz'),
              protected_metadata_entries=len(before['metadata']), protected_byte_hashes=len(before['critical_and_evidence_hashes']),
              allocated_archive_bytes=plan['estimated_allocated_reclaim_bytes'], files_removed=0,
              runtime=idle, VM_started=False, model_gradient_or_training_calls=0, seconds=time.monotonic()-start))
        print(json.dumps(dict(complete=True, phase='verify', candidate_count=4, files_removed=0)), flush=True)
        return
    assert OUT.is_dir() and not (OUT / 'cleanup_receipt.json').exists() and not (OUT / 'deletion_ledger.jsonl').exists()
    verification = read(OUT / 'verify_receipt.json')
    assert verification['complete'] and verification['plan_sha256'] == args.plan_sha
    assert sha(OUT / 'protected_before.json.gz') == verification['protected_file_sha256']
    with gzip.open(OUT / 'protected_before.json.gz', 'rb') as stream:
        before = json.load(stream)
    assert canonical(before) == verification['protected_canonical_sha256']
    assert protected() == before, 'Protected state changed after verify; retain files'
    idle = workload(targets)
    for row in plan['candidates']:
        archive_check(row)
    disk_before = shutil.disk_usage(ROOT).free
    removed = []
    with (OUT / 'deletion_ledger.jsonl').open('x') as ledger:
        for row in plan['candidates']:
            workload(targets)
            archive_check(row)
            Path(row['path']).unlink()
            removed.append(row['path'])
            ledger.write(json.dumps(dict(path=row['path'], sha256=row['sha256'], backup_verified=True,
                                         allocated_bytes=row['allocated_bytes'], inode=row['inode'], nlink=row['nlink'])) + '\n')
            ledger.flush()
            os.fsync(ledger.fileno())
    after = protected()
    assert after == before, 'Research state differs; preserve deletion ledger'
    protected_write('protected_after.json.gz', after)
    disk_after = shutil.disk_usage(ROOT).free
    write(OUT / 'cleanup_receipt.json', dict(complete=True, plan_sha256=args.plan_sha, files_removed=4, removed_paths=removed,
          protected_canonical_before=canonical(before), protected_canonical_after=canonical(after),
          protected_metadata_entries=len(after['metadata']), protected_byte_hashes=len(after['critical_and_evidence_hashes']),
          all_scientific_cache_bytes_rehashed=False, disjoint_nlink1_home_only_deletion=True,
          free_before_bytes=disk_before, free_after_bytes=disk_after, recovered_bytes=disk_after-disk_before,
          allocated_archive_bytes=plan['estimated_allocated_reclaim_bytes'], runtime_before_removal=idle,
          research_assets_and_failures_unchanged=True, VM_started=False, model_gradient_or_training_calls=0,
          goal_complete=False, seconds=time.monotonic()-start))
    write(OUT / 'manifest.json', dict(complete=True, plan_sha256=args.plan_sha,
          files_sha256={file.name: sha(file) for file in sorted(OUT.iterdir()) if file.is_file()}))
    import tarfile
    archive = HOME / 'dgp-bank-archive-cleanup-v1-receipts.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in sorted(OUT.iterdir()):
            tar.add(file, arcname='receipts/' + file.name, recursive=False)
    write(HOME / 'dgp-bank-archive-cleanup-v1-export.json', dict(complete=True,
          archive_sha256=sha(archive), bytes=archive.stat().st_size, model_gradient_or_training_calls=0))
    print(json.dumps(dict(complete=True, phase='apply', files_removed=4, free_GiB=disk_after/1024**3)), flush=True)


if __name__ == '__main__':
    main()
