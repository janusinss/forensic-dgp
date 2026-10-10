"""Maintenance only: remove exact inactive output copies after local backup proof."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import stat
import subprocess
import tarfile
import time
import urllib.request

HOME = Path('/home/janusdominic0')
ROOT = HOME / 'forensic-dgp'
OUT = HOME / 'dgp_output_offload_20261010_v1_receipts'
GROUPS = {
    'cctv_dgp_head4_capacity_vm_v1', 'cctv_dgp_actual_step_review_v1_vm',
    'cctv_dgp_loss_cone_probe_v33_vm', 'cctv_dgp_profile_batches_vm_v31',
    'cctv_dgp_feature_fusion_vm_v32_r2', 'cctv_dgp_pcgrad_fit_vm_v41',
    'cctv_dgp_spatial_fit_vm_v40',
}
CRITICAL = {'.py', '.md', '.json', '.jsonl', '.pth', '.pt', '.onnx', '.sha256', '.log', '.csv', '.sh'}


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


def metadata(path):
    s = path.lstat()
    return [path.relative_to(ROOT).as_posix(), s.st_mode, s.st_size, s.st_mtime_ns,
            s.st_ctime_ns, s.st_ino, s.st_nlink, s.st_uid,
            os.readlink(path) if stat.S_ISLNK(s.st_mode) else None]


def workload(targets):
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), 'Active GPU work; retain output copies'
    tmux = subprocess.run(['tmux', 'list-panes', '-a', '-F',
                           '#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}|#{pane_dead}'],
                          capture_output=True, text=True, timeout=15)
    assert tmux.returncode == 0 or (tmux.returncode == 1 and 'No such file or directory' in tmux.stderr)
    for pane in tmux.stdout.splitlines():
        fields = pane.split('|')
        assert len(fields) == 6 and fields[2] in {'bash', 'sh', 'zsh', 'fish'} and fields[5] == '0', 'Active tmux task'
    active, readers = [], []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            comm = (folder / 'comm').read_text().strip()
            cmd = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = (folder / 'cwd').resolve()
            if (str(ROOT) + '/' in cmd and comm not in {'bash', 'sh', 'zsh', 'sshd'}) or (
                    cwd.is_relative_to(ROOT) and comm in {'python', 'python3', 'python3.10', 'tar', 'gzip', 'rsync', 'cp', 'mv'}):
                active.append(dict(pid=int(folder.name), comm=comm, cwd=str(cwd)))
            for fd in (folder / 'fd').iterdir():
                try:
                    path = str(fd.resolve())
                    if path in targets:
                        readers.append(dict(pid=int(folder.name), path=path))
                except (PermissionError, FileNotFoundError, OSError):
                    pass
        except (PermissionError, FileNotFoundError, OSError):
            pass
    assert not active and not readers, 'Active research work or open output file; retain copies'
    return dict(UTC=datetime.now(timezone.utc).isoformat(), GPU_compute_idle=True,
                tmux_stdout=tmux.stdout, tmux_stderr=tmux.stderr,
                research_processes=active, open_output_readers=readers)


def validate_scope(row):
    rel = PurePosixPath(row['relative_path'])
    assert not rel.is_absolute() and '..' not in rel.parts and len(rel.parts) >= 4
    assert rel.as_posix() == row['relative_path'] and rel.parts[0] in GROUPS and rel.parts[1] == 'outputs'
    assert rel.suffix in {'.png', '.npy', '.npz'} and 'embedding' not in rel.name
    assert not any(any(word in part.lower() for word in ('cache', 'gradient', 'direction')) for part in rel.parts[2:])
    assert row['payload_kind'] in {'PNG_image', 'raw_RGB_array', 'lossless_RGB_output_pack'}
    path = ROOT.joinpath(*rel.parts)
    assert path.resolve() == path and path.is_relative_to(ROOT) and str(path) == row['path']
    assert row['metadata'][0] == row['relative_path'] and row['metadata'][6:8] == [1, 1001]
    assert row['sha256'] == row['local_backup_sha256'] and row['local_complete_return_verified']
    return path


def candidate_check(row, hash_bytes):
    path = validate_scope(row)
    assert metadata(path) == row['metadata'], 'Frozen output metadata changed: ' + row['relative_path']
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1 and not path.is_symlink()
    if hash_bytes:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            opened = os.fstat(stream.fileno())
            assert (opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_nlink, opened.st_uid) == (
                s.st_ino, s.st_size, s.st_mtime_ns, 1, 1001)
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024**2), b''):
                h.update(chunk)
            assert h.hexdigest() == row['sha256'], 'Remote output differs from local recovery copy'
        assert metadata(path) == row['metadata']
    return s.st_blocks * 512


def protected(extra_array_paths):
    rows, hashes, inode_hashes = [], {}, {}
    for folder, dirs, files in os.walk(ROOT, followlinks=False):
        dirs.sort(); files.sort()
        for name in dirs + files:
            path = Path(folder) / name
            row = metadata(path)
            rows.append(row)
            s = path.lstat()
            if stat.S_ISREG(s.st_mode) and '.venv' not in path.parts and (path.suffix in CRITICAL or row[0] in extra_array_paths):
                key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
                if key not in inode_hashes:
                    inode_hashes[key] = sha(path)
                hashes[row[0]] = inode_hashes[key]
            if len(rows) % 100000 == 0:
                print(json.dumps(dict(protected_metadata_entries=len(rows), protected_byte_hashes=len(hashes))), flush=True)
    return dict(metadata=sorted(rows), critical_and_evidence_hashes=hashes,
                all_scientific_cache_bytes_rehashed=False,
                proof='Exact backed-up nlink1 image-output removal; all retained metadata and critical bytes compared')


def save_protected(name, value):
    with gzip.open(OUT / name, 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def retained_check(before, after, rows):
    removed = {row['relative_path'] for row in rows}
    touched = set()
    for rel in removed:
        touched.add(PurePosixPath(rel).parent.as_posix())
    left = {row[0]: row for row in before['metadata'] if row[0] not in removed}
    right = {row[0]: row for row in after['metadata']}
    assert set(left) == set(right), 'Unexpected retained/new/deleted research path'
    for rel, original in left.items():
        current = right[rel]
        if rel in touched:
            assert stat.S_ISDIR(original[1]) and stat.S_ISDIR(current[1])
            assert [original[i] for i in (0, 1, 5, 6, 7, 8)] == [current[i] for i in (0, 1, 5, 6, 7, 8)]
        else:
            assert original == current, 'Retained research metadata changed: ' + rel
    assert before['critical_and_evidence_hashes'] == after['critical_and_evidence_hashes'], 'Checkpoint, cache receipt, source or failure bytes differ'
    return dict(retained_metadata_entries=len(left), permitted_parent_directory_changes=len(touched),
                retained_critical_byte_hashes=len(after['critical_and_evidence_hashes']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--phase', choices=['verify', 'apply'], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert Path.home().resolve() == HOME and ROOT.is_dir() and os.getuid() == 1001
    assert socket.gethostname().split('.')[0] == 'forensic-dgp-thesis'
    instance = urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',
                                      headers={'Metadata-Flavor': 'Google'}), timeout=5).read().decode()
    assert instance == '4410777042005672095' and sha(args.plan) == args.plan_sha
    plan = read(args.plan)
    assert plan['complete'] and plan['instance_id'] == instance and plan['research_root'] == str(ROOT)
    assert plan['authorization_scope'] == 'inactive image-output copies offloaded to verified Windows recovery backup'
    assert plan['remote_script_sha256'] == sha(Path(__file__)) and plan['portable_backup_independently_verified']
    rows = plan['candidates']
    assert len(rows) == plan['candidate_count'] and {PurePosixPath(row['relative_path']).parts[0] for row in rows} == GROUPS
    assert len({row['path'] for row in rows}) == len(rows)
    assert len({row['metadata'][5] for row in rows}) == len(rows), 'Shared inode candidate'
    assert not {row['relative_path'] for row in rows}.intersection(plan['current_packet_VM_source_paths'])
    assert not (ROOT / 'cctv_dgp_bank_comparison_v1_vm').exists(), 'Current pilot already installed; retain historical outputs'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Maintenance900s stop; retain recovery backup and ledger')))
    signal.alarm(900)
    targets = {row['path'] for row in rows}
    extra_array_paths = {row['relative_path'] for row in plan['retained_non_image_arrays']}
    assert not extra_array_paths.intersection({row['relative_path'] for row in rows})
    idle = workload(targets)
    allocated = 0
    for n, row in enumerate(rows, 1):
        allocated += candidate_check(row, True)
        if n % 5000 == 0:
            print(json.dumps(dict(verified_output_copies=n, of=len(rows))), flush=True)
    if args.phase == 'verify':
        assert not OUT.exists(), 'Keep previous maintenance evidence'
        OUT.mkdir()
        before = protected(extra_array_paths)
        for row in plan['retained_non_image_arrays']:
            assert before['critical_and_evidence_hashes'][row['relative_path']] == row['sha256']
        save_protected('protected_before.json.gz', before)
        write(OUT / 'verify_receipt.json', dict(complete=True, plan_sha256=args.plan_sha, files_removed=0,
              candidate_count=len(rows), protected_canonical_sha256=canonical(before),
              protected_file_sha256=sha(OUT / 'protected_before.json.gz'),
              protected_metadata_entries=len(before['metadata']), protected_byte_hashes=len(before['critical_and_evidence_hashes']),
              output_payload_bytes=sum(row['metadata'][2] for row in rows), allocated_output_bytes=allocated,
              portable_backup_sha256=plan['portable_backup_sha256'], runtime=idle,
              model_gradient_or_training_calls=0, VM_started=False, seconds=time.monotonic()-started))
        print(json.dumps(dict(complete=True, phase='verify', files_removed=0, output_copies=len(rows), reclaim_GiB=allocated/1024**3)), flush=True)
        return
    verification = read(OUT / 'verify_receipt.json')
    assert verification['complete'] and verification['plan_sha256'] == args.plan_sha and verification['files_removed'] == 0
    assert verification['allocated_output_bytes'] == allocated and verification['portable_backup_sha256'] == plan['portable_backup_sha256']
    assert sha(OUT / 'protected_before.json.gz') == verification['protected_file_sha256']
    with gzip.open(OUT / 'protected_before.json.gz', 'rb') as stream:
        before = json.load(stream)
    assert canonical(before) == verification['protected_canonical_sha256']
    assert protected(extra_array_paths) == before, 'Research state changed since verification; retain copies'
    idle = workload(targets)
    disk_before = shutil.disk_usage(ROOT).free
    removed_count = 0
    with (OUT / 'deletion_ledger.jsonl').open('x') as ledger:
        for n, row in enumerate(rows, 1):
            if (n - 1) % 250 == 0:
                workload(targets)
            size = candidate_check(row, True)
            path = Path(row['path'])
            assert metadata(path) == row['metadata']
            path.unlink()
            removed_count += 1
            ledger.write(json.dumps(dict(path=row['path'], relative_path=row['relative_path'], sha256=row['sha256'],
                       local_backup=row['local_backup'], portable_backup_sha256=plan['portable_backup_sha256'],
                       allocated_bytes=size, inode=row['metadata'][5], nlink=1, uid=1001, backup_verified=True)) + '\n')
            if n % 250 == 0 or n == len(rows):
                ledger.flush(); os.fsync(ledger.fileno())
            if n % 5000 == 0:
                print(json.dumps(dict(offloaded_output_copies=n, of=len(rows))), flush=True)
    after = protected(extra_array_paths)
    result = retained_check(before, after, rows)
    assert all(not Path(row['path']).exists() for row in rows)
    save_protected('protected_after.json.gz', after)
    idle_after = workload(targets)
    disk_after = shutil.disk_usage(ROOT).free
    write(OUT / 'cleanup_receipt.json', dict(complete=True, plan_sha256=args.plan_sha, files_removed=removed_count,
          protected_canonical_before=canonical(before), protected_canonical_after=canonical(after),
          retention_check=result, all_scientific_cache_bytes_rehashed=False,
          output_payload_bytes=sum(row['metadata'][2] for row in rows), allocated_output_bytes=allocated,
          portable_backup_sha256=plan['portable_backup_sha256'], free_before_bytes=disk_before,
          free_after_bytes=disk_after, recovered_bytes=disk_after-disk_before,
          runtime_before_removal=idle, runtime_after_removal=idle_after,
          checkpoints_splits_caches_sources_logs_and_gate_bytes_unchanged=True,
          permitted_changes='Exact planned regular image-output files removed; their direct parent directory times/size change only',
          VM_started=False, model_gradient_or_training_calls=0, goal_complete=False, seconds=time.monotonic()-started))
    write(OUT / 'manifest.json', dict(complete=True, plan_sha256=args.plan_sha,
          files_sha256={file.name: sha(file) for file in sorted(OUT.iterdir()) if file.is_file()}))
    archive = HOME / 'dgp-output-offload-20261010-v1-receipts.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in sorted(OUT.iterdir()):
            tar.add(file, arcname='receipts/' + file.name, recursive=False)
    write(HOME / 'dgp-output-offload-20261010-v1-export.json', dict(complete=True, archive_sha256=sha(archive),
          bytes=archive.stat().st_size, model_gradient_or_training_calls=0))
    print(json.dumps(dict(complete=True, phase='apply', output_copies_offloaded=removed_count,
                         free_GiB=disk_after/1024**3, recovered_GiB=(disk_after-disk_before)/1024**3)), flush=True)


if __name__ == '__main__':
    main()
