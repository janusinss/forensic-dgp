"""Free VM storage only after exact Windows backup verification.

Two frozen phases: duplicate archives/pip cache, then inactive feature-cache files.
Checkpoints, source, splits, logs and data stay on the VM. No training or restart.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
MAINTENANCE_ROOT = REPO / 'maintenance_storage_20261006_v1'
RECORDS = None
CACHE_ROOTS = (REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/anatomical_cache',
               REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/fixed_cache',
               REPO / 'cctv_dgp_broader_codes_vm_v16_r2/outputs/broader_codes_v16_r2/cache')
TIME_LIMIT = 900
PIP = HOME / '.cache/pip'
FORMAT = 'user-authorized-verified-backed-VM-storage-cleanup-20261006-v1'
START = time.monotonic()


def clock():
    if time.monotonic() - START > TIME_LIMIT:
        raise TimeoutError('Finite maintenance deadline exceeded')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            clock()
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def checked_file(name):
    path = Path(name)
    if not path.is_absolute() or str(path) != name or not path.is_relative_to(HOME):
        raise ValueError('Absolute canonical path within the named VM home required')
    for component in [path] + list(path.parents):
        if component == HOME.parent:
            break
        if component.is_symlink():
            raise ValueError('No symlink paths permitted: ' + name)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise ValueError('Require owned regular non-hardlinked file: ' + name)
    if path.resolve(strict=True) != path:
        raise ValueError('Resolved path differs: ' + name)
    return path, info


def file_stamp(info):
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'inode': info.st_ino, 'allocated_bytes': info.st_blocks * 512}


def idle():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=20)
    if gpu.stdout.strip():
        raise RuntimeError('GPU task active; no cleanup or process termination')
    panes = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{pane_current_command}\t#{pane_dead}'],
                           capture_output=True, text=True, timeout=20)
    if panes.returncode not in [0, 1]:
        raise RuntimeError('Cannot verify tmux process state')
    for line in panes.stdout.splitlines():
        command, dead = line.split('\t')
        if dead != '1' and command not in {'bash', 'sh', 'zsh', 'fish', 'tail', 'less', 'watch', 'top', 'htop', 'tmux'}:
            raise RuntimeError('Active tmux program; no cleanup: ' + command)
    processes = subprocess.run(['ps', '-eo', 'pid,comm,args'], capture_output=True, text=True,
                               check=True, timeout=20)
    for line in processes.stdout.splitlines()[1:]:
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        if fields[1].startswith(('python', 'pip', 'apt', 'conda')) and any(
                marker in fields[2] for marker in ['train_', 'supervise_', 'pip install', 'pip download', 'pip cache', 'conda install']):
            raise RuntimeError('Training/package task active; preserve it')
        if fields[1].startswith('python') and 'cctv_dgp' in fields[2] and any(mode in fields[2] for mode in ['--run', '--preflight', '--export']):
            raise RuntimeError('Current pilot work active; preserve it')


def protected_snapshot(plan):
    expected = dict(plan['protected_assets_sha256'])
    cleanup_targets = {row['path'] for row in plan['files']}
    extra = {}
    metadata_suffixes = {'.pth', '.pt', '.onnx', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.sha256', '.log'}
    tensors = {}
    for directory, names, files in os.walk(REPO, followlinks=False):
        names[:] = [name for name in names if name not in {'.git', '.venv', 'venv', 'maintenance_storage_20261005', 'maintenance_storage_20261005_r2', 'maintenance_storage_20261006_v1'}
                    and not (Path(directory) / name).is_symlink()]
        for name in files:
            clock()
            path = Path(directory) / name
            if path.is_symlink():
                continue
            if path.suffix in {'.npz', '.bin'}:
                if str(path) not in cleanup_targets:
                    tensors[str(path)] = file_stamp(path.stat())
            if path.suffix in metadata_suffixes or name == 'cuda_runtime_before.txt':
                extra[str(path)] = None
    for name in extra:
        expected.setdefault(name, None)
    result = {}
    for index, (name, digest) in enumerate(sorted(expected.items()), 1):
        clock()
        path = Path(name)
        if not path.is_relative_to(HOME) or path.is_symlink() or not path.is_file():
            raise ValueError('Protected file missing/changed: ' + name)
        actual = sha(path)
        if digest is not None and actual != digest:
            raise ValueError('Protected pinned asset differs before cleanup: ' + name)
        result[name] = actual
        if index % 2500 == 0:
            print(json.dumps({'protected_assets_verified': index}), flush=True)
    return {'sha256': result, 'scientific_tensor_stamps': tensors}


def verify(plan, pin):
    if RECORDS.exists():
        raise RuntimeError('Preserve existing maintenance evidence; no automatic repeat')
    idle()
    protected = protected_snapshot(plan)
    checked = []
    if not plan['files'] or len({row['path'] for row in plan['files']}) != len(plan['files']):
        raise ValueError('Exact nonempty backed-file inventory required')
    for index, row in enumerate(plan['files'], 1):
        path, info = checked_file(row['path'])
        if row['path'] in protected['sha256'] or any(v in path.name for v in ('detail-prior-v22', 'detail-skip-v23')):
            raise ValueError('Protected scientific/current target: ' + str(path))
        if row['kind'] == 'duplicate_archive':
            if path.suffixes[-2:] != ['.tar', '.gz']:
                raise ValueError('Only exact duplicate tar.gz archives')
        elif row['kind'] == 'backed_up_inactive_feature_cache':
            if not any(path.is_relative_to(root) for root in CACHE_ROOTS) or path.suffix not in {'.bin', '.npz'}:
                raise ValueError('Only backed inactive .bin/.npz files in three named cache roots')
        else:
            raise ValueError('Unknown cleanup file kind')
        if not row.get('local_backup') or not row.get('local_backup_verified'):
            raise ValueError('Exact Windows backup evidence required')
        if info.st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Archive differs from verified Windows backup: ' + str(path))
        checked.append({**row, **file_stamp(info)})
        if index % 20 == 0:
            print(json.dumps({'VM_backed_files_matched': index, 'total': len(plan['files'])}), flush=True)
    if plan['clear_pip_download_cache'] and (PIP.resolve(strict=True) != PIP or PIP.is_symlink()):
        raise ValueError('Pip cache path differs')
    cache = []
    for directory, names, files in (os.walk(PIP, followlinks=False) if plan['clear_pip_download_cache'] else []):
        if any((Path(directory) / name).is_symlink() for name in names):
            raise ValueError('Unexpected symlink in pip cache; preserve it')
        for name in files:
            path, info = checked_file(str(Path(directory) / name))
            cache.append({'path': str(path), **file_stamp(info)})
    RECORDS.mkdir(parents=True)
    write_new(RECORDS / 'protected_before.json', protected)
    disk = shutil.disk_usage('/')
    receipt = {'complete': True, 'mode': 'verify_only', 'plan_sha256': pin,
        'script_sha256': sha(Path(__file__)), 'archives': checked, 'pip_cache_files': cache,
        'archive_bytes': sum(row['bytes'] for row in checked if row['kind'] == 'duplicate_archive'),
        'inactive_cache_bytes': sum(row['bytes'] for row in checked if row['kind'] == 'backed_up_inactive_feature_cache'),
        'pip_cache_bytes': sum(row['bytes'] for row in cache),
        'protected_hashed_files': len(protected['sha256']),
        'scientific_tensor_files_retained': len(protected['scientific_tensor_stamps']),
        'protected_snapshot_sha256': sha(RECORDS / 'protected_before.json'),
        'disk_free_bytes': disk.free, 'files_removed': 0, 'training_started': False,
        'seconds': time.monotonic() - START}
    write_new(RECORDS / 'verification.json', receipt)
    print(json.dumps({key: receipt[key] for key in ['complete', 'mode', 'archive_bytes', 'pip_cache_bytes',
        'protected_hashed_files', 'scientific_tensor_files_retained', 'disk_free_bytes', 'files_removed', 'seconds']}, indent=2), flush=True)


def apply(plan, pin):
    if (RECORDS / 'cleanup_receipt.json').exists() or (RECORDS / 'deletions.jsonl').exists():
        raise RuntimeError('Preserve existing cleanup evidence; no repeat')
    receipt = json.loads((RECORDS / 'verification.json').read_text())
    if not receipt['complete'] or receipt['plan_sha256'] != pin or receipt['script_sha256'] != sha(Path(__file__)):
        raise ValueError('Verified maintenance plan/script changed')
    if sha(RECORDS / 'protected_before.json') != receipt['protected_snapshot_sha256']:
        raise ValueError('Protected snapshot changed')
    idle()
    # Recheck every intended file and verified backup binding before the first unlink.
    for row in receipt['archives'] + receipt['pip_cache_files']:
        path, info = checked_file(row['path'])
        if file_stamp(info) != {key: row[key] for key in file_stamp(info)}:
            raise ValueError('Intended file changed since verification: ' + str(path))
        if path.is_relative_to(PIP):
            continue
        if row['path'] not in {item['path'] for item in plan['files']} or sha(path) != row['sha256']:
            raise ValueError('Archive backup binding changed')
    before = shutil.disk_usage('/')
    count = 0
    with (RECORDS / 'deletions.jsonl').open('x', encoding='utf-8') as log:
        for row in receipt['archives'] + receipt['pip_cache_files']:
            clock()
            path, info = checked_file(row['path'])
            if file_stamp(info) != {key: row[key] for key in file_stamp(info)}:
                raise ValueError('File changed immediately before unlink')
            path.unlink()  # Exact validated regular file only; never recursive deletion.
            count += 1
            log.write(json.dumps({'path': row['path'], 'bytes': row['bytes'],
                'local_backup': row.get('local_backup'), 'removed_unix': time.time()}) + '\n')
            log.flush()
    protected_before = json.loads((RECORDS / 'protected_before.json').read_text())
    protected_after = protected_snapshot(plan)
    if protected_before != protected_after:
        raise ValueError('Protected research/runtime metadata or tensors changed; retain evidence')
    after = shutil.disk_usage('/')
    final = {'complete': True, 'date_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': pin, 'script_sha256': sha(Path(__file__)), 'archives_removed': sum(row['kind'] == 'duplicate_archive' for row in receipt['archives']),
        'inactive_cache_files_removed': sum(row['kind'] == 'backed_up_inactive_feature_cache' for row in receipt['archives']),
        'inactive_cache_logical_bytes_removed': receipt['inactive_cache_bytes'],
        'pip_cache_files_removed': len(receipt['pip_cache_files']), 'files_removed': count,
        'archive_logical_bytes_removed': receipt['archive_bytes'],
        'pip_cache_logical_bytes_removed': receipt['pip_cache_bytes'],
        'free_bytes_before': before.free, 'free_bytes_after': after.free,
        'free_bytes_increase': after.free - before.free,
        'total_bytes': after.total, 'used_bytes_after': after.used,
        'protected_hashed_files_unchanged': len(protected_after['sha256']),
        'scientific_tensor_files_unchanged': len(protected_after['scientific_tensor_stamps']),
        'original_checkpoints_sources_splits_logs_preserved': True,
        'current_V22_V23_inputs_weights_outputs_archives_preserved': True, 'all_local_backups_retained': True,
        'training_started': False, 'VM_stopped': False,
        'seconds': time.monotonic() - START}
    write_new(RECORDS / 'cleanup_receipt.json', final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha', required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--verify-only', action='store_true')
    choice.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or os.uname().nodename.split('.')[0] != 'forensic-dgp-thesis' or Path.home().resolve() != HOME:
        raise RuntimeError('Only the named existing Linux VM/user is authorized')
    path = args.plan.resolve(strict=True)
    if not path.is_relative_to(HOME) or sha(path) != args.expected_plan_sha:
        raise ValueError('Plan path/hash differs')
    plan = json.loads(path.read_text())
    if (plan['format'] != FORMAT or plan['home'] != str(HOME) or plan['repo'] != str(REPO)
            or plan['pip_download_cache'] != str(PIP) or plan['training_launch_authorized']
            or not plan['original_checkpoints_splits_logs_preserved']):
        raise ValueError('User-authorized scope differs')
    if plan['maintenance_id'] not in {'archive-duplicates', 'inactive-feature-caches'}:
        raise ValueError('Unknown frozen maintenance phase')
    RECORDS = MAINTENANCE_ROOT / plan['maintenance_id']
    TIME_LIMIT = plan['maintenance_cap_seconds']
    if type(TIME_LIMIT) is not int or not 600 <= TIME_LIMIT <= 1800:
        raise ValueError('Finite 600-1800 second maintenance cap required')
    if args.verify_only:
        verify(plan, args.expected_plan_sha)
    else:
        apply(plan, args.expected_plan_sha)
