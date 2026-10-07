"""Separate read-only post-cleanup verification on the named VM; no removal/model call."""
import argparse
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
BASE = REPO / 'maintenance_storage_20261007_v2'
START = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)


def clock():
    require(time.monotonic() - START < 1800, 'Independent read-only audit exceeds30 minutes')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            clock(); digest.update(block)
    return digest.hexdigest()


def stamp(path):
    value = path.stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns, 'inode': value.st_ino,
            'allocated_bytes': value.st_blocks * 512}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('archive-duplicates', 'inactive-feature-caches'), required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--backend-sha', required=True)
    args = parser.parse_args()
    require(sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis' and
            Path.home().resolve() == HOME, 'Only named existing VM/user')
    phase = BASE / args.phase
    plan_path = HOME / (args.phase + '_plan_20261007_v2.json')
    require(sha(plan_path) == args.plan_sha, 'Plan differs')
    require(sha(HOME / 'cleanup_cctv_dgp_vm_storage_20261007_v2.py') == args.backend_sha, 'Backend differs')
    plan = json.loads(plan_path.read_text())
    verify = json.loads((phase / 'verification.json').read_text())
    cleanup = json.loads((phase / 'cleanup_receipt.json').read_text())
    protected_path = phase / 'protected_before.json'
    require(sha(protected_path) == verify['protected_snapshot_sha256'], 'Protected snapshot differs')
    protected = json.loads(protected_path.read_text())
    require(verify['complete'] and cleanup['complete'] and verify['plan_sha256'] == cleanup['plan_sha256'] == args.plan_sha,
            'Verification/apply receipts differ')
    require(verify['script_sha256'] == cleanup['script_sha256'] == args.backend_sha, 'Receipt backend differs')
    require(cleanup['training_started'] is False and cleanup['VM_stopped'] is False, 'Cleanup exceeded authorization')
    deleted = [json.loads(row) for row in (phase / 'deletions.jsonl').read_text().splitlines()]
    expected = verify['archives'] + verify['pip_cache_files']
    require(len(deleted) == len(expected) == cleanup['files_removed'] and
            len({row['path'] for row in deleted}) == len(deleted), 'Deletion count/uniqueness differs')
    expected_map = {row['path']: row for row in expected}
    require({row['path'] for row in deleted} == set(expected_map), 'Deletion scope differs')
    for row in deleted:
        path = Path(row['path'])
        require(path.is_absolute() and path.is_relative_to(HOME) and str(path) == row['path'], 'Deleted path leaves approved home')
        require(row['bytes'] == expected_map[row['path']]['bytes'] and
                row.get('local_backup') == expected_map[row['path']].get('local_backup') and
                not os.path.lexists(path), 'Deletion ledger/absence differs')
    require(cleanup['archives_removed'] == sum(row['kind'] == 'duplicate_archive' for row in verify['archives']) and
            cleanup['inactive_cache_files_removed'] == sum(row['kind'] == 'backed_up_inactive_feature_cache' for row in verify['archives']), 'Per-family count differs')
    require(cleanup['duplicate_recognizer_files_removed'] == sum(row['kind'] == 'backed_up_duplicate_recognizer' for row in verify['archives']) and cleanup['duplicate_recognizer_logical_bytes_removed'] == sum(row['bytes'] for row in verify['archives'] if row['kind'] == 'backed_up_duplicate_recognizer'), 'Duplicate recognizer count/bytes differ')
    for name, digest in protected['sha256'].items():
        clock(); path = Path(name)
        require(path.is_absolute() and path.is_relative_to(HOME) and not path.is_symlink() and
                path.is_file() and sha(path) == digest, 'Protected research file differs: ' + name)
    for name, expected_stamp in protected['scientific_tensor_stamps'].items():
        clock(); path = Path(name)
        require(path.is_relative_to(REPO) and not path.is_symlink() and path.is_file() and stamp(path) == expected_stamp,
                'Retained scientific tensor changed: ' + name)
    require(cleanup['protected_hashed_files_unchanged'] == len(protected['sha256']) and
            cleanup['scientific_tensor_files_unchanged'] == len(protected['scientific_tensor_stamps']), 'Protected-count claim differs')
    require(set(plan['protected_assets_sha256']).issubset(protected['sha256']) and
            all(protected['sha256'][name] == digest for name, digest in plan['protected_assets_sha256'].items()), 'Explicit current research bindings differ')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=20)
    require(not gpu.strip(), 'A GPU task started during maintenance')
    disk = shutil.disk_usage('/')
    report = {'complete': True, 'scope': 'Independent live read-only absence/hash/stamp audit after exact backed-file removal',
        'phase': args.phase, 'plan_sha256': args.plan_sha, 'backend_sha256': args.backend_sha,
        'verification_sha256': sha(phase / 'verification.json'), 'cleanup_receipt_sha256': sha(phase / 'cleanup_receipt.json'),
        'deletions_sha256': sha(phase / 'deletions.jsonl'), 'protected_snapshot_sha256': sha(protected_path),
        'deleted_files_verified_absent': len(deleted), 'protected_hashed_files_live_verified': len(protected['sha256']),
        'retained_tensor_stamps_live_verified': len(protected['scientific_tensor_stamps']),
        'explicit_current_research_bindings_live_verified': len(plan['protected_assets_sha256']),
        'free_bytes_live': disk.free, 'total_bytes_live': disk.total, 'used_bytes_live': disk.used,
        'GPU_idle': True, 'files_removed_by_this_audit': 0, 'training_started': False, 'seconds': time.monotonic() - START}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
