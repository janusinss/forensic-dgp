"""Freeze an archive-only cleanup, retaining independently hashed Windows backups."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1'
HOME = PurePosixPath('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
PARENT = 'cctv_dgp_actual_step_review_v1_vm'
TAIL = 'cctv_dgp_actual_step_tail_v1_vm'
CURRENT_ARCHIVES = {'cctv-dgp-actual-step-review-v1-execution.tar.gz',
                    'cctv-dgp-actual-step-tail-v1-execution.tar.gz',
                    'cctv-dgp-actual-step-tail-v1-results.tar.gz'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after, 1)


def main():
    started = time.monotonic()
    assert not (OUT / 'archive-duplicates_plan_20261009_v1.json').exists()
    guest = read(OUT / 'inventory_initial/guest_inventory.json')
    assert guest['complete'] and not guest['gpu_processes']['stdout'].strip()
    assert guest['tmux']['exit_code'] in (0, 1) and not guest['tmux']['stdout'].strip()
    assert guest['current_diagnostic_metadata']['protocol_sha256'] == '339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5'
    assert not guest['tail_root_present'], 'Review an installed tail separately before planning'
    candidates = []
    skipped = []
    for row in sorted(guest['home_transfer_archive_metadata_only'], key=lambda row: row['path']):
        name = PurePosixPath(row['path']).name
        backup = ROOT / 'outputs' / name
        if name in CURRENT_ARCHIVES:
            skipped.append({'path': row['path'], 'reason': 'Retain current manual diagnostic packet'})
            continue
        assert PurePosixPath(row['path']).parent == HOME
        assert row['regular_file'] and not row['symlink'] and row['uid'] == guest['current_uid'] and row['nlink'] == 1
        if not backup.is_file():
            skipped.append({'path': row['path'], 'reason': 'No exact local archive backup'})
            continue
        info = backup.lstat()
        assert stat.S_ISREG(info.st_mode) and not backup.is_symlink() and info.st_size == row['bytes']
        checksum = backup.with_name(name + '.sha256')
        parts = checksum.read_text(encoding='utf-8').strip().split(maxsplit=1)
        assert len(parts) == 2 and parts[1].lstrip('*') == name
        digest = sha(backup)
        assert digest == parts[0]
        export = None
        if name.endswith('-results.tar.gz'):
            export = backup.with_name(name.replace('-results.tar.gz', '-export.json'))
            receipt = read(export)
            assert receipt['complete'] and receipt['archive_sha256'] == digest and receipt['bytes'] == info.st_size
        candidates.append({**row, 'sha256': digest, 'kind': 'duplicate_archive',
                           'local_backup': str(backup), 'local_backup_verified': True,
                           'local_backup_mtime_ns': info.st_mtime_ns,
                           'local_checksum': str(checksum), 'local_checksum_sha256': sha(checksum),
                           'local_export_receipt': str(export) if export else None,
                           'local_export_receipt_sha256': sha(export) if export else None})
        print({'Windows_archive_backup_verified': len(candidates), 'name': name}, flush=True)
    assert candidates
    protected = {}
    local_protected = {}

    def pin(remote, local, expected):
        assert PurePosixPath(remote).is_relative_to(HOME) and Path(local).is_file()
        assert sha(local) == expected, str(local)
        if remote in protected:
            assert protected[remote] == expected
        protected[remote] = expected
        local_protected[str(local)] = expected

    packet = ROOT / 'outputs' / PARENT
    returned = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm_return'
    tail_packet = ROOT / 'outputs' / TAIL
    parent_protocol = read(packet / 'protocol.json')
    tail_protocol = read(tail_packet / 'protocol.json')
    assert sha(tail_packet / 'protocol.json') == 'ec6981ce0e6961b765fb5dc1e82f191bc3511db2a1ca6b87e2849e83ea40eef9'
    for relative, digest in parent_protocol['assets_sha256'].items():
        pin(str(REPO / PARENT / relative), packet / relative, digest)
    for table in ('retained_parent_return_sha256', 'overlap_files_sha256'):
        for relative, digest in tail_protocol[table].items():
            pin(str(REPO / PARENT / relative), returned / relative, digest)
    for name in sorted(CURRENT_ARCHIVES):
        existing = next((row for row in guest['home_transfer_archive_metadata_only'] if PurePosixPath(row['path']).name == name), None)
        if existing:
            local = ROOT / 'outputs' / name
            pin(str(HOME / name), local, sha(local))
            local_sum = local.with_name(name + '.sha256')
            pin(str(HOME / (name + '.sha256')), local_sum, sha(local_sum))
    write(OUT / 'local_protected_bindings.json', local_protected)

    old_backend = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261007_v1.py'
    backend_path = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261009_v1.py'
    old_auditor = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py'
    auditor_path = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261009_v1.py'
    assert not backend_path.exists() and not auditor_path.exists()
    backend = old_backend.read_text(encoding='utf-8').replace('20261007_v1', '20261009_v1').replace('20261007-v1', '20261009-v1')
    backend = replace_once(backend, 'Two frozen phases: duplicate archives/pip cache, then inactive feature-cache files.',
                           'One frozen phase: top-level home archives with exact retained Windows backups.')
    backend = replace_once(backend, "'maintenance_storage_20261006_v2', 'maintenance_storage_20261009_v1'",
                           "'maintenance_storage_20261006_v2', 'maintenance_storage_20261007_v1', 'maintenance_storage_20261007_v2', 'maintenance_storage_20261009_v1'")
    backend = replace_once(backend, "any(mode in fields[2] for mode in ['--run', '--preflight', '--export'])", 'True')
    backend = replace_once(backend, "any(v in path.name for v in ('broader-mean-v30',))",
                           "any(v in path.name for v in ('actual-step-review-v1-execution', 'actual-step-tail-v1'))")
    backend = replace_once(backend, "        if row['kind'] == 'duplicate_archive':",
                           "        if row['kind'] != 'duplicate_archive' or path.parent != HOME or not path.name.startswith('cctv-dgp-'):\n            raise ValueError('This phase authorizes only exact top-level duplicate CCTV archives')\n        if any(getattr(info, attribute) != row[key] for attribute, key in [('st_uid','uid'),('st_nlink','nlink'),('st_ino','inode'),('st_mtime_ns','mtime_ns')]):\n            raise ValueError('Candidate differs from the fresh inventory')\n        if row['kind'] == 'duplicate_archive':")
    backend = replace_once(backend, "if plan['maintenance_id'] not in {'archive-duplicates', 'inactive-feature-caches'}:",
                           "if plan['maintenance_id'] != 'archive-duplicates' or plan['clear_pip_download_cache']:")
    backend = backend.replace('current_V30_dependencies_and_all_trained_DGP_checkpoints_preserved',
                              'current_diagnostic_dependencies_and_all_trained_DGP_checkpoints_preserved')
    old_tree = ast.parse(old_backend.read_text(encoding='utf-8'))
    new_tree = ast.parse(backend)
    for name in ('checked_file', 'file_stamp', 'sha'):
        before = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        after = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(before) == ast.dump(after), 'Retain the proven file safety checks'
    assert backend.count('path.unlink()') == 1 and '.rmtree(' not in backend
    with backend_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(backend)
    auditor = old_auditor.read_text(encoding='utf-8').replace('20261007_v1', '20261009_v1')
    auditor = replace_once(auditor, "choices=('archive-duplicates', 'inactive-feature-caches')", "choices=('archive-duplicates',)")
    auditor = replace_once(auditor, "    disk = shutil.disk_usage('/')",
                           "    require(not plan['clear_pip_download_cache'] and not verify['pip_cache_files'], 'No cache deletion authorized')\n    require(all(row['kind'] == 'duplicate_archive' and Path(row['path']).parent == HOME for row in verify['archives']), 'Top-level archive-only scope differs')\n    require((REPO / 'cctv_dgp_vm_bundle/.venv/bin/python').is_file() and os.access(REPO / 'cctv_dgp_vm_bundle/.venv/bin/python', os.X_OK), 'Shared VM Python runtime unavailable')\n    disk = shutil.disk_usage('/')")
    auditor = replace_once(auditor, "'GPU_idle': True,", "'GPU_idle': True, 'shared_VM_python_available': True, 'manual_tail_free_requirement_met': disk.free >= 2 * 1024**3,")
    ast.parse(auditor)
    with auditor_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(auditor)
    write(OUT / 'source_adaptation.json', {
        'complete': True, 'original_backend_sha256': sha(old_backend), 'backend_sha256': sha(backend_path),
        'original_auditor_sha256': sha(old_auditor), 'auditor_sha256': sha(auditor_path),
        'owned_regular_non_symlink_single_link_checks_AST_unchanged': True,
        'single_exact_unlink_retained': True, 'recursive_delete_calls': 0,
        'scope_narrowed_to_top_level_backed_archives': True,
        'current_diagnostic_protected_instead_of_old_V30_label': True,
        'other_CCTV_Python_work_blocks_cleanup': True})
    plan = {'format': 'user-authorized-verified-backed-VM-storage-cleanup-20261009-v1',
            'UTC': datetime.now(timezone.utc).isoformat(), 'maintenance_id': 'archive-duplicates',
            'home': str(HOME), 'repo': str(REPO), 'pip_download_cache': str(HOME / '.cache/pip'),
            'clear_pip_download_cache': False, 'training_launch_authorized': False,
            'original_checkpoints_splits_logs_preserved': True, 'maintenance_cap_seconds': 1800,
            'authorization': 'Do a clean up then ill manual input the Finish the final state of the storage-stopped diagnostic',
            'inventory_sha256': sha(OUT / 'inventory_initial/guest_inventory.json'),
            'files': candidates, 'skipped': skipped, 'protected_assets_sha256': protected,
            'current_manual_diagnostic': TAIL, 'parent_diagnostic_root': str(REPO / PARENT),
            'parent_overlap_files_exactly_preserved': 828,
            'current_execution_packets_retained': True, 'all_local_backups_retained': True}
    write(OUT / 'archive-duplicates_plan_20261009_v1.json', plan)
    write(OUT / 'local_backup_verification.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(OUT / 'archive-duplicates_plan_20261009_v1.json'),
        'archives': len(candidates), 'logical_bytes': sum(row['bytes'] for row in candidates),
        'allocated_bytes_VM_inventory': sum(row['allocated_bytes'] for row in candidates),
        'current_explicit_protected_assets': len(protected), 'all_local_backups_retained': True,
        'file_deletions': 0, 'model_or_gradient_calls': 0, 'seconds': time.monotonic() - started})
    print({'prepared': True, 'archive_copies': len(candidates),
           'potential_reclaim_GiB': sum(row['allocated_bytes'] for row in candidates) / 1024**3,
           'protected_current_bindings': len(protected), 'files_removed': 0}, flush=True)


if __name__ == '__main__':
    main()
