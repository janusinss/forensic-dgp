"""Bind fresh VM inventory to independently verified actual Windows cache copies."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2'
BACKUP = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1'
PIN = 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    start = time.monotonic()
    source = BACKUP / 'cache_source_inventory.json'
    assert sha(source) == PIN
    frozen = read(source)
    live = read(OUT / 'inventory_before.json')
    audit_path = OUT / 'local_backup_fresh_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['verified_cache_files'] == 4431
    assert audit['verified_cache_bytes'] == 39448585279 and audit['manifest_sha256'] == PIN
    assert live['complete'] and live['GPU_idle'] and live['GPU'] == 'NVIDIA L4'
    assert live['hostname'] == 'forensic-dgp-thesis' and live['files_removed'] == 0
    assert len(live['files']) == len(frozen['files']) == 4431
    expected = {row['path']: (row['bytes'], row['sha256']) for row in frozen['files']}
    assert {row['path']: (row['bytes'], row['sha256']) for row in live['files']} == expected
    prior = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/archive-duplicates_plan_20261007_v1.json'
    dependency = read(OUT / 'V30_local_dependency_audit.json')
    assert dependency['complete'] and dependency['prior_plan_sha256'] == sha(prior)
    bindings = read(prior)['protected_assets_sha256']
    assert len(bindings) == 5757
    rows = []
    for row in live['files']:
        if not row['cleanup_candidate']:
            assert Path(row['path']).name == 'state.json'
            continue
        local = BACKUP / 'cache_files' / row['path'].lstrip('/')
        assert local.is_file() and not local.is_symlink() and local.stat().st_size == row['bytes']
        assert local.resolve().is_relative_to((BACKUP / 'cache_files').resolve())
        rows.append({**row, 'kind': 'backed_up_inactive_feature_cache',
                     'local_backup': str(local.resolve()), 'local_backup_verified': True})
    assert len(rows) == 4429 and sum(row['bytes'] for row in rows) == 39442892400
    assert not set(bindings) & {row['path'] for row in rows}
    home = '/home/janusdominic0'
    plan = {'format': 'user-authorized-verified-backed-VM-storage-cleanup-20261007-v2',
            'maintenance_id': 'inactive-feature-caches', 'home': home, 'repo': home + '/forensic-dgp',
            'pip_download_cache': home + '/.cache/pip', 'clear_pip_download_cache': False,
            'maintenance_cap_seconds': 1800, 'training_launch_authorized': False,
            'original_checkpoints_splits_logs_preserved': True,
            'scientific_cache_removal_authorized': True, 'current_V30_dependencies_retained': True,
            'local_backup_full_sha256_audit_complete': True, 'local_backup_manifest_sha256': PIN,
            'local_backup_audit_sha256': sha(audit_path), 'local_backup_audit': str(audit_path),
            'files': rows, 'protected_assets_sha256': bindings,
            'backup_policy': 'Fresh SHA256 verification of all actual Windows and VM cache files; exact inactive .bin/.npz unlinks only; cache state.json metadata retained',
            'source_inventory_sha256': sha(OUT / 'inventory_before.json'),
            'V30_protocol_sha256': dependency['V30_protocol_sha256'], 'created_unix': time.time()}
    plan_path = OUT / 'inactive-feature-caches_plan_20261007_v2.json'
    write(plan_path, plan)
    manifest = OUT / 'cache_manifest_for_cleanup_20261007_v2.json'
    with manifest.open('xb') as stream:
        stream.write(source.read_bytes())
    receipt = {'complete': True, 'plan_sha256': sha(plan_path), 'manifest_sha256': sha(manifest),
               'backend_sha256': sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261007_v2.py'),
               'independent_auditor_sha256': sha(ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261007_v2.py'),
               'backup_files_verified': 4431, 'cache_files_planned': len(rows),
               'cache_bytes_planned': sum(row['bytes'] for row in rows),
               'cache_allocated_bytes_planned': sum(row['allocated_bytes'] for row in rows),
               'cache_metadata_files_retained': 2, 'V30_dependency_bindings': 5757,
               'free_bytes_before': live['free_bytes'], 'files_removed': 0, 'training_started': False,
               'seconds': time.monotonic() - start}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
