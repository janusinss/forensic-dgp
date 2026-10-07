"""Independent local receipt, backup-stamp and preserved-history verification."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2'
BACKUP = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1'
PIN = 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify():
    prep = read(OUT / 'preparation_r1.json')
    plan_path = OUT / 'inactive-feature-caches_plan_20261007_v2.json'
    plan = read(plan_path)
    assert sha(plan_path) == prep['plan_sha256']
    assert len(plan['files']) == 4429 and sum(row['bytes'] for row in plan['files']) == 39442892400
    assert all(row['kind'] == 'backed_up_inactive_feature_cache' for row in plan['files'])
    assert plan['scientific_cache_removal_authorized'] and not plan['clear_pip_download_cache']
    assert len(plan['protected_assets_sha256']) == 5757
    remote = OUT / 'remote_receipts'
    v, c = read(remote / 'verification.json'), read(remote / 'cleanup_receipt.json')
    protected = read(remote / 'protected_before.json')
    a = read(OUT / 'audit_r1.log')
    runtime = read(OUT / 'final_runtime.json')
    assert all(row['complete'] for row in (v, c, a, runtime))
    assert v['plan_sha256'] == c['plan_sha256'] == a['plan_sha256'] == sha(plan_path)
    assert v['script_sha256'] == c['script_sha256'] == a['backend_sha256'] == prep['backend_sha256']
    assert sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261007_v2_r1.py') == prep['backend_sha256']
    assert sha(ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261007_v2_r1.py') == prep['independent_auditor_sha256']
    for field, name in (('verification_sha256', 'verification.json'), ('cleanup_receipt_sha256', 'cleanup_receipt.json'),
                        ('deletions_sha256', 'deletions.jsonl'), ('protected_snapshot_sha256', 'protected_before.json')):
        assert a[field] == sha(remote / name)
    assert v['protected_snapshot_sha256'] == sha(remote / 'protected_before.json')
    assert c['inactive_cache_files_removed'] == c['files_removed'] == a['deleted_files_verified_absent'] == 4429
    assert c['inactive_cache_logical_bytes_removed'] == 39442892400
    assert c['archives_removed'] == c['duplicate_recognizer_files_removed'] == c['pip_cache_files_removed'] == 0
    assert c['protected_hashed_files_unchanged'] == a['protected_hashed_files_live_verified'] == len(protected['sha256'])
    assert c['scientific_tensor_files_unchanged'] == a['retained_tensor_stamps_live_verified'] == len(protected['scientific_tensor_stamps'])
    assert a['explicit_current_research_bindings_live_verified'] == 5757
    assert all(protected['sha256'][name] == digest for name, digest in plan['protected_assets_sha256'].items())
    ledger = [json.loads(line) for line in (remote / 'deletions.jsonl').read_text().splitlines()]
    expected = {row['path']: row for row in plan['files']}
    assert len(ledger) == len({row['path'] for row in ledger}) == len(expected)
    assert {row['path'] for row in ledger} == set(expected)
    for row in ledger:
        assert row['bytes'] == expected[row['path']]['bytes'] and row['local_backup'] == expected[row['path']]['local_backup']
    manifest = BACKUP / 'cache_source_inventory.json'
    assert sha(manifest) == PIN
    audit_path = OUT / 'local_backup_pre_apply_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['verified_cache_files'] == 4431
    assert audit['verified_cache_bytes'] == 39448585279 and audit['manifest_sha256'] == PIN
    stamps = read(OUT / 'local_backup_pre_apply_stamps.json')
    assert stamps['complete'] and stamps['full_sha256_audit_receipt_sha256'] == sha(audit_path)
    assert len(stamps['files']) == 4431 and sum(row['bytes'] for row in stamps['files']) == 39448585279
    frozen = {row['path']: row for row in read(manifest)['files']}
    assert {row['source'] for row in stamps['files']} == set(frozen)
    for row in stamps['files']:
        local = Path(row['path'])
        assert local.resolve().is_relative_to((BACKUP / 'cache_files').resolve()) and not local.is_symlink()
        info = local.stat()
        assert (info.st_size, info.st_mtime_ns, info.st_ino) == (row['bytes'], row['mtime_ns'], row['inode'])
        assert row['bytes'] == frozen[row['source']]['bytes'] and row['sha256'] == frozen[row['source']]['sha256']
    assert runtime['retained_cache_metadata_files'] == 2 and runtime['source_cache_data_files_verified_absent'] == 4429
    assert runtime['runtime']['CUDA_available'] and runtime['GPU'] == 'NVIDIA L4' and runtime['GPU_idle']
    assert runtime['free_bytes'] >= 6 * 1024**3 and c['free_bytes_increase'] >= 36 * 1024**3
    assert not any(row['training_started'] for row in (v, c, a, runtime, audit))
    assert not c['VM_stopped'] and not runtime['VM_stopped'] and runtime['model_loaded'] is False
    identity = read(OUT / 'source_instance_identity_after_SSH_stop.log')
    disk = read(OUT / 'source_disk_identity.log')
    assert identity['id'] == '4410777042005672095' and identity['status'] == 'RUNNING'
    assert disk['id'] == '2705643655052542111' and disk['sizeGb'] == '100'
    history = read(OUT / 'local_history_pre_cleanup_audit.json')
    assert history['complete'] and history['app22_bindings_verified']
    packet = read(OUT / 'V30_packet_fresh_audit.json')
    archive = ROOT / 'outputs/cctv-dgp-broader-mean-v30-execution.tar.gz'
    assert packet['complete'] and sha(archive) == packet['archive_sha256'] and packet['finite_updates'] == 800
    return {'complete': True, 'removed_inactive_cache_files': 4429,
            'removed_logical_bytes': 39442892400, 'initial_inventory_free_bytes': prep['free_bytes_before'],
            'measured_cleanup_free_bytes_increase': c['free_bytes_increase'],
            'final_live_free_bytes': runtime['free_bytes'],
            'actual_Windows_cache_files_preserved': 4431, 'actual_Windows_cache_bytes_preserved': 39448585279,
            'protected_VM_file_hashes_verified': len(protected['sha256']),
            'protected_VM_tensor_stamps_verified': len(protected['scientific_tensor_stamps']),
            'V30_dependency_bindings_verified': 5757, 'cache_metadata_files_retained_on_VM': 2,
            'CUDA_available': True, 'GPU_idle': True, 'training_started': False,
            'V30_installed': runtime['V30_installed'], 'V30_outputs_present': runtime['V30_outputs_present'],
            'goal_complete': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
