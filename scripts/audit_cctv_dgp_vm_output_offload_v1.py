"""Independently audit portable backup bytes and returned deletion/retention evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2'
BACKUP_NAME = 'dgp-image-output-recovery-20261010-v1.tar'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b''):
        h.update(chunk)
    return h.hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return digest(stream)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def audit_backup():
    started = time.monotonic()
    assert not (OUT / 'independent_backup_audit.json').exists()
    export = read(OUT / 'recovery_export.json')
    recovery = read(OUT / 'recovery_manifest.json')
    archive = OUT / BACKUP_NAME
    assert export['complete'] and archive.stat().st_size == export['bytes'] and sha(archive) == export['archive_sha256']
    assert sha(OUT / 'recovery_manifest.json') == export['recovery_manifest_sha256']
    assert export['candidate_count'] == recovery['candidate_count'] == len(recovery['candidates'])
    rows = {row['recovery_member']: row for row in recovery['candidates']}
    assert len(rows) == len(recovery['candidates'])
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers(); names = set()
        assert len(members) == len(rows) + 2
        for n, member in enumerate(members, 1):
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts and path.as_posix() == member.name
            assert ':' not in member.name and '\\' not in member.name and member.name not in names
            names.add(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            with tar.extractfile(member) as source:
                actual = digest(source)
            if member.name in rows:
                row = rows[member.name]
                assert member.size == row['metadata'][2] and actual == row['sha256'] == row['local_backup_sha256']
                assert member.name == 'forensic-dgp/' + row['relative_path']
                assert sha(ROOT / row['local_backup']) == row['sha256']
            elif member.name == '_recovery/recovery_manifest.json':
                assert actual == sha(OUT / 'recovery_manifest.json')
            else:
                assert member.name == '_recovery/restore_cctv_dgp_vm_output_offload_v1.py'
                assert actual == recovery['restore_script_sha256'] == sha(ROOT / 'scripts/restore_cctv_dgp_vm_output_offload_v1.py')
            if n % 10000 == 0:
                print(dict(backup_members_independently_verified=n, of=len(members)), flush=True)
        assert names == set(rows) | {'_recovery/recovery_manifest.json', '_recovery/restore_cctv_dgp_vm_output_offload_v1.py'}
    bindings = read(OUT / 'local_return_bindings.json')
    for name, expected in bindings.items():
        assert sha(ROOT / name) == expected, name
    result = dict(complete=True, checker_sha256=sha(Path(__file__)), archive_sha256=export['archive_sha256'],
        recovery_manifest_sha256=export['recovery_manifest_sha256'], candidate_count=len(rows),
        every_payload_member_hashed=True, complete_local_return_bindings_verified=len(bindings),
        portable_restore_helper_embedded=True, model_gradient_or_training_calls=0, files_removed=0,
        seconds=time.monotonic()-started)
    write(OUT / 'independent_backup_audit.json', result)
    print(dict(complete=True, portable_backup_outputs=len(rows), full_local_return_bindings=len(bindings)), flush=True)


def audit_cleanup():
    started = time.monotonic()
    destination = OUT / 'remote_receipts'
    assert not destination.exists() and not (OUT / 'independent_cleanup_audit.json').exists()
    plan = read(OUT / 'plan.json'); pin = sha(OUT / 'plan.json')
    execution = read(OUT / 'apply_execution.json')
    assert execution['complete'] and execution['plan_sha256'] == pin
    assert execution['driver_sha256'] == plan['local_driver_sha256'] == sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_v1.py')
    assert execution['remote_script_sha256'] == plan['remote_script_sha256'] == sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_v1_vm.py')
    assert sha(OUT / BACKUP_NAME) == plan['portable_backup_sha256']
    assert sha(OUT / 'independent_backup_audit.json') == plan['local_backup_audit_sha256']
    assert sha(OUT / 'recovery_manifest.json') == plan['recovery_manifest_sha256']
    export = read(OUT / 'dgp-output-offload-20261010-v1-export.json')
    archive = OUT / 'dgp-output-offload-20261010-v1-receipts.tar.gz'
    assert export['complete'] and export['model_gradient_or_training_calls'] == 0
    assert archive.stat().st_size == export['bytes'] and sha(archive) == export['archive_sha256']
    total = 0
    with gzip.open(archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            total += len(chunk); assert total < 256*1024**2
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); names = set()
        assert len(members) == 6 and sum(member.size for member in members) < 256*1024**2
        for member in members:
            path = PurePosixPath(member.name)
            assert len(path.parts) == 2 and path.parts[0] == 'receipts' and '..' not in path.parts
            assert member.isfile() and not member.islnk() and not member.issym()
            assert path.as_posix() == member.name and member.name not in names and ':' not in member.name and '\\' not in member.name
            names.add(member.name)
        with tar.extractfile('receipts/manifest.json') as stream:
            manifest = json.load(stream)
        assert manifest['complete'] and manifest['plan_sha256'] == pin
        assert names == {'receipts/' + name for name in manifest['files_sha256']} | {'receipts/manifest.json'}
        for name, expected in manifest['files_sha256'].items():
            with tar.extractfile('receipts/' + name) as stream:
                assert digest(stream) == expected, name
        destination.mkdir()
        for member in members:
            with tar.extractfile(member) as source, (destination / PurePosixPath(member.name).name).open('xb') as target:
                for chunk in iter(lambda: source.read(1024**2), b''):
                    target.write(chunk)
    with gzip.open(destination / 'protected_before.json.gz', 'rb') as stream:
        before = json.load(stream)
    with gzip.open(destination / 'protected_after.json.gz', 'rb') as stream:
        after = json.load(stream)
    verification = read(destination / 'verify_receipt.json'); receipt = read(destination / 'cleanup_receipt.json')
    assert verification['complete'] and receipt['complete']
    assert verification['plan_sha256'] == receipt['plan_sha256'] == pin and verification['files_removed'] == 0
    assert canonical(before) == verification['protected_canonical_sha256'] == receipt['protected_canonical_before']
    assert canonical(after) == receipt['protected_canonical_after']
    assert sha(destination / 'protected_before.json.gz') == verification['protected_file_sha256']
    rows = {row['relative_path']: row for row in plan['candidates']}; removed = set(rows)
    assert len(rows) == plan['candidate_count'] == verification['candidate_count'] == receipt['files_removed']
    left = {row[0]: row for row in before['metadata']}; right = {row[0]: row for row in after['metadata']}
    assert set(left) - set(right) == removed and not set(right) - set(left)
    touched = {PurePosixPath(rel).parent.as_posix() for rel in removed}
    for rel, row in right.items():
        original = left[rel]
        if rel in touched:
            assert stat.S_ISDIR(row[1]) and stat.S_ISDIR(original[1])
            assert [row[i] for i in (0, 1, 5, 6, 7, 8)] == [original[i] for i in (0, 1, 5, 6, 7, 8)]
        else:
            assert row == original, rel
    assert before['critical_and_evidence_hashes'] == after['critical_and_evidence_hashes']
    for row in plan['retained_non_image_arrays']:
        assert after['critical_and_evidence_hashes'][row['relative_path']] == row['sha256']
        assert sha(ROOT / row['local_backup']) == row['sha256']
    assert receipt['retention_check']['retained_metadata_entries'] == len(right)
    assert receipt['retention_check']['permitted_parent_directory_changes'] == len(touched)
    assert receipt['retention_check']['retained_critical_byte_hashes'] == len(after['critical_and_evidence_hashes'])
    ledger = [json.loads(line) for line in (destination / 'deletion_ledger.jsonl').read_text().splitlines()]
    assert len(ledger) == len(removed) and {row['relative_path'] for row in ledger} == removed
    allocated = 0
    for row in ledger:
        original = rows[row['relative_path']]
        assert left[row['relative_path']] == original['metadata']
        assert row['path'] == original['path'] == '/home/janusdominic0/forensic-dgp/' + row['relative_path']
        assert row['inode'] == original['metadata'][5] and row['nlink'] == 1 and row['uid'] == 1001
        assert stat.S_ISREG(original['metadata'][1]) and original['metadata'][6:8] == [1, 1001]
        assert row['backup_verified'] and row['sha256'] == original['sha256'] == sha(ROOT / row['local_backup'])
        assert row['portable_backup_sha256'] == plan['portable_backup_sha256']
        allocated += row['allocated_bytes']
    assert allocated == verification['allocated_output_bytes'] == receipt['allocated_output_bytes']
    assert receipt['free_after_bytes']-receipt['free_before_bytes'] == receipt['recovered_bytes']
    assert receipt['recovered_bytes'] >= allocated - 128*1024**2
    assert receipt['checkpoints_splits_caches_sources_logs_and_gate_bytes_unchanged']
    assert receipt['model_gradient_or_training_calls'] == 0 and not receipt['VM_started'] and not receipt['goal_complete']
    for runtime in [receipt['runtime_before_removal'], receipt['runtime_after_removal']]:
        assert runtime['GPU_compute_idle'] and not runtime['research_processes'] and not runtime['open_output_readers']
    for phase in ['verify', 'apply']:
        instance = read(OUT / ('instance_' + phase + '.json'))
        assert instance['id'] == plan['instance_id'] == '4410777042005672095' and instance['status'] == 'RUNNING'
        assert instance['machineType'].endswith('/g2-standard-4') and instance['guestAccelerators'][0]['acceleratorType'].endswith('/nvidia-l4')
    protected_file = ROOT / plan['local_protected_manifest']
    assert sha(protected_file) == plan['local_protected_manifest_sha256']
    protected = read(protected_file)
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    result = dict(complete=True, checker_sha256=sha(Path(__file__)), plan_sha256=pin,
        receipt_archive_sha256=export['archive_sha256'], portable_backup_sha256=plan['portable_backup_sha256'],
        output_copies_offloaded=len(ledger), every_removed_file_backed_up_and_reverified=True, shared_links_deleted=0,
        retained_research_metadata_entries=len(right), permitted_direct_parent_directory_updates=len(touched),
        retained_critical_and_evidence_byte_hashes=len(after['critical_and_evidence_hashes']),
        local_protected_bindings_unchanged=len(protected), all_scientific_cache_bytes_rehashed=False,
        scientific_cache_retention_proof='All non-candidate metadata exact; singly-linked image-output candidates disjoint from cache and source paths',
        observed_recovered_GiB=receipt['recovered_bytes']/1024**3, reported_free_after_GiB=receipt['free_after_bytes']/1024**3,
        model_gradient_or_training_calls=0, VM_started=False, goal_complete=False,
        fresh_post_inventory_required=True, seconds=time.monotonic()-started)
    write(OUT / 'independent_cleanup_audit.json', result)
    print({k: result[k] for k in ['complete', 'output_copies_offloaded', 'observed_recovered_GiB', 'reported_free_after_GiB']}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['backup', 'cleanup'], required=True)
    args = parser.parse_args()
    audit_backup() if args.phase == 'backup' else audit_cleanup()


if __name__ == '__main__':
    main()
