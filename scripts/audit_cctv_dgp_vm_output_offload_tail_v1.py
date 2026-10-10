"""Independent byte/metadata audit of a maintenance timeout's non-deleting tail."""
import gzip
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time

import audit_cctv_dgp_vm_output_offload_v1 as audit

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2'
OUT = PARENT / 'tail_recovery_v1'
read, sha, write, canonical = audit.read, audit.sha, audit.write, audit.canonical


def main():
    started = time.monotonic()
    destination = OUT / 'remote_receipts'
    assert not destination.exists() and not (PARENT / 'independent_cleanup_audit.json').exists()
    original = read(PARENT / 'plan.json'); pin = sha(PARENT / 'plan.json')
    p = read(OUT / 'tail_plan.json'); execution = read(OUT / 'closure_execution.json')
    assert execution['complete'] and execution['tail_plan_sha256'] == sha(OUT / 'tail_plan.json')
    assert execution['original_plan_sha256'] == p['original_plan_sha256'] == pin
    assert execution['additional_files_removed'] == 0 and not execution['original_apply_complete']
    assert execution['driver_sha256'] == sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_tail_v1.py')
    assert p['recovery_worker_sha256'] == sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_tail_v1_vm.py')
    assert original['local_driver_sha256'] == sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_v1.py')
    assert original['remote_script_sha256'] == p['original_worker_sha256'] == sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_v1_vm.py')
    stopped = read(PARENT / 'transport/apply_transport.json'); stop = read(OUT / 'original_stop.json')
    assert not stopped['complete'] and not stop['original_apply_complete'] and stopped['exit_code'] == 1
    assert sha(OUT / 'original_stop.json') == p['original_apply_stop_sha256']
    assert sha(PARENT / 'transport/apply_transport.json') == p['original_apply_transport_sha256'] == stop['transport_sha256']
    assert sha(PARENT / 'transport/apply_stderr.log') == stop['stderr_sha256']
    assert sha(PARENT / 'transport/apply_stdout.log') == stop['stdout_sha256']
    assert 'TimeoutError: Maintenance900s stop' in (PARENT / 'transport/apply_stderr.log').read_text()
    assert sha(PARENT / 'dgp-image-output-recovery-20261010-v1.tar') == original['portable_backup_sha256']
    assert sha(PARENT / 'independent_backup_audit.json') == original['local_backup_audit_sha256']
    assert sha(PARENT / 'recovery_manifest.json') == original['recovery_manifest_sha256']
    export = read(OUT / 'dgp-output-offload-tail-v1-export.json')
    archive = OUT / 'dgp-output-offload-tail-v1-receipts.tar.gz'
    assert export['complete'] and export['additional_files_removed'] == 0 and export['model_gradient_or_training_calls'] == 0
    assert export['tail_plan_sha256'] == sha(OUT / 'tail_plan.json')
    assert archive.stat().st_size == export['bytes'] and sha(archive) == export['archive_sha256']
    total = 0
    with gzip.open(archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            total += len(chunk); assert total < 256*1024**2
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); names = set()
        assert len(members) == 6
        for member in members:
            path = PurePosixPath(member.name)
            assert len(path.parts) == 2 and path.parts[0] == 'receipts' and '..' not in path.parts
            assert member.isfile() and not member.islnk() and not member.issym() and member.name not in names
            assert path.as_posix() == member.name and ':' not in member.name and '\\' not in member.name
            names.add(member.name)
        with tar.extractfile('receipts/manifest.json') as stream:
            manifest = json.load(stream)
        assert manifest['complete'] and manifest['tail_plan_sha256'] == sha(OUT / 'tail_plan.json')
        assert names == {'receipts/' + name for name in manifest['files_sha256']} | {'receipts/manifest.json'}
        for name, expected in manifest['files_sha256'].items():
            with tar.extractfile('receipts/' + name) as stream:
                assert audit.digest(stream) == expected
        destination.mkdir()
        for member in members:
            with tar.extractfile(member) as source, (destination / PurePosixPath(member.name).name).open('xb') as target:
                for chunk in iter(lambda: source.read(1024**2), b''):
                    target.write(chunk)
    with gzip.open(destination / 'protected_before.json.gz', 'rb') as stream:
        before = json.load(stream)
    with gzip.open(destination / 'protected_after.json.gz', 'rb') as stream:
        after = json.load(stream)
    verify = read(destination / 'verify_receipt.json'); receipt = read(destination / 'audit_receipt.json')
    assert verify['complete'] and receipt['complete'] and verify['files_removed'] == 0 and receipt['additional_files_removed'] == 0
    assert verify['plan_sha256'] == receipt['original_plan_sha256'] == pin
    assert receipt['tail_plan_sha256'] == sha(OUT / 'tail_plan.json')
    assert canonical(before) == receipt['protected_canonical_before'] == verify['protected_canonical_sha256']
    assert canonical(after) == receipt['protected_canonical_after']
    assert sha(destination / 'protected_before.json.gz') == verify['protected_file_sha256']
    assert sha(destination / 'deletion_ledger.jsonl') == p['expected_ledger_sha256'] == read(OUT / 'inspection.json')['ledger_sha256']
    rows = {r['relative_path']: r for r in original['candidates']}
    ledger = [json.loads(line) for line in (destination / 'deletion_ledger.jsonl').read_text().splitlines()]
    removed = set(rows); left = {r[0]: r for r in before['metadata']}; right = {r[0]: r for r in after['metadata']}
    assert len(rows) == len(ledger) == receipt['original_output_copies_offloaded'] == 76973
    assert {r['relative_path'] for r in ledger} == removed
    assert set(left)-set(right) == removed and not set(right)-set(left)
    parents = {PurePosixPath(rel).parent.as_posix() for rel in removed}
    for rel, row in right.items():
        old = left[rel]
        if rel in parents:
            assert stat.S_ISDIR(row[1]) and stat.S_ISDIR(old[1])
            assert [row[i] for i in (0,1,5,6,7,8)] == [old[i] for i in (0,1,5,6,7,8)]
        else:
            assert row == old, rel
    assert before['critical_and_evidence_hashes'] == after['critical_and_evidence_hashes']
    for row in original['retained_non_image_arrays']:
        assert after['critical_and_evidence_hashes'][row['relative_path']] == row['sha256'] == sha(ROOT / row['local_backup'])
    allocated = 0
    for n, row in enumerate(ledger, 1):
        old = rows[row['relative_path']]
        assert left[row['relative_path']] == old['metadata']
        assert row['path'] == old['path'] == '/home/janusdominic0/forensic-dgp/' + row['relative_path']
        assert row['sha256'] == old['sha256'] == old['local_backup_sha256'] == sha(ROOT / row['local_backup'])
        assert row['inode'] == old['metadata'][5] and row['nlink'] == 1 and row['uid'] == 1001 and row['backup_verified']
        assert stat.S_ISREG(old['metadata'][1]) and old['metadata'][6:8] == [1,1001]
        assert row['portable_backup_sha256'] == original['portable_backup_sha256']
        allocated += row['allocated_bytes']
        if n % 20000 == 0:
            print(dict(offloaded_copies_independently_audited=n, of=len(ledger)), flush=True)
    assert allocated == receipt['allocated_removed_bytes'] == verify['allocated_output_bytes']
    assert receipt['retention_check']['retained_metadata_entries'] == len(right)
    assert receipt['retention_check']['permitted_parent_directory_changes'] == len(parents)
    assert receipt['retention_check']['retained_critical_byte_hashes'] == len(after['critical_and_evidence_hashes'])
    assert receipt['original_apply_stop_sha256'] == p['original_apply_stop_sha256'] and not receipt['original_apply_complete']
    assert receipt['original_apply_stopped_in_post_removal_hashing'] and receipt['model_gradient_or_training_calls'] == 0
    assert receipt['exact_immediate_pre_removal_free_bytes_unavailable'] and receipt['free_gain_includes_intervening_plan_and_receipt_storage']
    previous = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1/inventory_after.json'
    assert sha(previous) == p['previous_guest_inventory_sha256'] == receipt['previous_guest_inventory_sha256']
    assert read(previous)['disk']['free_bytes'] == receipt['previous_guest_free_bytes']
    assert receipt['observed_interval_free_gain_bytes'] == receipt['free_bytes_after_receipt_copies']-receipt['previous_guest_free_bytes']
    for runtime in [receipt['runtime_before'], receipt['runtime_after']]:
        assert runtime['GPU_compute_idle'] and not runtime['research_processes'] and not runtime['open_output_readers']
    for name in ['instance_inspect.json', 'instance_close.json']:
        instance = read(OUT / name)
        assert instance['id'] == original['instance_id'] == '4410777042005672095' and instance['status'] == 'RUNNING'
        assert instance['machineType'].endswith('/g2-standard-4') and instance['guestAccelerators'][0]['acceleratorType'].endswith('/nvidia-l4')
    assert sha(ROOT / original['local_protected_manifest']) == original['local_protected_manifest_sha256']
    protected = read(ROOT / original['local_protected_manifest'])
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    result = dict(complete=True, checker_sha256=sha(Path(__file__)), plan_sha256=pin,
        tail_plan_sha256=sha(OUT / 'tail_plan.json'), original_apply_complete=False,
        original_apply_timeout_preserved=True, closed_by_independent_post_removal_audit=True, additional_files_removed=0,
        receipt_archive_sha256=export['archive_sha256'], portable_backup_sha256=original['portable_backup_sha256'],
        output_copies_offloaded=len(ledger), every_removed_file_backed_up_and_reverified=True, shared_links_deleted=0,
        retained_research_metadata_entries=len(right), retained_critical_and_evidence_byte_hashes=len(after['critical_and_evidence_hashes']),
        permitted_direct_parent_directory_updates=len(parents), local_protected_bindings_unchanged=len(protected),
        all_scientific_cache_bytes_rehashed=False, allocated_removed_GiB=allocated/1024**3,
        observed_recovered_GiB=receipt['observed_interval_free_gain_bytes']/1024**3,
        free_gain_measurement='Net interval free-space change from prior guest inventory; includes intervening plan and receipt storage',
        reported_free_after_GiB=receipt['free_bytes_after_receipt_copies']/1024**3,
        model_gradient_or_training_calls=0, VM_started=False, goal_complete=False, fresh_post_inventory_required=True,
        seconds=time.monotonic()-started)
    write(PARENT / 'independent_cleanup_audit.json', result)
    print({key: result[key] for key in ['complete','output_copies_offloaded','retained_research_metadata_entries','allocated_removed_GiB']}, flush=True)


if __name__ == '__main__':
    main()
