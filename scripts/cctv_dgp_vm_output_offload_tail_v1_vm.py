"""Audit a completed removal ledger after a timed stop; no deletion operation."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import tarfile
import time
import urllib.request

HOME = Path('/home/janusdominic0')
ROOT = HOME / 'forensic-dgp'
OLD = HOME / 'dgp_output_offload_20261010_v1_receipts'
OUT = HOME / 'dgp_output_offload_tail_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    a = parser.parse_args(); started = time.monotonic()
    assert Path.home().resolve() == HOME and ROOT.is_dir() and os.getuid() == 1001
    assert socket.gethostname().split('.')[0] == 'forensic-dgp-thesis'
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id', headers={'Metadata-Flavor': 'Google'})
    assert urllib.request.urlopen(request, timeout=5).read().decode() == '4410777042005672095'
    assert sha(a.plan) == a.plan_sha
    recovery = read(a.plan)
    assert recovery['complete'] and not recovery['additional_deletions_permitted'] and recovery['cap_seconds'] == 900
    assert sha(Path(__file__)) == recovery['recovery_worker_sha256']
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Audit-only900s stop; preserve all records')))
    signal.alarm(900)
    original_plan = HOME / 'dgp_output_offload_20261010_v1_plan.json'
    original_worker = HOME / 'cctv_dgp_vm_output_offload_v1_vm.py'
    assert sha(original_plan) == recovery['original_plan_sha256']
    assert sha(original_worker) == recovery['original_worker_sha256']
    p = read(original_plan)
    assert p['remote_script_sha256'] == recovery['original_worker_sha256'] and p['candidate_count'] == 76973
    spec = importlib.util.spec_from_file_location('frozen_maintenance_definitions', original_worker)
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    assert not OUT.exists() and not (OLD / 'cleanup_receipt.json').exists() and not (OLD / 'protected_after.json.gz').exists()
    verify = read(OLD / 'verify_receipt.json')
    assert verify['complete'] and verify['plan_sha256'] == recovery['original_plan_sha256'] and verify['files_removed'] == 0
    assert sha(OLD / 'protected_before.json.gz') == verify['protected_file_sha256']
    with gzip.open(OLD / 'protected_before.json.gz', 'rb') as stream:
        before = json.load(stream)
    assert frozen.canonical(before) == verify['protected_canonical_sha256']
    ledger_path = OLD / 'deletion_ledger.jsonl'
    s = ledger_path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == 1001 and s.st_nlink == 1 and not ledger_path.is_symlink()
    assert sha(ledger_path) == recovery['expected_ledger_sha256']
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rows = {r['relative_path']: r for r in p['candidates']}
    assert len(rows) == len(ledger) == recovery['candidate_count'] == 76973
    assert {r['relative_path'] for r in ledger} == set(rows)
    metadata = {r[0]: r for r in before['metadata']}
    allocated = 0
    for row in ledger:
        original = rows[row['relative_path']]
        path = frozen.validate_scope(original)
        assert not path.exists() and not path.is_symlink()
        assert metadata[row['relative_path']] == original['metadata']
        assert row['path'] == original['path'] and row['sha256'] == original['sha256'] == original['local_backup_sha256']
        assert row['local_backup'] == original['local_backup'] and row['backup_verified']
        assert row['portable_backup_sha256'] == p['portable_backup_sha256']
        assert row['inode'] == original['metadata'][5] and row['nlink'] == 1 and row['uid'] == 1001
        assert stat.S_ISREG(original['metadata'][1]) and original['metadata'][6:8] == [1, 1001]
        allocated += row['allocated_bytes']
    assert allocated == verify['allocated_output_bytes']
    assert not (ROOT / 'cctv_dgp_bank_comparison_v1_vm').exists()
    targets = {r['path'] for r in rows.values()}
    idle_before = frozen.workload(targets)
    arrays = {r['relative_path'] for r in p['retained_non_image_arrays']}
    after = frozen.protected(arrays)
    retention = frozen.retained_check(before, after, p['candidates'])
    for row in p['retained_non_image_arrays']:
        assert after['critical_and_evidence_hashes'][row['relative_path']] == row['sha256']
    idle_after = frozen.workload(targets)
    OUT.mkdir()
    for source_name in ['protected_before.json.gz', 'verify_receipt.json', 'deletion_ledger.jsonl']:
        shutil.copyfile(OLD / source_name, OUT / source_name)
        assert sha(OUT / source_name) == sha(OLD / source_name)
    with gzip.open(OUT / 'protected_after.json.gz', 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(after, sort_keys=True, allow_nan=False).encode())
    free = shutil.disk_usage(ROOT).free
    receipt = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        original_plan_sha256=recovery['original_plan_sha256'], tail_plan_sha256=a.plan_sha,
        original_apply_complete=False, original_apply_stopped_in_post_removal_hashing=True,
        original_apply_transport_sha256=recovery['original_apply_transport_sha256'],
        original_apply_stop_sha256=recovery['original_apply_stop_sha256'],
        additional_files_removed=0, original_output_copies_offloaded=len(ledger), allocated_removed_bytes=allocated,
        expected_ledger_sha256=recovery['expected_ledger_sha256'],
        protected_canonical_before=frozen.canonical(before), protected_canonical_after=frozen.canonical(after),
        retention_check=retention, retained_non_image_arrays=len(arrays),
        retained_checkpoint_split_source_log_gate_hashes_unchanged=True, all_retained_cache_metadata_unchanged=True,
        all_scientific_cache_bytes_rehashed=False, runtime_before=idle_before, runtime_after=idle_after,
        free_bytes_after_receipt_copies=free, previous_guest_free_bytes=recovery['previous_guest_free_bytes'],
        previous_guest_inventory_sha256=recovery['previous_guest_inventory_sha256'], previous_inventory_UTC=recovery['previous_inventory_UTC'],
        observed_interval_free_gain_bytes=free-recovery['previous_guest_free_bytes'],
        free_gain_includes_intervening_plan_and_receipt_storage=True, exact_immediate_pre_removal_free_bytes_unavailable=True,
        model_gradient_or_training_calls=0, VM_started=False, goal_complete=False, seconds=time.monotonic()-started)
    write(OUT / 'audit_receipt.json', receipt)
    write(OUT / 'manifest.json', dict(complete=True, tail_plan_sha256=a.plan_sha,
        files_sha256={f.name: sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}))
    archive = HOME / 'dgp-output-offload-tail-v1-receipts.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in sorted(OUT.iterdir()):
            tar.add(file, arcname='receipts/' + file.name, recursive=False)
    write(HOME / 'dgp-output-offload-tail-v1-export.json', dict(complete=True, bytes=archive.stat().st_size,
        archive_sha256=sha(archive), tail_plan_sha256=a.plan_sha, additional_files_removed=0,
        model_gradient_or_training_calls=0, original_apply_complete=False, seconds=time.monotonic()-started))
    print(json.dumps(dict(complete=True, post_removal_audit=True, original_output_copies_offloaded=len(ledger),
          retained_metadata=retention['retained_metadata_entries'], allocated_offloaded_GiB=allocated/1024**3,
          free_GiB=free/1024**3, additional_files_removed=0)), flush=True)


if __name__ == '__main__':
    main()
