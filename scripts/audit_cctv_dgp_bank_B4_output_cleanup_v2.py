"""Independent maintenance closure: receipt namespace, exact ledger and backups."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_bank_B4_output_cleanup_v2'


def sha_stream(stream):
    h = hashlib.sha256()
    for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream: return sha_stream(stream)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    start = time.monotonic(); receipt_path = OUT/'independent_audit.json'; assert not receipt_path.exists()
    plan = read(OUT/'plan.json')
    worker = ROOT/'scripts/cctv_dgp_bank_B4_output_cleanup_v2_vm.py'
    assert plan['complete'] and plan['worker_sha256'] == sha(worker)
    export = read(OUT/'dgp-bank-B4-output-cleanup-v2-export.json')
    archive = OUT/'dgp-bank-B4-output-cleanup-v2-receipts.tar.gz'
    assert export['complete'] and archive.stat().st_size == export['bytes'] and sha(archive) == export['archive_sha256']
    wanted = {'verify_receipt.json','cleanup_receipt.json','deletion_ledger.jsonl','protected_before.json.gz','protected_after.json.gz','manifest.json'}
    dest = OUT/'remote_receipts'; assert not dest.exists(); dest.mkdir()
    seen = set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            value = PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not value.is_absolute() and value.parts[0] == 'receipts' and len(value.parts) == 2
            assert ':' not in member.name and '\\' not in member.name and '..' not in value.parts
            name = value.parts[1]; assert name in wanted and name not in seen
            with tar.extractfile(member) as source, (dest/name).open('xb') as target:
                for block in iter(lambda:source.read(1024**2),b''): target.write(block)
            assert (dest/name).stat().st_size == member.size; seen.add(name)
    assert seen == wanted
    manifest = read(dest/'manifest.json')
    assert manifest['complete'] and manifest['plan_sha256'] == sha(OUT/'plan.json')
    assert set(manifest['files_sha256']) == wanted-{'manifest.json'}
    assert all(sha(dest/name) == expected for name,expected in manifest['files_sha256'].items())
    verify, cleanup = read(dest/'verify_receipt.json'), read(dest/'cleanup_receipt.json')
    assert verify['complete'] and cleanup['complete'] and verify['files_removed'] == 0
    assert verify['plan_sha256'] == cleanup['plan_sha256'] == sha(OUT/'plan.json')
    assert verify['worker_sha256'] == cleanup['worker_sha256'] == sha(worker)
    with gzip.open(dest/'protected_before.json.gz','rb') as stream: before = json.load(stream)
    with gzip.open(dest/'protected_after.json.gz','rb') as stream: after = json.load(stream)
    assert before == after and canonical(before) == cleanup['protected_canonical_before'] == cleanup['protected_canonical_after']
    assert canonical(before) == verify['protected_canonical_sha256']
    assert sha(dest/'protected_before.json.gz') == verify['protected_sha256']
    planned = plan['candidates']
    ledger = [json.loads(line) for line in (dest/'deletion_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(ledger) == cleanup['files_removed'] == verify['candidate_count'] == len(planned) == 17347
    assert cleanup['removed_paths'] == [row['path'] for row in planned] == [row['path'] for row in ledger]
    backup_bytes = 0
    for original,actual in zip(planned,ledger):
        for key in ['path','sha256','local_backup','inode','nlink','allocated_bytes']: assert actual[key] == original[key]
        assert actual['backup_verified'] and actual['nlink'] == 1 and original['uid'] == 1001
        path = PurePosixPath(original['path'])
        if 'cctv_dgp_residual_epochs_vm_v42' in path.parts:
            assert path.parent.name in {'update0','update50'} and path.suffix == '.png'
        elif original['path'].startswith('/home/janusdominic0/forensic-dgp/cctv_dgp_bank_comparison_v1_r4_B_vm/'):
            assert path.parent.name in {'packs','native'} and path.suffix in {'.npz','.png','.npy'}
        else: assert original['path'] == '/home/janusdominic0/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz'
        backup = Path(actual['local_backup'])
        assert backup.resolve().is_relative_to(ROOT) and backup.is_file() and not backup.is_symlink()
        assert backup.stat().st_size == original['bytes'] and sha(backup) == actual['sha256']
        backup_bytes += original['bytes']
    assert backup_bytes == plan['exact_candidate_bytes']
    assert cleanup['allocated_reclaim_bytes'] == sum(row['allocated_bytes'] for row in planned)
    assert cleanup['recovered_bytes'] == cleanup['free_after_bytes']-cleanup['free_before_bytes']
    assert cleanup['runtime_before_removal']['GPU_compute_idle'] and not cleanup['runtime_before_removal']['active_research_processes']
    assert not cleanup['runtime_before_removal']['open_target_readers']
    assert not cleanup['all_scientific_cache_bytes_rehashed'] and not before['all_scientific_cache_bytes_rehashed']
    assert not cleanup['model_gradient_or_training_calls'] and not cleanup['VM_started']
    assert cleanup['checkpoints_inputs_splits_logs_metrics_provenance_and_failures_retained']
    assert cleanup['protected_metadata_entries'] == len(before['metadata'])
    assert cleanup['retained_bank_hashes'] == len(before['bank_retained_bytes_sha256'])
    # Retained bytes include B4 states plus V42 checkpoint/state/raw arrays;
    # candidate images are absent from protected metadata, all else is equal.
    protected_names = before['bank_retained_bytes_sha256']
    assert 'cctv_dgp_bank_comparison_v1_r4_B_vm/outputs/bank_comparison_v1/state_B50/full_state.pth' in protected_names
    assert 'cctv_dgp_residual_epochs_vm_v42/outputs/stopped_training_state.pt' in protected_names
    assert any(name.endswith('.npy') and 'cctv_dgp_residual_epochs_vm_v42/outputs/update50/' in name for name in protected_names)
    bindings = read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == expected for name,expected in bindings.items())
    result = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        plan_sha256=sha(OUT/'plan.json'),worker_sha256=sha(worker),receipt_archive_sha256=sha(archive),
        exact_deletions_checked=len(ledger),complete_local_backup_hashes_rechecked=len(ledger),
        local_candidate_backup_bytes=backup_bytes,protected_metadata_entries=len(before['metadata']),
        retained_bank_hashes=len(protected_names),retained_state_exactly_equal=True,
        allocated_reclaim_bytes=cleanup['allocated_reclaim_bytes'],observed_free_bytes_change=cleanup['recovered_bytes'],
        remote_receipt_free_after_bytes=cleanup['free_after_bytes'],all_scientific_cache_bytes_rehashed=False,
        V42_raw_arrays_and_full_states_retained=True,local_app_and_original_checkpoint_hashes_unchanged=True,
        model_gradient_or_training_calls=0,VM_started=False,manual_training_pending=True,goal_complete=False,
        seconds=time.monotonic()-start)
    with receipt_path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,exact_deletions=len(ledger),protected_metadata=len(before['metadata']),
        retained_hashes=len(protected_names),observed_recovered_GiB=cleanup['recovered_bytes']/1024**3,seconds=result['seconds']),flush=True)


if __name__ == '__main__': main()
