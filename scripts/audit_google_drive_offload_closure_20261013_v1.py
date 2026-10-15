"""Independent plan, Drive backup, exact removal and whole-project audit."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/google_drive_offload_20261012_v1'
OUT = PARENT / 'closure_20261013_v1'
PLAN_PIN = '2390ae75ca39b6a60383931b2d5fc658459a6051b4a640ef7cf608a1f38f716f'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def frozen_evidence(pin):
    assert sha(PARENT / 'plan.json') == PLAN_PIN
    plan = read(PARENT / 'plan.json')
    removal = read(OUT / 'removal_plan.json')
    assert sha(OUT / 'removal_plan.json') == pin and removal['complete']
    assert removal['plan_sha256'] == PLAN_PIN
    assert sha(ROOT / 'scripts/close_google_drive_offload_20261013_v1.py') == removal['closure_driver_sha256']
    assert sha(OUT / 'cloud_inventory_before.json') == removal['cloud_namespace_sha256']
    listing = read(OUT / 'cloud_inventory_before.json')
    namespace = {row['Path']: row for row in listing}
    assert len(namespace) == len(listing) == 4457 and len({row['ID'] for row in listing}) == len(listing)
    assert set(namespace) == {batch + '/' + row['path'] for batch, data in plan['batches'].items() for row in data['files']}
    for batch, data in plan['batches'].items():
        for row in data['files']:
            actual = namespace[batch + '/' + row['path']]
            hashes = {name.lower(): value for name, value in actual['Hashes'].items()}
            assert actual['Size'] == row['bytes'] and hashes['sha256'] == row['sha256']
        for mode in ['upload', 'verify']:
            paths = removal['manual_operations'][batch + '_' + mode]
            assert paths
            for relative in paths:
                path = ROOT / relative
                receipt = read(path)
                assert receipt['Complete'] and receipt['RcloneExitCode'] == 0 and receipt['PlanSHA256'] == PLAN_PIN
                assert receipt['ExpectedFiles'] == data['file_count'] and receipt['ExpectedBytes'] == data['upload_bytes']
                if mode == 'verify':
                    assert receipt['DownloadedSHA256Verification']
                    lines = (path.parent / 'combined.txt').read_text(encoding='utf-8').splitlines()
                    assert Counter(lines) == Counter('= ' + row['path'] for row in data['files'])
    expected_candidates = {row['path']: row for data in plan['batches'].values() for row in data['files'] if row['role'].startswith('inactive')}
    actual_candidates = {row['path']: row for row in removal['candidates']}
    assert set(expected_candidates) == set(actual_candidates) and len(actual_candidates) == 4432
    assert sum(row['metadata']['bytes'] for row in actual_candidates.values()) == 52028568639
    source = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1/cache_source_inventory.json'
    assert sha(source) == 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'
    source_rows = read(source)['files']
    source_hashes = {'outputs/cctv_dgp_local_research_cache_backup_20261007_v1/cache_files' + row['path']: row['sha256'] for row in source_rows}
    candidate_cache = {name: row['sha256'] for name, row in actual_candidates.items() if '/cache_files/' in name}
    assert candidate_cache == source_hashes
    for relative, row in actual_candidates.items():
        assert row['sha256'] == row['cloud_sha256'] == expected_candidates[relative]['sha256']
        assert row['metadata']['nlink'] == 1 and row['cloud_id']
        assert relative not in removal['protected_hashes']
    for relative, expected in removal['receipt_hashes'].items():
        assert sha(ROOT / relative) == expected
    return plan, removal, namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['before', 'after'])
    parser.add_argument('--removal-plan-sha', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan, removal, namespace = frozen_evidence(args.removal_plan_sha)
    target = OUT / ('independent_' + args.phase + '_audit.json')
    assert not target.exists()
    for relative, expected in removal['protected_hashes'].items():
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink() and sha(path) == expected, relative
    candidates = {row['path']: row for row in removal['candidates']}
    if args.phase == 'before':
        assert not (OUT / 'apply_receipt.json').exists()
        for relative, row in candidates.items():
            path = ROOT / relative
            info = path.stat()
            assert path.is_file() and not path.is_symlink() and info.st_size == row['metadata']['bytes']
            assert info.st_ino == row['metadata']['inode'] and info.st_nlink == 1 and info.st_mtime_ns == row['metadata']['mtime_ns']
        assert sha(OUT / 'namespace_before.json.gz') == removal['namespace_before_sha256']
        result = dict(complete=True, phase='before', verified_cloud_files=len(namespace),
                      user_streamed_sha256_receipts_audited=True, current_cloud_hashes_match=True,
                      candidate_namespace_exact=True, candidate_count=len(candidates),
                      logical_candidate_bytes=52028568639, active_hashes_verified=len(removal['protected_hashes']),
                      files_removed=0, training_calls=0)
    else:
        applied = read(OUT / 'apply_receipt.json')
        assert applied['complete'] and applied['removal_plan_sha256'] == args.removal_plan_sha
        assert sha(OUT / 'deletion_ledger.jsonl') == applied['deletion_ledger_sha256']
        events = [json.loads(line) for line in (OUT / 'deletion_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        removed = [row for row in events if row['event'] == 'removed']
        intent = [row for row in events if row['event'] == 'removal_intent']
        assert len(removed) == len(intent) == len(candidates)
        assert Counter(row['path'] for row in removed) == Counter(candidates)
        for row in removed:
            frozen = candidates[row['path']]
            assert not (ROOT / row['path']).exists()
            assert row['sha256'] == frozen['sha256'] and row['cloud_id'] == frozen['cloud_id']
            assert row['bytes'] == frozen['metadata']['bytes'] and row['allocated_bytes'] == frozen['allocated_bytes']
            assert row['exclusive_handle_hash_verified'] and row['native_exact_file_removal']
        assert sha(OUT / 'namespace_before.json.gz') == removal['namespace_before_sha256']
        after_receipt = read(OUT / 'namespace_after_receipt.json')
        assert after_receipt['complete'] and sha(OUT / 'namespace_after.json.gz') == after_receipt['SHA256']
        with gzip.open(OUT / 'namespace_before.json.gz', 'rt', encoding='utf-8') as stream:
            before = json.load(stream)
        with gzip.open(OUT / 'namespace_after.json.gz', 'rt', encoding='utf-8') as stream:
            after = json.load(stream)
        assert set(before) - set(after) == set(candidates), 'Unexpected lost file'
        assert not set(after) - set(before), 'Concurrent new file needs explicit audit'
        differences = [relative for relative in after if after[relative] != before[relative]]
        assert not differences, ('Retained metadata changed; preserve and review', differences[:10])
        assert sum(row['bytes'] for row in removed) == applied['logical_bytes_removed'] == 52028568639
        assert sum(row['allocated_bytes'] for row in removed) == applied['allocated_bytes_removed']
        result = dict(complete=True, phase='after', exact_files_removed=len(removed),
            logical_bytes_removed=applied['logical_bytes_removed'], allocated_bytes_removed=applied['allocated_bytes_removed'],
            observed_free_increase_bytes=applied['observed_free_increase_bytes'], free_bytes_at_apply_end=applied['free_after_bytes'],
            retained_namespace_files=len(after), unexpected_missing_files=0, retained_metadata_differences=0,
            current_active_hashes_verified=len(removal['protected_hashes']), downloaded_backup_and_cloud_hash_evidence_preserved=True,
            directories_removed=0, cloud_files_removed=0, VM_calls=0, training_calls=0, research_goal_complete=False)
    result.update(removal_plan_sha256=args.removal_plan_sha, checker_sha256=sha(Path(__file__)), seconds=time.monotonic()-started)
    save(target, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
