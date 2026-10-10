"""Independent returned-maintenance audit using saved bytes, without model calls."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1'
NAMES = {'cctv-dgp-multiscale-calibration-v1-execution.tar.gz', 'cctv-dgp-multiscale-calibration-v1-results.tar.gz', 'dgp-head4-archive-cleanup-v1-receipts.tar.gz', 'dgp-multiscale-archive-cleanup-v1-receipts.tar.gz'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def gzip_json(path):
    chunks = []
    size = 0
    with gzip.open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            size += len(chunk)
            assert size < 256*1024**2
            chunks.append(chunk)
    return json.loads(b''.join(chunks))


def main():
    started = time.monotonic()
    destination = OUT / 'remote_receipts'
    assert not destination.exists() and not (OUT / 'independent_audit.json').exists()
    plan = read(OUT / 'plan.json')
    pin = sha(OUT / 'plan.json')
    execution = read(OUT / 'apply_execution.json')
    assert execution['complete'] and execution['plan_sha256'] == pin
    assert execution['driver_sha256'] == sha(ROOT / 'scripts/run_cctv_dgp_bank_archive_cleanup_v1.py')
    assert execution['remote_script_sha256'] == plan['remote_script_sha256'] == sha(ROOT / 'scripts/cctv_dgp_bank_archive_cleanup_v1_vm.py')
    assert plan['preparation_script_sha256'] == sha(ROOT / 'scripts/prepare_cctv_dgp_bank_archive_cleanup_v1.py')
    export = read(OUT / 'dgp-bank-archive-cleanup-v1-export.json')
    archive = OUT / 'dgp-bank-archive-cleanup-v1-receipts.tar.gz'
    assert export['complete'] and export['model_gradient_or_training_calls'] == 0
    assert archive.stat().st_size == export['bytes'] and sha(archive) == export['archive_sha256']
    total = 0
    with gzip.open(archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            total += len(chunk)
            assert total < 256*1024**2
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == 6 and sum(row.size for row in members) < 256*1024**2
        names = set()
        for row in members:
            path = PurePosixPath(row.name)
            assert row.isfile() and not row.issym() and not row.islnk() and row.size >= 0
            assert len(path.parts) == 2 and path.parts[0] == 'receipts' and '..' not in path.parts
            assert row.name == path.as_posix() and ':' not in row.name and '\\' not in row.name and row.name not in names
            names.add(row.name)
        with tar.extractfile('receipts/manifest.json') as stream:
            manifest = json.load(stream)
        assert manifest['complete'] and manifest['plan_sha256'] == pin
        assert names == {'receipts/' + name for name in manifest['files_sha256']} | {'receipts/manifest.json'}
        for name, digest in manifest['files_sha256'].items():
            with tar.extractfile('receipts/' + name) as stream:
                h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    h.update(chunk)
            assert h.hexdigest() == digest, name
        destination.mkdir()
        for row in members:
            with tar.extractfile(row) as source, (destination / PurePosixPath(row.name).name).open('xb') as target:
                for chunk in iter(lambda: source.read(1024**2), b''):
                    target.write(chunk)
    before = gzip_json(destination / 'protected_before.json.gz')
    after = gzip_json(destination / 'protected_after.json.gz')
    assert before == after
    verification = read(destination / 'verify_receipt.json')
    receipt = read(destination / 'cleanup_receipt.json')
    digest = canonical(before)
    assert verification['complete'] and receipt['complete']
    assert verification['plan_sha256'] == receipt['plan_sha256'] == pin
    assert verification['protected_canonical_sha256'] == receipt['protected_canonical_before'] == receipt['protected_canonical_after'] == digest
    assert sha(destination / 'protected_before.json.gz') == verification['protected_file_sha256']
    assert verification['files_removed'] == 0 and receipt['files_removed'] == 4
    assert verification['candidate_count'] == 4 and receipt['disjoint_nlink1_home_only_deletion']
    assert len(before['metadata']) == verification['protected_metadata_entries'] == receipt['protected_metadata_entries']
    assert len(before['critical_and_evidence_hashes']) == verification['protected_byte_hashes'] == receipt['protected_byte_hashes']
    assert before['all_scientific_cache_bytes_rehashed'] is False and receipt['all_scientific_cache_bytes_rehashed'] is False
    ledger = [json.loads(line) for line in (destination / 'deletion_ledger.jsonl').read_text().splitlines()]
    targets = {row['path']: row for row in plan['candidates']}
    assert len(ledger) == len(targets) == 4
    assert {Path(path).name for path in targets} == NAMES
    assert {row['path'] for row in ledger} == set(targets) == set(receipt['removed_paths'])
    for row in ledger:
        original = targets[row['path']]
        assert original['regular_file'] and not original['symlink'] and original['uid'] == 1001 and original['nlink'] == 1
        assert Path(row['path']).parent.as_posix() == '/home/janusdominic0'
        assert row['nlink'] == 1 and row['inode'] == original['inode'] and row['backup_verified']
        assert row['sha256'] == original['sha256'] == sha(ROOT / original['local_backup'])
        assert row['allocated_bytes'] == original['allocated_bytes'] and original['full_gzip_CRC_verified']
    assert receipt['allocated_archive_bytes'] == plan['estimated_allocated_reclaim_bytes'] == sum(row['allocated_bytes'] for row in ledger)
    assert receipt['free_after_bytes'] - receipt['free_before_bytes'] == receipt['recovered_bytes']
    assert receipt['recovered_bytes'] >= plan['estimated_allocated_reclaim_bytes'] - 64*1024**2
    for phase in ['verify', 'apply']:
        state = read(OUT / ('instance_' + phase + '.json'))
        assert state['id'] == plan['instance_id'] == '4410777042005672095' and state['status'] == 'RUNNING'
        assert state['machineType'].endswith('/g2-standard-4') and state['guestAccelerators'][0]['acceleratorType'].endswith('/nvidia-l4')
    assert not receipt['VM_started'] and receipt['model_gradient_or_training_calls'] == 0 and receipt['research_assets_and_failures_unchanged']
    assert receipt['runtime_before_removal']['GPU_compute_idle'] and not receipt['runtime_before_removal']['research_processes']
    assert not receipt['runtime_before_removal']['open_archive_readers']
    local_bindings = read(OUT / 'local_protected_sha256.json')
    assert sha(OUT / 'local_protected_sha256.json') == plan['local_protected_manifest_sha256']
    assert len(local_bindings) == plan['local_protected_bindings']
    for name, expected in local_bindings.items():
        assert sha(ROOT / name) == expected, name
    packet = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_vm'
    protocol = read(packet / 'protocol.json')
    assert sha(packet / 'protocol.json') == plan['current_packet_protocol_sha256']
    assert sha(ROOT / 'outputs/cctv-dgp-bank-comparison-v1-execution.tar.gz') == plan['current_packet_archive_sha256']
    for name, expected in protocol['assets_sha256'].items():
        assert sha(packet / name) == expected, name
    result = dict(complete=True, plan_sha256=pin, cleanup_archive_sha256=export['archive_sha256'],
                  checker_sha256=sha(Path(__file__)), exact_backed_up_archives_removed=4,
                  local_backups_reverified=4, shared_links_deleted=0,
                  research_metadata_entries_unchanged=len(before['metadata']),
                  protected_byte_hashes_unchanged=len(before['critical_and_evidence_hashes']),
                  local_protected_bindings_unchanged=len(local_bindings), current_packet_assets_unchanged=len(protocol['assets_sha256']),
                  all_scientific_cache_bytes_rehashed=False, cache_preservation_proof=before['preservation_proof'],
                  reported_free_after_GiB=receipt['free_after_bytes']/1024**3,
                  observed_recovered_GiB=receipt['recovered_bytes']/1024**3,
                  model_gradient_or_training_calls=0, VM_started=False, goal_complete=False,
                  fresh_post_apply_inventory_required=True, seconds=time.monotonic()-started)
    with (OUT / 'independent_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({key: result[key] for key in ['complete', 'exact_backed_up_archives_removed', 'observed_recovered_GiB', 'reported_free_after_GiB']}, flush=True)


if __name__ == '__main__':
    main()
