"""Independent saved-plan/namespace audit; no upload, deletion or model calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/google_drive_offload_20261012_v1'
CACHE = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def small_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-name', default='independent_preparation_audit.json')
    args = parser.parse_args()
    assert Path(args.receipt_name).name == args.receipt_name and args.receipt_name.endswith('.json')
    target = OUT / args.receipt_name
    assert not target.exists(), 'Retain any earlier audit'
    plan = read(OUT / 'plan.json')
    assert small_sha(ROOT / 'scripts/run_google_drive_offload_20261012_v1.ps1') == plan['driver_sha256']
    protected = read(OUT / 'protected_active_bindings.json')
    source = read(CACHE / 'cache_source_inventory.json')
    cache_audit = read(CACHE / 'drive_preflight_20261012_v1.json')
    expected = {'outputs/cctv_dgp_local_research_cache_backup_20261007_v1/cache_files' + item['path']: item['sha256']
                for item in source['files']}
    selected = {item['path']: item['sha256'] for item in plan['batches']['cache']['files'] if item['role'].startswith('inactive')}
    assert selected == expected
    assert cache_audit['complete'] and cache_audit['verified_cache_files'] == 4431
    assert small_sha(CACHE / 'cache_source_inventory.json') == 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'
    candidates = [item for batch in plan['batches'].values() for item in batch['files'] if item['role'].startswith('inactive')]
    assert not set(item['path'] for item in candidates) & set(protected)
    assert all((ROOT / item['path']).is_file() and (ROOT / item['path']).stat().st_size == item['bytes'] for item in candidates)
    for name, batch in plan['batches'].items():
        paths = (OUT / f'{name}_files.txt').read_text(encoding='utf-8').splitlines()
        sums = (OUT / f'{name}_sha256.txt').read_text(encoding='utf-8').splitlines()
        assert paths == [item['path'] for item in batch['files']]
        assert sums == [item['sha256'] + '  ' + item['path'] for item in batch['files']]
        assert small_sha(OUT / f'{name}_files.txt') == batch['file_list_sha256']
        assert small_sha(OUT / f'{name}_sha256.txt') == batch['sha256_list_sha256']
    original_pins = read(OUT / 'documents_before/sha256.json')
    preserved = {name: small_sha(OUT / 'documents_before' / name) == pin for name, pin in original_pins.items()}
    assert all(preserved.values())
    historical = {name: (ROOT / name).read_text(encoding='utf-8').endswith(
        (OUT / 'documents_before' / name).read_text(encoding='utf-8')) for name in original_pins}
    assert all(historical.values())
    fixtures = read(ROOT / 'scratch/drive_offload_checksum_20261012_v1/receipt.json')
    assert [item['ExitCode'] for item in fixtures] == [0, 1, 1]
    assert [item['Combined'] for item in fixtures] == [['= sample.bin'], ['* sample.bin'], ['+ sample.bin']]
    result = dict(complete=True, prepared_only=True, plan_sha256=small_sha(OUT / 'plan.json'),
                  checker_sha256=small_sha(Path(__file__)), selected_cache_namespace_exact=True,
                  exact_cache_hashes_match_frozen_source=True, active_bindings=len(protected),
                  candidates=len(candidates), candidate_logical_bytes=sum(item['bytes'] for item in candidates),
                  selected_local_files_still_present=True, historical_document_text_preserved=historical,
                  original_document_bytes_preserved=preserved, checksum_matching_changed_missing_fixtures_passed=True,
                  files_uploaded=0, files_removed=0, training_calls=0)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
