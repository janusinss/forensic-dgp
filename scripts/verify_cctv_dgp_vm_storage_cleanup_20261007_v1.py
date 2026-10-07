"""Independent local maintenance/backup/history readback; no model or deletion."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1'
PREVIOUS = ROOT / 'outputs/dgp_v29_development_and_broader_mean_v30_milestone/milestone.json'
PREVIOUS_PIN = '8992a934f6f2217501c6e0878f9f0ae1e8d91c9d3d9130819f2ff6fd63919e5d'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(bindings, mapping=None):
    for name, digest in bindings.items():
        path = (ROOT / (mapping or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def history(mapping=None):
    assert sha(PREVIOUS) == PREVIOUS_PIN
    previous = read(PREVIOUS)
    counts = [verify(previous['new_evidence_sha256'], mapping)]
    for name, key in (
        ('outputs/dgp_v28_preservation_diagnostic_and_mean_centered_v29_milestone/milestone.json', 'previous371_original_locations'),
        ('outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone/milestone.json', 'previous1041_original_locations'),
        ('outputs/dgp_original_decoder_r2_audit_and_active_decoder_v28_milestone/milestone.json', 'previous586_original_locations'),
        ('outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json', 'previous50_original_locations'),
        ('outputs/dgp_feature_skips_v27_audit_milestone/milestone.json', 'previous668_original_locations'),
    ):
        path = ROOT / name
        assert sha(path) == previous['previous_milestone_sha256']
        older = read(path)
        counts.append(verify(older['new_evidence_sha256'], previous[key]))
        previous = older
    v27 = previous
    path = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
    assert sha(path) == v27['cleanup_closure_sha256']
    counts.append(verify(read(path)['new_evidence_sha256'], v27['cleanup60_original_locations']))
    path = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
    assert sha(path) == v27['previous_milestone_sha256']
    previous = read(path)
    counts.append(verify(previous['new_evidence_sha256'], v27['previous309_original_locations']))
    for name, key in (
        ('outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json', 'previous66_original_locations'),
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json', 'previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json', 'previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json', 'previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json', 'previous513_original_locations'),
    ):
        path = ROOT / name
        assert sha(path) == previous['previous_milestone_sha256']
        older = read(path)
        counts.append(verify(older['new_evidence_sha256'], previous[key]))
        previous = older
    assert counts == [5934, 371, 1041, 586, 50, 668, 60, 309, 66, 692, 299, 697, 513]
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == read(PREVIOUS)['app_record_sha256']
    app = read(app_path)
    assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    return counts


def maintenance():
    plan_path = OUT / 'archive-duplicates_plan_20261007_v1.json'
    plan, prep = read(plan_path), read(OUT / 'preparation.json')
    assert sha(plan_path) == prep['plan_sha256']
    assert prep['duplicate_archives'] == 10 and prep['duplicate_recognizers'] == 4
    assert prep['backed_file_bytes'] == 1915161691 and prep['V30_dependency_bindings'] == 5757
    assert plan['clear_pip_download_cache'] and not plan['scientific_cache_removal_authorized']
    assert plan['all_trained_DGP_checkpoints_retained'] and plan['shared_V26_recognizer_retained']
    remote = OUT / 'remote_receipts'
    v, c, protected = (read(remote / name) for name in ('verification.json', 'cleanup_receipt.json', 'protected_before.json'))
    a = read(OUT / 'independent_VM_audit.log')
    runtime = read(OUT / 'final_runtime.json')
    assert all(row['complete'] for row in (v, c, a, runtime))
    assert v['plan_sha256'] == c['plan_sha256'] == a['plan_sha256'] == sha(plan_path)
    assert v['script_sha256'] == c['script_sha256'] == a['backend_sha256'] == prep['backend_sha256']
    assert sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261007_v1.py') == prep['backend_sha256']
    assert sha(ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py') == prep['auditor_sha256']
    for field, name in (('verification_sha256', 'verification.json'), ('cleanup_receipt_sha256', 'cleanup_receipt.json'),
                        ('deletions_sha256', 'deletions.jsonl'), ('protected_snapshot_sha256', 'protected_before.json')):
        assert a[field] == sha(remote / name)
    assert v['protected_snapshot_sha256'] == sha(remote / 'protected_before.json')
    assert c['archives_removed'] == 10 and c['duplicate_recognizer_files_removed'] == 4 and c['inactive_cache_files_removed'] == 0
    assert c['archive_logical_bytes_removed'] + c['duplicate_recognizer_logical_bytes_removed'] == prep['backed_file_bytes']
    assert c['pip_cache_files_removed'] == len(v['pip_cache_files'])
    assert c['files_removed'] == a['deleted_files_verified_absent'] == 14 + len(v['pip_cache_files'])
    assert c['protected_hashed_files_unchanged'] == a['protected_hashed_files_live_verified'] == len(protected['sha256'])
    assert c['scientific_tensor_files_unchanged'] == a['retained_tensor_stamps_live_verified'] == len(protected['scientific_tensor_stamps'])
    assert a['explicit_current_research_bindings_live_verified'] == 5757
    assert all(protected['sha256'][name] == digest for name, digest in plan['protected_assets_sha256'].items())
    expected = {row['path']: row for row in v['archives'] + v['pip_cache_files']}
    ledger = [json.loads(line) for line in (remote / 'deletions.jsonl').read_text().splitlines()]
    assert len(ledger) == len(expected) and {row['path'] for row in ledger} == set(expected)
    for row in ledger:
        assert row['bytes'] == expected[row['path']]['bytes']
        assert row.get('local_backup') == expected[row['path']].get('local_backup')
    for row in plan['files']:
        local = Path(row['local_backup']).resolve()
        assert local.is_relative_to(ROOT / 'outputs') and local.stat().st_size == row['bytes'] and sha(local) == row['sha256']
    assert c['free_bytes_increase'] >= 1.3 * 1024**3 and runtime['free_bytes'] >= 6 * 1024**3
    assert runtime['runtime']['CUDA_available'] and runtime['GPU'] == 'NVIDIA L4' and runtime['GPU_idle']
    assert runtime['retained_cache_files'] == 4431 and runtime['retained_cache_bytes'] == 39448585279
    assert not any(row['training_started'] for row in (v, c, a, runtime)) and not c['VM_stopped']
    assert runtime['model_loaded'] is False
    return {'backed_files_verified': 14, 'backed_file_bytes': prep['backed_file_bytes'],
            'pip_cache_files_removed': c['pip_cache_files_removed'], 'deleted_files_verified_absent': c['files_removed'],
            'protected_hashed_files_live_verified': len(protected['sha256']),
            'scientific_tensor_stamps_live_verified': len(protected['scientific_tensor_stamps']),
            'V30_dependency_bindings_live_verified': 5757, 'measured_free_bytes_increase': c['free_bytes_increase'],
            'final_free_bytes': runtime['free_bytes'], 'CUDA_available': True, 'GPU_idle': True}


def main():
    started = time.monotonic()
    closure_path = OUT / 'closure_manifest.json'
    c = read(closure_path)
    assert c['complete'] and c['goal_status'] == 'active' and not c['goal_complete']
    current = verify(c['new_evidence_sha256'])
    counts = history(c['previous5934_original_locations'])
    checked = maintenance()
    assert all(c[key] == value for key, value in checked.items())
    for name, backup in c['previous5934_original_locations'].items():
        old, new = (ROOT / backup).read_bytes(), (ROOT / name).read_bytes()
        split = old.index(b'\n') + 1
        assert new.startswith(old[:split]) and new.endswith(old[split:])
        prefix = new[split:len(new) - len(old[split:])].decode('utf-8')
        for text in ('VM cleanup complete', 'manual VM execution', '5757', 'trained DGP checkpoints', 'Goal active/incomplete'):
            assert text in prefix, (name, text)
    result = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'closure_sha256': sha(closure_path),
        'new_evidence_bindings_verified': current, 'historical_binding_counts': counts,
        'app22_bindings_preserved': True, 'complete_prior_document_bodies_preserved': 3,
        **checked, 'training_started': False, 'model_or_gradient_calls': 0,
        'files_removed_by_this_readback': 0, 'goal_complete': False, 'seconds': time.monotonic() - started}
    with (OUT / 'independent_closure_readback.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
