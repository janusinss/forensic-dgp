"""Independent local readback of maintenance receipts, backups and history."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(bindings, mapping=None):
    for name, digest in bindings.items():
        path = (ROOT / (mapping or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    start = time.monotonic()
    path = OUT / 'closure_manifest.json'
    closure = read(path)
    assert closure['complete'] and closure['archives_removed'] == 14 and closure['goal_status'] == 'active'
    assert not closure['goal_complete'] and not closure['training_started'] and not closure['VM_started_or_stopped']
    assert closure['scientific_cache_files_removed'] == 0 and closure['current_V27_archives_retained']
    counts = [verify(closure['new_evidence_sha256'])]
    previous_path = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
    assert sha(previous_path) == closure['previous_milestone_sha256']
    previous = read(previous_path)
    counts.append(verify(previous['new_evidence_sha256'], closure['previous309_original_locations']))
    for name, key in (
        ('outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json', 'previous66_original_locations'),
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json', 'previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json', 'previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json', 'previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json', 'previous513_original_locations'),
    ):
        next_path = ROOT / name
        assert sha(next_path) == previous['previous_milestone_sha256']
        next_milestone = read(next_path)
        counts.append(verify(next_milestone['new_evidence_sha256'], previous[key]))
        previous = next_milestone
    assert counts[1:] == [309, 66, 692, 299, 697, 513]
    for name, backup in closure['previous309_original_locations'].items():
        original = (ROOT / backup).read_bytes()
        updated = (ROOT / name).read_bytes()
        split = original.index(b'\n') + 1
        assert updated.startswith(original[:split]) and updated.endswith(original[split:])
        prefix = updated[split:len(updated) - len(original[split:])].decode('utf-8')
        assert 'VM cleanup complete; full DGP goal active' in prefix
        assert 'actual training remains manual' in prefix and 'Goal incomplete' in prefix
    goal = read(OUT / 'active_goal_snapshot.json')['goal']['objective'].split(' Latest explicit authorization:', 1)[0]
    assert goal in (ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md').read_text(encoding='utf-8')
    plan = read(OUT / 'archive-duplicates_plan_20261006_v2.json')
    total = 0
    for row in plan['files']:
        backup = Path(row['local_backup']).resolve()
        assert backup.is_relative_to(ROOT / 'outputs') and backup.stat().st_size == row['bytes'] and sha(backup) == row['sha256']
        total += row['bytes']
    assert total == closure['logical_bytes_removed'] == 1756556632
    live = read(OUT / 'independent_VM_audit.log')
    assert live['complete'] and live['deleted_files_verified_absent'] == 14
    assert live['protected_hashed_files_live_verified'] == closure['protected_hashes_live_verified'] == 207967
    assert live['retained_tensor_stamps_live_verified'] == 4817
    runtime = read(OUT / 'final_runtime.json')
    assert runtime['free_bytes'] == closure['final_free_bytes'] and runtime['GPU_idle']
    assert runtime['runtime']['CUDA_available'] and runtime['retained_cache_bytes'] == 39448585279
    result = {'complete': True, 'closure_sha256': sha(path), 'checker_sha256': sha(Path(__file__)),
              'binding_counts': counts, 'archived_Windows_recovery_copies_verified': 14,
              'all_previous_research_histories_preserved': True, 'exact_reaffirmed_goal_preserved': True,
              'full_document_history_bodies_preserved': 3, 'files_removed_by_this_readback': 0,
              'model_or_gradient_calls': 0, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'independent_closure_readback.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
