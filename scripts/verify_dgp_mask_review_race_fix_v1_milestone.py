"""Closure audit with explicit archived-source resolution for historical UI evidence."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_mask_review_race_fix_v1'
PREVIOUS = ROOT / 'outputs/dgp_sampling_review_and_profile_batches_v31_milestone'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None, historical=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        row = (historical or {}).get(name)
        if row is not None and row['sha256'] == digest: path = (ROOT / row['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    start = time.monotonic(); m = read(OUT / 'milestone.json'); count = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['ordering_regressions'] == 3 and m['real_model_generated_outputs'] == 11
    assert m['families_functionally_verified'] == 7 and not m['quality_qualification'] and m['goal_status'] == 'active'
    assert m['local_gradient_calls'] == m['local_optimizer_updates'] == 0 and not m['goal_complete']
    assert m['VM_actions'] == ['read-only V31 status', 'read-only corrected V31 transfer verification']
    assert not m['V31_actual_training_started_by_agent'] and m['V31_original_launch_failures_preserved'] == 2
    assert m['V31_command_only_import_fix_verified']
    for row in m['documents']:
        before = (ROOT / row['before_path']).read_bytes(); after = (ROOT / row['name']).read_bytes()
        assert sha(ROOT / row['before_path']) == row['before_sha256'] and sha(ROOT / row['name']) == row['after_sha256']
        split = before.index(b'\n') + 1; assert after[:split] + after[split + row['addition_bytes']:] == before
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == 'a6dd026c8d343163ed7ba599cca7924af4085579960bece9568484c2280caa3f'
    previous = read(PREVIOUS / 'milestone.json'); prior_count = verify(previous['new_evidence_sha256'], m['previous_document_locations']); assert prior_count == 41
    old_final = read(PREVIOUS / 'final_readback.json'); final_count = verify(old_final['evidence_sha256'])
    assert old_final['complete'] and old_final['independent_closure_pass']
    prior_path = ROOT / 'outputs/dgp_v30_return_and_sampling_gradient_v1_milestone'
    older = read(prior_path / 'milestone.json'); older_count = verify(older['new_evidence_sha256'], previous['previous56_original_locations']); assert older_count == 56
    corrections = read(prior_path / 'final_readback.json'); correction_count = verify(corrections['correction_evidence_sha256'])
    cleanup = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/closure_manifest.json')
    cleanup_count = verify(cleanup['new_evidence_sha256'], older['previous_cleanup110_original_locations']); assert cleanup_count == 110
    earlier = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/closure_manifest.json')
    backup_docs = {r['name']: 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1/recovery_metadata/project_instructions/' + r['name'] for r in m['documents']}
    earlier_count = verify(earlier['new_evidence_sha256'], backup_docs); assert earlier_count == 71
    import verify_cctv_dgp_vm_storage_cleanup_20261007_v1 as history_module
    original_verifier = history_module.verify
    historical = m['historical_source_locations']
    assert historical == {'static/face_workflow.js': {'sha256': '81edfc86fcd16d976f1ed98d146680b38ada9af7504f6bdc4d60a5471cb249b0', 'path': 'outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js'}}
    # Resolve one historical SHA to its preserved bytes, never the new frontend.
    # Historical source files and gate checks remain unmodified.
    history_module.verify = lambda mapping, aliases=None: verify(mapping, aliases, historical)
    historic_docs = {r['name']: 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/before_docs/' + r['name'] for r in m['documents']}
    try: history_counts = history_module.history(historic_docs)
    finally: history_module.verify = original_verifier
    from verify_cctv_dgp_vm_storage_cleanup_20261007_v2 import verify as verify_cache_stamps
    cache = verify_cache_stamps(); assert cache['actual_Windows_cache_files_preserved'] == 4431
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == m['original_app_record_sha256'] == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(app_path); assert verify({**app['sources_sha256'], **app['evidence_sha256']}, historical=historical) == 22
    assert verify({name: digest for name, digest in app['sources_sha256'].items() if name != 'static/face_workflow.js'}) == 12
    assert sha(ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json') == m['V31_protocol_sha256']
    assert sha(ROOT / 'outputs/cctv-dgp-profile-batches-v31-execution.tar.gz') == m['V31_archive_sha256']
    saved = read(OUT / 'independent_saved_output_audit.json')
    assert saved['complete'] and saved['checker_sha256'] == sha(ROOT / 'scripts/audit_dgp_mask_review_race_fix_v1.py')
    assert saved['downloaded_PNGs_verified'] == 11 and saved['raw_bundle_and_exact_composition_verified']
    launch_path = ROOT / 'outputs/cctv_dgp_v31_launch_import_v1/independent_launch_audit.json'
    assert sha(launch_path) == m['V31_launch_import_audit_sha256']
    launch = read(launch_path); launch_count = verify(launch['evidence_sha256'])
    assert launch['complete'] and launch['corrected_transfer_check_passed'] and not launch['VM_training_started_by_agent']
    assert launch['original_missing_module_failures_retained'] == 2 and launch['packet_protocol_and_training_recipe_unchanged']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': count, 'previous41_bindings_preserved': prior_count, 'previous_final_bindings': final_count,
               'older56_bindings_preserved': older_count, 'older_correction_bindings': correction_count,
               'cleanup110_bindings': cleanup_count, 'earlier_cleanup71_bindings': earlier_count,
               'historical_research_bindings': history_counts, 'archived_UI_source_resolution_explicit': historical,
               'other12_app_sources_unchanged': True, 'actual_Windows_backup_stamps_preserved': 4431,
               'complete_previous_document_bodies_preserved': 3, 'V31_packet_unchanged': True,
               'quality_qualification': False, 'reserved_final_used': False, 'training': False,
               'read_only_VM_actions_verified': m['VM_actions'], 'V31_launch_recovery_bindings_verified': launch_count,
               'V31_launch_recipe_unchanged_and_command_fix_verified': True,
               'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
