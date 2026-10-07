"""Independent final readback of return, visual review and retained history."""
import base64
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_milestone'
PREVIOUS = ROOT / 'outputs/completion_pixel_support_v1_r1'
REVIEW = ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    start = time.monotonic()
    m = read(OUT / 'milestone.json')
    new_count = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    assert m['local_gradient_calls'] == m['local_optimizer_updates'] == 0
    assert not m['new_training_packet_released'] and not m['app_or_model_changes'] and not m['app_promotion']
    assert not m['reserved_final_used'] and not m['quality_qualification']
    assert not m['VM_workloads_started_stopped_or_modified_by_agent']
    doc = m['document']
    before, after = [(ROOT / doc[key]).read_bytes() for key in ['before_path', 'name']]
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert sha(ROOT / doc['before_path']) == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == '9d0e0a9393fe09897d10dc57b1a86ec9d9c517984cfbfdea0d591ff273659de8'
    prior_count = verify(previous['new_evidence_sha256'], m['previous_document_locations'])
    assert prior_count == 19
    old_final = read(PREVIOUS / 'final_readback.json')
    assert old_final['complete'] and old_final['independent_milestone_pass']
    prior_final_count = verify(old_final['evidence_sha256'], m['previous_document_locations'])
    old_closure = read(PREVIOUS / 'independent_milestone_audit.json')
    assert old_closure['complete'] and old_closure['milestone_sha256'] == m['previous_milestone_sha256']
    old_app = ROOT / 'outputs/dgp_mask_review_race_fix_v1'
    older = read(old_app / 'milestone.json')
    older_count = verify(older['new_evidence_sha256'], previous['previous_document_locations'])
    assert older_count == 72
    for name in m['scope_documents_unchanged']:
        assert sha(ROOT / name) == previous['new_evidence_sha256'][name]
    pixel_plan = read(PREVIOUS / 'plan.json')
    pixel_bindings = verify(pixel_plan['sources_sha256'])
    assert pixel_bindings == 300
    audit_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['training_completed800']
    assert not audit['necessary_capacity_pass'] and audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_return.py')
    assert audit['protocol_sha256'] == m['V31_protocol_sha256'] and audit['archive_sha256'] == m['V31_return_sha256']
    assert audit['local_gradient_calls'] == audit['local_backward_calls'] == audit['local_optimizer_updates'] == 0
    assert not audit['native_or_reserved_used'] and not audit['app_promotion'] and not audit['goal_complete']
    assert [row['update'] for row in audit['snapshots_audited']] == [0, 50]
    assert not audit['partial_snapshots_retained'] and audit['completed_snapshot_PNG_metrics_vectors_and_mean_controls_audited'] == 7810
    imported = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_import.json')
    returned = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
    returned_count = verify({(returned / name).relative_to(ROOT).as_posix(): digest
                             for name, digest in imported['files_sha256'].items()})
    assert returned_count == audit['members_verified'] == m['V31_return_members']
    failure = read(returned / 'outputs/failure.json')
    early = read(returned / 'outputs/early_structure_stop.json')
    assert failure['optimizer_updates'] == early['update'] == m['V31_optimizer_updates'] == 50
    assert failure['resume_permitted'] is False and not m['V31_resume_permitted'] and not early['pass']
    assert early['minimum'] == m['V31_early_requirement'] == .01 and early['relative_feature_error_gain'] == m['V31_early_gain'] < .01
    assert not (returned / 'outputs/results.json').exists() and (returned / 'outputs/stopped_dgp_candidate_v31.pth').is_file()
    prepared, visual = read(REVIEW / 'preparation.json'), read(REVIEW / 'visual_review.json')
    reviewed = read(REVIEW / 'independent_preparation_audit.json')
    review_bindings = verify(prepared['source_bindings_sha256'])
    assert reviewed['complete'] and reviewed['preparation_sha256'] == sha(REVIEW / 'preparation.json')
    assert reviewed['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_failure_review.py')
    assert reviewed['exact_256_cells_verified'] == prepared['exact_cells_checked'] == 250
    assert not reviewed['preservation_regressions_at50'] and m['V31_all17_preservation_groups_pass_at50']
    assert visual['complete'] and visual['cases_reviewed'] == 50 and visual['comparison_cells_reviewed'] == 250
    assert visual['recorder_source_sha256'] == sha(ROOT / 'scripts/record_cctv_dgp_profile_batches_v31_visual_review.py')
    assert visual['independent_return_audit_sha256'] == sha(audit_path)
    assert visual['preparation_sha256'] == sha(REVIEW / 'preparation.json')
    assert visual['independent_preparation_audit_sha256'] == sha(REVIEW / 'independent_preparation_audit.json')
    assert not visual['useful_whole_face_gain_established'] and not visual['independent_final_review']
    assert [r['id'] for r in visual['rows']] == read(REVIEW / 'plan.json')['case_ids']
    assert all(row['regions_inspected'] == read(REVIEW / 'plan.json')['regions'] for row in visual['rows'])
    assert len(visual['actually_viewed_sheets']) == 10
    assert all(row['requested_detail'] == 'original' and row['actually_viewed'] for row in visual['actually_viewed_sheets'])
    assert all(set(row['observations']) == set(row['regions_inspected']) and
               all(isinstance(note, str) and len(note) > 10 for note in row['observations'].values())
               for row in visual['rows'])
    feature = read(ROOT / 'outputs/cctv_dgp_post_v31_feature_path_review_v1/review.json')
    feature_bindings = verify(feature['source_bindings_sha256'])
    assert feature['complete'] and feature['reviewer_sha256'] == sha(ROOT / 'scripts/review_cctv_dgp_post_v31_feature_path_v1.py')
    fusion_names = {'fpn.lateral' + str(i) + '.weight' for i in range(5)} | {
        'fpn.td' + str(i) + '.0.' + suffix for i in range(1, 4) for suffix in ['weight', 'bias']}
    assert {row['name'] for row in feature['feature_fusion_tensors']} == fusion_names
    assert sum(math.prod(row['shape']) for row in feature['feature_fusion_tensors']) == feature['feature_fusion_parameters'] == 479616
    assert all(row['unchanged_at_stopped50'] and not row['connectivity_gradient_measured'] for row in feature['feature_fusion_tensors'])
    assert feature['all_FPN_state_entries_unchanged'] and feature['changed_decoder_parameters'] == 498627
    assert feature['neural_calls'] == feature['gradient_calls'] == feature['backwards'] == feature['optimizer_updates'] == 0
    assert not feature['training_packet_released'] and not feature['new_training_recipe_selected'] and feature['no_unique_cause_established']
    verify({(REVIEW / name).relative_to(ROOT).as_posix(): digest for name, digest in prepared['sheet_sha256'].items()})
    status_root = ROOT / 'outputs/cctv_dgp_v31_progress_status_v1_r1'
    live, t = read(status_root / 'launch_status.log'), read(status_root / 'launch_status_transport.json')
    assert live['complete'] and live['read_only'] and live['snapshot_utc'] == m['VM_snapshot_utc']
    assert not live['V31_processes'] and not live['GPU']['stdout'] and not live['training_started_by_agent']
    assert t['exit_code'] == 0 and t['stdout_sha256'] == sha(status_root / 'launch_status.log')
    meta_root = ROOT / 'outputs/cctv_dgp_v31_return_metadata_v1'
    meta, mt = read(meta_root / 'original_return_metadata.log'), read(meta_root / 'original_return_metadata_transport.json')
    assert meta['complete'] and meta['read_only'] and meta['VM_writes'] == 0 and not meta['training_started_by_agent']
    assert mt['exit_code'] == 0 and mt['stdout_sha256'] == sha(meta_root / 'original_return_metadata.log')
    for row in meta['files']:
        raw = base64.b64decode(row['content_base64'], validate=True)
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        assert (ROOT / 'outputs' / row['name']).read_bytes() == raw
    source_root = ROOT / 'outputs/completion_mat_source_review_v1_r1'
    source = read(source_root / 'independent_source_audit.json')
    assert source['complete'] and source['checker_sha256'] == sha(ROOT / 'scripts/audit_completion_mat_source_review_v1.py')
    assert source['official_source_files_verified'] == 5 and not source['checkpoint_downloaded'] and not source['source_executed']
    assert source['model_or_gradient_calls'] == source['optimizer_updates'] == 0
    app = read(ROOT / 'outputs/dgp_app_v3_integration_record.json')
    unchanged_app = verify({name: digest for name, digest in app['sources_sha256'].items() if name != 'static/face_workflow.js'})
    archived = ROOT / 'outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js'
    assert sha(archived) == app['sources_sha256']['static/face_workflow.js']
    assert sha(ROOT / 'static/face_workflow.js') == '9be6cfab57e0041f87571cd38ba6d0d1e0850742b8fafd498f1b1d126137a060'
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == app['checkpoint_sha256']
    from verify_cctv_dgp_vm_storage_cleanup_20261007_v2 import verify as verify_cache_stamps
    backup = verify_cache_stamps()
    assert backup['actual_Windows_cache_files_preserved'] == 4431
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': new_count, 'prior19_bindings_preserved': prior_count,
               'prior_final_readback_bindings_preserved': prior_final_count, 'older72_bindings_preserved': older_count,
               'previous_pixel_diagnostic_sources_preserved': pixel_bindings,
               'returned_files_preserved': returned_count, 'review_source_bindings_verified': review_bindings,
               'fixed50_TRAIN_cases_and250_cells_reviewed': True, 'all17_preservation_groups_pass_at_stopped50': True,
               'one_percent_structure_failure_and_stopped_checkpoint_preserved': True,
               'exact_original_VM_export_sidecars_verified': 2, 'terminal_read_only_VM_snapshot_verified': True,
               'MAT_source_files_verified_without_execution': 5,
               'feature_path_review_bindings_verified': feature_bindings, 'feature_fusion_tensors_reviewed': 11,
               'unchanged_app_sources': unchanged_app, 'archived_prior_UI_source_preserved': True,
               'actual_Windows_backup_stamps_preserved': 4431, 'full_previous_handoff_preserved': True,
               'scope_documents_unchanged': 2, 'app_model_checkpoint_unchanged': True,
               'new_training_or_local_gradient_calls': 0, 'reserved_final_used': False,
               'quality_qualification': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
