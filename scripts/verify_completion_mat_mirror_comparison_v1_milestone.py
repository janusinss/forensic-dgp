"""Independent closure of converted-MAT evidence and the unchanged original app."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_mirror_comparison_v1_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_milestone'
COMPARISON = ROOT / 'outputs/completion_mat_mirror_comparison_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json')
    assert m['complete'] and m['goal_status'] == 'active' and not m['goal_complete']
    new_count = verify(m['new_evidence_sha256'])
    prior = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256']
    old_count = verify(prior['new_evidence_sha256'], m['previous_document_locations'])
    closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert closure['complete'] and closure['milestone_sha256'] == m['previous_milestone_sha256']
    final_count = verify(read(PREVIOUS / 'final_readback.json')['evidence_sha256'], m['previous_document_locations'])
    doc = m['document']
    before, after = (ROOT / doc['before_path']).read_bytes(), (ROOT / doc['name']).read_bytes()
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest() == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    protocol = read(COMPARISON / 'protocol.json')
    sources = verify(protocol['sources_sha256'])
    app = verify(protocol['app_preservation_sha256'])
    assert len(protocol['cases']) == 32 and protocol['excluded_before_generation'] == 4
    assert protocol['assisted_only'] and not protocol['new_training'] and not protocol['native_CCTV_or_reserved_final_used']
    results, audit, visual = map(read, [COMPARISON / 'results.json', COMPARISON / 'independent_saved_output_audit.json', COMPARISON / 'visual_review.json'])
    assert results['complete'] and results['protocol_sha256'] == audit['protocol_sha256'] == visual['protocol_sha256'] == sha(COMPARISON / 'protocol.json')
    assert audit['results_sha256'] == visual['results_sha256'] == sha(COMPARISON / 'results.json')
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_completion_mat_mirror_comparison_v1.py')
    assert visual['recorder_sha256'] == sha(ROOT / 'scripts/record_completion_mat_mirror_comparison_v1_review.py')
    assert visual['independent_saved_output_audit_sha256'] == sha(COMPARISON / 'independent_saved_output_audit.json')
    assert results['requests'] == 32 and results['pretrained_forwards'] == 28 and results['empty_mask_bypasses'] == 4
    assert audit['exact_visible_source_bytes'] == m['exact_visible_source_bytes'] == 5225232
    assert audit['all32_saved_outputs_verified'] and audit['all28_raw512_to256_compositions_verified']
    assert audit['all4_clear_controls_exact_bypasses'] and audit['independent_CPU_replay']['maximum_raw_error'] == 0
    assert audit['independent_CPU_replay']['exact_PNG'] and results['generator_state_before'] == results['generator_state_after']
    assert [row['id'] for row in results['rows']] == [row['id'] for row in protocol['cases']]
    assert [row['id'] for row in visual['rows']] == [row['id'] for row in protocol['cases']]
    for row in results['rows']:
        assert sha(COMPARISON / row['output']) == row['output_sha256']
        assert sha(COMPARISON / row['completed256']) == row['completed256_sha256']
        if row['raw512'] is not None:
            assert sha(COMPARISON / row['raw512']) == row['raw512_sha256']
        assert row['mode'] == 'assisted' and row['exact_outside_reviewed_mask']
        assert row['hidden_PSNR_SSIM'] is None and not row['exact_hidden_identity_claim']
    assert len(visual['sheets']) == 8 and visual['all32_cases_reviewed'] and visual['all8_sheets_viewed_at_original_detail']
    for sheet in visual['sheets']:
        assert sheet['actually_viewed'] and sheet['requested_detail'] == 'original'
        assert sha(COMPARISON / sheet['path']) == sheet['sha256']
    assert not any(visual[k] for k in ['automatic_results_generated', 'automatic_quality_qualification',
                                      'assisted_quality_qualification', 'MAT_original_author_model_failure_claim',
                                      'app_changes', 'training', 'goal_complete'])
    supervision = read(COMPARISON / 'supervisor_receipt.json')
    assert supervision['complete'] and supervision['child_exit_code'] == 0
    assert 0 < supervision['seconds'] < supervision['cap_seconds'] == 1230
    assert results['seconds'] < protocol['budgets']['total_seconds']
    asset_root = ROOT / 'outputs/completion_mat_mirror_assets_v1'
    assets = read(asset_root / 'independent_assets_audit.json')
    assert assets['complete'] and assets['adapter_regressions_passed'] == 11
    assert assets['source_routing_is_only_MAT_source_difference'] and assets['all_helpers_byte_identical']
    assert assets['original_tensor_count'] == 465 and assets['all62612683_weight_values_finite']
    assert assets['checker_sha256'] == sha(ROOT / 'scripts/audit_completion_mat_mirror_assets_v1.py')
    assert assets['original_weight_sha256'] == sha(asset_root / 'MAT_FFHQ_512_fp16.safetensors')
    assert assets['adapter_test_output_sha256'] == sha(asset_root / 'adapter_regressions.txt')
    assert sha(ROOT / 'mat_mirror_completion_v1.py') == read(asset_root / 'preparation.json')['adapter_sha256']
    source_root = ROOT / 'outputs/completion_mat_mirror_review_v1_r1'
    source_audit = read(source_root / 'independent_source_audit.json')
    assert source_audit['complete'] and source_audit['source_files_verified'] == 11
    assert source_audit['acquisition_sha256'] == sha(source_root / 'acquisition.json')
    assert source_audit['checker_sha256'] == sha(ROOT / 'scripts/audit_completion_mat_mirror_review_v1.py')
    r1 = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_independent_audit.json')
    imported = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_return_import.json')
    assert r1['complete'] and r1['members_verified'] == 180 and r1['saved_gradient_values_checked'] == 75324711
    assert r1['snapshots_audited'] == [] and r1['initial50_CPU_replay'] == 50
    assert r1['local_optimizer_updates'] == r1['local_gradient_calls'] == 0
    r1_count = verify({'outputs/cctv_dgp_feature_fusion_v32_r1_return/' + name: digest for name, digest in imported['files_sha256'].items()})
    download_root = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_download_v1'
    download = read(download_root / 'download_receipt.json')
    assert download['complete'] and download['VM_writes'] == 0 and not download['training_started_by_agent']
    for row in download['placements']:
        assert sha(ROOT / 'outputs' / row['name']) == row['sha256']
    assert sha(ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r2-results.tar.gz') == download['archive_sha256']
    assert closure['previous_actual_Windows_backup_receipt_preserved'] == 4431
    assert sha(ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md') == '2aad6635473dad349c85ab177ec0d1fc09d1089c350bf620298f4a06f67d2299'
    assert sha(ROOT / 'PRACTICAL_OUTPUT_SCOPE.md') == '47a02049970719379727aca9f5a6707177e5894a35c211814076aa7c628bdb2b'
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': new_count, 'previous_R2_repair_bindings_preserved': old_count,
               'previous_final_readback_bindings_preserved': final_count, 'full_previous_handoff_preserved': True,
               'MAT_source_and_comparison_bindings_verified': sources, 'unchanged_app_bindings_verified': app,
               'all32_assisted_outputs_and8_development_sheets_preserved': True,
               'MAT_quality_qualification': False, 'original_author_equivalence_verified': False,
               'previous_actual_Windows_backup_receipt_preserved': 4431, 'R1_returned_files_verified': r1_count,
               'R1_full_gradient_audit_complete': True, 'R1_optimizer_updates': 0, 'R2_full_audit_pending': True,
               'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_writes': 0,
               'training_started_by_agent': False, 'manual_VM_training_required': True,
               'app_promotion': False, 'independent_final_review': False, 'goal_complete': False,
               'seconds': time.monotonic() - started}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
