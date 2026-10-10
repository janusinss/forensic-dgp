"""Independent hash, scope, diagnostic coverage and handoff closure audit."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_audited_milestone'
PRIOR = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1'


def main():
    start = time.monotonic()
    target = OUT / 'independent_closure_audit.json'
    assert not target.exists(), 'Never overwrite a closure receipt'
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))

    def sha(path):
        assert path.is_file() and not path.is_symlink(), str(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert time.monotonic() - start < 600, 'Finite local closure audit cap exceeded'
        return digest

    milestone = read(OUT / 'milestone.json')
    for name, digest in milestone['new_evidence_sha256'].items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert sha(path) == digest, name
    print({'new_evidence_bindings_checked': len(milestone['new_evidence_sha256'])}, flush=True)
    old = read(PRIOR / 'milestone.json')
    old_closure = read(PRIOR / 'independent_closure_audit.json')
    assert old_closure['complete'] and sha(PRIOR / 'milestone.json') == old_closure['milestone_sha256']
    assert sha(PRIOR / 'milestone.json') == milestone['previous_milestone_sha256']
    assert sha(PRIOR / 'independent_closure_audit.json') == milestone['previous_closure_sha256']
    before = (OUT / 'PROJECT_HANDOFF_before_tail.md').read_bytes()
    current = (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    assert hashlib.sha256(before).hexdigest() == old['new_evidence_sha256']['PROJECT_HANDOFF.md']
    assert hashlib.sha256(before).hexdigest() == milestone['previous_handoff_sha256']
    assert current[milestone['new_intro_bytes']:] == before
    for name, digest in old['new_evidence_sha256'].items():
        if name != 'PROJECT_HANDOFF.md':
            assert milestone['new_evidence_sha256'][name] == digest and sha(ROOT / name) == digest, name
    assert milestone['app_bindings_sha256'] == old['app_bindings_sha256']
    for name, digest in milestone['app_bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert len(milestone['app_bindings_sha256']) == 14
    imported_counts = {}
    for stem in ['cctv_dgp_actual_step_review_v1', 'cctv_dgp_actual_step_tail_v1']:
        base = ROOT / 'outputs'
        imported = read(base / (stem + '_return_import.json'))
        returned = base / (stem + '_vm_return')
        assert imported['complete'] and not imported['returned_code_executed']
        for name, digest in imported['files_sha256'].items():
            assert sha(returned / name) == digest, stem + '/' + name
        imported_counts[stem] = len(imported['files_sha256'])
        print({'retained_return_members_rehashed': stem, 'count': len(imported['files_sha256'])}, flush=True)
    assert imported_counts == {'cctv_dgp_actual_step_review_v1': 9369, 'cctv_dgp_actual_step_tail_v1': 959}
    science = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json')
    partial = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json')
    assert science['complete'] and science['all315_outputs_checked']
    assert science['members_verified'] == 959 and science['overlap_files_independently_checked'] == 828
    assert science['every_original_stored_row_and_categorical_gate_exact']
    assert science['all_control_and_partial_overlap_arrays_pixels_metrics_exact']
    assert science['original_storage_failure_retained'] and partial['original_storage_failure_retained']
    assert partial['complete'] and partial['all3045_raw_PNG_and_mean_only_outputs_checked']
    assert partial['CPU_replay']['cases'] == 145 and science['CPU_replay']['cases'] == 15
    assert milestone['tail_archive_sha256'] == science['archive_sha256']
    archive = ROOT / 'outputs/cctv-dgp-actual-step-tail-v1-results.tar.gz'
    assert sha(archive) == science['archive_sha256'] and archive.stat().st_size == 324974655
    original_return = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm_return'
    tail_packet = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_vm/protocol.json')
    assert tail_packet['prospective_auditor_sha256'] == science['checker_sha256']
    for name, digest in tail_packet['retained_parent_return_sha256'].items():
        assert sha(original_return / name) == digest, name
    for name, digest in tail_packet['overlap_files_sha256'].items():
        assert sha(original_return / name) == digest, name
    assert len(tail_packet['overlap_files_sha256']) == 828
    results = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_vm_return/outputs/results.json')
    assert results['complete'] and results['original_initial_reference_and_recognizer_unchanged']
    assert results['progress']['forward_slots'] == 315 and results['progress']['completed_conditions'] == 3
    assert not results['progress']['optimizer_constructed']
    for key in ['optimizer_updates', 'gradient_queries', 'backward_calls']:
        assert results['progress'][key] == 0
    assert not results['progress']['new_trained_checkpoint']
    analysis_root = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_analysis'
    analysis = read(analysis_root / 'analysis.json')
    arithmetic = read(analysis_root / 'independent_arithmetic_and_sheets_audit.json')
    visual = read(analysis_root / 'visual_review.json')
    previous_visual = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis/visual_review.json')
    original_plan = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json')
    assert arithmetic['complete'] and arithmetic['all150_new_cells_exact']
    assert arithmetic['analysis_sha256'] == visual['analysis_sha256'] == sha(analysis_root / 'analysis.json')
    assert visual['independent_pixel_audit_sha256'] == sha(analysis_root / 'independent_arithmetic_and_sheets_audit.json')
    assert visual['complete'] and visual['sheets_viewed'] == 5 and visual['rows_viewed'] == 25
    assert visual['exact256_pixel_cells_viewed'] == 150
    fields = ['update', 'role', 'id', 'source', 'profile']
    combined = previous_visual['rows'] + visual['rows']
    assert [{key: row[key] for key in fields} for row in combined] == original_plan['visual_rows']
    assert len(combined) == 250 and all(row['actually_viewed'] for row in combined)
    for record in [previous_visual, visual]:
        for sheet in record['sheets']:
            assert sheet['actually_viewed'] and sha(ROOT / sheet['path']) == sheet['sha256']
    assert visual['previous_visual_review_sha256'] == sha(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis/visual_review.json')
    assert analysis['original_run_remains_storage_stopped'] and analysis['original_full_checker_not_relabelled_as_pass']
    assert arithmetic['logical_condition_union_verified'] == milestone['all_planned_conditions_covered'] == 30
    assert arithmetic['unique_completed_diagnostic_slots'] == milestone['unique_output_slots_audited'] == 3150
    assert 3045 + 315 - 210 == 3150
    assert analysis['CPU_replays_executed_across_both_audits'] == milestone['CPU_replay_executions'] == 160
    assert analysis['unique_current_batch_proposal_replays'] == milestone['unique_CPU_replay_slots'] == 150
    for summary in analysis['summaries']:
        assert summary['states'] == 10
        if summary['role'] != 'current_batch' or summary['stage'] == 'PNG':
            assert summary['finite_preservation_pass_updates'] == []
        else:
            assert summary['finite_preservation_pass_updates'] == [27]
    geometry = ROOT / 'outputs/cctv_dgp_output_geometry_v1'
    oracle, oracle_audit = read(geometry / 'analysis.json'), read(geometry / 'independent_audit.json')
    assert oracle_audit['complete'] and oracle_audit['analysis_sha256'] == sha(geometry / 'analysis.json')
    assert oracle_audit['cases_checked'] == 145 and oracle_audit['metrics_rows_checked'] == 870
    assert oracle['oracle_target_access_is_unavailable_in_app'] and oracle['network_representability_not_tested']
    assert oracle['full_preservation_gate_not_evaluated'] and oracle['ArcFace_not_evaluated']
    assert oracle['neural_calls'] == oracle['optimizer_updates'] == oracle['gradient_queries'] == 0
    assert not oracle['native_or_DEV_or_reserved_final_used'] and not oracle['app_promotion']
    assert milestone['diagnostic_coverage_complete'] and not milestone['new_training_protocol_prepared']
    assert not milestone['training_or_diagnostic_launched_on_VM_by_agent']
    assert not milestone['app_promotion'] and not milestone['model_qualification']
    assert not milestone['independent_final_review'] and not milestone['goal_complete']
    value = {'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'),
        'new_evidence_bindings_verified': len(milestone['new_evidence_sha256']),
        'original_return_members_rehashed': 9369, 'tail_return_members_rehashed': 959,
        'source_checkpoint_split_and_failure_bindings_exact': True,
        'tail_archive_bytes': archive.stat().st_size, 'app_bindings_verified': 14,
        'previous_handoff_exact_suffix_verified': True, 'all30_conditions_covered': True,
        'unique_audited_slots': 3150, 'all50_frozen_visual_sheets_accounted_for': True,
        'all250_frozen_visual_rows_accounted_for': True, 'exact256_cells_reviewed': 1500,
        'geometry_is_oracle_arithmetic_not_model_qualification': True,
        'original_failed_run_and_auditor_status_preserved': True,
        'no_new_pilot_or_training_launched': True, 'goal_complete': False,
        'seconds': time.monotonic() - start, 'cap_seconds': 600}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(value, flush=True)


if __name__ == '__main__':
    main()
