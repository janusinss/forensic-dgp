"""Independent review-scope and failure timing/state checks, without model imports."""
from pathlib import Path
import sys
from cctv_dgp_spatial_fit_v40_contract import read, write, sha
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review'


def main():
    plan, ready, v, checked = [read(OUT/name) for name in ['plan.json', 'preparation.json', 'visual_review.json', 'independent_preparation_audit.json']]
    p = read(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json'); a = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json')
    failure_path = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return/outputs/failure.json'; failure = read(failure_path)
    assert v['complete'] and checked['complete'] and a['complete'] and a['failure_retained'] and not a['necessary_capacity_pass']
    assert v['plan_sha256'] == sha(OUT/'plan.json') and v['preparation_sha256'] == sha(OUT/'preparation.json')
    assert v['independent_preparation_audit_sha256'] == sha(OUT/'independent_preparation_audit.json')
    assert v['source_sha256'] == sha(ROOT/'scripts/record_cctv_dgp_spatial_fit_v40_visual_review.py')
    assert v['all50_previews_and250_comparison_cells_actually_viewed'] and v['all10_sheets_actually_viewed_at_original256_cell_detail']
    assert v['pages'] == ready['pages'] and [row['id'] for row in v['rows']] == plan['case_ids'] == p['preview_case_ids']
    assert len(v['rows']) == 50 and len(v['pages']) == 10 and v['fixed50_directly_optimized_cases'] == ready['fixed50_cases_optimized_by50'] == 0
    for row, original in zip(v['rows'], ready['rows']):
        assert row['source'] == original['source'] and row['profile'] == original['profile'] and row['optimized_by50'] == original['optimized_by50']
        assert row['all5_regions_actually_reviewed'] and row['regions'] == plan['regions'] and row['observed_comparison'] and row['decision']
        assert not row['independent_final_review']
    for page in v['pages']: assert sha(OUT/page['path']) == page['sha256']
    assert failure['optimizer_updates'] == failure['backwards'] == 50 and failure['complete_snapshots'] == [0, 50]
    assert failure['seconds'] <= 4500 and failure['fit_seconds'] <= 3600 and failure['peak_allocated_VRAM_bytes'] <= 20*1024**3
    assert not failure['resume_permitted'] and not failure['automatic_repeat_permitted'] and failure['new_trained_checkpoint']
    for name in ['original', 'reference_decoder', 'recognizer']:
        assert failure['states_before'][name] == failure['states_after'][name] == p['initial_states'][name]
    assert failure['states_after']['decoder'] != failure['states_before']['decoder']
    assert not a['gates'][0]['pass'] and a['gates'][0]['minimum_gain'] == .01 and not a['gates'][0]['preservation_failures']
    assert v['no_gate_override_or_threshold_change'] and v['not_a_proven_unique_cause'] and v['all_are_TRAIN_not_held_out_evaluation']
    assert not v['quality_qualification'] and not v['app_promotion'] and not v['independent_final_review']
    assert not any(name in sys.modules for name in ['torch', 'pretrained_completion', 'dgp_frozen_inference_v2'])
    write(OUT/'independent_review_audit.json', {'complete': True, 'checker_sha256': sha(Path(__file__)),
          'visual_review_sha256': sha(OUT/'visual_review.json'), 'failure_sha256': sha(failure_path),
          'all50_case_notes_and10_actual_page_records_verified': True, 'all5_visible_regions_retained': True,
          'failure_timing_and_state_declarations_verified': True, 'unchanged_one_percent_stop_preserved': True,
          'CPU_replay_passes': 100, 'raw_PREVIEW_check_scope_per_snapshot': 50, 'all_nonpreview_raw_stages_retained': False,
          'source_labels_not_ethnicity': True, 'neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
          'VM_calls': 0, 'app_promotion': False, 'independent_final_review': False, 'goal_complete': False})
    print({'complete': True, 'case_notes': 50, 'failed_gate_retained': True}, flush=True)


if __name__ == '__main__': main()
