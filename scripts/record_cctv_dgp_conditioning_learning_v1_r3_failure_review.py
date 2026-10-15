"""Retain incomplete-gradient evidence and the completed direct visual review."""
from datetime import datetime, timezone
import re
import numpy as np

from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

OBSERVATIONS = {
    36: 'Clear glasses, eyes, lips and skin detail are softened; degraded outputs remain diffuse.',
    51: 'Clear eye, forehead and mouth detail is softened; degraded features remain blurred.',
    76: 'Clear hair and eye/lip edges are softened; degraded eyes and mouth remain indistinct.',
    85: 'Clear smile remains visible but teeth/lip/hair detail is softened; degraded outputs stay soft.',
    92: 'Smile and face outline remain coarse; eye, teeth and hair detail are softened.',
    146: 'Clear eye and teeth detail is softened; degraded eyes and smile remain poorly resolved.',
    169: 'Clear smile is retained coarsely; eye and hair detail are softened in every degraded profile.',
    357: 'Clear eyes, mouth and hair texture are softened; turn/expression detail remains diffuse when degraded.',
    384: 'Clear eyes, stubble and mouth detail are softened; degraded face remains smooth and blurred.',
    386: 'Clear glasses placement and smile remain coarse; degraded eyes and glasses are indistinct.',
    391: 'Coarse mouth and facial-hair outline remains; fine details soften and degrade into blur.',
    392: 'Clear eye/mouth outline remains coarse; degraded eyes and open-mouth details are blurred.',
    393: 'Clear expression is retained coarsely; degraded eyes, nose and lip line are diffuse.',
    394: 'Clear gaze/open-mouth outline remains coarse; blurred profile loses eye and lip detail.',
    395: 'Visible hand stays in the input; clear brow, smile and facial texture are softened.',
    480: 'Clear smile remains coarse; degraded teeth/eye details are blurred rather than recovered.',
    615: 'Neutral side padding remains; visible hand/teeth/hair details are softened.',
    651: 'Neutral side padding remains; clear eye/lip/skin and fringe details are softened.',
    701: 'Neutral side padding remains; visible gaze, mouth and facial detail are soft when degraded.',
    745: 'Neutral side padding remains; clear smile/teeth are softened and degraded features remain diffuse.',
}


def main():
    a = read(ANALYSIS / 'failure_audit.json')
    imported = read(ANALYSIS / 'import.json')
    p = read(RETURN / 'protocol.json')
    stage = RETURN / 'outputs/conditioning_learning_v1_r3'
    f = read(stage / 'failure.json')
    red, green = [read(ANALYSIS / n) for n in ['loss_forward_red.json', 'loss_forward_green.json']]
    assert all(v['complete'] for v in [a, imported, green]) and not red['complete']
    assert f == a['failure'] and f['message'] == 'Scientific/surrogate loss alignment differs'
    counts = dict(B=34, recognizer=67, gradient_queries=420, backwards=0, accepted_updates=0, parameter_proposals=0)
    assert f['counts'] == f['progress']['counts'] == counts
    assert f['progress']['stage'] == 'fresh_group_gradients' and f['progress']['round'] == 0
    assert imported['returned_file_bindings'] == 166 and imported['parent_file_bindings'] == 5852
    assert a['completed_snapshots_audited'] == ['initial'] and not a['completed_trials']
    assert a['accepted_trials_verified'] == [] and a['actual_update_count_reported'] == 0
    assert not list(stage.glob('accepted*')) and not list(stage.glob('r??_trial*'))
    assert not (stage / 'run_results.json').exists() and not (stage / 'round00/gradients.json').exists()
    canonical = read(stage / 'canonical_pixels.json')
    assert canonical['complete'] and canonical['all256_byte_levels']
    completed_batches, query = divmod(counts['gradient_queries'], 32)
    assert completed_batches == 13 and query == 4
    block = p['reference_blocks'][completed_batches]
    expected_case = block['case_ids'][query // 3]
    assert expected_case == 'v9_tr_asian_00180_blur_lr24' and query % 3 == 1
    assert counts['B'] == 20 + completed_batches + 1 and counts['recognizer'] == 40 + 2 * completed_batches + 1
    failed_row = red['failures'][0]
    assert len(red['failures']) == 1 and failed_row['id'] == expected_case
    assert failed_row['stage'] == 'raw' and failed_row['term'] == 'SSIM_loss'
    assert red['real_saved_loss_checks'] == green['real_saved_loss_checks'] == 600
    assert not green['failures'] and green['maximum_RAW_SSIM_difference'] < 1e-13
    assert green['checker_sha256'] == red['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_conditioning_loss_forward_v1.py')
    partial = []
    for path in sorted((stage / 'round00/samples').glob('*.npy')):
        assert re.fullmatch(r'(RAW|PNG)_b(036|391)\.npy', path.name)
        v = np.load(path, allow_pickle=False)
        assert v.shape == (p['conditioning_parameters'],) and v.dtype == np.float32 and np.isfinite(v).all()
        assert np.linalg.norm(v.astype(np.float64)) > 0
        partial.append(dict(file=path.relative_to(RETURN).as_posix(), sha256=sha(path),
                            elements=len(v), norm=float(np.linalg.norm(v.astype(np.float64))),
                            single_clear_MSE_derivative_not_full_group_aggregate=True))
    assert len(partial) == 4
    projection = read(stage / 'round00/timing_projection.json')
    assert projection['cap_seconds'] == 600 and projection['projection_seconds'] <= 600
    assert np.isclose(max(projection['completed_batch_seconds']) * 20 * 1.5, projection['projection_seconds'])
    sheets = a['all_planned_visual_sheets']
    assert len(sheets) == 20
    rows = []
    for sheet in sheets:
        path = ROOT / sheet['file']
        assert sha(path) == sheet['sha256'] and sheet['cells'] == 15
        number = int(path.stem[1:])
        assert number in OBSERVATIONS
        rows.append(dict(sheet, all15_cells_directly_viewed=True, requested_image_detail='original',
                         observations=OBSERVATIONS[number], candidate_after_training=False))
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(ANALYSIS / 'visual_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          reviewer='Implementing assistant; direct saved-sheet inspection', reviewer_script_sha256=sha(__file__),
          failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'), sheets=rows,
          all20_sheets_and300_cells_directly_viewed=True, original_DGP_initial_controls_only=True,
          new_trained_outputs=0, no_visual_improvement_over_original_claimed=True,
          independent_final_review=False, native_CCTV_used=False, model_qualified=False, goal_complete=False))
    write(ANALYSIS / 'loss_alignment_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          reviewer_script_sha256=sha(__file__), import_sha256=sha(ANALYSIS / 'import.json'),
          failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'),
          protocol_sha256=PIN, returned_archive_sha256=imported['archive_sha256'],
          reported_VM_seconds=f['seconds'], reported_counts=counts,
          initial100_RAW_controls_exact_and200_RAW_PNG_rows_audited=True,
          complete_gradient_rounds=0, retained_partial_gradient_samples=partial,
          all420_neural_queries_not_independently_reexecuted=True,
          inferred_stopped_batch=block, inferred_stopped_stage='RAW', inferred_query=4,
          inference_basis='13 completed log batches then 4 additional queries; RAW queries0-15 precede PNG.',
          failed_CPU_row=failed_row, actual_failed_GPU_scalar_not_exported=True,
          original_RAW_float64_surrogate_versus_float32_scientific_mismatch_reproduced_on_CPU=True,
          red_regression_sha256=sha(ANALYSIS / 'loss_forward_red.json'),
          green_regression_sha256=sha(ANALYSIS / 'loss_forward_green.json'),
          current_scientific_quality_and_alignment_bounds_unchanged=True,
          corrected_helper_sha256=green['raw_helper_sha256'],
          canonical_forward_unchanged_approximate_derivative_new_manual_run_pending=True,
          no_initial_or_post_failure_hash_claim_beyond_audited_saved_initial_state=True,
          more_epochs_or_actual_structure_gate_not_reached=True,
          partial_run_failure_preserved=True, original_app_and_checkpoints_unchanged=True,
          local_neural_gradient_optimizer_parameter_displacement_calls=0,
          new_manual_training_unrun=True, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, actual_updates=0, inferred_failed_case=expected_case,
               partial_samples=4, sheets_directly_viewed=20, canonical_RAW_PNG_forward_checks=600))


if __name__ == '__main__':
    main()
