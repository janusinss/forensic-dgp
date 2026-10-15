"""Bind every directly viewed sheet and the human's diagnostic direction."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, RETURN, ANALYSIS, PIN, read, sha, write

OBSERVATIONS = {
    36: 'Clear glasses and lip/skin detail remain softened. Degraded eyes and glasses stay diffuse across all three trials; no convincing clarity advantage.',
    51: 'Clear eye/skin and hair detail remain softened. Blur and compound profiles lack well-defined eyes and mouth in all candidates.',
    76: 'Clear hair, eye and lip edges remain soft. All degraded candidates retain diffuse eyes, nose and lip line; trial differences are subtle.',
    85: 'Clear smile and teeth remain coarse. Degraded eyes, teeth and hair remain blurred; none of the trials resolves them convincingly.',
    92: 'Clear smile, teeth and hair are softened. Degraded face outline and eyes remain diffuse in all trial columns.',
    146: 'Clear smile and eye outline remain coarse. Degraded eyes and mouth stay poorly resolved with no substantial visible improvement.',
    169: 'Clear expression, teeth and hair retain the baseline softness. Degraded smile and eyes stay blurred in all three trials.',
    357: 'Clear eye, lip and hair boundaries soften. Turned-face degraded profiles remain diffuse; none of the candidates establishes better visible structure.',
    384: 'Clear brow, facial hair and mouth details are soft. Degraded eyes and nose/lip contours remain blurred throughout the trial columns.',
    386: 'Clear glasses and teeth remain coarse. Degraded eyewear, eyes and mouth remain indistinct in all trial outputs.',
    391: 'Clear facial-hair and eye/lip contours retain softened detail. Degraded eyes and mouth remain blurred; no substantial whole-face advantage.',
    392: 'Clear eye and open-mouth outlines remain coarse. The degraded candidates leave eyes, lips and teeth diffuse.',
    393: 'Clear eye and lip lines remain coarse. Degraded eye and mouth boundaries remain blurred in all three trials.',
    394: 'Clear gaze and open-mouth outlines are softened. Degraded eye and lip details remain poorly resolved across all candidates.',
    395: 'The visible hand remains in every column. Clear brows and smile are softened; degraded face details remain diffuse. This is not a completion test.',
    480: 'Clear smile and teeth remain coarse. Degraded eyes, nose and teeth show no convincing structural clarity gain.',
    615: 'Neutral side padding and the visible hand remain. Clear teeth and hair are softened; degraded eyes and smile remain blurred. No completion claim.',
    651: 'Neutral padding and visible hair remain. Clear skin and lip details are softened; degraded eye and mouth outlines remain diffuse.',
    701: 'Neutral side padding remains. Clear visible gaze and lip lines are soft; degraded eye and mouth detail remains blurred across trials.',
    745: 'Neutral padding remains. Clear eye and teeth details are softened; degraded eyes and smile remain poorly resolved.',
}


def main():
    audited = read(ANALYSIS / 'failure_audit.json')
    prediction = read(ANALYSIS / 'gradient_prediction_review.json')
    assert audited['complete'] and prediction['complete'] and audited['protocol_sha256'] == PIN
    assert audited['completed_trials'] == 3 and not audited['accepted_trials_verified']
    assert audited['completed_snapshots_audited'] == ['initial', 'r00_trial00', 'r00_trial01', 'r00_trial02']
    rows = []
    for sheet in audited['all_planned_visual_sheets']:
        path = ROOT / sheet['file']
        index = int(path.stem[1:])
        assert index in OBSERVATIONS and sha(path) == sheet['sha256'] and sheet['cells'] == 30
        rows.append(dict(file=sheet['file'], sha256=sheet['sha256'], cells=30,
                         all30_cells_directly_viewed=True, requested_image_detail='original',
                         observations=OBSERVATIONS[index], useful_whole_face_improvement_qualified=False))
    assert len(rows) == 20 and {int(Path(v['file']).stem[1:]) for v in rows} == set(OBSERVATIONS)
    write(ANALYSIS / 'visual_review.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), reviewer='Implementing assistant; direct saved-sheet inspection',
        reviewer_sha256=sha(__file__), failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'), sheets=rows,
        all20_sheets_and600_cells_directly_viewed=True, exact_original_initial_control_and_three_rejected_trials=True,
        photographic_TRAIN_references=20, synthetic_paired_cases=100, saved_RAW_and_PNG_model_outputs=400,
        no_convincing_whole_face_visual_advantage=True, no_candidate_selected_or_promoted=True,
        independent_final_review=False, native_CCTV_used=False, hidden_identity_or_ethnicity_or_Zamboanga_claim=False,
        automatic_or_assisted_completion_qualification=False, model_qualified=False, goal_complete=False))
    write(ANALYSIS / 'diagnostic_decision.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), recorder_sha256=sha(__file__),
        human_question='Before another training pilot, which cause should the design review investigate first? '
                       'R4’s predicted improvements did not hold in the actual raw/PNG preservation checks.',
        human_answer='apply best approach',
        chosen_direction='Audit saved gradient predictions against actual RAW/PNG changes before another training pilot',
        invalid_assumption='Favorable local descent constraints and a small relative parameter step establish finite image preservation',
        three_stops_retained=['R2 startup shape stop', 'R3 SSIM-forward alignment stop', 'R4 actual preservation stop'],
        distinct_startup_versus_model_quality_causes=True, code_changes_stopped_before_question=True,
        diagnostic_question_answered_before_further_diagnostic_review=True,
        saved_prediction_review_sha256=sha(ANALYSIS / 'gradient_prediction_review.json'),
        no_automatic_failed_recipe_continuation_or_new_training_launch=True,
        actual_neural_gradient_or_training_calls_in_this_review=0,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    assert all(sha(ROOT / n) == h for n, h in read(ANALYSIS / 'local_protected_before.json')['files_sha256'].items())
    print(dict(complete=True, sheets_directly_viewed=20, cells=600,
               human_direction_recorded=True, new_incumbent=False))


if __name__ == '__main__':
    main()
