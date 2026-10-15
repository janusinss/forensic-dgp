"""Review saved signed slopes, actual float32 changes, PNGs and inspected sheets."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_bank_group_step_check_v1_return import ROOT, RETURN, ANALYSIS, PARENT, read, sha, write
from cctv_dgp_signed_step_audit import group_loss

NOTES = {
    'b036': 'Clear glasses, lips and fine hair are softened in the retained output and all four proposals. Degraded eyes/mouth remain indistinct; no convincing proposal advantage.',
    'b051': 'Clear eyelids, skin texture and white hair are softened. All four degraded proposals remain close to the blurred retained estimate; no useful new detail.',
    'b076': 'Hair strands and eye/lip borders are soft in all estimates. The four signed proposals do not visibly resolve the degraded face or add convincing clarity.',
    'b085': 'Clear eyes and individual teeth are softer than the input. Degraded mouth/eye areas remain blurred across all proposals; no convincing useful gain.',
    'b092': 'Smile/teeth, eyelids and bangs remain softened. The four proposals have no convincing advantage over the retained result in the degraded rows.',
    'b146': 'Clear iris edges and teeth are smoothed. The degraded eye/mouth estimates remain very similar for both signs and radii; useful new detail is absent.',
    'b169': 'The visible eyes, smile/teeth and graduation clothing remain recognizable but soft. All four proposals leave degraded facial detail indistinct.',
    'b357': 'Visible hair/eye/lip borders are smoother than the clear input; ordinary hair remains present. All four degraded estimates remain soft with no convincing clarity gain.',
    'b384': 'Clear beard/skin detail, eyebrows and hair are softened. The signed candidates do not visibly sharpen the degraded eyes, mouth or beard over the retained output.',
    'b386': 'Clear glasses, teeth and hair are soft after restoration. The degraded eyewear/eye area remains indistinct in all proposals; no convincing benefit.',
    'b391': 'Moustache, eye and mouth edges remain smooth. Both signed radii produce visually similar degraded estimates; no convincing whole-face detail gain.',
    'b392': 'The open mouth and dark eye outlines remain soft, especially in blur/compound rows. All proposals are close to the retained output without convincing extra structure.',
    'b393': 'The cheek, nose and mouth remain smooth across all signed outputs. Degraded eyes/lips remain vague; the new proposals do not visibly recover useful details.',
    'b394': 'Visible gaze and open mouth are retained as rough shapes but are soft in degraded outputs. No convincing improvement among the four proposals.',
    'b395': 'Visible hand beside the face is retained; this is no completion test. Clear skin/eyes are smoothed and degraded eyes/mouth remain indistinct in every proposal.',
    'b480': 'The smile and teeth remain recognizable but soft. Degraded eyelids and nose/mouth detail show no convincing gain from either signed radius.',
    'b615': 'Side padding stays fixed. Visible smile/teeth and ordinary hair remain soft; the forehead hand is not being completed. No convincing detail gain among the four proposals.',
    'b651': 'Side padding stays fixed. Ordinary hair and the visible eye/mouth shapes remain present but soft. The four degraded estimates have no convincing advantage; no covering completion is tested.',
    'b701': 'Side padding and the visible head covering remain present. The face is softened, especially eyes/lips in degraded rows; no convincing signed-step clarity gain.',
    'b745': 'Side padding stays fixed. Smile/teeth and eyes are softened; degraded estimates remain indistinct across the four signed proposals.',
}


def rgb(path, mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256, 256)
        return np.asarray(image.convert(mode)).copy()


def main():
    start = time.monotonic()
    for name in ['actual_step_direction_review.json', 'quantization_review.json', 'visual_review.json']:
        assert not (ANALYSIS/name).exists(), 'Retain immutable review: ' + name
    audit_path = ANALYSIS/'independent_audit_r1.json'
    audit, finite, signed, protocol = [read(path) for path in [audit_path,
        ANALYSIS/'finite_metrics.json', ANALYSIS/'signed_derivative_review.json', RETURN/'protocol.json']]
    assert all(record['complete'] for record in [audit, finite, signed])
    assert sha(ANALYSIS/'finite_metrics.json') == audit['finite_metrics_sha256']
    assert sha(ANALYSIS/'signed_derivative_review.json') == audit['signed_derivative_review_sha256']
    assert audit['quality_gates_and_exact_parameter_hashes_unchanged']
    old = read(PARENT/'protocol.json')
    group_root = ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_r1_return/outputs/group_gradients_v1_r1'
    groups = read(group_root/'run_results.json')['groups']
    predictions = {row['key']: row for row in read(RETURN/'direction_review.json')['float32_group_predictions']}
    assert len(groups) == len(predictions) == 32
    assert all(predictions[row['key']]['loss'] == row['value'] for row in groups)
    initial = torch.load(PARENT/'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    names = protocol['learning_names']
    assert len(names) == 37
    increments = {}
    changes = []
    with np.load(RETURN/'directions.npz', allow_pickle=False) as data:
        for proposal in protocol['proposals']:
            label, rate = proposal['label'], np.float32(proposal['rate'])
            receipt = read(RETURN/'outputs/group_step_check_v1'/label/'parameter_receipt.json')
            delta = []
            for i, name in enumerate(names):
                value = initial[name].numpy() + rate*data[f'balanced_p{i:02d}']
                assert hashlib.sha256(value.tobytes()).hexdigest() == receipt['selected_parameter_value_sha256'][name]
                delta.append((value.astype(np.float64)-initial[name].numpy().astype(np.float64)).reshape(-1))
            increments[label] = np.concatenate(delta)
            statistics = list(receipt['selected_parameter_change_statistics'].values())
            changes.append(dict(proposal=label, changed_values=sum(s['changed_values'] for s in statistics),
                values=sum(s['values'] for s in statistics),
                actual_L2=math.sqrt(math.fsum(s['squared_L2'] for s in statistics)),
                nominal_L2=math.sqrt(math.fsum(s['nominal_squared_L2'] for s in statistics)),
                rounding_error_L2=math.sqrt(math.fsum(s['rounding_error_squared_L2'] for s in statistics))))
    gradient_records = []
    for row in groups:
        path = group_root/row['file']
        assert sha(path) == row['sha256']
        vector = np.load(path, allow_pickle=False)
        assert vector.dtype == np.float64 and vector.shape == (3364643,)
        prediction = predictions[row['key']]
        step_predictions = {label: float(vector@increment) for label, increment in increments.items()}
        pairs = []
        for h, rate in [(.001, '0.001'), (.0001, '0.0001')]:
            plus = group_loss(finite['proposals']['balanced_'+rate]['groups']['raw'], prediction)
            minus = group_loss(finite['proposals']['balanced_-'+rate]['groups']['raw'], prediction)
            central = (plus-minus)/(2*h)
            actual_dot = (step_predictions['balanced_'+rate]-step_predictions['balanced_-'+rate])/(2*h)
            pairs.append(dict(radius=h, measured_central_slope=central,
                nominal_saved_gradient_dot=prediction['dot'],
                gradient_dot_for_actual_float32_parameter_changes=actual_dot,
                error_after_accounting_for_actual_parameter_rounding=central-actual_dot,
                relative_error_after_parameter_rounding=abs(central-actual_dot)/max(abs(actual_dot), 1e-30)))
        gradient_records.append(dict(key=row['key'], source=row['source'], metric=row['metric'],
            saved_gradient_sha256=row['sha256'], first_order_change_for_actual_saved_steps=step_predictions, pairs=pairs))
    assert all(p['measured_central_slope'] < 0 for row in gradient_records for p in row['pairs'])
    direction = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__), independent_audit_sha256=sha(audit_path),
        finite_metrics_sha256=sha(ANALYSIS/'finite_metrics.json'), signed_derivative_review_sha256=sha(ANALYSIS/'signed_derivative_review.json'),
        original_group_run_sha256=sha(group_root/'run_results.json'), groups=gradient_records,
        parameter_changes=changes, raw_central_slopes_negative_at_both_radii=32,
        matched_initial_controls=100, no_gradient_sign_reversal_demonstrated_in_signed_central_test=True,
        this_does_not_prove_all_backward_paths_correct=True,
        float32_rounding_is_accounted_for_not_assumed_to_explain_every_difference=True,
        PNG_quantization_separate_from_raw_derivative=True,
        new_neural_gradient_optimizer_or_actual_model_displacements=0,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False)
    write(ANALYSIS/'actual_step_direction_review.json', direction)

    sys.path.insert(0, str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    refs = {row['id']: row for row in old['references']}
    metric_rows = {label: {row['id']: row for row in trial['rows']}
                   for label, trial in finite['proposals'].items()}
    rows = []
    run = read(RETURN/'outputs/group_step_check_v1/run_results.json')
    for control in run['controls']:
        begin = control['begin']
        cases = old['cases'][begin:begin+5]
        baseline, _ = unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz', [c['id'] for c in cases])
        for proposal in protocol['proposals']:
            label = proposal['label']
            values = np.load(RETURN/f'outputs/group_step_check_v1/{label}/b{begin//5:03d}.npy', allow_pickle=False)
            for index, case in enumerate(cases):
                ref = refs[case['source_person_or_reference']]
                area = rgb(PARENT/ref['observed'], 'L') > 0
                raw_before, raw_after = baseline[index], values[index]
                png_before = np.floor(raw_before*np.float32(255)).astype(np.uint8)
                png_after = np.floor(raw_after*np.float32(255)).astype(np.uint8)
                raw_shift = raw_after[area].astype(np.float64)-raw_before[area].astype(np.float64)
                png_shift = png_after[area].astype(np.int16)-png_before[area].astype(np.int16)
                assert not np.any(raw_after[~area] != raw_before[~area])
                metric = metric_rows[label][case['id']]
                prior_metric = next(row for row in finite['baseline_rows'] if row['id'] == case['id'])
                raw_change = metric['raw']['MSE']-prior_metric['raw']['MSE']
                png_change = metric['png']['MSE']-prior_metric['png']['MSE']
                rows.append(dict(proposal=label, id=case['id'], source=case['source'], profile=case['profile'],
                    observed_channels=int(raw_shift.size),
                    raw_changed_channels=int(np.count_nonzero(raw_shift)),
                    PNG_changed_channels=int(np.count_nonzero(png_shift)),
                    PNG_changed_pixels=int(np.count_nonzero(np.any(png_shift != 0, axis=1))),
                    maximum_absolute_raw_channel_change=float(np.max(abs(raw_shift))),
                    maximum_absolute_PNG_channel_change=int(np.max(abs(png_shift))),
                    raw_MSE_change=raw_change, PNG_MSE_change=png_change,
                    PNG_minus_raw_MSE_change=png_change-raw_change,
                    raw_MSE_improved_but_PNG_MSE_worsened=raw_change < 0 and png_change > 0))
    assert len(rows) == 400
    summary = {}
    for label in finite['proposals']:
        subset = [row for row in rows if row['proposal'] == label]
        summary[label] = dict(cases=len(subset), raw_changed_channels=sum(r['raw_changed_channels'] for r in subset),
            PNG_changed_channels=sum(r['PNG_changed_channels'] for r in subset),
            PNG_changed_pixels=sum(r['PNG_changed_pixels'] for r in subset),
            maximum_raw_channel_change=max(r['maximum_absolute_raw_channel_change'] for r in subset),
            maximum_PNG_channel_change=max(r['maximum_absolute_PNG_channel_change'] for r in subset),
            raw_MSE_improved_PNG_worsened_cases=sum(r['raw_MSE_improved_but_PNG_MSE_worsened'] for r in subset))
    write(ANALYSIS/'quantization_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__), independent_audit_sha256=sha(audit_path), finite_metrics_sha256=sha(ANALYSIS/'finite_metrics.json'),
        rows=rows, summaries=summary, all400_cases_recounted=True,
        converter='Existing floor(float32*255); no alternative image generated',
        this_is_processing_origin_not_visible_identity_or_completion_accuracy=True,
        ArcFace_is_saved_embedding_evidence_not_recomputed_by_new_neural_inference=True,
        zero_new_neural_or_gradient_or_optimizer_calls=True, app_edit=False,
        thresholds_and_floor_policy_unchanged=True, model_qualified=False, goal_complete=False))
    sheets = []
    for row in audit['comparison_sheets']:
        path = ROOT/row['file']
        assert sha(path) == row['sha256'] and len(row['cells']) == 40
        assert path.stem in NOTES
        sheets.append(dict(file=row['file'], sha256=row['sha256'], all40_cells_inspected=True,
            requested_display_detail='original', source_dimensions=[2048, 1460],
            observations=NOTES[path.stem], convincing_useful_gain_over_retained_DGP=False))
    assert len(sheets) == len(NOTES) == 20
    write(ANALYSIS/'visual_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer='Primary assistant direct development visual review; not independent final human review',
        record_script_sha256=sha(__file__), raw_and_PNG_audit_sha256=sha(audit_path),
        all20_sheets_and800_cells_inspected=True, sheets=sheets, cells_inspected=800,
        proposals=4, unique_TRAIN_references=20, inputs=100,
        positive_small_step_raw_preservation_does_not_establish_delivered_PNG_qualification=True,
        no_convincing_whole_face_detail_gain_observed=True,
        model_adoption=False, independent_final_review=False, native_CCTV_pixels_reviewed=0,
        final_identity_pixels_reviewed=0, completion_families_tested=0,
        no_new_neural_calls_gradients_updates_or_VM_connections=True, goal_complete=False))
    print(dict(complete=True, saved_gradient_groups=32, actual_change_predictions=128,
        quantization_cases=400, visually_inspected_sheets=20, seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
