"""Bind the completed visual review and diagnose saved first-order/finite evidence."""
import ast
from datetime import datetime, timezone
import json
import numpy as np
from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write

VIEWED_NOTES = json.loads(r'''[{"sheet":"b036","note":"Clear glasses, eyes and lips remain softer than the target. All four degraded profiles remain blurred around glasses and mouth; no convincing clarity gain. The 1e-4 outputs show small colour changes."},{"sheet":"b051","note":"Clear forehead and hair detail remain softened. Degraded eyes and mouth remain blurred; ordinary 1e-4 adds a warmer facial cast, particularly lowlight and compound."},{"sheet":"b076","note":"Clear eye and hair detail is softened. Strong ordinary proposal changes colour more than facial clarity; all degraded outputs remain soft around eyes and lips."},{"sheet":"b085","note":"Clear eyebrows and teeth remain softened. Larger ordinary proposal adds warmth/redness; degraded eyes and teeth remain indistinct across proposals."},{"sheet":"b092","note":"Smile, eyes and hair keep their broad layout; teeth and eyelid edges remain softer than the target. Degraded outputs remain blurred; larger ordinary proposal changes facial colour."},{"sheet":"b146","note":"Eyes and teeth remain softened, with no convincing recovery in degraded profiles. Larger ordinary proposal adds red/warm facial tone; smaller proposals are close to baseline."},{"sheet":"b169","note":"Clear eyelashes, teeth and earrings remain softened. Larger proposal changes skin and dark/blue background colour; degraded eyes and teeth remain indistinct."},{"sheet":"b357","note":"Eye and lip edges remain soft. Larger ordinary proposal changes dark hair/background and facial tone more than clarity; degraded faces remain blurred."},{"sheet":"b384","note":"Clear eyebrows and beard remain softer than input and target. Larger ordinary proposal warms/reddens skin and darkens hair in blur and lowlight. Degraded details remain soft; no convincing whole-face clarity gain."},{"sheet":"b386","note":"Clear glasses rims and teeth remain softened. Larger ordinary proposal changes skin and background toward a purplish tone; degraded glasses and eyes stay blurred across proposals."},{"sheet":"b391","note":"This is a derivation reference, not independent development evidence. Clear moustache and eye detail remain soft. Larger proposals alter skin and grey background colour, without convincing eye or moustache clarity in degraded profiles."},{"sheet":"b392","note":"Clear mouth and eyelid edges remain soft. Larger ordinary proposal warms/reddens midtones across profiles; blurred and motion-degraded mouths remain soft, with little convincing clarity difference."},{"sheet":"b393","note":"Clear eyelid and mouth boundaries remain softened; degraded eyes and lips are blurred. Larger ordinary proposal lightens/reddens darker profiles, without convincing detail recovery."},{"sheet":"b394","note":"Clear eyelids and open-mouth edges remain soft compared with target. Degraded eyes and mouth remain blurry; larger ordinary proposal changes skin/background colour more than clarity."},{"sheet":"b395","note":"Clear brows, eye edges and smile are smoothed, while the visible hand remains present. Larger ordinary proposal warms/redens the face. Degraded profiles retain soft eye and mouth detail across proposals."},{"sheet":"b480","note":"Clear tooth boundaries and eye wrinkles remain softer than target. Larger ordinary proposal changes facial tone; all four degraded profiles still lack crisp eye and tooth detail."},{"sheet":"b615","note":"Clear eyes, hair strands and tooth edges remain soft. The visible hand and non-obstructing hair remain present. Degraded outputs remain heavily blurred; larger ordinary proposal alters skin tone without convincing recovery."},{"sheet":"b651","note":"Clear hair and eyelid details are softened. Larger ordinary proposal changes red/purple skin and hair tone. Degraded eyes and lips remain indistinct across proposals; no convincing clarity gain."},{"sheet":"b701","note":"Clear eyelids, mouth and skin lines are smoothed. Larger ordinary proposal changes facial colour; degraded eyes and lips remain blurred, with white fabric retained."},{"sheet":"b745","note":"Clear eye and tooth edges remain softened. Larger ordinary proposal changes skin and dark hair colour. All degraded profiles remain blurred around eyes and teeth, with no convincing whole-face clarity gain."}]''')


def reference_rows(rows, reference):
    selected = [r for r in rows if r['id'].startswith('v9_'+reference+'_')]
    assert len(selected) == 5 and selected[0]['profile'] == 'clear'
    return selected


def losses(rows, stage):
    all_metrics = [r[stage] for r in rows]
    degraded = [r[stage] for r in rows if r['profile'] != 'clear']
    assert len(rows) == 5 and len(degraded) == 4
    return dict(detail=float(np.mean([r['landmark_high_frequency_MSE'] for r in degraded])),
                clear_MSE=all_metrics[0]['MSE'],
                degraded_MSE=float(np.mean([r['MSE'] for r in degraded])),
                SSIM=float(np.mean([1-r['SSIM'] for r in all_metrics])),
                ArcFace=float(np.mean([1-r['ArcFace_observed_fixed'] for r in all_metrics])))


def summarize(before, after, stage):
    assert len(before) == len(after) and [r['id'] for r in before] == [r['id'] for r in after]
    names = ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE']
    a = {n:float(np.mean([r[stage][n] for r in before])) for n in names}
    b = {n:float(np.mean([r[stage][n] for r in after])) for n in names}
    return dict(cases=len(before), baseline=a, candidate=b,
                structure_gain_fraction=1-b['landmark_high_frequency_MSE']/a['landmark_high_frequency_MSE'],
                metric_changes={n:b[n]-a[n] for n in names})


def main():
    audit_path = ANALYSIS/'independent_audit.json'
    finite_path = ANALYSIS/'finite_metrics.json'
    audit, finite = read(audit_path), read(finite_path)
    assert audit['complete'] and finite['complete'] and sha(finite_path) == audit['finite_metrics_sha256']
    assert audit['finite_raw_case_outputs_checked'] == audit['exact_floor_PNGs_checked'] == 400
    assert audit['independent_raw_and_PNG_metric_rows'] == 800
    visual_path, review_path = ANALYSIS/'visual_review.json', ANALYSIS/'finite_direction_review.json'
    assert not visual_path.exists() and not review_path.exists()
    sheets = {str(v['file']).replace('\\','/').split('/')[-1][:-4]:v for v in audit['comparison_sheets']}
    assert len(VIEWED_NOTES) == 20 and {r['sheet'] for r in VIEWED_NOTES} == set(sheets)
    visual_rows = []
    for row in VIEWED_NOTES:
        bound = sheets[row['sheet']]
        assert sha(ROOT/bound['file']) == bound['sha256'] and len(bound['cells']) == 40
        visual_rows.append(dict(**row, file=bound['file'], sha256=bound['sha256'],
            cases=[v['case'] for v in bound['cells'] if v['column'] == 0],
            exact_bound_cells=40, inspected_at_original_resolution=True,
            convincing_whole_face_clarity_gain=False, promotion_recommended=False))
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == value for name,value in protected.items())
    write(visual_path, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), independent_audit_sha256=sha(audit_path),
        finite_metrics_sha256=sha(finite_path), all20_sheets_inspected=True,
        all800_bound_cells_inspected=True, all400_candidate_PNGs_inspected=True,
        review_scope='Human-visible assistant review of paired photographic TRAIN sheets; not an independent final reviewer',
        reviewed_sheets=visual_rows,
        observations='Degraded eyes, mouths and teeth remain soft; larger ordinary proposals change facial colour. No convincing whole-face clarity gain on the reviewed sheets.',
        existing_clear_glasses_hair_hands_and_fabric_reviewed_but_completion_not_tested=True,
        automatic_and_assisted_completion_results=0, native_CCTV_results=0, final_identity_results=0,
        raw_metrics_do_not_establish_PNG_preservation=True, paired_reference_results_not_native_or_Zamboanga_performance=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        model_qualified=False, goal_complete=False))

    direction_path = OUT/'cctv_dgp_bank_saved_direction_v1_analysis/direction_review.json'
    direction = read(direction_path)
    hinge_path = OUT/'cctv_dgp_bank_hinge_boundary_v1_return/outputs/loss_alignment_v1/run_results.json'
    hinge = read(hinge_path)
    assert sha(hinge_path) == direction['run_results_sha256']
    old = read(PARENT/'protocol.json')
    definitions = OUT/'cctv_dgp_bank_hinge_boundary_v1_return/cctv_dgp_bank_loss_alignment_v1.py'
    source = definitions.read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'components')
    assert "terms['landmark_structure'][1:].mean()" in ast.unparse(function)
    assert "terms['SSIM_loss'].mean()" in ast.unparse(function)
    assert 'arc.mean()' in ast.unparse(function)
    references = [row['reference'] for row in direction['sampled_gradient_bindings']]
    assert references == ['tr_ffhq_00084','tr_asian_00048']
    cohorts, predicted = {}, []
    weights, normalizers = old['reconstruction_weights'], old['normalizers']
    anchors = old['active_anchor_normalizers']
    assert anchors == [.0006738122269695833,.007011396786714702]
    scalar_checks = []
    for reference in references:
        row = next(r for r in hinge['rows'] if r['state'] == 'initial' and r['reference'] == reference)
        b = losses(reference_rows(finite['baseline_rows'], reference),'raw')
        recomputed = dict(detail=weights[3]*b['detail']/normalizers[3],
            clear_MSE=b['clear_MSE']/anchors[0], degraded_MSE=b['degraded_MSE']/normalizers[0],
            SSIM=weights[1]*b['SSIM']/normalizers[1], ArcFace=weights[2]*b['ArcFace']/normalizers[2])
        for name,value in recomputed.items():
            reported = row['component_values'][hinge['component_names'].index(name)]
            assert np.isclose(reported,value,rtol=0,atol=1e-6), (reference,name,reported,value)
            scalar_checks.append(dict(reference=reference,component=name,saved_scalar=reported,
                independent_scalar=value,maximum_allowed_error=1e-6))
        for name,proposal in finite['proposals'].items():
            rate = float(name.split('_')[1])
            label = name.split('_')[0]
            for stage in ['raw','png']:
                before = losses(reference_rows(finite['baseline_rows'],reference),stage)
                actual = losses(reference_rows(proposal['rows'],reference),stage)
                for record in [r for r in direction['constraint_records'] if r['reference'] == reference]:
                    component = record['component']
                    scalar = row['component_values'][hinge['component_names'].index(component)]
                    product = record['nominal_'+label+'_direction_product']
                    expected = -rate*product/scalar
                    observed = 1-actual[component]/before[component]
                    predicted.append(dict(proposal=name,stage=stage,reference=reference,component=component,
                        baseline_loss=before[component],candidate_loss=actual[component],
                        first_order_predicted_gain_fraction=expected,actual_gain_fraction=observed,
                        finite_minus_first_order_fraction=observed-expected,
                        prediction_is_raw_derivative_not_PNG_derivative=True,
                        nonlinear_or_numeric_finite_difference_not_an_optimizer_step=True))
    sources = sorted({r['source'] for r in finite['baseline_rows']})
    assert sources == ['dataset/asian_faces','dataset/thumbnails128x128']
    for name, proposal in finite['proposals'].items():
        record = dict(gates={}, cohorts={})
        for stage in ['raw','png']:
            gate = proposal['gates'][stage]
            failures = gate['preservation_failures']
            record['gates'][stage] = dict(structure_gain_percent=100*gate['relative_feature_gain'],
                source_structure_gain_percent={k:100*v for k,v in gate['source_feature_gains'].items()},
                preservation_group_metric_failures=len(failures),
                failures_by_metric={n:sum(r['metric']==n for r in failures) for n in ['MSE','SSIM','ArcFace_observed_fixed']},
                failures_are_group_metric_checks_not_counts_of_faces=True,
                preview_pass=gate['pass_all_preview_requirements'],brightness_gain_fraction=gate['brightness_gain_fraction'])
            for source_label in sources:
                for included in [True,False]:
                    ids=[r['id'] for r in finite['baseline_rows'] if r['source']==source_label and r['profile']!='clear' and
                        any(r['id'].startswith('v9_'+ref+'_') for ref in references)==included]
                    before=[r for r in finite['baseline_rows'] if r['id'] in ids]
                    after=[r for r in proposal['rows'] if r['id'] in ids]
                    assert len(before)==(4 if included else 36)
                    record['cohorts'].setdefault(stage,{})[source_label+('/derivation' if included else '/other9_TRAIN_references')]=summarize(before,after,stage)
            raw_set={(r['group'],r['metric']) for r in proposal['gates']['raw']['preservation_failures']}
            png_set={(r['group'],r['metric']) for r in proposal['gates']['png']['preservation_failures']}
            record['delivered_only_failures']=[dict(group=g,metric=m) for g,m in sorted(png_set-raw_set)]
            record['raw_only_failures']=[dict(group=g,metric=m) for g,m in sorted(raw_set-png_set)]
        assert not proposal['all_preview_requirements_pass']
        cohorts[name]=record
    larger = [k for k in cohorts if k.endswith('_0.0001')]
    smaller = [k for k in cohorts if k.endswith('_0.00001')]
    assert all(cohorts[k]['gates'][stage]['structure_gain_percent']>=1 for k in larger for stage in ['raw','png'])
    assert all(cohorts[k]['gates'][stage]['structure_gain_percent']<1 for k in smaller for stage in ['raw','png'])
    assert all(cohorts[k]['gates'][stage]['preservation_group_metric_failures']>0 for k in cohorts for stage in ['raw','png'])
    assert all(cohorts[k]['gates'][stage]['source_structure_gain_percent']['dataset/asian_faces/degraded']<0
        for k in larger for stage in ['raw','png'])
    sampled_constrained_small=[r for r in predicted if r['proposal']=='constrained_0.00001' and r['stage']=='raw']
    assert all(r['actual_gain_fraction']>0 for r in sampled_constrained_small if r['component']=='detail')
    assert any(r['actual_gain_fraction']<0 and r['first_order_predicted_gain_fraction']>0
        for r in sampled_constrained_small), 'Expected finite local counterexample missing'
    write(review_path,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),reviewer_sha256=sha(__file__),
        evidence_sha256={str(path.relative_to(ROOT).as_posix()):sha(path) for path in
            [audit_path,finite_path,visual_path,direction_path,hinge_path,definitions,PARENT/'protocol.json']},
        controls_and_masks_already_independently_audited=True, proposals=cohorts,
        first_order_vs_finite=predicted, sampled_scalar_recomputation=scalar_checks,
        detail_constraint_and_gate_both_use_four_degraded_profiles=True,
        SSIM_and_ArcFace_derivative_constraints_average_all5_not_individual_profiles=True,
        only2_of20_TRAIN_references_covered_by_gradient_projection=True,
        saved_gradient_and_finite_losses_match_on_sampled_baseline=True,
        numerical_execution_succeeded_scientific_preservation_failed=True,
        bigger_proposals_cross1percent_mean_structure_but_source_and_preservation_fail=True,
        smaller_proposals_have_less_structure_and_still_fail_preservation=True,
        invalid_assumption='Two-reference, first-order aggregate constraints would preserve all source/profile groups after a finite update.',
        next_diagnostic='Measure gradients for the ten source/profile groups and two source/degraded feature groups over all20 frozen TRAIN references; audit before deriving another direction. Any later finite trial must use bounded line search and actual raw/PNG gates.',
        no_unconditional_anchor_restoration_or_gate_relaxation=True,
        no_unchanged_recipe_rerun=True,no_new_VM_packet_or_training_in_this_review=True,
        no_local_neural_calls_gradients_optimizer_or_parameter_displacement=True,
        protected_files_sha256=protected,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,visual_sheets=20,visual_candidate_PNGs=400,
        actual_metric_rows=800,first_order_comparisons=len(predicted),
        larger_mean_structure_pass_but_preservation_fail=True,
        new_model_qualified=False))


if __name__ == '__main__':
    main()

