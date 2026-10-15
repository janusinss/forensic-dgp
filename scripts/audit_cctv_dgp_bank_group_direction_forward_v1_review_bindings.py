"""Independently bind the visual and finite-prediction review to audited arrays."""
import math
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_direction_forward_v1_return import ROOT, RETURN, ANALYSIS, read, sha, write
from cctv_dgp_signed_step_audit import group_loss


def main():
    receipt=ANALYSIS/'review_binding_audit.json'
    assert not receipt.exists()
    metric=read(ANALYSIS/'finite_metrics.json')
    review=read(ANALYSIS/'finite_direction_review.json')
    visual=read(ANALYSIS/'visual_review.json')
    audit=read(ANALYSIS/'independent_audit.json')
    direction=read(RETURN/'direction_review.json')
    assert all(d['complete'] for d in [metric,review,visual,audit,direction])
    assert review['finite_metrics_sha256']==audit['finite_metrics_sha256']==sha(ANALYSIS/'finite_metrics.json')
    assert review['direction_review_sha256']==sha(RETURN/'direction_review.json')
    expected={row['key']:row for row in direction['float32_group_predictions']}
    assert len(expected)==len(review['groups'])==32
    comparisons=0
    for row in review['groups']:
        prediction=expected[row['key']]
        baseline=group_loss(metric['baseline_groups']['raw'],prediction)
        assert row['raw_gradient_loss']==prediction['loss']
        assert row['directional_derivative']==prediction['dot']<0
        assert row['independent_raw_baseline_loss']==baseline
        assert row['baseline_loss_difference']==baseline-prediction['loss']
        assert len(row['finite'])==4
        for value in row['finite']:
            rate=float(value['proposal'].removeprefix('balanced_'))
            after=group_loss(metric['proposals'][value['proposal']]['groups']['raw'],prediction)
            change=after-baseline
            check={'rate':rate,'actual_loss_change':change,
                'predicted_first_order_loss_change':rate*prediction['dot'],
                'actual_percent_loss_reduction':-100*change/baseline,
                'predicted_percent_loss_reduction':-100*rate*prediction['dot']/prediction['loss'],
                'change_minus_linear_prediction':change-rate*prediction['dot']}
            for key,number in check.items():
                assert math.isclose(value[key],number,rel_tol=1e-13,abs_tol=1e-12),(row['key'],key)
            comparisons+=1
    sheets={row['file']:row for row in audit['comparison_sheets']}
    assert visual['raw_and_PNG_audit_sha256']==sha(ANALYSIS/'independent_audit.json')
    assert len(sheets)==len(visual['sheets'])==20 and visual['cells_inspected']==800
    for row in visual['sheets']:
        assert row['sha256']==sheets[row['file']]['sha256']==sha(ROOT/row['file'])
        assert row['all40_cells_inspected'] and row['observations']
        assert not row['convincing_useful_gain_over_retained_DGP']
    for row in review['summary']:
        proposal=metric['proposals'][row['proposal']]
        assert not row['all_preview_requirements_pass'] and not proposal['all_preview_requirements_pass']
        for stage,label in [('raw','raw'),('png','PNG')]:
            assert row[label+'_structure_gain_percent']==100*proposal['gates'][stage]['relative_feature_gain']
            assert row[label+'_preservation_failure_count']==len(proposal['gates'][stage]['preservation_failures'])
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        finite_review_sha256=sha(ANALYSIS/'finite_direction_review.json'),visual_review_sha256=sha(ANALYSIS/'visual_review.json'),
        all32_raw_loss_group_mappings_checked=True,finite_prediction_comparisons=128,
        all20_visual_sheet_hashes_and800_cell_bindings_checked=True,independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        protected_original_model_app_hashes_unchanged=True,new_neural_or_gradient_calls=0,
        finite_quality_proposals_rejected=4,individual_layer_or_bug_cause_not_isolated=True,
        manual_small_signed_step_diagnosis_selected_by_user_best_approach=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,loss_comparisons=comparisons,visual_sheets=20))


if __name__=='__main__':main()
