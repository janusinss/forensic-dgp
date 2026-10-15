"""Analytical signed-slope tests and actual negative-step selection expression."""
import ast
import copy
import math
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT,OUT,read,sha,write
from cctv_dgp_signed_step_audit import signed_differences

BASE='cctv_dgp_bank_group_step_check_v1'


def main():
    receipt=OUT/(BASE+'_signed_derivative_fixture_audit.json')
    assert not receipt.exists()
    p=read(OUT/(BASE+'_vm')/'protocol.json')
    predictions=copy.deepcopy(read(OUT/(BASE+'_vm')/'direction_review.json')['float32_group_predictions'])
    before=copy.deepcopy(read(OUT/'cctv_dgp_bank_group_direction_forward_v1_analysis/finite_metrics.json')['baseline_groups'])
    for groups in before.values():
        for value in groups.values():
            value.update(MSE=1.,SSIM=0.,ArcFace_observed_fixed=0.,landmark_high_frequency_MSE=1.)
    proposals={a['label']:dict(groups=copy.deepcopy(before)) for a in p['proposals']}
    for i,prediction in enumerate(predictions):
        derivative=-.2-.01*i
        prediction.update(dot=derivative,loss=1.)
        key=prediction['source']+'/'+prediction['profile']
        for a in p['proposals']:
            t=a['rate']
            # Independent analytic polynomial: L(t)=1+d*t+2*t²+3*t³.
            loss=1+derivative*t+2*t*t+3*t*t*t
            field={'MSE':'MSE','SSIM_loss':'SSIM','ArcFace_loss':'ArcFace_observed_fixed',
                   'feature_MSE':'landmark_high_frequency_MSE'}[prediction['metric']]
            proposals[a['label']]['groups']['raw'][key][field]=1-loss if field in ['SSIM','ArcFace_observed_fixed'] else loss
    result=signed_differences(proposals,before,predictions)
    for row,prediction in zip(result['groups'],predictions,strict=True):
        for pair in row['pairs']:
            h=pair['radius'];expected=prediction['dot']+3*h*h
            assert math.isclose(pair['central_directional_derivative'],expected,rel_tol=0,abs_tol=3e-12)
            assert math.isclose(pair['second_difference_per_squared_radius'],4,rel_tol=0,abs_tol=1e-7)
            assert math.isclose(pair['forward_directional_derivative'],prediction['dot']+2*h+3*h*h,rel_tol=0,abs_tol=3e-12)
            assert math.isclose(pair['backward_directional_derivative'],prediction['dot']-2*h+3*h*h,rel_tol=0,abs_tol=3e-12)
    missing=copy.deepcopy(proposals);missing.pop('balanced_-0.001')
    try:signed_differences(missing,before,predictions)
    except AssertionError:pass
    else:raise AssertionError('Missing negative pair must fail')
    auditor=ROOT/'scripts'/f'audit_{BASE}_return.py'
    tree=ast.parse(auditor.read_text(encoding='utf-8'),feature_version=(3,10))
    call=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript)
        and ast.unparse(n.targets[0])=='results[label]')
    expr=next(k.value for k in call.keywords if k.arg=='all_preview_requirements_pass')
    code=compile(ast.Expression(body=expr),str(auditor),'eval')
    checks=[]
    for t in [-.001,-.0001,.0001,.001]:
        observed=eval(code,{'proposal':{'rate':t},'source_pass':True,
            'gates':{'raw':{'pass_all_preview_requirements':True},'png':{'pass_all_preview_requirements':True}}})
        assert observed==(t>0)
        checks.append(dict(rate=t,synthetic_gate_pass_fixture=True,selection_expression_result=observed))
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),groups_checked=32,
        checker_sha256=sha(ROOT/'scripts/cctv_dgp_signed_step_audit.py'),fixture_checker_sha256=sha(__file__),
        future_auditor_sha256=sha(auditor),protocol_sha256=sha(OUT/(BASE+'_vm')/'protocol.json'),
        known_analytic_slopes_curvature_and_one_sided_slopes_checked=64,missing_signed_pair_rejected=True,
        actual_negative_step_model_selection_expression_checked=checks,
        fixtures_are_not_model_or_restoration_results=True,no_neural_model_or_gradient_calls=True,model_qualified=False))
    print(dict(complete=True,groups=32,analytic_pairs=64,negative_step_expression_checks=4))


if __name__=='__main__':main()
