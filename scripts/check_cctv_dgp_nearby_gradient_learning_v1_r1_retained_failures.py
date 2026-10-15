"""Retain actual finite-output rejection and the actual original repeat bound miss."""
from datetime import datetime,timezone
import copy
import numpy as np

from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import ROOT,RETURN,RESPONSE,ANALYSIS,read,sha,write
from cctv_dgp_raw_conditioning_learning_v1_r1_contract import decision
from raw_conditioning_decision_receipt_check_v1_r1 import validate_receipt
from cctv_dgp_RGB_gradient_parity_v1 import report


def main():
    audited=read(ANALYSIS/'failure_audit.json');assert audited['complete'] and audited['accepted_trials']==[]
    out=RETURN/'outputs/nearby_gradient_learning_v1';base=read(out/'initial/snapshot.json')['groups']
    checked=[]
    for label in ['trial00','trial01','trial02']:
        saved=read(out/label/'decision.json')['decision'];after=read(out/label/'snapshot.json')['groups']
        actual=decision(base,base,after);assert not actual['accept_step'] and not actual['one_percent_checkpoint_pass']
        assert validate_receipt(actual,saved,(base,base,after))['exact_saved_score_receipt_binding']
        erased=copy.deepcopy(saved);erased['stages']['png']['initial_preservation_failures']=[]
        try:validate_receipt(actual,erased,(base,base,after))
        except AssertionError:pass
        else:raise AssertionError('Existing PNG failures were erased')
        promoted=copy.deepcopy(saved);promoted['accept_step']=True;promoted['one_percent_checkpoint_pass']=True
        try:validate_receipt(actual,promoted,(base,base,after))
        except AssertionError:pass
        else:raise AssertionError('Rejected output falsely promoted')
        checked.append(dict(label=label,rejection_retained=True,erased_failure_and_false_promotion_rejected=True))
    expected=next(r for r in audited['original_initial_gradient_parity_reports'] if not r['original_parity_check_pass'])
    current=read(out/'gradients_initial/gradients.json');index=next(i for i,r in enumerate(current['groups']) if r['key']==expected['key'])
    new=current['groups'][index];old=read(RESPONSE/'outputs/conditioning_response_v1/repeat00/gradients.json')['groups'][index]
    a=np.load(out/new['file'],allow_pickle=False);b=np.load(RESPONSE/'outputs/conditioning_response_v1'/old['file'],allow_pickle=False)
    fresh=report(a,b);assert not fresh['original_parity_check_pass'] and fresh['mismatched_elements']==1 and fresh['norm_check_pass']
    assert dict(key=expected['key'],**fresh)==expected
    point=fresh['violations'][0];i=point['index']
    assert abs(a[i]-b[i])>1e-8+1e-4*abs(b[i]) and point['absolute_difference']>point['original_bound']
    write(ANALYSIS/'retained_failure_regression.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),independent_failure_audit_sha256=sha(ANALYSIS/'failure_audit.json'),
        actual_three_trial_rejections_checked=checked,actual_original_parity_miss_reproduced=fresh,original_parity_key=expected['key'],
        original_bounds_unchanged=True,actual_images_or_vectors_not_modified=True,local_neural_or_training_calls=0,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,rejected_trials_retained=3,actual_parity_miss_retained=1,neural_calls=0))


if __name__=='__main__':main()
