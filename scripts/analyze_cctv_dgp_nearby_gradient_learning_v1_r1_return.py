"""Compare saved derivatives with already executed proposals; no model or new direction."""
from collections import Counter
from datetime import datetime,timezone
import time
import numpy as np

from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import ROOT,OUT,RETURN,ANALYSIS,PIN,read,sha,write


def loss(groups,row):
    g=groups['raw'][row['source']+'/'+row['profile']]
    field=dict(MSE='MSE',SSIM_loss='SSIM',ArcFace_loss='ArcFace_observed_fixed',feature_MSE='landmark_high_frequency_MSE')[row['metric']]
    return 1-g[field] if row['metric'] in ['SSIM_loss','ArcFace_loss'] else g[field]


def main():
    started=time.monotonic();audited=read(ANALYSIS/'failure_audit.json')
    assert audited['complete'] and audited['protocol_sha256']==PIN and not audited['completed_pilot']
    assert audited['trials_replayed']==3 and audited['accepted_trials']==[] and audited['certificate_replay']['complete']
    assert audited['complete_gradient_sites']==['initial','plus02','minus02']
    folder=RETURN/'outputs/nearby_gradient_learning_v1';p=read(RETURN/'protocol.json')
    failure=read(folder/'failure.json');assert failure['counts']['gradient_queries']==960 and failure['counts']['accepted_updates']==0
    record=read(folder/'direction_certificate.json');rows=record['groups'];assert len(rows)==96
    vectors=[]
    for r in rows:
        path=folder/r['file'];assert sha(path)==r['sha256'];vectors.append(np.load(path,allow_pickle=False,mmap_mode='r'))
    direction=np.load(folder/'direction.npy',allow_pickle=False).astype(np.float64)
    initial=np.load(folder/'initial_conditioning.npy',allow_pickle=False).astype(np.float64)
    labels=[f'trial{i:02d}' for i in range(3)]
    trials=[read(folder/label/'decision.json') for label in labels]
    snapshots=[read(folder/label/'snapshot.json')['groups'] for label in labels]
    base=read(folder/'initial/snapshot.json')['groups']
    deltas=np.column_stack([np.load(folder/label/'conditioning.npy',allow_pickle=False).astype(np.float64)-initial for label in labels])
    dots=np.zeros(96);rounded=np.zeros((96,3));nearby_pair_dot=np.zeros((32,3))
    for begin in range(0,len(direction),32768):
        assert time.monotonic()-started<240,'Saved-array analysis stop'
        block=np.vstack([v[begin:begin+32768] for v in vectors])
        dots+=np.einsum('ij,j->i',block,direction[begin:begin+32768],optimize=True)
        rounded+=np.einsum('ij,jk->ik',block,deltas[begin:begin+32768],optimize=True)
        for column,(a,b) in enumerate([(0,32),(0,64),(32,64)]):
            nearby_pair_dot[:,column]+=np.einsum('ij,ij->i',block[a:a+32],block[b:b+32],optimize=True)
    np.testing.assert_allclose(dots,record['certificate']['direction_dots'],rtol=1e-8,atol=1e-10)
    assert (dots<0).all()
    comparisons=[];summaries=[]
    for column,(label,t,after) in enumerate(zip(labels,trials,snapshots)):
        assert not t['decision']['accept_step'] and not t['decision']['one_percent_checkpoint_pass']
        counts=Counter();feature_indices=[]
        for i,r in enumerate(rows[:32]):
            before,value=loss(base,r),loss(after,r)
            assert abs(before-r['value'])<=(1e-6 if r['metric']=='ArcFace_loss' else 1e-13)
            tolerance=1e-12 if r['metric']=='MSE' else 1e-6 if r['metric'] in ['SSIM_loss','ArcFace_loss'] else 0.
            prediction=float(rounded[i,column]);actual=float(value-before)
            bad=r['metric']!='feature_MSE' and actual>tolerance
            counts['initial_RAW_groups']+=1
            counts['actual_RAW_preservation_losses_failed']+=bad
            counts['actual_RAW_loss_descent_sign_disagrees']+=actual>=0
            comparisons.append(dict(trial=label,key=r['key'],source=r['source'],profile=r['profile'],metric=r['metric'],
                before_loss=before,after_loss=value,actual_loss_delta=actual,initial_gradient_actual_rounded_displacement_dot=prediction,
                finite_minus_initial_linear_residual=actual-prediction,original_regression_bound=tolerance,
                RAW_preservation_regression_beyond_bound=bad))
            if r['metric']=='feature_MSE':feature_indices.append(i)
        weights=np.asarray([base['raw'][rows[i]['source']+'/degraded']['cases'] for i in feature_indices])
        predicted_gain=-float(np.average(rounded[feature_indices,column],weights=weights))/base['raw']['degraded']['landmark_high_frequency_MSE']
        stages={}
        for stage,v in t['decision']['stages'].items():
            failures=v['initial_preservation_failures']
            stages[stage]=dict(global_structure_gain=v['relative_feature_gain'],source_structure_gains=v['source_feature_gains'],
                failed_checks=len(failures),failed_metrics=dict(Counter(f['metric'] for f in failures)),
                failures=failures,brightness_gain_fraction=v['brightness_gain_fraction'],
                source_global_progress_positive=all(g>0 for g in v['step_source_and_global_progress'].values()),
                one_percent_checkpoint_pass=v['one_percent_checkpoint_pass'])
        intended=direction*t['rate']
        summaries.append(dict(trial=label,rate=t['rate'],cumulative_relative_parameter_distance=t['cumulative_relative_parameter_distance'],
            initial_linear_predicted_RAW_structure_gain=predicted_gain,stages=stages,counts=dict(counts),
            float32_parameter_rounding_relative_error=float(np.linalg.norm(deltas[:,column]-intended)/np.linalg.norm(intended)),
            all96_rounded_displacement_dots_are_negative=bool((rounded[:,column]<0).all()),
            near_state_derivatives_are_not_predictions_of_initial_to_proposal_finite_loss=True,
            accept_step=False,endpoint_pass=False))
    gram=np.asarray(record['certificate']['normalized_Gram']);eig=np.linalg.eigvalsh(gram)
    bound=record['certificate']['rank_bound'];kept=eig[eig>bound]
    assert len(kept)==record['certificate']['numerical_span_rank']
    pairs=[];norms=np.asarray([r['gradient_norm'] for r in rows])
    for i,r in enumerate(rows[:32]):
        assert rows[32+i]['key']==rows[64+i]['key']==r['key']
        values={}
        for column,(a,b,label) in enumerate([(0,32,'initial_plus02'),(0,64,'initial_minus02'),(32,64,'plus02_minus02')]):
            dot=nearby_pair_dot[i,column];na,nb=norms[a+i],norms[b+i]
            values[label]=dict(cosine=float(dot/(na*nb)),relative_L2_difference_to_first=float(np.sqrt(max(0.,na*na+nb*nb-2*dot))/na))
        pairs.append(dict(key=r['key'],metric=r['metric'],comparisons=values))
    previous_path=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_analysis/gradient_prediction_review.json'
    previous=read(previous_path);assert previous['complete'] and previous['accepted_updates']==0
    previous_folder=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_return/outputs/raw_conditioning_learning_v1_r1'
    np.testing.assert_array_equal(initial.astype(np.float32),np.load(previous_folder/'initial_conditioning.npy',allow_pickle=False))
    assert read(previous_folder/'initial/snapshot.json')['groups']==base
    previous_direction=np.load(previous_folder/'round00/direction.npy',allow_pickle=False).astype(np.float64)
    cosine=float(direction@previous_direction/(np.linalg.norm(direction)*np.linalg.norm(previous_direction)))
    old_trials=[dict(trial=t['trial'],RAW_gain=t['stages']['raw']['actual_global_feature_gain'],PNG_gain=t['stages']['png']['actual_global_feature_gain'],
        RAW_failures=t['stages']['raw']['failed_checks'],PNG_failures=t['stages']['png']['failed_checks']) for t in previous['trials']]
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/'finite_response_review.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),analyst_sha256=sha(__file__),
        protocol_sha256=PIN,independent_failure_audit_sha256=sha(ANALYSIS/'failure_audit.json'),direction_certificate_sha256=sha(folder/'direction_certificate.json'),
        trials=summaries,initial_RAW_group_trial_comparisons=comparisons,nearby_gradient_pairs=pairs,
        normalized_Gram_smallest_eigenvalue=float(eig[0]),normalized_Gram_numerical_span_rank=len(kept),
        normalized_Gram_resolved_condition_number=float(kept[-1]/kept[0]),all96_original_constraints_certified=True,
        previous_RAW32_direction_cosine=cosine,previous_executed_RAW32_trials=old_trials,previous_review_sha256=sha(previous_path),
        original_initial_gradient_parity_reports=audited['original_initial_gradient_parity_reports'],
        accepted_updates=0,finite_preservation_and_1percent_endpoint_not_qualified=True,
        inference='Nearby sampled derivatives still do not establish actual finite RAW/PNG preservation. A certified local derivative direction is insufficient evidence of a useful training step.',
        exclusive_root_cause_not_established=True,new_neural_models_or_derivatives_or_direction_or_proposal_computed=False,
        native_CCTV_and_reserved_final_identities_used=False,seconds=time.monotonic()-started,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,initial_group_trial_comparisons=len(comparisons),accepted_updates=0,
        previous_direction_cosine=cosine,trials=[dict(label=t['trial'],RAW_gain=t['stages']['raw']['global_structure_gain'],
            PNG_gain=t['stages']['png']['global_structure_gain'],RAW_failures=t['stages']['raw']['failed_checks'],
            PNG_failures=t['stages']['png']['failed_checks']) for t in summaries],local_neural_or_training_calls=0))


if __name__=='__main__':main()
