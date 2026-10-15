"""Combine audited saved raw/PNG derivatives; no model or neural operations."""
from datetime import datetime, timezone
import time
import numpy as np
from scipy.optimize import nnls, minimize
from import_cctv_dgp_png_group_gradients_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def main():
    start=time.monotonic()
    dest=ANALYSIS/'joint_raw_png_direction_v1_r2';assert not dest.exists()
    raw_return=OUT/'cctv_dgp_bank_group_gradients_v1_r1_return'
    raw_analysis=OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis'
    png_audit=read(ANALYSIS/'independent_audit.json');raw_audit=read(raw_analysis/'independent_audit.json')
    assert png_audit['complete'] and raw_audit['complete']
    png_path=RETURN/'outputs/png_group_gradients_v1_r2/run_results.json'
    raw_path=raw_return/'outputs/group_gradients_v1_r1/run_results.json'
    assert sha(png_path)==png_audit['run_results_sha256'] and sha(raw_path)==raw_audit['run_results_sha256']
    png,raw=read(png_path),read(raw_path)
    p,old=read(RETURN/'protocol.json'),read(raw_return/'protocol.json')
    for key in ['case_ids','initial_model_state','learning_names','parameters','parent_protocol_sha256','preserved_scientific_thresholds']:
        assert p[key]==old[key]
    assert png['model_state']==raw['model_state']==p['initial_model_state']
    assert png['parameters']==raw['parameters']==3364643
    groups,vectors,bindings=[],[],[]
    for stage,folder,run in [('RAW',raw_return/'outputs/group_gradients_v1_r1',raw),('PNG',RETURN/'outputs/png_group_gradients_v1_r2',png)]:
        assert len(run['groups'])==32
        for row in run['groups']:
            path=folder/row['file'];assert sha(path)==row['sha256']
            v=np.load(path,allow_pickle=False,mmap_mode='r')
            assert v.dtype==np.float64 and v.shape==(3364643,)
            key=row['key'] if row['key'].startswith('PNG/') else 'RAW/'+row['key']
            groups.append(dict(row,key=key,metric_stage=stage));vectors.append(v)
            bindings.append(dict(key=key,stage=stage,file=path.relative_to(ROOT).as_posix(),sha256=row['sha256']))
    assert len(groups)==64 and len({r['key'] for r in groups})==64
    norms=np.array([row['gradient_norm'] for row in groups]);losses=np.array([row['value'] for row in groups])
    assert np.all(norms>0) and np.all(losses>0)
    gram=np.zeros((64,64))
    for begin in range(0,3364643,65536):
        block=np.stack([v[begin:begin+65536] for v in vectors])/norms[:,None]
        assert np.isfinite(block).all();gram+=block@block.T
        assert time.monotonic()-start<180
    assert np.allclose(gram,gram.T,rtol=0,atol=1e-12) and np.allclose(gram.diagonal(),1,rtol=0,atol=1e-10)
    eigen,basis=np.linalg.eigh(gram)
    roundoff_bound=64*np.finfo(np.float64).eps*eigen[-1]
    assert eigen[0]>roundoff_bound, 'All64 measured dimensions must remain above Gram roundoff bound'
    # Keep every dimension and loss. Orthogonal coordinates condition the
    # minimum physical-norm objective without adding a ridge or dropping rows.
    Q=basis*np.sqrt(eigen)[None,:]
    assert np.allclose(Q@Q.T,gram,rtol=0,atol=1e-12)
    fractions=np.array([1. if row['metric']=='feature_MSE' else .1 for row in groups])
    targets=fractions*losses/norms
    primal=minimize(lambda z:.5*float(z@z),np.zeros(64),jac=lambda z:z,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda z:Q@z-targets,jac=lambda z:Q)],options=dict(ftol=1e-12,maxiter=2000))
    assert primal.success,primal.message
    actual_slack=Q@primal.x-targets
    assert actual_slack.min()>-1e-9
    active=actual_slack<=1e-9
    dual=np.zeros(64)
    dual[active],residual=nnls(Q[active].T,primal.x,maxiter=10000)
    stationary=float(np.linalg.norm(Q.T@dual-primal.x))
    slack=gram@dual-targets
    assert stationary<1e-8 and dual.min()>=0 and slack.min()>-1e-9
    assert np.max(abs(slack*dual))<1e-9
    discrepancy=stationary
    direction=np.empty(3364643,np.float32)
    for begin in range(0,3364643,65536):
        block=np.stack([v[begin:begin+65536] for v in vectors])/norms[:,None]
        direction[begin:begin+65536]=(-dual@block).astype(np.float32)
    actual_dots=np.array([float(v@direction.astype(np.float64)) for v in vectors])
    descent=-actual_dots/losses
    assert np.all(descent>=fractions-1e-6), 'Saved float32 direction fails a joint local constraint'
    previous=read(RETURN/'direction_review.json')
    shapes=previous['shapes'];sizes=[int(np.prod(s)) for s in shapes];offsets=np.cumsum([0]+sizes)
    assert sum(sizes)==3364643 and previous['names']==p['learning_names']
    initial_norm=previous['initial_parameter_norm'];norm=float(np.linalg.norm(direction.astype(np.float64)))
    rates=[.0025,.005,.01,.02]
    assert max(rates)*norm/initial_norm<.001, 'Joint direction exceeds retained finite parameter-distance bound'
    assert all(100*max(rates)*descent[i]>=1 for i,r in enumerate(groups) if r['metric']=='feature_MSE')
    dest.mkdir()
    path=dest/'directions.npz'
    arrays={f'balanced_p{i:02d}':direction[offsets[i]:offsets[i+1]].reshape(shape) for i,shape in enumerate(shapes)}
    np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as z:assert all(np.array_equal(z[k],v) for k,v in arrays.items())
    assert sha(RETURN/'directions.npz')==previous['direction_file_sha256']
    with np.load(RETURN/'directions.npz',allow_pickle=False) as oldpack:
        assert set(oldpack.files)=={f'balanced_p{i:02d}' for i in range(37)}
        old_direction_vector=np.concatenate([oldpack[f'balanced_p{i:02d}'].reshape(-1) for i in range(37)]).astype(np.float64)
    rows=[dict(key=row['key'],source=row['source'],profile=row['profile'],metric=row['metric'],metric_stage=row['metric_stage'],
        loss=row['value'],gradient_norm=row['gradient_norm'],dot=float(actual_dots[i]),
        fractional_descent_per_unit_rate=float(descent[i]),target_fraction_per_unit_rate=float(fractions[i]),
        predicted_percent_improvement={str(rate):float(100*rate*descent[i]) for rate in rates},
        old_raw_balanced_direction_dot=float(vectors[i]@old_direction_vector),first_order_only=True) for i,row in enumerate(groups)]
    assert sum(row['old_raw_balanced_direction_dot']>0 for row in rows if row['metric_stage']=='PNG')==1
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    assert time.monotonic()-start<180
    write(dest/'direction_review.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),reviewer_sha256=sha(__file__),
        raw_return_audit_sha256=sha(raw_analysis/'independent_audit.json'),PNG_return_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        raw_run_results_sha256=sha(raw_path),PNG_run_results_sha256=sha(png_path),
        input_protocol_sha256=sha(RETURN/'protocol.json'),raw_input_protocol_sha256=sha(raw_return/'protocol.json'),
        aggregate_gradient_bindings=bindings,names=p['learning_names'],shapes=shapes,sizes=sizes,parameters=3364643,
        direction_file_sha256=sha(path),direction_file_bytes=path.stat().st_size,
        initial_file_sha256=previous['initial_file_sha256'],initial_parameter_norm=initial_norm,direction_norm=norm,
        initial_model_state=p['initial_model_state'],rate_candidates=rates,
        normalized_Gram=gram.tolist(),normalized_targets=targets.tolist(),dual_multipliers=dual.tolist(),
        smallest_Gram_eigenvalue=float(eigen[0]),Gram_condition_number=float(eigen[-1]/eigen[0]),NNLS_residual=float(residual),
        maximum_primal_violation=float(-slack.min()),maximum_complementarity_error=float(np.max(abs(slack*dual))),
        orthonormal_primal_stationarity_error=stationary,primary_primal_iterations=int(primal.nit),
        all64_eigendimensions_retained=True,no_ridge_or_constraint_removal=True,
        Gram_roundoff_rank_bound=float(roundoff_bound),float32_group_predictions=rows,
        maximum_prospective_displacement_relative_to_parameter_L2=max(rates)*norm/initial_norm,
        groups_RAW=32,groups_PNG=32,old_raw_direction_PNG_groups_initially_worsening=1,
        all64_float32_predicted_derivatives_descend=True,zero_groups_removed_and_no_quality_threshold_change=True,
        mathematical_method='Minimum physical L2 direction with64 constraints in a full orthonormal derivative span; SLSQP primal and active NNLS KKT witness',
        theoretical_scope='Local approximate derivative design only; no MGDA implementation, convergence or finite preservation guarantee',
        no_local_neural_calls_gradients_optimizer_or_parameter_displacement=True,
        scope='Same20 paired photographic TRAIN references/100 cases; no native or final pixels',
        future_actual_raw_and_PNG_outputs_and_all_visual_sheets_required=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False,seconds=time.monotonic()-start))
    print(dict(complete=True,groups=64,minimum_eigenvalue=float(eigen[0]),direction_norm=norm,
        max_relative_distance=max(rates)*norm/initial_norm,old_PNG_worsening_groups=1,local_model_or_gradient_calls=0))


if __name__=='__main__':main()
