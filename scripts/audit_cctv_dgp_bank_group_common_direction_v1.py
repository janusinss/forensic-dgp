"""Independent saved-array/KKT checks; never construct or move model parameters."""
from datetime import datetime, timezone
import ast
import numpy as np
from scipy.optimize import minimize
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, RETURN, ANALYSIS, read, sha, write


def main():
    dest = ANALYSIS/'common_direction'
    receipt = dest/'independent_audit.json'
    assert not receipt.exists()
    review_path = dest/'direction_review.json'
    review = read(review_path)
    evidence = read(ANALYSIS/'independent_audit.json')
    assert review['complete'] and evidence['complete']
    assert sha(ANALYSIS/'independent_audit.json') == review['audited_return_sha256']
    assert sha(RETURN/'outputs/group_gradients_v1_r1/run_results.json') == review['run_results_sha256']
    source = ROOT/'scripts/review_cctv_dgp_bank_group_gradients_v1_r1.py'
    assert sha(source) == review['reviewer_sha256']
    tree = ast.parse(source.read_text(encoding='utf-8'))
    assert not any(isinstance(n, ast.Import) and any(a.name.split('.')[0]=='torch' for a in n.names) for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                   n.func.attr in ['backward','grad','step','copy_','load_state_dict'] for n in ast.walk(tree))
    assert sha(dest/'directions.npz') == review['direction_file_sha256']
    with np.load(dest/'directions.npz', allow_pickle=False) as pack:
        assert set(pack.files) == {f'balanced_p{i:02d}' for i in range(37)}
        arrays = [pack[f'balanced_p{i:02d}'] for i in range(37)]
        assert all(a.dtype==np.float32 and list(a.shape)==review['shapes'][i] and np.isfinite(a).all()
                   for i,a in enumerate(arrays))
        direction = np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)
    vectors=[]
    for bound in review['aggregate_gradient_bindings']:
        path=RETURN/'outputs/group_gradients_v1_r1'/bound['file']
        assert sha(path)==bound['sha256']
        vectors.append(np.load(path,allow_pickle=False,mmap_mode='r'))
    rows=review['float32_group_predictions']
    norms=np.array([r['gradient_norm'] for r in rows])
    multipliers=np.array(review['dual_multipliers'])
    gram=np.zeros((32,32));reconstructed=np.empty(direction.size)
    for begin in range(0,direction.size,32768):
        block=np.vstack([v[begin:begin+32768]/n for v,n in zip(vectors,norms,strict=True)])
        gram += np.einsum('ij,kj->ik',block,block,optimize=True)
        reconstructed[begin:begin+32768]=np.einsum('i,ij->j',-multipliers,block)
    assert np.allclose(gram,review['normalized_Gram'],rtol=0,atol=1e-10)
    assert np.array_equal(reconstructed.astype(np.float32),direction.astype(np.float32))
    targets=np.array(review['normalized_targets'])
    slack=gram@multipliers-targets
    assert multipliers.min()>=0 and slack.min()>-1e-9
    assert np.max(abs(slack*multipliers))<1e-9
    # Independent primal solve starts at zero, unlike the proposal generator.
    result=minimize(lambda c:.5*float(c@gram@c),np.zeros(32),jac=lambda c:gram@c,
        method='SLSQP',constraints=[dict(type='ineq',fun=lambda c:gram@c-targets,jac=lambda c:gram)],
        options=dict(ftol=1e-12,maxiter=3000))
    assert result.success, result.message
    delta=result.x-multipliers
    error=float(np.sqrt(max(0,delta@gram@delta)))
    assert error<1e-5 and np.min(gram@result.x-targets)>-1e-9
    dots=[float(v@direction) for v in vectors]
    assert all(np.isclose(d,r['dot'],rtol=1e-9,atol=1e-12) for d,r in zip(dots,rows,strict=True))
    features=[r for r in rows if r['metric']=='feature_MSE']
    assert len(features)==2 and len({r['source'] for r in features})==2
    assert all(r['fractional_descent_per_unit_rate']>=r['target_fraction_per_unit_rate']-1e-6 for r in rows)
    assert review['rate_candidates']==[.005,.01,.02,.04]
    norm=float(np.linalg.norm(direction))
    assert np.isclose(norm,review['direction_norm'],rtol=1e-12)
    assert .04*norm/review['initial_parameter_norm']<.001
    assert review['old_directions_initially_regressing_group_counts']=={
        label:sum(r['old_direction_dots'][label]>0 for r in rows) for label in ['ordinary','constrained']}
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        direction_review_sha256=sha(review_path),direction_file_sha256=sha(dest/'directions.npz'),
        original_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        aggregate_hashes_checked=32,Gram_recomputed_with_different_chunk_size=True,
        exact_float32_direction_reconstructed=True,all32_float32_slopes_and_KKT_constraints_checked=True,
        zero_initialized_independent_primal_direction_L2_error=error,
        no_neural_calls_gradients_optimizer_or_model_parameter_displacements=True,
        first_order_predictions_not_restoration_or_training_results=True,
        future_manual_finite_raw_and_PNG_review_required=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,groups=32,zero_initialized_primal_error=error,local_neural_calls=0))


if __name__=='__main__':
    main()
