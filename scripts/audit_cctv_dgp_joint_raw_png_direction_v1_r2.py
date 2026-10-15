"""Independent full-vector and convex KKT audit; no model construction."""
import ast
from datetime import datetime, timezone
import time
import numpy as np
from import_cctv_dgp_png_group_gradients_v1_r2_return import ROOT,OUT,ANALYSIS,RETURN,PARENT,read,sha,write


def main():
    start=time.monotonic();dest=ANALYSIS/'joint_raw_png_direction_v1_r2'
    receipt=dest/'independent_audit.json';assert not receipt.exists()
    review_path=dest/'direction_review.json';r=read(review_path)
    assert r['complete'] and r['groups_RAW']==r['groups_PNG']==32
    assert sha(ANALYSIS/'independent_audit.json')==r['PNG_return_audit_sha256']
    assert sha(OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis/independent_audit.json')==r['raw_return_audit_sha256']
    source=ROOT/'scripts/review_cctv_dgp_joint_raw_png_direction_v1_r2.py'
    assert sha(source)==r['reviewer_sha256']
    tree=ast.parse(source.read_text())
    assert not any(isinstance(n,ast.Import) and any(a.name.split('.')[0]=='torch' for a in n.names) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        ['grad','backward','step','copy_','load_state_dict'] for n in ast.walk(tree))
    rows=r['float32_group_predictions'];bindings=r['aggregate_gradient_bindings']
    assert len(rows)==len(bindings)==64 and len({row['key'] for row in rows})==64
    assert all(row['metric_stage']==bound['stage'] and row['key']==bound['key'] for row,bound in zip(rows,bindings,strict=True))
    vectors=[]
    for item in bindings:
        path=ROOT/item['file'];assert sha(path)==item['sha256']
        v=np.load(path,allow_pickle=False,mmap_mode='r');assert v.dtype==np.float64 and v.shape==(r['parameters'],)
        vectors.append(v)
    norms=np.array([row['gradient_norm'] for row in rows]);losses=np.array([row['loss'] for row in rows])
    expected_fractions=np.array([1. if row['metric']=='feature_MSE' else .1 for row in rows])
    targets=expected_fractions*losses/norms
    np.testing.assert_array_equal(targets,r['normalized_targets'])
    dual=np.array(r['dual_multipliers']);assert dual.shape==(64,) and np.isfinite(dual).all() and dual.min()>=0
    gram=np.zeros((64,64));reconstructed=np.empty(r['parameters'])
    for begin in range(0,r['parameters'],32768):
        block=np.vstack([v[begin:begin+32768]/norm for v,norm in zip(vectors,norms,strict=True)])
        gram+=np.einsum('ij,kj->ik',block,block,optimize=True)
        reconstructed[begin:begin+32768]=np.einsum('i,ij->j',-dual,block,optimize=True)
        assert time.monotonic()-start<180
    assert np.allclose(gram,r['normalized_Gram'],rtol=0,atol=1e-10)
    eigen=np.linalg.eigvalsh(gram)
    rank_bound=64*np.finfo(np.float64).eps*eigen[-1]
    assert eigen[0]>rank_bound and r['all64_eigendimensions_retained'] and r['no_ridge_or_constraint_removal']
    slack=gram@dual-targets
    assert slack.min()>-1e-9 and np.max(abs(slack*dual))<1e-9
    # Convex primal/dual KKT witness: all constraints, nonnegative multipliers,
    # complementary slackness, and d=-sum(lambda_i*normalized_gradient_i).
    assert sha(dest/'directions.npz')==r['direction_file_sha256']
    with np.load(dest/'directions.npz',allow_pickle=False) as z:
        assert set(z.files)=={f'balanced_p{i:02d}' for i in range(37)}
        arrays=[z[f'balanced_p{i:02d}'] for i in range(37)]
        assert all(a.dtype==np.float32 and list(a.shape)==r['shapes'][i] and np.isfinite(a).all() for i,a in enumerate(arrays))
        direction=np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)
    np.testing.assert_array_equal(reconstructed.astype(np.float32),direction.astype(np.float32))
    dots=np.array([float(v@direction) for v in vectors]);fractions=-dots/losses
    assert np.all(fractions>=expected_fractions-1e-6)
    assert all(np.isclose(dot,row['dot'],rtol=1e-9,atol=1e-12) for dot,row in zip(dots,rows,strict=True))
    assert r['rate_candidates']==[.0025,.005,.01,.02]
    assert np.isclose(np.linalg.norm(direction),r['direction_norm'],rtol=1e-12)
    assert .02*np.linalg.norm(direction)/r['initial_parameter_norm']<.001
    assert sha(PARENT/'initializers/initial_B.pth')==r['initial_file_sha256']
    assert r['initial_model_state']==read(RETURN/'protocol.json')['initial_model_state']
    with np.load(RETURN/'directions.npz',allow_pickle=False) as old:
        previous=np.concatenate([old[f'balanced_p{i:02d}'].reshape(-1) for i in range(37)]).astype(np.float64)
    olddots=[float(v@previous) for v in vectors]
    assert all(np.isclose(dot,row['old_raw_balanced_direction_dot'],rtol=1e-9,atol=1e-12) for dot,row in zip(olddots,rows,strict=True))
    png_bad=[row['key'] for row,dot in zip(rows,olddots,strict=True) if row['metric_stage']=='PNG' and dot>0]
    assert png_bad==['PNG/dataset/thumbnails128x128/clear/MSE']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        direction_review_sha256=sha(review_path),direction_file_sha256=sha(dest/'directions.npz'),
        RAW_gradient_files_checked=32,PNG_gradient_files_checked=32,
        independent_Gram_and_direction_reconstructed_with_different_chunks=True,
        full64_convex_primal_dual_KKT_witness_verified=True,all64_actual_saved_float32_derivatives_descend=True,
        all64_positive_eigendimensions_retained=True,no_ridge_or_loss_group_removal=True,
        old_raw_direction_PNG_worsening_groups=png_bad,physical_distance_bound_retained=True,
        visible_raw_PNG_structure_and_brightness_quality_gates_not_changed=True,
        no_local_neural_gradient_optimizer_or_parameter_displacement_calls=True,
        independent_actual_finite_outputs_and_all_visual_sheets_still_required=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False,seconds=time.monotonic()-start))
    print(dict(complete=True,RAW_groups=32,PNG_groups=32,all64_local_constraints_pass=True,new_model_or_epoch=False))


if __name__=='__main__':main()
