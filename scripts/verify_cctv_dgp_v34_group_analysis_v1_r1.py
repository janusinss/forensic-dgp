"""Independent saved-array group membership and projection-certificate readback."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'
RETURN = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_return'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    import numpy as np
    a = read(OUT / 'analysis.json')
    assert a['complete'] and a['all108_nonzero_constraints_retained']
    for name, digest in a['bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in a['saved_arrays_sha256'].items():
        assert sha(OUT / name) == digest, name
    failed = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1'
    f = read(failed / 'failure_preservation.json')
    for name,digest in f['retained_files_sha256'].items():
        assert sha(failed / name) == digest
    assert f['serialization_regression_passed'] and not f['geometry_or_quality_gate_changes']
    for name in ['group_guard_matrix.npy', 'projected_displacement.npy', 'mean_original_displacement.npy', 'theta_before.npy', 'projection.json']:
        assert sha(OUT / name) == sha(failed / name), 'Serialization fix must retain every numeric artifact'
    p = read(RETURN / 'protocol.json')
    matrix = np.load(OUT / 'group_guard_matrix.npy', allow_pickle=False)
    assert matrix.shape == (102,978243) and matrix.dtype == np.float64
    worst = 0.
    old = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs'
    component_rows = []
    for cindex,cohort in enumerate(p['cohorts']):
        cases = cohort['cases']
        folder = RETURN / 'outputs' / cohort['name']
        labels = a['constraint_labels'][51*cindex:51*(cindex+1)]
        weights = np.zeros((51,150), np.float64)
        for i,row in enumerate(labels):
            group = row['group']
            if group == 'all':
                mask = np.ones(50,bool)
            elif group in ['clear','degraded']:
                mask = np.array([v['profile']=='clear' for v in cases])
                if group == 'degraded': mask = ~mask
            else:
                source, kind = group.rsplit('/',1)
                mask = np.array([v['source']==source and (kind=='all' or
                    (kind=='degraded' and v['profile']!='clear') or v['profile']==kind) for v in cases])
            assert mask.any()
            metric = p['guard_metrics'].index(row['metric'])
            weights[i,3*np.flatnonzero(mask)+metric] = 1. / int(mask.sum())
        rebuilt = np.zeros((51,978243),np.float64)
        for batch in range(10):
            g = np.load(folder / f'batch{batch}_guard_gradients.npy',allow_pickle=False)
            rebuilt += weights[:,15*batch:15*(batch+1)] @ g.astype(np.float64)
        error = float(np.abs(rebuilt-matrix[51*cindex:51*(cindex+1)]).max())
        assert error <= 2e-12
        worst = max(worst,error)
        g = np.load(old / f'state0_{cohort["name"]}/gradient_components.npy',allow_pickle=False)
        assert np.count_nonzero(g[3:]) == 0
        component_rows.append(g[:3].copy())
        del rebuilt
    full = np.concatenate([matrix,*component_rows],axis=0)
    del matrix
    norms = np.linalg.norm(full,axis=1)
    assert (norms>0).all()
    unit = full / norms[:,None]
    direction = np.load(OUT/'projected_displacement.npy',allow_pickle=False)
    proposal = np.load(OUT/'mean_original_displacement.npy',allow_pickle=False)
    proof = read(OUT/'projection.json')
    multipliers = np.asarray(proof['multipliers'],np.float64)
    stationarity = float(np.linalg.norm(direction-proposal-multipliers@unit))
    dots = unit@direction
    assert (multipliers>=0).all() and dots.min()>=-proof['KKT_tolerance']
    assert stationarity<=1e-12 and np.max(np.abs(multipliers*dots))<=1e-12
    assert np.allclose(full@direction,proof['raw_dots_after'],rtol=2e-10,atol=1e-11)
    assert proof['independent_primal_relative_L2_error']<=2e-7
    theta = np.load(OUT/'theta_before.npy',allow_pickle=False)
    for r in a['fixed_float32_scales']:
        delta=theta.astype(np.float64)-(theta.astype(np.float64)-r['scale']*direction).astype(np.float32).astype(np.float64)
        changes=-(full@delta)
        assert np.allclose(changes[:102],r['linear_guard_changes'],rtol=2e-10,atol=1e-11)
        assert np.allclose(changes[102:],r['linear_existing_loss_changes'],rtol=2e-10,atol=1e-11)
    # Meaningful array fixtures: conflicting source rows cannot cancel away;
    # positive scaling and reordering cannot change the feasible displacement.
    from cctv_dgp_group_guard_cone_v35 import project_vectors
    base=np.array([[1.,0.],[-1.,0.],[0.,1.]])
    desired=np.array([3.,2.])
    result,_=project_vectors(base,desired)
    assert np.allclose(result,[0.,2.],atol=1e-9)
    reordered,_=project_vectors(base[[2,0,1]]*np.array([[7.],[2.],[3.]]),desired)
    assert np.allclose(result,reordered,atol=1e-9)
    rejected=0
    for g,q in [(np.ones((110,2)),desired),(base,np.array([np.nan,2.])),(base,np.ones(3)),(base[0],desired)]:
        try: project_vectors(g,q)
        except ValueError: rejected+=1
        else: raise AssertionError('Invalid geometry must stop')
    receipt={'complete':True,'analysis_sha256':sha(OUT/'analysis.json'),'checker_sha256':sha(Path(__file__)),
        'group_rows_independently_reassembled':102,'existing_restoration_rows_retained':6,
        'maximum_group_element_error':worst,'KKT_stationarity_error':stationarity,
        'minimum_unit_guard_dot':float(dots.min()),'positive_scaling_and_reorder_passed':True,
        'aggregate_cancellation_fixture_passed':True,'invalid_geometry_rejections':rejected,
        'serialization_failure_and_numeric_artifacts_retained':True,
        'neural_calls':0,'gradient_calls':0,'optimizer_updates':0,'VM_calls':0,
        'finite_output_guarantee':False,'training_capacity_pass':False,'app_promotion':False,
        'goal_complete':False,'seconds':time.monotonic()-started}
    assert receipt['seconds']<300
    with (OUT/'independent_analysis_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'constraints':108,'seconds':receipt['seconds']}))


if __name__=='__main__':
    main()
