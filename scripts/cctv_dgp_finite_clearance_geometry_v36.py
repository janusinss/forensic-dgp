"""Verify empirical affine clearances using pinned saved arrays; finite images still decide."""
def verify_geometry(root, p, sha, read):
    from pathlib import Path
    import numpy as np

    guard_root = root.parent / 'cctv_dgp_group_guard_grad_v34_vm'
    old = root.parent / 'cctv_dgp_v32_loss_gradient_v1_vm'
    grouped, components = [], []
    for ci,cohort in enumerate(p['cohorts']):
        cases=cohort['cases']
        labels=p['constraint_labels'][51*ci:51*(ci+1)]
        weights=np.zeros((51,150),np.float64)
        for index,row in enumerate(labels):
            group=row['group']
            if group=='all': selected=list(range(50))
            elif group=='clear': selected=[i for i,c in enumerate(cases) if c['profile']=='clear']
            elif group=='degraded': selected=[i for i,c in enumerate(cases) if c['profile']!='clear']
            else:
                source,kind=group.rsplit('/',1)
                selected=[i for i,c in enumerate(cases) if c['source']==source and
                    (kind=='all' or (kind=='degraded' and c['profile']!='clear') or c['profile']==kind)]
            assert selected and row['cohort']==cohort['name']
            metric=p['guard_metrics'].index(row['metric'])
            weights[index,3*np.asarray(selected)+metric]=1./len(selected)
        matrix=np.zeros((51,978243),np.float64)
        for batch in range(10):
            path=guard_root/f'outputs/{cohort["name"]}/batch{batch}_guard_gradients.npy'
            assert sha(path)==p['guard_return_sha256'][path.relative_to(guard_root).as_posix()]
            g=np.load(path,allow_pickle=False)
            assert g.dtype==np.float32 and g.shape==(15,978243) and np.isfinite(g).all()
            matrix+=weights[:,15*batch:15*(batch+1)]@g.astype(np.float64)
        grouped.append(matrix)
        g=np.load(old/f'outputs/state0_{cohort["name"]}/gradient_components.npy',allow_pickle=False)
        assert g.shape==(7,978243) and g.dtype==np.float64 and np.count_nonzero(g[3:])==0
        components.append(g[:3].copy())
    full=np.concatenate(grouped+components,axis=0)
    assert full.shape==(108,978243)
    norms=np.linalg.norm(full,axis=1)
    assert (norms>0).all()
    unit=full/norms[:,None]
    proposal=np.load(root/'mean_original_displacement.npy',allow_pickle=False)
    direction=np.load(root/'projected_displacement.npy',allow_pickle=False)
    theta=np.load(root/'theta_before.npy',allow_pickle=False)
    assert proposal.dtype==direction.dtype==np.float64 and theta.dtype==np.float32
    assert proposal.shape==direction.shape==theta.shape==(978243,)
    proof=read(root/'projection.json')
    multipliers=np.asarray(proof['multipliers'],np.float64)
    targets=np.asarray(proof["clearance_targets"],np.float64)
    assert targets.shape==(108,) and np.isfinite(targets).all() and (targets>=0).all()
    assert np.count_nonzero(targets)==65 and np.count_nonzero(targets[102:])==0
    dots=unit@direction-targets/norms
    stationarity=float(np.linalg.norm(direction-proposal-multipliers@unit))
    assert multipliers.shape==(108,) and (multipliers>=0).all()
    assert stationarity<=1e-12 and dots.min()>=-proof['KKT_tolerance']
    assert np.max(np.abs(multipliers*dots))<=1e-12
    assert np.allclose(full@direction,proof['raw_dots_after'],rtol=2e-10,atol=1e-11)
    certificate={'complete':True,'preservation_guard_rows':102,'existing_restoration_rows':6,
        'KKT_stationarity_error':stationarity,'minimum_unit_clearance_residual':float(dots.min()),
        'projection_sha256':sha(root/'projection.json'),'empirical_clearance_not_a_validated_loss_bound':True,
        'nonzero_clearance_rows':int(np.count_nonzero(targets)),'finite_outputs_still_required':True,
        'gradient_calls':0,'optimizer_updates':0,'neural_calls':0}
    return theta,direction,full,certificate
