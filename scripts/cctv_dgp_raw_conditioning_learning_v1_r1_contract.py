"""Pure arithmetic for the finite conditioning pilot; no neural imports."""
import time
import numpy as np
from scipy.optimize import minimize, nnls


def solve_direction(vectors, records):
    start = time.monotonic()
    assert len(vectors) == len(records) == 32
    norms = np.array([float(np.linalg.norm(v)) for v in vectors])
    losses = np.array([r['value'] for r in records])
    assert np.isfinite(norms).all() and (norms > 0).all()
    assert np.isfinite(losses).all() and (losses > 0).all()
    gram = np.zeros((32,32))
    for begin in range(0,len(vectors[0]),65536):
        block = np.stack([v[begin:begin+65536] / norm for v,norm in zip(vectors,norms)])
        gram += block @ block.T
        assert time.monotonic()-start < 60
    eigen,basis = np.linalg.eigh(gram)
    rank_bound = 32*np.finfo(np.float64).eps*max(float(eigen[-1]),1.)
    assert eigen[0] > rank_bound, 'Retain all32 RAW loss dimensions; stop if numerically deficient'
    q = basis*np.sqrt(eigen)[None,:]
    fractions = np.array([1. if r['metric']=='feature_MSE' else .1 for r in records])
    target = fractions*losses/norms
    primal = minimize(lambda z:.5*float(z@z),np.zeros(32),jac=lambda z:z,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda z:q@z-target,jac=lambda z:q)],
        options=dict(ftol=1e-12,maxiter=2000))
    assert primal.success, primal.message
    active = q@primal.x-target <= 1e-9
    dual = np.zeros(32)
    dual[active],residual = nnls(q[active].T,primal.x,maxiter=10000)
    slack = gram@dual-target
    assert dual.min()>=0 and slack.min()>-1e-9
    assert np.linalg.norm(q.T@dual-primal.x)<1e-8 and np.max(abs(slack*dual))<1e-9
    direction = np.empty(len(vectors[0]),np.float32)
    for begin in range(0,len(direction),65536):
        block = np.stack([v[begin:begin+65536]/norm for v,norm in zip(vectors,norms)])
        direction[begin:begin+65536] = (-dual@block).astype(np.float32)
    dots = np.array([float(v@direction.astype(np.float64)) for v in vectors])
    assert np.all(-dots/losses >= fractions-1e-6), 'Float32 direction loses a constraint'
    assert time.monotonic()-start<60
    return direction,dict(normalized_Gram=gram.tolist(),gradient_norms=norms.tolist(),
        normalized_targets=target.tolist(),dual_multipliers=dual.tolist(),
        direction_norm=float(np.linalg.norm(direction.astype(np.float64))),
        direction_dots=dots.tolist(),fractional_derivative_descent=(-dots/losses).tolist(),
        target_fractions=fractions.tolist(),rank_bound=rank_bound,smallest_eigenvalue=float(eigen[0]),
        all32_RAW_loss_dimensions_retained=True,no_ridge_or_loss_removal=True,NNLS_residual=float(residual),
        seconds=time.monotonic()-start,local_derivatives_do_not_guarantee_finite_preservation=True)


def bounded_rate(initial,current,direction,initial_all37_norm):
    # Keep the original physical parameter-distance limit over the ENTIRE run,
    # not a fresh limit for each accepted step.
    delta = current.astype(np.float64)-initial.astype(np.float64)
    d = direction.astype(np.float64)
    radius = .001*initial_all37_norm
    aa,bb,cc = float(d@d),float(delta@d),float(delta@delta)-radius**2
    assert aa>0 and cc <= 1e-12
    root = (-bb+np.sqrt(max(0.,bb*bb-aa*cc)))/aa
    value = min(.02,float(root)*.999)
    assert np.isfinite(value) and value>1e-8, 'No remaining finite trust-region step'
    return value


def preservation_failures(before,after):
    assert len(before)==len(after)==17 and set(before)==set(after)
    failures=[]
    for key in sorted(before):
        assert before[key]['cases']==after[key]['cases']
        for metric in ['MSE','SSIM','ArcFace_observed_fixed']:
            b,a = before[key][metric],after[key][metric]
            bad = a>b+1e-12 if metric=='MSE' else a<b-1e-6
            if bad:failures.append(dict(group=key,metric=metric,before=b,after=a))
    return failures


def decision(initial,previous,after):
    for collection in [initial,previous,after]:
        assert set(collection)=={'raw','png'}
        for groups in collection.values():
            assert len(groups)==17
            for row in groups.values():
                assert row['cases']>0
                assert all(np.isfinite(row[m]) for m in ['MSE','SSIM','ArcFace_observed_fixed',
                    'landmark_high_frequency_MSE','constant_mean_shift_only_MSE']), 'Nonfinite scientific metric; retain stop'
                assert row['MSE']>=0 and row['landmark_high_frequency_MSE']>0
    stages={}
    for stage in ['raw','png']:
        baseline,old,new = initial[stage],previous[stage],after[stage]
        failures=preservation_failures(baseline,new)
        current_failures=preservation_failures(old,new)
        source_keys=[key for key in baseline if key.endswith('/degraded')]
        gain=1-new['degraded']['landmark_high_frequency_MSE']/baseline['degraded']['landmark_high_frequency_MSE']
        source_gain={k:1-new[k]['landmark_high_frequency_MSE']/baseline[k]['landmark_high_frequency_MSE'] for k in source_keys}
        progress={k:1-new[k]['landmark_high_frequency_MSE']/old[k]['landmark_high_frequency_MSE'] for k in ['degraded']+source_keys}
        b,a=baseline['degraded'],new['degraded']
        fraction=max(0.,b['MSE']-a['constant_mean_shift_only_MSE'])/max(b['MSE']-a['MSE'],1e-12)
        safe=not failures and not current_failures and fraction<=.2 and all(v>0 for v in progress.values())
        stages[stage]=dict(initial_preservation_failures=failures,current_preservation_failures=current_failures,
            relative_feature_gain=gain,source_feature_gains=source_gain,
            step_source_and_global_progress=progress,brightness_gain_fraction=fraction,
            accept_step=safe,one_percent_checkpoint_pass=safe and gain>=.01 and all(v>=.01 for v in source_gain.values()))
    return dict(stages=stages,accept_step=all(r['accept_step'] for r in stages.values()),
                one_percent_checkpoint_pass=all(r['one_percent_checkpoint_pass'] for r in stages.values()))
