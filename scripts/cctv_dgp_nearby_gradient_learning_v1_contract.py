"""96-row nearby-state constrained direction; pure saved-array arithmetic only."""
import time
import numpy as np
from scipy.optimize import minimize, nnls
from cctv_dgp_raw_conditioning_learning_v1_r1_contract import bounded_rate, decision, preservation_failures


def solve_direction(vectors, records):
    started = time.monotonic()
    def clock(*_args):
        assert time.monotonic() - started < 60, 'Finite nearby-state solve stop'
    assert len(vectors) == len(records) == 96
    assert [r['site'] for r in records] == ['initial'] * 32 + ['plus02'] * 32 + ['minus02'] * 32
    assert all(v.ndim == 1 and v.shape == vectors[0].shape and v.dtype == np.float64 and np.isfinite(v).all() for v in vectors)
    norms = np.asarray([float(np.linalg.norm(v)) for v in vectors])
    losses = np.asarray([r['value'] for r in records], dtype=np.float64)
    assert np.isfinite(norms).all() and (norms > 0).all() and np.isfinite(losses).all() and (losses > 0).all()
    fractions = np.asarray([1. if r['metric'] == 'feature_MSE' else .1 for r in records])
    target = fractions * losses / norms
    gram = np.zeros((96, 96), dtype=np.float64)
    for begin in range(0, len(vectors[0]), 32768):
        block = np.stack([v[begin:begin+32768] / norm for v, norm in zip(vectors, norms)])
        gram += block @ block.T; clock()
    eigen, basis = np.linalg.eigh(gram)
    rank_bound = 96 * np.finfo(np.float64).eps * max(1., float(eigen[-1]))
    assert eigen[0] >= -10 * rank_bound, 'Normalized Gram is not numerically PSD'
    kept = eigen > rank_bound; assert kept.any()
    # Represent the span of gradient vectors; keep ALL 96 inequality rows.
    q = basis[:, kept] * np.sqrt(eigen[kept])[None, :]
    reconstruction_error = float(np.max(np.abs(q @ q.T - gram)))
    assert reconstruction_error <= max(1e-10, 10 * rank_bound)
    primal = minimize(lambda z: .5 * float(z @ z), np.zeros(int(kept.sum())), jac=lambda z: z,
        method='SLSQP', constraints=[dict(type='ineq', fun=lambda z: q @ z - target, jac=lambda z: q)],
        callback=clock, options=dict(ftol=1e-12, maxiter=2000))
    clock(); assert primal.success, 'No certified common nearby direction: ' + str(primal.message)
    active = q @ primal.x - target <= 1e-9
    assert active.any()
    dual = np.zeros(96)
    dual[active], residual = nnls(q[active].T, primal.x, maxiter=10000)
    slack = gram @ dual - target
    assert dual.min() >= 0 and slack.min() > -1e-9
    assert np.linalg.norm(q.T @ dual - primal.x) < 1e-8 and np.max(abs(slack * dual)) < 1e-9
    direction = np.empty(len(vectors[0]), dtype=np.float32)
    for begin in range(0, len(direction), 32768):
        block = np.stack([v[begin:begin+32768] / norm for v, norm in zip(vectors, norms)])
        direction[begin:begin+32768] = (-dual @ block).astype(np.float32); clock()
    dots = np.asarray([float(v @ direction.astype(np.float64)) for v in vectors])
    assert np.isfinite(direction).all() and np.all(-dots / losses >= fractions - 1e-6), 'Rounded direction loses a retained constraint'
    clock()
    return direction, dict(normalized_Gram=gram.tolist(), gradient_norms=norms.tolist(),
        normalized_targets=target.tolist(), dual_multipliers=dual.tolist(),
        direction_norm=float(np.linalg.norm(direction.astype(np.float64))), direction_dots=dots.tolist(),
        fractional_derivative_descent=(-dots / losses).tolist(), target_fractions=fractions.tolist(),
        rank_bound=rank_bound, smallest_eigenvalue=float(eigen[0]), numerical_span_rank=int(kept.sum()),
        span_Gram_reconstruction_max_error=reconstruction_error,
        inequality_rows=96, all96_constraints_retained=True, no_ridge_or_loss_removal=True,
        NNLS_residual=float(residual), seconds=time.monotonic() - started,
        initial_only_direction_not_reused=True, nearby_derivatives_do_not_guarantee_finite_RAW_PNG_preservation=True)
