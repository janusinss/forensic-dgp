"""Compare learning subspaces using returned arrays only; no neural execution."""
from datetime import datetime, timezone
import time

import numpy as np
from scipy.optimize import minimize, nnls

from import_cctv_dgp_live_conditioning_v1_r1_return import (
    ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write)


def solve(gram, losses, norms, groups):
    eigen, basis = np.linalg.eigh(gram)
    bound = len(groups) * np.finfo(np.float64).eps * max(float(eigen[-1]), 1.)
    if eigen[0] <= bound:
        return dict(supported=False, reason='Full loss span is numerically rank deficient; no rows removed',
                    smallest_eigenvalue=float(eigen[0]), rank_bound=bound)
    q = basis * np.sqrt(eigen)[None, :]
    fractions = np.array([1. if g['metric'] == 'feature_MSE' else .1 for g in groups])
    target = fractions * losses / norms
    result = minimize(lambda z: .5 * float(z @ z), np.zeros(len(groups)),
        jac=lambda z: z, method='SLSQP',
        constraints=[dict(type='ineq', fun=lambda z: q @ z - target, jac=lambda z: q)],
        options=dict(ftol=1e-12, maxiter=2000))
    if not result.success:
        return dict(supported=False, reason=str(result.message), no_constraint_removal=True)
    active = q @ result.x - target <= 1e-9
    dual = np.zeros(len(groups))
    dual[active], residual = nnls(q[active].T, result.x, maxiter=10000)
    slack = gram @ dual - target
    stationarity = float(np.linalg.norm(q.T @ dual - result.x))
    assert dual.min() >= 0 and slack.min() > -1e-9 and stationarity < 1e-8
    assert np.max(abs(slack * dual)) < 1e-9
    return dict(supported=True, normalized_Gram=gram.tolist(), normalized_targets=target.tolist(),
        dual_multipliers=dual.tolist(), smallest_eigenvalue=float(eigen[0]),
        condition_number=float(eigen[-1] / eigen[0]), rank_bound=bound,
        all32_dimensions_retained=True, no_ridge_or_constraint_removal=True,
        stationary_error=stationarity, NNLS_residual=float(residual),
        maximum_complementarity_error=float(np.max(abs(slack * dual))),
        minimum_primal_slack=float(slack.min()),
        direction_norm=float(np.linalg.norm(result.x)),
        ideal_1percent_detail_step_norm=float(.01 * np.linalg.norm(result.x)),
        target_fractions_per_unit_rate=fractions.tolist())


def main():
    started = time.monotonic()
    receipt = ANALYSIS / 'learning_subspace_review.json'
    assert not receipt.exists()
    audit_path = ANALYSIS / 'independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['protocol_sha256'] == PIN
    p = read(RETURN / 'protocol.json')
    out = RETURN / 'outputs/live_conditioning_v1_r1'
    run = read(out / 'run_results.json')
    assert sha(out / 'run_results.json') == audit['run_results_sha256']
    groups = run['groups']
    names = p['learning_names']
    sizes = [int(np.prod(p['selected_parameter_shapes'][name])) for name in names]
    edges = np.cumsum([0] + sizes)
    assert edges[-1] == p['parameters'] == 3364643
    condition = [i for i, name in enumerate(names) if name.startswith(
        ('live.style_conditioner.', 'live.spatial_conditioners.'))]
    assert len(condition) == 12
    vectors, bindings = [], []
    for group in groups:
        path = out / group['file']
        assert sha(path) == group['sha256']
        vector = np.load(path, allow_pickle=False, mmap_mode='r')
        assert vector.dtype == np.float64 and vector.shape == (3364643,)
        vectors.append(vector)
        bindings.append(dict(key=group['key'], file=path.relative_to(ROOT).as_posix(), sha256=group['sha256']))
    losses = np.array([g['value'] for g in groups])
    assert np.isfinite(losses).all() and (losses > 0).all()
    reviews = {}
    for label, partitions in [('all37', list(range(37))), ('conditioning12', condition)]:
        unscaled = np.zeros((32, 32))
        for i in partitions:
            for begin in range(edges[i], edges[i + 1], 65536):
                end = min(begin + 65536, edges[i + 1])
                block = np.stack([v[begin:end] for v in vectors])
                assert np.isfinite(block).all()
                unscaled += block @ block.T
                assert time.monotonic() - started < 180
        norms = np.sqrt(unscaled.diagonal())
        assert (norms > 0).all()
        gram = unscaled / norms[:, None] / norms[None, :]
        result = solve(gram, losses, norms, groups)
        result.update(tensor_count=len(partitions), values=sum(sizes[i] for i in partitions),
                      names=[names[i] for i in partitions], gradient_norms=norms.tolist())
        if result['supported']:
            dual = np.array(result['dual_multipliers'])
            # A discarded plain-array direction is only arithmetic. Never load
            # it into a neural model or change any checkpoint.
            dots = np.zeros(32)
            norm2 = 0.
            for i in partitions:
                for begin in range(edges[i], edges[i + 1], 65536):
                    end = min(begin + 65536, edges[i + 1])
                    block = np.stack([v[begin:end] for v in vectors])
                    direction = (-dual @ (block / norms[:, None])).astype(np.float32).astype(np.float64)
                    dots += block @ direction
                    norm2 += float(direction @ direction)
            fractions = -dots / losses
            target = np.array(result['target_fractions_per_unit_rate'])
            assert np.all(fractions >= target - 1e-6)
            result.update(float32_direction_norm=float(np.sqrt(norm2)),
                          actual_float32_direction_dots=dots.tolist(),
                          fractional_derivative_descent_per_unit_rate=fractions.tolist(),
                          all32_local_float32_constraints_pass=True)
        reviews[label] = result
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), independent_return_audit_sha256=sha(audit_path),
        run_results_sha256=sha(out / 'run_results.json'), protocol_sha256=PIN,
        aggregate_gradient_bindings=bindings, group_keys=[g['key'] for g in groups],
        scientific_loss_values=losses.tolist(), subspaces=reviews,
        comparison_not_a_neural_ablation_or_new_model_result=True,
        PNG_derivatives_not_measured_for_this_new_architecture=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        raw_common_descent_does_not_qualify_finite_PNG_preservation_or_epochs=True,
        no_new_checkpoint=True, model_qualified=False, goal_complete=False,
        seconds=time.monotonic() - started))
    print({key: {k: value.get(k) for k in ['supported', 'direction_norm', 'ideal_1percent_detail_step_norm',
           'condition_number']} for key, value in reviews.items()})


if __name__ == '__main__':
    main()
