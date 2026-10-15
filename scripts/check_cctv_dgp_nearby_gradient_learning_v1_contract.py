"""Real arithmetic regression: initial-only descent can fail a nearby constraint."""
import importlib.util
import numpy as np


def main():
    assert importlib.util.find_spec('cctv_dgp_nearby_gradient_learning_v1_contract'), 'Nearby-state solver is not implemented'
    from cctv_dgp_nearby_gradient_learning_v1_contract import solve_direction
    gradients = []; records = []
    for site, vector in zip(['initial', 'plus02', 'minus02'], [[1., 0.], [-1., 2.], [1., -.5]]):
        for i in range(32):
            gradients.append(np.asarray(vector, dtype=np.float64))
            records.append(dict(site=site, metric='feature_MSE' if i == 31 else 'MSE', value=1., key=f'{site}/g{i:02d}'))
    initial_only = np.asarray([-1., 0.], dtype=np.float32)
    assert float(gradients[32] @ initial_only) > 0, 'Fixture must reproduce an unsafe nearby descent prediction'
    direction, proof = solve_direction(gradients, records)
    assert direction.dtype == np.float32 and proof['all96_constraints_retained'] and proof['numerical_span_rank'] == 2
    fractions = np.asarray([1. if r['metric'] == 'feature_MSE' else .1 for r in records])
    actual = -np.asarray([g @ direction.astype(np.float64) for g in gradients])
    assert np.all(actual >= fractions - 1e-6)
    np.testing.assert_allclose(direction, [-5/3, -4/3], rtol=1e-5, atol=1e-6)
    # An opposing nearby derivative is truly infeasible; no group can be discarded.
    impossible = [g.copy() for g in gradients]; impossible[32] = np.asarray([-1., 0.])
    try:
        solve_direction(impossible, records)
    except AssertionError:
        rejected_infeasible = True
    else:
        raise AssertionError('Infeasible nearby-state constraints were removed')
    invalid = [g.copy() for g in gradients]; invalid[0][0] = np.nan
    try:
        solve_direction(invalid, records)
    except AssertionError:
        rejected_nonfinite = True
    else:
        raise AssertionError('Nonfinite gradient accepted')
    print(dict(complete=True, unsafe_initial_only_direction_reproduced=True,
        retained96_constraints=True, rank2_with_all_rows=True,
        infeasible_and_nonfinite_rejected=rejected_infeasible and rejected_nonfinite, neural_calls=0))


if __name__ == '__main__':
    main()
