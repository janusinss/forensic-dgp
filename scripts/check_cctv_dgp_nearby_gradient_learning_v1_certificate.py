"""Independent certificate rejects a forged direction and a dropped nearby row."""
import importlib.util
import numpy as np
from cctv_dgp_nearby_gradient_learning_v1_contract import solve_direction


def main():
    assert importlib.util.find_spec('cctv_dgp_nearby_gradient_certificate_v1'), 'Independent nearby certificate replay is not implemented'
    from cctv_dgp_nearby_gradient_certificate_v1 import audit_certificate
    vectors = []; records = []
    for site, g in zip(['initial', 'plus02', 'minus02'], [[1., 0.], [-1., 2.], [1., -.5]]):
        for i in range(32):
            v = np.asarray(g, np.float64); vectors.append(v)
            records.append(dict(site=site, metric='feature_MSE' if i == 31 else 'MSE', value=1., key=f'{site}/g{i:02d}'))
    direction, proof = solve_direction(vectors, records)
    assert audit_certificate(vectors, records, direction, proof)['all96_original_dot_inequalities_pass']
    wrong = direction.copy(); wrong[0] = .1
    try:audit_certificate(vectors, records, wrong, proof)
    except AssertionError:pass
    else:raise AssertionError('Forged nearby direction accepted')
    try:audit_certificate(vectors[:-1], records[:-1], direction, proof)
    except AssertionError:pass
    else:raise AssertionError('Dropped nearby inequality accepted')
    print(dict(complete=True, rank2_96_row_certificate_independently_replayed=True,
               forged_direction_and_dropped_constraint_rejected=True, neural_calls=0))


if __name__ == '__main__':main()
