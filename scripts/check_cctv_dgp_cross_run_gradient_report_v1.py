"""Known failed return value plus exact, boundary and invalid-array fixtures."""
import sys
import numpy as np
from cctv_dgp_cross_run_gradient_report_v1 import report
from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ANALYSIS, sha, write


def main():
    actual = np.array([-0.00012899858411401406], np.float64)
    expected = np.array([-0.00012897558044642089], np.float64)
    failed = report(actual, expected)
    assert not failed['original_parity_check_pass'] and failed['mismatched_elements'] == 1
    assert failed['violations'][0]['absolute_difference'] > failed['violations'][0]['original_bound']
    assert not np.allclose(actual, expected, rtol=1e-4, atol=1e-8)
    fixtures = []
    for name, a, b in [('exact', np.array([1.,2.]), np.array([1.,2.])),
                        ('within_bound', np.array([1.+1e-6,2.]), np.array([1.,2.])),
                        ('outside_bound', np.array([1.001,2.]), np.array([1.,2.]))]:
        row = report(a,b)
        assert row['original_parity_check_pass'] == bool(np.allclose(a,b,rtol=1e-4,atol=1e-8) and np.isclose(np.linalg.norm(a),np.linalg.norm(b),rtol=1e-4,atol=1e-10))
        fixtures.append(dict(name=name, pass_reported=row['original_parity_check_pass']))
    rejected = []
    for name, a, b in [('NaN',np.array([np.nan]),np.array([1.])),('Inf',np.array([np.inf]),np.array([1.])),
                        ('shape',np.array([1.,2.]),np.array([1.]))]:
        try: report(a,b)
        except AssertionError: rejected.append(name)
        else: raise AssertionError(name)
    write(ANALYSIS/'cross_run_report_check.json',dict(complete=True,checker_sha256=sha(__file__),
        known_original_stop_still_fails=failed, fixtures=fixtures, invalid_inputs_rejected=rejected,
        original_bounds_not_relaxed=True, new_neural_or_training_calls=0))
    print(dict(complete=True, original_failure_retained=True, invalid_inputs_rejected=rejected))


if __name__=='__main__':main()
