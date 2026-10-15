"""Original parity report, with explicit undefined relative error for an exact zero reference."""
import numpy as np
from cctv_dgp_cross_run_gradient_report_v1 import report as original_report


def report(actual,expected):
    assert actual.shape==expected.shape and actual.ndim==1 and actual.size>0
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    if np.any(expected):return original_report(actual,expected)
    difference=np.abs(actual-expected);bad=difference>1e-8;indices=np.flatnonzero(bad)
    norm_pass=bool(np.isclose(np.linalg.norm(actual),np.linalg.norm(expected),rtol=1e-4,atol=1e-10))
    return dict(original_elementwise_rtol=1e-4,original_elementwise_atol=1e-8,
        original_norm_rtol=1e-4,original_norm_atol=1e-10,elements=int(actual.size),mismatched_elements=int(bad.sum()),
        norm_check_pass=norm_pass,original_parity_check_pass=not bad.any() and norm_pass,
        maximum_absolute_difference=float(difference.max()),relative_vector_difference=None,
        zero_reference_relative_error_is_undefined=True,
        violations=[dict(index=int(i),actual=float(actual[i]),expected=float(expected[i]),
                         absolute_difference=float(difference[i]),original_bound=1e-8) for i in indices])
