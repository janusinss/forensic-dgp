"""Report the existing cross-run bound without changing it or passing a miss."""
import numpy as np


def report(actual, expected):
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    bound = 1e-8 + 1e-4 * np.abs(expected)
    difference = np.abs(actual - expected)
    bad = difference > bound
    norm_pass = bool(np.isclose(np.linalg.norm(actual), np.linalg.norm(expected), rtol=1e-4, atol=1e-10))
    indices = np.flatnonzero(bad)
    return dict(original_elementwise_rtol=1e-4, original_elementwise_atol=1e-8,
        original_norm_rtol=1e-4, original_norm_atol=1e-10, elements=int(actual.size),
        mismatched_elements=int(bad.sum()), norm_check_pass=norm_pass,
        original_parity_check_pass=not bad.any() and norm_pass,
        maximum_absolute_difference=float(difference.max()),
        relative_vector_difference=float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
        violations=[dict(index=int(i), actual=float(actual[i]), expected=float(expected[i]),
                         absolute_difference=float(difference[i]), original_bound=float(bound[i])) for i in indices])
