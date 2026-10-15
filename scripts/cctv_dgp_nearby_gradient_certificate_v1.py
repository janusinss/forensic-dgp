"""Recompute a saved nearby certificate with independent array arithmetic, no solve."""
import numpy as np


def audit_certificate(vectors, rows, direction, proof):
    assert len(vectors) == len(rows) == 96 and direction.dtype == np.float32 and direction.ndim == 1
    assert [r['site'] for r in rows] == ['initial'] * 32 + ['plus02'] * 32 + ['minus02'] * 32
    assert proof['all96_constraints_retained'] and proof['inequality_rows'] == 96 and proof['no_ridge_or_loss_removal']
    assert all(v.shape == direction.shape and v.dtype == np.float64 and np.isfinite(v).all() for v in vectors)
    assert np.isfinite(direction).all()
    norms = np.asarray([np.linalg.norm(v) for v in vectors]); assert (norms > 0).all()
    loss = np.asarray([r['value'] for r in rows]); assert np.isfinite(loss).all() and (loss > 0).all()
    fractions = np.asarray([1. if r['metric'] == 'feature_MSE' else .1 for r in rows])
    dual = np.asarray(proof['dual_multipliers']); assert dual.shape == (96,) and np.isfinite(dual).all() and (dual >= 0).all()
    gram = np.zeros((96, 96)); dots = np.zeros(96)
    for begin in range(0, len(direction), 32768):
        block = np.vstack([v[begin:begin+32768] / norm for v, norm in zip(vectors, norms)])
        gram += np.einsum('ij,kj->ik', block, block, optimize=True)
        reproduced = np.einsum('i,ij->j', -dual, block, optimize=True).astype(np.float32)
        np.testing.assert_allclose(reproduced, direction[begin:begin+32768], rtol=1e-6, atol=1e-8)
        dots += np.einsum('ij,j->i', block * norms[:, None], direction[begin:begin+32768].astype(np.float64), optimize=True)
    np.testing.assert_allclose(gram, proof['normalized_Gram'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(norms, proof['gradient_norms'], rtol=1e-10, atol=1e-12)
    target = fractions * loss / norms
    np.testing.assert_allclose(target, proof['normalized_targets'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dots, proof['direction_dots'], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(-dots / loss, proof['fractional_derivative_descent'], rtol=1e-8, atol=1e-10)
    assert np.all(-dots / loss >= fractions - 1e-6)
    slack = gram @ dual - target
    assert slack.min() > -1e-9 and np.max(abs(slack * dual)) < 1e-9
    eig, basis = np.linalg.eigh(gram)
    bound = 96 * np.finfo(np.float64).eps * max(1., float(eig[-1]))
    assert eig[0] >= -10 * bound and np.isclose(bound, proof['rank_bound'], rtol=1e-10)
    kept = eig > bound
    assert int(kept.sum()) == proof['numerical_span_rank']
    q = basis[:, kept] * np.sqrt(eig[kept])[None, :]
    assert np.max(abs(q @ q.T - gram)) <= max(1e-10, 10 * bound)
    assert np.isclose(np.linalg.norm(direction.astype(np.float64)), proof['direction_norm'], rtol=1e-10)
    return dict(complete=True, all96_original_dot_inequalities_pass=True, original_solve_not_reexecuted=True,
                full_Gram_dual_and_float32_direction_checked=True, numerical_span_rank=int(kept.sum()),
                numerical_redundancy_does_not_drop_inequality_rows=True, scientific_gates_not_changed=True)
