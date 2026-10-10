"""Deterministic PCGrad arithmetic only; no models, derivatives or optimizer."""
import numpy as np

SEED = 20261008
TERMS = 7


def task_orders(update):
    assert type(update) is int and update >= 1
    rng = np.random.default_rng(SEED + update - 1)
    return [[int(j) for j in rng.permutation(TERMS) if j != i] for i in range(TERMS)]


def combine(components, update):
    a = np.asarray(components)
    assert a.ndim == 2 and a.shape[0] == TERMS and a.shape[1] > 0
    assert a.dtype in (np.float32, np.float64) and np.isfinite(a).all()
    original = a.astype(np.float64, copy=True)
    projected = original.copy(); orders = task_orders(update); conflicts = 0
    for i, order in enumerate(orders):
        for j in order:
            dot = float(projected[i] @ original[j])
            norm2 = float(original[j] @ original[j])
            if norm2 > 0 and dot < 0:
                projected[i] -= (dot / norm2) * original[j]
                conflicts += 1
    merged = projected.sum(0)
    assert np.isfinite(merged).all()
    return merged.astype(np.float32), {
        'orders': orders, 'conflict_projections': conflicts,
        'component_norms': np.linalg.norm(original, axis=1).tolist(),
        'component_values_not_reweighted': True,
        'sum_direction_derivatives': (original @ -original.sum(0)).tolist(),
        'projected_direction_derivatives': (original @ -merged).tolist(),
        'combined_float64_norm': float(np.linalg.norm(merged)),
        'first_order_only_no_finite_preservation_guarantee': True,
    }
