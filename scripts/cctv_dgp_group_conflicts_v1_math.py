"""Saved-vector arithmetic only. The dual solve never updates a model parameter."""
import time
import numpy as np
from scipy.optimize import minimize


def common_direction(matrix, minimum_cosine=1e-7):
    start = time.monotonic()
    assert matrix.dtype == np.float64 and matrix.ndim == 2 and np.isfinite(matrix).all()
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        return None, {'complete': True, 'common_direction_found': False,
                      'reason': 'Zero group gradient; no strictly improving certificate',
                      'gradient_norms': norms.tolist(), 'dual_solver_iterations': 0,
                      'global_infeasibility_proven': False}
    unit = matrix / norms[:, None]
    gram = unit @ unit.T
    assert np.allclose(gram, gram.T, rtol=0, atol=1e-12)
    count = len(matrix)
    result = minimize(lambda w: .5 * float(w @ gram @ w), np.full(count, 1. / count),
        jac=lambda w: gram @ w, method='SLSQP', bounds=[(0., 1.)] * count,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.,
                      'jac': lambda w: np.ones(count)}],
        options={'ftol': 1e-12, 'maxiter': 500, 'disp': False})
    assert time.monotonic() - start <= 120, 'Finite dual solver cap'
    weights = np.clip(result.x, 0., 1.)
    assert np.isfinite(weights).all() and weights.sum() > 0
    weights /= weights.sum()
    combined = weights @ unit
    norm = float(np.linalg.norm(combined))
    direction = -combined / norm if norm > 0 else np.zeros(matrix.shape[1], np.float64)
    cosines = -(unit @ direction)
    found = bool(result.success and norm > 0 and np.min(cosines) >= minimum_cosine)
    objective = float(weights @ gram @ weights)
    gap = 2. * (objective - float(np.min(gram @ weights)))
    receipt = {'complete': True, 'common_direction_found': found,
        'reason': 'Certified negative derivative for every recorded group' if found else
                  'No certified strictly improving direction from this bounded solve',
        'global_infeasibility_proven': False,
        'gradient_norms': norms.tolist(), 'normalized_Gram': gram.tolist(),
        'simplex_weights': weights.tolist(), 'combined_gradient_norm': norm,
        'minimum_normalized_descent_cosine': float(cosines.min()),
        'normalized_descent_cosines': cosines.tolist(),
        'raw_directional_derivatives': (matrix @ direction).tolist(),
        'dual_solver_iterations': int(result.nit), 'dual_solver_success': bool(result.success),
        'dual_solver_message': str(result.message), 'duality_gap': gap,
        'minimum_descent_cosine_requirement': minimum_cosine,
        'model_optimizer_updates': 0, 'seconds': time.monotonic() - start}
    return direction if found else None, receipt
