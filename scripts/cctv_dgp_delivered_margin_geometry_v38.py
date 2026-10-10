"""Finite array-only affine projection; empirical margins are not loss bounds."""
import time
import numpy as np

MAX_ROWS = 210
MAX_PARAMETERS = 978243
RELATIVE_KKT_TOLERANCE = 2e-10


def project_affine(gradients, proposal, targets, cap_seconds=120):
    from scipy.optimize import LinearConstraint, linprog, minimize, nnls
    started = time.monotonic()
    gradients = np.asarray(gradients, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64); targets = np.asarray(targets, dtype=np.float64)
    if (gradients.ndim != 2 or not 1 <= gradients.shape[0] <= MAX_ROWS or
            gradients.shape[1] > MAX_PARAMETERS or proposal.shape != (gradients.shape[1],) or
            targets.shape != (len(gradients),) or not np.isfinite(gradients).all() or
            not np.isfinite(proposal).all() or not np.isfinite(targets).all() or (targets < 0).any()):
        raise ValueError('Finite matched gradients, proposal and nonnegative margins required')
    norms = np.linalg.norm(gradients, axis=1)
    if not (norms > 0).all(): raise ValueError('Retain and investigate zero guard; no row omission')
    length = float(np.linalg.norm(proposal))
    if length <= 0: raise ValueError('Nonzero original proposal required')
    unit = gradients / norms[:, None]
    gram = unit @ unit.T; dots = unit @ (proposal / length)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -2e-10: raise ArithmeticError('Invalid guard Gram')
    keep = eigenvalues > 1e-12
    A = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    y = (eigenvectors[:, keep].T @ dots) / np.sqrt(eigenvalues[keep])
    reduction = float(np.max(np.abs(A @ y - dots)))
    if reduction > 2e-10: raise ArithmeticError('Row-space reduction changes a preservation dot')
    bounds = targets / (norms * length)
    feasibility = linprog(np.zeros(len(y)), A_ub=-A, b_ub=-bounds,
                          bounds=[(None, None)] * len(y), method='highs',
                          options={'maxiter': 2000, 'time_limit': min(60, cap_seconds)})
    receipt = {'complete': True, 'rows': len(gradients), 'rank': int(keep.sum()),
               'linear_feasible': bool(feasibility.success), 'LP_status': int(feasibility.status),
               'LP_message': str(feasibility.message), 'gram_reduction_dot_error': reduction,
               'geometry_qualified_for_disposable_probe': False, 'empirical_margins_not_validated_bounds': True,
               'finite_preservation_implied': False, 'local_gradient_calls': 0, 'optimizer_updates': 0,
               'neural_calls': 0, 'proposal_norm': length}
    if not feasibility.success:
        receipt['seconds'] = time.monotonic() - started
        return None, receipt

    def deadline(_):
        if time.monotonic() - started > cap_seconds: raise TimeoutError('Finite geometry calculation limit')

    primal = minimize(lambda z: .5 * float((z - y) @ (z - y)), feasibility.x,
                      jac=lambda z: z - y, constraints=[LinearConstraint(A, bounds, np.inf)],
                      method='SLSQP', callback=deadline, options={'ftol': 1e-14, 'maxiter': 2000})
    slack = A @ primal.x - bounds; active = np.flatnonzero(slack <= 2e-8)
    multipliers = np.zeros(len(gradients))
    multipliers[active], residual = nnls(A[active].T, primal.x - y, maxiter=100 * len(gradients))
    dual = minimize(lambda z: .5 * float(z @ gram @ z) + float((dots - bounds) @ z), np.zeros(len(gradients)),
                    jac=lambda z: gram @ z + dots - bounds, bounds=[(0., None)] * len(gradients),
                    method='SLSQP', callback=deadline, options={'ftol': 1e-14, 'maxiter': 2000})
    difference = float(np.linalg.norm(primal.x - (y + A.T @ dual.x)))
    # Reconstruct from the independently checked nonnegative KKT multipliers.
    candidate = proposal + length * (multipliers @ unit)
    direct_slack = unit @ candidate - targets / norms
    complementarity = float(np.max(np.abs(multipliers * slack)))
    stationarity = float(np.linalg.norm(A.T @ multipliers - (primal.x - y)))
    ratio = float(np.linalg.norm(candidate) / length)
    stable = bool(primal.success and dual.success and slack.min() >= -2e-10 and
                  stationarity <= 2e-10 and complementarity <= 2e-10 and difference <= 2e-7 and
                  direct_slack.min() >= -2e-10 * length and (multipliers >= 0).all())
    receipt.update({'primal_success': bool(primal.success), 'dual_success': bool(dual.success),
        'primal_iterations': int(primal.nit), 'dual_iterations': int(dual.nit),
        'minimum_normalized_slack': float(slack.min()), 'minimum_direct_unit_clearance_residual': float(direct_slack.min()),
        'KKT_stationarity_relative_error': stationarity, 'KKT_complementarity_relative_error': complementarity,
        'independent_primal_dual_relative_L2_error': difference, 'nnls_residual': float(residual),
        'candidate_magnitude_ratio': ratio, 'maximum_permitted_magnitude_ratio': 2.,
        'multipliers': (length * multipliers).tolist(), 'clearance_targets': targets.tolist(),
        'raw_dots_after': (gradients @ candidate).tolist(), 'KKT_tolerance': 2e-10 * length,
        'geometry_qualified_for_disposable_probe': stable and ratio <= 2.,
        'seconds': time.monotonic() - started, 'cap_seconds': cap_seconds})
    if receipt['seconds'] > cap_seconds: raise TimeoutError('Geometry calculation limit')
    return candidate, receipt
