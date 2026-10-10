"""Project saved displacements against every non-hinged preservation row.

Array geometry only: no models, differentiation, training or torch optimizer.
For theta <- theta - d, enforce U @ d >= 0. Positive row scaling leaves
the constraint set unchanged. Finite raw and delivered PNG outputs still need
all original preservation checks; this is only a local linear approximation.
"""
import numpy as np

POLICY = 'own-all-group-nonhinged-displacement-cone-v35'
MAX_ROWS = 109
KKT_RELATIVE_TOLERANCE = 2e-10


def project_vectors(gradients, proposal):
    from scipy.optimize import nnls, minimize, LinearConstraint

    gradients = np.asarray(gradients, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if (gradients.ndim != 2 or not 1 <= gradients.shape[0] <= MAX_ROWS
            or proposal.ndim != 1 or gradients.shape[1] != len(proposal)
            or not np.isfinite(gradients).all() or not np.isfinite(proposal).all()):
        raise ValueError('Finite preservation rows and matching displacement required')
    norms = np.linalg.norm(gradients, axis=1)
    nonzero = np.flatnonzero(norms > 0)
    unit = gradients[nonzero] / norms[nonzero, None]
    length = float(np.linalg.norm(proposal))
    if length == 0 or len(nonzero) == 0:
        return proposal.copy(), {'policy': POLICY, 'nonzero_indices': nonzero.tolist(),
            'proposal_norm': length, 'projected_norm': length, 'multipliers': [0.] * len(nonzero),
            'KKT_tolerance': KKT_RELATIVE_TOLERANCE * max(length, 1e-12),
            'raw_dots_before': (gradients @ proposal).tolist(),
            'raw_dots_after': (gradients @ proposal).tolist(),
            'gradient_calls': 0, 'neural_calls': 0, 'optimizer_updates': 0}
    gram = unit @ unit.T
    dots = unit @ (proposal / length)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -2e-10:
        raise ArithmeticError('Preservation Gram is not positive semidefinite')
    keep = eigenvalues > 1e-12
    values, vectors = eigenvalues[keep], eigenvectors[:, keep]
    # A contains every original constraint in orthonormal row-space coordinates.
    A = vectors * np.sqrt(values)[None, :]
    b = (vectors.T @ dots) / np.sqrt(values)
    null_error = float(np.max(np.abs(dots - A @ b)))
    if null_error > KKT_RELATIVE_TOLERANCE:
        raise ArithmeticError('Gram reduction loses a preservation dot; do not drop constraints')
    multipliers, residual = nnls(A.T, -b, maxiter=100 * len(nonzero))
    corrected = proposal + length * (multipliers @ unit)
    actual = unit @ corrected
    tolerance = KKT_RELATIVE_TOLERANCE * length
    if (actual.min() < -tolerance or multipliers.min() < 0
            or np.max(np.abs(multipliers * (actual / length))) > KKT_RELATIVE_TOLERANCE):
        raise ArithmeticError('Dual projection fails direct full-row KKT checks')
    # Independent primal solve, starting from the feasible zero displacement.
    primal = minimize(lambda t: .5 * float((t-b) @ (t-b)), np.zeros_like(b),
        jac=lambda t: t-b, constraints=[LinearConstraint(A, 0., np.inf)],
        method='SLSQP', options={'ftol': 1e-14, 'maxiter': 2000})
    primal_error = float(np.linalg.norm(primal.x - (b + A.T @ multipliers)))
    if (not primal.success or (A @ primal.x).min() < -2e-8
            or primal_error > 2e-7):
        raise ArithmeticError('Independent primal solve disagrees; stop rather than change a guard')
    return corrected, {'policy': POLICY, 'nonzero_indices': nonzero.tolist(),
        'constraint_count': len(nonzero), 'gram_rank': int(keep.sum()),
        'gram_minimum_eigenvalue': float(eigenvalues.min()),
        'gram_reduction_dot_error': null_error, 'proposal_norm': length,
        'projected_norm': float(np.linalg.norm(corrected)),
        'correction_norm': float(np.linalg.norm(corrected-proposal)),
        'multipliers': (length * multipliers).tolist(),
        'unit_dots_before': (unit @ proposal).tolist(), 'unit_dots_after': actual.tolist(),
        'raw_dots_before': (gradients @ proposal).tolist(),
        'raw_dots_after': (gradients @ corrected).tolist(),
        'KKT_tolerance': tolerance, 'independent_primal_relative_L2_error': primal_error,
        'primal_iterations': primal.nit, 'nnls_residual': float(residual),
        'gradient_calls': 0, 'neural_calls': 0, 'optimizer_updates': 0,
        'finite_output_preservation_implied': False}
