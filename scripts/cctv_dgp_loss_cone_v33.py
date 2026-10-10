"""Own finite half-space projection; NumPy arithmetic, no model or optimizer.

For a proposed parameter displacement p, minimize ||d-p||^2 subject to
g_i dot d >= 0 for every nonzero existing loss gradient. Parameters take -d.
Positive row scaling leaves the feasible cone unchanged. The <=7-row dual is
solved by enumerating every active set; no constraint is relaxed on failure.
This is a local linear approximation, not a guarantee for finite neural outputs.
"""
from itertools import combinations
import numpy as np

POLICY = 'own-seven-loss-displacement-cone-v33'
MAX_CONSTRAINTS = 7
KKT_RELATIVE_TOLERANCE = 2e-10


def solve_dual(gram, dots, proposal_norm):
    gram = np.asarray(gram, dtype=np.float64)
    dots = np.asarray(dots, dtype=np.float64)
    n = len(dots)
    if not (0 <= n <= MAX_CONSTRAINTS and gram.shape == (n, n)):
        raise ValueError('At most seven square Gram constraints')
    if not (np.isfinite(gram).all() and np.isfinite(dots).all()
            and np.isfinite(proposal_norm) and proposal_norm >= 0):
        raise ValueError('Finite Gram, dots and nonnegative proposal norm required')
    if n and (not np.allclose(gram, gram.T, rtol=0, atol=2e-12)
              or not np.allclose(np.diag(gram), 1, rtol=0, atol=2e-12)
              or np.linalg.eigvalsh(gram).min() < -2e-10):
        raise ValueError('Unit-row symmetric positive-semidefinite Gram required')
    tolerance = KKT_RELATIVE_TOLERANCE * max(float(proposal_norm), 1e-12)
    best = None
    subsets = 0
    for count in range(n + 1):
        for subset in combinations(range(n), count):
            subsets += 1
            multipliers = np.zeros(n, dtype=np.float64)
            if subset:
                indices = np.asarray(subset, dtype=int)
                block = gram[np.ix_(indices, indices)]
                solution = np.linalg.lstsq(block, -dots[indices], rcond=1e-12)[0]
                if np.max(np.abs(block @ solution + dots[indices])) > tolerance:
                    continue
                if np.min(solution) < -tolerance:
                    continue
                multipliers[indices] = np.maximum(solution, 0)
            after = dots + gram @ multipliers
            if n and np.min(after) < -tolerance:
                continue
            if subset and np.max(np.abs(after[list(subset)])) > tolerance:
                continue
            cost = float(multipliers @ gram @ multipliers)
            if best is None or cost < best[0] - tolerance * tolerance:
                best = (cost, multipliers, after, list(subset))
    if best is None:
        raise ArithmeticError('No verified cone projection; stop rather than drop a loss')
    cost, multipliers, after, active = best
    return multipliers, {'policy': POLICY, 'active_constraints': active,
                         'active_sets_checked': subsets, 'KKT_tolerance': tolerance,
                         'correction_squared_norm': cost,
                         'unit_gradient_dots_before': dots.tolist(),
                         'unit_gradient_dots_after': after.tolist(),
                         'multipliers': multipliers.tolist()}


def project_vectors(components, proposal):
    gradients = np.asarray(components, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if (gradients.ndim != 2 or not 1 <= gradients.shape[0] <= 7
            or proposal.ndim != 1 or gradients.shape[1] != len(proposal)
            or not np.isfinite(gradients).all() or not np.isfinite(proposal).all()):
        raise ValueError('Finite <=7 gradient rows and matching proposal required')
    norms = np.linalg.norm(gradients, axis=1)
    indices = np.flatnonzero(norms > 0)
    unit = gradients[indices] / norms[indices, None]
    proposal_norm = float(np.linalg.norm(proposal))
    gram, dots = unit @ unit.T, unit @ proposal
    multipliers, receipt = solve_dual(gram, dots, proposal_norm)
    corrected = proposal + multipliers @ unit
    actual_dots = unit @ corrected
    if len(indices) and np.min(actual_dots) < -receipt['KKT_tolerance']:
        raise ArithmeticError('Projection readback violates a loss constraint')
    receipt.update({'nonzero_term_indices': indices.tolist(),
                    'component_norms': norms.tolist(),
                    'proposal_norm': proposal_norm,
                    'projected_norm': float(np.linalg.norm(corrected)),
                    'correction_norm': float(np.linalg.norm(corrected - proposal)),
                    'raw_component_dots_before': (gradients @ proposal).tolist(),
                    'raw_component_dots_after': (gradients @ corrected).tolist(),
                    'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0})
    return corrected, receipt
