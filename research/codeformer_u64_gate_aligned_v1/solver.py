"""Bounded pure NumPy common-descent arithmetic on supplied saved gradients.

Every nonempty simplex face is examined, at most 255 for eight gate vectors.
No model, automatic differentiation, neural optimizer or parameter application
exists in this module. Returned direction arrays are mathematical data only.
"""
import math
import time

import numpy as np

MAX_VECTORS = 8
MAX_COORDINATES = 168960
MAX_FACES = 255
MAX_SECONDS = 10.0
MIN_STRICT_NORMALIZED_MARGIN = 1e-6
WEIGHT_NEGATIVE_TOLERANCE = 1e-10
KKT_RESIDUAL_TOLERANCE = 1e-10
LABEL_ORDER = tuple((role, query) for role in (
    'local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35'
) for query in ('gate_mae', 'gate_adjacent'))


class CommonDescentError(ValueError):
    """No usable certified direction; callers must stop rather than fall back."""


def need(condition, message):
    if not condition:
        raise CommonDescentError(message)


def norm(array):
    value = float(np.sum(array * array, dtype=np.float64))
    need(math.isfinite(value) and value >= 0, 'Finite representable FP64 squared norm required')
    return math.sqrt(value)


def solve_common_descent(gradients, *, max_seconds=MAX_SECONDS):
    """Return a unit direction strictly descending every supplied pooled gate.

    Input rows are one to eight finite, nonzero gradient vectors in a common
    coordinate system. Each row is normalized once. Enumerating simplex faces
    minimizes ||sum_i w_i unit(g_i)||^2 with nonnegative weights summing to one.
    An actual full-vector dot certificate, independent of the Gram prediction,
    must have negative normalized margin greater than 1e-6 for every row.
    """
    began = time.monotonic()
    need(type(max_seconds) in (int, float) and 0 < max_seconds <= MAX_SECONDS,
         'A positive wall cap no greater than10 seconds is required')
    def bounded():
        need(time.monotonic() - began <= max_seconds, 'Common-descent arithmetic exceeded wall cap; no fallback')
    need(isinstance(gradients, np.ndarray) and gradients.ndim == 2 and gradients.dtype.kind == 'f',
         'Floating NumPy matrix required')
    count, coordinates = gradients.shape
    need(1 <= count <= MAX_VECTORS and 0 < coordinates <= MAX_COORDINATES,
         'One to eight vectors with at most168960 common coordinates required')
    need(np.isfinite(gradients).all(), 'Every supplied gradient must be finite')
    values = gradients.astype(np.float64, copy=False)
    lengths = np.array([norm(row) for row in values])
    need(np.all(lengths > 0), 'Zero/underflow gradient cannot certify strict common decrease; no fallback')
    units = values / lengths[:, None]
    need(np.isfinite(units).all(), 'Finite normalized vectors required')
    gram = units @ units.T
    gram = (gram + gram.T) * .5
    need(np.isfinite(gram).all(), 'Finite symmetric Gram required')
    face_count = (1 << count) - 1
    need(face_count <= MAX_FACES, 'At most255 nonempty faces required')
    best, best_squared, accepted, rejected = None, math.inf, 0, 0
    best_face = None
    maximum_accepted_kkt_residual = 0.
    for mask in range(1, face_count + 1):
        bounded()
        ids = [i for i in range(count) if mask & (1 << i)]
        size = len(ids)
        # G_face*w = lambda*1 and sum(w)=1; least squares also handles
        # singular faces. A singular result is accepted only after its residual,
        # nonnegative weights and normalization are independently checked.
        system = np.zeros((size + 1, size + 1), dtype=np.float64)
        system[:size, :size] = gram[np.ix_(ids, ids)]
        system[:size, size] = -1.
        system[size, :size] = 1.
        rhs = np.zeros(size + 1)
        rhs[size] = 1.
        solution, _, _, _ = np.linalg.lstsq(system, rhs, rcond=1e-12)
        residual = norm(system @ solution - rhs)
        candidate = solution[:size]
        if (not np.isfinite(solution).all() or residual > KKT_RESIDUAL_TOLERANCE
                or np.min(candidate) < -WEIGHT_NEGATIVE_TOLERANCE):
            rejected += 1
            continue
        candidate = np.maximum(candidate, 0.)
        total = float(candidate.sum())
        if not math.isfinite(total) or abs(total - 1.) > 1e-9 or total <= 0:
            rejected += 1
            continue
        weights = np.zeros(count)
        weights[ids] = candidate / total
        squared = float(weights @ gram @ weights)
        if not math.isfinite(squared) or squared < -1e-12:
            rejected += 1
            continue
        squared = max(0., squared)
        accepted += 1
        maximum_accepted_kkt_residual = max(maximum_accepted_kkt_residual, residual)
        if squared < best_squared:
            best, best_squared, best_face = weights, squared, ids
    bounded()
    need(best is not None and accepted > 0, 'No admissible simplex face; no fallback')
    combined = np.sum(best[:, None] * units, axis=0, dtype=np.float64)
    combined_norm = norm(combined)
    need(combined_norm > 0, 'Zero minimum-norm combination; no strict direction and no fallback')
    direction = -combined / combined_norm
    unit_dots = np.array([float(np.sum(row * direction, dtype=np.float64)) for row in units])
    raw_dots = np.array([float(np.sum(row * direction, dtype=np.float64)) for row in values])
    gram_dots = -(gram @ best) / combined_norm
    need(np.allclose(unit_dots, gram_dots, rtol=2e-10, atol=2e-12), 'Actual-vector/Gram derivative mismatch')
    margin = -float(np.max(unit_dots))
    need(np.isfinite(direction).all() and math.isfinite(margin)
         and margin > MIN_STRICT_NORMALIZED_MARGIN and np.all(raw_dots < 0),
         'Strict normalized common-descent margin greater than1e-6 absent; no fallback')
    bounded()
    return {'direction': direction, 'simplex_coefficients': best,
        'input_gradient_norms': lengths, 'unit_gram': gram,
        'unit_gradient_dot_direction': unit_dots, 'original_gradient_dot_direction': raw_dots,
        'combination_norm': combined_norm, 'minimum_squared_norm_from_faces': best_squared,
        'minimum_normalized_descent_margin': margin,
        'minimum_required_margin': MIN_STRICT_NORMALIZED_MARGIN,
        'all_actual_derivatives_strict_negative': True, 'vector_arithmetic_only': True,
        'face_count': face_count, 'accepted_faces': accepted, 'rejected_faces': rejected,
        'best_face_indices': best_face, 'maximum_accepted_face_KKT_residual': maximum_accepted_kkt_residual,
        'elapsed_seconds': time.monotonic() - began, 'wall_cap_seconds': max_seconds,
        'parameter_applied': False, 'model_constructed': False, 'optimizer_constructed': False,
        'finite_step_or_PNG_gain_certified': False, 'training_recipe_qualified': False}


def solve_eight_pooled_gates(pooled):
    """Exact four-role/two-query wrapper; never drops a mandatory gate."""
    need(type(pooled) is dict and set(pooled) == {r for r, _ in LABEL_ORDER}, 'Exact four roles required')
    rows, shape = [], None
    for role, query in LABEL_ORDER:
        need(type(pooled[role]) is dict and set(pooled[role]) == {'gate_mae', 'gate_adjacent'}, 'Both gate queries required per role')
        value = pooled[role][query]
        need(isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0 and value.dtype.kind == 'f', 'Floating gate vector required')
        shape = value.shape if shape is None else shape
        need(value.shape == shape, 'Common parameter coordinates required')
        rows.append(value)
    return solve_common_descent(np.stack(rows))
