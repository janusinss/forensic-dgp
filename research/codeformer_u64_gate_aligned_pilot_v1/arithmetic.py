"""Pure NumPy FP32 proposal arithmetic; never mutates a model or input array.

The single 0.99 factor provides fixed numerical headroom before FP32 rounding.
It is independent of metrics and has no retry, line search or fallback path.
"""
import math

import numpy as np

from research.codeformer_u64_role_gradients_v1.contract import head_shapes

MAX_STEP = 2.5e-5
MAX_CUMULATIVE = 1e-4
NUMERICAL_HEADROOM = .99
MAX_COORDINATES = 168960
MIN_DIRECTION_MARGIN = 1e-6


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _norm(value):
    squared = float(np.sum(value * value, dtype=np.float64))
    need(math.isfinite(squared) and squared > 0, 'Finite nonzero representable FP64 norm required')
    return math.sqrt(squared)


def propose_fp32_vector_step(current, original, direction, pooled_gradients):
    """Synthetic/reduced-vector helper; supplied arrays only, at most168960D.

    Returns a new FP32 candidate, actual rounded FP64 displacement and a JSON
    certificate. Rejecting any bound/sign guard ends this call without retry.
    No candidate returned by this function has been applied to any model.
    """
    need(isinstance(current, np.ndarray) and current.ndim == 1
         and current.dtype == np.float32 and 0 < current.size <= MAX_COORDINATES,
         'Nonempty FP32 current vector with at most168960 coordinates required')
    need(isinstance(original, np.ndarray) and original.shape == current.shape
         and original.dtype == np.float32, 'Matching original FP32 vector required')
    need(isinstance(direction, np.ndarray) and direction.shape == current.shape
         and direction.dtype == np.float64, 'Matching exact FP64 direction vector required')
    need(isinstance(pooled_gradients, np.ndarray)
         and pooled_gradients.shape == (8, current.size) and pooled_gradients.dtype == np.float64,
         'All eight exact FP64 pooled gate gradients required')
    need(all(np.isfinite(value).all() for value in (current, original, direction, pooled_gradients)),
         'Every input coordinate must be finite')
    current_before, original_before = current.tobytes(), original.tobytes()
    direction_before, gradients_before = direction.tobytes(), pooled_gradients.tobytes()
    now64, start64 = current.astype(np.float64), original.astype(np.float64)
    need(float(np.max(np.abs(now64 - start64))) <= MAX_CUMULATIVE,
         'Current state already exceeds original-u64 cumulative coordinate bound')
    direction_norm = _norm(direction)
    gradient_norms = np.array([_norm(row) for row in pooled_gradients])
    unrounded_dots = np.array([float(np.sum(row * direction, dtype=np.float64)) for row in pooled_gradients])
    unrounded_cosines = unrounded_dots / (gradient_norms * direction_norm)
    need(np.isfinite(unrounded_cosines).all()
         and -float(np.max(unrounded_cosines)) > MIN_DIRECTION_MARGIN,
         'Supplied direction lacks strict eight-gate normalized margin greater than1e-6')
    peak = float(np.max(np.abs(direction)))
    scalar = NUMERICAL_HEADROOM * MAX_STEP / peak
    need(math.isfinite(scalar) and scalar > 0, 'Finite strictly positive fixed step scalar required')
    # One FP64 addition followed by one explicit FP32 rounding. Candidate data
    # remains detached from both input arrays and any neural model.
    candidate = (now64 + scalar * direction).astype(np.float32)
    need(np.isfinite(candidate).all(), 'FP32 candidate must remain finite')
    rounded64 = candidate.astype(np.float64)
    actual_delta = rounded64 - now64
    cumulative_delta = rounded64 - start64
    actual_max = float(np.max(np.abs(actual_delta)))
    cumulative_max = float(np.max(np.abs(cumulative_delta)))
    need(actual_max <= MAX_STEP, 'Actual FP32 displacement exceeds2.5e-5 per-coordinate bound; stop without retry')
    need(cumulative_max <= MAX_CUMULATIVE, 'Actual FP32 candidate exceeds1e-4 cumulative coordinate bound; stop without retry')
    actual_dots = np.array([float(np.sum(row * actual_delta, dtype=np.float64)) for row in pooled_gradients])
    need(np.isfinite(actual_dots).all() and np.all(actual_dots < 0),
         'Rounded actual displacement lacks strict descent for every gate; stop without retry')
    delta_norm = _norm(actual_delta)
    actual_cosines = actual_dots / (gradient_norms * delta_norm)
    need(np.isfinite(actual_cosines).all(), 'Finite actual rounded displacement cosines required')
    need(current.tobytes() == current_before and original.tobytes() == original_before
         and direction.tobytes() == direction_before and pooled_gradients.tobytes() == gradients_before,
         'Arithmetic input array changed during proposal')
    need(not np.shares_memory(candidate, current) and not np.shares_memory(candidate, original),
         'Candidate must own detached array data')
    return {'candidate': candidate, 'actual_delta': actual_delta,
        'certificate': {'pass': True, 'arithmetic_only': True, 'coordinates': current.size,
            'fixed_numerical_headroom': NUMERICAL_HEADROOM,
            'step_scalar': scalar, 'step_scalar_formula': '0.99*2.5e-5/maxabs(direction)',
            'scalar_chosen_before_metrics': True, 'step_attempts': 1,
            'actual_max_absolute_coordinate_delta': actual_max,
            'actual_max_cumulative_coordinate_delta': cumulative_max,
            'maximum_absolute_coordinate_step': MAX_STEP,
            'maximum_cumulative_absolute_coordinate_delta': MAX_CUMULATIVE,
            'unrounded_direction_gradient_dots': unrounded_dots.tolist(),
            'unrounded_minimum_normalized_descent_margin': -float(np.max(unrounded_cosines)),
            'actual_FP32_displacement_gradient_dots': actual_dots.tolist(),
            'actual_FP32_minimum_normalized_descent_margin': -float(np.max(actual_cosines)),
            'all8_actual_displacement_dots_strict_negative': True,
            'inputs_byte_exact_after': True, 'candidate_detached_copy': True,
            'parameter_applied': False, 'optimizer_constructed': False,
            'metric_line_search_or_adaptive_retry': False, 'finite_step_or_PNG_gain_proved': False}}


def propose_fp32_step(current, original, direction, pooled_gradients):
    """Exact36-head adapter in sorted canonical168960-coordinate order.

    current/original must be plain dictionaries containing the exact retained
    head names/shapes and finite FP32 arrays. Only candidate copies are returned.
    """
    shapes = head_shapes()
    need(type(current) is dict and type(original) is dict
         and set(current) == set(original) == set(shapes), 'Exact36-head namespace required')
    before = {}
    for label, state in (('current', current), ('original', original)):
        for name, shape in shapes.items():
            value = state[name]
            need(isinstance(value, np.ndarray) and value.dtype == np.float32
                 and value.shape == shape and np.isfinite(value).all(),
                 'Exact finite FP32 head schema required: ' + label + '/' + name)
            before[label, name] = value.tobytes(order='C')
    now = np.concatenate([current[name].reshape(-1, order='C') for name in shapes])
    initial = np.concatenate([original[name].reshape(-1, order='C') for name in shapes])
    need(now.size == initial.size == MAX_COORDINATES, 'Exact168960 head coordinates required')
    proposal = propose_fp32_vector_step(now, initial, direction, pooled_gradients)
    candidate, offset = {}, 0
    for name, shape in shapes.items():
        count = math.prod(shape)
        candidate[name] = proposal['candidate'][offset:offset + count].reshape(shape).copy()
        offset += count
    for label, state in (('current', current), ('original', original)):
        need(all(state[name].tobytes(order='C') == before[label, name] for name in shapes),
             'Input head state changed during arithmetic')
    proposal['candidate'] = candidate
    proposal['certificate'].update({'head_tensors': 36, 'head_coordinate_order': list(shapes),
                                    'actual_model_state_inspected_or_modified': False})
    return proposal
