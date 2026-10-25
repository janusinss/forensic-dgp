"""Pure NumPy arithmetic on supplied saved gradients; no framework or solver.

Objective vectors are already gradients of objective/4. No function applies a
parameter vector, constructs an optimizer or predicts delivered PNG quality.
"""
import math

from . import contract

ROLES = contract.ROLES
QUERIES = contract.QUERIES


def need(ok, message):
    if not ok:
        raise ValueError(message)


def vector(value, *, np):
    need(isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0
         and value.dtype.kind == 'f' and np.isfinite(value).all(), 'Finite nonempty floating vector required')
    return value.astype(np.float64, copy=False)


def comparison(left, right, *, np):
    left, right = vector(left, np=np), vector(right, np=np)
    need(left.shape == right.shape, 'Matching vector coordinates required')
    dot = float(np.dot(left, right))
    ln, rn = math.sqrt(float(np.dot(left, left))), math.sqrt(float(np.dot(right, right)))
    need(all(math.isfinite(value) for value in (dot, ln, rn)), 'Finite FP64 dot/norm arithmetic required')
    cosine = None if ln == 0 or rn == 0 else dot / (ln * rn)
    if cosine is not None:
        need(-1 - 1e-12 <= cosine <= 1 + 1e-12, 'Cauchy-consistent FP64 cosine required')
        cosine = max(-1.0, min(1.0, cosine))
    return {'dot': dot, 'left_norm': ln, 'right_norm': rn, 'cosine': cosine,
            'left_zero': ln == 0, 'right_zero': rn == 0}


def pack_gradients(gradients, *, np):
    """Pack all36 finite FP32 tensors; this function never fills missing values.

    A caller may supply explicit zero tensors for autograd-unused coordinates
    only after recording/authenticating that unused status in query receipts.
    Such provenance is not inferred from a supplied numerical zero array.
    """
    shapes = contract.head_shapes()
    need(type(gradients) is dict and set(gradients) == set(shapes), 'All36 actual head-gradient tensors required')
    parts = []
    for name, shape in shapes.items():
        value = gradients[name]
        need(isinstance(value, np.ndarray) and value.dtype == np.float32
             and value.shape == shape and np.isfinite(value).all(), 'Exact finite FP32 head-gradient schema required: ' + name)
        parts.append(value.astype(np.float64).reshape(-1, order='C'))
    packed = np.concatenate(parts)
    need(packed.shape == (168960,), 'Exact168960 head coordinates required')
    return packed


def hypothetical_fresh_adam(gradient, *, np):
    """First warmup step at empty moments as FP64 vector arithmetic ONLY.

    Bias correction at step1 reduces Adam to g/(abs(g)+eps). Clipping must be
    applied first: its scale still changes the denominator-epsilon effect.
    The returned displacement is mathematical data, never a parameter proposal.
    """
    gradient = vector(gradient, np=np)
    norm = math.sqrt(float(np.dot(gradient, gradient)))
    need(math.isfinite(norm) and (norm > 0 or not np.count_nonzero(gradient)),
         'Finite representable FP64 norm required; underflow is not a measured zero')
    coefficient = min(1.0, 1.0 / (norm + 1e-6))
    clipped = gradient * coefficient
    direction = -0.000025 * clipped / (np.abs(clipped) + 1e-8)
    need(np.isfinite(direction).all(), 'Finite hypothetical arithmetic required')
    return {'direction': direction, 'gradient_norm': norm, 'clip_coefficient': coefficient,
            'LR': 0.000025, 'clip_denominator_epsilon': 1e-6, 'Adam_denominator_epsilon': 1e-8,
            'empty_moments': True, 'vector_arithmetic_only': True,
            'optimizer_constructed': False, 'parameter_applied': False}


def summarize_vectors(vectors, *, np):
    """Small-vector helper for analytical fixtures and full packed gradients."""
    need(type(vectors) is dict and set(vectors) == set(ROLES), 'All four roles required')
    checked = {}
    shape = None
    for role in ROLES:
        need(type(vectors[role]) is dict and set(vectors[role]) == set(QUERIES), 'All three mandatory query vectors required')
        checked[role] = {}
        for query in QUERIES:
            value = vector(vectors[role][query], np=np)
            shape = value.shape if shape is None else shape
            need(value.shape == shape, 'One common parameter coordinate schema required')
            checked[role][query] = value
    objective = {role: checked[role]['objective_div4'] for role in ROLES}
    # These are already divided by4 exactly once by the connected query.
    joint = sum((objective[role] for role in ROLES), np.zeros(shape, np.float64))
    descent = -joint
    adam = hypothetical_fresh_adam(joint, np=np)
    joint_info = comparison(joint, joint, np=np)
    gram = {left: {right: comparison(objective[left], objective[right], np=np) for right in ROLES} for left in ROLES}
    norms = {role: {query: comparison(value, value, np=np)['left_norm'] for query, value in checked[role].items()} for role in ROLES}
    derivatives, contributions = {}, {}
    for query in QUERIES:
        derivatives[query], contributions[query] = {}, {}
        for role in ROLES:
            gradient = checked[role][query]
            derivatives[query][role] = {
                'steepest_descent_first_order_derivative': float(np.dot(gradient, descent)),
                'hypothetical_fresh_Adam_first_order_derivative': float(np.dot(gradient, adam['direction'])),
                'cosine_with_joint_objective': comparison(gradient, joint, np=np)['cosine'],
            }
            contributions[query][role] = {source: {
                'gradient_dot': float(np.dot(gradient, objective[source])),
                'steepest_descent_contribution': -float(np.dot(gradient, objective[source])),
                'cosine': comparison(gradient, objective[source], np=np)['cosine'],
            } for source in ROLES}
    return {'gradient_norms': norms, 'objective_role_pairwise': gram,
        'joint_objective_div4_sum_norm': joint_info['left_norm'], 'joint_objective_zero': joint_info['left_zero'],
        'joint_first_order_derivatives': derivatives, 'role_directional_contributions': contributions,
        'hypothetical_fresh_Adam': {key: value for key, value in adam.items() if key != 'direction'},
        'objective_div4_sum_is_not_divided_again': True,
        'negative_derivative_means_local_predicted_decrease_only': True,
        'finite_step_or_PNG_prediction': False}


def analyze(case_rows, selection, *, np):
    """Validate all64 complete cases, then report same-photo and pooled geometry.

    Caller must independently bind arrays to actual saved query/graph receipts.
    This arithmetic cannot authenticate that an autograd call occurred.
    """
    contract.validate_selection(selection)
    expected = selection['optimization']
    need(type(case_rows) is list and len(case_rows) == 64 and all(type(row) is dict for row in case_rows),
         'Complete ordered64-case saved-gradient bank required')
    packed = {}
    for actual, reference in zip(case_rows, expected):
        need(all(actual.get(key) == reference[key] for key in ('id', 'reference_id', 'role'))
             and actual.get('metric_region') == contract.metric_region(reference['role']),
             'Actual case/role/region differs from admitted photograph-major order')
        gradients = actual['gradients']
        need(type(gradients) is dict and set(gradients) == set(QUERIES), 'Exact objective_div4/MAE/adjacent query set required')
        packed[actual['id']] = {query: pack_gradients(gradients[query], np=np) for query in QUERIES}
    order = selection['optimization_reference_order']
    by = {(row['reference_id'], row['role']): row['id'] for row in expected}
    pooled = {role: {query: np.zeros(168960, np.float64) for query in QUERIES} for role in ROLES}
    photo, original_sampler = [], []
    individual_norm_sum = {role: dict.fromkeys(QUERIES, 0.0) for role in ROLES}
    for reference in order:
        bank = {role: packed[by[(reference, role)]] for role in ROLES}
        summary = summarize_vectors(bank, np=np)
        photo.append({'reference_id': reference, 'case_ids': {role: by[(reference, role)] for role in ROLES}, **summary})
        for role in ROLES:
            for query in QUERIES:
                pooled[role][query] += bank[role][query] / 16.0
                individual_norm_sum[role][query] += summary['gradient_norms'][role][query] / 16.0
    for index in range(16):
        refs = {role: order[(index + offset) % 16] for role, offset in zip(ROLES, (0, 4, 8, 12))}
        bank = {role: packed[by[(refs[role], role)]] for role in ROLES}
        original_sampler.append({'old_schedule_update': index + 1, 'reference_ids': refs,
            'all_vectors_at_same_unchanged_initializer': True, **summarize_vectors(bank, np=np)})
    pooled_summary = summarize_vectors(pooled, np=np)
    cancellation = {role: {query: {
        'mean_individual_norm': individual_norm_sum[role][query],
        'pooled_mean_norm': pooled_summary['gradient_norms'][role][query],
        'pooled_over_mean_individual_norm': (None if individual_norm_sum[role][query] == 0 else
            pooled_summary['gradient_norms'][role][query] / individual_norm_sum[role][query]),
    } for query in QUERIES} for role in ROLES}
    return {'complete': True, 'saved_gradient_arithmetic_pass': True,
        'cases': 64, 'photographs': 16, 'head_tensors': 36, 'head_coordinates': 168960,
        'role_order': list(ROLES), 'query_order': list(QUERIES),
        'objective_gradient_weighting': 'Input objective_div4 vectors summed once; pooled role vectors are mean16',
        'metric_regions': {role: contract.metric_region(role) for role in ROLES},
        'per_photograph': photo, 'pooled_role_mean': pooled_summary,
        'pooled_cancellation_diagnostics': cancellation,
        'original_offset_sampler_at_common_state': original_sampler,
        'zero_gradient_policy': 'Supplied measured zeros retained; undefined cosine is null. Caller authenticates explicitly unused coordinates and records their zeros; analysis never fills missing/disconnected/None vectors.',
        'caller_must_bind_actual_saved_query_bytes_and_state': True,
        'actual_autograd_or_state_integrity_proved_here': False,
        'hypothetical_directions_only': True, 'Adam_trajectory_or_PNG_quality_predicted': False,
        'optimizer_constructions': 0, 'parameter_applications': 0,
        'common_descent_or_feasibility_certificate': False, 'new_recipe_selected': False, 'model_qualified': False}
