"""Pure NumPy geometry of supplied RAW/biased-PNG-surrogate gradient arrays.

No framework, solver, parameter direction, update or finite-step prediction.
The caller authenticates each saved query and common unchanged state first.
"""
import math

from . import contract

ROLES, QUERIES = contract.ROLES, contract.QUERIES
LABELS = tuple((role, query) for role in ROLES for query in QUERIES)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def vector(value, *, np):
    need(isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0
         and value.dtype.kind == 'f' and np.isfinite(value).all(), 'Finite nonempty floating vector required')
    return value.astype(np.float64, copy=False)


def comparison(left, right, *, np):
    a, b = vector(left, np=np), vector(right, np=np)
    need(a.shape == b.shape, 'Common vector coordinates required')
    dot = float(np.sum(a * b, dtype=np.float64))
    an = math.sqrt(float(np.sum(a * a, dtype=np.float64)))
    bn = math.sqrt(float(np.sum(b * b, dtype=np.float64)))
    need(all(math.isfinite(v) for v in (dot, an, bn)), 'Finite representable FP64 dot/norm required')
    return {'dot': dot, 'left_norm': an, 'right_norm': bn,
        'cosine': None if an == 0 or bn == 0 else max(-1., min(1., dot / (an * bn))),
        'left_zero': an == 0, 'right_zero': bn == 0}


def pack_gradients(gradients, *, np):
    shapes = contract.head_shapes()
    need(type(gradients) is dict and set(gradients) == set(shapes), 'All36 actually supplied head tensors required')
    parts = []
    for name, shape in shapes.items():
        value = gradients[name]
        need(isinstance(value, np.ndarray) and value.dtype == np.float32
             and value.shape == shape and np.isfinite(value).all(), 'Exact finite FP32 head schema required: ' + name)
        parts.append(value.reshape(-1, order='C').astype(np.float64))
    result = np.concatenate(parts)
    need(result.shape == (168960,), 'Exact168960 coordinates required')
    return result


def summarize_vectors(bank, *, np):
    need(type(bank) is dict and set(bank) == set(ROLES), 'All four roles required')
    checked, shape = {}, None
    for role in ROLES:
        need(type(bank[role]) is dict and set(bank[role]) == set(QUERIES), 'All four mandatory domain/query vectors required')
        checked[role] = {}
        for query in QUERIES:
            value = vector(bank[role][query], np=np)
            shape = value.shape if shape is None else shape
            need(value.shape == shape, 'Common coordinate schema for all16 vectors required')
            checked[role][query] = value
    # One16x16 Gram avoids repeatedly reducing the same168960 coordinates.
    vectors = np.stack([checked[r][q] for r, q in LABELS])
    gram = vectors @ vectors.T
    need(np.isfinite(gram).all(), 'Finite16-vector FP64 Gram required')
    indices = {label: i for i, label in enumerate(LABELS)}
    norms = [math.sqrt(max(0., float(gram[i, i]))) for i in range(16)]
    def pair(a, b):
        i, j = indices[a], indices[b]
        an, bn, dot = norms[i], norms[j], float(gram[i, j])
        return {'dot': dot, 'left_norm': an, 'right_norm': bn,
            'cosine': None if an == 0 or bn == 0 else max(-1., min(1., dot / (an * bn))),
            'left_zero': an == 0, 'right_zero': bn == 0}
    matrix = [[pair(a, b) for b in LABELS] for a in LABELS]
    return {'vector_count': 16, 'coordinate_count': shape[0],
        'label_order': [{'role': r, 'query': q} for r, q in LABELS],
        'gradient_norms': {r: {q: pair((r, q), (r, q))['left_norm'] for q in QUERIES} for r in ROLES},
        'pairwise16': matrix,
        'RAW_vs_PNG_surrogate_by_role': {r: {
            'mae': pair((r, 'raw_mae'), (r, 'png_surrogate_mae')),
            'adjacent': pair((r, 'raw_adjacent'), (r, 'png_surrogate_adjacent'))} for r in ROLES},
        'measured_zero_vectors': [{'role': r, 'query': q} for r, q in LABELS
                                  if not np.any(checked[r][q])],
        'surrogate_is_actual_PNG_derivative': False, 'parameter_directions_computed': False,
        'common_direction_solver_used': False, 'finite_step_prediction': False}


def analyze(case_rows, selection, *, np):
    contract.validate_selection(selection)
    expected = selection['optimization']
    need(type(case_rows) is list and len(case_rows) == 64, 'Complete ordered64-rolecase bank required')
    pooled = {r: {q: np.zeros(168960, dtype=np.float64) for q in QUERIES} for r in ROLES}
    norm_sums = {r: dict.fromkeys(QUERIES, 0.) for r in ROLES}
    packed, photos = {}, []
    for actual, reference in zip(case_rows, expected):
        need(type(actual) is dict and all(actual.get(k) == reference[k] for k in ('id', 'reference_id', 'role'))
             and actual.get('metric_region') == contract.metric_region(reference['role']),
             'Actual photograph/role/order/region differs from admitted TRAIN selection')
        need(type(actual.get('gradients')) is dict and set(actual['gradients']) == set(QUERIES),
             'Four mandatory actual RAW/PNG-surrogate query banks required')
        packed[actual['id']] = {q: pack_gradients(actual['gradients'][q], np=np) for q in QUERIES}
        for query in QUERIES:
            value = packed[actual['id']][query]
            pooled[actual['role']][query] += value / 16.
            norm_sums[actual['role']][query] += comparison(value, value, np=np)['left_norm'] / 16.
    lookup = {(r['reference_id'], r['role']): r['id'] for r in expected}
    for reference in selection['optimization_reference_order']:
        bank = {r: packed[lookup[reference, r]] for r in ROLES}
        photos.append({'reference_id': reference, 'case_ids': {r: lookup[reference, r] for r in ROLES},
                       **summarize_vectors(bank, np=np)})
    pooled_summary = summarize_vectors(pooled, np=np)
    cancellation = {r: {q: {'mean_individual_norm': norm_sums[r][q],
        'pooled_mean_norm': pooled_summary['gradient_norms'][r][q],
        'pooled_over_mean_individual_norm': None if norm_sums[r][q] == 0 else
            pooled_summary['gradient_norms'][r][q] / norm_sums[r][q]} for q in QUERIES} for r in ROLES}
    return {'complete': True, 'saved_array_arithmetic_pass': True, 'cases': 64,
        'photographs': 16, 'mandatory_query_vectors': 256, 'pooled_vectors': 16,
        'head_tensors': 36, 'head_coordinates': 168960,
        'query_order': list(QUERIES), 'per_photograph': photos,
        'pooled_role_mean16': pooled_summary, 'pooled_cancellation_diagnostics': cancellation,
        'zero_policy': 'Supplied measured zeros retained; undefined cosine is null; no missing or disconnected vector is inferred/imputed.',
        'pooling': 'Mean16 same-state photographs per role/query, normalization only for pairwise cosine after pooling.',
        'caller_must_bind_actual_saved_queries_and_unchanged_state': True,
        'actual_autograd_or_state_integrity_proved_here': False,
        'PNG_surrogate_is_biased_not_actual_discrete_derivative': True,
        'common_direction_solver_used': False, 'parameter_directions_computed': False,
        'parameter_applications': 0, 'optimizer_constructions': 0,
        'finite_step_or_PNG_gain_prediction': False, 'new_training_recipe_selected': False,
        'model_qualified': False}
