"""NEW offline16-gate feasibility arithmetic from authenticated saved gradients.

No neural framework, new autograd, model, optimizer, parameter application or
VM call. A common local direction is hypothetical geometry, never a finite PNG
certificate or a chosen training recipe. All16 role/domain/gate rows retained.
"""
import hashlib
import importlib.abc
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/feasibility_v1.json'
AUDIT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_gradients/receipt_v1.json'
INVENTORY = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/inventory_v1.json'
PROTOCOL = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm/protocol.json'
PROTOCOL_SHA = '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
QUERIES = ('raw_mae', 'raw_adjacent', 'png_surrogate_mae', 'png_surrogate_adjacent')
LABELS = tuple((role, query) for role in ROLES for query in QUERIES)
MAX_FACES, MAX_SECONDS = 65535, 120.


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        need(fullname.split('.')[0] not in {'torch', 'tensorflow', 'jax'}, 'Framework import forbidden')
        return None


def inner(a, b):
    return float(np.sum(a * b, dtype=np.float64))


def norm(a):
    return math.sqrt(inner(a, a))


def exhaustive_minimum_norm(values, *, deadline_seconds=MAX_SECONDS):
    """Enumerate every nonempty simplex face; no mandatory vector dropped."""
    began = time.monotonic()
    need(isinstance(values, np.ndarray) and values.ndim == 2 and values.dtype == np.float64
         and 1 <= values.shape[0] <= 16 and 0 < values.shape[1] <= 168960
         and np.isfinite(values).all(), 'One to16 finite FP64 vectors in common headspace required')
    need(0 < deadline_seconds <= MAX_SECONDS, 'Explicit wall cap at most120s required')
    lengths = np.array([norm(v) for v in values])
    need(np.all(lengths > 0), 'Zero mandatory vector: stop, do not drop endpoint')
    units = values / lengths[:, None]
    gram = units @ units.T
    gram = .5 * (gram + gram.T)
    count, coordinates = values.shape
    faces = (1 << count) - 1
    need(faces <= MAX_FACES, '65535-face cap exceeded')
    accepted, rejected, best_norm, best_weights, best_ids = 0, 0, math.inf, None, None
    maximum_residual = 0.
    for mask in range(1, faces + 1):
        need(time.monotonic() - began <= deadline_seconds, '120s exhaustive solve deadline reached; no partial certificate')
        ids = [i for i in range(count) if mask & (1 << i)]
        size = len(ids)
        # Equality-constrained face optimum; singular systems are accepted only
        # if their complete KKT residual and nonnegative weight checks pass.
        matrix = np.zeros((size + 1, size + 1))
        matrix[:size, :size] = gram[np.ix_(ids, ids)]
        matrix[:size, size] = 1.
        matrix[size, :size] = 1.
        rhs = np.zeros(size + 1); rhs[-1] = 1.
        solved = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
        residual = norm(matrix @ solved - rhs)
        weights = solved[:size]
        if not np.isfinite(solved).all() or residual > 1e-10 or weights.min() < -1e-10:
            rejected += 1
            continue
        weights = np.maximum(weights, 0.)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0 or abs(total - 1.) > 1e-9:
            rejected += 1
            continue
        full = np.zeros(count)
        full[ids] = weights / total
        squared = float(full @ gram @ full)
        if not math.isfinite(squared) or squared < -1e-12:
            rejected += 1
            continue
        accepted += 1
        maximum_residual = max(maximum_residual, residual)
        squared = max(0., squared)
        if squared < best_norm:
            best_norm, best_weights, best_ids = squared, full, ids
    need(best_weights is not None, 'No admissible face; no common-descent claim')
    combined = np.sum(best_weights[:, None] * units, axis=0, dtype=np.float64)
    combined_norm = norm(combined)
    # Independent KKT certificate uses actual full-coordinate products rather
    # than the Gram/lstsq residual that selected the candidate.
    products = np.array([inner(row, combined) for row in units])
    squared_actual = inner(combined, combined)
    active = best_weights > 1e-8
    active_error = float(np.max(np.abs(products[active] - squared_actual)))
    lower_violation = float(max(0., squared_actual - products.min()))
    simplex_error = abs(float(best_weights.sum()) - 1.)
    kkt_pass = active_error <= 1e-8 and lower_violation <= 1e-8 and simplex_error <= 1e-10 and best_weights.min() >= -1e-12
    need(kkt_pass, 'Independent actual-coordinate KKT certificate failed')
    direction = None if combined_norm < 1e-12 else -combined / combined_norm
    original_dots = None if direction is None else np.array([inner(row, direction) for row in values])
    normalized_dots = None if direction is None else np.array([inner(row, direction) for row in units])
    margin = None if normalized_dots is None else -float(normalized_dots.max())
    strict = direction is not None and margin > 1e-6 and bool(np.all(original_dots < 0))
    if direction is not None:
        need(np.allclose(normalized_dots, -(gram @ best_weights) / combined_norm, rtol=3e-9, atol=3e-12), 'Gram/full-coordinate directional dot mismatch')
    need(time.monotonic() - began <= deadline_seconds, '120s solve deadline exceeded')
    return {'direction': direction, 'units': units, 'gradient_norms': lengths,
        'unit_gram': gram, 'simplex_coefficients': best_weights,
        'minimum_norm_combination_norm': combined_norm,
        'minimum_squared_norm_from_faces': best_norm,
        'minimum_actual_squared_norm': squared_actual,
        'actual_original_gradient_dot_direction': original_dots,
        'actual_unit_gradient_dot_direction': normalized_dots,
        'minimum_normalized_descent_margin': margin,
        'all_mandatory_gates_strict_common_descent': strict,
        'strict_margin_threshold': 1e-6, 'best_face_indices': best_ids,
        'faces_enumerated': faces, 'accepted_faces': accepted, 'rejected_faces': rejected,
        'maximum_accepted_face_KKT_residual': maximum_residual,
        'independent_actual_KKT': {'pass': kkt_pass, 'active_threshold': 1e-8,
            'active_equal_dot_max_error': active_error, 'all_dot_lower_bound_violation': lower_violation,
            'simplex_sum_error': simplex_error, 'tolerance': 1e-8},
        'deadline_seconds': deadline_seconds, 'elapsed_seconds': time.monotonic() - began}


def fixtures():
    # Identical rows require a known unique vector direction despite nonunique
    # simplex weights; opposed rows place zero inside the hull.
    identical = exhaustive_minimum_norm(np.tile(np.array([[3., 4.]]), (4, 1)))
    need(np.allclose(identical['direction'], [-.6, -.8], atol=1e-12)
         and identical['all_mandatory_gates_strict_common_descent'], 'Identical-vector fixture failed')
    opposed = exhaustive_minimum_norm(np.array([[1., 0.], [-1., 0.]]))
    need(opposed['minimum_norm_combination_norm'] < 1e-12
         and not opposed['all_mandatory_gates_strict_common_descent'], 'Opposed-vector infeasibility fixture failed')
    # Sixteen differently scaled orthogonal coordinates normalize to identity:
    # optimum weights1/16, combinednorm1/4, normalized direction each-1/4.
    known = exhaustive_minimum_norm(np.diag(np.arange(1., 17.)))
    need(known['faces_enumerated'] == 65535 and np.allclose(known['simplex_coefficients'], np.full(16, 1/16), atol=1e-12)
         and np.allclose(known['direction'], np.full(16, -.25), atol=1e-12)
         and abs(known['minimum_norm_combination_norm'] - .25) < 1e-12, 'Nontrivial16-vector analytical solution failed')
    try:
        exhaustive_minimum_norm(np.array([[1., 0.], [0., 0.]]))
        raise AssertionError('Zero endpoint was silently dropped')
    except ValueError as error:
        need('Zero mandatory' in str(error), 'Expected mandatory-zero rejection')
    return {'pass': True, 'identical_known_direction': [-.6, -.8],
        'opposed_zero_hull_retained_not_strict': True,
        'nontrivial16_orthogonal_scaled_solution': {'weights_each': 1/16, 'direction_each': -.25,
            'combination_norm': .25, 'faces': 65535, 'elapsed_seconds': known['elapsed_seconds']},
        'mandatory_zero_rejected_no_drop': True}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main():
    sys.meta_path.insert(0, NoFramework())
    consumed = {}
    def read(path, pin=None):
        raw = path.read_bytes()
        actual = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        if pin is not None:
            need(actual == {k: pin[k] for k in actual}, 'Exact consumed byte pin differs: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        need(key not in consumed or consumed[key] == actual, 'Source changed during read')
        consumed[key] = actual
        return raw
    protocol_raw = read(PROTOCOL)
    need(hashlib.sha256(protocol_raw).hexdigest() == PROTOCOL_SHA, 'Exact zero-update protocol required')
    protocol = json.loads(protocol_raw)
    receipt = json.loads(read(AUDIT))
    need(receipt['complete'] and receipt['saved_state_gradient_geometry_audit_pass']
         and receipt['protocol_sha256'] == PROTOCOL_SHA and receipt['gradient_NPZ_files'] == 256, 'Completed independent256-gradient audit required')
    inventory = json.loads(read(INVENTORY))
    evidence = Path(inventory['evidence_root'])
    lookup = {(r['id'], r['query']): r for r in receipt['gradient_receipts']}
    pooled = {r: {q: np.zeros(168960) for q in QUERIES} for r in ROLES}
    cases = {}
    for row in protocol['selection']['optimization']:
        cases[row['id']] = {}
        for query in QUERIES:
            saved = lookup[row['id'], query]
            raw = read(evidence / saved['file']['path'], saved['file'])
            with np.load(io.BytesIO(raw), allow_pickle=False) as decoded:
                vector = np.concatenate([decoded[name].reshape(-1).astype(np.float64) for name in sorted(decoded.files)])
            need(vector.shape == (168960,) and hashlib.sha256(vector.tobytes()).hexdigest() == saved['packed_FP64_sha256'], 'Independently audited actual vector changed')
            cases[row['id']][query] = vector
            pooled[row['role']][query] += vector / 16.
    matrix = np.stack([pooled[r][q] for r, q in LABELS])
    fixture_result = fixtures()
    solved = exhaustive_minimum_norm(matrix)
    need(solved['all_mandatory_gates_strict_common_descent'], 'No strict16-gate common direction at measured pooled state')
    direction = solved['direction']
    # Compare every pooled Gram element and same-state normalized magnitude with
    # the prior independent measurement receipt before interpreting feasibility.
    for i, row in enumerate(receipt['pooled_role_mean16']['pairwise16']):
        for j, pair in enumerate(row):
            expected_cos = pair['cosine']
            need(expected_cos is not None and abs(float(solved['unit_gram'][i, j]) - expected_cos) < 2e-10,
                 'Actual16 pooled geometry differs from prior independent audit')
    raw_ids = [i for i, (_, q) in enumerate(LABELS) if q.startswith('raw_')]
    raw_solved = exhaustive_minimum_norm(matrix[raw_ids])
    raw_direction = raw_solved['direction']
    need(raw_direction is not None, 'Raw-only comparison has no direction')
    raw_all_dots = np.array([inner(v, raw_direction) for v in matrix])
    raw_all_unit = np.array([inner(v, raw_direction) for v in solved['units']])
    # Authenticate the exact historical first-step direction from the rejected
    # pilot separately; this is a historical saved-array comparator only.
    prior_inventory = json.loads(read(ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/inventory_v1.json'))
    prior_audit = json.loads(read(ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_updates/receipt_v1.json'))
    old_relative = 'outputs/u64_gate_aligned_pilot_v1/updates/u0001/direction.npy'
    old_path = Path(prior_inventory['evidence_root']) / old_relative
    expected_old = prior_inventory['extracted_files'][old_relative]
    need(prior_audit['saved_state_gradient_update_audit_pass'] and prior_audit['consumed_files'][old_path.relative_to(ROOT).as_posix()] == {k: expected_old[k] for k in ('bytes', 'sha256')}, 'Historical direction not independently bound')
    old_direction = np.load(io.BytesIO(read(old_path, expected_old)), allow_pickle=False)
    need(old_direction.dtype == np.float64 and old_direction.shape == (168960,), 'Exact historical RAW direction schema required')
    old_dots = np.array([inner(v, old_direction) for v in matrix])
    old_unit = np.array([inner(v, old_direction) for v in solved['units']])
    case_results = []
    for row in protocol['selection']['optimization']:
        case_results.append({k: row[k] for k in ('id', 'reference_id', 'role')} | {
            'four_query_gradient_dots_hypothetical16_direction': {q: inner(cases[row['id']][q], direction) for q in QUERIES}})
    counts = {r: {q: {'decrease': sum(c['four_query_gradient_dots_hypothetical16_direction'][q] < 0 for c in case_results if c['role'] == r),
        'zero': sum(c['four_query_gradient_dots_hypothetical16_direction'][q] == 0 for c in case_results if c['role'] == r),
        'increase': sum(c['four_query_gradient_dots_hypothetical16_direction'][q] > 0 for c in case_results if c['role'] == r)}
        for q in QUERIES} for r in ROLES}
    read(Path(__file__).resolve())
    for name, pin in consumed.items():
        raw = (ROOT / name).read_bytes()
        need({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == pin, 'Consumed source changed before/after: ' + name)
    need(not any(n.split('.')[0] in {'torch', 'tensorflow', 'jax'} for n in sys.modules), 'Framework unexpectedly loaded')
    no_arrays = {k: v for k, v in solved.items() if k not in ('direction', 'units')}
    result = {'complete': True, 'status': 'NEW_OFFLINE_FULL16_SAVED_GEOMETRY_FEASIBILITY_NOT_A_TRAINING_RECIPE',
        'outside_returned_measurement_protocol': True, 'protocol_sha256': PROTOCOL_SHA,
        'all16_mandatory_role_domain_gate_vectors_retained': True,
        'pooling': 'Mean16 actual saved photographs per role/query, normalized only after pooling.',
        'label_order': [{'role': r, 'query': q} for r, q in LABELS],
        'actual_gradient_files_reconstructed': 256, 'head_coordinates': 168960,
        'solver': no_arrays, 'synthetic_checks': fixture_result,
        'comparison_new8RAW_only_direction': {'RAW_label_indices': raw_ids,
            'all16_original_gradient_dots': raw_all_dots, 'all16_unit_gradient_dots': raw_all_unit,
            'all16_negative': bool(np.all(raw_all_dots < 0)),
            'minimum16_normalized_margin': -float(raw_all_unit.max()),
            'cosine_with_full16_direction': inner(raw_direction, direction),
            'L2_difference_from_full16_direction': norm(raw_direction - direction)},
        'comparison_authenticated_historical8RAW_u1_direction': {'source': expected_old,
            'all16_original_gradient_dots': old_dots, 'all16_unit_gradient_dots': old_unit,
            'all16_negative': bool(np.all(old_dots < 0)),
            'minimum16_normalized_margin': -float(old_unit.max()),
            'cosine_with_full16_direction': inner(old_direction, direction),
            'L2_difference_from_new8RAW_direction': norm(old_direction - raw_direction)},
        'per_role_query_photograph_signs_under_hypothetical16_direction': counts,
        'all64_rolecase_four_query_dots': case_results,
        'consumed_files': consumed, 'all_consumed_before_after_exact': True,
        'direction_or_parameter_NPZ_written': False, 'new_model_or_optimizer_or_actual_gradient_or_parameter_VM_calls': 0,
        'limitations': ['All16 pooled gradients descend locally; individual cases, adaptive monitors and guards are separate, unproved constraints.',
            'PNG vectors are biased identity-backward surrogates, not true discrete PNG derivatives.',
            'Even all16 negative derivatives cannot certify a finite actual PNG change or useful visible restoration.',
            'The rejected prior RAW pilot remains rejected, including earlieru1; this comparison does not repair its PNG gate.',
            'Weights and directions describe one saved initializer geometry only, not a selected training recipe or a manual execution packet.'],
        'new_training_recipe_selected': False, 'finite_step_or_actual_PNG_gain_certified': False,
        'quality_gates_or_quantizer_changed': False, 'prior_failure_overruled': False, 'model_qualified': False}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(jsonable(result), stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(jsonable({'complete': True, 'all16_common_descent': solved['all_mandatory_gates_strict_common_descent'],
        'margin': solved['minimum_normalized_descent_margin'], 'faces': solved['faces_enumerated'],
        'solver_seconds': solved['elapsed_seconds'], 'independent_KKT': solved['independent_actual_KKT'],
        'simplex_coefficients': solved['simplex_coefficients'],
        'RAW8_comparison': result['comparison_new8RAW_only_direction'],
        'historicalRAW8_comparison': result['comparison_authenticated_historical8RAW_u1_direction'],
        'percase_signs': counts}), indent=2))


if __name__ == '__main__':
    main()
