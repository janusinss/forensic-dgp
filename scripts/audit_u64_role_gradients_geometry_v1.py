"""Audit saved FP32 gradient data with NumPy only; never enter a model/optimizer.

The independent arithmetic below is compared with the sealed inert analysis
module only after its exact source pins are verified. No returned direction is
written into a model or a parameter file.
"""
import argparse
import hashlib
import importlib.abc
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm/protocol.json'
INVENTORY = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit/inventory_v1.json'
OUT = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_geometry'
PROTOCOL_SHA = '07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
QUERIES = ('objective_div4', 'gate_mae', 'gate_adjacent')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        obj = {}
        for key, val in items:
            need(key not in obj, 'Duplicate JSON key: ' + key)
            obj[key] = val
        return obj
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject)


def expected_shapes():
    result = {}
    for site, channels in (('32', 256), ('64', 256), ('128', 128), ('256', 128)):
        for part, shape in (
            ('encoder.weight', (32, channels, 1, 1)), ('encoder.bias', (32,)),
            ('decoder.weight', (32, channels, 1, 1)), ('fused.weight', (32, channels, 1, 1)),
            ('coarse.weight', (32, 256, 1, 1)), ('spatial.weight', (32, 32, 3, 3)),
            ('spatial.bias', (32,)), ('output.weight', (channels, 32, 1, 1)),
            ('output.bias', (channels,)),
        ):
            result['heads.' + site + '.' + part] = shape
    result = dict(sorted(result.items()))
    need(len(result) == 36 and sum(math.prod(s) for s in result.values()) == 168960, 'Independent schema failed')
    return result


def canonical(arrays):
    digest, schema = hashlib.sha256(), []
    for name in sorted(arrays):
        value = arrays[name]
        payload = value.tobytes(order='C')
        header = json.dumps([name, 'torch.float32', list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(payload)
        schema.append({'name': name, 'shape': list(value.shape), 'dtype': 'float32',
                       'bytes': len(payload), 'bytes_sha256': sha(payload)})
    return digest.hexdigest(), schema


def inner(left, right):
    # An explicit elementwise product/reduction is independent of np.dot in the
    # sealed analyzer. FP64 accumulation follows the supplied FP32 measurements.
    return float(np.sum(left * right, dtype=np.float64))


def norm(value):
    return math.sqrt(inner(value, value))


def compare(left, right):
    a, b = norm(left), norm(right)
    dot = inner(left, right)
    return {'dot': dot, 'left_norm': a, 'right_norm': b,
            'cosine': None if not a or not b else max(-1.0, min(1.0, dot / (a * b))),
            'left_zero': a == 0, 'right_zero': b == 0}


def fresh_adam_arithmetic(gradient):
    size = norm(gradient)
    clip = min(1.0, 1.0 / (size + 1e-6))
    clipped = gradient * clip
    direction = (-2.5e-5 * clipped) / (np.abs(clipped) + 1e-8)
    return direction, {'gradient_norm': size, 'clip_coefficient': clip,
        'LR': 2.5e-5, 'clip_denominator_epsilon': 1e-6, 'Adam_denominator_epsilon': 1e-8,
        'empty_moments': True, 'vector_arithmetic_only': True,
        'optimizer_constructed': False, 'parameter_applied': False}


def summarize(bank):
    objective = [bank[role]['objective_div4'] for role in ROLES]
    joint = np.stack(objective).sum(axis=0, dtype=np.float64)
    direction, adam = fresh_adam_arithmetic(joint)
    return {
        'gradient_norms': {r: {q: norm(bank[r][q]) for q in QUERIES} for r in ROLES},
        'objective_role_pairwise': {a: {b: compare(bank[a]['objective_div4'], bank[b]['objective_div4']) for b in ROLES} for a in ROLES},
        'joint_objective_div4_sum_norm': norm(joint), 'joint_objective_zero': not np.any(joint),
        'joint_first_order_derivatives': {q: {r: {
            'steepest_descent_first_order_derivative': -inner(bank[r][q], joint),
            'hypothetical_fresh_Adam_first_order_derivative': inner(bank[r][q], direction),
            'cosine_with_joint_objective': compare(bank[r][q], joint)['cosine'],
        } for r in ROLES} for q in QUERIES},
        'role_directional_contributions': {q: {r: {source: {
            'gradient_dot': inner(bank[r][q], bank[source]['objective_div4']),
            'steepest_descent_contribution': -inner(bank[r][q], bank[source]['objective_div4']),
            'cosine': compare(bank[r][q], bank[source]['objective_div4'])['cosine'],
        } for source in ROLES} for r in ROLES} for q in QUERIES},
        'hypothetical_fresh_Adam': adam,
        'objective_div4_sum_is_not_divided_again': True,
        'negative_derivative_means_local_predicted_decrease_only': True,
        'finite_step_or_PNG_prediction': False,
    }


def stats(values):
    vals = [x for x in values if x is not None]
    if not vals:
        return {'n': len(values), 'defined': 0, 'null': len(values)}
    return {'n': len(values), 'defined': len(vals), 'null': len(values) - len(vals),
        'negative': sum(x < 0 for x in vals), 'zero': sum(x == 0 for x in vals),
        'positive': sum(x > 0 for x in vals), 'min': min(vals), 'max': max(vals),
        'mean': math.fsum(vals) / len(vals), 'median': float(np.median(vals))}


def extra_geometry(bank):
    objectives = {r: bank[r]['objective_div4'] for r in ROLES}
    joint = np.stack(list(objectives.values())).sum(axis=0)
    non_dark = np.stack([objectives[r] for r in ROLES[:3]]).sum(axis=0)
    dark = ROLES[-1]
    sum_norms = math.fsum(norm(v) for v in objectives.values())
    return {
        'objective_vs_same_role_gate': {r: {q: compare(objectives[r], bank[r][q]) for q in QUERIES[1:]} for r in ROLES},
        'objective_norm_fraction_of_role_norm_sum': {r: None if sum_norms == 0 else norm(objectives[r]) / sum_norms for r in ROLES},
        'dark_to_nondark_sum_norm_ratio': None if norm(non_dark) == 0 else norm(objectives[dark]) / norm(non_dark),
        'joint_vs_dark_objective': compare(joint, objectives[dark]),
        'dark_ablation_at_same_state': {q: {r: {
            'all_roles_steepest_derivative': -inner(bank[r][q], joint),
            'nondark_roles_steepest_derivative': -inner(bank[r][q], non_dark),
            'dark_only_steepest_derivative': -inner(bank[r][q], objectives[dark]),
        } for r in ROLES} for q in QUERIES},
        'shared_state_only_not_optimizer_trajectory': True,
    }


class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        need(fullname.split('.')[0] not in {'torch', 'tensorflow', 'jax'}, 'Framework import forbidden: ' + fullname)
        return None


def numeric_crosscheck(actual, expected):
    checks, max_error = 0, 0.0
    def visit(a, b, path):
        nonlocal checks, max_error
        if isinstance(b, dict):
            need(type(a) is dict and set(a) == set(b), 'Analyzer keys differ at ' + path)
            for k in b:
                visit(a[k], b[k], path + '/' + k)
        elif isinstance(b, list):
            need(type(a) is list and len(a) == len(b), 'Analyzer list differs: ' + path)
            for i in range(len(b)):
                visit(a[i], b[i], path + '/' + str(i))
        elif type(b) in (int, float) and type(b) is not bool:
            checks += 1
            error = abs(a - b)
            max_error = max(max_error, error)
            need(math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-13), 'Independent arithmetic mismatch ' + path + ': ' + repr((a, b)))
        else:
            need(type(a) == type(b) and a == b, 'Analyzer semantic mismatch ' + path)
    visit(actual, expected, '')
    return {'pass': True, 'numeric_leaves': checks, 'maximum_absolute_error': max_error,
            'relative_tolerance': 2e-10, 'absolute_tolerance': 2e-13}


def fixtures(sealed):
    # Orthogonal, opposed, and zero measurements expose sign/cancellation bugs.
    a = np.array([3., 4., 0.])
    b = np.array([-3., 0., 4.])
    zero = np.zeros(3)
    need(norm(a) == 5 and inner(a, b) == -9 and compare(a, zero)['cosine'] is None, 'Analytical fixture failed')
    d, metadata = fresh_adam_arithmetic(a)
    clip = 1 / (5 + 1e-6)
    expected = np.array([-2.5e-5 * 3 * clip / (3 * clip + 1e-8), -2.5e-5 * 4 * clip / (4 * clip + 1e-8), 0.])
    need(np.allclose(d, expected, atol=1e-20, rtol=0), 'Clipping-before-epsilon fixture failed')
    bank = {r: {q: (a if i % 2 == 0 else -a).copy() for q in QUERIES} for i, r in enumerate(ROLES)}
    independent = summarize(bank)
    need(independent['joint_objective_zero'] and all(independent['joint_first_order_derivatives'][q][r]['cosine_with_joint_objective'] is None for q in QUERIES for r in ROLES), 'Cancellation/null fixture failed')
    cross = numeric_crosscheck(independent, sealed.summarize_vectors(bank, np=np))
    need(sealed.comparison(zero, a, np=np)['cosine'] is None, 'Sealed zero semantics failed')
    conflict = {r: {'objective_div4': factor * a, 'gate_mae': b.copy(), 'gate_adjacent': zero.copy()}
                for r, factor in zip(ROLES, (1., 2., -1., 4.))}
    conflict_summary = summarize(conflict)
    need(conflict_summary['joint_objective_div4_sum_norm'] == 30., 'Known nonzero joint norm failed')
    need(conflict_summary['joint_first_order_derivatives']['objective_div4']['clear']['steepest_descent_first_order_derivative'] == 150., 'Opposed objective sign fixture failed')
    need(all(conflict_summary['joint_first_order_derivatives']['gate_mae'][r]['steepest_descent_first_order_derivative'] == 54. for r in ROLES), 'Gate/objective conflict fixture failed')
    conflict_cross = numeric_crosscheck(conflict_summary, sealed.summarize_vectors(conflict, np=np))
    return {'pass': True, 'known_3_4_5_norm': 5, 'known_dot': -9, 'zero_cosine_is_null': True,
        'exact_cancellation_is_retained': True, 'clipping_before_adam_epsilon': metadata,
        'independent_vs_sealed': cross, 'known_nonzero_joint_norm': 30.,
        'known_opposed_objective_derivative': 150., 'known_gate_conflict_derivative': 54.,
        'conflict_independent_vs_sealed': conflict_cross}


def run():
    sys.meta_path.insert(0, NoFramework())
    consumed = {}
    def read(path, pin=None):
        path = path.resolve()
        raw = path.read_bytes()
        stamp = {'bytes': len(raw), 'sha256': sha(raw)}
        if pin is not None:
            need(stamp == {k: pin[k] for k in ('bytes', 'sha256')}, 'Consumed byte pin differs: ' + str(path))
        key = str(path.relative_to(ROOT)).replace('\\', '/')
        if key in consumed:
            need(consumed[key] == stamp, 'Consumed file changed during read: ' + key)
        consumed[key] = stamp
        return raw
    protocol_raw = read(PROTOCOL)
    need(sha(protocol_raw) == PROTOCOL_SHA, 'Exact sealed diagnostic protocol required')
    protocol = decode(protocol_raw)
    inventory = decode(read(INVENTORY))
    need(inventory['complete'] and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete bound inventory required')
    evidence = Path(inventory['evidence_root']).resolve()
    need(evidence.is_relative_to(ROOT), 'Evidence must be in workspace')
    pins = inventory['extracted_files']
    def evidence_read(name):
        need(name in pins, 'Missing inventory namespace binding: ' + name)
        return read(evidence / name, pins[name])
    for name in ('research/__init__.py', 'research/codeformer_u64_role_gradients_v1/__init__.py',
                 'research/codeformer_u64_role_gradients_v1/analysis.py', 'research/codeformer_u64_role_gradients_v1/contract.py'):
        read(ROOT / name, protocol['files'][name])
    sys.path.insert(0, str(ROOT))
    from research.codeformer_u64_role_gradients_v1 import analysis as sealed
    shapes = expected_shapes()
    need(shapes == sealed.contract.head_shapes(), 'Independent head schema differs from sealed schema')
    selection, plan = protocol['selection'], protocol['plan'][8:]
    rows, vectors, receipts = [], {}, []
    need(len(plan) == len(selection['optimization']) == 64, 'Exact64 planned photographs/roles required')
    for call, selected in zip(plan, selection['optimization']):
        slot = call['slot']
        need(slot == len(rows) + 9 and all(call[k] == selected[k] for k in ('id', 'reference_id', 'role')), 'Ordered plan/selection join failed')
        need(call['metric_region'] == ('central128' if call['role'] == ROLES[0] else 'whole'), 'Region binding failed')
        stem = 'outputs/u64_role_gradients_v1/calls/' + str(slot).zfill(4)
        scalar = decode(evidence_read(stem + '/gradients.json'))
        output = decode(evidence_read(stem + '/record.json'))
        need(scalar['call'] == output['call'] == call and scalar['metric_region'] == call['metric_region'], 'Saved scalar/call/region join failed')
        need(scalar['complete'] and scalar['objective_divided_exactly_once'] and scalar['RAW_differentiable_gate_metrics_only'] and scalar['PNG_gradient_claimed'] is False, 'Saved scalar scope differs')
        need(all(output[k] == selected[k] for k in ('id', 'reference_id', 'role', 'input', 'target')), 'Output photographic binding differs')
        need(scalar['output_record'] == pins[stem + '/record.json'], 'Scalar output-record pin differs')
        need(set(scalar['gradients']) == set(QUERIES), 'Exact three mandatory queries required')
        row = {k: call[k] for k in ('id', 'reference_id', 'role', 'metric_region')}
        row['gradients'] = {}
        vectors[call['id']] = {}
        for query in QUERIES:
            saved = scalar['gradients'][query]
            filename = stem + '/gradients_' + query + '.npz'
            need(saved['file'] == pins[filename], 'Gradient file/receipt pin differs')
            raw = evidence_read(filename)
            with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
                names = zipped.namelist()
                need(len(names) == len(set(names)) == 36 and set(names) == {n + '.npy' for n in shapes}, 'Exact unique36 ZIP members required')
            with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
                need(len(loaded.files) == len(set(loaded.files)) == 36 and set(loaded.files) == set(shapes), 'Exact36 NPZ namespace required')
                arrays = {n: loaded[n] for n in shapes}
            for name, val in arrays.items():
                need(val.dtype == np.dtype('float32') and val.shape == shapes[name] and np.isfinite(val).all(), 'Exact finite FP32 shape required: ' + name)
            digest, description = canonical(arrays)
            need(saved['description'] == {'canonical_sha256': digest, 'arrays': description, 'allow_pickle': False} and saved['canonical_sha256'] == digest, 'Canonical/schema receipt differs')
            zeros = [n for n in shapes if not np.any(arrays[n])]
            unused = saved['unused_names_explicit_zero']
            need(type(unused) is list and len(unused) == len(set(unused)) and set(unused).issubset(zeros), 'Unused receipt must name explicit zero tensors')
            need(saved['all_zero_names'] == zeros, 'Zero tensor metadata differs from actual bytes')
            need(type(saved['scalar']) in (int, float) and math.isfinite(saved['scalar']), 'Finite scalar required')
            if query == 'objective_div4':
                need(saved['scalar'] == scalar['objective'] / 4, 'Objective divided exactly once scalar differs')
            packed = np.concatenate([arrays[n].reshape(-1).astype(np.float64) for n in shapes])
            need(packed.size == 168960, 'Exact168960 coordinates required')
            row['gradients'][query] = arrays
            vectors[call['id']][query] = packed
            receipts.append({'slot': slot, 'id': call['id'], 'reference_id': call['reference_id'], 'role': call['role'],
                'query': query, 'metric_region': call['metric_region'], 'file': saved['file'],
                'canonical_sha256': digest, 'scalar': saved['scalar'], 'unused_names_explicit_zero': unused,
                'all_zero_names': zeros, 'nonzero_coordinates': int(np.count_nonzero(packed)),
                'packed_FP64_sha256': sha(packed.tobytes()), 'norm': norm(packed)})
        rows.append(row)
    need(len(receipts) == 192, 'Exact192 actual vectors required')
    fixture_result = fixtures(sealed)
    sealed_result = sealed.analyze(rows, selection, np=np)
    by = {(r['reference_id'], r['role']): r['id'] for r in rows}
    order = selection['optimization_reference_order']
    pooled = {r: {q: np.zeros(168960) for q in QUERIES} for r in ROLES}
    photos, old = [], []
    cancellation_norm = {r: dict.fromkeys(QUERIES, 0.) for r in ROLES}
    crosschecks = []
    for index, reference in enumerate(order):
        bank = {r: vectors[by[reference, r]] for r in ROLES}
        summary = summarize(bank)
        crosschecks.append(numeric_crosscheck(summary, {k: sealed_result['per_photograph'][index][k] for k in summary}))
        photos.append({'reference_id': reference, 'case_ids': {r: by[reference, r] for r in ROLES}, **summary, **extra_geometry(bank)})
        for r in ROLES:
            for q in QUERIES:
                pooled[r][q] += bank[r][q] / 16
                cancellation_norm[r][q] += norm(bank[r][q]) / 16
    for index in range(16):
        refs = {r: order[(index + offset) % 16] for r, offset in zip(ROLES, (0, 4, 8, 12))}
        bank = {r: vectors[by[refs[r], r]] for r in ROLES}
        summary = summarize(bank)
        crosschecks.append(numeric_crosscheck(summary, {k: sealed_result['original_offset_sampler_at_common_state'][index][k] for k in summary}))
        old.append({'old_schedule_update': index + 1, 'reference_ids': refs, **summary, **extra_geometry(bank)})
    pooled_summary = summarize(pooled)
    crosschecks.append(numeric_crosscheck(pooled_summary, sealed_result['pooled_role_mean']))
    cancellation = {r: {q: {'mean_individual_norm': cancellation_norm[r][q],
        'pooled_mean_norm': norm(pooled[r][q]),
        'pooled_over_mean_individual_norm': None if cancellation_norm[r][q] == 0 else norm(pooled[r][q]) / cancellation_norm[r][q],
    } for q in QUERIES} for r in ROLES}
    crosschecks.append(numeric_crosscheck(cancellation, sealed_result['pooled_cancellation_diagnostics']))
    geometry = {'complete': True, 'per_photograph': photos, 'pooled_role_mean': {**pooled_summary, **extra_geometry(pooled)},
        'original_offset_sampler_at_common_state': old, 'pooled_cancellation_diagnostics': cancellation}
    diagnosis = {
        'per_photograph_first_order_signs': {q: {r: {method: stats([p['joint_first_order_derivatives'][q][r][method] for p in photos])
            for method in ('steepest_descent_first_order_derivative', 'hypothetical_fresh_Adam_first_order_derivative')}
            for r in ROLES} for q in QUERIES},
        'old_offset_common_state_first_order_signs': {q: {r: {method: stats([p['joint_first_order_derivatives'][q][r][method] for p in old])
            for method in ('steepest_descent_first_order_derivative', 'hypothetical_fresh_Adam_first_order_derivative')}
            for r in ROLES} for q in QUERIES},
        'objective_same_role_gate_cosines': {r: {q: stats([p['objective_vs_same_role_gate'][r][q]['cosine'] for p in photos]) for q in QUERIES[1:]} for r in ROLES},
        'objective_pair_cosines': {a: {b: stats([p['objective_role_pairwise'][a][b]['cosine'] for p in photos]) for b in ROLES} for a in ROLES},
        'objective_norm_fraction_of_role_norm_sum': {r: stats([p['objective_norm_fraction_of_role_norm_sum'][r] for p in photos]) for r in ROLES},
        'dark_to_nondark_sum_norm_ratio': stats([p['dark_to_nondark_sum_norm_ratio'] for p in photos]),
        'joint_vs_dark_objective_cosine': stats([p['joint_vs_dark_objective']['cosine'] for p in photos]),
        'per_photograph_dark_ablation_signs': {q: {r: {method: stats([p['dark_ablation_at_same_state'][q][r][method] for p in photos])
            for method in ('all_roles_steepest_derivative', 'nondark_roles_steepest_derivative', 'dark_only_steepest_derivative')}
            for r in ROLES} for q in QUERIES},
        'scope': 'TRAIN initializer-local derivatives only; matched16 photographs and original offset schedule recombined at the same unchanged state.',
        'unique_historical_cause_established': False, 'finite_step_prediction': False,
        'Adam_trajectory_prediction': False, 'PNG_prediction': False, 'new_training_selected': False,
    }
    # Complete all read-only computations before proving consumed inputs stable.
    for name, stamp in consumed.items():
        raw = (ROOT / name).read_bytes()
        need({'bytes': len(raw), 'sha256': sha(raw)} == stamp, 'Consumed file changed during audit: ' + name)
    need(not any(name.split('.')[0] in {'torch', 'tensorflow', 'jax'} for name in sys.modules), 'Framework unexpectedly loaded')
    result = {'complete': True, 'format': 'u64-role-gradient-independent-Numpy-audit-v1',
        'protocol_sha256': PROTOCOL_SHA, 'gradient_files': 192, 'case_roles': 64, 'photographs': 16,
        'head_tensors_each': 36, 'head_coordinates_each': 168960,
        'all_192_exact_schema_canonical_description_file_bindings_pass': True,
        'unused_tensor_receipts': sum(bool(r['unused_names_explicit_zero']) for r in receipts),
        'zero_tensor_receipts': sum(bool(r['all_zero_names']) for r in receipts),
        'complete_zero_vectors': sum(r['nonzero_coordinates'] == 0 for r in receipts),
        'independent_crosschecks': {'pass': True, 'summary_sets': len(crosschecks),
            'numeric_leaves': sum(c['numeric_leaves'] for c in crosschecks),
            'maximum_absolute_error': max(c['maximum_absolute_error'] for c in crosschecks)},
        'analytical_fixtures': fixture_result, 'consumed_before_after_exact': True,
        'consumed_file_count': len(consumed), 'consumed_files': consumed,
        'execution_boundaries': {'framework_loaded': False, 'model_constructed': False,
            'optimizer_constructed': False, 'actual_gradient_computed': False,
            'parameter_applied': False, 'training_or_VM_called': False},
        'zero_policy': 'Measured zeros retained; receipt-authenticated unused tensors remain explicit zero; zero-norm cosine is null.',
        'model_qualified': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in (('audit.json', result), ('vectors.json', receipts), ('geometry.json', geometry),
                        ('diagnosis.json', diagnosis), ('sealed_analysis_crosschecked.json', sealed_result)):
        (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('consumed_files', 'analytical_fixtures')}, indent=2))


if __name__ == '__main__':
    run()
