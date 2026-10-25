"""Independent saved-state/gradient geometry audit; NumPy files only.

No framework/model/autograd/optimizer/parameter/VM call or direction solver.
The stored biased-surrogate vectors are measurements, not true PNG derivatives.
"""
import ast
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/inventory_v1.json'
PROTOCOL = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm/protocol.json'
OUT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_gradients'
PREFIX = 'outputs/u64_png_surrogate_measurement_v1/'
PROTOCOL_SHA = '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'
INITIAL_FILE_SHA = '5a2d55fe9f136a1fcf035de3c3ca46e363defa5874572bf0dac261838845247b'
FULL_CANONICAL = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
HEAD_CANONICAL = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
FROZEN_CANONICAL = 'f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
QUERIES = ('raw_mae', 'raw_adjacent', 'png_surrogate_mae', 'png_surrogate_adjacent')
LABELS = tuple((r, q) for r in ROLES for q in QUERIES)


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        need(fullname.split('.')[0] not in {'torch', 'tensorflow', 'jax'}, 'Framework import forbidden')
        return None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def doc(raw):
    def pairs(items):
        obj = {}
        for key, val in items:
            need(key not in obj, 'Duplicate JSON key')
            obj[key] = val
        return obj
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def shapes():
    result = {}
    for site, channels in (('32', 256), ('64', 256), ('128', 128), ('256', 128)):
        for part, shape in (('encoder.weight', (32, channels, 1, 1)), ('encoder.bias', (32,)),
            ('decoder.weight', (32, channels, 1, 1)), ('fused.weight', (32, channels, 1, 1)),
            ('coarse.weight', (32, 256, 1, 1)), ('spatial.weight', (32, 32, 3, 3)),
            ('spatial.bias', (32,)), ('output.weight', (channels, 32, 1, 1)), ('output.bias', (channels,))):
            result['heads.' + site + '.' + part] = shape
    return dict(sorted(result.items()))


SHAPES = shapes()


def canonical_update(digest, name, array):
    need(array.dtype == np.float32 and np.isfinite(array).all(), 'Finite FP32 tensor required')
    header = json.dumps([name, 'torch.float32', list(array.shape)], separators=(',', ':')).encode()
    payload = array.tobytes(order='C')
    digest.update(len(header).to_bytes(8, 'big')); digest.update(header); digest.update(payload)
    return {'name': name, 'shape': list(array.shape), 'dtype': 'float32',
            'bytes': len(payload), 'bytes_sha256': sha(payload)}


def canonical(arrays):
    digest, rows = hashlib.sha256(), []
    for name in sorted(arrays):
        rows.append(canonical_update(digest, name, arrays[name]))
    return digest.hexdigest(), rows


def inner(a, b):
    return float(np.sum(a * b, dtype=np.float64))


def norm(a):
    return math.sqrt(inner(a, a))


def compare(a, b):
    an, bn, dot = norm(a), norm(b), inner(a, b)
    return {'dot': dot, 'left_norm': an, 'right_norm': bn,
        'cosine': None if not an or not bn else max(-1., min(1., dot / (an * bn))),
        'left_zero': an == 0, 'right_zero': bn == 0}


def summary(bank):
    vectors = [bank[r][q] for r, q in LABELS]
    # Independent elementwise/reduction Gram, avoiding the analyzer matmul.
    lengths = [norm(v) for v in vectors]
    gram = [[inner(a, b) for b in vectors] for a in vectors]
    matrix = []
    for i in range(16):
        row = []
        for j in range(16):
            an, bn, dot = lengths[i], lengths[j], gram[i][j]
            row.append({'dot': dot, 'left_norm': an, 'right_norm': bn,
                'cosine': None if not an or not bn else max(-1., min(1., dot / (an * bn))),
                'left_zero': an == 0, 'right_zero': bn == 0})
        matrix.append(row)
    return {'vector_count': 16, 'coordinate_count': vectors[0].size,
        'label_order': [{'role': r, 'query': q} for r, q in LABELS],
        'gradient_norms': {r: {q: lengths[i] for i, (rr, q) in enumerate(LABELS) if rr == r} for r in ROLES},
        'pairwise16': matrix,
        'RAW_vs_PNG_surrogate_by_role': {r: {'mae': compare(bank[r]['raw_mae'], bank[r]['png_surrogate_mae']),
            'adjacent': compare(bank[r]['raw_adjacent'], bank[r]['png_surrogate_adjacent'])} for r in ROLES},
        'measured_zero_vectors': [{'role': r, 'query': q} for i, (r, q) in enumerate(LABELS) if lengths[i] == 0],
        'surrogate_is_actual_PNG_derivative': False, 'parameter_directions_computed': False,
        'common_direction_solver_used': False, 'finite_step_prediction': False}


def crosscheck(actual, recorded):
    count, maximum = 0, 0.
    def visit(a, b, path):
        nonlocal count, maximum
        if type(b) is dict:
            need(type(a) is dict and set(a) == set(b), 'Recorded geometry keys differ: ' + path)
            for k in b:
                visit(a[k], b[k], path + '/' + k)
        elif type(b) is list:
            need(type(a) is list and len(a) == len(b), 'Recorded geometry list differs: ' + path)
            for i in range(len(b)):
                visit(a[i], b[i], path + '/' + str(i))
        elif type(b) in (int, float):
            count += 1; error = abs(a - b); maximum = max(maximum, error)
            need(math.isclose(a, b, rel_tol=3e-9, abs_tol=3e-12), 'Recorded geometry arithmetic differs: ' + path)
        else:
            need(type(a) is type(b) and a == b, 'Recorded geometry semantics differ: ' + path)
    visit(actual, recorded, '')
    return {'numeric_leaves': count, 'maximum_absolute_error': maximum, 'pass': True}


def stats(values):
    defined = [v for v in values if v is not None]
    return {'count': len(values), 'defined': len(defined), 'null': len(values) - len(defined),
        'negative': sum(v < 0 for v in defined), 'zero': sum(v == 0 for v in defined),
        'positive': sum(v > 0 for v in defined), 'minimum': min(defined) if defined else None,
        'maximum': max(defined) if defined else None,
        'mean': math.fsum(defined) / len(defined) if defined else None}


def main():
    sys.meta_path.insert(0, NoFramework())
    consumed = {}
    def pin(path):
        before = path.stat(); digest = hashlib.sha256()
        need(not path.is_symlink() and before.st_nlink == 1, 'Ordinary single-link file required')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        after = path.stat()
        need((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed during hash')
        return {'bytes': before.st_size, 'sha256': digest.hexdigest()}
    def register(path, expected=None):
        actual = pin(path)
        if expected is not None:
            need(actual == {k: expected[k] for k in actual}, 'Exact byte pin differs: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        need(key not in consumed or consumed[key] == actual, 'Consumed source changed')
        consumed[key] = actual
        return path
    protocol = doc(register(PROTOCOL).read_bytes())
    need(consumed[PROTOCOL.relative_to(ROOT).as_posix()]['sha256'] == PROTOCOL_SHA, 'Exact sealed measurement protocol required')
    inventory = doc(register(INVENTORY).read_bytes())
    need(inventory['complete'] and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete exact return inventory required')
    evidence, pins = Path(inventory['evidence_root']), inventory['extracted_files']
    def bound(name, expected=None):
        need(name in pins and (expected is None or pins[name] == expected), 'Returned receipt namespace/pin differs: ' + name)
        return register(evidence / name, pins[name])
    def read(name, expected=None):
        return doc(bound(name, expected).read_bytes())
    result_record = read(PREFIX + 'result.json')
    need(result_record['complete'] and result_record['protocol_sha256'] == PROTOCOL_SHA
         and result_record['parameter_applications'] == result_record['optimizer_constructions'] == result_record['optimizer_steps'] == 0
         and result_record['operation_outcome_unknown'] is False and not result_record['unknown_reservations'], 'Complete known zero-update result required')
    need(result_record['counters'] == {'full_forward_attempts': 72, 'full_forward_returns': 72,
        'gradient_query_attempts': 256, 'gradient_query_returns': 256,
        'parity_forwards': 4, 'repeat_forwards': 4, 'diagnostic_forwards': 64}, 'Exact72/256 counts required')
    schema = read(PREFIX + 'head_schema.json')
    need(schema['head_shapes'] == {n: list(s) for n, s in SHAPES.items()}
         and schema['total_tensors'] == 575 and schema['frozen_tensors'] == 539, 'Exact retained head schema required')
    need(result_record['PNG_forward_exact_64_cases'] and result_record['PNG_surrogate_is_biased']
         and result_record['true_PNG_derivative_claimed'] is False
         and result_record['all575_parameters_bitwise_unchanged'] and result_record['all539_frozen_exact'],
         'Exact zero-update biased-surrogate result labels required')
    states = [read(PREFIX + 'full_states/' + label + '.json') for label in ('initial', 'closing')]
    state_paths = [bound(s['state']['path'], s['state']) for s in states]
    need(all(consumed[p.relative_to(ROOT).as_posix()] == {'bytes': 377504068, 'sha256': INITIAL_FILE_SHA} for p in state_paths), 'Both complete575 NPZs must equal original trained-u64 bytes')
    prior = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit/extracted/outputs/u64_role_gradients_v1/full_states/initial.npz'
    register(prior, {'bytes': 377504068, 'sha256': INITIAL_FILE_SHA})
    descriptions = []
    for state in states:
        rows = state['description']['arrays']
        need(len(rows) == len({r['name'] for r in rows}) == 575, 'Unique complete575 descriptions required')
        descriptions.append({r['name']: r for r in rows})
    digests = [hashlib.sha256(), hashlib.sha256()]
    heads_digest, frozen_digest = hashlib.sha256(), hashlib.sha256()
    state_rows, counts = [], {'heads': 0, 'base': 0, 'adapters': 0}
    with np.load(state_paths[0], allow_pickle=False) as initial, np.load(state_paths[1], allow_pickle=False) as closing:
        need(len(initial.files) == len(set(initial.files)) == len(closing.files) == 575
             and set(initial.files) == set(closing.files), 'Exact575 complete NPZ namespaces required')
        for name in sorted(initial.files):
            a, b = initial[name], closing[name]
            need(a.dtype == b.dtype == np.float32 and a.shape == b.shape and a.tobytes() == b.tobytes()
                 and np.isfinite(a).all(), 'Initial/closing tensor changed: ' + name)
            rows = [canonical_update(digests[0], name, a), canonical_update(digests[1], name, b)]
            need(rows[0] == descriptions[0][name] and rows[1] == descriptions[1][name], 'Full description/schema byte hash differs')
            kind = name.split('.')[0]; need(kind in counts, 'Unknown model namespace'); counts[kind] += 1
            if kind == 'heads':
                need(name in SHAPES and a.shape == SHAPES[name], 'Exact36-head full-state schema required')
                canonical_update(heads_digest, name, a)
            else:
                canonical_update(frozen_digest, name, a)
            state_rows.append(rows[0])
    need(counts == {'heads': 36, 'base': 515, 'adapters': 24}, 'Exact575 tensor census required')
    need(all(d.hexdigest() == FULL_CANONICAL for d in digests) and heads_digest.hexdigest() == HEAD_CANONICAL
         and frozen_digest.hexdigest() == FROZEN_CANONICAL, 'Exact original/frozen/head canonical values required')
    for state in states:
        need(state['complete'] and state['canonical_sha256'] == state['description']['canonical_sha256'] == FULL_CANONICAL
             and state['all575_exact_initial_values'] and state['all539_frozen_exact'] and state['gradient_fields_all_None'], 'Unchanged state receipts differ')
    rng0, rng1 = read(PREFIX + 'RNG_sampler_initial.json'), read(PREFIX + 'RNG_sampler_closing.json')
    need(rng0['complete'] and rng1['complete'] and rng0['RNG'] == rng1['RNG']
         and rng0['optimizer'] is rng1['optimizer'] is None, 'Complete unchanged RNG/null-optimizer required')
    need(rng0['sampler'] == {'next_slot': 1, 'fixed_photo_major_plan': True, 'automatic_resume_permitted': False}
         and rng1['sampler'] == {'next_slot': None, 'fixed_photo_major_plan': True, 'automatic_resume_permitted': False}, 'Exact fixed sampler start/end required')
    need(rng0['all575_canonical_sha256'] == FULL_CANONICAL and rng1['all575_unchanged'],
         'RNG sampler unchanged-state receipt differs')
    rng = rng0['RNG']
    need(set(rng) == {'Python', 'NumPy', 'CPU', 'CUDA'} and type(ast.literal_eval(rng['Python'])) is tuple
         and rng['NumPy'][0] == 'MT19937' and len(bytes.fromhex(rng['NumPy'][1])) == 2496
         and bytes.fromhex(rng['CPU']) and len(rng['CUDA']) == 1 and bytes.fromhex(rng['CUDA'][0]), 'Complete RNG payload required')
    selected = protocol['selection']['optimization']; plan = protocol['plan'][8:]
    need(len(selected) == len(plan) == 64 and len(result_record['diagnostic_records']) == 64, 'Exact64 rolecases required')
    packed, gradient_rows = {}, []
    for index, (call, source) in enumerate(zip(plan, selected)):
        folder = PREFIX + 'calls/' + str(call['slot']).zfill(4)
        scalar = read(folder + '/gradients.json', result_record['diagnostic_records'][index])
        output = read(folder + '/record.json', scalar['output_record'])
        need(scalar['call'] == output['call'] == call and all(call[k] == source[k] for k in ('id', 'reference_id', 'role'))
             and all(output[k] == source[k] for k in ('id', 'reference_id', 'role', 'split', 'input', 'target')),
             'Photograph/role/query/output/protocol join differs')
        need(call['slot'] == index + 9 and source['split'] == 'TRAIN'
             and source['cohort'] == 'optimize' and source['optimization_permitted'] is True
             and source['target'] is not None, 'Exact ordered admitted optimizing TRAIN query scope required')
        need(scalar['metric_region'] == call['metric_region'] == ('central128' if call['role'] == ROLES[0] else 'whole')
             and call['gradient_queries'] == list(QUERIES) and set(scalar['gradients']) == set(QUERIES), 'Exact four queries and gate ROI required')
        need(scalar['PNG_forward_actual_decoded_FP32_exact'] and scalar['PNG_biased_surrogate_gradient_only']
             and scalar['true_PNG_derivative_claimed'] is False and scalar['PNG_RGB8_file'] == output['outputs']['png'], 'PNG biased-surrogate provenance label differs')
        bound(scalar['PNG_forward_reference']['path'], scalar['PNG_forward_reference'])
        bound(scalar['PNG_RGB8_file']['path'], scalar['PNG_RGB8_file'])
        bound(output['outputs']['raw']['path'], output['outputs']['raw'])
        packed[call['id']] = {}
        for query in QUERIES:
            receipt = scalar['gradients'][query]
            path = bound(receipt['file']['path'], receipt['file'])
            need(receipt['file']['path'] == folder + '/gradients_' + query + '.npz', 'Exact query filename required')
            with zipfile.ZipFile(path) as zipped:
                names = zipped.namelist()
                need(len(names) == len(set(names)) == 36 and set(names) == {n + '.npy' for n in SHAPES}, 'Exact unique36 gradient ZIP members required')
            with np.load(path, allow_pickle=False) as arrays:
                need(set(arrays.files) == set(SHAPES), 'Exact36 gradient names required')
                decoded = {n: arrays[n] for n in SHAPES}
            need(all(v.dtype == np.float32 and v.shape == SHAPES[n] and np.isfinite(v).all() for n, v in decoded.items()), 'Exact finite FP32 gradient schema required')
            digest, description = canonical(decoded)
            need(receipt['description'] == {'canonical_sha256': digest, 'arrays': description, 'allow_pickle': False}
                 and receipt['canonical_sha256'] == digest and receipt['head_state_canonical_sha256'] == HEAD_CANONICAL,
                 'Gradient canonical/description/current-state binding differs')
            zeros = [n for n in SHAPES if not np.any(decoded[n])]
            unused = receipt['unused_names_explicit_zero']
            need(receipt['all_zero_names'] == zeros and type(unused) is list and len(unused) == len(set(unused))
                 and set(unused).issubset(zeros), 'Unused/zero receipt differs from actual tensors')
            need(receipt['surrogate_backward_identity'] == query.startswith('png_surrogate_')
                 and receipt['true_delivered_PNG_derivative'] is False
                 and type(receipt['scalar']) in (int, float) and math.isfinite(receipt['scalar']), 'Query scalar/backward/derivative label differs')
            vector = np.concatenate([decoded[n].reshape(-1).astype(np.float64) for n in SHAPES])
            need(vector.shape == (168960,), 'Exact168960 vector coordinates required')
            packed[call['id']][query] = vector
            gradient_rows.append({'slot': call['slot'], 'id': call['id'], 'reference_id': call['reference_id'],
                'role': call['role'], 'query': query, 'metric_region': call['metric_region'], 'scalar': receipt['scalar'],
                'canonical_sha256': digest, 'file': receipt['file'], 'norm': norm(vector),
                'all_zero_names': zeros, 'unused_names_explicit_zero': unused,
                'packed_FP64_sha256': sha(vector.tobytes())})
    need(len(gradient_rows) == 256 and sum('/gradients_' in n and n.endswith('.npz') for n in pins) == 256,
         'Exactly256 actual mandatory gradient witnesses required')
    stored = read(result_record['saved_gradient_geometry']['path'], result_record['saved_gradient_geometry'])
    need(stored['complete'] and stored['mandatory_query_vectors'] == 256 and stored['pooled_vectors'] == 16
         and stored['head_tensors'] == 36 and stored['head_coordinates'] == 168960
         and stored['common_direction_solver_used'] is False and stored['parameter_directions_computed'] is False,
         'Stored geometry scope must remain16 pooled zero-direction arithmetic')
    lookup = {(r['reference_id'], r['role']): r['id'] for r in selected}
    photos, crosschecks = [], []
    order = protocol['selection']['optimization_reference_order']
    pooled = {r: {q: np.zeros(168960) for q in QUERIES} for r in ROLES}
    means_norm = {r: dict.fromkeys(QUERIES, 0.) for r in ROLES}
    for i, reference in enumerate(order):
        bank = {r: packed[lookup[reference, r]] for r in ROLES}
        measured = summary(bank)
        need(stored['per_photograph'][i]['reference_id'] == reference
             and stored['per_photograph'][i]['case_ids'] == {r: lookup[reference, r] for r in ROLES},
             'Stored per-photograph geometry binding differs')
        crosschecks.append(crosscheck(measured, {k: stored['per_photograph'][i][k] for k in measured}))
        photos.append({'reference_id': reference, 'case_ids': {r: lookup[reference, r] for r in ROLES}, **measured})
        for role in ROLES:
            for query in QUERIES:
                pooled[role][query] += bank[role][query] / 16.
                means_norm[role][query] += norm(bank[role][query]) / 16.
    pooled_summary = summary(pooled)
    crosschecks.append(crosscheck(pooled_summary, stored['pooled_role_mean16']))
    cancellation = {r: {q: {'mean_individual_norm': means_norm[r][q],
        'pooled_mean_norm': norm(pooled[r][q]), 'pooled_over_mean_individual_norm':
        None if means_norm[r][q] == 0 else norm(pooled[r][q]) / means_norm[r][q]} for q in QUERIES} for r in ROLES}
    crosschecks.append(crosscheck(cancellation, stored['pooled_cancellation_diagnostics']))
    alignment = {r: {metric: stats([photo['RAW_vs_PNG_surrogate_by_role'][r][metric]['cosine'] for photo in photos])
                    for metric in ('mae', 'adjacent')} for r in ROLES}
    conflicts = []
    for i, (r, q) in enumerate(LABELS):
        for j in range(i + 1, 16):
            other_r, other_q = LABELS[j]
            cosines = [p['pairwise16'][i][j]['cosine'] for p in photos]
            conflicts.append({'left': {'role': r, 'query': q}, 'right': {'role': other_r, 'query': other_q},
                'per_photo_cosines': stats(cosines), 'pooled_cosine': pooled_summary['pairwise16'][i][j]['cosine']})
    # Known analytical zero/opposition fixture keeps undefined-cosine semantics.
    need(compare(np.zeros(3), np.ones(3))['cosine'] is None
         and compare(np.array([3., 4.]), np.array([-3., -4.]))['cosine'] == -1., 'Analytical zero/opposition fixture failed')
    register(Path(__file__))
    for name, expected in consumed.items():
        need(pin(ROOT / name) == expected, 'Consumed file changed before/after: ' + name)
    need(not any(n.split('.')[0] in {'torch', 'tensorflow', 'jax'} for n in sys.modules), 'Unexpected neural framework loaded')
    report = {'complete': True, 'saved_state_gradient_geometry_audit_pass': True,
        'protocol_sha256': PROTOCOL_SHA, 'all575_initial_closing_exact_original_u64': True,
        'original_full_NPZ_sha256': INITIAL_FILE_SHA, 'full575_canonical_sha256': FULL_CANONICAL,
        'head36_canonical_sha256': HEAD_CANONICAL, 'frozen539_canonical_sha256': FROZEN_CANONICAL,
        'tensor_census': counts, 'RNG_and_fixed_sampler_null_optimizer_receipts_pass': True,
        'gradient_NPZ_files': 256, 'head_tensors_each': 36, 'coordinates_each': 168960,
        'query_order': list(QUERIES), 'zero_tensor_receipts': sum(bool(r['all_zero_names']) for r in gradient_rows),
        'unused_tensor_receipts': sum(bool(r['unused_names_explicit_zero']) for r in gradient_rows),
        'complete_zero_vectors': sum(r['norm'] == 0 for r in gradient_rows),
        'independent_geometry_crosscheck': {'pass': True, 'summary_sets': len(crosschecks),
            'numeric_leaves': sum(c['numeric_leaves'] for c in crosschecks),
            'maximum_absolute_error': max(c['maximum_absolute_error'] for c in crosschecks)},
        'RAW_PNG_surrogate_alignment_per_photo': alignment,
        'pooled_role_mean16': pooled_summary, 'pooled_cancellation_diagnostics': cancellation,
        'all120_pair_conflict_counts': conflicts, 'per_photograph': photos,
        'gradient_receipts': gradient_rows, 'full575_tensor_rows': state_rows,
        'consumed_files': consumed, 'all_consumed_before_after_exact': True,
        'caller_saved_PNG_forward_pixels_independent_audit_separate': True,
        'PNG_gradient_is_biased_identity_surrogate_not_true_derivative': True,
        'direction_solver_or_parameter_directions_used': False,
        'new_training_recipe_selected': False, 'neural_model_autograd_optimizer_parameter_VM_calls': 0,
        'measurement_completion_does_not_qualify_model': True, 'model_qualified': False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'receipt_v1.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('pooled_role_mean16', 'pooled_cancellation_diagnostics',
        'all120_pair_conflict_counts', 'per_photograph', 'gradient_receipts', 'full575_tensor_rows', 'consumed_files')}, indent=2))


if __name__ == '__main__':
    main()
