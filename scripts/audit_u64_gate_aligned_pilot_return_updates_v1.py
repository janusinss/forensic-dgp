"""Independent saved-array audit of returned finite head updates; no framework.

Reconstructs arithmetic from retained arrays only. It never instantiates a
model, applies weights, computes an actual gradient, or contacts a VM.
"""
import ast
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
INVENTORY = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/inventory_v1.json'
PROTOCOL = ROOT / 'outputs/cctv_codeformer_u64_gate_aligned_pilot_v1_vm/protocol.json'
OUT = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_updates'
PREFIX = 'outputs/u64_gate_aligned_pilot_v1/'
PROTOCOL_SHA = '35a5cfd877a97355622c6c69d9792fd9238ccf312335dd679081a8636ba23b48'
INITIAL_FILE_SHA = '5a2d55fe9f136a1fcf035de3c3ca46e363defa5874572bf0dac261838845247b'
INITIAL_CANONICAL = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
HEAD_CANONICAL = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
FROZEN_CANONICAL = 'f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
QUERIES = ('gate_mae', 'gate_adjacent')


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
        result = {}
        for key, val in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = val
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def shapes():
    result = {}
    for site, channels in (('32', 256), ('64', 256), ('128', 128), ('256', 128)):
        for name, shape in (('encoder.weight', (32, channels, 1, 1)), ('encoder.bias', (32,)),
            ('decoder.weight', (32, channels, 1, 1)), ('fused.weight', (32, channels, 1, 1)),
            ('coarse.weight', (32, 256, 1, 1)), ('spatial.weight', (32, 32, 3, 3)),
            ('spatial.bias', (32,)), ('output.weight', (channels, 32, 1, 1)), ('output.bias', (channels,))):
            result['heads.' + site + '.' + name] = shape
    return dict(sorted(result.items()))


SHAPES = shapes()


def canonical_update(digest, name, array):
    need(array.dtype == np.float32 and np.isfinite(array).all(), 'Finite FP32 array required')
    header = json.dumps([name, 'torch.float32', list(array.shape)], separators=(',', ':')).encode()
    payload = array.tobytes(order='C')
    digest.update(len(header).to_bytes(8, 'big'))
    digest.update(header)
    digest.update(payload)
    return {'name': name, 'shape': list(array.shape), 'dtype': 'float32',
            'bytes': len(payload), 'bytes_sha256': sha(payload)}


def canonical(arrays):
    digest, desc = hashlib.sha256(), []
    for name in sorted(arrays):
        desc.append(canonical_update(digest, name, arrays[name]))
    return digest.hexdigest(), desc


def flat(arrays):
    return np.concatenate([arrays[n].reshape(-1).astype(np.float64) for n in SHAPES])


def inner(a, b):
    return float(np.sum(a * b, dtype=np.float64))


def norm(a):
    return math.sqrt(inner(a, a))


def close(actual, expected, message, rtol=2e-9, atol=2e-12):
    need(np.allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol), message)


def independent_faces(vectors):
    lengths = np.array([norm(v) for v in vectors])
    need(np.all(lengths > 0), 'All eight measured gates nonzero required')
    units = vectors / lengths[:, None]
    # Explicit product/reduction Gram is separate from the worker matmul path.
    gram = np.array([[inner(a, b) for b in units] for a in units])
    best = None
    for mask in range(1, 256):
        ids = [i for i in range(8) if mask & (1 << i)]
        count = len(ids)
        matrix = np.zeros((count + 1, count + 1))
        matrix[:count, :count] = gram[np.ix_(ids, ids)]
        matrix[:count, count] = 1.
        matrix[count, :count] = 1.
        rhs = np.zeros(count + 1)
        rhs[count] = 1.
        answer = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
        if norm(matrix @ answer - rhs) > 1e-10 or answer[:count].min() < -1e-10:
            continue
        weights = np.maximum(answer[:count], 0.)
        if abs(weights.sum() - 1.) > 1e-9:
            continue
        weights /= weights.sum()
        full = np.zeros(8)
        full[ids] = weights
        squared = float(full @ gram @ full)
        if best is None or squared < best[0]:
            best = squared, full
    need(best is not None, 'Independent feasible simplex face required')
    combination = np.sum(best[1][:, None] * units, axis=0)
    direction = -combination / norm(combination)
    unit_dots = np.array([inner(v, direction) for v in units])
    need(-unit_dots.max() > 1e-6, 'Independent eight-gate strict margin required')
    return {'weights': best[1], 'direction': direction, 'gram': gram, 'lengths': lengths,
            'combination_norm': norm(combination), 'margin': -unit_dots.max()}


def main():
    sys.meta_path.insert(0, NoFramework())
    consumed = {}
    def pin(path):
        path = path.resolve()
        stat = path.stat()
        need(not path.is_symlink() and stat.st_nlink == 1, 'Ordinary single-link source required')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        after = path.stat()
        need((stat.st_size, stat.st_mtime_ns, stat.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'Source changed during hash')
        return {'bytes': stat.st_size, 'sha256': digest.hexdigest()}
    def register(path, expected=None):
        actual = pin(path)
        if expected is not None:
            need(actual == {k: expected[k] for k in actual}, 'File pin differs: ' + str(path))
        key = path.resolve().relative_to(ROOT).as_posix()
        need(key not in consumed or consumed[key] == actual, 'Consumed file changed')
        consumed[key] = actual
        return path
    protocol = doc(register(PROTOCOL).read_bytes())
    need(consumed[PROTOCOL.relative_to(ROOT).as_posix()]['sha256'] == PROTOCOL_SHA, 'Exact sealed protocol required')
    inventory = doc(register(INVENTORY).read_bytes())
    need(inventory['complete'] and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete bound inventory required')
    evidence, pins = Path(inventory['evidence_root']), inventory['extracted_files']
    def bound(name, expected=None):
        need(name in pins, 'Missing returned file: ' + name)
        if expected is not None:
            need(pins[name] == expected, 'Receipt/inventory file join differs: ' + name)
        return register(evidence / name, pins[name])
    def read(name, expected=None):
        return doc(bound(name, expected).read_bytes())
    def npz(name, count=36, expected=None, description=None):
        path = bound(name, expected)
        with zipfile.ZipFile(path) as stream:
            names = stream.namelist()
            need(len(names) == len(set(names)) == count and all(n.endswith('.npy') and '/' not in n and '\\' not in n for n in names), 'Exact unique flat NPZ schema required')
        with np.load(path, allow_pickle=False) as data:
            need(set(data.files) == set(SHAPES), 'Exact36 head names required')
            arrays = {n: data[n] for n in SHAPES}
        need(all(a.dtype == np.float32 and a.shape == SHAPES[n] and np.isfinite(a).all() for n, a in arrays.items()), 'Exact finite FP32 head schema required')
        canonical_sha, desc = canonical(arrays)
        if description is not None:
            need(description == {'canonical_sha256': canonical_sha, 'arrays': desc, 'allow_pickle': False}, 'Complete canonical/schema description differs')
        return arrays, canonical_sha
    def npy(name, shape, expected=None):
        with bound(name, expected).open('rb') as stream:
            value = np.load(stream, allow_pickle=False)
        need(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), 'Exact finite FP64 arithmetic array required')
        return value
    failure = read(PREFIX + 'failure.json')
    need(failure['quality_stop'] and failure['active_update'] == 2 and failure['active_slot'] == 302
         and failure['operation_outcome_unknown'] is False and not failure['unknown_reservations'], 'Known quality stop at update2 required')
    need(failure['counters']['parameter_applications'] == 2 and failure['counters']['gradient_query_returns'] == 256, 'Exactly two applied updates and256 query returns required')
    need(sum(name.endswith('.npz') and '/gradients_' in name for name in pins) == 256,
         'Exactly256 returned gradient NPZs required, no unaccounted gradient bank')
    need(not any('optimizer' in Path(name).name.lower() for name in pins),
         'No optimizer-state file belongs to this optimizer-free run')
    def prefix_counts(slot):
        counts = dict.fromkeys(failure['counters'], 0)
        for call in protocol['plan'][:slot]:
            if call['kind'] == 'forward':
                counts['full_forward_attempts'] += 1
                counts['full_forward_returns'] += 1
                counts[call['category']] += 1
                counts['gradient_query_attempts'] += len(call['gradient_queries'])
                counts['gradient_query_returns'] += len(call['gradient_queries'])
            else:
                counts['parameter_application_attempts'] += 1
                counts['parameter_applications'] += 1
        return counts
    need(failure['counters'] == prefix_counts(302), 'Failure counters differ from exact completed plan prefix')
    preflight = read(PREFIX + 'preflight.json')
    need(preflight['optimizer'] is None and preflight['full575_canonical_sha256'] == INITIAL_CANONICAL
         and preflight['frozen539_canonical_sha256'] == FROZEN_CANONICAL, 'Original/frozen/null-optimizer preflight required')
    # Head checkpoints authenticate each current query state and actual commits.
    checkpoints, heads, head_shas, resumes = {}, {}, {}, {}
    for update in range(3):
        stem = PREFIX + 'checkpoints/u' + str(update).zfill(4)
        saved = read(stem + '/checkpoint.json')
        arrays, digest = npz(stem + '/heads.npz', expected=saved['heads'], description=saved['head_description'])
        resume = read(stem + '/resume.json', saved['resume'])
        need(saved['complete'] and saved['update'] == resume['update'] == update
             and saved['frozen539_canonical_sha256'] == resume['frozen539_canonical_sha256'] == FROZEN_CANONICAL,
             'Checkpoint state/frozen/update binding differs')
        need(resume['optimizer'] is None and resume['optimizer_constructions'] == 0
             and resume['gradient_descent_parameter_applications'] == resume['counters']['parameter_applications'] == update,
             'Checkpoint must contain explicit null optimizer and actual application count')
        need(resume['parameter_names'] == list(SHAPES) and resume['head_canonical_sha256'] == digest
             and resume['initial_head_canonical_sha256'] == HEAD_CANONICAL and resume['protocol_sha256'] == PROTOCOL_SHA,
             'Checkpoint coordinate/initializer/protocol binding differs')
        next_slot = 1 if update == 0 else next(c['slot'] + 1 for c in protocol['plan'] if c['kind'] == 'parameter_application' and c['new_update'] == update)
        need(resume['sampler'] == {'fixed_plan': True, 'next_slot': next_slot, 'automatic_resume_permitted': False}, 'Exact fixed sampler receipt required')
        need(resume['counters'] == prefix_counts(next_slot - 1), 'Checkpoint counters differ from completed application plan prefix')
        checkpoints[update], heads[update], head_shas[update], resumes[update] = saved, arrays, digest, resume
    need(head_shas[0] == HEAD_CANONICAL, 'Exact original trained-u64 head checkpoint required')
    need(resumes[0]['RNG'] == resumes[1]['RNG'] == resumes[2]['RNG'], 'Run RNG changed across head checkpoints')
    rng = resumes[0]['RNG']
    need(set(rng) == {'Python', 'NumPy', 'CPU', 'CUDA'} and type(ast.literal_eval(rng['Python'])) is tuple
         and rng['NumPy'][0] == 'MT19937' and len(bytes.fromhex(rng['NumPy'][1])) == 2496
         and bytes.fromhex(rng['CPU']) and len(rng['CUDA']) == 1 and bytes.fromhex(rng['CUDA'][0]), 'Complete actual RNG representation required')
    failed_rng = read(failure['failed_RNG_sampler']['path'], failure['failed_RNG_sampler'])
    need(failed_rng['RNG'] == rng and failed_rng['optimizer'] is None and failed_rng['automatic_resume_permitted'] is False
         and failed_rng['active_update'] == 2 and failed_rng['active_plan_slot'] == 302
         and failed_rng['counters'] == failure['counters'] and failed_rng['operation_outcome_unknown'] is False,
         'Failed state RNG/sampler binding differs')
    need(failure['last_checkpoint'] == pins[PREFIX + 'checkpoints/u0002/checkpoint.json']
         and failure['failed_current_state'] == pins[PREFIX + 'full_states/failed_current.json']
         and failure['last_complete_state'] == pins[PREFIX + 'full_states/initial.json'],
         'Failure state/checkpoint pins differ from retained payloads')
    # Complete575 witnesses are streamed tensor by tensor to bound local memory.
    state_records = [read(PREFIX + 'full_states/' + label + '.json') for label in ('initial', 'failed_current')]
    state_paths = [bound(row['state']['path'], row['state']) for row in state_records]
    need(consumed[state_paths[0].relative_to(ROOT).as_posix()] == {'bytes': 377504068, 'sha256': INITIAL_FILE_SHA}, 'Original complete575 file differs from retained trained-u64')
    prior_initial = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit/extracted/outputs/u64_role_gradients_v1/full_states/initial.npz'
    register(prior_initial, {'bytes': 377504068, 'sha256': INITIAL_FILE_SHA})
    descriptions = []
    for row in state_records:
        arrays = row['description']['arrays']
        need(len(arrays) == len({d['name'] for d in arrays}) == 575, 'Unique complete575 descriptions required')
        descriptions.append({d['name']: d for d in arrays})
    full_digests = [hashlib.sha256(), hashlib.sha256()]
    frozen_digests = [hashlib.sha256(), hashlib.sha256()]
    rows575, changed_frozen, changed_heads = [], [], []
    with np.load(state_paths[0], allow_pickle=False) as initial, np.load(state_paths[1], allow_pickle=False) as failed:
        need(len(initial.files) == len(set(initial.files)) == 575 and set(initial.files) == set(failed.files)
             and len(failed.files) == 575, 'Exact two complete575 NPZ namespaces required')
        for name in sorted(initial.files):
            a, b = initial[name], failed[name]
            need(a.dtype == b.dtype == np.float32 and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Exact finite575 state schema required')
            desc0, desc1 = canonical_update(full_digests[0], name, a), canonical_update(full_digests[1], name, b)
            need(descriptions[0][name] == desc0 and descriptions[1][name] == desc1, 'Full state descriptions differ')
            same = a.tobytes() == b.tobytes()
            if name.startswith('heads.'):
                need(name in SHAPES and a.shape == SHAPES[name] and a.tobytes() == heads[0][name].tobytes()
                     and b.tobytes() == heads[2][name].tobytes(), 'Full-state/checkpoint head mismatch')
                if not same:
                    changed_heads.append(name)
            else:
                canonical_update(frozen_digests[0], name, a)
                canonical_update(frozen_digests[1], name, b)
                if not same:
                    changed_frozen.append(name)
            rows575.append({'name': name, 'shape': list(a.shape), 'initial_bytes_sha256': desc0['bytes_sha256'],
                            'failed_bytes_sha256': desc1['bytes_sha256'], 'exact_equal': same})
    need(not changed_frozen and len(rows575) - len(SHAPES) == 539, 'All frozen539 must be bitwise unchanged')
    full_shas = [h.hexdigest() for h in full_digests]
    need(full_shas[0] == INITIAL_CANONICAL and all(h.hexdigest() == FROZEN_CANONICAL for h in frozen_digests), 'Original/frozen canonical identity differs')
    for i, row in enumerate(state_records):
        need(row['complete'] and row['canonical_sha256'] == row['description']['canonical_sha256'] == full_shas[i]
             and row['head36_canonical_sha256'] == head_shas[(0, 2)[i]] and row['all539_frozen_exact'], 'Full state canonical/checkpoint receipt differs')
    results, gradients = [], []
    selected = {r['id']: r for r in protocol['selection']['optimization']}
    for update in (1, 2):
        stem = PREFIX + 'updates/u' + str(update).zfill(4)
        precommit, commit, solver_receipt = [read(stem + '/' + n + '.json') for n in ('precommit', 'commit', 'solver')]
        calls = [c for c in protocol['plan'] if c['new_update'] == update and c['gradient_queries']]
        need(len(calls) == 64 and len(precommit['all_query_receipts']) == 64, 'Complete64 same-state optimization rows required')
        bank = {r: {q: [] for q in QUERIES} for r in ROLES}
        for index, call in enumerate(calls):
            folder = PREFIX + 'calls/' + str(call['slot']).zfill(4)
            scalar, output = read(folder + '/gradients.json'), read(folder + '/record.json')
            source = selected[call['id']]
            need(scalar['call'] == output['call'] == call and scalar['head_before_sha256'] == output['head_before_sha256'] == head_shas[update - 1], 'Gradient call/current-state join differs')
            need(output['update'] == update and all(output[k] == source[k] for k in ('id', 'reference_id', 'role', 'split', 'input', 'target')),
                 'Original admitted photographic role/input/target join differs')
            need(scalar['output_record'] == pins[folder + '/record.json'] and scalar['complete']
                 and scalar['RAW_differentiable_metrics_only'] and scalar['PNG_gradient_claimed'] is False
                 and set(scalar['gradients']) == set(QUERIES), 'Complete exact two-query RAW scalar receipt required')
            need(precommit['all_query_receipts'][index] == {'call': call, 'queries': scalar['gradients']}, 'Update gradient receipt list differs')
            for query in QUERIES:
                receipt = scalar['gradients'][query]
                need(read(folder + '/gradient_' + query + '_receipt.json') == receipt, 'Per-query receipt differs')
                arrays, digest = npz(folder + '/gradients_' + query + '.npz', expected=receipt['file'], description=receipt['description'])
                need(receipt['canonical_sha256'] == digest and receipt['head_before_sha256'] == head_shas[update - 1], 'Gradient canonical/current-state differs')
                zeros = [n for n in SHAPES if not np.any(arrays[n])]
                need(receipt['all_zero_names'] == zeros == [] and receipt['unused_names_explicit_zero'] == [], 'Measured zero/unused metadata differs or stops mandatory query')
                need(type(receipt['scalar']) in (float, int) and math.isfinite(receipt['scalar']), 'Finite scalar required')
                field = 'mae' if query == 'gate_mae' else 'adjacent_rgb_error'
                measured = output['regional_metrics']['raw'][call['metric_region']][field]
                close(receipt['scalar'], measured, 'Connected scalar vs saved FP64 RAW metric differs beyond FP32 tolerance', rtol=3e-6, atol=1e-8)
                vector = flat(arrays)
                need(vector.shape == (168960,) and norm(vector) > 0, 'Complete nonzero168960 vector required')
                bank[call['role']][query].append(vector)
                gradients.append({'update': update, 'slot': call['slot'], 'id': call['id'], 'query': query,
                    'head_before_sha256': receipt['head_before_sha256'], 'canonical_sha256': digest,
                    'scalar': receipt['scalar'], 'norm': norm(vector), 'file': receipt['file']})
        pooled = np.stack([np.sum(np.stack(bank[r][q]), axis=0, dtype=np.float64) / 16. for r in ROLES for q in QUERIES])
        saved_pooled = npy(stem + '/pooled_gate_gradients.npy', (8, 168960), precommit['pooled_gate_gradients'])
        need(np.array_equal(pooled, saved_pooled), 'Saved pooled mean16 differs from actual256-gradient source bank')
        independent = independent_faces(pooled)
        direction = npy(stem + '/direction.npy', (168960,))
        close(direction, independent['direction'], 'Saved direction differs from independent exhaustive-face solve', rtol=1e-6, atol=1e-9)
        close(solver_receipt['unit_gram'], independent['gram'], 'Solver Gram differs', rtol=2e-10)
        close(solver_receipt['simplex_coefficients'], independent['weights'], 'Solver face weights differ', rtol=1e-6, atol=1e-8)
        close(solver_receipt['input_gradient_norms'], independent['lengths'], 'Solver gradient norms differ')
        close(solver_receipt['combination_norm'], independent['combination_norm'], 'Solver combination norm differs')
        close(solver_receipt['minimum_normalized_descent_margin'], independent['margin'], 'Solver strict margin differs')
        need(solver_receipt['face_count'] == 255 and solver_receipt['all_actual_derivatives_strict_negative'], 'Complete exhaustive-face strict certificate required')
        units = pooled / independent['lengths'][:, None]
        raw_dots = np.array([inner(v, direction) for v in pooled])
        unit_dots = np.array([inner(v, direction) for v in units])
        close(raw_dots, solver_receipt['original_gradient_dot_direction'], 'Solver actual direction dots differ')
        close(unit_dots, solver_receipt['unit_gradient_dot_direction'], 'Solver normalized direction dots differ')
        need(np.all(raw_dots < 0) and -unit_dots.max() > 1e-6, 'Saved direction lacks all-eight strict local descent')
        proposed, proposed_sha = npz(stem + '/proposed_heads.npz', expected=precommit['candidate'], description=precommit['candidate_description'])
        now, original = flat(heads[update - 1]), flat(heads[0])
        alpha = .99 * 2.5e-5 / float(np.max(np.abs(direction)))
        expected_candidate = (now + alpha * direction).astype(np.float32).astype(np.float64)
        proposed_flat = flat(proposed)
        need(np.array_equal(expected_candidate, proposed_flat) and proposed_sha == head_shas[update]
             and all(proposed[n].tobytes() == heads[update][n].tobytes() for n in SHAPES), 'Actual committed checkpoint differs from reconstructed fixed0.99 FP32 candidate')
        delta = proposed_flat - now
        saved_delta = npy(stem + '/actual_rounded_delta.npy', (168960,), precommit['actual_delta'])
        need(np.array_equal(delta, saved_delta), 'Actual rounded saved displacement differs')
        cumulative = proposed_flat - original
        actual_dots = np.array([inner(v, delta) for v in pooled])
        actual_cosines = actual_dots / (independent['lengths'] * norm(delta))
        need(np.max(np.abs(delta)) <= 2.5e-5 and np.max(np.abs(cumulative)) <= 1e-4 and np.all(actual_dots < 0), 'Actual FP32 coordinate/cumulative/eight-gate dot bound failed')
        certificate = precommit['certificate']
        need(commit['rounded_certificate'] == certificate and commit['candidate'] == precommit['candidate']
             and commit['actual_delta'] == precommit['actual_delta'] and commit['head_before_sha256'] == precommit['head_before_sha256'] == head_shas[update - 1]
             and commit['head_after_sha256'] == head_shas[update] and commit['optimizer'] is None
             and commit['new_update'] == update and commit['counters']['parameter_applications'] == update,
             'Precommit/commit/current-checkpoint join differs')
        application_slot = next(c['slot'] for c in protocol['plan'] if c['kind'] == 'parameter_application' and c['new_update'] == update)
        need(commit['counters'] == resumes[update]['counters'] == prefix_counts(application_slot),
             'Actual commit/checkpoint counters differ from exact completed plan prefix')
        close(certificate['step_scalar'], alpha, 'Fixed0.99 alpha differs')
        close(certificate['actual_max_absolute_coordinate_delta'], float(np.max(np.abs(delta))), 'Actual maxstep receipt differs')
        close(certificate['actual_max_cumulative_coordinate_delta'], float(np.max(np.abs(cumulative))), 'Actual cumulative receipt differs')
        close(certificate['actual_FP32_displacement_gradient_dots'], actual_dots, 'Actual rounded gradient dot receipt differs')
        close(certificate['actual_FP32_minimum_normalized_descent_margin'], -actual_cosines.max(), 'Actual rounded cosine receipt differs')
        need(certificate['fixed_numerical_headroom'] == .99 and certificate['step_attempts'] == 1
             and certificate['scalar_chosen_before_metrics'] and certificate['metric_line_search_or_adaptive_retry'] is False,
             'Fixed single attempt/nonadaptive scalar receipt required')
        results.append({'update': update, 'head_before_sha256': head_shas[update - 1], 'head_after_sha256': head_shas[update],
            'gradient_NPZs': 128, 'pooled_mean16_byte_exact': True,
            'direction_independent_L2_difference': norm(direction - independent['direction']),
            'direction_minimum_normalized_margin': float(-unit_dots.max()),
            'simplex_coefficients': independent['weights'].tolist(), 'fixed_alpha': alpha,
            'actual_max_absolute_coordinate_delta': float(np.max(np.abs(delta))),
            'actual_max_cumulative_coordinate_delta': float(np.max(np.abs(cumulative))),
            'actual_rounded_gradient_dots': actual_dots.tolist(),
            'actual_rounded_minimum_normalized_margin': float(-actual_cosines.max()),
            'actual_candidate_equals_checkpoint': True, 'optimizer': None})
    need(len(gradients) == 256, 'Complete returned256 gradient NPZ audit required')
    # Saved terminal scalar failure is reported separately from independent pixels.
    banks = [read(PREFIX + 'banks/u' + str(u).zfill(4) + '/bank.json') for u in (1, 2)]
    failed_checks = [c for c in banks[1]['gate']['role_mean_nonincrease_checks'] if not c['pass']]
    need(banks[0]['gate']['pass'] is True and banks[1]['gate']['pass'] is False, 'Saved quality stop sequence differs')
    register(Path(__file__))
    for relative, stamp in consumed.items():
        need(pin(ROOT / relative) == stamp, 'Consumed source changed before/after: ' + relative)
    need(not any(n.split('.')[0] in {'torch', 'tensorflow', 'jax'} for n in sys.modules), 'Unexpected framework loaded')
    result = {'complete': True, 'saved_state_gradient_update_audit_pass': True,
        'protocol_sha256': PROTOCOL_SHA, 'original575_initializer_exact': True,
        'initial_matches_prior_role_gradient_original_u64_file_exact': True,
        'full575_canonical_sha256': {'initial': full_shas[0], 'failed_current_u2': full_shas[1]},
        'all539_frozen_bitwise_exact': True, 'frozen539_canonical_sha256': FROZEN_CANONICAL,
        'changed_head_tensors_initial_to_failed': changed_heads,
        'head_checkpoint_canonical_sha256': head_shas, 'RNG_checkpoint0_1_2_failed_byte_exact': True,
        'fixed_sampler_null_optimizer_receipts_pass': True, 'gradient_NPZ_files': 256,
        'all_gradient_schema_current_state_query_photograph_bindings_pass': True,
        'actual_two_committed_updates_reconstructed_exact': True, 'updates': results,
        'saved_scalar_stop': {'update1_pass': True, 'update2_pass': False,
            'inherited_gate_pass_at_update2': banks[1]['inherited']['pass'],
            'failed_added_checks': failed_checks, 'independent_pixel_audit_separate': True},
        'quality_stop_retained_no_promotion': True, 'pilot_screen_pass': False,
        'consumed_files': consumed, 'all_consumed_before_after_exact': True,
        'neural_framework_model_autograd_optimizer_parameter_VM_calls': 0,
        'audit_reconstructs_saved_array_arithmetic_only': True, 'model_qualified': False,
        'tensor_rows575': rows575, 'gradient_receipts': gradients}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'receipt_v1.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('consumed_files', 'tensor_rows575', 'gradient_receipts')}, indent=2))


if __name__ == '__main__':
    main()
