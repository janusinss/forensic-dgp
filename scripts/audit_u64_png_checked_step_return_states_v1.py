"""Independent saved-array audit of returned PNG-checked finite applications; no framework.

Reconstructs arithmetic from retained arrays only. It never instantiates a
model, applies weights, computes an actual gradient, or contacts a VM.
"""
import ast
import hashlib
import importlib.abc
import importlib.util
import os
import stat
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit/inventory_v1.json'
PROTOCOL = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm/protocol.json'
OUT = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_states'
PREFIX = 'outputs/u64_png_checked_step_v1/'
PROTOCOL_SHA = '7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60'
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


CAPS = (2e-5, 1.5e-5, 1e-5, 5e-6)
MEASUREMENT_PROTOCOL_SHA = "0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c"


def main():
    sys.meta_path.insert(0, NoFramework())
    consumed = {}
    def pin(path):
        path = Path(path).absolute()
        need(path.is_relative_to(ROOT), 'Confined local source required')
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not path.is_symlink()
             and not getattr(before, 'st_file_attributes', 0) & 1024, 'Ordinary single-link file required')
        for parent in path.parents:
            info = parent.lstat()
            need(stat.S_ISDIR(info.st_mode) and not parent.is_symlink()
                 and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary source ancestry required')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(1024**2), b''): digest.update(part)
        after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hash')
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size, 'sha256': digest.hexdigest()}
    def payload(row): return {key: row[key] for key in ('bytes', 'sha256')}
    def register(path, expected=None):
        path = Path(path).absolute(); actual = pin(path)
        if expected is not None: need(payload(actual) == payload(expected), 'File bytes differ: ' + str(path))
        need(actual['path'] not in consumed or consumed[actual['path']] == actual, 'Consumed file changed')
        consumed[actual['path']] = actual
        return path
    def localdoc(path, expected=None): return doc(register(path, expected).read_bytes())
    register(Path(__file__))
    protocol = localdoc(PROTOCOL)
    need(pin(PROTOCOL)['sha256'] == PROTOCOL_SHA, 'Exact sealed finite-step protocol required')
    inventory = localdoc(INVENTORY)
    need(inventory['complete'] is True and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Completed bound extraction required')
    evidence, pins = Path(inventory['evidence_root']), inventory['extracted_files']
    need(evidence == ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit/extracted', 'Exact extraction root')
    def bound(name, expected=None):
        need(name in pins and pins[name]['path'] == name, 'Missing/misnamed returned witness: ' + name)
        if expected is not None: need(pins[name] == expected, 'Receipt/inventory binding differs: ' + name)
        return register(evidence / name, pins[name])
    def read(name, expected=None): return doc(bound(name, expected).read_bytes())
    def load_heads(path, description=None):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            need(len(names) == len(set(names)) == 36 and set(names) == {name + '.npy' for name in SHAPES}, 'Exact36 NPZ member namespace')
            need(all(not info.is_dir() and info.file_size <= 4 * 1024**2 for info in archive.infolist()), 'Bounded ordinary NPZ entries')
        with np.load(path, allow_pickle=False) as loaded:
            need(set(loaded.files) == set(SHAPES), 'Exact sorted head coordinate schema')
            values = {name: loaded[name] for name in SHAPES}
        need(all(value.dtype == np.float32 and value.shape == SHAPES[name] and np.isfinite(value).all()
                 for name, value in values.items()), 'Finite exact FP32 head tensors required')
        digest, rows = canonical(values)
        if description is not None:
            need(description == {'allow_pickle': False, 'arrays': rows, 'canonical_sha256': digest}, 'Complete head description differs')
        return values, digest
    def heads(name, expected=None, description=None): return load_heads(bound(name, expected), description)
    def npy(name, shape):
        with bound(name).open('rb') as stream: value = np.load(stream, allow_pickle=False)
        need(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), 'Exact finite FP64 saved array')
        return value
    plan = protocol['plan']
    need(len(plan) == 445 and sum(row['kind'] == 'forward' for row in plan) == 436
         and sum(row['kind'] == 'parameter_application' for row in plan) == 9
         and not any(row['gradient_queries'] for row in plan), 'Exact436/9/0 finite plan')
    failure = read(PREFIX + 'failure.json')
    need(failure['quality_stop'] is True and failure['operation_outcome_unknown'] is False
         and failure['unknown_reservations'] == [] and PREFIX + 'result.json' not in pins,
         'Retained known scientific failure required, no success reclassification')
    terminal_slot = failure['active_slot']; applications = failure['counters']['parameter_applications']
    need((terminal_slot, applications) in ((445, 9), (388, 8)), 'Known final monitor or noeligible restored prefix required')
    def prefix_counts(slot):
        counters = dict.fromkeys(failure['counters'], 0)
        actions = {'trial_apply': 'trial_parameter_applications', 'trial_restore': 'original_restoration_applications',
                   'selected_commit': 'selected_parameter_commits'}
        for row in plan[:slot]:
            if row['kind'] == 'forward':
                counters['full_forward_attempts'] += 1; counters['full_forward_returns'] += 1; counters[row['category']] += 1
            else:
                counters['parameter_application_attempts'] += 1; counters['parameter_applications'] += 1
                counters[actions[row['action']]] += 1
        return counters
    need(failure['counters'] == prefix_counts(terminal_slot), 'Terminal counters differ from completed exact plan prefix')
    need(not any('/gradients_' in name or 'optimizer' in Path(name).name.lower() for name in pins), 'No new gradients or optimizer files allowed')
    preflight = read(PREFIX + 'preflight.json')
    need(preflight['optimizer'] is None and preflight['full575_canonical_sha256'] == INITIAL_CANONICAL
         and preflight['frozen539_canonical_sha256'] == FROZEN_CANONICAL, 'Exact original initializer/frozen/null optimizer')
    application_calls = [row for row in plan[:terminal_slot] if row['kind'] == 'parameter_application']
    checkpoint_rows, head_values, head_shas, resumes = {}, {}, {}, {}
    for index in range(applications + 1):
        stem = PREFIX + 'applications/a' + str(index).zfill(4)
        saved = read(stem + '/checkpoint.json')
        values, digest = heads(stem + '/heads.npz', saved['heads'], saved['head_description'])
        resume = read(stem + '/resume.json', saved['resume'])
        next_slot = 1 if index == 0 else application_calls[index - 1]['slot'] + 1
        need(saved['complete'] is True and saved['parameter_application_index'] == resume['parameter_application_index'] == index
             and saved['frozen539_canonical_sha256'] == resume['frozen539_canonical_sha256'] == FROZEN_CANONICAL,
             'Checkpoint index/frozen identity differs')
        need(resume['format'] == 'u64-png-checked-closed-application-state-v1' and resume['protocol_sha256'] == PROTOCOL_SHA
             and resume['optimizer'] is None and resume['optimizer_constructions'] == 0
             and resume['all_trial_restore_commit_parameter_applications'] == index
             and resume['counters'] == prefix_counts(next_slot - 1)
             and resume['parameter_names'] == list(SHAPES) and resume['head_canonical_sha256'] == digest
             and resume['initial_head_canonical_sha256'] == HEAD_CANONICAL,
             'Checkpoint closed application/protocol/null optimizer/source names differ')
        need(resume['sampler'] == {'fixed_plan': True, 'next_slot': next_slot, 'automatic_resume_permitted': False}, 'Exact fixed sampler')
        checkpoint_rows[index], head_values[index], head_shas[index], resumes[index] = saved, values, digest, resume
    need(head_shas[0] == HEAD_CANONICAL, 'Exact original trained-u64 head checkpoint required')
    rng = resumes[0]['RNG']
    need(all(row['RNG'] == rng for row in resumes.values()), 'Actual checkpoint RNG changed')
    need(set(rng) == {'Python', 'NumPy', 'CPU', 'CUDA'} and type(ast.literal_eval(rng['Python'])) is tuple
         and rng['NumPy'][0] == 'MT19937' and len(bytes.fromhex(rng['NumPy'][1])) == 2496
         and bytes.fromhex(rng['CPU']) and len(rng['CUDA']) == 1 and bytes.fromhex(rng['CUDA'][0]), 'Complete RNG representation required')
    failed_rng = read(failure['failed_RNG_sampler']['path'], failure['failed_RNG_sampler'])
    need(failed_rng['RNG'] == rng and failed_rng['optimizer'] is None
         and failed_rng['automatic_resume_permitted'] is False and failed_rng['active_application'] == applications
         and failed_rng['active_plan_slot'] == terminal_slot and failed_rng['counters'] == failure['counters']
         and failed_rng['operation_outcome_unknown'] is False, 'Retained failed RNG/sampler/null optimizer differs')
    need(failure['last_checkpoint'] == pins[PREFIX + 'applications/a' + str(applications).zfill(4) + '/checkpoint.json']
         and failure['failed_current_state'] == pins[PREFIX + 'full_states/failed_current.json'], 'Failure checkpoint/state links differ')
    # Full-state arrays are streamed tensor by tensor; no model or state object is built.
    states = [read(PREFIX + 'full_states/' + label + '.json') for label in ('initial', 'failed_current')]
    state_paths = [bound(row['state']['path'], row['state']) for row in states]
    need(payload(pin(state_paths[0])) == {'bytes': 377504068, 'sha256': INITIAL_FILE_SHA}, 'Original575 NPZ exact historical bytes')
    prior = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit/extracted/outputs/u64_role_gradients_v1/full_states/initial.npz'
    register(prior, {'bytes': 377504068, 'sha256': INITIAL_FILE_SHA})
    descriptions = [{row['name']: row for row in saved['description']['arrays']} for saved in states]
    need(all(len(rows) == 575 for rows in descriptions), 'Unique575 state descriptions required')
    full_digests, frozen_digests = [hashlib.sha256(), hashlib.sha256()], [hashlib.sha256(), hashlib.sha256()]
    rows575, changed_frozen, changed_heads = [], [], []
    for path in state_paths:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            need(len(names) == len(set(names)) == 575 and all(name.endswith('.npy') and '/' not in name and '\\' not in name for name in names), 'Flat unique575 NPZ namespace')
    with np.load(state_paths[0], allow_pickle=False) as initial, np.load(state_paths[1], allow_pickle=False) as current:
        need(len(initial.files) == len(current.files) == 575 and set(initial.files) == set(current.files), 'Complete same575 namespace')
        for name in sorted(initial.files):
            left, right = initial[name], current[name]
            need(left.dtype == right.dtype == np.float32 and left.shape == right.shape
                 and np.isfinite(left).all() and np.isfinite(right).all(), 'Exact finite575 current schema')
            before, after = canonical_update(full_digests[0], name, left), canonical_update(full_digests[1], name, right)
            need(descriptions[0][name] == before and descriptions[1][name] == after, 'Complete state description differs')
            same = left.tobytes() == right.tobytes()
            if name.startswith('heads.'):
                need(name in SHAPES and left.shape == SHAPES[name]
                     and left.tobytes() == head_values[0][name].tobytes()
                     and right.tobytes() == head_values[applications][name].tobytes(), 'Full/checkpoint head bits differ')
                if not same: changed_heads.append(name)
            else:
                canonical_update(frozen_digests[0], name, left); canonical_update(frozen_digests[1], name, right)
                if not same: changed_frozen.append(name)
            rows575.append({'name': name, 'shape': list(left.shape), 'initial_bytes_sha256': before['bytes_sha256'],
                            'failed_current_bytes_sha256': after['bytes_sha256'], 'exact_equal': same})
    full_shas = [digest.hexdigest() for digest in full_digests]
    need(not changed_frozen and len(rows575) - len(SHAPES) == 539 and full_shas[0] == INITIAL_CANONICAL
         and all(digest.hexdigest() == FROZEN_CANONICAL for digest in frozen_digests), 'Original/frozen full-state canonical identity differs')
    for index, saved in enumerate(states):
        need(saved['complete'] is True and saved['canonical_sha256'] == saved['description']['canonical_sha256'] == full_shas[index]
             and saved['head36_canonical_sha256'] == head_shas[(0, applications)[index]]
             and saved['parameter_application_index'] == (0, applications)[index] and saved['all539_frozen_exact'] is True,
             'State description/current checkpoint canonical differs')
    # Authenticate exactly the retained original-u64 RAW128 files through the sealed dependency map.
    preparation = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation'
    mproof = localdoc(PROTOCOL.parent / 'evidence/local_measurement_readback.json')
    mbindings = {row['measurement_relative']: row['actual_local'] for row in mproof['files']}
    manifest = protocol['retained_measurement']['files']
    mprotocol_path = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm/protocol.json'
    measurement = localdoc(mprotocol_path)
    need(pin(mprotocol_path)['sha256'] == MEASUREMENT_PROTOCOL_SHA and measurement['selection'] == protocol['selection'], 'Exact retained original measurement selection')
    def mbound(name, expected=None):
        need(name in manifest and name in mbindings, 'Unbound retained measurement source')
        binding = mbindings[name]
        if expected is not None: need(expected == {'path': name, **manifest[name]}, 'Retained receipt/manifest differs')
        path = ROOT / binding['path']
        direct = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm' / name
        alias = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/extracted' / name
        need(path in (direct, alias) and (path == direct or name not in measurement['files']), 'Only explicit retained runtime extraction alias permitted')
        register(path, manifest[name]); need(pin(path) == binding, 'Exact retained source proof differs')
        return path
    def mread(name): return doc(mbound(name).read_bytes())
    pieces = {role: {query: [] for query in QUERIES} for role in ROLES}
    gradient_rows, dependencies = [], []
    for slot, source in enumerate(protocol['selection']['optimization'], 9):
        stem = 'outputs/u64_png_surrogate_measurement_v1/calls/' + str(slot).zfill(4)
        receipt, output = mread(stem + '/gradients.json'), mread(stem + '/record.json')
        call = measurement['plan'][slot - 1]
        need(receipt['complete'] is True and receipt['call'] == output['call'] == call
             and all(call[key] == source[key] for key in ('id', 'reference_id', 'role'))
             and receipt['metric_region'] == ('central128' if source['role'] == ROLES[0] else 'whole')
             and output['input'] == source['input'] and output['target'] == source['target']
             and output['split'] == 'TRAIN' and source['optimization_permitted'] is True
             and receipt['output_record'] == {'path': stem + '/record.json', **manifest[stem + '/record.json']}, 'Retained current-original photographic query provenance differs')
        for raw_query, gate_query in (('raw_mae', 'gate_mae'), ('raw_adjacent', 'gate_adjacent')):
            saved = receipt['gradients'][raw_query]
            values, digest = load_heads(mbound(saved['file']['path'], saved['file']), saved['description'])
            zeros = [name for name in SHAPES if not np.any(values[name])]
            need(saved['canonical_sha256'] == digest and saved['head_state_canonical_sha256'] == HEAD_CANONICAL
                 and saved['unused_names_explicit_zero'] == [] and saved['all_zero_names'] == zeros == [], 'Connected original-state zero/unused gradient metadata differs')
            vector = flat(values)
            need(vector.shape == (168960,) and norm(vector) > 0 and math.isfinite(saved['scalar']), 'Complete nonzero finite retained gradient')
            pieces[source['role']][gate_query].append(vector)
            dependencies.append({'case_id': source['id'], 'query': raw_query, 'file': saved['file'], 'canonical_sha256': digest})
            gradient_rows.append({'slot': slot, 'id': source['id'], 'role': source['role'], 'query': raw_query,
                'canonical_sha256': digest, 'head_canonical_sha256': HEAD_CANONICAL, 'norm': norm(vector), 'unused_names': [], 'all_zero_names': []})
    need(len(gradient_rows) == 128 and all(len(pieces[role][query]) == 16 for role in ROLES for query in QUERIES), 'All16photo/fourrole/twoRAW query vectors')
    pooled_dict = {role: {query: np.mean(np.stack(parts), axis=0, dtype=np.float64)
                   for query, parts in bank.items()} for role, bank in pieces.items()}
    pooled = np.stack([pooled_dict[role][query] for role in ROLES for query in QUERIES])
    saved_pooled = npy(PREFIX + 'direction/pooled_RAW8.npy', (8, 168960))
    need(np.array_equal(pooled, saved_pooled), 'Saved eight pooled gradients are not exact FP64 means16')
    dependency_receipt = read(PREFIX + 'direction/retained_RAW_dependencies.json')
    need(dependency_receipt['complete'] is True and dependency_receipt['queries'] == dependencies
         and dependency_receipt['new_autograd_queries'] == 0 and dependency_receipt['all_current_original_state'] is True
         and dependency_receipt['finite_PNG_gain_certified'] is False, 'Actual direction retained source receipt differs')
    independent = independent_faces(pooled)
    direction = npy(PREFIX + 'direction/direction.npy', (168960,))
    solver_receipt = read(PREFIX + 'direction/solver.json')
    close(direction, independent['direction'], 'Direction differs from independent255face numerical solve', rtol=1e-6, atol=1e-9)
    for field, value in (('unit_gram', independent['gram']), ('simplex_coefficients', independent['weights']),
        ('input_gradient_norms', independent['lengths']), ('combination_norm', independent['combination_norm']),
        ('minimum_normalized_descent_margin', independent['margin'])):
        close(solver_receipt[field], value, 'Saved solver certificate differs: ' + field, rtol=1e-6, atol=1e-9)
    solver_path = PROTOCOL.parent / 'research/codeformer_u64_gate_aligned_v1/solver.py'
    register(solver_path, protocol['files']['research/codeformer_u64_gate_aligned_v1/solver.py'])
    spec = importlib.util.spec_from_file_location('exact_inert_eight_gate_solver', solver_path)
    solver = importlib.util.module_from_spec(spec); spec.loader.exec_module(solver)
    exact_solved = solver.solve_eight_pooled_gates(pooled_dict)
    close(direction, exact_solved['direction'], 'Saved direction differs from exact reviewed pure solver', rtol=1e-10, atol=1e-12)
    raw_dots = np.array([inner(row, direction) for row in pooled])
    unit_dots = raw_dots / independent['lengths']
    close(solver_receipt['original_gradient_dot_direction'], raw_dots, 'Actual RAW direction dots differ')
    close(solver_receipt['unit_gradient_dot_direction'], unit_dots, 'Normalized direction dots differ')
    need(np.all(raw_dots < 0) and -unit_dots.max() > 1e-6 and solver_receipt['face_count'] == 255
         and solver_receipt['all_actual_derivatives_strict_negative'] is True, 'Actual eight-gate strict certificate absent')
    # Reconstruct four independent-original FP32 trial values without applying any model state.
    original = flat(head_values[0]); trials, trial_shas, certificates, trial_results = {}, {}, {}, []
    for index, cap in enumerate(CAPS):
        alpha = .99 * cap / float(np.max(np.abs(direction)))
        expected = (original + alpha * direction).astype(np.float32).astype(np.float64)
        proposed, digest = heads(PREFIX + 'trials/' + str(index) + '/proposed_heads.npz')
        actual = flat(proposed); delta = actual - original
        saved_delta = npy(PREFIX + 'trials/' + str(index) + '/rounded_delta.npy', (168960,))
        need(np.array_equal(actual, expected) and np.array_equal(delta, saved_delta), 'Independent original/.99 FP32 proposal or delta differs')
        dots = np.array([inner(row, delta) for row in pooled]); peak = float(np.max(np.abs(delta)))
        need(peak <= cap <= 2.5e-5 and peak <= 1e-4 and np.all(dots < 0), 'Actual rounded trial cap/cumulative/all8 RAW sign failure')
        cert = {'pass': True, 'registered_bound': float(cap), 'fixed_numerical_headroom': .99,
            'alpha': alpha, 'actual_maxabs_delta_from_original': peak, 'original_per_step_cap': 2.5e-5,
            'original_cumulative_cap': 1e-4, 'all8_actual_RAW_dots': dots.tolist(),
            'independent_original_initialization': True, 'inputs_unchanged': True, 'parameter_applied': False,
            'metric_adaptive_scalar': False, 'retry_or_fallback': False, 'finite_PNG_gain_certified': False,
            'restore_or_nine_application_accounting_proved': False}
        trials[index], trial_shas[index], certificates[index] = proposed, digest, cert
        trial_results.append({'trial_index': index, 'registered_cap': cap, 'fixed_alpha': alpha,
            'actual_FP32_maxabs_delta': peak, 'all8_actual_RAW_dots': dots.tolist(), 'head_canonical_sha256': digest,
            'proposal_independent_original_and_saved_delta_byte_exact': True})
    choice = read(PREFIX + 'selection.json')
    selected_index = choice['selected_index']
    need(choice['complete'] is True and choice['all_four_trials_evaluated'] is True
         and choice['candidate_registry'] == list(CAPS) and choice['only_optimization_records_supplied'] is True
         and choice['all_four_original_restores_completed'] is True and choice['monitor_records_supplied'] == 0
         and selected_index == next((row['candidate_index'] for row in choice['trial_results'] if row['eligible']), None), 'Fixed complete four-trial TRAIN-only selection receipt differs')
    need((applications == 9 and selected_index in range(4)) or (applications == 8 and selected_index is None), 'Selected commit/noeligible branch differs')
    application_results = []
    for index, call in enumerate(application_calls, 1):
        stem = PREFIX + 'applications/a' + str(index).zfill(4)
        precommit, commit = read(stem + '/precommit.json'), read(stem + '/commit.json')
        proposed, digest = heads(stem + '/precommit_heads.npz', precommit['candidate'], precommit['description'])
        action, trial = call['action'], call['trial_index']
        expected_values = head_values[0] if action == 'trial_restore' else trials[trial if action == 'trial_apply' else selected_index]
        need(all(proposed[name].tobytes() == expected_values[name].tobytes() == head_values[index][name].tobytes() for name in SHAPES), 'Application precommit/copied checkpoint differs from actual expected trial/restore')
        delta, cumulative = flat(proposed) - flat(head_values[index - 1]), flat(proposed) - original
        peak, cumulative_peak = float(np.max(np.abs(delta))), float(np.max(np.abs(cumulative)))
        need(peak <= 2.5e-5 and cumulative_peak <= 1e-4, 'Actual application coordinate/cumulative cap failed')
        dots = [inner(row, delta) for row in pooled]
        if action == 'trial_restore':
            certificate = {'restore_reversal_not_descent': True, 'expected_after': HEAD_CANONICAL}
            need(digest == HEAD_CANONICAL and trial in range(4), 'Explicit restore must exactly recover original heads')
        else:
            certificate = certificates[trial if action == 'trial_apply' else selected_index]
            need(all(value < 0 for value in dots), 'Actual nonrestore displacement lacks all8 strict RAW descent signs')
        need(precommit['complete'] is True and precommit['call'] == commit['call'] == call
             and precommit['before'] == commit['actual_before'] == head_shas[index - 1]
             and commit['actual_after'] == digest == head_shas[index] and commit['candidate'] == precommit['candidate']
             and commit['all539_frozen_exact'] is True and commit['selected_training_updates'] == int(action == 'selected_commit')
             and commit['counters'] == resumes[index]['counters'] == prefix_counts(call['slot'])
             and precommit['restore_is_explicit_reversal'] == (action == 'trial_restore'), 'Actual precommit/commit/checkpoint chain differs')
        close(precommit['actual_maxabs_step'], peak, 'Actual application step scalar differs')
        close(precommit['actual_maxabs_cumulative'], cumulative_peak, 'Actual application cumulative scalar differs')
        for key, value in certificate.items():
            actual = precommit['certificate'][key]
            if type(value) in (float, list): close(actual, value, 'Application fixed numerical certificate differs: ' + key, rtol=1e-12, atol=1e-14)
            else: need(actual == value, 'Application certificate bool/value differs: ' + key)
        application_results.append({'index': index, 'plan_slot': call['slot'], 'action': action, 'trial_index': trial,
            'before_head_canonical_sha256': head_shas[index - 1], 'after_head_canonical_sha256': digest,
            'actual_maxabs_step': peak, 'actual_maxabs_cumulative': cumulative_peak,
            'all8_actual_RAW_displacement_dots': dots, 'restore_is_reversal_not_descent': action == 'trial_restore',
            'actual_precommit_checkpoint_bytes_exact': True})
    # Every saved forward is bound to the expected actual checkpoint state.
    active = 0; forward_rows = []
    for call in plan[:terminal_slot]:
        if call['kind'] == 'parameter_application': active += 1; continue
        record = read(PREFIX + 'calls/' + str(call['slot']).zfill(4) + '/record.json')
        need(record['complete'] is True and record['call'] == call and record['head_before_sha256'] == head_shas[active]
             and record['parameter_application_index'] == active and record['frozen539_canonical_sha256'] == FROZEN_CANONICAL,
             'Saved forward actual application-state provenance differs')
        forward_rows.append({'slot': call['slot'], 'id': call['id'], 'parameter_application_index': active,
                             'head_canonical_sha256': head_shas[active]})
    journal = [doc(line) for line in bound(PREFIX + 'events.jsonl').read_bytes().splitlines()]
    returned_apps = [row for row in journal if row['kind'] == 'parameter_application_returned']
    need(len(returned_apps) == applications, 'Exactly all application returns required')
    for index, (event, call) in enumerate(zip(returned_apps, application_calls), 1):
        need(event['id'] == call['id'] and event['action'] == call['action'] and event['trial_index'] == call['trial_index']
             and event['plan_slot'] == call['slot'] and event['synchronized_return'] is True
             and event['actual_before_head_canonical_sha256'] == head_shas[index - 1]
             and event['actual_after_head_canonical_sha256'] == head_shas[index]
             and event['counters'] == prefix_counts(call['slot']), 'Durable actual application event chain differs')
    final_bank = read(PREFIX + 'banks/selected_quality.json') if applications == 9 else None
    if final_bank is not None: need(final_bank['gate']['pass'] is False and final_bank['monitor_fallback_forbidden'] is True, 'Retained terminal quality failure/no fallback required')
    for relative, value in consumed.items(): need(pin(ROOT / relative) == value, 'Consumed bytes changed before/after: ' + relative)
    need(not any(name.split('.')[0] in {'torch', 'tensorflow', 'jax'} for name in sys.modules), 'Unexpected framework loaded')
    result = {'complete': True, 'saved_state_direction_application_audit_pass': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': PROTOCOL_SHA,
        'unresolved_blockers': [], 'original575_initializer_exact': True, 'initial575_NPZ_prior_exact': True,
        'full575_canonical_sha256': {'initial': full_shas[0], 'failed_current': full_shas[1]},
        'all539_frozen_bitwise_exact': True, 'frozen539_canonical_sha256': FROZEN_CANONICAL,
        'changed_head_tensors_initial_to_failed': changed_heads, 'head_checkpoint_canonical_sha256': head_shas,
        'RNG_all10_checkpoints_and_failed_exact': True, 'fixed_sampler_null_optimizer_receipts_pass': True,
        'retained_original_RAW_gradient_NPZ_files': 128, 'new_gradient_query_files': 0,
        'all_gradient_schema_state_query_photograph_bindings_pass': True, 'pooled_RAW8_FP64_mean16_byte_exact': True,
        'direction': {'independent_255face_L2_difference': norm(direction - independent['direction']),
            'exact_reviewed_pure_solver_L2_difference': norm(direction - exact_solved['direction']),
            'exact_reviewed_pure_solver_bitwise_equal': np.array_equal(direction, exact_solved['direction']),
            'minimum_normalized_margin': float(-unit_dots.max()), 'all8_RAW_dots': raw_dots.tolist(),
            'simplex_coefficients': solver_receipt['simplex_coefficients']},
        'all_four_independent_original_FP32_proposals_exact': True, 'trials': trial_results,
        'explicit_parameter_applications': applications, 'trial_applies': 4, 'original_restores': 4,
        'selected_commits': int(applications == 9), 'selected_candidate_index': selected_index,
        'all_actual_application_prepost_state_chains_exact': True, 'applications': application_results,
        'all_forward_current_checkpoint_bindings': forward_rows,
        'terminal': {'quality_stop': True, 'active_slot': terminal_slot, 'error': failure['error'],
            'operation_outcome_unknown': False, 'counters': failure['counters'], 'pilot_quality_pass': False,
            'selected_monitor_gate_pass': None if final_bank is None else final_bank['gate']['pass']},
        'quality_stop_retained_no_promotion': True, 'model_qualified': False,
        'tensor_rows575': rows575, 'retained_gradient_rows': gradient_rows,
        'consumed_files': list(consumed.values()), 'all_consumed_before_after_exact': True,
        'audit_reconstructs_saved_array_arithmetic_only': True,
        'neural_framework_model_autograd_optimizer_parameter_VM_calls': 0}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'receipt_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'receipt': pin(target), 'states_pass': True, 'retained_RAW_NPZ': 128,
        'applications': applications, 'selected_index': selected_index, 'direction': result['direction'],
        'trials': trial_results, 'terminal': result['terminal'], 'all_consumed_before_after_exact': True}, indent=2))


if __name__ == '__main__':
    main()
