"""Manual L4 normal-tone diagnostic: cached gradients, eight disposable Adam steps.

Import and static CLI modes use only the standard library. Actual model,
gradient and optimizer execution is guarded by the exact manual VM root.
No training updates are committed and no model checkpoint is written.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

try:
    import resource
except ImportError:
    resource = None


VM_NAME = 'cctv_codeformer_normal_tone_v1_vm'
COMPONENTS = ('token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
              'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl')
BRANCHES = ('baseline', 'full', 'omit_normal_coarse',
            'center_normal_luminance', 'momentum_only')
BASES = ('all_coarse', 'normal_coarse', 'dark_coarse',
         'centered_normal', 'canonical_hybrid')
COUNTS = {'candidate_source_forwards': 32, 'candidate_probe_forwards': 160,
          'restoration_repeat_forwards': 32, 'coarse_gradient_queries': 32,
          'centered_normal_gradient_queries': 16, 'optimizer_step_attempts': 8,
          'disposable_optimizer_steps': 8, 'committed_training_updates': 0,
          'backward_calls': 0}
LUMINANCE = (.2126, .7152, .0722)
OLD_OUTPUTS = 'outputs/loss_response_v1'
STOPPED_SHA = '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DiagnosticStop(RuntimeError):
    """Stop retaining all already written evidence and completed real-call counts."""


def call_totals(counts):
    return {'neural_forwards': sum(counts[key] for key in
                ('candidate_source_forwards', 'candidate_probe_forwards', 'restoration_repeat_forwards')),
            'gradient_queries': counts['coarse_gradient_queries'] + counts['centered_normal_gradient_queries'],
            'actual_completed_disposable_Adam_steps': counts['disposable_optimizer_steps'],
            'Adam_step_attempt_reservations': counts['optimizer_step_attempts'],
            'committed_training_updates': 0, 'backward_calls': 0}


def transfer_cli_summary(transfer):
    """Keep supervisor stdout bounded; full verification stays in memory."""
    summary = {key: transfer[key] for key in ('format', 'complete', 'protocol_sha256',
        'cases_verified', 'readonly', 'neural_calls', 'gradient_queries', 'optimizer_steps')}
    summary['static_files_verified_including_protocol'] = len(transfer['static_files']) + 1
    summary['dependencies_verified'] = dict(transfer['dependencies_verified'])
    require(len(json.dumps(summary).encode('utf-8')) < 4096, 'Compact preflight receipt exceeds4096bytes')
    return summary


def array_digest(array, np):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def accumulate(destination, current, names, np):
    """Add each per-source weighted /16 FP32 vector in its canonical order."""
    for name in names:
        value = current[name]
        if value is None:
            continue
        require(value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid source gradient')
        if destination[name] is None:
            destination[name] = value.copy()
        else:
            require(destination[name].shape == value.shape, 'Gradient accumulation geometry differs')
            np.add(destination[name], value, out=destination[name])


def compose(components, coarse, names, np, *, momentum_only=False):
    """Use the frozen seven-slot order, replacing only the coarse slot."""
    result = {}
    for name in names:
        value = None
        for term in COMPONENTS:
            item = (coarse if term == 'coarse_rgb_l1' else components[term])[name]
            if item is None:
                continue
            if value is None:
                value = item.copy()
            else:
                np.add(value, item, out=value)
        require(value is not None and value.dtype == np.float32 and np.isfinite(value).all(),
                'Every disposable branch must assign all350 finite FP32 gradients')
        result[name] = np.zeros_like(value) if momentum_only else value
    return result


def exact_arrays(actual, expected, names, np):
    """Digest equality is mandatory; numerical discrepancy is diagnostic only."""
    rows, maximum, squared = {}, 0.0, 0.0
    for name in names:
        a, b = actual[name], expected[name]
        require((a is None) == (b is None), 'Cached/fresh None topology differs: ' + name)
        if a is None:
            rows[name] = {'is_none': True, 'exact': True}
            continue
        require(a.shape == b.shape and a.dtype == b.dtype == np.float32, 'Cached/fresh gradient shape differs')
        difference = a.astype(np.float64) - b.astype(np.float64)
        maximum = max(maximum, float(np.abs(difference).max()))
        squared += float(np.square(difference).sum(dtype=np.float64))
        digest_a, digest_b = array_digest(a, np), array_digest(b, np)
        rows[name] = {'is_none': False, 'exact': digest_a == digest_b,
                      'fresh_sha256': digest_a, 'cached_sha256': digest_b}
    return {'pass': all(row['exact'] for row in rows.values()), 'parameters': rows,
            'maximum_absolute': maximum, 'difference_l2': math.sqrt(squared),
            'accepted_maximum_error': 0.0, 'digest_equality_required': True}


def coarse_terms(prediction, clean512, torch):
    """One native unclipped RGB grid; luminance mean remains in the graph."""
    functional = torch.nn.functional
    rgb = (prediction.float() + 1) / 2
    target_grid = functional.avg_pool2d(clean512.float(), 32, 32)
    predicted_grid = functional.avg_pool2d(rgb, 32, 32)
    error = predicted_grid - target_grid
    weights = error.new_tensor(LUMINANCE).view(1, 3, 1, 1)
    mean_luminance = (error * weights).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    ordinary = functional.l1_loss(predicted_grid, target_grid)
    centered = (error - mean_luminance).abs().mean()
    require(error.shape == (1, 3, 16, 16) and mean_luminance.shape == (1, 1, 1, 1),
            'Per-observation native coarse geometry differs')
    require(mean_luminance.requires_grad and torch.isfinite(ordinary).item() and
            torch.isfinite(centered).item(), 'Centered luminance detached or nonfinite')
    return ordinary, centered, mean_luminance


@contextmanager
def pid_lock(root, protocol_sha, write_json):
    path = Path(root) / 'normal_tone_v1.lock'
    payload = {'pid': os.getpid(), 'host': socket.gethostname(), 'protocol_sha256': protocol_sha,
               'UTC': datetime.now(timezone.utc).isoformat()}
    write_json(path, payload)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) == payload:
            path.unlink()


class Budget:
    """Prospective call/write reservations with measured process/GPU/disk peaks."""
    def __init__(self, root, out, limits, torch, request, helper, started):
        self.root, self.out, self.limits = Path(root), Path(out), limits
        self.torch, self.request, self.helper = torch, request, helper
        self.started = started
        self.counts = dict.fromkeys(COUNTS, 0)
        self.actual_steps, self.optimizer_inflight = 0, False
        self.phase, self.phase_started = 'initialization', started
        self.elapsed = dict.fromkeys(('initialization', 'source', 'probe'), 0.0)
        self.event = self.payload_bytes = self.metadata_bytes = self.regular_files = 0
        self.samples = 0
        self.peak_RSS_bytes = self.peak_CUDA_reserved_bytes = self.peak_CUDA_allocated_bytes = 0
        self.minimum_free_bytes = shutil.disk_usage(root).free
        self.last_measurement = None
        self.sample()

    def sample(self):
        free = shutil.disk_usage(self.root).free
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 if resource is not None else 0
        reserved = max(self.torch.cuda.memory_reserved(), self.torch.cuda.max_memory_reserved())
        allocated = max(self.torch.cuda.memory_allocated(), self.torch.cuda.max_memory_allocated())
        self.minimum_free_bytes = min(self.minimum_free_bytes, free)
        self.peak_RSS_bytes = max(self.peak_RSS_bytes, rss)
        self.peak_CUDA_reserved_bytes = max(self.peak_CUDA_reserved_bytes, reserved)
        self.peak_CUDA_allocated_bytes = max(self.peak_CUDA_allocated_bytes, allocated)
        self.samples += 1
        self.last_measurement = {'seconds': time.monotonic() - self.started,
            'RSS_high_water_bytes': rss, 'CUDA_reserved_peak_bytes': reserved,
            'CUDA_allocated_peak_bytes': allocated, 'free_bytes': free}

    def measurements(self):
        return {'measurement_samples': self.samples, 'measured_peak_RSS_bytes': self.peak_RSS_bytes,
            'measured_peak_CUDA_reserved_bytes': self.peak_CUDA_reserved_bytes,
            'measured_peak_CUDA_allocated_bytes': self.peak_CUDA_allocated_bytes,
            'measured_minimum_sampled_free_bytes': self.minimum_free_bytes,
            'free_disk_measurement_path': str(self.root),
            'free_disk_scope': 'Worker root filesystem only; HOME archive has separate transport guards',
            'free_disk_minimum_is_sampled_not_continuous': True,
            'RSS_source': 'Linux getrusage(RUSAGE_SELF).ru_maxrss *1024 high water',
            'CUDA_source': 'PyTorch current and process-lifetime allocator peak counters',
            'last_measurement': self.last_measurement}

    def timings(self):
        result = dict(self.elapsed)
        result[self.phase] += time.monotonic() - self.phase_started
        return result

    def set_phase(self, phase):
        now = time.monotonic()
        self.elapsed[self.phase] += now - self.phase_started
        self.phase, self.phase_started = phase, now
        self.check()

    def check(self):
        self.sample()
        if self.request['signal'] is not None:
            raise DiagnosticStop('Manual signal requested finite diagnostic stop')
        if time.monotonic() - self.started > self.limits['worker_seconds']:
            raise DiagnosticStop('Worker wall-time cap reached')
        if any(seconds > self.limits[name + '_seconds'] for name, seconds in self.timings().items()):
            raise DiagnosticStop('Cumulative diagnostic phase time cap reached')
        if self.minimum_free_bytes < self.limits['reserve_free_bytes']:
            raise DiagnosticStop('Declared4GiB free disk reserve reached')
        if self.peak_RSS_bytes > self.limits['RSS_bytes']:
            raise DiagnosticStop('Declared8GiB process RSS bound reached')
        if self.peak_CUDA_reserved_bytes > self.limits['CUDA_reserved_bytes']:
            raise DiagnosticStop('Declared22GiB CUDA reserved bound reached')
        if self.payload_bytes > self.limits['result_payload_bytes'] or \
                self.metadata_bytes > self.limits['metadata_bytes'] or \
                self.regular_files > self.limits['regular_file_cap']:
            raise DiagnosticStop('Declared output byte/metadata/file bound reached')

    def reserve(self, name, context=None):
        self.check()
        require(name in COUNTS and self.counts[name] < COUNTS[name], 'Finite call cap reached: ' + name)
        self.counts[name] += 1
        self.event += 1
        self.json(self.out / 'call_trace' / ('%05d.json' % self.event),
            {'event': self.event, 'before_call': name, 'context': context, 'counts': dict(self.counts),
             'actual_completed_disposable_Adam_steps': self.actual_steps,
             'committed_training_updates': 0, 'backward_calls': 0,
             'seconds': time.monotonic() - self.started, 'resources': self.measurements()})

    def reserve_write(self, estimate):
        self.check()
        require(type(estimate) is int and estimate >= 0, 'Invalid prospective write estimate')
        require(self.regular_files < self.limits['regular_file_cap'], 'Prospective file cap reached')
        require(self.payload_bytes + estimate <= self.limits['result_payload_bytes'] - 1024**2,
                'Prospective output exceeds2GiB payload cap')
        require(shutil.disk_usage(self.root).free >= estimate + self.limits['reserve_free_bytes'],
                'Prospective write consumes declared disk reserve')

    def json(self, path, value):
        payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
        require(self.metadata_bytes + len(payload) <= self.limits['metadata_bytes'] - 65536,
                'Prospective metadata cap reached')
        self.reserve_write(len(payload))
        self.helper.atomic_json(path, value)
        self.payload_bytes += Path(path).stat().st_size
        self.metadata_bytes += Path(path).stat().st_size
        self.regular_files += 1
        self.sample()

    def array(self, path, array, np):
        require(array.dtype.kind != 'O' and np.isfinite(array).all(), 'Invalid output array')
        require(array.nbytes + 1024 <= self.limits['array_file_bytes'], 'Single array cap exceeded')
        self.reserve_write(array.nbytes + 1024)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), 'Retain existing output array')
        temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
        with temporary.open('xb') as stream:
            np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        require(temporary.stat().st_size <= array.nbytes + 1024, 'Array header cap exceeded')
        os.replace(temporary, path)
        self.payload_bytes += path.stat().st_size; self.regular_files += 1
        self.sample()
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': self.helper.sha256(path),
                'bytes': path.stat().st_size, 'shape': list(array.shape), 'dtype': str(array.dtype),
                'array_sha256': array_digest(array, np)}

    def png(self, path, array, Image):
        require(array.dtype.name == 'uint8' and array.shape == (256, 256, 3), 'Invalid PNG geometry')
        cap = array.nbytes + 4096
        self.reserve_write(cap)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), 'Retain existing PNG output')
        temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
        with temporary.open('xb') as stream:
            Image.fromarray(array).save(stream, format='PNG')
            stream.flush(); os.fsync(stream.fileno())
        require(temporary.stat().st_size <= cap, 'PNG byte cap exceeded')
        os.replace(temporary, path)
        self.payload_bytes += path.stat().st_size; self.regular_files += 1
        self.sample()
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': self.helper.sha256(path),
                'bytes': path.stat().st_size}


def dependency_path(parents, protocol, alias, relative, contract, *, verify=True):
    require(alias in parents and relative in protocol['dependencies'][alias],
            'Undeclared read-only dependency: ' + alias + ':' + relative)
    path = contract.safe_path(parents[alias], relative)
    record = protocol['dependencies'][alias][relative]
    require(path.stat().st_size == record['bytes'], 'Dependency size changed: ' + relative)
    if verify:
        require(contract.sha256(path) == record['sha256'], 'Dependency bytes changed: ' + relative)
    return path


def old_json(parents, protocol, relative, contract):
    return contract.read_json(dependency_path(parents, protocol, 'loss_response', relative, contract))


def load_cases(root, parents, protocol, contract, data, np, Image, budget):
    document = contract.read_json(contract.safe_path(root, protocol['cases_file']))
    contract.verify_cases(document)
    cases = document['cases']
    adain = parents['adain']
    reference_path = dependency_path(parents, protocol, 'adain', 'references.json', contract)
    references = data.load_reference_manifest(adain, 'train', reference_path,
                                             expected_manifest_sha256=contract.sha256(reference_path))
    by_id, arrays = {reference.id: reference for reference in references}, {}
    for update in (1, 6, 15, 24):
        group = [case for case in cases if case['update'] == update]
        require(group[0]['reference_id'] == group[1]['reference_id'] and
                group[2]['reference_id'] == group[3]['reference_id'], 'Matched reference pair differs')
        pairs = data.make_training_update(by_id[group[0]['reference_id']], by_id[group[2]['reference_id']], update)
        for case, pair in zip(group, pairs):
            budget.check()
            record = case['record']
            source_path = dependency_path(parents, protocol, 'adain', case['source_path'], contract)
            input_path = dependency_path(parents, protocol, 'adain', case['input_path'], contract)
            pair_path = dependency_path(parents, protocol, 'adain', case['pair_record_path'], contract)
            # Emit the actual generated record before testing equality: a mismatch
            # remains reconstructible alongside all already completed case receipts.
            parity = {'case': case['id'], 'update': update, 'slot': case['slot'],
                'reference_id': case['reference_id'], 'generated_record': pair.record,
                'expected_record': record, 'generated_parameters': pair.parameters,
                'source_file_sha256': contract.sha256(source_path),
                'saved_input_file_sha256': contract.sha256(input_path),
                'generated_input_float32_sha256': array_digest(pair.input256, np),
                'generated_clean512_float32_sha256': array_digest(pair.clean512, np),
                'pass': False}
            keys = ('input256_rgb8_sha256', 'input256_float32_sha256', 'clean512_float32_sha256',
                    'clean512_rgb8_sha256', 'source_sha256', 'reference_id', 'update', 'slot')
            with Image.open(input_path) as image:
                require(image.mode == 'RGB' and image.size == (256, 256), 'Saved input geometry differs')
                saved = np.asarray(image, dtype=np.float32) / np.float32(255)
            with Image.open(source_path) as image:
                require(image.mode == 'RGB' and image.size == (1024, 1024), 'Source geometry differs')
                target = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / np.float32(255)
            target_prefix = OLD_OUTPUTS + '/targets/' + case['reference_id'] + '/'
            retained_target_path = dependency_path(parents, protocol, 'loss_response',
                                                   target_prefix + 'clean256.npy', contract)
            retained_target_record = old_json(parents, protocol, target_prefix + 'parity.json', contract)
            retained_target = np.load(retained_target_path, allow_pickle=False)
            target_digest, retained_target_digest = array_digest(target, np), array_digest(retained_target, np)
            checks = {'record_fields': all(record[key] == pair.record[key] for key in keys),
                'parameters': case['parameters'] == pair.parameters,
                'profile': case['profile'] == pair.record['profile'],
                'source_join': case['source_path'] == record['source_path'] == pair.record['source_path'],
                'source_sha256': parity['source_file_sha256'] == record['source_sha256'],
                'saved_input_exact': np.array_equal(saved, pair.input256),
                'saved_pair_input_hash': contract.read_json(pair_path)['input256_float32_sha256'] ==
                                         record['input256_float32_sha256'],
                'target256_metadata': retained_target_record['pass'] is True and
                    retained_target_record['reference_id'] == case['reference_id'] and
                    retained_target_record['clean256_policy'] == 'TRAIN source RGB1024 Pillow LANCZOS256; metric-only',
                'target256_shape_dtype': retained_target.shape == target.shape == (256, 256, 3) and
                                         retained_target.dtype == target.dtype == np.float32,
                'target256_exact': np.array_equal(target, retained_target) and
                    target_digest == retained_target_digest == retained_target_record['clean256_array_sha256']}
            parity.update({'checks': checks, 'pass': all(checks.values()),
                'generated_target256_float32_sha256': target_digest,
                'retained_target256_float32_sha256': retained_target_digest,
                'retained_target256_path': target_prefix + 'clean256.npy',
                'retained_target256_parity_path': target_prefix + 'parity.json',
                'target256_role': 'Metric-only fixed TRAIN target; not candidate supervision or inference correction'})
            budget.json(budget.out / 'camera_parity' / (case['id'] + '.json'), parity)
            require(parity['pass'], 'Canonical TRAIN regeneration differs: ' + case['id'])
            arrays[case['id']] = {'input': pair.input256, 'clean512': pair.clean512, 'target256': target}
    require(len(arrays) == 16, 'Exactly16 canonical source/probe inputs required')
    return cases, arrays


def load_cached_components(label, names, parameters, parents, protocol, contract, helper, np):
    components, bindings = {}, {}
    expected_counts = {'token_ce': 278, 'latent_mse': 164, 'pixel_l1_rgb': 72,
        'lpips_vgg': 72, 'coarse_rgb_l1': 72, 'behavior_rgb_l1': 72, 'behavior_token_kl': 278}
    for term in COMPONENTS:
        relative = OLD_OUTPUTS + '/' + label + '/gradients/' + term + '/manifest.json'
        declaration = protocol['cached_basis'][label][term]
        require(declaration == {'root': 'loss_response', 'manifest_path': relative}, 'Cached basis path differs')
        manifest = old_json(parents, protocol, relative, contract)
        require(manifest['component'] == term and manifest['weighted'] is True and
                manifest['source_exposures'] == 16 and manifest['aggregation_divisor'] == 16 and
                manifest['accumulation_dtype'] == 'float32' and manifest['None_is_not_zero'] is True and
                manifest['accumulation_order'] == ['u%04d_s%d' % (u, s) for u in (1, 6, 15, 24) for s in (1, 2, 3, 4)] and
                list(manifest['parameters']) == names, 'Cached gradient contract differs')
        current = {}
        for name in names:
            row = manifest['parameters'][name]
            if row['is_none']:
                require(row.get('file') is None and row['sha256'] is None, 'None cache has a physical array')
                current[name] = None
                continue
            item = row['file']
            expected = OLD_OUTPUTS + '/' + label + '/gradients/' + term + '/' + name + '.npy'
            require(item['path'] == expected, 'Cached array path differs')
            path = dependency_path(parents, protocol, 'loss_response', expected, contract)
            require(contract.sha256(path) == item['sha256'] and path.stat().st_size == item['bytes'],
                    'Cached array file fingerprint differs')
            array = np.load(path, allow_pickle=False)
            require(array.dtype == np.float32 and list(array.shape) == row['shape'] == list(parameters[name].shape) and
                    np.isfinite(array).all() and array_digest(array, np) == row['sha256'] == item['array_sha256'],
                    'Cached physical gradient bytes differ')
            current[name] = array
        require(sum(value is not None for value in current.values()) == expected_counts[term],
                'Cached physical/None route census differs')
        components[term] = current
        bindings[term] = {'root': 'loss_response', 'manifest_path': relative,
            'manifest_sha256': contract.sha256(dependency_path(parents, protocol, 'loss_response', relative, contract)),
            'physical_arrays': expected_counts[term], 'recomputed': False}
    return components, bindings


def compare_outputs(exported, old_record, parents, protocol, contract, *, source=False):
    keys = ('native512', 'raw256', 'tokens', 'logits', 'features') if source else ('native512', 'raw256', 'tokens')
    result = {}
    for key in keys:
        a, b = exported[key], old_record['outputs'][key]
        dependency_path(parents, protocol, 'loss_response', b['path'], contract)
        require(a['shape'] == b['shape'] and a['dtype'] == b['dtype'] and
                a['array_sha256'] == b['array_sha256'], 'Prior output parity failed: ' + key)
        result[key] = {'exact': True, 'array_sha256': a['array_sha256'], 'prior_path': b['path']}
    # Exact RAW and the unchanged floor policy imply identical delivered pixels;
    # compressed PNG byte equality is not assumed across encoder versions.
    return {'pass': True, 'arrays': result, 'PNG_pixels_implied_by_exact_RAW_and_unchanged_floor': True}


def save_basis(label, basis_name, arrays, names, cases, budget, helper, np):
    stats = helper.gradient_statistics(names, arrays, np)
    require(stats['groups']['encoder']['none'] == 164 and stats['groups']['predictor']['none'] == 114 and
            stats['groups']['fusion']['none'] == 0, 'Fresh coarse/centered route is not fusion-only')
    for name in names:
        if arrays[name] is not None:
            stats['parameters'][name]['file'] = budget.array(
                budget.out / label / 'fresh_gradients' / basis_name / (name + '.npy'), arrays[name], np)
    source_cases = [case['id'] for case in cases if
        basis_name in ('all_coarse', 'canonical_hybrid') or
        (case['slot'] in (1, 3) if basis_name in ('normal_coarse', 'centered_normal') else case['slot'] in (2, 4))]
    stats.update({'basis': basis_name, 'snapshot': label, 'weighted': True, 'coarse_weight': 4.,
        'aggregation_divisor': 16, 'accumulation_dtype': 'float32', 'accumulation_order': source_cases,
        'None_is_not_zero': True, 'only_nonNone_physical_arrays_saved': True,
        'per_source_gradient_vectors_exported': False,
        'canonical_hybrid_policy': 'normal:centered;dark:original;direct interleaved16-source FP32 accumulation',
        'reaggregation_is_source_bound': True})
    budget.json(budget.out / label / 'fresh_gradients' / basis_name / 'manifest.json', stats)
    return {'basis': basis_name, 'physical_arrays': 72, 'groups': stats['groups']}


def save_fusion_deltas(net, base_model, names, directory, budget, helper, torch, np, delta_record):
    rows = {}
    state = net.state_dict()
    for name in names:
        if not name.startswith('fuse_convs_dict.'):
            continue
        after = state[name].detach().cpu().float().numpy()
        before = base_model[name].detach().cpu().float().numpy()
        delta = np.ascontiguousarray(after - before, dtype=np.float32)
        require(array_digest(delta, np) == delta_record['parameters'][name]['delta_sha256'],
                'Exported actual fusion delta differs from recorded actual CUDA state')
        rows[name] = budget.array(directory / (name + '.npy'), delta, np)
    require(len(rows) == 72, 'All72 actual fusion delta arrays required')
    record = {'parameters': rows, 'meaning': 'Actual CUDA parameter after minus exact saved CPU base, float32',
        'not_a_model_checkpoint': True, 'full_model_delta_arrays_exported': False}
    budget.json(directory / 'manifest.json', record)
    return record


def common_terms(prediction, captured, case, arrays, parents, protocol, contract, model, torch, np):
    """Reduction-only witness of common losses; no teacher or LPIPS forward."""
    functional = torch.nn.functional
    device = prediction.device
    indices = np.load(dependency_path(parents, protocol, 'adain', case['hq_indices_path'], contract),
                      allow_pickle=False)
    features = np.load(dependency_path(parents, protocol, 'adain', case['hq_features_path'], contract),
                       allow_pickle=False)
    logits = torch.from_numpy(captured['logits'].copy()).to(device=device)
    feature = torch.from_numpy(captured['features'].copy()).to(device=device)
    clean = torch.from_numpy(arrays[case['id']]['clean512'].copy()).permute(2, 0, 1).unsqueeze(0).to(device=device)
    rgb = (prediction.detach().float() + 1) / 2
    values = {'token_ce': functional.cross_entropy(logits.transpose(1, 2),
                      torch.from_numpy(indices.copy()).to(device=device)),
              'latent_mse': functional.mse_loss(feature,
                      torch.from_numpy(features.copy()).to(device=device))}
    if case['parameters']['image_eligible']:
        values['pixel_l1_rgb'] = functional.l1_loss(rgb, clean)
    if case['slot'] == 3:
        delivered = np.load(dependency_path(parents, protocol, 'adain', case['behavior_delivered_path'], contract),
                            allow_pickle=False)
        target_logits = np.load(dependency_path(parents, protocol, 'adain', case['behavior_logits_path'], contract),
                                allow_pickle=False)
        values['behavior_rgb_l1'] = functional.l1_loss(model.delivered_rgb(prediction.detach()),
            torch.from_numpy(delivered.copy()).to(device=device))
        values['behavior_token_kl'] = model.token_distillation(logits,
            torch.from_numpy(target_logits.copy()).to(device=device))
    result = {key: float(value.detach().cpu()) for key, value in values.items()}
    require(all(math.isfinite(value) for value in result.values()), 'Nonfinite common scalar witness')
    return result


def query_gradient(loss, names, parameters, torch, np, *, retain_graph):
    gradients = torch.autograd.grad(loss, [parameters[name] for name in names],
        allow_unused=True, retain_graph=retain_graph, create_graph=False)
    current = {}
    for name, gradient in zip(names, gradients):
        if gradient is None:
            current[name] = None
            continue
        require(gradient.dtype == torch.float32 and torch.isfinite(gradient).all().item(),
                'Nonfinite/non-FP32 fresh gradient')
        require(name.startswith('fuse_convs_dict.'), 'Coarse/centered gradient escaped isolated fusion route')
        current[name] = np.ascontiguousarray(gradient.detach().cpu().numpy().copy(), dtype=np.float32)
    require(sum(value is not None for value in current.values()) == 72,
            'Fresh coarse query must return72 physical fusion arrays and278 None')
    return current


def validate_prior_branch(summary, label, condition, base_model_digest, base_optimizer_digest,
                          base_rng_digest, contract):
    require(summary['snapshot'] == label and summary['condition'] == condition and
            summary['base_model_sha256'] == base_model_digest and
            summary['base_Adam_sha256'] == base_optimizer_digest and
            summary['base_RNG_sha256'] == base_rng_digest and
            summary['committed_training_updates'] == 0 and summary['branch_chain'] is False and
            summary['model_checkpoint_written'] is False and
            summary['restored_independently_before_branch'] is True,
            'Prior control identity/base/restoration differs')


def run_snapshot(label, before_counter, checkpoint, cases, arrays, net, optimizer, names, parameters,
                 model, route, architecture, parents, protocol, contract, helper, budget, torch, np, Image):
    directory = budget.out / label
    base_model, base_rng = checkpoint['model'], checkpoint['rng']
    base_optimizer = deepcopy(checkpoint['optimizer'])
    rates = protocol['optimizer']['fixed_group_rates']
    require(len(base_optimizer['param_groups']) == 2 and
            [group['name'] for group in base_optimizer['param_groups']] == ['encoder_token', 'fusion'] and
            [group['params'] for group in base_optimizer['param_groups']] ==
                [list(range(278)), list(range(278, 350))], 'Exact Adam group/parameter order differs')
    for group in base_optimizer['param_groups']:
        require(tuple(group['betas']) == (.9, .99) and group['eps'] == 1e-8 and group['weight_decay'] == 0. and
                group.get('amsgrad') is False and group.get('maximize') is False and
                group.get('capturable') is False and group.get('differentiable') is False,
                'Exact saved Adam semantics differ')
        group['lr'] = rates[group['name']]
    base_model_digest = helper.tree_digest(base_model, torch)
    base_optimizer_digest = helper.tree_digest(base_optimizer, torch)
    base_rng_digest = helper.tree_digest(base_rng, torch)
    frozen_digest = model.frozen_state_digest(net)
    prior_source_summary = old_json(parents, protocol, OLD_OUTPUTS + '/' + label + '/source_summary.json', contract)
    require(prior_source_summary['base_model_sha256'] == base_model_digest and
            prior_source_summary['base_Adam_sha256'] == base_optimizer_digest and
            prior_source_summary['base_RNG_sha256'] == base_rng_digest and
            prior_source_summary['fixed_component_order'] == list(COMPONENTS) and
            prior_source_summary['source_extraction_committed_updates'] == 0,
            'Cached component source base differs from exact checkpoint/Adam/RNG')
    old_controls = {condition: old_json(parents, protocol,
        OLD_OUTPUTS + '/' + label + '/conditions/' + condition + '/branch.json', contract)
        for condition in ('full', 'momentum_only')}
    for condition, summary in old_controls.items():
        validate_prior_branch(summary, label, condition, base_model_digest, base_optimizer_digest, base_rng_digest, contract)

    def restore_base():
        net.load_state_dict(base_model, strict=True)
        optimizer.load_state_dict(deepcopy(base_optimizer))
        optimizer.zero_grad(set_to_none=True)
        net.eval(); net.quantize.eval(); net.generator.eval()
        helper.restore_rng(base_rng, torch, np)
        require(helper.tree_digest(net.state_dict(), torch) == base_model_digest and
                helper.tree_digest(optimizer.state_dict(), torch) == base_optimizer_digest and
                helper.tree_digest(helper.rng_state(torch, np), torch) == base_rng_digest,
                'Exact base model/Adam/RNG restoration failed')
        require(helper.optimizer_step_values(optimizer) == [float(before_counter)] * 350 and
                all(parameter.grad is None for parameter in net.parameters()) and
                model.frozen_state_digest(net) == frozen_digest, 'Restored counter/gradient/frozen state differs')

    restore_base()
    budget.set_phase('source')
    components, cached_bindings = load_cached_components(label, names, parameters, parents, protocol, contract, helper, np)
    fresh = {basis: dict.fromkeys(names) for basis in BASES}
    source_outputs, source_receipts = {}, []
    for case in cases:
        budget.check()
        net.train(); net.quantize.eval(); net.generator.eval()
        before_rng = helper.tree_digest(helper.rng_state(torch, np), torch)
        require(before_rng == base_rng_digest, 'Canonical source started with a different saved RNG')
        normal = case['slot'] in (1, 3)
        require(normal == case['profile'].endswith('_normal'), 'Normal/dark TRAIN role differs')
        prior_record = old_json(parents, protocol,
            OLD_OUTPUTS + '/' + label + '/sources/' + case['id'] + '/record.json', contract)
        require(prior_record['snapshot'] == label and prior_record['case'] == case['id'] and
                prior_record['reference_id'] == case['reference_id'] and
                prior_record['profile'] == case['profile'] and prior_record['aggregation_divisor'] == 16,
                'Prior source observation identity differs')
        budget.reserve('candidate_source_forwards', {'snapshot': label, 'case': case['id']})
        with torch.autocast(device_type='cuda', enabled=False), route.isolated_style(architecture) as route_capture, \
                helper.capture_candidate_path(net, arrays=True) as captured:
            prediction = model.infer(net, helper._tensor(arrays[case['id']]['input'], torch))
        route.validate_capture(route_capture)
        require(all(value == 'torch.float32' for value in captured['dtypes'].values()), 'Source non-FP32 dtype')
        exported = helper._save_prediction(budget, directory / 'sources' / case['id'], prediction, captured,
            arrays[case['id']]['target256'], model, np, Image)
        output_parity = compare_outputs(exported, prior_record, parents, protocol, contract, source=True)
        source_outputs[case['id']] = exported
        clean = helper._tensor(arrays[case['id']]['clean512'], torch)
        ordinary, centered, mean_luminance = coarse_terms(prediction, clean, torch)
        scalar_witnesses = common_terms(prediction, captured, case, arrays, parents, protocol,
                                        contract, model, torch, np)
        scalar_witnesses['coarse_rgb_l1'] = float(ordinary.detach().cpu())
        scalar_checks = {term: value == prior_record['terms'][term] for term, value in scalar_witnesses.items()}
        require(all(scalar_checks.values()), 'Fresh common scalar loss witness differs from prior source')
        budget.reserve('coarse_gradient_queries', {'snapshot': label, 'case': case['id'], 'weight': 4., 'divisor': 16})
        ordinary_gradient = query_gradient(4. * ordinary / 16, names, parameters, torch, np, retain_graph=normal)
        accumulate(fresh['all_coarse'], ordinary_gradient, names, np)
        accumulate(fresh['normal_coarse' if normal else 'dark_coarse'], ordinary_gradient, names, np)
        centered_stats = None
        if normal:
            budget.reserve('centered_normal_gradient_queries', {'snapshot': label, 'case': case['id'],
                'weight': 4., 'divisor': 16, 'mean_is_differentiated': True})
            centered_gradient = query_gradient(4. * centered / 16, names, parameters, torch, np, retain_graph=False)
            accumulate(fresh['centered_normal'], centered_gradient, names, np)
            accumulate(fresh['canonical_hybrid'], centered_gradient, names, np)
            centered_stats = helper.gradient_statistics(names, centered_gradient, np)
            del centered_gradient
        else:
            accumulate(fresh['canonical_hybrid'], ordinary_gradient, names, np)
        after_rng = helper.tree_digest(helper.rng_state(torch, np), torch)
        require(after_rng == before_rng and all(parameter.grad is None for parameter in net.parameters()),
                'Source query changed RNG or parameter.grad')
        ordinary_stats = helper.gradient_statistics(names, ordinary_gradient, np)
        record = {'snapshot': label, 'case': case['id'], 'reference_id': case['reference_id'],
            'update': case['update'], 'slot': case['slot'], 'profile': case['profile'], 'normal': normal,
            'source_RNG_before_sha256': before_rng, 'source_RNG_after_sha256': after_rng,
            'outputs': exported, 'prior_source_output_parity': output_parity, 'route': route_capture,
            'fresh_scalar_witnesses': scalar_witnesses, 'fresh_scalar_exact_checks': scalar_checks,
            'LPIPS_scalar': prior_record['terms'].get('lpips_vgg'),
            'LPIPS_scalar_is_cached_not_freshly_revalidated': True,
            'coarse_rgb_l1': float(ordinary.detach().cpu()), 'centered_l1': float(centered.detach().cpu()),
            'mean_coarse_luminance_error': float(mean_luminance.detach().cpu()),
            'mean_luminance_differentiated': True, 'aggregation_divisor': 16, 'coarse_weight': 4.,
            'coarse_gradient_groups': ordinary_stats['groups'],
            'coarse_gradient_sha256_by_tensor': {name: ordinary_stats['parameters'][name]['sha256'] for name in names},
            'centered_gradient_groups': None if centered_stats is None else centered_stats['groups'],
            'centered_gradient_sha256_by_tensor': None if centered_stats is None else
                {name: centered_stats['parameters'][name]['sha256'] for name in names},
            'per_source_gradient_vectors_exported': False, 'parameter_grad_remains_None': True}
        budget.json(directory / 'sources' / case['id'] / 'record.json', record)
        source_receipts.append({'case': case['id'], 'output_parity': True, 'scalar_parity': True,
                                'RNG_unchanged': True})
        del ordinary_gradient, ordinary_stats, centered_stats, ordinary, centered, mean_luminance, clean, prediction, captured
    parity = exact_arrays(fresh['all_coarse'], components['coarse_rgb_l1'], names, np)
    # Record any disagreement before rejecting it; never relax a zero tolerance.
    budget.json(directory / 'fresh_all_cached_coarse_parity.json', parity)
    require(parity['pass'], 'Fresh all-coarse gradient does not bitmatch cached original; no branches permitted')
    basis_records = [save_basis(label, basis, fresh[basis], names, cases, budget, helper, np) for basis in BASES]
    require(helper.tree_digest(net.state_dict(), torch) == base_model_digest and
            helper.tree_digest(optimizer.state_dict(), torch) == base_optimizer_digest and
            helper.tree_digest(helper.rng_state(torch, np), torch) == base_rng_digest,
            'Fresh source extraction changed base model/Adam/RNG')
    budget.json(directory / 'source_summary.json', {'snapshot': label, 'sources': source_receipts,
        'base_model_sha256': base_model_digest, 'base_Adam_sha256': base_optimizer_digest,
        'base_RNG_sha256': base_rng_digest, 'cached_components': cached_bindings,
        'fresh_bases': basis_records, 'fresh_all_coarse_cache_bit_exact': True,
        'fixed_component_order': list(COMPONENTS), 'source_extraction_committed_updates': 0,
        'fresh_direct_full_gradient_queries': 0, 'new_full_autograd_parity_claimed': False,
        'old_full_gradient_checks_remain_source_bound': True,
        'hybrid_accumulated_per_source_not_regrouped_or_subtracted': True})

    @contextmanager
    def independent_branch(condition, branch_directory):
        restore_base()
        try:
            yield
        finally:
            active_error = sys.exc_info()[0] is not None
            recovery = {'snapshot': label, 'condition': condition,
                'actual_completed_disposable_Adam_steps': budget.actual_steps,
                'optimizer_inflight': budget.optimizer_inflight, 'committed_training_updates': 0}
            try:
                restore_base()
                recovery.update({'exact_base_model_Adam_RNG_restored': True,
                    'model_sha256': base_model_digest, 'Adam_sha256': base_optimizer_digest,
                    'RNG_sha256': base_rng_digest, 'Adam_counter': before_counter,
                    'mode': 'eval; frozen quantize/generator eval; gradients None'})
            except BaseException as error:
                recovery.update({'exact_base_model_Adam_RNG_restored': False,
                                  'restoration_error': type(error).__name__ + ': ' + str(error)})
                helper.atomic_json(branch_directory / 'restoration.json', recovery)
                if not active_error:
                    raise
            else:
                path = branch_directory / 'restoration.json'
                helper.atomic_json(path, recovery)
                budget.payload_bytes += path.stat().st_size
                budget.metadata_bytes += path.stat().st_size; budget.regular_files += 1
                budget.sample()

    budget.set_phase('probe')
    baseline, branch_summaries = {}, []
    for condition in BRANCHES:
        branch_directory = directory / 'conditions' / condition
        with independent_branch(condition, branch_directory):
            step_record = None
            if condition != 'baseline':
                coarse = fresh['dark_coarse'] if condition == 'omit_normal_coarse' else \
                         fresh['canonical_hybrid'] if condition == 'center_normal_luminance' else components['coarse_rgb_l1']
                assigned = compose(components, coarse, names, np, momentum_only=condition == 'momentum_only')
                assigned_stats = helper.gradient_statistics(names, assigned, np)
                assigned_hashes = {name: assigned_stats['parameters'][name]['sha256'] for name in names}
                control = old_controls.get(condition)
                if control is not None:
                    require(assigned_hashes == control['step']['assigned_gradient_sha256_by_tensor'],
                            'Prior full/momentum assigned gradient digests differ')
                if condition in ('omit_normal_coarse', 'center_normal_luminance'):
                    require(all(assigned_hashes[name] == old_controls['full']['step']['assigned_gradient_sha256_by_tensor'][name]
                                for name in names if not name.startswith('fuse_convs_dict.')),
                            'Normal-tone intervention changed cached encoder/predictor gradient')
                for name in names:
                    parameters[name].grad = torch.from_numpy(assigned[name].copy()).to(device='cuda', dtype=torch.float32)
                require(all(parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all().item()
                            for name in names), 'All350 explicit finite assigned gradients required')
                if condition == 'momentum_only':
                    require(all(parameters[name].grad.count_nonzero().item() == 0 for name in names),
                            'Momentum-only must assign explicit350 zeros, never None')
                budget.reserve('optimizer_step_attempts', {'snapshot': label, 'condition': condition,
                    'base_counter': before_counter, 'disposable_counter_after': before_counter + 1})
                budget.optimizer_inflight = True
                optimizer.step()  # REAL disposable Adam step; never a committed update.
                budget.optimizer_inflight = False
                budget.actual_steps += 1; budget.counts['disposable_optimizer_steps'] += 1
                require(helper.optimizer_step_values(optimizer) == [float(before_counter + 1)] * 350,
                        'Actual disposable Adam counter did not advance0to1/50to51')
                require(all(torch.isfinite(value).all().item() for value in net.state_dict().values()),
                        'Nonfinite disposable model state')
                delta_record = helper.parameter_delta_record(net, base_model, torch, np)
                adam_after = helper.tree_digest(optimizer.state_dict(), torch)
                model_after = helper.tree_digest(net.state_dict(), torch)
                if control is not None:
                    require(adam_after == control['step']['Adam_after_sha256'] and
                            model_after == control['branch_model_sha256'], 'Prior full/momentum model/Adam digest differs')
                    for name, row in delta_record['parameters'].items():
                        previous = control['step']['parameter_deltas']['parameters'][name]
                        require(all(row[key] == previous[key] for key in ('base_sha256', 'after_sha256', 'delta_sha256')),
                                'Prior full/momentum actual parameter/delta digest differs')
                if condition in ('omit_normal_coarse', 'center_normal_luminance'):
                    previous = old_controls['full']['step']['parameter_deltas']['parameters']
                    require(all(row['after_sha256'] == previous[name]['after_sha256']
                                for name, row in delta_record['parameters'].items()
                                if not name.startswith('fuse_convs_dict.')),
                            'Normal-tone proposal changed a nonfusion parameter relative to full')
                delta_exports = save_fusion_deltas(net, base_model, names, branch_directory / 'fusion_deltas',
                    budget, helper, torch, np, delta_record)
                step_record = {'actual_Adam_step': True, 'step_before': before_counter, 'step_after': before_counter + 1,
                    'trainable_tensors_with_explicit_gradients': 350, 'fixed_rates': rates, 'schedule_advanced': False,
                    'momentum_only_explicit_zero': condition == 'momentum_only',
                    'coarse_basis': 'dark_coarse' if condition == 'omit_normal_coarse' else
                                    'canonical_hybrid' if condition == 'center_normal_luminance' else 'cached_original_coarse',
                    'assigned_gradient_groups': assigned_stats['groups'],
                    'assigned_gradient_sha256_by_tensor': assigned_hashes, 'Adam_after_sha256': adam_after,
                    'parameter_deltas': delta_record, 'fusion_delta_manifest':
                        (branch_directory / 'fusion_deltas' / 'manifest.json').relative_to(budget.root).as_posix(),
                    'prior_control_exact': control is not None}
                del assigned, assigned_stats, delta_record, delta_exports
            net.eval(); net.quantize.eval(); net.generator.eval()
            branch_model_digest = helper.tree_digest(net.state_dict(), torch)
            probe_records = []
            for case in cases:
                before_rng = helper.tree_digest(helper.rng_state(torch, np), torch)
                require(before_rng == base_rng_digest, 'Probe RNG differs from exact saved base')
                budget.reserve('candidate_probe_forwards', {'snapshot': label, 'condition': condition, 'case': case['id']})
                with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), \
                        route.isolated_style(architecture) as route_capture, helper.capture_candidate_path(net) as captured:
                    prediction = model.infer(net, helper._tensor(arrays[case['id']]['input'], torch))
                require(len(route_capture['calls']) == 1 and route_capture['original_callable_restored'],
                        'Probe AdaIN/hook scope failed to restore')
                exported = helper._save_prediction(budget, branch_directory / case['id'], prediction, captured,
                    arrays[case['id']]['target256'], model, np, Image)
                prior_parity = None
                if condition in ('full', 'momentum_only'):
                    old_record = old_json(parents, protocol, OLD_OUTPUTS + '/' + label + '/conditions/' +
                                         condition + '/' + case['id'] + '/record.json', contract)
                    require(old_record['snapshot'] == label and old_record['condition'] == condition and
                            old_record['case'] == case['id'], 'Prior control probe identity differs')
                    prior_parity = compare_outputs(exported, old_record, parents, protocol, contract)
                if condition == 'baseline':
                    require(all(exported[key]['array_sha256'] == source_outputs[case['id']][key]['array_sha256']
                                for key in ('native512', 'raw256', 'tokens')), 'Baseline/source alias differs')
                    baseline[case['id']] = exported
                after_rng = helper.tree_digest(helper.rng_state(torch, np), torch)
                require(after_rng == before_rng, 'Probe consumed saved RNG')
                budget.json(branch_directory / case['id'] / 'record.json',
                    {'snapshot': label, 'condition': condition, 'case': case['id'],
                     'reference_id': case['reference_id'], 'profile': case['profile'], 'outputs': exported,
                     'route': route_capture, 'RNG_before_sha256': before_rng, 'RNG_after_sha256': after_rng,
                     'prior_control_output_parity': prior_parity})
                probe_records.append({'case': case['id'], 'raw_metrics': exported['raw_metrics'],
                                      'png_metrics': exported['png_metrics']})
                del prediction, captured
            require(helper.tree_digest(net.state_dict(), torch) == branch_model_digest and
                    helper.tree_digest(optimizer.state_dict(), torch) ==
                        (base_optimizer_digest if condition == 'baseline' else step_record['Adam_after_sha256']) and
                    helper.tree_digest(helper.rng_state(torch, np), torch) == base_rng_digest and
                    model.frozen_state_digest(net) == frozen_digest, 'Probes changed model/Adam/RNG/frozen state')
            summary = {'snapshot': label, 'condition': condition, 'base_model_sha256': base_model_digest,
                'base_Adam_sha256': base_optimizer_digest, 'base_RNG_sha256': base_rng_digest,
                'branch_model_sha256': branch_model_digest, 'step': step_record, 'probes': probe_records,
                'actual_disposable_step_count_so_far': budget.actual_steps, 'committed_training_updates': 0,
                'restored_independently_before_branch': True, 'branch_chain': False, 'model_checkpoint_written': False}
            budget.json(branch_directory / 'branch.json', summary)
            branch_summaries.append({'condition': condition, 'model_sha256': branch_model_digest,
                                     'actual_Adam_step': condition != 'baseline'})
    restore_base()
    repeats = []
    for case in cases:
        budget.reserve('restoration_repeat_forwards', {'snapshot': label, 'case': case['id']})
        before_rng = helper.tree_digest(helper.rng_state(torch, np), torch)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), \
                route.isolated_style(architecture) as capture_route, helper.capture_candidate_path(net) as captured:
            prediction = model.infer(net, helper._tensor(arrays[case['id']]['input'], torch))
        require(capture_route['original_callable_restored'] and len(capture_route['calls']) == 1,
                'Repeat AdaIN/hook scope did not restore')
        native = prediction.detach().float().cpu().numpy().copy()
        raw = model.delivered_rgb(prediction).detach()[0].permute(1, 2, 0).cpu().numpy().copy()
        exported = {}
        for key, value in (('native512', native), ('raw256', raw), ('tokens', captured['tokens'])):
            require(array_digest(value, np) == baseline[case['id']][key]['array_sha256'], 'Saved repeat baseline parity failed')
            exported[key] = budget.array(directory / 'repeats' / case['id'] / (key + '.npy'), value, np)
        after_rng = helper.tree_digest(helper.rng_state(torch, np), torch)
        require(after_rng == before_rng == base_rng_digest, 'Repeat changed saved RNG')
        record = {'case': case['id'], 'native_RAW_tokens_bit_exact': True, 'outputs': exported,
                  'RNG_before_sha256': before_rng, 'RNG_after_sha256': after_rng}
        budget.json(directory / 'repeats' / case['id'] / 'record.json', record)
        repeats.append({'case': case['id'], 'native_RAW_tokens_bit_exact': True})
        del prediction, captured, native, raw
    restore_base()
    record = {'snapshot': label, 'base_counter': before_counter, 'branches': branch_summaries,
        'final_repeat': repeats, 'final_base_model_sha256': base_model_digest,
        'final_base_Adam_sha256': base_optimizer_digest, 'final_base_RNG_sha256': base_rng_digest,
        'final_exact_model_Adam_RNG_restoration': True, 'model_checkpoint_written': False,
        'committed_training_updates': 0, 'actual_disposable_Adam_steps_this_snapshot': 4,
        'resources': budget.measurements()}
    budget.json(directory / 'snapshot_complete.json', record)
    return record


def run(root, protocol, protocol_sha):
    from research.codeformer_normal_tone_v1 import contract
    # These checks precede Torch/model imports, asset loading and construction.
    root = contract.manual_root(root)
    require(platform.system() == 'Linux' and root.name == VM_NAME and bool(os.environ.get('TMUX')),
            'Exact manually opened Linux L4/tmux root required')
    contract.verify_protocol(protocol)
    require(contract.sha256(contract.safe_path(root, 'protocol.json')) == protocol_sha and
            contract.read_json(contract.safe_path(root, 'protocol.json')) == protocol, 'Current protocol bytes differ')
    parents = contract.resolve_parents(root)
    require(COUNTS == contract.COUNTS and list(COMPONENTS) == contract.COMPONENTS and
            list(BRANCHES) == contract.BRANCHES, 'Worker/contract call or objective census differs')
    require(call_totals(COUNTS)['neural_forwards'] == 224 and call_totals(COUNTS)['gradient_queries'] == 48 and
            COUNTS['disposable_optimizer_steps'] == 8, 'Source-derived finite arithmetic differs')
    limits = protocol['limits']
    require(limits['result_payload_bytes'] == 2 * 1024**3 and limits['archive_bytes'] == 9 * 1024**3 // 4 and
            limits['required_free_bytes'] == 10 * 1024**3 and limits['reserve_free_bytes'] == 4 * 1024**3 and
            limits['RSS_bytes'] == 8 * 1024**3 and limits['CUDA_reserved_bytes'] == 22 * 1024**3 and
            limits['regular_file_cap'] == 5000 and limits['worker_seconds'] == 1800,
            'Prospective finite resource bounds differ')
    require(shutil.disk_usage(root).free >= limits['required_free_bytes'], 'Require10GiB free after installation')
    out = contract.safe_path(root, protocol['outputs'], must_exist=False)
    require(not out.exists(), 'No repeat/resume; retain existing diagnostic evidence')
    out.mkdir(parents=True)
    started, budget = time.monotonic(), None
    request = {'signal': None}
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, lambda value, frame: request.__setitem__('signal', value))
    try:
        with pid_lock(root, protocol_sha, contract.write_new_json):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import research
            for alias in ('loss_response', 'adain'):
                location = str(parents[alias] / 'research')
                if location not in research.__path__:
                    research.__path__.append(location)
            if str(parents['adain']) not in sys.path:
                sys.path.append(str(parents['adain']))
            from research.codeformer_loss_response_v1 import worker as helper
            require(Path(helper.__file__).resolve() == dependency_path(parents, protocol, 'loss_response',
                    'research/codeformer_loss_response_v1/worker.py', contract), 'Frozen helper resolved outside bound source')
            import torch
            import numpy as np
            import PIL
            from PIL import Image
            from research.codeformer_adaptation_v2 import model, data
            from research.codeformer_adain_gradient_isolation_v1 import gradient_route
            from third_party.codeformer import codeformer_arch as architecture
            for module, relative in ((model, 'research/codeformer_adaptation_v2/model.py'),
                    (data, 'research/codeformer_adaptation_v2/data.py'),
                    (gradient_route, 'research/codeformer_adain_gradient_isolation_v1/gradient_route.py'),
                    (architecture, 'third_party/codeformer/codeformer_arch.py')):
                require(Path(module.__file__).resolve() == dependency_path(parents, protocol, 'adain', relative, contract),
                        'Model/data/route source resolved outside bound dependency')
            helper.require_l4(torch)
            require(resource is not None and torch.get_default_dtype() == torch.float32,
                    'Linux RSS high-water measurement and default FP32 required')
            budget = Budget(root, out, limits, torch, request, helper, started)
            random.seed(protocol['seed']); np.random.seed(protocol['seed'])
            torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            prior_initialization = old_json(parents, protocol, OLD_OUTPUTS + '/initialization.json', contract)
            require(helper.algorithm_metadata(torch) == prior_initialization['algorithm_settings'],
                    'Cached-control numerical environment differs; do not change determinism to force parity')
            require(model.LOSS_WEIGHTS == protocol['loss_weights'], 'Frozen objective weights differ')
            versions = {'Python': sys.version, 'Torch': torch.__version__, 'CUDA': torch.version.cuda,
                'NumPy': np.__version__, 'Pillow': PIL.__version__, 'OpenCV': data.cv2.__version__,
                'device': torch.cuda.get_device_name(0), 'platform': platform.platform()}
            driver_command = ['nvidia-smi', '--query-gpu=driver_version,name,uuid', '--format=csv,noheader']
            driver = subprocess.run(driver_command, capture_output=True, text=True, check=True, timeout=10)
            require(len(driver.stdout.encode('utf-8')) <= 4096 and len(driver.stderr.encode('utf-8')) <= 4096,
                    'NVIDIA driver provenance output exceeds4096byte bound')
            driver_rows = [line.strip() for line in driver.stdout.splitlines() if line.strip()]
            require(len(driver_rows) == 1 and len(driver_rows[0].split(',')) == 3,
                    'Exactly one driver/name/UUID provenance row required')
            driver_version, driver_name, driver_uuid = [value.strip() for value in driver_rows[0].split(',')]
            require(driver_name == 'NVIDIA L4' and driver_version and driver_uuid.startswith('GPU-'),
                    'NVIDIA L4 driver/UUID provenance differs')
            versions.update({'NVIDIA_driver': driver_version, 'GPU_UUID': driver_uuid,
                             'driver_query': {'argv': driver_command, 'stdout': driver.stdout,
                                              'stderr': driver.stderr, 'timeout_seconds': 10}})
            budget.json(out / 'runtime_versions.json', {'versions': versions,
                'algorithm_settings': helper.algorithm_metadata(torch), 'FP32_autocast_disabled': True,
                'extra_teacher_perceptual_benchmark_forwards': 0,
                'cached_six_noncoarse_components_are_not_fresh_gradients': True,
                'current_resource_measurement': budget.measurements()})
            cases, arrays = load_cases(root, parents, protocol, contract, data, np, Image, budget)
            asset_paths = {}
            for name in ('initializer', 'original_checkpoint', 'stopped_checkpoint'):
                asset = protocol['assets'][name]
                path = dependency_path(parents, protocol, asset['root'], asset['path'], contract)
                require(contract.sha256(path) == asset['sha256'], 'Asset fingerprint differs: ' + name)
                asset_paths[name] = path
            require(contract.sha256(asset_paths['stopped_checkpoint']) == STOPPED_SHA, 'Exact retained stopped50 checkpoint required')
            # Conservative old simultaneous-live estimate, plus all five new
            # fusion aggregates. The old teacher/perceptual allowances remain
            # deliberately included despite constructing neither model here.
            projection = helper.projected_resources(asset_paths['stopped_checkpoint'].stat().st_size)
            projection['projected_RAM_bytes'] += 5 * 10659584 * 4
            projection['new_five_fusion_basis_element_bytes'] = 5 * 10659584 * 4
            projection['projected_output_payload_bytes'] = 1978003456
            require(projection['projected_RAM_bytes'] <= limits['RSS_bytes'] and
                    projection['projected_CUDA_bytes'] <= limits['CUDA_reserved_bytes'] and
                    projection['projected_output_payload_bytes'] < limits['result_payload_bytes'],
                    'Conservative prospective resource/output estimate exceeds cap')
            available_RAM = helper.linux_available_memory_bytes()
            available_CUDA, total_CUDA = torch.cuda.mem_get_info()
            require(available_RAM >= projection['projected_RAM_bytes'] and
                    available_CUDA >= projection['projected_CUDA_bytes'], 'Available RAM/CUDA below conservative allocation plan')
            budget.json(out / 'resource_plan.json', {**projection, 'available_RAM_bytes_before_model_load': available_RAM,
                'available_CUDA_bytes_before_model_load': available_CUDA, 'total_CUDA_bytes': total_CUDA,
                'limits': limits, 'measured_peaks_are_recorded_separately': True,
                'physical_output_census': {'source_sets': 32, 'probe_sets': 160, 'repeat_native_RAW_tokens_sets': 32,
                    'fresh_fusion_basis_arrays': 720, 'actual_fusion_delta_arrays': 576},
                'teacher_perceptual_behavior_benchmark_models_constructed': 0,
                'no_model_or_optimizer_checkpoint_output': True})
            budget.check()
            net, census = model.load_trainable(asset_paths['initializer'], device='cuda')
            names = [name for name, parameter in net.named_parameters() if parameter.requires_grad]
            parameters = dict(net.named_parameters())
            require(census == prior_initialization['census'] and len(names) == 350 and
                    census['trainable_parameters'] == 59788608 and census['frozen_parameters'] == 34324099,
                    'Frozen parameter census differs')
            frozen = model.frozen_state_digest(net)
            require(frozen == prior_initialization['frozen']['candidate'], 'Frozen initializer/codebook/generator differs')
            rates = protocol['optimizer']['fixed_group_rates']
            groups = []
            for group_name in ('encoder_token', 'fusion'):
                chosen = [parameter for name, parameter in net.named_parameters() if parameter.requires_grad and
                          (name.startswith('fuse_convs_dict.') == (group_name == 'fusion'))]
                require(len(chosen) == (72 if group_name == 'fusion' else 278), 'Adam partition census differs')
                groups.append({'params': chosen, 'lr': rates[group_name], 'base_lr': rates[group_name], 'name': group_name})
            optimizer = torch.optim.Adam(groups, betas=tuple(protocol['optimizer']['betas']),
                                         eps=protocol['optimizer']['eps'], weight_decay=0.)
            optimizer_metadata = [{key: value for key, value in group.items() if key != 'params'}
                                  for group in optimizer.param_groups]
            require(json.loads(json.dumps(optimizer_metadata)) == prior_initialization['optimizer_default_metadata'],
                    'Actual Adam defaults differ from cached-control implementation')
            budget.check()
            original = torch.load(asset_paths['original_checkpoint'], map_location='cpu', weights_only=True)
            require(original['completed_updates'] == 0 and original['partial_microbatches_discarded'] == 0 and
                    not original['optimizer']['state'] and
                    helper.tree_digest(original['model'], torch) == helper.tree_digest(net.state_dict(), torch) and
                    model.state_digest(net) == '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
                    'Original0 exact checkpoint differs from untouched initializer/empty Adam')
            budget.check()
            stopped = torch.load(asset_paths['stopped_checkpoint'], map_location='cpu', weights_only=True)
            require(stopped['completed_updates'] == 50 and stopped['partial_microbatches_discarded'] == 0 and
                    stopped['protocol_sha256'] == contract.ROOTS['adain']['protocol_sha256'] and
                    len(stopped['optimizer']['state']) == 350, 'Stopped50 exact model/Adam state differs')
            budget.check()
            budget.json(out / 'initialization.json', {'pass': True, 'census': census, 'versions': versions,
                'algorithm_settings': helper.algorithm_metadata(torch), 'FP32_autocast_disabled': True,
                'frozen_candidate_sha256': frozen, 'source_observations': 16, 'source_TRAIN_references': 8,
                'source_image_eligible': 12, 'source_normal': 8, 'source_dark': 8,
                'source_equals_probe': True, 'fixed_diagnostic_rates': rates, 'schedule_extension': False,
                'optimizer_default_metadata': optimizer_metadata,
                'original0_model_sha256': helper.tree_digest(original['model'], torch),
                'original0_checkpoint_RNG_sha256': helper.tree_digest(original['rng'], torch),
                'stopped50_model_sha256': helper.tree_digest(stopped['model'], torch),
                'stopped50_checkpoint_RNG_sha256': helper.tree_digest(stopped['rng'], torch),
                'fresh_teacher_or_initializer_benchmark_parity_calls': 0,
                'initializer_proof': 'Exact old checkpoint/initializer state plus32 exact prior source output witnesses',
                'teacher_perceptual_common_gradient_basis': 'Unchanged, hash-bound old six-component arrays; not rerun',
                'resources': budget.measurements()})
            summaries = []
            for label, before_counter, checkpoint in (('original0', 0, original), ('stopped50', 50, stopped)):
                summaries.append(run_snapshot(label, before_counter, checkpoint, cases, arrays, net, optimizer,
                    names, parameters, model, gradient_route, architecture, parents, protocol, contract,
                    helper, budget, torch, np, Image))
            budget.check()
            require(budget.counts == COUNTS and budget.actual_steps == 8 and not budget.optimizer_inflight,
                    'Actual diagnostic census differs from224forwards/48queries/8disposable steps')
            require(model.frozen_state_digest(net) == frozen, 'Final frozen candidate changed')
            completion = {'format': 'codeformer-normal-tone-completion-v1', 'complete': True,
                'protocol_sha256': protocol_sha, 'counts': dict(budget.counts), **call_totals(budget.counts),
                'actual_disposable_Adam_steps': budget.actual_steps, 'optimizer_inflight': False,
                'committed_training_updates': 0, 'backward_calls': 0, 'model_checkpoint_written': False,
                'branches_chain': False, 'model_qualified': False, 'resume_authorized': False,
                'fresh_direct_full_gradient_queries': 0, 'new_full_autograd_parity_claimed': False,
                'cached_gradient_reuse_explicit': True, 'snapshots': summaries,
                'phase_seconds': budget.timings(), 'worker_seconds': time.monotonic() - started,
                'algorithm_settings': helper.algorithm_metadata(torch), 'resources': budget.measurements(),
                'final_frozen_candidate_sha256': frozen}
            budget.json(out / 'resource_measurements.json', budget.measurements())
            budget.json(out / 'completion.json', completion)
            return completion
    except BaseException as error:
        if budget is not None:
            try:
                budget.sample()
            except BaseException:
                pass  # Preserve the last successfully measured counters too.
        failure = {'format': 'codeformer-normal-tone-failure-v1', 'complete': False,
            'protocol_sha256': protocol_sha, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc()[-32768:], 'worker_seconds': time.monotonic() - started,
            'counts': None if budget is None else dict(budget.counts),
            'actual_completed_disposable_Adam_steps': 0 if budget is None else budget.actual_steps,
            'actual_disposable_Adam_steps': 0 if budget is None else budget.actual_steps,
            'optimizer_inflight_at_failure': False if budget is None else budget.optimizer_inflight,
            'partial_disposable_step_possible': False if budget is None else budget.optimizer_inflight,
            'committed_training_updates': 0, 'backward_calls': 0, 'model_checkpoint_written': False,
            'model_qualified': False, 'resume_authorized': False, 'all_failure_evidence_retained': True,
            'resources': None if budget is None else budget.measurements()}
        if not (out / 'failure.json').exists():
            contract.write_new_json(out / 'failure.json', failure)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', '--protocol-sha256', dest='protocol_sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-transfer', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--export', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from research.codeformer_normal_tone_v1 import transport
    transfer = transport.verify_transfer(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps(transfer_cli_summary(transfer)), flush=True)
    elif args.export:
        print(json.dumps(transport.export_results(root, args.protocol_sha)), flush=True)
    else:
        print(json.dumps(run(root, transfer['protocol'], args.protocol_sha)), flush=True)


if __name__ == '__main__':
    main()
