"""Finite, manual L4 loss-response measurement; disposable Adam steps are real.

Static verification/export never imports a neural framework. The diagnostic
does not commit training, chain branches, write a model checkpoint, resume a
failed recipe, change an app, or establish model qualification.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

try:
    import resource
except ImportError:  # Static Windows checks do not execute the Linux worker.
    resource = None


PACKAGE_DIR = 'cctv_codeformer_loss_response_v1_vm'
PARENT_DIR = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
STOPPED_CHECKPOINT_SHA256 = '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b'
COMPONENTS = ('token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
              'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl')
CONDITIONS = ('baseline', 'full', *('drop:' + name for name in COMPONENTS), 'momentum_only')
EXPECTED_CAPS = {'candidate_source_forwards': 32, 'candidate_probe_forwards': 320,
                 'restoration_repeat_forwards': 32, 'behavior_teacher_forwards': 4,
                 'initializer_parity_candidate_forwards': 2,
                 'initializer_parity_benchmark_forwards': 2,
                 'HQ_teacher_forwards': 8, 'perceptual_source_forwards': 24,
                 'perceptual_prerequisite_forwards': 1, 'component_gradient_queries': 160,
                 'full_gradient_queries': 32, 'perceptual_prerequisite_gradient_queries': 1,
                 'optimizer_step_attempts': 18, 'disposable_optimizer_steps': 18,
                 'committed_training_updates': 0, 'backward_calls': 0}
FORWARD_COUNTERS = tuple(name for name in EXPECTED_CAPS if name.endswith('_forwards'))


class DiagnosticStop(RuntimeError):
    """Finite resource/control failure; all already written evidence is retained."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def bounded(root, relative, *, must_exist=True):
    require(isinstance(relative, str) and relative, 'Empty relative path')
    name = PurePosixPath(relative)
    require(not name.is_absolute() and name.as_posix() == relative and '..' not in name.parts
            and ':' not in relative and '\\' not in relative, 'Noncanonical path: ' + relative)
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*name.parts)
    require(path.resolve(strict=must_exist).is_relative_to(root), 'Path escapes declared root')
    for current in [path, *path.parents]:
        if current == root.parent:
            break
        if current.exists():
            require(not current.is_symlink() and not getattr(current.lstat(), 'st_file_attributes', 0) & 0x400,
                    'Symlink/reparse point forbidden')
    if must_exist:
        require(path.is_file(), 'Expected exact regular file: ' + relative)
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Retain existing evidence: ' + str(path))
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def pid_lock(root, protocol_sha):
    path = Path(root) / 'loss_response_v1.lock'
    payload = {'pid': os.getpid(), 'host': socket.gethostname(), 'protocol_sha256': protocol_sha,
               'UTC': datetime.now(timezone.utc).isoformat()}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream); stream.flush(); os.fsync(stream.fileno())
    try:
        yield
    finally:
        if path.exists() and read_json(path) == payload:
            path.unlink()  # Only this process's exact lock; no stale-lock removal.


def require_vm_path(root):
    """Called before Torch, model source, checkpoint loads, or neural execution."""
    require(platform.system() == 'Linux', 'Actual diagnostic is Linux L4 VM only')
    base = (Path.home() / 'forensic-dgp').resolve(strict=True)
    root = Path(root).resolve(strict=True)
    require(root == base / PACKAGE_DIR and root.is_relative_to(base),
            'Diagnostic root must be exact new packet below ~/forensic-dgp')
    require(bool(os.environ.get('TMUX')), 'Human-opened tmux session required')
    require((root.parent / PARENT_DIR).is_dir(), 'Complete frozen sibling parent required')
    return root


def require_l4(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One available CUDA GPU required')
    require(torch.cuda.get_device_name(0) == 'NVIDIA L4', 'NVIDIA L4 required')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    others = [int(line.strip()) for line in result.stdout.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    require(not others, 'Competing GPU workload; diagnostic refused')


def call_totals(counts):
    return {'neural_forwards': sum(counts[name] for name in FORWARD_COUNTERS),
            'gradient_queries': sum(counts[name] for name in
                ('component_gradient_queries', 'full_gradient_queries', 'perceptual_prerequisite_gradient_queries')),
            'Adam_step_attempt_reservations': counts['optimizer_step_attempts'],
            'committed_training_updates': 0, 'backward_calls': 0}


def cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item, torch) for item in value)
    return deepcopy(value)


def tree_digest(value, torch):
    """Exact tensor bytes plus type-preserving metadata; no neural call."""
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            header = json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        elif isinstance(item, dict):
            digest.update(b'dict[')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                add(key); add(item[key])
            digest.update(b']'); return
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b'[')
            for child in item:
                add(child)
            digest.update(b']'); return
        else:
            header = json.dumps([type(item).__name__, item], separators=(',', ':'), allow_nan=False).encode()
            payload = b''
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(len(payload).to_bytes(8, 'big')); digest.update(payload)
    add(value)
    return digest.hexdigest()


def rng_state(torch, np):
    value = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'algorithm': value[0], 'keys': value[1].tolist(),
        'position': value[2], 'has_gauss': value[3], 'cached_gaussian': value[4]},
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]}


def restore_rng(state, torch, np):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['algorithm'], np.asarray(value['keys'], dtype=np.uint32),
        value['position'], value['has_gauss'], value['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def algorithm_metadata(torch):
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'CUDA_LAUNCH_BLOCKING': os.environ.get('CUDA_LAUNCH_BLOCKING')}


@contextmanager
def capture_candidate_path(net, *, arrays=False):
    """Observe one already-counted forward, without extra neural calls."""
    captured = {}
    original = net.quantize.get_codebook_feat
    def lookup(indices, shape):
        require('tokens' not in captured, 'Repeated candidate dictionary lookup')
        captured['tokens'] = indices.detach().cpu().numpy().copy().reshape(256)
        return original(indices, shape)
    def output_hook(module, arguments, output):
        require(isinstance(output, tuple) and len(output) == 3, 'Unexpected candidate return')
        captured['dtypes'] = {'native': str(output[0].dtype), 'logits': str(output[1].dtype),
                              'features': str(output[2].dtype)}
        if arrays:
            captured['logits'] = output[1].detach().float().cpu().numpy().copy()
            captured['features'] = output[2].detach().float().cpu().numpy().copy()
    hook = net.register_forward_hook(output_hook)
    try:
        net.quantize.get_codebook_feat = lookup
        yield captured
    finally:
        net.quantize.get_codebook_feat = original
        hook.remove()


def optimizer_step_values(optimizer):
    values = []
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state = optimizer.state.get(parameter, {})
            value = state.get('step', 0)
            values.append(float(value.item()) if hasattr(value, 'item') else float(value))
    return values


def parameter_partition(name):
    if name.startswith('encoder.'):
        return 'encoder'
    if name.startswith('fuse_convs_dict.'):
        return 'fusion'
    return 'predictor'


def array_digest(array, np):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def gradient_statistics(names, arrays, np):
    groups = {name: {'tensors': 0, 'none': 0, 'nonzero': 0, 'squared_l2': 0.0,
                     'maximum_absolute': 0.0} for name in ('encoder', 'predictor', 'fusion')}
    rows = {}
    for name in names:
        value = arrays.get(name)
        group = groups[parameter_partition(name)]; group['tensors'] += 1
        if value is None:
            group['none'] += 1
            rows[name] = {'is_none': True, 'shape': None, 'sha256': None}
            continue
        require(value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid CPU gradient array')
        squared = float(np.square(value.astype(np.float64)).sum(dtype=np.float64))
        maximum = float(np.abs(value).max())
        group['nonzero'] += int(maximum > 0); group['squared_l2'] += squared
        group['maximum_absolute'] = max(group['maximum_absolute'], maximum)
        rows[name] = {'is_none': False, 'shape': list(value.shape), 'dtype': 'float32',
                      'sha256': array_digest(value, np), 'l2': math.sqrt(squared),
                      'maximum_absolute': maximum}
    for group in groups.values():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
    return {'groups': groups, 'parameters': rows}


def compose_gradients(components, names, np, *, omitted=None, momentum_only=False):
    """CPU FP32 composition in declared order; None remains distinct from zero."""
    result = {}
    for name in names:
        value = None
        for component in COMPONENTS:
            if component == omitted:
                continue
            item = components[component].get(name)
            if item is not None:
                if value is None:
                    value = item.copy()
                else:
                    np.add(value, item, out=value)
        require(value is not None, 'Every disposable branch must address all350 trainable tensors')
        result[name] = np.zeros_like(value) if momentum_only else value
    return result


def projected_resources(stopped_checkpoint_bytes):
    """Conservative admission estimate; observed peaks remain separately guarded."""
    require(type(stopped_checkpoint_bytes) is int and stopped_checkpoint_bytes > 0, 'Invalid stopped-state bytes')
    return {'projected_RAM_bytes': stopped_checkpoint_bytes + 3 * 59788608 * 4 +
            2 * 681249536 + 2 * 94112707 * 4 + 128 * 1024**2 + 1024**3,
            'projected_CUDA_bytes': 2 * 94112707 * 4 + 63740100 * 4 + 64 * 1024**2 +
            5 * 59788608 * 4 + 12 * 1024**3,
            'CUDA_graph_workspace_allowance_bytes': 12 * 1024**3,
            'projection_is_not_measured_peak': True}


def linux_available_memory_bytes():
    """Read host availability only; the separate8GiB process RSS cap remains."""
    fields = {}
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if value.strip().endswith('kB'):
                fields[key] = int(value.split()[0]) * 1024
    require('MemAvailable' in fields, 'Cannot establish available host RAM')
    return fields['MemAvailable']


class Budget:
    """Reserve calls/writes before execution; preserve incomplete evidence."""
    def __init__(self, root, out, limits, torch, request):
        self.root, self.out, self.limits = Path(root), Path(out), limits
        self.torch, self.request = torch, request
        self.started = time.monotonic()
        self.counts = {name: 0 for name in EXPECTED_CAPS}
        self.actual_steps = 0
        self.optimizer_inflight = False
        self.phase = None
        self.phase_started = None
        self.elapsed = {'initialization': 0.0, 'source': 0.0, 'probe': 0.0}
        self.event = 0
        self.payload_bytes = 0
        self.metadata_bytes = 0
        self.regular_files = 0

    def set_phase(self, phase):
        now = time.monotonic()
        if self.phase is not None:
            self.elapsed[self.phase] += now - self.phase_started
        self.phase, self.phase_started = phase, now
        self.check()

    def timings(self):
        result = dict(self.elapsed)
        if self.phase is not None:
            result[self.phase] += time.monotonic() - self.phase_started
        return result

    def check(self):
        if self.request['signal'] is not None:
            raise DiagnosticStop('Manual signal requested finite diagnostic stop')
        if time.monotonic() - self.started > self.limits['worker_seconds']:
            raise DiagnosticStop('Worker wall-time cap reached')
        caps = {'initialization': self.limits['initialization_seconds'],
                'source': self.limits['source_seconds'], 'probe': self.limits['probe_seconds']}
        if any(seconds > caps[name] for name, seconds in self.timings().items()):
            raise DiagnosticStop('Cumulative diagnostic phase time cap reached')
        if shutil.disk_usage(self.root).free < self.limits['reserve_free_bytes']:
            raise DiagnosticStop('Declared4GiB disk reserve reached')
        if self.torch.cuda.memory_reserved() > self.limits['CUDA_reserved_bytes'] or \
                self.torch.cuda.max_memory_reserved() > self.limits['CUDA_reserved_bytes']:
            raise DiagnosticStop('Declared22GiB CUDA reserved memory bound reached')
        if resource is not None and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > self.limits['RSS_bytes']:
            raise DiagnosticStop('Declared8GiB process RSS bound reached')
        if self.payload_bytes > self.limits['result_payload_bytes']:
            raise DiagnosticStop('Declared4GiB diagnostic payload bound reached')
        if self.metadata_bytes > self.limits['metadata_bytes'] or self.regular_files > self.limits['regular_file_cap']:
            raise DiagnosticStop('Declared metadata/file-count bound reached')

    def reserve(self, name, context=None):
        self.check()
        require(name in EXPECTED_CAPS and self.counts[name] < EXPECTED_CAPS[name],
                'Finite call cap reached: ' + name)
        self.counts[name] += 1
        self.event += 1
        self.json(self.out / 'call_trace' / ('%05d.json' % self.event),
                  {'event': self.event, 'before_call': name, 'context': context,
                   'counts': dict(self.counts), 'actual_completed_Adam_steps': self.actual_steps,
                   'committed_training_updates': 0, 'backward_calls': 0,
                   'seconds': time.monotonic() - self.started})

    def reserve_write(self, estimate):
        self.check()
        require(type(estimate) is int and estimate >= 0, 'Invalid prospective write reservation')
        require(self.regular_files < self.limits['regular_file_cap'], 'Prospective write exceeds regular-file cap')
        require(self.payload_bytes + estimate <= self.limits['result_payload_bytes'] - 1024**2,
                'Prospective write exceeds diagnostic payload cap')
        require(shutil.disk_usage(self.root).free >= estimate + self.limits['reserve_free_bytes'],
                'Prospective write consumes declared disk reserve')

    def json(self, path, value):
        payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
        require(self.metadata_bytes + len(payload) <= self.limits['metadata_bytes'] - 65536,
                'Prospective metadata write exceeds declared cap')
        self.reserve_write(len(payload))
        atomic_json(path, value)
        self.payload_bytes += Path(path).stat().st_size
        self.metadata_bytes += Path(path).stat().st_size
        self.regular_files += 1

    def array(self, path, array, np):
        require(array.dtype.kind != 'O' and np.isfinite(array).all(), 'Invalid/nonfinite array output')
        require(array.nbytes + 1024 <= self.limits['array_file_bytes'], 'Array exceeds single-file cap')
        self.reserve_write(array.nbytes + 1024)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), 'Retain existing array evidence')
        temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
        with temporary.open('xb') as stream:
            np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        require(temporary.stat().st_size <= array.nbytes + 1024, 'Array header exceeds reserved bytes')
        os.replace(temporary, path)
        self.payload_bytes += path.stat().st_size
        self.regular_files += 1
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': sha256(path),
                'bytes': path.stat().st_size, 'shape': list(array.shape), 'dtype': str(array.dtype),
                'array_sha256': array_digest(array, np)}

    def png(self, path, array, Image):
        require(array.dtype.name == 'uint8' and array.shape == (256, 256, 3), 'Invalid PNG pixels')
        cap = array.nbytes + 4096
        self.reserve_write(cap)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), 'Retain existing PNG evidence')
        temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
        with temporary.open('xb') as stream:
            Image.fromarray(array).save(stream, format='PNG')
            stream.flush(); os.fsync(stream.fileno())
        require(temporary.stat().st_size <= cap, 'PNG exceeds reserved bytes')
        os.replace(temporary, path)
        self.payload_bytes += path.stat().st_size
        self.regular_files += 1
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': sha256(path),
                'bytes': path.stat().st_size}


def gradient_sum_check(component_arrays, full_arrays, names, np, *, atol, rtol):
    """Check finite vector agreement; scalar norms cannot establish attribution."""
    maximum, squared, full_maximum = 0.0, 0.0, 0.0
    for name in names:
        expected = None
        for arrays in component_arrays:
            value = arrays.get(name)
            if value is not None:
                if expected is None:
                    expected = value.copy()
                else:
                    np.add(expected, value, out=expected)
        actual = full_arrays.get(name)
        require((expected is None) == (actual is None), 'Full/component connectivity differs: ' + name)
        if expected is None:
            continue
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = max(maximum, float(np.abs(difference).max()))
        full_maximum = max(full_maximum, float(np.abs(actual).max()))
        squared += float(np.square(difference).sum(dtype=np.float64))
    bound = atol + rtol * full_maximum
    require(maximum <= bound, 'Full/component gradient vector discrepancy exceeds prespecified bound')
    return {'pass': True, 'maximum_absolute': maximum, 'difference_l2': math.sqrt(squared),
            'maximum_full_gradient_absolute': full_maximum, 'bound': bound,
            'atol': atol, 'rtol': rtol, 'full_gradient_vector_exported': False}


def output_metrics(rgb, target, np):
    """CPU descriptors on TRAIN paired probes; no DEV/FINAL acceptance claim."""
    error = rgb.astype(np.float64) - target.astype(np.float64)
    mean = error.mean(axis=(0, 1))
    residual = error - mean
    p = rgb.astype(np.float64).reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
    t = target.astype(np.float64).reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
    return {'mae': float(np.abs(error).mean()), 'mse': float(np.square(error).mean()),
            'signed_mean_RGB_error': mean.tolist(),
            'absolute_luminance_bias': float(abs(mean @ np.asarray([.2126, .7152, .0722]))),
            'coarse_rgb_mae': float(np.abs(p - t).mean()),
            'residual_mse': float(np.square(residual).mean())}


def parameter_delta_record(net, base, torch, np):
    """Record every tensor delta plus group totals, without model checkpoint output."""
    rows, groups = {}, {}
    for name, value in net.state_dict().items():
        before = base[name]
        actual = value.detach().cpu().contiguous()
        require(actual.shape == before.shape and actual.dtype == before.dtype, 'Parameter state geometry changed')
        delta = actual.float().numpy() - before.float().numpy()
        squared = float(np.square(delta.astype(np.float64)).sum(dtype=np.float64))
        maximum = float(np.abs(delta).max())
        group_name = name.split('.')[0]
        group = groups.setdefault(group_name, {'tensors': 0, 'changed_tensors': 0,
                                              'squared_l2': 0.0, 'maximum_absolute': 0.0})
        group['tensors'] += 1; group['changed_tensors'] += int(maximum > 0)
        group['squared_l2'] += squared; group['maximum_absolute'] = max(group['maximum_absolute'], maximum)
        rows[name] = {'shape': list(actual.shape), 'dtype': str(actual.dtype),
                      'base_sha256': array_digest(before.numpy(), np),
                      'after_sha256': array_digest(actual.numpy(), np),
                      'delta_sha256': array_digest(delta, np), 'delta_l2': math.sqrt(squared),
                      'maximum_absolute_delta': maximum}
    for group in groups.values():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
    require(all(groups[name]['changed_tensors'] == 0 for name in ('quantize', 'generator')),
            'Disposable diagnostic changed frozen dictionary/generator')
    return {'parameters': rows, 'groups': groups, 'model_checkpoint_written': False}


def _tensor(array, torch):
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device='cuda', dtype=torch.float32)


def _parent_asset(parent, protocol, name):
    value = protocol['assets'][name]
    path = bounded(parent, value['path'])
    require(sha256(path) == value['sha256'], 'Asset fingerprint differs: ' + name)
    return path


def _load_cases(root, parent, protocol, data, np, Image):
    from research.codeformer_loss_response_v1 import contract
    declared = protocol['cases_file']
    relative = declared['path'] if isinstance(declared, dict) else declared
    cases = read_json(bounded(root, relative))
    contract.verify_cases(cases)
    require(cases.get('source_updates') == [1, 6, 15, 24], 'Fixed source update selection differs')
    rows = cases['cases']
    require([(row['update'], row['slot']) for row in rows] ==
            [(update, slot) for update in (1, 6, 15, 24) for slot in range(1, 5)],
            'Source/probe observations must be exact ordered16 canonical TRAIN exposures')
    require(len({row['reference_id'] for row in rows}) == 8, 'Exactly8 TRAIN photographs required')
    references = data.load_reference_manifest(parent, 'train', parent / 'references.json',
        expected_manifest_sha256=sha256(parent / 'references.json'))
    by_id = {reference.id: reference for reference in references}
    regenerated = {}
    for update in (1, 6, 15, 24):
        group = [row for row in rows if row['update'] == update]
        require(group[0]['reference_id'] == group[1]['reference_id'] and
                group[2]['reference_id'] == group[3]['reference_id'], 'Matched normal/dark reference differs')
        pairs = data.make_training_update(by_id[group[0]['reference_id']], by_id[group[2]['reference_id']], update)
        for row, pair in zip(group, pairs):
            require(re.fullmatch(r'tr_ffhq_[0-9]{5}', row['reference_id']) is not None,
                    'Only approved photographic TRAIN references permitted')
            record = row['record']
            require(record['role'] == 'train' and row['parameters'] == pair.parameters and
                    row['profile'] == pair.record['profile'], 'Camera/role/profile contract differs')
            require(row['source_path'] == record['source_path'] == pair.record['source_path'] and
                    row['update'] == record['update'] and row['slot'] == record['slot'] and
                    row['reference_id'] == record['reference_id'], 'Top-level/source TRAIN case binding differs')
            require(sha256(bounded(parent, row['source_path'])) == record['source_sha256'],
                    'Case source path does not identify the exact retained TRAIN photograph')
            for key in ('input256_rgb8_sha256', 'input256_float32_sha256', 'clean512_float32_sha256',
                        'clean512_rgb8_sha256', 'source_sha256', 'reference_id', 'update', 'slot'):
                require(record[key] == pair.record[key], 'Canonical TRAIN regeneration differs: ' + key)
            with Image.open(bounded(parent, row['input_path'])) as image:
                require(image.mode == 'RGB' and image.size == (256, 256), 'Saved TRAIN input geometry differs')
                input_array = np.asarray(image, dtype=np.float32) / np.float32(255)
            require(np.array_equal(input_array, pair.input256), 'Saved PNG differs from regenerated canonical TRAIN input')
            require(read_json(bounded(parent, row['pair_record_path']))['input256_float32_sha256'] ==
                    record['input256_float32_sha256'], 'Saved canonical pair record differs')
            with Image.open(bounded(parent, row['source_path'])) as image:
                require(image.mode == 'RGB' and image.size == (1024, 1024), 'TRAIN source geometry differs')
                target256 = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / np.float32(255)
            regenerated[row['id']] = {'input': pair.input256, 'clean512': pair.clean512,
                                      'target256': target256, 'record': record}
    require(sum(bool(row['parameters']['image_eligible']) for row in rows) == 12,
            'Exactly12 image-eligible observations required')
    return rows, regenerated


def _save_prediction(budget, directory, prediction, captured, target, model, np, Image):
    native = prediction.detach().float().cpu().numpy().copy()
    raw = model.delivered_rgb(prediction).detach()[0].permute(1, 2, 0).cpu().numpy().copy()
    require(native.shape == (1, 3, 512, 512) and native.dtype == np.float32 and
            raw.shape == (256, 256, 3) and raw.dtype == np.float32, 'Prediction export geometry differs')
    tokens = captured['tokens']
    require(tokens.shape == (256,) and tokens.dtype == np.int64 and
            int(tokens.min()) >= 0 and int(tokens.max()) < 1024, 'Selected token export differs')
    png = np.floor(np.clip(raw, 0, 1) * 255).astype(np.uint8)
    record = {'native512': budget.array(directory / 'native512.npy', native, np),
              'raw256': budget.array(directory / 'raw256.npy', raw, np),
              'png256': budget.png(directory / 'result.png', png, Image),
              'tokens': budget.array(directory / 'tokens.npy', tokens, np),
              'dtypes': captured['dtypes'],
              'raw_metrics': output_metrics(raw, target, np),
              'png_metrics': output_metrics(png.astype(np.float32) / np.float32(255), target, np),
              'support': 'complete256x256 TRAIN crop; no padding or covering mask',
              'native_domain': 'raw CodeFormer normalized domain; no clipping',
              'raw_policy': 'native(output+1)/2 clamp[0,1], bilinear512to256, align_cornersFalse',
              'PNG_policy': 'floor(clipped RAW*255) RGB8; no display enhancement'}
    if 'logits' in captured:
        record['logits'] = budget.array(directory / 'logits.npy', captured['logits'], np)
        record['features'] = budget.array(directory / 'features.npy', captured['features'], np)
    return record


def _run_snapshot(label, before_counter, base_model, base_optimizer, base_rng,
                  cases, arrays, clean_codes, behavior_targets, net, optimizer, names, parameters,
                  model, route, architecture, teacher, perceptual, frozen,
                  budget, rates, torch, np, Image):
    directory = budget.out / label
    base_optimizer = deepcopy(base_optimizer)
    require(len(base_optimizer['param_groups']) == 2 and
            [group['name'] for group in base_optimizer['param_groups']] == ['encoder_token', 'fusion'],
            'Base Adam named group order differs')
    expected_ids = [list(range(278)), list(range(278, 350))]
    require([group['params'] for group in base_optimizer['param_groups']] == expected_ids,
            'Base Adam parameter order differs from exact candidate census')
    for group in base_optimizer['param_groups']:
        require(tuple(group['betas']) == (.9, .99) and group['eps'] == 1e-8 and
                group['weight_decay'] == 0.0 and group.get('amsgrad') is False and
                group.get('maximize') is False and group.get('capturable') is False and
                group.get('differentiable') is False, 'Base Adam semantics differ')
        group['lr'] = rates[group['name']]
    base_model_digest = tree_digest(base_model, torch)
    base_optimizer_digest = tree_digest(base_optimizer, torch)
    base_rng_digest = tree_digest(base_rng, torch)
    def restore_base():
        net.load_state_dict(base_model, strict=True)
        optimizer.load_state_dict(deepcopy(base_optimizer))
        optimizer.zero_grad(set_to_none=True)
        net.eval(); net.quantize.eval(); net.generator.eval()
        restore_rng(base_rng, torch, np)
        require(tree_digest(net.state_dict(), torch) == base_model_digest and
                tree_digest(optimizer.state_dict(), torch) == base_optimizer_digest and
                tree_digest(rng_state(torch, np), torch) == base_rng_digest,
                'Exact base model/Adam/RNG restoration failed')
        require(optimizer_step_values(optimizer) == [float(before_counter)] * 350,
                'Base Adam counters differ from0/50 snapshot')
        require(all(parameter.grad is None for parameter in net.parameters()), 'Base retained parameter.grad exists')
        require(model.frozen_state_digest(net) == frozen['candidate'], 'Restored frozen model differs')
    restore_base()
    budget.set_phase('source')
    components = {component: {name: None for name in names} for component in COMPONENTS}
    full_aggregate = {name: None for name in names}
    source_receipts = []
    for case in cases:
        budget.check()
        net.train(); net.quantize.eval(); net.generator.eval()
        observed = arrays[case['id']]
        cached = clean_codes[case['reference_id']]
        codes = model.CleanCodes(cached.indices.to('cuda'), cached.features.to('cuda'))
        behavior = behavior_targets.get(case['id'])
        if behavior is not None:
            behavior = model.BehaviorTargets(behavior.delivered256.to('cuda').clone(), behavior.logits.to('cuda').clone())
        eligible = bool(case['parameters']['image_eligible'])
        budget.reserve('candidate_source_forwards', {'snapshot': label, 'case': case['id']})
        if eligible:
            budget.reserve('perceptual_source_forwards', {'snapshot': label, 'case': case['id']})
        with torch.autocast(device_type='cuda', enabled=False), route.isolated_style(architecture) as route_capture, \
                capture_candidate_path(net, arrays=True) as captured:
            total, terms, prediction = model.forward_training_terms(net, teacher, perceptual,
                _tensor(observed['input'], torch), _tensor(observed['clean512'], torch),
                image_eligible=eligible, codes=codes, behavior=behavior)
        route.validate_capture(route_capture)
        require(all(value == 'torch.float32' for value in captured['dtypes'].values()),
                'Source forward inherited non-FP32 dtype')
        source_directory = directory / 'sources' / case['id']
        exported = _save_prediction(budget, source_directory, prediction, captured,
                                    observed['target256'], model, np, Image)
        require(set(terms) == ({'token_ce', 'latent_mse', 'coarse_rgb_l1'} |
                ({'pixel_l1_rgb', 'lpips_vgg'} if eligible else set()) |
                ({'behavior_rgb_l1', 'behavior_token_kl'} if behavior is not None else set())),
                'Actual source objective eligibility differs')
        exposure_arrays = []
        component_norms = {}
        for component in COMPONENTS:
            if component not in terms:
                continue
            budget.reserve('component_gradient_queries', {'snapshot': label, 'case': case['id'], 'component': component})
            loss = model.LOSS_WEIGHTS[component] * terms[component] / 16
            gradients = torch.autograd.grad(loss, [parameters[name] for name in names],
                                            allow_unused=True, retain_graph=True, create_graph=False)
            current = {}
            for name, gradient in zip(names, gradients):
                if gradient is None:
                    current[name] = None
                    continue
                require(gradient.dtype == torch.float32 and torch.isfinite(gradient).all().item(),
                        'Nonfinite/non-FP32 source gradient')
                value = np.ascontiguousarray(gradient.detach().cpu().numpy().copy(), dtype=np.float32)
                current[name] = value
                if components[component][name] is None:
                    components[component][name] = value.copy()
                else:
                    np.add(components[component][name], value, out=components[component][name])
            if component in route.IMAGE_TERMS:
                require(all(current[name] is None for name in names if parameter_partition(name) != 'fusion'),
                        'Isolated image gradient unexpectedly reaches encoder/predictor')
            if component == 'latent_mse':
                require(all(current[name] is None for name in names if parameter_partition(name) != 'encoder'),
                        'HQ-feature gradient unexpectedly reaches predictor/fusion')
            if component in ('token_ce', 'behavior_token_kl'):
                require(all(current[name] is None for name in names if parameter_partition(name) == 'fusion'),
                        'Token objective unexpectedly reaches fusion')
            stats = gradient_statistics(names, current, np)
            component_norms[component] = stats['groups']
            exposure_arrays.append(current)
            del gradients, loss, stats
        budget.reserve('full_gradient_queries', {'snapshot': label, 'case': case['id']})
        gradients = torch.autograd.grad(total / 16, [parameters[name] for name in names],
                                        allow_unused=True, retain_graph=False, create_graph=False)
        full_arrays = {name: None if gradient is None else
            np.ascontiguousarray(gradient.detach().cpu().numpy().copy(), dtype=np.float32)
            for name, gradient in zip(names, gradients)}
        sum_check = gradient_sum_check(exposure_arrays, full_arrays, names, np, atol=2e-5, rtol=2e-4)
        for name, value in full_arrays.items():
            if value is not None:
                if full_aggregate[name] is None:
                    full_aggregate[name] = value.copy()
                else:
                    np.add(full_aggregate[name], value, out=full_aggregate[name])
        record = {'snapshot': label, 'case': case['id'], 'update': case['update'], 'slot': case['slot'],
            'reference_id': case['reference_id'], 'profile': case['profile'], 'image_eligible': eligible,
            'terms': {name: float(value.detach().cpu()) for name, value in terms.items()},
            'weighted_terms_before_division16': {name: model.LOSS_WEIGHTS[name] * float(value.detach().cpu())
                                                for name, value in terms.items()},
            'aggregation_divisor': 16, 'component_group_gradient_summaries': component_norms,
            'full_component_vector_agreement': sum_check, 'route': route_capture, 'outputs': exported,
            'parameter_grad_remains_None': all(parameter.grad is None for parameter in net.parameters())}
        require(record['parameter_grad_remains_None'], 'autograd.grad unexpectedly accumulated parameter.grad')
        budget.json(source_directory / 'record.json', record)
        source_receipts.append({'case': case['id'], 'full_component_vector_agreement': sum_check})
        del total, terms, prediction, gradients, full_arrays, exposure_arrays, current, captured, behavior, codes
    aggregate_sum_check = gradient_sum_check([components[name] for name in COMPONENTS],
        full_aggregate, names, np, atol=2e-5, rtol=2e-4)
    del full_aggregate
    gradient_exports = {}
    for component in COMPONENTS:
        stats = gradient_statistics(names, components[component], np)
        for name in names:
            value = components[component][name]
            if value is not None:
                require(re.fullmatch(r'[A-Za-z0-9_.]+', name) is not None, 'Unsafe tensor export name')
                stats['parameters'][name]['file'] = budget.array(
                    directory / 'gradients' / component / (name + '.npy'), value, np)
        stats.update({'component': component, 'weighted': True, 'source_exposures': 16,
                      'aggregation_divisor': 16, 'accumulation_dtype': 'float32',
                      'accumulation_order': [case['id'] for case in cases],
                      'None_is_not_zero': True, 'only_nonNone_physical_arrays_saved': True})
        budget.json(directory / 'gradients' / component / 'manifest.json', stats)
        gradient_exports[component] = {'groups': stats['groups'],
            'manifest': (directory / 'gradients' / component / 'manifest.json').relative_to(budget.root).as_posix()}
    require(tree_digest(net.state_dict(), torch) == base_model_digest and
            tree_digest(optimizer.state_dict(), torch) == base_optimizer_digest and
            tree_digest(rng_state(torch, np), torch) == base_rng_digest,
            'Gradient-source extraction changed base model/Adam/RNG')
    budget.json(directory / 'source_summary.json', {'snapshot': label, 'sources': source_receipts,
        'aggregate_vector_agreement': aggregate_sum_check, 'components': gradient_exports,
        'base_model_sha256': base_model_digest, 'base_Adam_sha256': base_optimizer_digest,
        'base_RNG_sha256': base_rng_digest, 'fixed_component_order': list(COMPONENTS),
        'full_query_vectors_not_exported': True, 'source_extraction_committed_updates': 0})
    branch_summaries, baseline = [], {}
    budget.set_phase('probe')
    @contextmanager
    def independent_branch(condition, branch_directory):
        restore_base()
        try:
            yield
        finally:
            active_error = sys.exc_info()[0] is not None
            recovery = {'snapshot': label, 'condition': condition,
                        'actual_completed_disposable_Adam_steps': budget.actual_steps,
                        'optimizer_inflight': budget.optimizer_inflight,
                        'committed_training_updates': 0}
            try:
                restore_base()
                recovery.update({'exact_base_model_Adam_RNG_restored': True,
                    'model_sha256': base_model_digest, 'Adam_sha256': base_optimizer_digest,
                    'RNG_sha256': base_rng_digest, 'Adam_counter': before_counter})
            except BaseException as error:
                recovery.update({'exact_base_model_Adam_RNG_restored': False,
                                 'restoration_error': type(error).__name__ + ': ' + str(error)})
                atomic_json(branch_directory / 'restoration.json', recovery)
                if not active_error:
                    raise
            else:
                # A reserved1MiB payload margin permits terminal/recovery metadata
                # even if a phase or normal evidence reservation has just stopped.
                atomic_json(branch_directory / 'restoration.json', recovery)
                budget.payload_bytes += (branch_directory / 'restoration.json').stat().st_size
                budget.metadata_bytes += (branch_directory / 'restoration.json').stat().st_size
                budget.regular_files += 1
    for condition in CONDITIONS:
        safe_condition = condition.replace(':', '__')
        branch_directory = directory / 'conditions' / safe_condition
        with independent_branch(condition, branch_directory):
            step_record = None
            if condition != 'baseline':
                omitted = condition.split(':', 1)[1] if condition.startswith('drop:') else None
                assigned = compose_gradients(components, names, np, omitted=omitted,
                                            momentum_only=condition == 'momentum_only')
                for name in names:
                    parameters[name].grad = torch.from_numpy(assigned[name].copy()).to(device='cuda', dtype=torch.float32)
                require(all(parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all().item()
                            for name in names), 'Disposable Adam branch must assign all350 finite gradients')
                if condition == 'momentum_only':
                    require(all(parameters[name].grad.count_nonzero().item() == 0 for name in names),
                            'Momentum-only branch needs explicit zero gradients, never None')
                assigned_stats = gradient_statistics(names, assigned, np)
                budget.reserve('optimizer_step_attempts', {'snapshot': label, 'condition': condition,
                    'base_counter': before_counter, 'disposable_counter_after': before_counter + 1})
                budget.optimizer_inflight = True
                optimizer.step()  # A REAL disposable Adam update; never a committed training step.
                budget.optimizer_inflight = False
                budget.actual_steps += 1
                budget.counts['disposable_optimizer_steps'] += 1
                require(optimizer_step_values(optimizer) == [float(before_counter + 1)] * 350,
                        'Actual disposable Adam counters failed0to1/50to51')
                require(all(torch.isfinite(value).all().item() for value in net.state_dict().values()),
                        'Nonfinite disposable model state')
                step_record = {'actual_Adam_step': True, 'step_before': before_counter,
                    'step_after': before_counter + 1, 'trainable_tensors_with_explicit_gradients': 350,
                    'momentum_only_explicit_zero': condition == 'momentum_only', 'omitted_component': omitted,
                    'fixed_rates': rates, 'schedule_advanced': False,
                    'assigned_gradient_groups': assigned_stats['groups'],
                    'assigned_gradient_sha256_by_tensor': {name: assigned_stats['parameters'][name]['sha256'] for name in names},
                    'Adam_after_sha256': tree_digest(optimizer.state_dict(), torch),
                    'parameter_deltas': parameter_delta_record(net, base_model, torch, np)}
                del assigned, assigned_stats
            net.eval(); net.quantize.eval(); net.generator.eval()
            branch_model_digest = tree_digest(net.state_dict(), torch)
            probe_records = []
            for case in cases:
                budget.reserve('candidate_probe_forwards', {'snapshot': label, 'condition': condition, 'case': case['id']})
                with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), \
                        route.isolated_style(architecture) as route_capture, capture_candidate_path(net) as captured:
                    prediction = model.infer(net, _tensor(arrays[case['id']]['input'], torch))
                require(len(route_capture['calls']) == 1 and route_capture['original_callable_restored'],
                        'Probe isolation scope failed to restore callable')
                exported = _save_prediction(budget, branch_directory / case['id'], prediction, captured,
                    arrays[case['id']]['target256'], model, np, Image)
                record = {'snapshot': label, 'condition': condition, 'case': case['id'], 'reference_id': case['reference_id'],
                    'profile': case['profile'], 'outputs': exported, 'route': route_capture}
                budget.json(branch_directory / case['id'] / 'record.json', record)
                probe_records.append({'case': case['id'], 'raw_metrics': exported['raw_metrics'],
                                      'png_metrics': exported['png_metrics']})
                if condition == 'baseline':
                    baseline[case['id']] = exported
                del prediction, captured
            require(tree_digest(net.state_dict(), torch) == branch_model_digest and
                    tree_digest(rng_state(torch, np), torch) == base_rng_digest,
                    'Probe forwards changed branch model or restored RNG')
            require(model.frozen_state_digest(net) == frozen['candidate'] and
                    model.state_digest(teacher) == frozen['teacher'] and model.state_digest(perceptual) == frozen['perceptual'],
                    'Diagnostic branch changed frozen module')
            require(all(parameter.grad is None for module in (teacher, perceptual) for parameter in module.parameters()),
                    'Frozen teacher/perceptual accumulated gradients')
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
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), \
                route.isolated_style(architecture), capture_candidate_path(net) as captured:
            prediction = model.infer(net, _tensor(arrays[case['id']]['input'], torch))
        native = prediction.detach().float().cpu().numpy()
        raw = model.delivered_rgb(prediction).detach()[0].permute(1, 2, 0).cpu().numpy()
        expected = baseline[case['id']]
        for name, value in (('native512', native), ('raw256', raw), ('tokens', captured['tokens'])):
            retained = np.load(bounded(budget.root, expected[name]['path']), allow_pickle=False)
            require(np.array_equal(value, retained), 'Final repeat baseline parity failed: ' + name)
        repeats.append({'case': case['id'], 'native_RAW_tokens_bit_exact': True})
        del prediction, captured, native, raw, retained
    require(tree_digest(net.state_dict(), torch) == base_model_digest and
            tree_digest(optimizer.state_dict(), torch) == base_optimizer_digest and
            tree_digest(rng_state(torch, np), torch) == base_rng_digest,
            'Final base model/Adam/RNG restoration failed')
    record = {'snapshot': label, 'base_counter': before_counter, 'branches': branch_summaries,
        'final_repeat': repeats, 'final_base_model_sha256': base_model_digest,
        'final_base_Adam_sha256': base_optimizer_digest, 'final_base_RNG_sha256': base_rng_digest,
        'final_exact_model_Adam_RNG_restoration': True, 'model_checkpoint_written': False,
        'committed_training_updates': 0, 'actual_disposable_Adam_steps_this_snapshot': 9}
    budget.json(directory / 'snapshot_complete.json', record)
    return record


def run(root, protocol, protocol_sha):
    root = require_vm_path(root)  # MUST precede framework/model imports/checkpoint loading.
    from research.codeformer_loss_response_v1 import contract
    contract.verify_protocol(protocol)
    root = contract.manual_root(root)
    parent = contract.resolve_parent(root)
    require(protocol['snapshots'] == ['original0', 'stopped50'],
            'Exactly original0/stopped50 diagnostic snapshots required')
    require(protocol['branches'] == list(CONDITIONS),
            'Fixed10 diagnostic conditions differ')
    require(EXPECTED_CAPS == contract.COUNTS and list(COMPONENTS) == contract.COMPONENTS,
            'Worker/contract finite census or fixed component order differs')
    require(protocol['seed'] == 20261015 and protocol['outputs'] == 'outputs/loss_response_v1',
            'Fixed diagnostic seed/output namespace differs')
    limits = protocol['limits']
    require(limits['worker_seconds'] == 1800 and limits['initialization_seconds'] == 300 and
            limits['source_seconds'] == 600 and limits['probe_seconds'] == 900,
            'Prospective finite wall-time limits differ')
    require(limits['CUDA_reserved_bytes'] == 22 * 1024**3 and limits['RSS_bytes'] == 8 * 1024**3 and
            limits['result_payload_bytes'] == 4 * 1024**3 and
            limits['required_free_bytes'] == 14 * 1024**3 and limits['reserve_free_bytes'] == 4 * 1024**3,
            'Prospective memory/storage limits differ')
    require(shutil.disk_usage(root).free >= limits['required_free_bytes'], 'Require14GiB free after installation')
    out = bounded(root, protocol['outputs'], must_exist=False)
    require(not out.exists(), 'No automatic repeat/resume; retain existing diagnostic output')
    out.mkdir(parents=True)
    request = {'signal': None}
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, lambda value, frame: request.__setitem__('signal', value))
    budget = None
    started = time.monotonic()
    try:
        with pid_lock(root, protocol_sha):
            # Extend only the already-selected new research namespace. The parent
            # is a frozen dependency root, never a destination for diagnostic writes.
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import research
            research.__path__.append(str(parent / 'research'))
            if str(parent) not in sys.path:
                sys.path.append(str(parent))
            import torch
            import numpy as np
            from PIL import Image
            from research.codeformer_adaptation_v2 import model, data
            from research.codeformer_adain_gradient_isolation_v1 import gradient_route
            from research.codeformer_adaptation_v1.perceptual import load_perceptual
            from third_party.codeformer import codeformer_arch as architecture
            require_l4(torch)
            budget = Budget(root, out, limits, torch, request)
            budget.started = started
            budget.elapsed['initialization'] = time.monotonic() - started
            budget.set_phase('initialization')
            require(sum(EXPECTED_CAPS[name] for name in FORWARD_COUNTERS) == 425,
                    'Prospective neural-forward arithmetic differs')
            random.seed(protocol['seed']); np.random.seed(protocol['seed'])
            torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            optimizer_contract = protocol['optimizer']
            require(optimizer_contract['betas'] == [.9, .99] and optimizer_contract['eps'] == 1e-8 and
                    optimizer_contract['weight_decay'] == 0.0, 'Adam metadata contract differs')
            rates = optimizer_contract['fixed_group_rates']
            require(rates == {'encoder_token': 1e-5, 'fusion': 2e-5}, 'Fixed diagnostic LR differs')
            require(model.LOSS_WEIGHTS == protocol['loss_weights'], 'Loss weights differ from frozen model')
            cases, arrays = _load_cases(root, parent, protocol, data, np, Image)
            initializer = _parent_asset(parent, protocol, 'initializer')
            original_path = _parent_asset(parent, protocol, 'original_checkpoint')
            stopped_path = _parent_asset(parent, protocol, 'stopped_checkpoint')
            require(sha256(stopped_path) == STOPPED_CHECKPOINT_SHA256, 'Exact stopped50 checkpoint required')
            # Conservative simultaneous-live CPU estimate, before model construction.
            # Includes checkpoint/model/moments, CPU component/full/query copies,
            # source arrays and1GiB Python/runtime allowance; never claims measured peak.
            projection = projected_resources(stopped_path.stat().st_size)
            require(projection['projected_RAM_bytes'] <= limits['RSS_bytes'],
                    'Projected CPU memory exceeds8GiB; refuse before model load')
            require(projection['projected_CUDA_bytes'] <= limits['CUDA_reserved_bytes'],
                    'Projected GPU memory exceeds22GiB; refuse before model load')
            available_RAM = linux_available_memory_bytes()
            available_CUDA, total_CUDA = torch.cuda.mem_get_info()
            require(available_RAM >= projection['projected_RAM_bytes'] and
                    available_CUDA >= projection['projected_CUDA_bytes'],
                    'Available host/GPU memory is below conservative prospective allocation plan')
            budget.json(out / 'resource_plan.json', {**projection,
                'RAM_cap_bytes': limits['RSS_bytes'], 'GPU_reserved_cap_bytes': limits['CUDA_reserved_bytes'],
                'available_RAM_bytes_before_model_load': available_RAM,
                'available_CUDA_bytes_before_model_load': available_CUDA, 'total_CUDA_bytes': total_CUDA,
                'GPU_activation_peak_uses_conservative_allowance_and_runtime_guard': True,
                'source_component_array_bytes_per_state': 681249536,
                'native_RAW_PNG_source_probe_outputs': '32 sources and320 probes;32 repeat outputs compared only',
                'no_model_or_optimizer_checkpoint_output': True})
            net, census = model.load_trainable(initializer, device='cuda')
            names = [name for name, parameter in net.named_parameters() if parameter.requires_grad]
            parameters = dict(net.named_parameters())
            require(len(names) == 350 and census['trainable_parameters'] == 59788608,
                    'Frozen trainable census differs')
            teacher = model.load_teacher(_parent_asset(parent, protocol, 'teacher'), device='cuda')
            perceptual = load_perceptual(_parent_asset(parent, protocol, 'perceptual_features').parent, device='cuda')
            compatibility = model.assert_shared_codebook(net, teacher)
            frozen = {'teacher': model.state_digest(teacher), 'perceptual': model.state_digest(perceptual),
                      'candidate': model.frozen_state_digest(net)}
            groups = []
            for name in ('encoder_token', 'fusion'):
                chosen = [parameter for key, parameter in net.named_parameters() if parameter.requires_grad and
                          (key.startswith('fuse_convs_dict.') == (name == 'fusion'))]
                require(len(chosen) == (72 if name == 'fusion' else 278), 'Adam partition census differs')
                groups.append({'params': chosen, 'lr': rates[name], 'base_lr': rates[name], 'name': name})
            optimizer = torch.optim.Adam(groups, betas=tuple(optimizer_contract['betas']),
                                         eps=optimizer_contract['eps'], weight_decay=0.0)
            original = torch.load(original_path, map_location='cpu', weights_only=True)
            require(original['completed_updates'] == 0 and original['partial_microbatches_discarded'] == 0,
                    'Original0 snapshot counter differs')
            original_model = original['model']
            original_optimizer = original['optimizer']
            require(not original_optimizer['state'] and
                    tree_digest(original_model, torch) == tree_digest(net.state_dict(), torch),
                    'Original0 checkpoint differs from untouched initializer/empty Adam state')
            initial_model_digest = model.state_digest(net)
            require(initial_model_digest == '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
                    'Untouched original0 model state differs')
            # One declared prerequisite query only; no extra CUDA gradient fixture.
            fixture = torch.linspace(-.8, .8, 3 * 64 * 64, device='cuda', dtype=torch.float32).reshape(1, 3, 64, 64)
            fixture.requires_grad_(True)
            fixture_target = torch.flip(fixture.detach(), dims=[3]) * .5
            budget.reserve('perceptual_prerequisite_forwards')
            prerequisite_loss = perceptual(fixture, fixture_target)
            budget.reserve('perceptual_prerequisite_gradient_queries')
            prerequisite_gradient, = torch.autograd.grad(prerequisite_loss, fixture)
            require(torch.isfinite(prerequisite_gradient).all().item() and
                    prerequisite_gradient.abs().max().item() > 0, 'LPIPS input gradient prerequisite failed')
            budget.json(out / 'perceptual_prerequisite.json', {'pass': True,
                'loss': float(prerequisite_loss.detach().cpu()), 'gradient_maximum': float(prerequisite_gradient.abs().max().cpu()),
                'frozen_state_unchanged': model.state_digest(perceptual) == frozen['perceptual'],
                'query_is_budgeted': True})
            del fixture, fixture_target, prerequisite_loss, prerequisite_gradient
            clean_codes = {}
            for case in cases:
                reference_id = case['reference_id']
                if reference_id in clean_codes:
                    continue
                observed = arrays[case['id']]
                budget.reserve('HQ_teacher_forwards', {'reference_id': reference_id})
                codes = model.hq_targets(teacher, _tensor(observed['clean512'], torch), codebook=net.quantize)
                retained_indices = np.load(bounded(parent, case['hq_indices_path']), allow_pickle=False)
                retained_features = np.load(bounded(parent, case['hq_features_path']), allow_pickle=False)
                record = read_json(bounded(parent, case['hq_record_path']))
                require(np.array_equal(codes.indices.cpu().numpy(), retained_indices) and
                        np.array_equal(codes.features.cpu().numpy(), retained_features), 'Fresh8 HQ targets differ from retained arrays')
                require(record['clean512_float32_sha256'] == array_digest(observed['clean512'], np) and
                        record['teacher_state_sha256'] == frozen['teacher'] and
                        record['reference_id'] == reference_id and record['role'] == 'train' and
                        record['source_path'] == case['source_path'] and
                        record['restoration_codebook_state_sha256'] == model.state_digest(net.quantize),
                        'Fresh HQ provenance differs')
                clean_codes[reference_id] = model.CleanCodes(codes.indices.cpu().clone(), codes.features.cpu().clone())
                target_dir = out / 'targets' / reference_id
                budget.array(target_dir / 'clean512.npy', observed['clean512'], np)
                budget.array(target_dir / 'clean256.npy', observed['target256'], np)
                budget.array(target_dir / 'indices.npy', retained_indices, np)
                budget.array(target_dir / 'features.npy', retained_features, np)
                budget.json(target_dir / 'parity.json', {'pass': True, 'reference_id': reference_id,
                    'retained_HQ_record': case['hq_record_path'], 'clean256_policy': 'TRAIN source RGB1024 Pillow LANCZOS256; metric-only',
                    'clean256_array_sha256': array_digest(observed['target256'], np)})
                del codes, retained_indices, retained_features
            # Four initializer comparisons: two candidate plus two unchanged benchmark.
            from pretrained_face_restoration import load_face_restorer
            from research.codeformer_adaptation_v1.common import file_digest_compat
            from unittest.mock import patch
            with patch.object(hashlib, 'file_digest', file_digest_compat, create=True):
                benchmark, benchmark_provenance = load_face_restorer(initializer, device='cuda')
            parity = []
            for case in cases[:2]:
                value = _tensor(arrays[case['id']]['input'], torch)
                with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                    budget.reserve('initializer_parity_candidate_forwards', {'case': case['id']})
                    actual = model.delivered_rgb(model.infer(net, value))
                    budget.reserve('initializer_parity_benchmark_forwards', {'case': case['id']})
                    expected = benchmark(value)
                maximum_error = float((actual - expected).abs().max().cpu())
                require(maximum_error <= protocol['numerical']['initializer_parity_maximum_error'],
                        'Fresh original initializer/benchmark parity failed')
                parity.append({'case': case['id'], 'bit_exact': torch.equal(actual, expected),
                               'maximum_absolute_error': maximum_error,
                               'bound': protocol['numerical']['initializer_parity_maximum_error']})
                del value, actual, expected
            del benchmark
            torch.cuda.empty_cache()
            behavior_teacher = model.load_behavior_teacher(initializer, device='cuda')
            behavior_targets = {}
            for case in cases:
                if case['slot'] != 3:
                    continue
                budget.reserve('behavior_teacher_forwards', {'case': case['id']})
                fresh = model.behavior_targets(behavior_teacher, _tensor(arrays[case['id']]['input'], torch))
                rgb = np.load(bounded(parent, case['behavior_delivered_path']), allow_pickle=False)
                logits = np.load(bounded(parent, case['behavior_logits_path']), allow_pickle=False)
                anchor_record = read_json(bounded(parent, case['behavior_record_path']))
                require(anchor_record['update'] == case['update'] and anchor_record['slot'] == 3 and
                        anchor_record['reference_id'] == case['reference_id'] and
                        anchor_record['input256_float32_sha256'] == case['record']['input256_float32_sha256'] and
                        anchor_record['teacher_state_sha256'] == initial_model_digest,
                        'Retained behavior anchor provenance differs')
                require(np.array_equal(fresh.delivered256.cpu().numpy(), rgb) and
                        np.array_equal(fresh.logits.cpu().numpy(), logits), 'Fresh4 behavior targets differ from retained loss arrays')
                behavior_targets[case['id']] = model.BehaviorTargets(
                    torch.from_numpy(rgb.copy()).clone(), torch.from_numpy(logits.copy()).clone())
                budget.array(out / 'behavior_targets' / case['id'] / 'delivered256.npy', rgb, np)
                budget.array(out / 'behavior_targets' / case['id'] / 'logits.npy', logits, np)
                del fresh, rgb, logits
            require(not any(parameter.grad is not None for parameter in behavior_teacher.parameters()), 'Behavior teacher gradient exists')
            del behavior_teacher
            torch.cuda.empty_cache()
            baseline_rng = original['rng']
            budget.json(out / 'initialization.json', {'pass': True, 'census': census,
                'algorithm_settings': algorithm_metadata(torch), 'FP32_autocast_disabled': True,
                'torch': torch.__version__, 'CUDA': torch.version.cuda, 'Python': sys.version,
                'device': torch.cuda.get_device_name(0), 'teacher_codebook_compatibility': compatibility,
                'initializer_parity': parity, 'benchmark_provenance': benchmark_provenance,
                'source_observations': 16, 'source_photographic_TRAIN_ids': 8, 'source_image_eligible': 12,
                'actual_gradient_source_scope': 'same16 fixed TRAIN observations as probes; no DEV/native/FINAL',
                'fixed_diagnostic_rates': rates, 'schedule_extension': False,
                'optimizer_default_metadata': [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups],
                'frozen': frozen, 'original0_checkpoint_RNG_sha256': tree_digest(baseline_rng, torch),
                'RNG_after_prerequisites_sha256': tree_digest(rng_state(torch, np), torch)})
            # Only after proving original parity are stopped weights/moments read.
            stopped = torch.load(stopped_path, map_location='cpu', weights_only=True)
            require(stopped['completed_updates'] == 50 and stopped['partial_microbatches_discarded'] == 0,
                    'Stopped base is not an exact committed50-update state')
            require(stopped['protocol_sha256'] == '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5',
                    'Stopped base protocol differs')
            require(len(stopped['optimizer']['state']) == 350, 'Stopped Adam must retain all350 moment states')
            state_summaries = []
            for label in ('original0', 'stopped50'):
                base_model = original_model if label == 'original0' else stopped['model']
                base_optimizer = original_optimizer if label == 'original0' else stopped['optimizer']
                base_rng = baseline_rng if label == 'original0' else stopped['rng']
                before_counter = 0 if label == 'original0' else 50
                state_summaries.append(_run_snapshot(label, before_counter, base_model, base_optimizer, base_rng,
                    cases, arrays, clean_codes, behavior_targets, net, optimizer, names, parameters,
                    model, gradient_route, architecture, teacher, perceptual, frozen,
                    budget, rates, torch, np, Image))
            budget.check()
            require(budget.counts == EXPECTED_CAPS and budget.actual_steps == 18,
                    'Completed diagnostic differs from exact declared calls/18 actual disposable steps')
            receipt = {'format': 'codeformer-loss-response-completion-v1', 'complete': True,
                'protocol_sha256': protocol_sha, 'counts': budget.counts, **call_totals(budget.counts),
                'actual_completed_disposable_Adam_steps': budget.actual_steps,
                'actual_disposable_Adam_steps': budget.actual_steps,
                'committed_training_updates': 0, 'backward_calls': 0, 'model_checkpoint_written': False,
                'branches_chain': False, 'model_qualified': False, 'resume_authorized': False,
                'snapshots': state_summaries, 'phase_seconds': budget.timings(),
                'worker_seconds': time.monotonic() - started, 'algorithm_settings': algorithm_metadata(torch),
                'final_frozen': {'teacher': model.state_digest(teacher), 'perceptual': model.state_digest(perceptual),
                                 'candidate': model.frozen_state_digest(net)}}
            require(receipt['final_frozen'] == frozen, 'Terminal frozen state differs')
            budget.json(out / 'completion.json', receipt)
            return receipt
    except BaseException as error:
        failure = {'format': 'codeformer-loss-response-failure-v1', 'complete': False,
            'protocol_sha256': protocol_sha, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc()[-32768:], 'worker_seconds': time.monotonic() - started,
            'counts': None if budget is None else budget.counts,
            'actual_completed_disposable_Adam_steps': 0 if budget is None else budget.actual_steps,
            'optimizer_inflight_at_failure': False if budget is None else budget.optimizer_inflight,
            'partial_disposable_step_possible': False if budget is None else budget.optimizer_inflight,
            'committed_training_updates': 0, 'backward_calls': 0, 'model_checkpoint_written': False,
            'model_qualified': False, 'resume_authorized': False, 'all_failure_evidence_retained': True}
        if not (out / 'failure.json').exists():
            atomic_json(out / 'failure.json', failure)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _transport():
    # This module must remain standard-library only and source-bound by protocol.
    from research.codeformer_loss_response_v1 import transport
    return transport


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
    transport = _transport()
    receipt = transport.verify_transfer(root, args.protocol_sha)
    protocol = receipt['protocol']
    if args.verify_transfer:
        print(json.dumps(receipt), flush=True)
    elif args.export:
        print(json.dumps(transport.export_results(root, args.protocol_sha)), flush=True)
    else:
        print(json.dumps(run(args.root, protocol, args.protocol_sha)), flush=True)


if __name__ == '__main__':
    main()
