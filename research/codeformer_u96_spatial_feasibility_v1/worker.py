"""Finite manual-L4 feasibility worker; importing performs standard-library work only.

No optimizer exists here. Two fixed, independent array applications follow the
same U96 initializer only after a common-descent certificate. All actual model
work is behind static seal, exact VM, tmux and supervisor guards inside run().
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import warnings


OUTPUT_RELATIVE = 'outputs/u96_spatial_feasibility_v1'
PACKAGE_RELATIVE = 'research/codeformer_u96_spatial_feasibility_v1'
STATUS = 'FINITE_MANUAL_FEASIBILITY_NOT_A_QUALIFIED_MODEL'
OPERATION_CAPS = {'forward': 218, 'gradient': 32, 'application': 2}
CATEGORY_CAPS = {'common_initializer_component_forwards': 16,
                 'initial_monitor_guard_forwards': 56,
                 'independent_parity_forwards': 2, 'proposal_forwards': 144}
FAILURE_CAPTURE_SECONDS = 30
FAILURE_RECEIPT_SECONDS = 5
FAILURE_FINALIZATION_SECONDS = 5
COUNTER_KEYS = ('full_forward_attempts', 'full_forward_returns',
                'gradient_query_attempts', 'gradient_queries',
                'proposal_application_attempts', 'proposal_applications',
                'optimizer_constructions', 'optimizer_steps',
                *CATEGORY_CAPS)


def need(condition, message):
    if not condition:
        raise ValueError(message)


class QualityStop(ValueError):
    """A scientific stop; preserve evidence, never retry or add epochs."""


def finite_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def linux_current_rss_bytes(text, page_size):
    fields = text.split()
    need(type(page_size) is int and page_size > 0 and len(fields) >= 2 and fields[1].isdigit(),
         'Positive Linux statm resident-page measurement required')
    return int(fields[1]) * page_size


def failure_capture_budget(elapsed_seconds):
    """Reserve10s for receipt/RNG inside1260s external worker deadline.

    Normal scientific work stops at1200s. Best-effort failure state capture is
    separately capped at30s; receipt and caller-RNG restoration each at5s. Thus
    a timely1200s internal stop gets at most40s finalization, within1260s.
    External SIGTERM/KILL remains authoritative if native code cannot yield.
    """
    need(type(elapsed_seconds) in (int, float) and math.isfinite(elapsed_seconds) and elapsed_seconds >= 0,
         'Finite nonnegative elapsed time required')
    return max(0.0, min(float(FAILURE_CAPTURE_SECONDS), 1260.0 - float(elapsed_seconds) -
                        FAILURE_RECEIPT_SECONDS - FAILURE_FINALIZATION_SECONDS))


def supervisor_context(expected_sha, *, environ, parent_pid):
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha),
         'Exact protocol SHA256 required')
    need(environ.get('FORENSIC_U96_SPATIAL_SUPERVISED') == '1' and
         environ.get('FORENSIC_U96_SPATIAL_PROTOCOL') == expected_sha and
         environ.get('FORENSIC_U96_SPATIAL_SUPERVISOR_PID') == str(parent_pid),
         'Exact finite supervisor environment and actual parent PID required')
    return True


class OperationLedger:
    """Pure reservation state machine: durable callback precedes each operation."""
    def __init__(self, persist):
        self.persist = persist
        self.counters = dict.fromkeys(COUNTER_KEYS, 0)
        self.pending = {}
        self.sequence = 0

    def reserve(self, kind, **metadata):
        need(kind in OPERATION_CAPS, 'Undeclared operation kind')
        need(not set(metadata) & {'kind', 'token', 'counters'}, 'Operation metadata cannot replace ledger fields')
        attempt = {'forward': 'full_forward_attempts', 'gradient': 'gradient_query_attempts',
                   'application': 'proposal_application_attempts'}[kind]
        need(self.counters[attempt] < OPERATION_CAPS[kind], 'Finite operation cap exhausted')
        if kind == 'forward':
            category = metadata.get('category')
            need(category in CATEGORY_CAPS and self.counters[category] < CATEGORY_CAPS[category],
                 'Finite declared forward category required')
            self.counters[category] += 1
        self.counters[attempt] += 1
        self.sequence += 1
        token = self.sequence
        self.pending[token] = {'kind': kind, **metadata}
        self.persist({'kind': kind + '_reserved', 'token': token,
                      'counters': dict(self.counters), **metadata})
        return token

    def returned(self, token, **metadata):
        need(type(token) is int and token in self.pending, 'One known unmatched reservation required')
        need(not set(metadata) & {'kind', 'token', 'counters'}, 'Return metadata cannot replace ledger fields')
        pending = self.pending[token]
        key = {'forward': 'full_forward_returns', 'gradient': 'gradient_queries',
               'application': 'proposal_applications'}[pending['kind']]
        # If this durable write fails, the prior reservation remains unknown.
        counters = dict(self.counters)
        counters[key] += 1
        self.persist({**pending, **metadata, 'kind': pending['kind'] + '_returned',
                      'token': token, 'counters': counters})
        self.counters = counters
        del self.pending[token]

    def snapshot(self):
        return {'counters': dict(self.counters),
                'unknown_reservations': [{'token': token, **row} for token, row in sorted(self.pending.items())],
                'operation_outcome_unknown': bool(self.pending)}


class DurableStore:
    """Exclusive bounded namespace; interrupted payloads remain retained.

    Payloads use exclusive creation, with their direct pins emitted only after
    flush/fsync succeeds. An interrupted payload remains at its original name,
    is tracked as incomplete in partial_files and is never automatically reused.
    No source, failed payload or old output is deleted. No hard links are made.
    """
    def __init__(self, root, out, maximum_bytes, pin):
        self.root = Path(root).absolute()
        self.out = Path(out).absolute()
        need(self.out.is_relative_to(self.root) and type(maximum_bytes) is int and maximum_bytes > 0,
             'Bounded output below exact root required')
        need(not self.out.exists() and not self.out.is_symlink(), 'Prior output is immutable')
        self.out.parent.mkdir(exist_ok=True)
        for path in (self.out.parent, *self.out.parent.parents):
            info = path.lstat()
            need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
                 not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked output ancestry refused')
        self.out.mkdir(exist_ok=False)
        self.maximum = maximum_bytes
        self.pin = pin
        self.sizes = {}
        self.partial_files = set()

    @property
    def bytes(self):
        return sum(self.sizes.values())

    def path(self, relative):
        relative = Path(relative)
        need(not relative.is_absolute() and '..' not in relative.parts and relative.parts,
             'Safe relative output path required')
        path = self.out / relative
        need(path.is_relative_to(self.out), 'Output scope escape refused')
        current = path.parent
        missing = []
        while not current.exists():
            missing.append(current)
            current = current.parent
        need(current.is_relative_to(self.out), 'Output parent escaped namespace')
        for ancestor in (current, *tuple(p for p in current.parents if p.is_relative_to(self.out))):
            info = ancestor.lstat()
            need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
                 not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked output directory refused')
        for directory in reversed(missing):
            directory.mkdir(exist_ok=False)
        return path

    def _sync_directory(self, path):
        if os.name == 'posix':
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def write(self, relative, payload, *, emergency=False):
        need(type(payload) is bytes, 'Exact bytes payload required')
        path = self.path(relative)
        need(not path.exists() and not path.is_symlink(), 'Existing complete/partial output is immutable')
        reserve = 0 if emergency else 131072
        need(self.bytes + len(payload) <= self.maximum - reserve, 'Finite output cap before payload')
        key = path.relative_to(self.out).as_posix()
        self.partial_files.add(key)
        try:
            with path.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if path.exists():
                self.sizes[key] = path.stat().st_size
        self._sync_directory(path.parent)
        self.partial_files.remove(key)
        return self.pin(self.root, path)

    def json(self, relative, value, *, emergency=False):
        return self.write(relative, finite_json(value), emergency=emergency)

    def append(self, relative, value):
        path = self.path(relative)
        key = path.relative_to(self.out).as_posix()
        info = path.lstat()
        need(key in self.sizes and stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             info.st_nlink == 1 and info.st_size == self.sizes[key], 'Owned unchanged single-link log required')
        payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        need(self.bytes + len(payload) <= self.maximum - 131072, 'Finite output cap before log')
        try:
            with path.open('ab') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            self.sizes[key] = path.stat().st_size

    def status(self, value):
        path = self.path('status.json')
        pending = self.path('status.pending.json')
        need(not pending.exists(), 'Interrupted status remains immutable')
        payload = finite_json(value)
        need(self.bytes + len(payload) <= self.maximum - 131072, 'Finite status cap')
        need(not path.is_symlink(), 'Linked status refused')
        try:
            with pending.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if pending.exists():
                self.sizes['status.pending.json'] = pending.stat().st_size
        if path.exists():
            info = path.lstat()
            need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1 and
                 info.st_size == self.sizes['status.json'], 'Owned status file differs')
        os.replace(pending, path)
        del self.sizes['status.pending.json']
        self.sizes['status.json'] = len(payload)
        self._sync_directory(path.parent)

    def audit_namespace(self):
        observed = {}
        for path in self.out.rglob('*'):
            info = path.lstat()
            need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                 'Linked output namespace refused')
            if stat.S_ISREG(info.st_mode):
                observed[path.relative_to(self.out).as_posix()] = info.st_size
            else:
                need(stat.S_ISDIR(info.st_mode), 'Nonregular output namespace entry')
        need(observed == self.sizes and self.bytes <= self.maximum,
             'Full retained namespace differs from durable byte ledger')
        return {'files': len(observed), 'bytes': self.bytes, 'complete': True}


def canonical_arrays(arrays, *, np):
    """Matches the model's sorted torch.float32 header/bytes canonical digest."""
    need(type(arrays) is dict and arrays, 'Nonempty named FP32 state required')
    digest = hashlib.sha256()
    schema = []
    for name, array in sorted(arrays.items()):
        need(type(name) is str and re.fullmatch('[A-Za-z0-9_.]{1,200}', name) and '..' not in name and
             isinstance(array, np.ndarray) and array.dtype == np.float32 and
             np.isfinite(array).all(), 'Finite named FP32 state arrays required')
        header = json.dumps([name, 'torch.float32', list(array.shape)], separators=(',', ':')).encode('utf-8')
        payload = array.tobytes(order='C')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(payload)
        schema.append({'name': name, 'shape': list(array.shape), 'dtype': 'float32',
                       'bytes': len(payload), 'bytes_sha256': byte_sha(payload)})
    return digest.hexdigest(), schema


def npz_state_payload(arrays, *, np):
    canonical, schema = canonical_arrays(arrays, np=np)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        need(set(saved.files) == set(arrays), 'Serialized complete state namespace differs')
        readback = {name: saved[name] for name in saved.files}
        need(canonical_arrays(readback, np=np)[0] == canonical and
             all(np.array_equal(readback[name], value) for name, value in arrays.items()),
             'Serialized complete state is not canonical/elementwise exact')
    return payload, {'canonical_sha256': canonical, 'arrays': schema, 'allow_pickle': False}


def run(root, protocol, expected_sha):
    """Only a source-sealed manual tmux supervisor on the exact L4 may enter."""
    from . import packet
    from . import contract

    root = Path(root).absolute()
    out = root / OUTPUT_RELATIVE
    need(not out.exists() and not out.is_symlink(), 'Prior output immutable; no resume or retry')
    start = time.monotonic()
    phase = 'static_VM_supervisor_guards'
    failed_phase = None
    store = None
    torch = None
    np = None
    candidate = None
    reference = None
    p = None
    previous_term = None
    caller_rng = None
    run_rng = None
    last_complete_state = None
    initial_arrays = None
    selected_items = []
    ledger = None
    resources_ready = False
    model_construction_attempted = False
    active_state = None
    active_case = None
    gradient_warning_scopes = []
    complete_state_witnesses = 0

    def status(label, *, complete=False, quality_stop=False):
        if store is None:
            return
        snapshot = ledger.snapshot() if ledger is not None else {
            'counters': dict.fromkeys(COUNTER_KEYS, 0), 'unknown_reservations': [],
            'operation_outcome_unknown': False}
        store.status({'complete': complete, 'status': STATUS, 'label': label, 'phase': phase,
                      'protocol_sha256': expected_sha, 'elapsed_seconds': time.monotonic() - start,
                      'active_state': active_state, 'active_case': active_case, 'quality_stop': quality_stop,
                      'last_complete_state': last_complete_state, 'model_qualified': False,
                      'manual_VM_only': True, **snapshot})

    def persist_event(row):
        store.append('events.jsonl', {'UTC': utc(), 'elapsed_seconds': time.monotonic() - start,
                                     'phase': phase, **row})

    def resource(label):
        elapsed = time.monotonic() - start
        free = shutil.disk_usage(root).free
        rss = linux_current_rss_bytes(Path('/proc/self/statm').read_text(), int(os.sysconf('SC_PAGE_SIZE')))
        peak_rss = int(linux_resource.getrusage(linux_resource.RUSAGE_SELF).ru_maxrss) * 1024
        reserved = peak_cuda = 0
        if torch is not None and torch.cuda.is_initialized():
            reserved = int(torch.cuda.memory_reserved(0))
            peak_cuda = int(torch.cuda.max_memory_reserved(0))
        row = {'label': label, 'UTC': utc(), 'elapsed_seconds': elapsed, 'phase': phase,
               'free_disk_bytes': free, 'current_RSS_bytes': rss, 'peak_RSS_bytes': peak_rss,
               'CUDA_reserved_bytes': reserved, 'CUDA_peak_reserved_bytes': peak_cuda,
               'output_bytes': store.bytes if store else 0}
        if resources_ready:
            store.append('resources.jsonl', row)
        limits = contract.LIMITS
        need(elapsed < limits['worker_seconds'], 'Finite1200s worker stop')
        need(free >= limits['continuing_free_disk_bytes'], 'Continuing4GiB disk reserve stop')
        need(max(rss, peak_rss) <= limits['maximum_RSS_bytes'], 'Current/peak RSS10GiB stop')
        need(max(reserved, peak_cuda) <= limits['maximum_CUDA_reserved_bytes'], 'CUDA reserved18GiB stop')
        need(store is None or store.bytes <= limits['maximum_output_bytes'], 'Output2GiB stop')
        return row

    @contextmanager
    def bounded_phase(name):
        nonlocal phase, failed_phase
        previous_phase = phase
        phase = name
        began = time.monotonic()
        resource('before')
        remaining = contract.LIMITS['worker_seconds'] - (time.monotonic() - start)
        need(remaining > 0, 'Worker budget exhausted before phase')
        prior = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError('Finite phase stop: ' + name)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(contract.LIMITS['per_phase_seconds'], remaining))
        try:
            yield
            resource('after')
            persist_event({'kind': 'phase_returned', 'phase_seconds': time.monotonic() - began,
                           'counters': dict(ledger.counters)})
        except BaseException:
            failed_phase = name
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)
            phase = previous_phase

    @contextmanager
    def failure_time_scope(seconds, label):
        need(seconds > 0, 'No time remains for bounded failure evidence')
        prior = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError('Finite failure-finalization stop: ' + label)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)

    def checked(binding):
        need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact file binding required')
        path = packet.regular(root / binding['path'])
        need(packet.pin(root, path) == binding, 'Bound file changed: ' + binding['path'])
        return path

    try:
        p = packet.verify_static(root, expected_sha)
        need(p == protocol and p['limits'] == contract.LIMITS and
             p['feasibility'] == contract.prospective_contract(), 'Exact verified finite contract required')
        need(Path(__file__).resolve() == (root / PACKAGE_RELATIVE / 'worker.py').resolve(),
             'Worker must run from exact staged source')
        worker_pin = packet.pin(root, Path(__file__).resolve())
        expected_worker = p['files'][PACKAGE_RELATIVE + '/worker.py']
        need(worker_pin['bytes'] == expected_worker['bytes'] and worker_pin['sha256'] == expected_worker['sha256'],
             'Loaded worker differs from exact source seal')
        facts = packet.require_vm(root)
        supervisor_context(expected_sha, environ=os.environ, parent_pid=os.getppid())
        parent_path = checked(p['parent_protocol'])
        selection = contract.select_parent_cases(parent_path.read_bytes())
        parent_protocol = json.loads(parent_path.read_bytes())
        need(p['cases'] == parent_protocol['cases'] and p['guards'] == parent_protocol['guards'] and
             p['quality_policy'] == parent_protocol['quality_policy'], 'Inherited cases/guards/gates changed')
        plan = contract.operation_plan(selection)
        checked(p['inherited_DEV_evidence'])
        need(shutil.disk_usage(root).free >= contract.LIMITS['initial_free_disk_bytes'], 'Initial8GiB free required')
        import resource as linux_resource
        store = DurableStore(root, out, contract.LIMITS['maximum_output_bytes'], packet.pin)
        store.json('started.json', {'UTC': utc(), 'protocol_sha256': expected_sha, 'host': facts,
                                  'status': STATUS, 'optimizer_steps': 0, 'model_qualified': False})
        store.write('events.jsonl', b'')
        store.write('resources.jsonl', b'')
        resources_ready = True
        ledger = OperationLedger(persist_event)
        previous_term = signal.getsignal(signal.SIGTERM)
        def external_stop(signum, frame):
            raise TimeoutError('External finite SIGTERM; retain partial state and unknown reservations')
        signal.signal(signal.SIGTERM, external_stop)
        store.json('operation_plan.json', {'complete': True, 'calls': plan, 'selection': selection,
                                          'gradient_queries': 32, 'optimizer_steps': 0})
        persist_event({'kind': 'all_static_VM_supervisor_guards_passed_before_Torch', 'Torch_imported': False,
                       'counters': dict(ledger.counters)})
        with bounded_phase('idle_GPU_guard_before_Torch'):
            apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True, timeout=10)
            need(not apps.stdout.strip(), 'Competing GPU application; do not construct models')
        with bounded_phase('manual_dependencies_and_exact_FP32_policy'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            need(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'), 'Deterministic cuBLAS config required')
            import torch as torch_module
            import numpy as numpy_module
            from PIL import Image
            from torch.nn import functional as F
            from . import model as prototype
            from . import objective
            from . import direction
            from . import metrics
            torch = torch_module
            np = numpy_module
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
                 torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one actual NVIDIA L4 required')
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
            need(not torch.is_inference_mode_enabled(), 'Ordinary autograd mode required')
            caller_rng = {'Python': random.getstate(), 'NumPy': np.random.get_state(),
                          'CPU': torch.get_rng_state().cpu().clone(),
                          'CUDA': [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]}
            random.seed(20261023)
            np.random.seed(20261023)
            torch.manual_seed(20261023)
            torch.cuda.manual_seed_all(20261023)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.cuda.reset_peak_memory_stats(0)
            for name, module in (('model', prototype), ('objective', objective), ('direction', direction),
                                 ('metrics', metrics), ('contract', contract)):
                relative = PACKAGE_RELATIVE + '/' + name + '.py'
                need(Path(module.__file__).resolve() == (root / relative).resolve() and
                     packet.sha256(Path(module.__file__)) == p['files'][relative]['sha256'],
                     'Loaded sealed component differs: ' + name)
        def tensors_as_arrays(state):
            return {name: value.detach().cpu().contiguous().numpy().copy()
                    for name, value in state.items()}

        def canonical_tensors(state):
            return canonical_arrays(tensors_as_arrays(state), np=np)[0]

        def save_array(relative, value):
            array = (value.detach().cpu().contiguous().numpy().copy()
                     if isinstance(value, torch.Tensor) else np.asarray(value))
            need(array.dtype.kind in ('f', 'u', 'i') and np.isfinite(array).all(), 'Finite numerical array required')
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            return store.write(relative, stream.getvalue())

        def retain_unvalidated_values(label, values):
            """Best-effort raw bytes of an actual bad return, never a new call."""
            rows = []
            sequence = values if isinstance(values, (list, tuple)) else (values,)
            for index, value in enumerate(sequence):
                row = {'index': index, 'accepted_as_valid': False, 'python_type': type(value).__name__}
                try:
                    if isinstance(value, torch.Tensor):
                        tensor = value.detach().cpu().contiguous()
                        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().copy()
                        row.update({'original_dtype': str(tensor.dtype), 'original_shape': list(tensor.shape),
                                    'raw_tensor_bytes': save_array('failed_returns/' + label + '/' + str(index) + '_bytes.npy', raw_bytes),
                                    'nonfinite_bit_patterns_preserved': True})
                    else:
                        row['description'] = repr(value)[:1000]
                except BaseException as retention_error:
                    row['retention_error'] = repr(retention_error)
                rows.append(row)
            return store.json('failed_returns/' + label + '/unvalidated.json',
                              {'complete': True, 'values': rows, 'neural_calls_added': 0,
                               'accepted_as_valid': False, 'model_qualified': False}, emergency=True)

        def capture_state(label, *, emergency=False):
            nonlocal last_complete_state, complete_state_witnesses
            need(complete_state_witnesses < contract.LIMITS['complete_model_state_witnesses'],
                 'Exactly three normal complete-state witnesses maximum')
            captured = candidate.capture_complete_state_for_audit()
            arrays = tensors_as_arrays(captured)
            need(len(arrays) == 539 and set(arrays) == set(dict(candidate.named_parameters())),
                 'Complete539 named state capture required')
            live_sha = canonical_tensors(candidate.state_dict())
            canonical, _ = canonical_arrays(arrays, np=np)
            need(canonical == live_sha, 'Live and captured complete states differ')
            if initial_arrays is not None:
                frozen = set(arrays) - {name for name, _ in direction.PARAMETER_SCHEMA}
                need(len(frozen) == 529 and all(np.array_equal(arrays[name], initial_arrays[name]) for name in frozen),
                     'All529 frozen values must be independently exportable and exact')
            payload, witness = npz_state_payload(arrays, np=np)
            binding = store.write('states/' + label + '/full_state.npz', payload, emergency=emergency)
            with np.load(checked(binding), allow_pickle=False) as serialized:
                durable_sha = canonical_arrays({name: serialized[name] for name in serialized.files}, np=np)[0]
            need(durable_sha == canonical == live_sha, 'Durable serialized canonical state differs from live model')
            witness.update({'complete': True, 'label': label, 'file': binding, 'tensor_count': 539,
                            'live_canonical_sha256': live_sha, 'durable_canonical_sha256': durable_sha,
                            'frozen529_exact_to_initializer': initial_arrays is not None,
                            'protocol_sha256': expected_sha, 'counters': dict(ledger.counters),
                            'model_qualified': False})
            witness['receipt'] = store.json('states/' + label + '/state.json', witness, emergency=emergency)
            last_complete_state = {key: witness[key] for key in
                                   ('label', 'file', 'receipt', 'canonical_sha256', 'tensor_count',
                                    'live_canonical_sha256', 'durable_canonical_sha256')}
            complete_state_witnesses += 1
            store.audit_namespace()
            return arrays, witness

        @contextmanager
        def gradient_warning_scope(task):
            before = {'enabled': torch.are_deterministic_algorithms_enabled(),
                      'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}
            need(before == {'enabled': True, 'warn_only': False}, 'Strict policy required before gradient query')
            row = {'task': task, 'id': active_case, 'before': before,
                   'during': {'enabled': True, 'warn_only': True},
                   'scope': 'One reserved autograd.grad query plus CUDA synchronize only',
                   'bitwise_CUDA_gradient_repeatability_claimed': False}
            gradient_warning_scopes.append(row)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    # CUDA bilinear interpolation backward is documented by
                    # this installed Torch runtime as nondeterministic. Keep
                    # forward strict; disclose only this bounded query scope.
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    yield
                finally:
                    torch.use_deterministic_algorithms(before['enabled'], warn_only=before['warn_only'])
                    row['after'] = {'enabled': torch.are_deterministic_algorithms_enabled(),
                                    'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}
                    row['warnings'] = [{'category': item.category.__name__, 'message': str(item.message)[:12000],
                                        'filename': str(item.filename), 'lineno': int(item.lineno)} for item in caught]
                    row['unexpected_nondeterminism_warnings'] = [item for item in row['warnings']
                        if ('deterministic' in item['message'].lower() or 'nondeterminism' in item['message'].lower())
                        and re.match(r'^upsample_bilinear2d_backward_out_cuda\b', item['message']) is None]
                    row['restored_exact'] = row['after'] == before
                    persist_event({'kind': 'gradient_determinism_scope', **row, 'counters': dict(ledger.counters)})
                    need(row['restored_exact'], 'Strict deterministic flags were not restored')
                    need(not row['unexpected_nondeterminism_warnings'], 'Unexpected nondeterministic gradient operation; retain stop')

        def rng_snapshot():
            numpy_state = np.random.get_state()
            cpu = torch.get_rng_state().cpu().numpy().tobytes()
            cuda = [value.cpu().numpy().tobytes() for value in torch.cuda.get_rng_state_all()]
            return {'CPU_sha256': byte_sha(cpu), 'CPU_hex': cpu.hex(),
                    'CUDA_sha256': [byte_sha(value) for value in cuda],
                    'CUDA_hex': [value.hex() for value in cuda],
                    'Python_state': random.getstate(),
                    'NumPy_state': {'algorithm': numpy_state[0], 'keys': numpy_state[1].tolist(),
                                    'position': int(numpy_state[2]), 'has_gauss': int(numpy_state[3]),
                                    'cached_gaussian': float(numpy_state[4])}}

        def assert_clean(net):
            need(all(not module._forward_pre_hooks and not module._forward_hooks for module in net.modules()),
                 'Temporary model hooks leaked')
            need(all(value.grad is None for value in net.parameters()), 'No accumulated .grad field is permitted')

        with bounded_phase('load_exact_R2_U96_initializers'):
            r2_path = checked(p['model']['r2'])
            u96_path = checked(p['model']['u96'])
            need(packet.sha256(r2_path) == prototype.R2_FILE_SHA256 and
                 packet.sha256(u96_path) == prototype.U96_FILE_SHA256, 'Fixed U96 artifact files differ')
            r2_saved = torch.load(r2_path, map_location='cpu', weights_only=True, mmap=True)
            u96_saved = torch.load(u96_path, map_location='cpu', weights_only=True, mmap=True)
            need(type(r2_saved) is dict and set(r2_saved) == {'params_ema'}, 'Exact inference-only R2 required')
            need(type(u96_saved) is dict and set(u96_saved) == {'format', 'adapter_state'} and
                 u96_saved['format'] == 'codeformer-spatial-adapter-inference-v1', 'Exact U96 adapter format required')
            base_state = r2_saved['params_ema']
            adapter_state = u96_saved['adapter_state']
            need(type(base_state) is dict and len(base_state) == 515 and
                 canonical_tensors(base_state) == prototype.R2_CANONICAL_SHA256,
                 'Complete R2 canonical515 initializer differs')
            need(type(adapter_state) is dict and len(adapter_state) == 24 and
                 canonical_tensors(adapter_state) == prototype.U96_CANONICAL_SHA256,
                 'Complete U96 canonical24 initializer differs')
            from third_party.codeformer.codeformer_arch import CodeFormer
            for module_name, (relative, expected_source_sha) in prototype.ARCHITECTURE_SOURCES.items():
                loaded = sys.modules.get(module_name)
                need(loaded is not None and Path(loaded.__file__).resolve() == (root / relative).resolve() and
                     packet.sha256(root / relative) == expected_source_sha, 'Exact loaded architecture required')

            def base_factory():
                nonlocal model_construction_attempted
                model_construction_attempted = True
                base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                  connect_list=list(prototype.SITES))
                base.load_state_dict(base_state, strict=True)
                return base.float().eval().requires_grad_(False).to(device)

            class ReferenceAdapter(torch.nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    options = {'device': 'cpu', 'dtype': torch.float32}
                    self.down = torch.nn.Conv2d(channels, 32, 1, bias=True, **options)
                    self.depthwise = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=True, **options)
                    self.up = torch.nn.Conv2d(32, channels, 1, bias=True, **options)
                    self.activation = torch.nn.SiLU(inplace=False)

                def forward(self, observed):
                    return observed + self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

            class IndependentReference(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base
                    self.adapters = torch.nn.ModuleDict({site: ReferenceAdapter(prototype.CHANNELS[site])
                                                       for site in prototype.SITES})
                    self.adapters.load_state_dict({name.removeprefix('adapters.'): value
                                                   for name, value in adapter_state.items()}, strict=True)
                    self.last_forward_sites = ()
                    self.last_base_calls = 0

                def forward(self, sample):
                    need(not torch.is_grad_enabled(), 'Independent frozen reference requires no_grad')
                    assert_clean(self)
                    handles = []
                    sites = []
                    self.last_base_calls = 0
                    def hook(site):
                        def adapt(module, args):
                            need(len(args) == 3 and args[2] == 1.0 and len(sites) < 4 and
                                 site == prototype.SITES[len(sites)], 'Exact ordered independent U96 hooks required')
                            observed, decoder, strength = args
                            shape = (1, prototype.CHANNELS[site], int(site), int(site))
                            need(observed.shape == decoder.shape == shape and observed.dtype == torch.float32 and
                                 not observed.requires_grad and not decoder.requires_grad,
                                 'Independent U96 frozen feature geometry differs')
                            used = self.adapters[site](observed)
                            need(used.shape == shape and torch.isfinite(used).all().item(), 'Independent adapter output invalid')
                            sites.append(site)
                            return used, decoder, strength
                        return adapt
                    try:
                        for site in prototype.SITES:
                            handles.append(self.base.fuse_convs_dict[site].register_forward_pre_hook(hook(site)))
                        self.last_base_calls = 1
                        return self.base(sample, w=1.0, detach_16=True, code_only=False, adain=True)
                    finally:
                        for handle in reversed(handles):
                            handle.remove()
                        self.last_forward_sites = tuple(sites)

            before_construction = rng_snapshot()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261023)
                reference = IndependentReference(base_factory()).float().eval().requires_grad_(False).to(device)
                candidate_base = base_factory()
            need(rng_snapshot() == before_construction, 'Base/reference construction changed caller RNG')
            candidate = prototype.build_restorer(candidate_base, adapter_state,
                r2_artifact=r2_path, u96_artifact=u96_path, manual_vm_root=root,
                prototype_source_sha256=p['files'][PACKAGE_RELATIVE + '/model.py']['sha256'],
                device='cuda:0', acknowledgement=prototype.MANUAL_ACKNOWLEDGEMENT).eval()
            need(rng_snapshot() == before_construction, 'Guarded candidate construction changed caller RNG')
            invariant = candidate.validate_invariants(check_frozen_digests=True)
            named = dict(candidate.named_parameters())
            selected_items = [(name, named[name]) for name, _ in direction.PARAMETER_SCHEMA]
            need(tuple((name, tuple(value.shape)) for name, value in selected_items) == direction.PARAMETER_SCHEMA and
                 set(name for name, value in named.items() if value.requires_grad) ==
                 {name for name, _ in direction.PARAMETER_SCHEMA}, 'Exact sorted ten-parameter packing differs')
            initial_arrays, initial_witness = capture_state('initializer')
            independent_arrays = tensors_as_arrays(reference.state_dict())
            need(canonical_arrays(independent_arrays, np=np)[0] == initial_witness['canonical_sha256'],
                 'Independent complete U96 and candidate initializer differ')
            reference_initial_sha = canonical_arrays(independent_arrays, np=np)[0]
            del independent_arrays
            initial_vector = np.concatenate([initial_arrays[name].reshape(-1) for name, _ in direction.PARAMETER_SCHEMA])
            need(initial_vector.shape == (75843,) and initial_vector.dtype == np.float32, 'Exact FP32 initializer vector required')
            save_array('states/initializer/trainable_vector.npy', initial_vector)
            run_rng = rng_snapshot()
            store.json('rng_initializer.json', run_rng)
            store.json('preflight.json', {'complete': True, 'protocol_sha256': expected_sha, 'host': facts,
                       'torch': torch.__version__, 'CUDA': torch.version.cuda, 'GPU': torch.cuda.get_device_name(0),
                       'candidate_invariants': invariant, 'complete_initial_state': initial_witness,
                       'independent_reference_canonical_sha256': reference_initial_sha,
                       'CPU_CUDA_factory_RNG_exact': True, 'limits': contract.LIMITS,
                       'optimizer_constructions': 0, 'optimizer_steps': 0, 'model_qualified': False})

        def png_array(binding):
            with Image.open(checked(binding)) as image:
                need(image.mode == 'RGB' and image.size == (256, 256) and getattr(image, 'n_frames', 1) == 1,
                     'Single RGB256 image required')
                return np.asarray(image).copy()

        def input_values(row):
            need(type(row['id']) is str and re.fullmatch('[A-Za-z0-9_-]{1,120}', row['id']), 'Safe witness ID required')
            arrays = []
            for field in ('input', 'target'):
                if row.get(field) is None:
                    need(field == 'target' and row['role'] == 'dark', 'Only inherited dark guard target may be null')
                    arrays.append(None)
                else:
                    image = png_array(row[field])
                    need(byte_sha(image.tobytes()) == row[field + '_RGB8_sha256'], 'Witness RGB8 bytes differ')
                    arrays.append(image.astype(np.float32) / np.float32(255))
            observed = torch.from_numpy(arrays[0].transpose(2, 0, 1).copy()).to(device)[None]
            sample = F.interpolate(observed, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            target = (None if arrays[1] is None else
                      torch.from_numpy(arrays[1].transpose(2, 0, 1).copy()).to(device)[None])
            return sample, target, arrays[1]

        def full_call(net, sample, category):
            need(torch.are_deterministic_algorithms_enabled() and
                 not torch.is_deterministic_algorithms_warn_only_enabled() and
                 not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and
                 not torch.backends.cudnn.benchmark, 'Strict deterministic FP32 policy changed')
            resource('before_forward_reservation')
            token = ledger.reserve('forward', category=category, state=active_state, id=active_case)
            status('forward_reserved')
            with torch.autocast(device_type='cuda', enabled=False):
                result = net(sample)
            torch.cuda.synchronize(0)
            ledger.returned(token, python_return_type=type(result).__name__,
                            synchronized_actual_return=True, validity_checked_separately=True)
            status('forward_returned')
            valid = isinstance(result, tuple) and len(result) == 3
            if valid:
                valid = all(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and
                            value.dtype == torch.float32 and value.device == device and torch.isfinite(value).all().item()
                            for value, shape in zip(result, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))))
            if not valid:
                retain_unvalidated_values('forward_' + str(token), result)
                raise ValueError('Actual returned stock tuple is invalid/nonfinite; retained without another call')
            need(net.last_base_calls == 1 and tuple(net.last_forward_sites) == prototype.SITES,
                 'One base call and four ordered U96 sites required')
            if net is candidate:
                need(net.last_spatial_calls == net.last_normalization_calls == net.last_readout_calls == 1,
                     'One actual block22/norm23/readout24 path required')
            assert_clean(net)
            persist_event({'kind': 'forward_validated', 'token': token,
                           'stock_shapes': [list(value.shape) for value in result],
                           'module_sites': list(net.last_forward_sites),
                           'module_calls_are_not_extra_full_forwards': True, 'counters': dict(ledger.counters)})
            resource('after_forward_return')
            return result

        def load_raw(binding):
            with checked(binding).open('rb') as stream:
                value = np.load(stream, allow_pickle=False)
            need(value.shape == (256, 256, 3) and value.dtype == np.float32 and np.isfinite(value).all(),
                 'Finite HWC256 FP32 RAW required')
            return value

        def tensor_summary(value):
            array = value.detach().cpu().contiguous().numpy()
            return {'shape': list(array.shape), 'dtype': str(array.dtype), 'bytes': int(array.nbytes),
                    'bytes_sha256': byte_sha(array.tobytes()), 'minimum': float(array.min()),
                    'maximum': float(array.max())}

        def saved_record(row, triple, *, baseline_record=None, parity=False):
            stem = 'banks/' + active_state + '/' + row['id']
            native = triple[0]
            if native.requires_grad:
                raw = prototype.connected_rgb256(native, torch=torch, manual_vm_root=root)
            else:
                raw = F.interpolate(((native.float() + 1) / 2).clamp(0, 1), (256, 256),
                                    mode='bilinear', align_corners=False)
            unclipped = F.interpolate((native.float() + 1) / 2, (256, 256), mode='bilinear', align_corners=False)
            array = raw.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
            quantized = np.floor(array * np.float32(255)).astype(np.uint8)
            pins = {'raw': save_array(stem + '/raw.npy', array),
                    'unclipped': save_array(stem + '/unclipped.npy',
                                            unclipped.detach().cpu().numpy()[0].transpose(1, 2, 0).copy())}
            stream = io.BytesIO()
            Image.fromarray(quantized, mode='RGB').save(stream, format='PNG')
            pins['png'] = store.write(stem + '/output.png', stream.getvalue())
            need(np.array_equal(png_array(pins['png']), quantized), 'Durable floor-PNG pixels differ')
            if parity:
                pins['stock'] = {name: save_array(stem + '/' + name + '.npy', value)
                                 for name, value in zip(('native512', 'logits', 'LQ'), triple)}
            target = (None if row.get('target') is None else
                      png_array(row['target']).astype(np.float32) / np.float32(255))
            teacher_raw = teacher_png = None
            if row.get('behavior_teacher') is not None:
                teacher_raw = load_raw(row['behavior_teacher']['raw'])
                teacher_png = png_array(row['behavior_teacher']['png'])
            baseline_raw = baseline_png = None
            if baseline_record is not None:
                baseline_raw = load_raw(baseline_record['outputs']['raw'])
                baseline_png = png_array(baseline_record['outputs']['png'])
            numerical = metrics.record_metrics(array, quantized, target, row, np=np,
                         teacher_raw=teacher_raw, teacher_png=teacher_png,
                         baseline_raw=baseline_raw, baseline_png=baseline_png)
            stock = {name: tensor_summary(value) for name, value in zip(('native512', 'logits', 'LQ'), triple)}
            if baseline_record is not None:
                need(stock['logits'] == baseline_record['stock']['logits'] and stock['LQ'] == baseline_record['stock']['LQ'],
                     'Frozen upstream logits/LQ changed')
            record = {**row, 'complete': True, 'state': active_state, 'outputs': pins, 'stock': stock,
                      'clipping': {'native_below_minus1': int((native.detach() < -1).sum().item()),
                                   'native_above1': int((native.detach() > 1).sum().item()),
                                   'native_values': int(native.numel())},
                      'counters': dict(ledger.counters), 'model_qualified': False, **numerical}
            record['receipt'] = store.json(stem + '/record.json', record)
            return record, raw

        by_id = {row['id']: row for row in selection['optimization'] + selection['monitors'] + selection['guards']}
        need(len(by_id) == 72, 'Exact72 selected bank required')
        initial_bank = {}
        query_records = []
        gradient_sums = {name: np.zeros(75843, dtype=np.float64) for name in objective.TASK_NAMES}
        gradient_counts = dict.fromkeys(objective.TASK_NAMES, 0)
        active_state = 'initializer'
        initial_groups = (
            ('initializer_gradient16', [row for row in plan if row['state'] == 'initializer' and row['gradient_components']]),
            ('initializer_monitor_guard56', [row for row in plan if row['state'] == 'initializer' and not row['gradient_components']]),
        )
        for group_name, group_calls in initial_groups:
            with bounded_phase(group_name):
                for call in group_calls:
                    active_case = call['id']
                    row = by_id[active_case]
                    sample, target, target_array = input_values(row)
                    category = ('common_initializer_component_forwards' if call['gradient_components'] else
                                'initial_monitor_guard_forwards')
                    with torch.enable_grad():
                        triple = full_call(candidate, sample, category)
                        record, delivered = saved_record(row, triple, parity=active_case in contract.PARITY_IDS)
                        initial_bank[active_case] = record
                        if call['gradient_components']:
                            need(target is not None and row['cohort'] == 'optimize' and row['optimization_permitted'] is True,
                                 'Only the16 selected photographic optimization cases may produce gradients')
                            components = objective.connected_components(delivered, target, row['role'], torch=torch)
                            saved_components, saved_terms = objective.numpy_components(
                                load_raw(record['outputs']['raw']), target_array, row['role'], np=np)
                            for component in objective.COMPONENTS:
                                resource('before_gradient_reservation')
                                task = row['role'] + '/' + component
                                token = ledger.reserve('gradient', state=active_state, id=active_case, task=task,
                                                       retain_graph=component == 'rgb')
                                status('gradient_reserved')
                                with gradient_warning_scope(task):
                                    gradients = torch.autograd.grad(components[component], [value for _, value in selected_items],
                                                 retain_graph=component == 'rgb', create_graph=False, allow_unused=False)
                                    torch.cuda.synchronize(0)
                                    ledger.returned(token, task=task, synchronized_actual_return=True, validity_checked_separately=True)
                                    status('gradient_returned')
                                valid = isinstance(gradients, tuple) and len(gradients) == 10 and all(
                                    isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
                                    tuple(value.shape) == tuple(parameter.shape) and torch.isfinite(value).all().item()
                                    for value, (_, parameter) in zip(gradients, selected_items))
                                if not valid:
                                    retain_unvalidated_values('gradient_' + str(token), gradients)
                                    raise ValueError('Actual gradient tuple is invalid/nonfinite; retain stop before application')
                                vector = np.concatenate([value.detach().cpu().contiguous().numpy().reshape(-1)
                                                         for value in gradients]).astype(np.float32, copy=False)
                                need(vector.shape == (75843,), 'Exact packed75843 component gradient required')
                                persist_event({'kind': 'gradient_validated', 'token': token, 'task': task,
                                               'vector_sha256': byte_sha(vector.tobytes()), 'counters': dict(ledger.counters)})
                                binding = save_array('gradients/cases/' + active_case + '/' + component + '.npy', vector)
                                gradient_sums[task] += vector.astype(np.float64)
                                gradient_counts[task] += 1
                                query_records.append({'id': active_case, 'task': task, 'gradient': binding,
                                      'connected_FP32_component': float(components[component].detach().item()),
                                      'saved_float64_component': saved_components[component],
                                      'saved_photographic_terms': saved_terms, 'parameter_order': list(direction.PARAMETER_SCHEMA),
                                      'query_token': token, 'counters': dict(ledger.counters)})
                                del gradients, vector
                            assert_clean(candidate)
                            del components
                    del sample, target, target_array, triple, delivered
        active_case = None
        initial_gate = metrics.bank_gate(initial_bank, None, p, selection, np=np, state='initializer')
        store.json('banks/initializer/bank.json', {'complete': True, 'records': initial_bank, 'gate': initial_gate})
        store.json('gradient_queries.json', {'complete': True, 'queries': query_records, 'counts': gradient_counts,
                                           'mean_count_per_task': 4, 'optimizer_steps': 0,
                                           'bounded_gradient_warning_scopes': gradient_warning_scopes,
                                           'bitwise_CUDA_gradient_repeatability_claimed': False})
        need(len(query_records) == 32 and all(value == 4 for value in gradient_counts.values()), 'Exact32 queries/eight4-case means required')
        if not initial_gate['pass']:
            raise QualityStop('Initializer inherited guard failure; retain stop before parameter applications')

        parity_rows = []
        active_state = 'independent_frozen_U96_parity'
        with bounded_phase('independent_frozen_U96_parity2'):
            for case_id in contract.PARITY_IDS:
                active_case = case_id
                sample, target, target_array = input_values(by_id[case_id])
                with torch.no_grad():
                    triple = full_call(reference, sample, 'independent_parity_forwards')
                    record, delivered = saved_record(by_id[case_id], triple, parity=True)
                baseline_record = initial_bank[case_id]
                need(record['stock'] == baseline_record['stock'] and
                     np.array_equal(load_raw(record['outputs']['raw']), load_raw(baseline_record['outputs']['raw'])) and
                     np.array_equal(png_array(record['outputs']['png']), png_array(baseline_record['outputs']['png'])),
                     'Exact native/logits/LQ/RAW/floorPNG independent initializer parity required')
                parity_rows.append({'id': case_id, 'complete': True, 'independent': record['receipt'],
                                    'candidate': baseline_record['receipt'], 'all5_domains_exact': True})
                del sample, target, target_array, triple, delivered
        need(canonical_tensors(reference.state_dict()) == reference_initial_sha,
             'Independent complete reference changed')
        store.json('independent_parity.json', {'complete': True, 'rows': parity_rows,
                   'reference_complete_state_exact': True, 'extra_full_forwards': 2})
        active_case = None
        active_state = 'direction_calculation'
        with bounded_phase('eight_task_analytical_subspaces_and_fixed_FP32_proposals'):
            means = {name: value / 4.0 for name, value in gradient_sums.items()}
            mean_pins = {name: save_array('gradients/means/' + name.replace('/', '__') + '.npy', value)
                         for name, value in means.items()}
            comparison = direction.compare_subspaces(means, np=np)
            for subspace in ('expanded', 'readout'):
                result = comparison[subspace]
                if 'combined_vector' in result:
                    result['combined_vector_file'] = save_array('direction/' + subspace + '_combined.npy', result['combined_vector'])
            def json_direction(result):
                return {name: value for name, value in result.items() if name != 'combined_vector'}
            store.json('direction/comparison.json', {**comparison,
                       'expanded': json_direction(comparison['expanded']),
                       'readout': json_direction(comparison['readout']), 'mean_vectors': mean_pins})
            if not comparison['expanded']['certified']:
                raise QualityStop('Expanded path has no certified eight-task common descent; feasibility unknown')
            proposals = direction.build_fp32_proposals(initial_vector, means, comparison['expanded'], np=np)
            proposal_pins = []
            for index, proposal in enumerate(proposals['proposals']):
                name = 'proposal_' + ('A' if index == 0 else 'B')
                bindings = {field: save_array('direction/' + name + '_' + field + '.npy', proposal[field])
                            for field in ('parameters_fp32', 'intended_displacement_float64', 'realized_displacement_float64')}
                proposal_pins.append({**{key: value for key, value in proposal.items() if key not in bindings},
                                      'arrays': bindings})
            store.json('direction/fixed_FP32_proposals.json', {**{name: value for name, value in proposals.items()
                       if name != 'proposals'}, 'proposals': proposal_pins})
            if not proposals['certified'] or proposals['stop_before_any_proposal']:
                raise QualityStop('Fixed FP32 displacement lost common descent; apply neither')

        trial_receipts = []
        prior_proposal_passed = True
        for trial_index, proposal in enumerate(proposals['proposals']):
            active_state = 'proposal_' + ('A' if trial_index == 0 else 'B')
            with bounded_phase(active_state + '_independent_array_application'):
                need(contract.permit_next_proposal(attempted=ledger.counters['proposal_application_attempts'],
                     initial_checks_passed=initial_gate['pass'], direction_certified=comparison['expanded']['certified'],
                     actual_FP32_direction_passed=proposals['certified'], prior_proposal_passed=prior_proposal_passed),
                     'Prospective application stop rule refused')
                need(np.array_equal(initial_vector, np.concatenate([initial_arrays[name].reshape(-1)
                     for name, _ in direction.PARAMETER_SCHEMA])), 'Immutable shared initializer vector changed')
                resource('before_application_reservation')
                token = ledger.reserve('application', state=active_state, maxabs_scale=proposal['maxabs_scale'],
                                       independent_initializer_sha256=byte_sha(initial_vector.tobytes()))
                status('application_reserved')
                offset = 0
                with torch.no_grad():
                    for name, parameter in selected_items:
                        count = parameter.numel()
                        array = proposal['parameters_fp32'][offset:offset + count].reshape(tuple(parameter.shape)).copy()
                        parameter.copy_(torch.from_numpy(array).to(device))
                        offset += count
                torch.cuda.synchronize(0)
                ledger.returned(token, synchronized_actual_return=True, validity_checked_separately=True)
                status('application_returned')
                actual_vector = np.concatenate([value.detach().cpu().numpy().reshape(-1) for _, value in selected_items])
                need(offset == 75843 and np.array_equal(actual_vector, proposal['parameters_fp32']),
                     'Actual parameter application differs from prechecked exact FP32 proposal')
                persist_event({'kind': 'application_validated', 'token': token,
                               'packed_vector_sha256': byte_sha(actual_vector.tobytes()),
                               'counters': dict(ledger.counters)})
                _, trial_state = capture_state(active_state)
            bank = {}
            with bounded_phase(active_state + '_72_case_bank'):
                for call in (row for row in plan if row['state'] == active_state):
                    active_case = call['id']
                    row = by_id[active_case]
                    sample, target, target_array = input_values(row)
                    with torch.enable_grad():
                        triple = full_call(candidate, sample, 'proposal_forwards')
                        record, delivered = saved_record(row, triple, baseline_record=initial_bank[active_case])
                    bank[active_case] = record
                    del sample, target, target_array, triple, delivered
            active_case = None
            gate = metrics.bank_gate(bank, initial_bank, p, selection, np=np, state=active_state)
            receipt = store.json('banks/' + active_state + '/bank.json', {'complete': True, 'records': bank,
                                  'gate': gate, 'state_witness': trial_state, 'counters': dict(ledger.counters)})
            trial_receipts.append(receipt)
            prior_proposal_passed = bool(gate['pass'])
            if not gate['pass']:
                raise QualityStop(active_state + ' actual RAW/PNG structure or preservation failed; stop before next proposal')
            print(json.dumps({'state': active_state, 'feasibility_pass': True,
                              'forwards': ledger.counters['full_forward_returns'], 'model_qualified': False}), flush=True)

        active_state = 'final_integrity'
        with bounded_phase('final_integrity_and_feasibility_receipt'):
            expected_counters = {'full_forward_attempts': 218, 'full_forward_returns': 218,
                                 'gradient_query_attempts': 32, 'gradient_queries': 32,
                                 'proposal_application_attempts': 2, 'proposal_applications': 2,
                                 'optimizer_constructions': 0, 'optimizer_steps': 0,
                                 **CATEGORY_CAPS}
            need(ledger.counters == expected_counters and not ledger.pending, 'Exact finite operation census differs')
            need(complete_state_witnesses == 3, 'All three complete539 state witnesses required')
            need(rng_snapshot() == run_rng, 'Fixed sampler/model operations changed CPU/CUDA/Python/NumPy RNG')
            candidate.validate_invariants(check_frozen_digests=True)
            need(canonical_tensors(reference.state_dict()) == reference_initial_sha, 'Frozen independent U96 changed')
            for relative, binding in p['files'].items():
                observed = packet.pin(root, root / relative)
                need(observed['bytes'] == binding['bytes'] and observed['sha256'] == binding['sha256'],
                     'Static source/model/data changed: ' + relative)
            store.json('rng_terminal.json', rng_snapshot())
            namespace = store.audit_namespace()
            result = {'complete': True, 'status': 'BOTH_FIXED_PROPOSALS_TRAIN_FEASIBLE_NOT_QUALIFIED',
                      'protocol_sha256': expected_sha, 'counters': dict(ledger.counters),
                      'initial_state': initial_witness, 'trial_receipts': trial_receipts,
                      'last_complete_state': last_complete_state, 'unknown_reservations': [],
                      'operation_outcome_unknown': False, 'elapsed_seconds': time.monotonic() - start,
                      'resources': resource('before_result'), 'namespace_before_result': namespace,
                      'all529_frozen_values_exported_exact': True, 'RNG_exact': True,
                      'optimizer_steps': 0, 'model_qualified': False,
                      'full_finite_training_pilot_DEV_human_FINAL_app_qualification_pending': True,
                      'automatic_training_resume_or_promotion': False}
            result_pin = store.json('result.json', result)
            status('finite_feasibility_complete', complete=True)
            print(json.dumps({'complete': True, 'result': result_pin, 'model_qualified': False}), flush=True)
            return result
    except BaseException as error:
        if store is not None:
            snapshot = ledger.snapshot() if ledger else {'counters': dict.fromkeys(COUNTER_KEYS, 0),
                        'unknown_reservations': [], 'operation_outcome_unknown': False}
            failure = {'complete': False, 'status': 'QUALITY_STOP' if isinstance(error, QualityStop) else
                       'INFRASTRUCTURE_OR_INTEGRITY_STOP', 'UTC': utc(), 'phase': failed_phase or phase,
                       'error_type': type(error).__name__, 'error': str(error)[:4000],
                       'traceback': traceback.format_exc()[-20000:], 'protocol_sha256': expected_sha,
                       'elapsed_seconds': time.monotonic() - start, 'active_state': active_state,
                       'active_case': active_case, 'last_complete_state': last_complete_state,
                       'Torch_imported_by_worker': torch is not None, 'model_construction_attempted': model_construction_attempted,
                       'partial_payloads': sorted(store.partial_files), 'model_qualified': False,
                       'bounded_gradient_warning_scopes': gradient_warning_scopes,
                       'failure_finalization_limits': {'normal_work_seconds': 1200,
                           'additional_state_capture_seconds_maximum': FAILURE_CAPTURE_SECONDS,
                           'additional_receipt_seconds_maximum': FAILURE_RECEIPT_SECONDS,
                           'additional_RNG_restore_seconds_maximum': FAILURE_FINALIZATION_SECONDS,
                           'external_worker_seconds': 1260},
                       'automatic_resume_retry_or_promotion': False, **snapshot}
            capture_budget = failure_capture_budget(time.monotonic() - start)
            failure['failure_capture_budget_seconds'] = capture_budget
            if candidate is not None and torch is not None and np is not None and capture_budget > 0:
                try:
                    with failure_time_scope(capture_budget, 'state_capture'):
                        try:
                            assert_clean(candidate)
                            captured = candidate.capture_complete_state_for_audit()
                            arrays = tensors_as_arrays(captured)
                            current_sha = canonical_arrays(arrays, np=np)[0]
                            if last_complete_state is not None and current_sha == last_complete_state['canonical_sha256']:
                                failure['retained_partial_state'] = {'exact_existing_full_state': last_complete_state,
                                    'current_complete_canonical_sha256': current_sha,
                                    'all539_current_values_exact_to_existing_witness': True}
                            else:
                                payload, witness = npz_state_payload(arrays, np=np)
                                witness['file'] = store.write('states/failure/full_partial_state.npz', payload, emergency=True)
                                witness['complete_tensor_count'] = len(arrays)
                                failure['retained_partial_state'] = witness
                            failure['RNG_at_failure'] = store.json('rng_failure.json', rng_snapshot(), emergency=True)
                        except TimeoutError:
                            raise
                        except BaseException as state_error:
                            failure['guarded_partial_state_error'] = repr(state_error)
                            # Invalid states are evidence, never accepted captures.
                            # This fallback shares the SAME alarm and cap; no retry.
                            arrays = tensors_as_arrays(candidate.state_dict())
                            stream = io.BytesIO()
                            np.savez(stream, **arrays)
                            failure['unvalidated_partial_state'] = store.write(
                                'states/failure/unvalidated_partial_state.npz', stream.getvalue(), emergency=True)
                            failure['unvalidated_partial_state_may_be_nonfinite'] = True
                except BaseException as state_error:
                    failure['partial_state_retention_error'] = repr(state_error)
                    failure['partial_state_complete_or_valid_unknown'] = True
            elif candidate is not None:
                failure['partial_state_capture_skipped_no_time'] = True
                failure['partial_state_complete_or_valid_unknown'] = True
            failure['partial_payloads'] = sorted(store.partial_files)
            try:
                receipt_budget = max(0.0, min(float(FAILURE_RECEIPT_SECONDS),
                    1260.0 - (time.monotonic() - start) - FAILURE_FINALIZATION_SECONDS))
                with failure_time_scope(receipt_budget, 'failure_receipt'):
                    try:
                        status('stopped', quality_stop=isinstance(error, QualityStop))
                    except TimeoutError:
                        raise
                    except BaseException as status_error:
                        failure['status_write_error'] = repr(status_error)
                    store.json('failure.json', failure, emergency=True)
            except BaseException as write_error:
                print('Failure receipt write failed: ' + repr(write_error), flush=True)
        if isinstance(error, QualityStop):
            raise SystemExit(2) from error
        raise
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        if caller_rng is not None and torch is not None and np is not None:
            try:
                final_budget = max(0.0, min(float(FAILURE_FINALIZATION_SECONDS),
                                           1260.0 - (time.monotonic() - start)))
                with failure_time_scope(final_budget, 'caller_RNG_restore'):
                    random.setstate(caller_rng['Python'])
                    np.random.set_state(caller_rng['NumPy'])
                    torch.set_rng_state(caller_rng['CPU'])
                    torch.cuda.set_rng_state_all(caller_rng['CUDA'])
            except BaseException as restore_error:
                print('Bounded caller RNG restoration failed: ' + repr(restore_error), flush=True)
