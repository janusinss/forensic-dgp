"""Inert-on-import, explicit manual-L4 four-update worker.

Only run() constructs a model, computes gradients or applies a parameter. Its
fresh staged root, tmux, exact parent cwd, venv and external supervisor are
mandatory. The historical parent is read-only; every new artifact is owned by
this new stage. No historical run, fallback, line search or resume is called.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings

from . import contract

PACKAGE, OUTPUT = contract.PACKAGE, contract.OUTPUT
PARENT_PACKAGE = 'research/codeformer_u64_cleananchors_continuation_v1'
CATEGORIES = contract.CATEGORIES
OPERATIONS = {
    'forward': ('full_forward_attempts', 'full_forward_returns', 540),
    'gradient_query': ('gradient_query_attempts', 'gradient_query_returns', 512),
    'parameter_application': ('parameter_application_attempts', 'parameter_applications', 4),
}
COUNTER_KEYS = ('full_forward_attempts', 'full_forward_returns',
                'gradient_query_attempts', 'gradient_query_returns',
                'parameter_application_attempts', 'parameter_applications', *CATEGORIES)


def need(condition, message):
    if not condition:
        raise ValueError(message)


class QualityStop(ValueError):
    """Terminal scientific stop. Preserve this state; do not append updates."""


class OperationLedger:
    def __init__(self, persist):
        self.persist = persist
        self.counters = dict.fromkeys(COUNTER_KEYS, 0)
        self.pending = {}
        self.sequence = 0

    def reserve(self, kind, **metadata):
        need(kind in OPERATIONS and not self.pending
             and not set(metadata) & {'kind', 'token', 'counters'},
             'One known nonoverlapping operation with unreserved metadata required')
        attempt, _, maximum = OPERATIONS[kind]
        need(self.counters[attempt] < maximum, 'Finite operation budget exhausted')
        if kind == 'forward':
            category = metadata.get('category')
            need(category in CATEGORIES and self.counters[category] < CATEGORIES[category],
                 'Known finite forward category required')
            self.counters[category] += 1
        self.counters[attempt] += 1
        self.sequence += 1
        self.pending[self.sequence] = {'kind': kind, **metadata}
        self.persist({'kind': kind + '_reserved', 'token': self.sequence,
                      'counters': dict(self.counters), **metadata})
        return self.sequence

    def returned(self, token, **metadata):
        need(type(token) is int and token in self.pending
             and not set(metadata) & {'kind', 'token', 'counters'}, 'Known pending return required')
        pending = self.pending[token]
        count = dict(self.counters)
        count[OPERATIONS[pending['kind']][1]] += 1
        self.persist({**pending, **metadata, 'kind': pending['kind'] + '_returned',
                      'token': token, 'counters': count})
        self.counters = count
        del self.pending[token]

    def snapshot(self):
        return {'counters': dict(self.counters),
                'unknown_reservations': [{'token': key, **row} for key, row in self.pending.items()],
                'operation_outcome_unknown': bool(self.pending)}


def supervisor_context(sha, *, environ, parent_pid):
    need(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha)
         and environ.get(contract.ENV + 'SUPERVISED') == '1'
         and environ.get(contract.ENV + 'PROTOCOL') == sha
         and environ.get(contract.ENV + 'SUPERVISOR_PID') == str(parent_pid),
         'Exact finite supervisor seal and actual parent PID required')


def run(root, parent, protocol, expected_sha):
    from . import runtime
    root, parent = Path(root).absolute(), Path(parent).absolute()
    began = time.monotonic()
    store = ledger = candidate = torch = np = helper = linux_resource = None
    caller_rng = frozen_initial = original_heads = None
    last_complete_state = last_checkpoint = None
    phase, failed_phase = 'static_guards', None
    active_update, active_slot, active_case, active_query = 0, None, None, None
    full_states_attempted, old_term = 0, None

    def event(row):
        store.append('events.jsonl', {'UTC': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - began, 'phase': phase, **row})

    def status(label, complete=False):
        if store is not None:
            snapshot = ledger.snapshot() if ledger else {'counters': dict.fromkeys(COUNTER_KEYS, 0),
                'unknown_reservations': [], 'operation_outcome_unknown': False}
            store.status({'complete': complete, 'protocol_sha256': expected_sha,
                'status': 'FINITE_GATE_ALIGNED_PILOT_NOT_QUALIFIED', 'label': label,
                'phase': failed_phase or phase, 'active_update': active_update,
                'active_slot': active_slot, 'active_case': active_case, 'active_query': active_query,
                'last_complete_state': last_complete_state, 'last_checkpoint': last_checkpoint,
                'elapsed_seconds': time.monotonic() - began, 'model_qualified': False, **snapshot})

    def checked(base, row):
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'Exact file binding required')
        runtime.relative(row['path'])
        path = runtime.regular(base / row['path'])
        need(runtime.pin(base, path) == row, 'Bound bytes differ: ' + row['path'])
        return path

    def resource(label):
        elapsed = time.monotonic() - began
        free = shutil.disk_usage(root).free
        rss = helper.linux_current_rss_bytes(Path('/proc/self/statm').read_text(), int(os.sysconf('SC_PAGE_SIZE')))
        peak = int(linux_resource.getrusage(linux_resource.RUSAGE_SELF).ru_maxrss) * 1024
        cuda = peak_cuda = 0
        if torch is not None and torch.cuda.is_initialized():
            cuda, peak_cuda = int(torch.cuda.memory_reserved(0)), int(torch.cuda.max_memory_reserved(0))
        row = {'label': label, 'elapsed_seconds': elapsed, 'free_disk_bytes': free,
               'current_RSS_bytes': rss, 'peak_RSS_bytes': peak, 'CUDA_reserved_bytes': cuda,
               'CUDA_peak_reserved_bytes': peak_cuda, 'output_bytes': store.bytes if store else 0}
        if store is not None:
            store.append('resources.jsonl', row)
        limits = contract.LIMITS
        need(elapsed < limits['worker_seconds'], 'Finite1800s worker stop')
        need(free >= limits['continuing_free_disk_bytes'], 'Continuing4GiB disk stop')
        need(max(rss, peak) <= limits['maximum_RSS_bytes'], 'RSS10GiB stop')
        need(max(cuda, peak_cuda) <= limits['maximum_CUDA_reserved_bytes'], 'CUDA18GiB stop')
        need(store is None or store.bytes <= limits['maximum_output_bytes'], 'Output2GiB stop')

    @contextmanager
    def timed(seconds, name):
        need(seconds > 0 and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Fresh positive timer required')
        prior = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError('Finite stop: ' + name)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)

    @contextmanager
    def bounded_phase(name):
        nonlocal phase, failed_phase
        previous, phase = phase, name
        start = time.monotonic()
        try:
            resource('before')
            with timed(min(contract.LIMITS['per_phase_seconds'],
                           contract.LIMITS['worker_seconds'] - (time.monotonic() - began)), name):
                yield
                resource('after')
                event({'kind': 'phase_returned', 'phase_seconds': time.monotonic() - start,
                       'counters': dict(ledger.counters)})
        except BaseException:
            failed_phase = name
            raise
        finally:
            phase = previous

    try:
        p = runtime.verify_static(root, parent, expected_sha)
        need(p == protocol and Path(__file__).resolve() == root / PACKAGE / 'worker.py'
             and Path(contract.__file__).resolve() == root / PACKAGE / 'contract.py', 'Own sealed worker required')
        facts = runtime.require_vm(root, parent)
        supervisor_context(expected_sha, environ=os.environ, parent_pid=os.getppid())
        need(Path.cwd().resolve() == parent and not (root / OUTPUT).exists(), 'Read-only parent cwd and fresh output required')
        need(shutil.disk_usage(root).free >= contract.LIMITS['initial_free_disk_bytes'], 'Initial8GiB free required')
        parent_raw = (parent / 'protocol.json').read_bytes()
        oldp = runtime.strict_json(parent_raw)
        standard_raw = checked(parent, oldp['parent_protocol']).read_bytes()
        audit_bytes = checked(parent, oldp['historical_pixel_audit']).read_bytes()
        selection = contract.select_cases(parent_raw, standard_raw)
        plan = contract.operation_plan(selection)
        need(selection == p['selection'] and plan == p['plan'], 'Exact current finite cohort/plan required')
        # Verified parent packages are imported only after every static binding
        # was checked. Their factory intentionally retains its original cwd guard.
        import research
        parent_research = str(parent / 'research')
        research.__path__ = [parent_research, *[str(v) for v in research.__path__ if str(v) != parent_research]]
        sys.path.insert(0, str(parent))
        helper = importlib.import_module('research.codeformer_u96_spatial_feasibility_v1.worker')
        model = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.model')
        metrics = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.metrics')
        for module, relative in ((helper, 'research/codeformer_u96_spatial_feasibility_v1/worker.py'),
                                 (model, PARENT_PACKAGE + '/model.py'), (metrics, PARENT_PACKAGE + '/metrics.py')):
            need(Path(module.__file__).resolve() == parent / relative, 'Read-only parent import origin differs')
        import resource as linux_resource
        rows = {row['id']: row for row in oldp['cases'] + oldp['guards']}
        store = helper.DurableStore(root, root / OUTPUT, contract.LIMITS['maximum_output_bytes'], runtime.pin)
        store.json('started.json', {'complete': False, 'protocol_sha256': expected_sha, 'host': facts,
                                  'parent_read_only': True, 'model_qualified': False})
        store.write('events.jsonl', b'')
        store.write('resources.jsonl', b'')
        ledger = OperationLedger(event)
        old_term = signal.getsignal(signal.SIGTERM)
        def stopped(signum, frame):
            raise TimeoutError('External termination: preserve actual state and unknown reservation')
        signal.signal(signal.SIGTERM, stopped)
        store.json('operation_plan.json', {'complete': True, 'calls': plan, 'selection': selection,
                                         'maximum_updates': 4, 'model_qualified': False})
        event({'kind': 'static_VM_supervisor_guards_passed_before_Torch', 'Torch_imported': False,
               'counters': dict(ledger.counters)})
        with bounded_phase('idle_GPU_before_framework'):
            probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                   capture_output=True, text=True, check=True, timeout=10)
            need(not probe.stdout.strip(), 'Competing GPU workload; do not construct model')
        with bounded_phase('dependencies_and_FP32_policy'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            need(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'), 'Deterministic cuBLAS config required')
            import torch as torch_module
            import numpy as numpy_module
            from PIL import Image
            from torch.nn import functional as F
            from . import arithmetic, quality
            from research.codeformer_u64_gate_aligned_v1 import solver
            torch, np = torch_module, numpy_module
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1
                 and torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one actual L4 required')
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
            need(not torch.is_inference_mode_enabled() and torch.get_default_dtype() == torch.float32,
                 'Ordinary FP32 autograd required')
            caller_rng = {'Python': random.getstate(), 'NumPy': np.random.get_state(),
                          'CPU': torch.get_rng_state().cpu().clone(),
                          'CUDA': [v.cpu().clone() for v in torch.cuda.get_rng_state_all()]}
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
            for module, relative in ((arithmetic, PACKAGE + '/arithmetic.py'), (quality, PACKAGE + '/quality.py'),
                                     (solver, 'research/codeformer_u64_gate_aligned_v1/solver.py')):
                need(Path(module.__file__).resolve() == root / relative, 'Own sealed numerical helper origin differs')

        def arrays(state):
            return {name: value.detach().cpu().contiguous().numpy().copy() for name, value in state.items()}

        def canonical(state):
            return helper.canonical_arrays(arrays(state), np=np)[0]

        def save_array(name, value, emergency=False):
            array = value.detach().cpu().contiguous().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value)
            need(array.dtype.kind in ('f', 'i', 'u') and np.isfinite(array).all(), 'Finite numeric array required')
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            return store.write(name, stream.getvalue(), emergency=emergency)

        def save_npz(name, values, emergency=False):
            payload, description = helper.npz_state_payload(values, np=np)
            binding = store.write(name, payload, emergency=emergency)
            with np.load(checked(root, binding), allow_pickle=False) as durable:
                need(set(durable.files) == set(values) and all(durable[key].dtype == values[key].dtype
                     and durable[key].shape == values[key].shape and durable[key].tobytes() == values[key].tobytes()
                     for key in values), 'Durable NPZ arrays differ')
            return binding, description

        def raw_file(binding):
            with checked(parent if binding['path'] in oldp['files'] else root, binding).open('rb') as stream:
                value = np.load(stream, allow_pickle=False)
            need(value.dtype == np.float32 and value.shape == (256, 256, 3) and np.isfinite(value).all()
                 and value.min() >= 0 and value.max() <= 1, 'Exact delivered FP32 RGB256 required')
            return value

        def rgb(binding):
            base = parent if binding['path'] in oldp['files'] else root
            with Image.open(checked(base, binding)) as image:
                need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256)
                     and getattr(image, 'n_frames', 1) == 1, 'Exact single RGB256 PNG required')
                return np.asarray(image).copy()

        def rng_record():
            ns = np.random.get_state()
            return {'Python': repr(random.getstate()),
                    'NumPy': [ns[0], ns[1].tobytes().hex(), int(ns[2]), int(ns[3]), float(ns[4])],
                    'CPU': torch.get_rng_state().cpu().numpy().tobytes().hex(),
                    'CUDA': [v.cpu().numpy().tobytes().hex() for v in torch.cuda.get_rng_state_all()]}

        with bounded_phase('exact_original_u64_initialization_no_forward'):
            r2_path, u96_path = (checked(parent, oldp['model'][key]) for key in ('r2', 'u96'))
            r2 = torch.load(r2_path, map_location='cpu', weights_only=True)
            u96 = torch.load(u96_path, map_location='cpu', weights_only=True)
            need(type(r2) is dict and set(r2) == {'params_ema'} and type(u96) is dict
                 and set(u96) == {'format', 'adapter_state'}
                 and u96['format'] == 'codeformer-spatial-adapter-inference-v1', 'Exact initializer schemas required')
            need(canonical(r2['params_ema']) == model.R2_CANONICAL_SHA256
                 and canonical(u96['adapter_state']) == model.U96_CANONICAL_SHA256, 'Exact frozen initializer values required')
            from third_party.codeformer.codeformer_arch import CodeFormer
            for relative, digest in model.ARCHITECTURE_SOURCES.items():
                module = sys.modules.get(relative[:-3].replace('/', '.'))
                need(module is not None and Path(module.__file__).resolve() == parent / relative
                     and runtime.sha256(parent / relative) == digest, 'Exact parent architecture origin required')
            before = rng_record()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261023)
                base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(model.SITES))
                base.load_state_dict(r2['params_ema'], strict=True)
                base = base.float().eval().requires_grad_(False).to(device)
            candidate = model.build_restorer(base, u96['adapter_state'], r2_artifact=r2_path, u96_artifact=u96_path,
                head_artifact=checked(parent, oldp['u64_head']), manual_vm_root=parent, device='cuda:0',
                acknowledgement=model.MANUAL_ACKNOWLEDGEMENT,
                prototype_source_sha256=oldp['files'][PARENT_PACKAGE + '/model.py']['sha256']).eval()
            need(before == rng_record(), 'Original-u64 construction changed caller RNG')
            named = dict(candidate.named_parameters())
            head_items = [(name, named[name]) for name in sorted(contract.head_shapes())]
            need(len(named) == 575 and len(head_items) == 36 and sum(v.numel() for _, v in head_items) == 168960,
                 'Exact575/36/168960 parameter census required')
            need(all(v.requires_grad == name.startswith('heads.') and v.grad is None for name, v in named.items()),
                 'Only36 heads require gradients; all .grad fields must be None')
            original_heads = arrays(dict(head_items))
            frozen_initial = arrays({name: value for name, value in named.items() if not name.startswith('heads.')})
            frozen_sha = helper.canonical_arrays(frozen_initial, np=np)[0]
            need(canonical(named) == contract.FULL_STATE_CANONICAL_SHA256
                 and helper.canonical_arrays(original_heads, np=np)[0] == contract.HEAD_CANONICAL_SHA256,
                 'Exact complete original-u64 state required, never failed-new8')
            del r2, u96
            run_rng = rng_record()
            store.json('preflight.json', {'complete': True, 'protocol_sha256': expected_sha,
                'initializer': candidate.u64_initializer_receipt, 'frozen539_canonical_sha256': frozen_sha,
                'full575_canonical_sha256': contract.FULL_STATE_CANONICAL_SHA256,
                'torch': torch.__version__, 'GPU': torch.cuda.get_device_name(0), 'limits': contract.LIMITS,
                'optimizer': None, 'model_qualified': False})

        def assert_frozen():
            candidate.validate_invariants(check_frozen_digests=True)
            actual = arrays({name: value for name, value in named.items() if not name.startswith('heads.')})
            need(set(actual) == set(frozen_initial) and len(actual) == 539
                 and all(actual[key].dtype == frozen_initial[key].dtype and actual[key].shape == frozen_initial[key].shape
                         and actual[key].tobytes() == frozen_initial[key].tobytes() for key in actual)
                 and helper.canonical_arrays(actual, np=np)[0] == frozen_sha, 'Every frozen539 value must remain exact')
            need(all(value.grad is None for value in candidate.parameters()), 'All .grad fields must stay None')

        def full_state(label, emergency=False):
            nonlocal full_states_attempted, last_complete_state
            need(full_states_attempted < 2, 'At most two full575 payload attempts')
            assert_frozen()
            actual = arrays(candidate.capture_complete_state_for_audit())
            need(len(actual) == 575, 'Complete575 values required')
            full_states_attempted += 1
            binding, description = save_npz('full_states/' + label + '.npz', actual, emergency)
            need(binding['bytes'] <= 384 * 1024**2, 'Complete575 state384MiB cap')
            last_complete_state = store.json('full_states/' + label + '.json', {
                'complete': True, 'update': active_update, 'state': binding, 'description': description,
                'canonical_sha256': helper.canonical_arrays(actual, np=np)[0], 'all539_frozen_exact': True,
                'head36_canonical_sha256': canonical(dict(head_items)), 'model_qualified': False}, emergency=emergency)
            return last_complete_state

        def checkpoint(update, next_slot):
            nonlocal last_checkpoint
            assert_frozen()
            folder = 'checkpoints/u' + str(update).zfill(4)
            head_file, description = save_npz(folder + '/heads.npz', arrays(dict(head_items)))
            resume = {'format': 'u64-gate-aligned-closed-finite-state-v1', 'protocol_sha256': expected_sha,
                'update': update, 'optimizer': None, 'optimizer_constructions': 0,
                'gradient_descent_parameter_applications': ledger.counters['parameter_applications'],
                'parameter_names': [name for name, _ in head_items], 'RNG': rng_record(),
                'sampler': {'fixed_plan': True, 'next_slot': next_slot, 'automatic_resume_permitted': False},
                'frozen539_canonical_sha256': frozen_sha, 'initial_head_canonical_sha256': contract.HEAD_CANONICAL_SHA256,
                'head_canonical_sha256': description['canonical_sha256'], 'counters': dict(ledger.counters),
                'model_qualified': False}
            resume_file = store.json(folder + '/resume.json', resume)
            need(runtime.read_json(checked(root, resume_file)) == resume, 'Durable RNG/sampler/null-optimizer differs')
            last_checkpoint = store.json(folder + '/checkpoint.json', {'complete': True, 'update': update,
                'heads': head_file, 'head_description': description, 'resume': resume_file,
                'frozen539_canonical_sha256': frozen_sha, 'model_qualified': False})
            status('head_checkpoint_saved')

        def individual_changes(current, original, prior):
            """Report every saved case, including unchanged legacy guards."""
            need(set(current) == set(original) == set(prior), 'Identical56 quality IDs required')
            changes = []
            for key in sorted(current):
                reference = rows[key]
                region = 'central128' if reference['role'] == contract.ROLES[0] else 'whole'
                for view in ('raw', 'native_png'):
                    for field in ('mae', 'adjacent_rgb_error'):
                        values = []
                        for bank in (current, original, prior):
                            row = bank[key]
                            values.append(row['regional_metrics'][view][region][field]
                                          if reference.get('target') is not None else row['metrics'][view][field])
                        now, start, preceding = values
                        if now is None or start is None or preceding is None:
                            need(now is start is preceding is None, 'Only null-target photographic metrics may be unavailable')
                            changes.append({'id': key, 'role': reference['role'], 'view': view, 'field': field,
                                            'region': region, 'available': False, 'reason': 'No aligned photographic target'})
                        else:
                            changes.append({'id': key, 'role': reference['role'], 'view': view, 'field': field,
                                'region': region, 'available': True, 'current': now, 'original_u64': start,
                                'prior_state': preceding, 'delta_original': now - start, 'delta_prior': now - preceding,
                                'regresses_original': now > start, 'regresses_prior': now > preceding})
            return changes

        controls = {row['id']: row for row in oldp['controls']}
        parity = {row['id']: row for row in oldp['u64_parity']}
        bank_ids = {row['id'] for row in selection['monitors'] + selection['guards']}
        need(set(controls) == bank_ids and set(parity) == {row['id'] for row in selection['monitors'][:4]},
             'Exact56 U96 controls and4 historical u64 parity witnesses required')
        frozen_records = {}
        with bounded_phase('cached_frozen_U96_controls_zero_neural'):
            for key, control in controls.items():
                a, q = raw_file(control['raw']), rgb(control['png'])
                need(np.array_equal(q, (np.clip(a, 0, 1) * 255).astype(np.uint8)), 'Saved U96 RAW/floorPNG differs')
                row = rows[key]
                target = None if row.get('target') is None else rgb(row['target']).astype(np.float32) / np.float32(255)
                teacher = row.get('behavior_teacher')
                options = {} if teacher is None else {'teacher_raw': raw_file(teacher['raw']), 'teacher_png': rgb(teacher['png'])}
                frozen_records[key] = {**{name: row[name] for name in ('id', 'reference_id', 'role', 'split')},
                    **metrics.record_metrics(a, q, target, row, np=np, frozen_U96_raw=a, frozen_U96_png=q, **options)}
                saved = runtime.strict_json(checked(parent, control['record']).read_bytes())
                need(saved['id'] == key and saved['metrics'] == frozen_records[key]['metrics']
                     and saved['regional_metrics'] == frozen_records[key]['regional_metrics'], 'Cached U96 metrics differ')
            store.json('frozen_U96_control_metrics.json', {'complete': True, 'records': frozen_records, 'model_qualified': False})
        with bounded_phase('complete_initial_u64_state_and_checkpoint'):
            full_state('initial')
            checkpoint(0, 1)

        @contextmanager
        def query_warning_scope():
            need(torch.are_deterministic_algorithms_enabled()
                 and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Strict forward policy before query required')
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    yield
            finally:
                torch.use_deterministic_algorithms(True, warn_only=False)
                actual = [{'category': value.category.__name__, 'message': str(value.message)} for value in captured]
                unexpected = [value for value in actual if ('deterministic' in value['message'].lower()
                    or 'nondeterminism' in value['message'].lower())
                    and re.match(r'^upsample_bilinear2d_backward_out_cuda\b', value['message']) is None]
                event({'kind': 'gradient_warning_scope', 'warnings': actual, 'unexpected': unexpected,
                       'strict_flags_restored': True, 'bitwise_CUDA_gradient_repeatability_claimed': False,
                       'counters': dict(ledger.counters)})
                need(not unexpected, 'Unexpected nondeterministic gradient operation')

        def execute_forward(call):
            row = rows[call['id']]
            observed = rgb(row['input'])
            need(hashlib.sha256(observed.tobytes()).hexdigest() == row['input_RGB8_sha256'], 'Admitted input RGB differs')
            sample = torch.from_numpy(observed.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)[None] / 255.
            sample = F.interpolate(sample, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            target_rgb = None if row.get('target') is None else rgb(row['target'])
            if target_rgb is not None:
                need(hashlib.sha256(target_rgb.tobytes()).hexdigest() == row['target_RGB8_sha256'], 'Admitted target RGB differs')
            target = None if target_rgb is None else torch.from_numpy(target_rgb.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)[None] / 255.
            before_sha = canonical(dict(head_items))
            resource('before_forward')
            token = ledger.reserve('forward', id=call['id'], plan_slot=call['slot'],
                                   new_update=call['new_update'], category=call['category'])
            status('forward_reserved')
            with torch.set_grad_enabled(bool(call['gradient_queries'])), torch.autocast(device_type='cuda', enabled=False):
                need(torch.are_deterministic_algorithms_enabled()
                     and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Strict deterministic full forward required')
                triple = candidate(sample)
                torch.cuda.synchronize(0)
                ledger.returned(token, synchronized_return=True)
                status('forward_returned')
                valid = isinstance(triple, tuple) and len(triple) == 3 and all(isinstance(value, torch.Tensor)
                    and value.dtype == torch.float32 and value.device == device and tuple(value.shape) == shape
                    and torch.isfinite(value).all().item() for value, shape in zip(triple,
                    ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))))
                if not valid:
                    retained = []
                    for index, value in enumerate(triple if isinstance(triple, (tuple, list)) else (triple,)):
                        if isinstance(value, torch.Tensor) and value.numel() * value.element_size() <= 8 * 1024**2:
                            bits = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
                            retained.append({'index': index, 'shape': list(value.shape), 'dtype': str(value.dtype),
                                'bits': save_array('invalid_returns/' + str(call['slot']) + '/' + str(index) + '.npy', bits)})
                    store.json('invalid_returns/' + str(call['slot']) + '/record.json', {'accepted_as_valid': False, 'values': retained})
                    raise ValueError('Invalid full return; actual bits retained within cap, no retry')
                need(candidate.last_base_calls == 1 and tuple(candidate.last_forward_sites) == tuple(model.SITES),
                     'One complete four-site forward required')
                native = (triple[0] + 1) / 2
                delivered = F.interpolate(native.clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
                raw = delivered.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
                quantized = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
                stem = 'calls/' + str(call['slot']).zfill(4)
                image = io.BytesIO()
                Image.fromarray(quantized).save(image, format='PNG')
                need(len(image.getvalue()) <= 256 * 1024, 'Per-image PNG256KiB cap')
                outputs = {'raw': save_array(stem + '/raw.npy', raw),
                           'png': store.write(stem + '/result.png', image.getvalue())}
                stock = {}
                witness = row['id'] in parity and call['new_update'] == 0
                if witness:
                    outputs['unclipped'] = save_array(stem + '/unclipped.npy', F.interpolate(native, (256, 256), mode='bilinear', align_corners=False))
                for name, value in zip(('native', 'logits', 'LQ'), triple):
                    array = value.detach().cpu().contiguous().numpy()
                    stock[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                   'bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()}
                    if witness:
                        stock[name]['file'] = save_array(stem + '/' + name + '.npy', value)
                need(np.array_equal(rgb(outputs['png']), quantized), 'Saved PNG readback differs')
                target_np = None if target_rgb is None else target_rgb.astype(np.float32) / np.float32(255)
                if row['id'] in controls:
                    control = controls[row['id']]
                    options = {'frozen_U96_raw': raw_file(control['raw']), 'frozen_U96_png': rgb(control['png'])}
                    if row.get('behavior_teacher') is not None:
                        teacher = row['behavior_teacher']
                        options.update(teacher_raw=raw_file(teacher['raw']), teacher_png=rgb(teacher['png']))
                    measured = metrics.record_metrics(raw, quantized, target_np, row, np=np, **options)
                else:
                    measured = metrics.record_training_metrics(raw, quantized, target_np, row, np=np)
                record = {'complete': True, **{name: row[name] for name in ('id', 'reference_id', 'role', 'split')},
                          'call': call, 'update': active_update, 'input': row['input'], 'target': row.get('target'),
                          'outputs': outputs, 'stock': stock, 'head_before_sha256': before_sha,
                          'frozen539_canonical_sha256': frozen_sha, 'model_qualified': False, **measured}
                record_pin = store.json(stem + '/record.json', record)
                record['receipt'] = record_pin
                query_records, vectors = {}, {}
                if call['gradient_queries']:
                    need(row.get('cohort') == 'optimize' and row.get('optimization_permitted') is True
                         and target is not None and row['split'] == 'TRAIN'
                         and tuple(call['gradient_queries']) == contract.QUERIES, 'Only original admitted TRAIN queries allowed')
                    residual = delivered - target
                    if call['metric_region'] == 'central128':
                        residual = residual[:, :, 64:192, 64:192]
                    else:
                        need(call['metric_region'] == 'whole', 'Declared gate metric region required')
                    values = {'gate_mae': residual.abs().mean(), 'gate_adjacent': .5 * (
                        (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().mean()
                        + (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().mean())}
                    for index, query in enumerate(contract.QUERIES):
                        nonlocal active_query
                        active_query = query
                        resource('before_gradient_query')
                        token = ledger.reserve('gradient_query', id=call['id'], plan_slot=call['slot'],
                                               new_update=call['new_update'], query=query)
                        status('gradient_query_reserved')
                        with query_warning_scope():
                            result = torch.autograd.grad(values[query], tuple(value for _, value in head_items),
                                retain_graph=index == 0, create_graph=False, allow_unused=True, materialize_grads=False)
                            torch.cuda.synchronize(0)
                            ledger.returned(token, synchronized_return=True)
                            status('gradient_query_returned')
                        actual, unused, zero = {}, [], []
                        for (name, parameter), gradient in zip(head_items, result):
                            if gradient is None:
                                unused.append(name)
                                gradient = torch.zeros_like(parameter)
                            need(gradient.dtype == torch.float32 and gradient.shape == parameter.shape
                                 and torch.isfinite(gradient).all().item(), 'Finite connected FP32 gradient schema required')
                            if not torch.count_nonzero(gradient).item():
                                zero.append(name)
                            actual[name] = gradient.detach().cpu().contiguous().numpy().copy()
                        binding, description = save_npz(stem + '/gradients_' + query + '.npz', actual)
                        query_records[query] = {'file': binding, 'description': description,
                            'canonical_sha256': helper.canonical_arrays(actual, np=np)[0],
                            'scalar': float(values[query].detach().item()), 'unused_names_explicit_zero': unused,
                            'all_zero_names': zero, 'head_before_sha256': before_sha}
                        store.json(stem + '/gradient_' + query + '_receipt.json', query_records[query])
                        need(not unused and not zero, 'Mandatory connected tensor gradient missing/all-zero; preserve measured receipt and stop')
                        vectors[query] = np.concatenate([actual[name].reshape(-1).astype(np.float64) for name, _ in head_items])
                        need(np.linalg.norm(vectors[query]) > 0, 'Zero mandatory gate vector; no fallback')
                        need(all(value.grad is None for value in candidate.parameters()), 'Autograd query changed a .grad field')
                        del result, actual
                        resource('after_gradient_query')
                    store.json(stem + '/gradients.json', {'complete': True, 'call': call,
                        'head_before_sha256': before_sha, 'output_record': record_pin, 'gradients': query_records,
                        'RAW_differentiable_metrics_only': True, 'PNG_gradient_claimed': False, 'model_qualified': False})
                need(canonical(dict(head_items)) == before_sha, 'Forward/query changed actual head values')
                return record, vectors, query_records

        initial_bank, prior_bank, current_bank = {}, {}, {}
        pooled_rows = None
        for call in plan:
            active_slot, active_case, active_update = call['slot'], call['id'], call['new_update']
            active_query = None
            if call['kind'] == 'forward':
                with bounded_phase('call_' + str(active_slot).zfill(4)):
                    record, vectors, queries = execute_forward(call)
                    if call['phase'] == 'initial_u64_quality_bank':
                        initial_bank[call['id']] = record
                        if call['id'] in parity:
                            witness = parity[call['id']]
                            need(np.array_equal(raw_file(record['outputs']['raw']), raw_file(witness['raw']))
                                 and np.array_equal(rgb(record['outputs']['png']), rgb(witness['png'])),
                                 'Exact original-u64 historical RAW/PNG parity failed')
                    elif call['phase'] == 'deterministic_same_u64_repeat':
                        original = initial_bank[call['id']]
                        for field in ('raw', 'unclipped'):
                            need({key: record['outputs'][field][key] for key in ('bytes', 'sha256')}
                                 == {key: original['outputs'][field][key] for key in ('bytes', 'sha256')},
                                 'Same-u64 deterministic array repeat differs')
                        need(np.array_equal(rgb(record['outputs']['png']), rgb(original['outputs']['png'])), 'Same-u64 PNG repeat differs')
                        for name in ('native', 'logits', 'LQ'):
                            need({key: value for key, value in record['stock'][name].items() if key != 'file'}
                                 == {key: value for key, value in original['stock'][name].items() if key != 'file'},
                                 'Same-u64 native/logits/LQ repeat differs')
                        if call['slot'] == 60:
                            gate = metrics.evaluate_bank(initial_bank, initial_bank, frozen_records, standard_raw,
                                                         phase='initializer', saved_audit_bytes=audit_bytes)
                            store.json('banks/u0000/bank.json', {'complete': True, 'records': initial_bank, 'gate': gate})
                            need(gate['pass'], 'Exact original-u64 initializer failed original screen')
                            prior_bank = initial_bank
                    elif call['phase'] == 'paired_TRAIN_gate_gradients':
                        if pooled_rows is None:
                            pooled_rows = {role: {query: [] for query in contract.QUERIES} for role in contract.ROLES}
                            update_gradient_records = []
                            gradient_state_sha = record['head_before_sha256']
                        need(gradient_state_sha == record['head_before_sha256'], 'All128 queries must share current head state')
                        for query in contract.QUERIES:
                            pooled_rows[call['role']][query].append(vectors[query])
                        update_gradient_records.append({'call': call, 'queries': queries})
                    else:
                        need(call['phase'] == 'current_quality_bank', 'Known quality phase required')
                        current_bank[call['id']] = record
                        if len(current_bank) == 56:
                            inherited = metrics.evaluate_bank(current_bank, initial_bank, frozen_records, standard_raw,
                                                              phase='u4', saved_audit_bytes=audit_bytes)
                            means = [metrics.cohort_means(bank, standard_raw) for bank in (current_bank, initial_bank, prior_bank)]
                            gate = quality.evaluate_quality(means[0], means[1], means[2], active_update == 4, inherited)
                            store.json('banks/u' + str(active_update).zfill(4) + '/bank.json', {
                                'complete': True, 'records': current_bank, 'gate': gate,
                                'inherited': inherited, 'individual_changes_reported': True,
                                'individual_changes': individual_changes(current_bank, initial_bank, prior_bank),
                                'current_means': means[0], 'original_means': means[1], 'prior_means': means[2]})
                            assert_frozen()
                            print(json.dumps({'gate_aligned_snapshot': active_update, 'pass': gate['pass']}), flush=True)
                            if not gate['pass']:
                                raise QualityStop('Structure/preservation screen failed; retain actual state, no continuation')
                            prior_bank, current_bank = current_bank, {}
            else:
                with bounded_phase('gate_aligned_update_' + str(active_update)):
                    need(call['kind'] == 'parameter_application' and pooled_rows is not None, 'All gate queries precede application')
                    for role in contract.ROLES:
                        for query in contract.QUERIES:
                            need(len(pooled_rows[role][query]) == 16, 'Exactly16 same-current-state photos per gate required')
                    pooled = {role: {query: np.mean(np.stack(pooled_rows[role][query]), axis=0, dtype=np.float64)
                              for query in contract.QUERIES} for role in contract.ROLES}
                    ordered = np.stack([pooled[role][query] for role in contract.ROLES for query in contract.QUERIES])
                    prefix = 'updates/u' + str(active_update).zfill(4)
                    # The parent's canonical NPZ writer intentionally accepts
                    # FP32 model tensors only. Preserve these actual FP64 pooled
                    # vectors without changing their precision for the solver.
                    pooled_file = save_array(prefix + '/pooled_gate_gradients.npy', ordered)
                    solved = solver.solve_eight_pooled_gates(pooled)
                    direction = solved.pop('direction')
                    certificate = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in solved.items()}
                    save_array(prefix + '/direction.npy', direction)
                    store.json(prefix + '/solver.json', certificate)
                    before_heads = arrays(dict(head_items))
                    need(helper.canonical_arrays(before_heads, np=np)[0] == gradient_state_sha, 'Current state differs from query state')
                    proposed = arithmetic.propose_fp32_step(before_heads, original_heads, direction, ordered)
                    candidate_file, candidate_description = save_npz(prefix + '/proposed_heads.npz', proposed['candidate'])
                    delta_file = save_array(prefix + '/actual_rounded_delta.npy', proposed['actual_delta'])
                    store.json(prefix + '/precommit.json', {'complete': True, 'certificate': proposed['certificate'],
                        'candidate': candidate_file, 'candidate_description': candidate_description, 'actual_delta': delta_file,
                        'pooled_gate_gradients': pooled_file,
                        'all_query_receipts': update_gradient_records, 'head_before_sha256': gradient_state_sha,
                        'no_finite_PNG_success_claim': True})
                    token = ledger.reserve('parameter_application', id=call['id'], plan_slot=call['slot'], new_update=active_update)
                    status('parameter_application_reserved')
                    with torch.no_grad():
                        for name, parameter in head_items:
                            parameter.copy_(torch.from_numpy(proposed['candidate'][name]).to(device=device, dtype=torch.float32))
                    torch.cuda.synchronize(0)
                    ledger.returned(token, synchronized_return=True)
                    status('parameter_application_returned')
                    actual = arrays(dict(head_items))
                    need(all(actual[name].tobytes() == proposed['candidate'][name].tobytes() for name in actual),
                         'Actual committed values differ from FP32 precommit candidate')
                    assert_frozen()
                    store.json(prefix + '/commit.json', {'complete': True, 'new_update': active_update,
                        'head_before_sha256': gradient_state_sha, 'head_after_sha256': helper.canonical_arrays(actual, np=np)[0],
                        'candidate': candidate_file, 'actual_delta': delta_file, 'rounded_certificate': proposed['certificate'],
                        'all539_frozen_exact': True, 'optimizer': None, 'counters': dict(ledger.counters), 'model_qualified': False})
                    checkpoint(active_update, call['slot'] + 1)
                    pooled_rows = None
                    del pooled, ordered, proposed, actual, before_heads, direction, solved
                    print(json.dumps({'gate_aligned_update': active_update, 'maximum': 4}), flush=True)
        with bounded_phase('closing_full575_and_terminal_receipt'):
            full_state('closing')
            need(rng_record() == run_rng, 'Fixed plan/model operations changed RNG')
            expected = {key: contract.LIMITS[key] for key in COUNTER_KEYS}
            need(ledger.counters == expected and not ledger.pending, 'Complete known540/512/4 execution required')
            store.audit_namespace()
            store.json('result.json', {'complete': True, 'status': 'FINITE_GATE_ALIGNED_TRAIN_SCREEN_PASSED_NOT_QUALIFIED',
                'protocol_sha256': expected_sha, 'last_complete_state': last_complete_state, 'last_checkpoint': last_checkpoint,
                'optimizer_constructions': 0, 'backward_calls': 0, 'model_qualified': False,
                'inherited_broader_DEV_failures_remain_binding': True, **ledger.snapshot()})
            event({'kind': 'run_result_saved', 'counters': dict(ledger.counters)})
            status('finite_screen_complete', complete=True)
        return 0
    except BaseException as error:
        if store is not None:
            failure = {'complete': False, 'quality_stop': isinstance(error, QualityStop),
                'protocol_sha256': expected_sha, 'phase': failed_phase or phase,
                'active_update': active_update, 'active_slot': active_slot, 'active_case': active_case, 'active_query': active_query,
                'error_type': type(error).__name__, 'error': str(error)[:4000], 'traceback': traceback.format_exc()[-20000:],
                'last_checkpoint': last_checkpoint, 'last_complete_state': last_complete_state,
                'model_qualified': False, **(ledger.snapshot() if ledger else {})}
            if candidate is not None and frozen_initial is not None and full_states_attempted < 2:
                budget = min(30., max(0., contract.LIMITS['external_seconds'] - (time.monotonic() - began) - 10.))
                if budget:
                    try:
                        with timed(budget, 'bounded_failed_current_state'):
                            try:
                                failure['failed_current_state'] = full_state('failed_current', emergency=True)
                            except TimeoutError:
                                raise
                            except BaseException as validation_error:
                                failure['validated_failed_state_error'] = type(validation_error).__name__ + ': ' + str(validation_error)[:1000]
                                if full_states_attempted < 2:
                                    actual = arrays(candidate.state_dict())
                                    need(len(actual) == 575 and all(value.dtype == np.float32 for value in actual.values()),
                                         'Unvalidated failed state still needs actual575 FP32 arrays')
                                    payload = io.BytesIO()
                                    np.savez(payload, **actual)
                                    need(len(payload.getvalue()) <= 384 * 1024**2, 'Failed-state384MiB cap')
                                    full_states_attempted += 1
                                    failure['unvalidated_failed575'] = store.write('failure_values/current575.npz', payload.getvalue(), emergency=True)
                                    failure['unvalidated_failed575_accepted_as_valid'] = False
                            failure['failed_RNG_sampler'] = store.json('failure_values/RNG_sampler.json', {
                                'complete': True, 'RNG': rng_record(), 'active_update': active_update,
                                'active_plan_slot': active_slot, 'active_case': active_case,
                                'active_query': active_query, 'optimizer': None,
                                'automatic_resume_permitted': False, **ledger.snapshot()}, emergency=True)
                    except BaseException as capture:
                        failure['failed_state_capture_error'] = type(capture).__name__ + ': ' + str(capture)[:1000]
            failure['partial_files'] = sorted(store.partial_files)
            try:
                with timed(5., 'bounded_failure_receipt'):
                    store.json('failure.json', failure, emergency=True)
                    status('retained_quality_stop' if isinstance(error, QualityStop) else 'retained_failure')
            except BaseException as failed:
                print(json.dumps({'failure_receipt_incomplete': type(failed).__name__, 'partial_outcome_retained': True}), flush=True)
        traceback.print_exc()
        return 2 if isinstance(error, QualityStop) else 1
    finally:
        if caller_rng is not None:
            with timed(5., 'caller_RNG_restoration'):
                random.setstate(caller_rng['Python'])
                np.random.set_state(caller_rng['NumPy'])
                torch.set_rng_state(caller_rng['CPU'])
                torch.cuda.set_rng_state_all(caller_rng['CUDA'])
                torch.cuda.synchronize(0)
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
