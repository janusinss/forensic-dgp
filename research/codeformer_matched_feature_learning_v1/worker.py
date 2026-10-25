"""Finite manual-L4 matched worker source; importing this module is inert.

Only run() in the independently sealed manual stage may import a numerical
framework or construct a model. The two arms share admitted data and the exact
original trained-u64 initializer; a finite TRAIN screen never qualifies one.
"""
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
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


PACKAGE = 'research/codeformer_matched_feature_learning_v1'
HELPER = 'research/codeformer_u96_spatial_feasibility_v1/worker.py'
HELPER_PIN = (72086, 'fae271100642604ae0682ebf52ec163e768fa6882eced87e0c34e587980126b6')
HEAD_DECODER = 'research/codeformer_u64_cleananchors_continuation_v1/model.py'
HEAD_DECODER_PIN = (21294, '6a67468fd6b94e938efe8548205282c048635fcc63e8e4efd8389a2910129829')
ARITHMETIC_ATOL = 2e-7
ARITHMETIC_RTOL = 2e-6
SEED = 20261024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def supervisor_context(sha, *, environ, parent_pid):
    need(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha)
         and environ.get('FORENSIC_MATCHED_FEATURE_SUPERVISED') == '1'
         and environ.get('FORENSIC_MATCHED_FEATURE_PROTOCOL') == sha
         and environ.get('FORENSIC_MATCHED_FEATURE_SUPERVISOR_PID') == str(parent_pid),
         'Exact seal and actual finite matched supervisor parent required')


def arithmetic_close(current, reference):
    """Numerical objective parity only; never a quality-gate tolerance."""
    return (type(current) in (int, float) and type(reference) in (int, float)
            and math.isfinite(current) and math.isfinite(reference)
            and abs(current - reference) <= ARITHMETIC_ATOL + ARITHMETIC_RTOL * abs(reference))


def run(root, protocol, expected_sha):
    """Execute only a fresh sealed, supervised, manual tmux/NVIDIA-L4 stage."""
    from . import packet, contract, execution
    root = Path(root).absolute()
    began = time.monotonic()
    budget = execution.TimeBudget(began)
    limits = contract.LIMITS
    store = ledger = helper = torch = np = None
    candidates, optimizers, named, trainable = {}, {}, {}, {}
    initial_trainable, initial_records, initial_gates = {}, {}, {}
    last_checkpoint, last_full_state, committed = {}, {}, dict.fromkeys(contract.ARMS, 0)
    arm_results, arm_stops = {}, {}
    frozen_baseline = None
    full_state_count = 0
    phase = 'static_guards'
    failed_phase = failed_arm = None
    active_arm = active_case = None
    active_update = 0
    old_term = None
    caller_rng = None
    fatal_error = None
    proposed_exit = 1

    def snapshot():
        return ledger.summary() if ledger is not None else {
            'status': 'NOT_STARTED', 'counters': {}, 'pending': None,
            'bookkeeping_complete': False, 'resume_permitted': False}

    def event(value):
        store.append('events.jsonl', {'UTC': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - began, 'phase': phase,
            'arm': active_arm, 'update': active_update, **value})

    def status(label, *, complete=False):
        if store is not None:
            state = snapshot()
            store.status({'complete': complete, 'protocol_sha256': expected_sha,
                'status': 'FINITE_MATCHED_TRAIN_COMPARISON_NOT_QUALIFIED', 'label': label,
                'phase': failed_phase or phase, 'active_arm': active_arm or failed_arm, 'active_case': active_case,
                'active_update': active_update, 'elapsed_seconds': time.monotonic() - began,
                'last_checkpoint': copy.deepcopy(last_checkpoint),
                'last_complete_state': copy.deepcopy(last_full_state),
                'arm_quality_stops': copy.deepcopy(arm_stops),
                'operation_outcome_unknown': state.get('pending') is not None
                    or state.get('status') == 'JOURNAL_FAILED',
                'model_qualified': False, **state})

    def checked(binding):
        need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'},
             'Exact file binding required')
        path = packet.regular(root / binding['path'])
        need(path.is_relative_to(root) and packet.pin(root, path) == binding,
             'Actual file differs from bound stage bytes')
        return path

    def source_module(relative, module_name, fixed=None):
        source = packet.regular(root / relative)
        actual = packet.pin(root, source)
        need(p['files'].get(relative) == {'bytes': actual['bytes'], 'sha256': actual['sha256']},
             'Active source missing from exact sealed manifest: ' + relative)
        if fixed is not None:
            need((actual['bytes'], actual['sha256']) == fixed, 'Exact retained dependency required: ' + relative)
        module = importlib.import_module(module_name)
        need(Path(module.__file__).resolve() == source, 'Active source import origin differs: ' + relative)
        return module

    def arm_used(arm, now):
        return budget.arm_seconds[arm] + (
            now - budget.active_since if budget.active_arm == arm else 0.0)

    def resource(label, *, scientific=True):
        now = time.monotonic()
        free = shutil.disk_usage(root).free
        rss = helper.linux_current_rss_bytes(Path('/proc/self/statm').read_text(),
                                             int(os.sysconf('SC_PAGE_SIZE')))
        peak = int(linux_resource.getrusage(linux_resource.RUSAGE_SELF).ru_maxrss) * 1024
        cuda = peak_cuda = 0
        if torch is not None and torch.cuda.is_initialized():
            cuda = int(torch.cuda.memory_reserved(0))
            peak_cuda = int(torch.cuda.max_memory_reserved(0))
        row = {'label': label, 'elapsed_seconds': now - began, 'free_disk_bytes': free,
            'current_RSS_bytes': rss, 'peak_RSS_bytes': peak,
            'CUDA_reserved_bytes': cuda, 'CUDA_peak_reserved_bytes': peak_cuda,
            'output_bytes': store.bytes, 'active_arm_seconds': {
                arm: arm_used(arm, now) for arm in contract.ARMS}}
        store.append('resources.jsonl', row)
        if scientific:
            need(now - began < limits['worker_seconds'], 'Finite5400s scientific deadline')
            if budget.active_arm is not None:
                need(arm_used(budget.active_arm, now) < limits['per_arm_seconds'],
                     'Finite2700s active-arm deadline')
        need(free >= limits['continuing_free_disk_bytes'], 'Continuing4GiB disk reserve')
        need(max(rss, peak) <= limits['maximum_RSS_bytes'], 'Finite10GiB RSS stop')
        need(max(cuda, peak_cuda) <= limits['maximum_CUDA_reserved_bytes'], 'Finite18GiB CUDA reserve stop')
        need(store.bytes <= limits['maximum_output_bytes'], 'Finite12GiB output stop')
        return row

    @contextmanager
    def timed(seconds, label):
        need(seconds > 0 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
             'Fresh positive finite timer required')
        old = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError('Finite matched stop: ' + label)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    @contextmanager
    def bounded_phase(label):
        nonlocal phase, failed_phase, failed_arm
        previous, phase = phase, label
        start = time.monotonic()
        try:
            resource('before_phase')
            allowance = budget.deadline(time.monotonic(), phase_started_at=start)
            with timed(allowance['remaining_seconds'], label):
                yield
                resource('after_phase')
                event({'kind': 'phase_returned', 'seconds': time.monotonic() - start})
        except BaseException:
            failed_phase, failed_arm = label, active_arm
            raise
        finally:
            phase = previous

    @contextmanager
    def charge_arm(arm):
        nonlocal active_arm
        need(active_arm is None, 'Only one arm is active at a time')
        active_arm = arm
        budget.enter_arm(arm, time.monotonic())
        try:
            yield
        finally:
            budget.leave_arm(time.monotonic())
            active_arm = None

    try:
        p = packet.verify_static(root, expected_sha)
        need(p == protocol and p['comparison'] == contract.prospective_contract()
             and p['limits'] == limits, 'Exact sealed matched comparison required')
        need(Path(__file__).resolve() == root / PACKAGE / 'worker.py', 'Use own staged matched worker')
        facts = packet.require_vm(root)
        supervisor_context(expected_sha, environ=os.environ, parent_pid=os.getppid())
        need(not (root / contract.OUTPUT).exists(), 'Prior complete or partial matched run immutable')
        helper = source_module(HELPER, 'research.codeformer_u96_spatial_feasibility_v1.worker', HELPER_PIN)
        import resource as linux_resource
        admission_raw = checked(p['admission']).read_bytes()
        candidate_raw = checked(p['candidate_data']).read_bytes()
        parent_raw = checked(p['parent_protocol']).read_bytes()
        audit_raw = checked(p['historical_pixel_audit']).read_bytes()
        checked(p['historical_archive_index'])
        selection = contract.select_cases(admission_raw, candidate_raw, parent_raw)
        need(selection == p['selection'], 'Selection must be rederived from sealed exact bytes')
        plan = contract.operation_plan(selection)
        store = helper.DurableStore(root, root / contract.OUTPUT,
                                    limits['maximum_output_bytes'], packet.pin)
        for filename in ('events.jsonl', 'resources.jsonl', 'operations.jsonl'):
            store.write(filename, b'')
        def persist(line):
            need(type(line) is bytes and line.endswith(b'\n'), 'Complete canonical journal line required')
            value = packet.strict_json(line)
            need((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode() == line,
                 'Journal sink must preserve exact canonical bytes')
            store.append('operations.jsonl', value)
        ledger = execution.ExecutionLedger(selection, persist=persist)
        store.json('started.json', {'complete': False, 'protocol_sha256': expected_sha,
            'host': facts, 'model_qualified': False})
        store.json('operation_plan.json', {'complete': True, 'calls': plan,
            'selection': selection, 'model_qualified': False})
        old_term = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise TimeoutError('Finite supervisor SIGTERM; retain current values and unknown reservations')
        signal.signal(signal.SIGTERM, interrupted)
        with bounded_phase('pre_framework_L4_and_idle_GPU'):
            names = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=6, check=True)
            need(not names.stderr.strip() and tuple(line.strip() for line in names.stdout.splitlines()
                 if line.strip()) == ('NVIDIA L4',), 'Exactly one pre-framework NVIDIA L4 required')
            processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                '--format=csv,noheader'], capture_output=True, text=True, timeout=6, check=True)
            need(not processes.stdout.strip() and not processes.stderr.strip(), 'Competing GPU workload; stop before construction')
            event({'kind': 'manual_static_supervisor_L4_guards_passed_before_Torch', 'Torch_imported': False})
        with bounded_phase('dependencies_and_deterministic_FP32_policy'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            need(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'), 'Explicit deterministic cuBLAS policy required')
            for name in ('contract', 'execution', 'objective', 'quality', 'records', 'graph', 'model'):
                source_module(PACKAGE + '/' + name + '.py', __package__ + '.' + name)
            from . import objective, quality, records, model
            import torch as torch_module
            import numpy as numpy_module
            from PIL import Image
            from torch.nn import functional as F
            torch, np = torch_module, numpy_module
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1
                 and torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one actual initialized L4 required')
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
            need(not torch.is_inference_mode_enabled() and torch.get_default_dtype() == torch.float32,
                 'Ordinary FP32 autograd runtime required')
            caller_rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
                'CPU': torch.get_rng_state().cpu().clone(),
                'CUDA': [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]}
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.cuda.reset_peak_memory_stats(0)

        def arrays(state):
            return {name: value.detach().cpu().contiguous().numpy().copy() for name, value in state.items()}

        def canonical(state):
            return helper.canonical_arrays(arrays(state), np=np)[0]

        def save_array(relative, value, *, emergency=False):
            array = value.detach().cpu().contiguous().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value)
            need(array.dtype.kind in ('f', 'i', 'u') and np.isfinite(array).all(), 'Finite actual numerical array required')
            stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
            binding = store.write(relative, stream.getvalue(), emergency=emergency)
            with checked(binding).open('rb') as durable:
                saved = np.load(durable, allow_pickle=False)
            need(saved.dtype == array.dtype and saved.shape == array.shape
                 and saved.tobytes() == array.tobytes(), 'Saved numerical array readback differs')
            return binding

        def load_raw(binding):
            with checked(binding).open('rb') as stream:
                value = np.load(stream, allow_pickle=False)
            need(value.dtype == np.float32 and value.shape == (256, 256, 3)
                 and np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1,
                 'Actual finite FP32 HWC256 cached RAW required')
            return value

        def rgb(binding):
            with Image.open(checked(binding)) as image:
                need(image.mode == 'RGB' and image.size == (256, 256)
                     and getattr(image, 'n_frames', 1) == 1, 'Exact single-frame RGB256 file required')
                value = np.asarray(image).copy()
            need(value.dtype == np.uint8 and value.shape == (256, 256, 3), 'Actual decoded RGB8 required')
            return value

        def tensor_summary(value):
            array = value.detach().cpu().contiguous().numpy()
            return {'shape': list(array.shape), 'dtype': str(array.dtype),
                'bytes': array.nbytes, 'bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()}

        def rng_record():
            state = np.random.get_state()
            return {'Python': repr(random.getstate()),
                'NumPy': {'algorithm': state[0], 'keys_hex': state[1].tobytes().hex(),
                    'position': int(state[2]), 'has_gauss': int(state[3]), 'cached_gaussian': float(state[4])},
                'CPU_hex': torch.get_rng_state().cpu().numpy().tobytes().hex(),
                'CUDA_hex': [value.cpu().numpy().tobytes().hex() for value in torch.cuda.get_rng_state_all()]}

        def cpu_tree(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
            if type(value) is dict: return {key: cpu_tree(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)): return type(value)(cpu_tree(child) for child in value)
            return value

        with bounded_phase('load_exact_R2_U96_trained_u64_states_no_forward'):
            r2_path, u96_path = (checked(p['model'][key]) for key in ('r2', 'u96'))
            head_path = checked(p['u64_head'])
            r2 = torch.load(r2_path, map_location='cpu', weights_only=True)
            u96 = torch.load(u96_path, map_location='cpu', weights_only=True)
            need(type(r2) is dict and set(r2) == {'params_ema'} and type(u96) is dict
                 and set(u96) == {'format', 'adapter_state'}
                 and u96['format'] == 'codeformer-spatial-adapter-inference-v1', 'Exact retained initializer formats required')
            need(canonical(r2['params_ema']) == model.R2_CANONICAL_SHA256
                 and canonical(u96['adapter_state']) == model.U96_CANONICAL_SHA256,
                 'Exact retained R2/U96 canonical values required')
            decoder = source_module(HEAD_DECODER,
                'research.codeformer_u64_cleananchors_continuation_v1.model', HEAD_DECODER_PIN)
            head_arrays = decoder.read_terminal_head_state(head_path, np=np)
            head_state = {name: torch.from_numpy(value.copy()) for name, value in head_arrays.items()}
            from third_party.codeformer.codeformer_arch import CodeFormer
            for module_name, binding in model.ARCHITECTURE_SOURCES.items():
                module = sys.modules.get(module_name)
                need(module is not None and Path(module.__file__).resolve() == root / binding['path']
                     and packet.sha256(root / binding['path']) == binding['sha256'],
                     'Exact retained architecture origin and bytes required')
            run_rng = rng_record()
        for arm in contract.ARMS:
            with charge_arm(arm), bounded_phase('construct_original_u64_arm_' + arm):
                before_rng = rng_record()
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(model.INITIALIZATION_SEED)
                    base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8,
                        n_layers=9, connect_list=list(model.SITES))
                    base.load_state_dict(r2['params_ema'], strict=True)
                    base = base.float().eval().requires_grad_(False).to(device)
                need(rng_record() == before_rng, 'Owned R2 construction changed caller RNG')
                candidate = model.build_restorer(base, u96['adapter_state'], head_state, arm=arm,
                    r2_artifact=r2_path, u96_artifact=u96_path, head_artifact=head_path).eval()
                need(rng_record() == before_rng, 'Matched constructor changed caller RNG')
                candidates[arm] = candidate
                named[arm] = dict(candidate.named_parameters())
                permitted = set(model.expected_head_shapes())
                if arm == 'B': permitted |= set(model.expected_adapter_shapes())
                trainable[arm] = [(name, named[arm][name]) for name in sorted(permitted)]
                need({name for name, value in named[arm].items() if value.requires_grad} == permitted,
                     'Exact arm-specific trainable permissions required')
                initial_trainable[arm] = arrays(dict(trainable[arm]))
                store.json('arms/' + arm + '/initializer.json', {
                    'complete': True, 'initializer': candidate.matched_initializer_receipt,
                    'invariants': candidate.validate_invariants(check_frozen_digests=True),
                    'RNG_restored': True, 'model_qualified': False})
        need(candidates['A'].base is not candidates['B'].base
             and all(id(a) != id(b) for a, b in zip(candidates['A'].parameters(), candidates['B'].parameters())),
             'Separate owned models and tensors required for matched arms')
        del base, candidate, r2, u96, head_state, head_arrays

        def assert_frozen(arm):
            receipt = candidates[arm].validate_invariants(check_frozen_digests=True)
            names = [name for name, value in named[arm].items() if not value.requires_grad]
            need(len(names) == (539 if arm == 'A' else 515), 'Exact frozen arm census required')
            for name in names:
                value = named[arm][name]
                need(value.grad is None and np.array_equal(
                    value.detach().cpu().numpy(), frozen_baseline[name]), 'An actual frozen parameter changed')
            return receipt

        def full_state(arm, label, *, emergency=False):
            nonlocal full_state_count
            need(full_state_count < 3, 'At most original baseline plus two arm terminal/failure full states')
            values = arrays(candidates[arm].capture_complete_state_for_audit())
            payload, schema = helper.npz_state_payload(values, np=np)
            need(len(values) == 575 and len(payload) <= 384 * 1024**2,
                 'Exact finite bounded full575 state required')
            full_state_count += 1
            folder = 'states/' + label
            binding = store.write(folder + '/full_state.npz', payload, emergency=emergency)
            with np.load(checked(binding), allow_pickle=False) as saved:
                restored = {name: saved[name] for name in saved.files}
            need(helper.canonical_arrays(restored, np=np)[0] == schema['canonical_sha256']
                 and all(restored[name].tobytes() == values[name].tobytes() for name in values),
                 'Durable complete state differs from actual captured values')
            last_full_state[arm] = store.json(folder + '/state.json', {
                'complete': True, 'arm': arm, 'update': committed[arm], 'file': binding,
                'canonical_sha256': schema['canonical_sha256'], 'arrays': schema['arrays'],
                'model_qualified': False}, emergency=emergency)
            return values

        with bounded_phase('original_complete575_baseline'):
            frozen_baseline = full_state('A', 'original_u64')
            need(helper.canonical_arrays(frozen_baseline, np=np)[0] == model.FULL_STATE_CANONICAL_SHA256,
                 'Exact original complete575 baseline required')
            assert_frozen('A'); assert_frozen('B')
        controls = {row['id']: row for row in p['controls']}
        parity = {row['id']: row for row in p['u64_parity']}
        need(len(p['controls']) == len(controls) == 56
             and set(controls) == {row['id'] for row in selection['old_quality']}
             and len(p['u64_parity']) == len(parity) == 4
             and set(parity) == {row['id'] for row in selection['old_monitors'][:4]},
             'Exact56 U96 controls and four historical RAW/PNG witnesses required')
        frozen_records, frozen_pixels, teacher_pixels = {}, {}, {}
        with bounded_phase('saved_U96_controls_recomputed_without_neural_calls'):
            for source in selection['old_quality']:
                row = {**source, 'bank': 'old_quality', 'case_key': 'old_quality:' + source['id']}
                control = controls[row['id']]
                raw, png = load_raw(control['raw']), rgb(control['png'])
                need(np.array_equal((np.clip(raw, 0, 1) * 255).astype(np.uint8), png), 'Cached U96 floorPNG differs from RAW')
                frozen_pixels[row['id']] = (raw, png)
                target = None if row.get('target') is None else rgb(row['target']).astype(np.float32) / np.float32(255)
                teacher = row.get('behavior_teacher')
                kwargs = {}
                if teacher is not None:
                    pair = (load_raw(teacher['raw']), rgb(teacher['png']))
                    teacher_pixels[row['id']] = pair
                    kwargs = {'teacher_raw': pair[0], 'teacher_png': pair[1]}
                frozen_records[row['id']] = records.record_inherited(raw, png, target, row,
                    np=np, frozen_U96_raw=raw, frozen_U96_png=png, **kwargs)
                saved = packet.strict_json(checked(control['record']).read_bytes())
                need(saved['id'] == row['id'] and saved['kind'] == 'reference' and saved['update'] == 0
                     and saved['role'] == row['role'] and saved['split'] == 'TRAIN'
                     and saved['metrics'] == frozen_records[row['id']]['metrics']
                     and saved['regional_metrics'] == frozen_records[row['id']]['regional_metrics'],
                     'Cached U96 freshly measured metrics differ from authenticated original record')
            store.json('frozen_U96_control_metrics.json', {'complete': True,
                'records': frozen_records, 'model_qualified': False})

        def input_values(row):
            image = rgb(row['input'])
            expected = row.get('prepared_input_RGB8_sha256', row.get('input_RGB8_sha256'))
            need(hashlib.sha256(image.tobytes()).hexdigest() == expected, 'Bound admitted input RGB differs')
            sample = torch.from_numpy(image.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)[None] / 255.0
            sample = F.interpolate(sample, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            target_array = None if row.get('target') is None else rgb(row['target'])
            if target_array is not None:
                expected = row.get('prepared_target_RGB8_sha256', row.get('target_RGB8_sha256'))
                need(hashlib.sha256(target_array.tobytes()).hexdigest() == expected, 'Bound photographic target RGB differs')
                target_array = target_array.astype(np.float32) / np.float32(255)
            target = None if target_array is None else torch.from_numpy(target_array.transpose(2, 0, 1).copy()).to(device)[None]
            return sample, target, target_array

        def full_call(sample, call):
            nonlocal active_case
            active_case = call['case_key']
            need(ledger.next_forward() == call and torch.are_deterministic_algorithms_enabled()
                 and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Exact next strict deterministic forward required')
            resource('before_forward')
            token = ledger.reserve_forward(call)
            status('forward_reserved')
            with torch.autocast(device_type='cuda', enabled=False):
                result = candidates[call['arm']](sample)
            torch.cuda.synchronize(0)
            ledger.returned(token)
            status('forward_returned')
            valid = isinstance(result, tuple) and len(result) == 3 and all(
                isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and value.device == device and tuple(value.shape) == shape and torch.isfinite(value).all().item()
                for value, shape in zip(result, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))))
            if not valid:
                retained = []
                for index, value in enumerate(result if isinstance(result, (tuple, list)) else (result,)):
                    if isinstance(value, torch.Tensor):
                        bits = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
                        retained.append({'index': index, 'shape': list(value.shape), 'dtype': str(value.dtype),
                            'bits': save_array('invalid_returns/' + str(call['slot']) + '/' + str(index) + '.npy', bits)})
                store.json('invalid_returns/' + str(call['slot']) + '/record.json', {
                    'accepted_as_valid': False, 'values': retained, 'model_qualified': False})
                raise ValueError('Invalid actual forward return retained; no retry')
            candidate = candidates[call['arm']]
            need(candidate.last_base_calls == 1 and tuple(candidate.last_forward_sites) == tuple(model.SITES),
                 'One actual owned four-site forward required')
            resource('after_forward')
            return result

        def saved_record(row, triple, call, target_array):
            stem = 'calls/' + str(call['slot']).zfill(4)
            normalized = (triple[0] + 1) / 2
            unclipped = F.interpolate(normalized, (256, 256), mode='bilinear', align_corners=False)
            raw = F.interpolate(normalized.clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
            value = raw.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
            png = (np.clip(value, 0, 1) * 255).astype(np.uint8)
            stream = io.BytesIO(); Image.fromarray(png).save(stream, format='PNG')
            need(len(stream.getvalue()) <= 256 * 1024, 'Finite per-image PNG cap')
            outputs = {'raw': save_array(stem + '/raw.npy', value),
                'unclipped': save_array(stem + '/unclipped.npy', unclipped),
                'png': store.write(stem + '/result.png', stream.getvalue())}
            need(np.array_equal(rgb(outputs['png']), png), 'Actual durable PNG RGB readback differs')
            stock = {name: tensor_summary(tensor) for name, tensor in zip(('native512', 'logits', 'LQ'), triple)}
            stock_files = {}
            if (call['bank'] == 'old_quality' and row['id'] in parity
                    and call['phase'] in ('initial_quality', 'same_state_deterministic_repeat')):
                stock_files = {name: save_array(stem + '/' + name + '.npy', tensor)
                    for name, tensor in zip(('native512', 'logits', 'LQ'), triple)}
            if call['bank'] == 'old_quality':
                frozen_raw, frozen_png = frozen_pixels[row['id']]
                kwargs = {}
                if row['id'] in teacher_pixels:
                    pair = teacher_pixels[row['id']]
                    kwargs = {'teacher_raw': pair[0], 'teacher_png': pair[1]}
                result = records.record_inherited(value, png, target_array, row, np=np,
                    frozen_U96_raw=frozen_raw, frozen_U96_png=frozen_png, **kwargs)
            else:
                result = records.record_new(value, png, target_array, row, np=np)
            result.update({'call': call, 'update': call['new_update'], 'outputs': outputs,
                'stock': stock, 'stock_files': stock_files,
                'RAW_pixel_bytes_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                'PNG_RGB8_sha256': hashlib.sha256(png.tobytes()).hexdigest(),
                'worker_input_target_file_and_RGB_verified': True,
                'worker_saved_RAW_PNG_readback_exact': True,
                'worker_RAW_floor_PNG_delivery_exact': True,
                'graph_witness': copy.deepcopy(candidates[call['arm']].last_graph_witness),
                'historical_stock_native_parity_claimed': False, 'model_qualified': False})
            result['receipt'] = store.json(stem + '/record.json', result)
            return result, raw

        def parity_equal(current, reference):
            return (current['stock'] == reference['stock']
                and current['RAW_pixel_bytes_sha256'] == reference['RAW_pixel_bytes_sha256']
                and current['PNG_RGB8_sha256'] == reference['PNG_RGB8_sha256']
                and current['outputs']['unclipped']['sha256'] == reference['outputs']['unclipped']['sha256']
                and current['input']['sha256'] == reference['input']['sha256']
                and current['target'] == reference['target'])

        def bank_gate(arm, update, bank):
            inherited = quality.evaluate_inherited(bank['old_quality'], initial_records[arm]['old_quality'],
                frozen_records, parent_raw, audit_raw, update=update)
            supplemental = quality.evaluate_new_bank(bank['new_monitors'], initial_records[arm]['new_monitors'],
                selection, update=update)
            gate = {'pass': inherited['pass'] and supplemental['pass'], 'update': update,
                'inherited': inherited, 'supplemental': supplemental,
                'same_original_references_and_zero_added_quality_tolerance': True,
                'model_qualified': False}
            receipt = store.json('arms/' + arm + '/banks/u' + str(update).zfill(4) + '/bank.json', {
                'complete': True, 'records': bank, 'gate': gate, 'model_qualified': False})
            return gate, receipt

        for arm in contract.ARMS:
            initial_records[arm] = {bank: {} for bank in ('old_quality', 'new_monitors', 'new_optimization')}
            calls = [call for call in plan if call['arm'] == arm and call['new_update'] == 0]
            with charge_arm(arm):
                for offset in range(0, len(calls), 24):
                    with bounded_phase('initializer_' + arm + '_chunk_' + str(offset // 24)):
                        for call in calls[offset:offset + 24]:
                            row = contract.resolve_case(selection, call)
                            row.update(bank=call['bank'], case_key=call['case_key'])
                            sample, target, target_array = input_values(row)
                            with torch.no_grad():
                                triple = full_call(sample, call)
                                record, raw = saved_record(row, triple, call, target_array)
                            if call['phase'] == 'same_state_deterministic_repeat':
                                baseline = initial_records[arm]['old_quality'][row['id']]
                                need(parity_equal(record, baseline), 'Actual same-arm deterministic repeat differs')
                            else:
                                initial_records[arm][call['bank']][row['id']] = record
                                if arm == 'B':
                                    need(parity_equal(record, initial_records['A'][call['bank']][row['id']]),
                                         'Actual A0/B0 native/logits/LQ/RAW/PNG values differ')
                                if call['bank'] == 'old_quality' and row['id'] in parity:
                                    witness = parity[row['id']]
                                    historical = packet.strict_json(checked(witness['record']).read_bytes())
                                    need(historical['id'] == row['id'] and historical['kind'] == 'snapshot'
                                         and historical['update'] == 64 and historical['role'] == row['role']
                                         and historical['split'] == 'TRAIN',
                                         'Exact historical terminal-u64 case witness record required')
                                    need(np.array_equal(load_raw(record['outputs']['raw']), load_raw(witness['raw']))
                                         and np.array_equal(rgb(record['outputs']['png']), rgb(witness['png'])),
                                         'Actual initializer differs from historical four terminal-u64 RAW/PNG witnesses')
                            del sample, target, target_array, triple, raw
                with bounded_phase('initializer_' + arm + '_complete_bank_gates'):
                    initial_gates[arm] = bank_gate(arm, 0, {key: initial_records[arm][key]
                        for key in ('old_quality', 'new_monitors')})
                    assert_frozen(arm)
        need(snapshot()['counters']['full_forward_returns'] == 440
             and snapshot()['counters']['optimizer_constructions'] == 0,
             'Both complete440 initializer/parity/repeat returns must precede any optimizer')
        store.json('initial_parity.json', {'complete': True,
            'actual_A0_B0_comparison_cases': 216, 'self_repeats_per_arm': 4,
            'historical_RAW_PNG_witnesses_per_arm': 4,
            'old56_complete_saved_scalar_join_per_arm': True,
            'saved_historical_stock_native_logits_LQ_parity_claimed': False,
            'initial_gates': {arm: value[0] for arm, value in initial_gates.items()},
            'model_qualified': False})

        def projection(label, *, measured_seconds, measured_calls, remaining_calls, extra_seconds=0.0):
            need(measured_calls > 0 and measured_seconds >= 0, 'Measured finite projection sample required')
            # A margin rejects work that cannot fit; it never grants an extension.
            projected = ((measured_seconds / measured_calls) * remaining_calls + extra_seconds) * 1.25
            now = time.monotonic()
            available = began + limits['worker_seconds'] - now
            if active_arm is not None:
                available = min(available, limits['per_arm_seconds'] - arm_used(active_arm, now))
            store.json('projections/' + label + '.json', {'complete': True,
                'measured_seconds': measured_seconds, 'measured_calls': measured_calls,
                'remaining_calls': remaining_calls, 'margin': 1.25,
                'additional_evaluation_seconds_before_margin': extra_seconds,
                'projected_additional_seconds': projected, 'available_seconds': available,
                'prediction_is_not_a_guarantee': True, 'deadline_extended': False,
                'pass': projected < available, 'model_qualified': False})
            need(projected < available, 'Conservative measured resource projection exceeds fixed deadline')

        initial_evaluation_seconds = sum(budget.arm_seconds.values())

        def tree_equal(left, right):
            if isinstance(left, torch.Tensor):
                return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
            if type(left) is not type(right): return False
            if type(left) is dict: return set(left) == set(right) and all(tree_equal(left[key], right[key]) for key in left)
            if isinstance(left, (tuple, list)): return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
            return left == right

        def checkpoint(arm, update):
            invariant = assert_frozen(arm)
            values = arrays(dict(trainable[arm]))
            payload, schema = helper.npz_state_payload(values, np=np)
            folder = 'arms/' + arm + '/checkpoints/u' + str(update).zfill(4)
            binding = store.write(folder + '/trainable.npz', payload)
            with np.load(checked(binding), allow_pickle=False) as saved:
                decoded = {name: saved[name] for name in saved.files}
            need(helper.canonical_arrays(decoded, np=np)[0] == schema['canonical_sha256']
                 and set(decoded) == set(values) and all(decoded[name].tobytes() == values[name].tobytes() for name in values),
                 'Durable actual36/60 trainable checkpoint differs')
            state = {'format': 'closed-matched-feature-state-v1', 'protocol_sha256': expected_sha,
                'arm': arm, 'update': update, 'automatic_resume_permitted': False,
                'optimizer': cpu_tree(optimizers[arm].state_dict()),
                'optimizer_parameter_names': [name for name, _ in trainable[arm]],
                'RNG': rng_record(), 'sampler_next_update': update + 1 if update < 48 else None,
                'sampler_next_plan_call': ledger.next_forward(),
                'initial_trainable_sha256': helper.canonical_arrays(initial_trainable[arm], np=np)[0],
                'frozen_invariants': invariant, 'execution': snapshot(), 'model_qualified': False}
            stream = io.BytesIO(); torch.save(state, stream)
            need(len(payload) + len(stream.getvalue()) <= 8 * 1024**2, 'Bounded trainable/Adam/RNG checkpoint8MiB required')
            resume = store.write(folder + '/closed_state.pt', stream.getvalue())
            restored = torch.load(checked(resume), map_location='cpu', weights_only=True)
            need(tree_equal(state, restored), 'Actual durable Adam/RNG/sampler readback differs')
            last_checkpoint[arm] = store.json(folder + '/checkpoint.json', {'complete': True,
                'arm': arm, 'update': update, 'trainable': binding, 'trainable_canonical_sha256': schema['canonical_sha256'],
                'closed_state': resume, 'frozen_invariants': invariant, 'model_qualified': False})
            status('arm_checkpoint_committed')

        @contextmanager
        def backward_warning_scope():
            need(torch.are_deterministic_algorithms_enabled()
                 and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Strict deterministic flags required before backward')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    yield
                finally:
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    rows = [{'category': warning.category.__name__, 'message': str(warning.message)[:12000]}
                            for warning in caught]
                    unexpected = [row for row in rows if ('deterministic' in row['message'].lower()
                        or 'nondeterminism' in row['message'].lower()) and re.match(
                            r'^upsample_bilinear2d_backward_out_cuda\b', row['message']) is None]
                    event({'kind': 'backward_warning_scope', 'warnings': rows, 'unexpected': unexpected,
                        'strict_flags_restored': True, 'bitwise_CUDA_gradient_repeatability_claimed': False})
                    need(not unexpected, 'Unexpected nondeterministic backward; fatal stop')

        def gradient_arrays(arm, *, allow_missing=False):
            result = {}
            for name, value in trainable[arm]:
                grad = value.grad
                if grad is None and allow_missing:
                    result[name] = np.zeros(tuple(value.shape), dtype=np.float32)
                    continue
                need(grad is not None and grad.dtype == torch.float32 and grad.device == device
                     and grad.shape == value.shape and torch.isfinite(grad).all().item(),
                     'Every declared trainable gradient must exist with finite exact FP32 schema')
                result[name] = grad.detach().cpu().contiguous().numpy().copy()
            return result

        def site_norms(values, arm):
            result = {}
            families = ('heads', 'adapters') if arm == 'B' else ('heads',)
            for family in families:
                for site in model.SITES:
                    prefix = family + '.' + site + '.'
                    selected = [value for name, value in values.items() if name.startswith(prefix)]
                    need(len(selected) == (9 if family == 'heads' else 6), 'Exact per-site gradient tensor scope required')
                    squared = sum(float(np.square(value.astype(np.float64)).sum()) for value in selected)
                    need(math.isfinite(squared), 'Finite per-site gradient norm required')
                    result[family + '/' + site] = math.sqrt(squared)
            return result

        def graph_guard(arm):
            witness = copy.deepcopy(candidates[arm].last_graph_witness)
            need(witness.get('status') == 'WITNESSED' and witness.get('complete') is True
                 and witness.get('grad_enabled') is True and witness.get('arm') == arm
                 and set(witness['sites']) == set(model.SITES), 'Actual current four-site autograd structure witness required')
            for site in model.SITES:
                branches = witness['sites'][site]
                need(set(branches) == {'stock_SFT', 'head_encoder'}, 'Both existing adapter routes must be witnessed')
                for branch in branches.values():
                    need(branch.get('pass') is True and branch.get('expected_present') is (arm == 'B')
                         and branch.get('current_adapter_leaves_reached') == (6 if arm == 'B' else 0)
                         and branch.get('autograd_queries_or_backward_calls') == 0,
                         'Actual arm-specific adapter reachability differs')
            return witness

        def close_quality(arm, update, gate, receipt):
            reason = 'Complete checkpoint' + str(update) + ' inherited56 or supplemental40 quality failed'
            ledger.quality_stop(arm, reason, receipt_sha256=receipt['sha256'])
            arm_stops[arm] = {'update': update, 'gate_receipt': receipt, 'reason': reason}
            arm_results[arm] = {'complete': False, 'quality_stop': True, **arm_stops[arm], 'model_qualified': False}
            event({'kind': 'arm_quality_stop', 'gate_receipt': receipt, 'update': update,
                'other_arm_precommitted_work_only': True})
            label = 'arm_' + arm + '_quality_failed_u' + str(update).zfill(4)
            if active_arm is None:
                with charge_arm(arm), bounded_phase('quality_failed_arm_' + arm + '_actual_state'):
                    full_state(arm, label)
            else:
                with bounded_phase('quality_failed_arm_' + arm + '_actual_state'):
                    full_state(arm, label)

        for arm in contract.ARMS:
            gate, receipt = initial_gates[arm]
            if not gate['pass']:
                close_quality(arm, 0, gate, receipt)

        with bounded_phase('initial_measured_projection'):
            state = snapshot()
            remaining_calls = len(plan) - len(state['returned_forward_slots']) - len(state['skipped_forward_slots'])
            projection('after440', measured_seconds=initial_evaluation_seconds,
                       measured_calls=440, remaining_calls=remaining_calls)

        for arm in contract.ARMS:
            if arm in arm_stops:
                continue
            active_update = 0
            with charge_arm(arm):
                with bounded_phase('fresh_Adam_' + arm):
                    need(ledger.next_forward()['arm'] == arm, 'Fresh optimizer must follow precommitted arm order')
                    token = ledger.reserve_optimizer_construction(arm)
                    status('optimizer_construction_reserved')
                    options = contract.OPTIMIZER
                    kwargs = {key: options[key] for key in ('eps', 'weight_decay', 'foreach', 'fused',
                        'capturable', 'differentiable', 'amsgrad', 'maximize')}
                    optimizers[arm] = torch.optim.Adam([value for _, value in trainable[arm]],
                        lr=options['maximum_learning_rate'], betas=tuple(options['betas']), **kwargs)
                    ledger.returned(token)
                    need(not optimizers[arm].state, 'Fresh empty per-arm Adam moments required')
                    checkpoint(arm, 0)
                for update in range(1, 49):
                    active_update = update
                    update_began = time.monotonic()
                    with bounded_phase('arm_' + arm + '_update_' + str(update)):
                        optimizer = optimizers[arm]
                        optimizer.zero_grad(set_to_none=True)
                        for group in optimizer.param_groups: group['lr'] = contract.learning_rate(update)
                        micro_receipts = []
                        previous_gradients = gradient_arrays(arm, allow_missing=True) if update == 1 else None
                        for role in objective.ROLES:
                            call = ledger.next_forward()
                            need(call is not None and call['arm'] == arm and call['new_update'] == update
                                 and call['backward'] and call['bank'] == 'new_optimization', 'Exact scheduled five-role update required')
                            row = contract.resolve_case(selection, call)
                            row.update(bank=call['bank'], case_key=call['case_key'])
                            need(row['role'] == role and row['cohort'] == 'optimize'
                                 and row['optimization_permitted'] is True, 'Only one admitted photograph in fixed role order required')
                            sample, target, target_array = input_values(row)
                            with torch.enable_grad(), torch.autocast(device_type='cuda', enabled=False):
                                triple = full_call(sample, call)
                                record, raw = saved_record(row, triple, call, target_array)
                                witness = graph_guard(arm)
                                value, terms = objective.connected_objective(raw, target, role, torch=torch)
                                raw_array = raw.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
                                reference, reference_terms = objective.objective_numpy(raw_array, target_array, role, np=np)
                                actual, actual_terms = float(value.detach().item()), {key: float(term.detach().item()) for key, term in terms.items()}
                                need(arithmetic_close(actual, reference) and all(arithmetic_close(
                                    actual_terms[key], reference_terms[key]) for key in objective.TERM_NAMES),
                                    'Connected FP32 objective differs from independent NumPy reference')
                                evidence = {'complete': True, 'call': call, 'arm': arm, 'update': update,
                                    'objective': actual, 'divided_objective': float((value / 5).detach().item()),
                                    'divisor_applied_once': 5, 'terms': actual_terms, 'NumPy_objective': reference,
                                    'NumPy_terms': reference_terms, 'arithmetic_only_atol': ARITHMETIC_ATOL,
                                    'arithmetic_only_rtol': ARITHMETIC_RTOL, 'quality_tolerance_added': False,
                                    'output_record': record['receipt'], 'graph_witness': witness,
                                    'LR': contract.learning_rate(update), 'model_qualified': False}
                                micro_receipts.append(store.json('calls/' + str(call['slot']).zfill(4) + '/objective.json', evidence))
                                token = ledger.reserve_backward(call['slot'])
                                status('backward_reserved')
                                with backward_warning_scope():
                                    (value / 5).backward()
                                    torch.cuda.synchronize(0)
                                    ledger.returned(token)
                                status('backward_returned')
                            need(all(parameter.grad is None for name, parameter in named[arm].items()
                                     if not parameter.requires_grad), 'Frozen arm gradients must stay None')
                            if update == 1:
                                gradients = gradient_arrays(arm, allow_missing=True)
                                difference = {name: gradients[name] - previous_gradients[name] for name in gradients}
                                norms = site_norms(difference, arm)
                                store.json('calls/' + str(call['slot']).zfill(4) + '/first_update_role_gradient_norms.json', {
                                    'complete': True, 'arm': arm, 'role': role, 'difference_of_accumulated_gradients': True,
                                    'site_norms': norms, 'zero_individual_role_site_allowed': True,
                                    'extra_autograd_or_backward_queries': 0, 'model_qualified': False})
                                previous_gradients = gradients
                            del sample, target, target_array, triple, raw, raw_array, value, terms, record
                        gradients = gradient_arrays(arm)
                        norms = site_norms(gradients, arm)
                        need(all(norm > 0 for norm in norms.values()) if update == 1
                             else any(norm > 0 for norm in norms.values()),
                             'First update needs positive norms per trainable site; later all-zero updates refused')
                        prefix = 'arms/' + arm + '/updates/u' + str(update).zfill(4)
                        before = arrays(dict(trainable[arm]))
                        pre_payload, _ = helper.npz_state_payload(gradients, np=np)
                        preclip = store.write(prefix + '/preclip_gradients.npz', pre_payload)
                        norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable[arm]],
                            max_norm=contract.OPTIMIZER['gradient_clip_max_norm'], error_if_nonfinite=True)
                        post_payload, _ = helper.npz_state_payload(gradient_arrays(arm), np=np)
                        need(len(pre_payload) + len(post_payload) <= 4 * 1024**2, 'Finite4MiB trainable gradient bundle required')
                        postclip = store.write(prefix + '/postclip_gradients.npz', post_payload)
                        if update == 1:
                            projection('first5_' + arm, measured_seconds=time.monotonic() - update_began,
                                measured_calls=5, remaining_calls=5 * 47,
                                extra_seconds=initial_evaluation_seconds / 440 * (3 * 96 + 120))
                        token = ledger.reserve_optimizer_update(arm, update)
                        status('optimizer_update_reserved')
                        optimizer.step()
                        torch.cuda.synchronize(0)
                        ledger.returned(token)
                        committed[arm] = update
                        status('optimizer_update_returned')
                        after = arrays(dict(trainable[arm]))
                        need(all(np.isfinite(value).all() for value in after.values()), 'Finite actual updated trainable parameters required')
                        scope = [name for name, _ in trainable[arm]]
                        displacement = contract.displacement_checks(
                            [float(item) for name in scope for item in before[name].reshape(-1)],
                            [float(item) for name in scope for item in after[name].reshape(-1)],
                            [float(item) for name in scope for item in initial_trainable[arm][name].reshape(-1)])
                        # Retain the actual post-step values before testing hard displacement bounds.
                        after_payload, after_schema = helper.npz_state_payload(after, np=np)
                        after_binding = store.write(prefix + '/actual_after_step.npz', after_payload)
                        store.json(prefix + '/commit.json', {'complete': True, 'arm': arm, 'update': update,
                            'LR': contract.learning_rate(update), 'microbatches': micro_receipts,
                            'preclip_gradients': preclip, 'postclip_gradients': postclip,
                            'per_site_preclip_norms': norms, 'preclip_norm': float(norm.detach().item()),
                            'before_canonical_sha256': helper.canonical_arrays(before, np=np)[0],
                            'actual_after_step': after_binding, 'after_canonical_sha256': after_schema['canonical_sha256'],
                            'actual_displacement': displacement, 'projection_or_rollback_performed': False,
                            'model_qualified': False})
                        need(displacement['pass'], 'Actual post-Adam step/cumulative Linf cap failed; preserve actual state without projection')
                        optimizer.zero_grad(set_to_none=True)
                        checkpoint(arm, update)
                        print(json.dumps({'arm': arm, 'update': update, 'maximum': 48,
                            'LR': contract.learning_rate(update), 'model_qualified': False}), flush=True)
                    if update in contract.SNAPSHOTS[1:]:
                        bank = {'old_quality': {}, 'new_monitors': {}}
                        with bounded_phase('arm_' + arm + '_quality_u' + str(update)):
                            for _ in range(96):
                                call = ledger.next_forward()
                                need(call is not None and call['arm'] == arm and call['phase'] == 'quality_snapshot'
                                     and call['new_update'] == update, 'Exact complete96 snapshot calls required')
                                row = contract.resolve_case(selection, call)
                                row.update(bank=call['bank'], case_key=call['case_key'])
                                sample, target, target_array = input_values(row)
                                with torch.no_grad():
                                    triple = full_call(sample, call)
                                    record, raw = saved_record(row, triple, call, target_array)
                                bank[call['bank']][row['id']] = record
                                del sample, target, target_array, triple, raw
                            gate, receipt = bank_gate(arm, update, bank)
                            assert_frozen(arm)
                            print(json.dumps({'arm': arm, 'snapshot': update, 'pass': gate['pass']}), flush=True)
                        if not gate['pass']:
                            close_quality(arm, update, gate, receipt)
                            break
                if arm in arm_stops:
                    continue
                terminal = {}
                for offset in range(0, 120, 24):
                    with bounded_phase('arm_' + arm + '_terminal_optimization_chunk_' + str(offset // 24)):
                        for _ in range(24):
                            call = ledger.next_forward()
                            need(call is not None and call['arm'] == arm and call['phase'] == 'terminal_optimization_readout',
                                 'Exact terminal optimization-only readout required')
                            row = contract.resolve_case(selection, call)
                            row.update(bank=call['bank'], case_key=call['case_key'])
                            sample, target, target_array = input_values(row)
                            with torch.no_grad():
                                triple = full_call(sample, call)
                                record, raw = saved_record(row, triple, call, target_array)
                            terminal[row['id']] = record
                            del sample, target, target_array, triple, raw
                with bounded_phase('arm_' + arm + '_terminal_full_state'):
                    full_state(arm, 'arm_' + arm + '_terminal_u0048')
                    assert_frozen(arm)
                    arm_results[arm] = {'complete': True, 'quality_stop': False, 'updates': 48,
                        'terminal_optimization_records': store.json('arms/' + arm + '/terminal_optimization.json', {
                            'complete': True, 'initial': initial_records[arm]['new_optimization'],
                            'terminal': terminal, 'optimization_set_not_independent_holdout': True,
                            'model_qualified': False}), 'model_qualified': False}
        active_case = None
        with bounded_phase('terminal_full_journal_and_namespace_audit'):
            need(rng_record() == run_rng, 'Fixed operations changed seeded caller RNG')
            for arm in contract.ARMS: assert_frozen(arm)
            ledger.finish()
            replay = execution.audit_journal(checked(packet.pin(root, root / contract.OUTPUT / 'operations.jsonl')), selection)
            need(replay['bookkeeping_complete'] is True and replay['counters'] == snapshot()['counters'],
                 'Independent exact journal replay differs from actual ledger')
            store.json('journal_replay.json', replay)
            store.audit_namespace()
            proposed_exit = 2 if arm_stops else 0
            result = {'complete': True, 'status': 'FINITE_MATCHED_TRAIN_RESULT_NOT_QUALIFIED',
                'protocol_sha256': expected_sha, 'arms': arm_results,
                'quality_stop': bool(arm_stops), 'quality_stops': arm_stops,
                'execution': snapshot(), 'counters': snapshot()['counters'],
                'last_checkpoint': last_checkpoint, 'last_complete_state': last_full_state,
                'independent_DEV_native_FINAL_or_broader_failure_clearance': False,
                'worker_exit': proposed_exit, 'model_qualified': False}
            store.json('result.json', result)
            event({'kind': 'finite_result_saved', 'worker_exit': proposed_exit})
            status('finite_comparison_complete', complete=True)
    except BaseException as error:
        fatal_error = error
        proposed_exit = 1
        if store is not None:
            fail = {'complete': False, 'fatal': True, 'quality_stop': False,
                'protocol_sha256': expected_sha, 'phase': failed_phase or phase, 'active_arm': active_arm or failed_arm,
                'active_update': active_update, 'active_case': active_case,
                'error_type': type(error).__name__, 'error': str(error)[:4000],
                'traceback': traceback.format_exc()[-20000:], 'execution': snapshot(),
                'last_checkpoint': last_checkpoint, 'last_complete_state': last_full_state,
                'model_qualified': False}
            try:
                if ledger is not None and snapshot()['status'] == 'OPEN':
                    ledger.fatal_stop((type(error).__name__ + ': ' + str(error))[:512])
                fail['execution'] = snapshot()
            except BaseException as journal_error:
                fail['fatal_journal_error'] = repr(journal_error)[:2000]
            try:
                now = time.monotonic()
                allowance = budget.deadline(now, phase_started_at=now, mode='failure_capture')
                if candidates and torch is not None and allowance['remaining_seconds'] > 0:
                    with timed(allowance['remaining_seconds'], 'best_effort_actual_failure_state'):
                        failure_order = ([failed_arm] if failed_arm in candidates else []) + [
                            arm for arm in contract.ARMS if arm != failed_arm]
                        for arm in failure_order:
                            if arm not in candidates:
                                continue
                            already_closed = arm_results.get(arm, {}).get('complete') is True or arm in arm_stops
                            if full_state_count < 3 and not already_closed:
                                try:
                                    full_state(arm, 'arm_' + arm + '_fatal_u' + str(committed[arm]).zfill(4), emergency=True)
                                except TimeoutError:
                                    raise
                                except BaseException as state_error:
                                    fail.setdefault('validated_capture_errors', {})[arm] = repr(state_error)[:2000]
                                    if full_state_count < 3:
                                        values = arrays(candidates[arm].state_dict())
                                        stream = io.BytesIO(); np.savez(stream, **values)
                                        payload = stream.getvalue()
                                        need(len(payload) <= 384 * 1024**2, 'Unvalidated actual575 capture cap')
                                        full_state_count += 1
                                        fail.setdefault('unvalidated_current575', {})[arm] = store.write(
                                            'failure_values/' + arm + '/current575.npz', payload, emergency=True)
                                        fail['unvalidated_current575_accepted_as_valid'] = False
                            stream = io.BytesIO()
                            torch.save({'arm': arm, 'update': committed[arm],
                                'trainable': cpu_tree(dict(trainable.get(arm, []))),
                                'gradients': cpu_tree({name: value.grad for name, value in trainable.get(arm, [])}),
                                'optimizer': None if arm not in optimizers else cpu_tree(optimizers[arm].state_dict()),
                                'RNG': rng_record(), 'execution': snapshot(), 'accepted_as_valid': False}, stream)
                            need(len(stream.getvalue()) <= 12 * 1024**2, 'Actual partial optimizer/gradient evidence cap')
                            fail.setdefault('partial_trainable_Adam_RNG', {})[arm] = store.write(
                                'failure_values/' + arm + '/partial_optimizer.pt', stream.getvalue(), emergency=True)
                fail['partial_files'] = sorted(store.partial_files)
                fail['last_complete_state'] = copy.deepcopy(last_full_state)
            except BaseException as capture_error:
                fail['extra_state_capture_error'] = repr(capture_error)[:2000]
            try:
                now = time.monotonic()
                allowance = budget.deadline(now, phase_started_at=now, mode='failure_receipt')
                with timed(allowance['remaining_seconds'], 'finite_failure_receipt'):
                    store.json('failure.json', fail, emergency=True)
                    event({'kind': 'fatal_failure_saved', 'execution': snapshot()})
                    status('fatal_retained_stop')
            except BaseException as receipt_error:
                print('Failure receipt error: ' + repr(receipt_error), flush=True)
        traceback.print_exc()
    finally:
        if torch is not None and caller_rng is not None:
            try:
                seconds = min(5.0, began + limits['external_seconds'] - time.monotonic())
                with timed(seconds, 'caller_RNG_restore'):
                    random.setstate(caller_rng['python']); np.random.set_state(caller_rng['numpy'])
                    torch.set_rng_state(caller_rng['CPU']); torch.cuda.set_rng_state_all(caller_rng['CUDA'])
                    current_np = np.random.get_state(); original_np = caller_rng['numpy']
                    need(random.getstate() == caller_rng['python'] and current_np[0] == original_np[0]
                         and np.array_equal(current_np[1], original_np[1]) and current_np[2:] == original_np[2:]
                         and torch.equal(torch.get_rng_state().cpu(), caller_rng['CPU'])
                         and all(torch.equal(a.cpu(), b) for a, b in zip(torch.cuda.get_rng_state_all(), caller_rng['CUDA'])),
                         'Actual Python/NumPy/CPU/CUDA caller RNG restore differs')
            except BaseException as restore_error:
                proposed_exit = 1
                print('Bounded caller RNG restore failed: ' + repr(restore_error), flush=True)
                if store is not None:
                    try:
                        store.json('rng_finalization_failure.json', {'complete': False,
                            'error': repr(restore_error)[:2000], 'prior_result_invalidated': True,
                            'model_qualified': False}, emergency=True)
                    except BaseException:
                        pass
        if old_term is not None: signal.signal(signal.SIGTERM, old_term)
        if ledger is not None: ledger.close()
    return proposed_exit
