"""Finite manual-L4 u64 warm-start worker; import is standard-library only.

No model, framework, optimizer, gradient or VM operation occurs on import. The
explicit run entry requires its sealed packet, tmux, exact retained L4 stage and
the finite supervisor's real parent PID before importing Torch.
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
import subprocess
import sys
import time
import traceback
import warnings

OUTPUT = 'outputs/u64_cleananchors_continuation_v1'
PACKAGE = 'research/codeformer_u64_cleananchors_continuation_v1'
HELPER = 'research/codeformer_u96_spatial_feasibility_v1/worker.py'
HELPER_PIN = (72086, 'fae271100642604ae0682ebf52ec163e768fa6882eced87e0c34e587980126b6')
CATEGORIES = {'initial_quality_bank_forwards': 56, 'deterministic_repeat_forwards': 4,
              'candidate_snapshot_forwards_after_initializer': 168,
              'candidate_training_forwards': 64}
COUNTER_KEYS = ('full_forward_attempts', 'full_forward_returns', 'backward_attempts', 'backward_returns',
                'optimizer_constructions', 'optimizer_construction_returns',
                'optimizer_update_attempts', 'optimizer_updates', *CATEGORIES)
OPERATIONS = {'forward': ('full_forward_attempts', 'full_forward_returns', 292),
              'backward': ('backward_attempts', 'backward_returns', 64),
              'optimizer_construction': ('optimizer_constructions', 'optimizer_construction_returns', 1),
              'optimizer_update': ('optimizer_update_attempts', 'optimizer_updates', 16)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


class QualityStop(ValueError):
    pass


class OperationLedger:
    """Durable reservation precedes entry; durable return precedes validation."""
    def __init__(self, persist):
        self.persist = persist
        self.counters = dict.fromkeys(COUNTER_KEYS, 0)
        self.pending = {}
        self.sequence = 0

    def reserve(self, kind, **metadata):
        need(kind in OPERATIONS and not set(metadata) & {'kind', 'counters', 'token'}, 'Known operation with nonreserved metadata required')
        attempt, _, cap = OPERATIONS[kind]
        need(self.counters[attempt] < cap, 'Finite operation cap reached')
        if kind == 'forward':
            category = metadata.get('category')
            need(category in CATEGORIES and self.counters[category] < CATEGORIES[category], 'Declared finite forward category required')
            self.counters[category] += 1
        self.counters[attempt] += 1
        self.sequence += 1
        self.pending[self.sequence] = {'kind': kind, **metadata}
        self.persist({'kind': kind + '_reserved', 'token': self.sequence, 'counters': dict(self.counters), **metadata})
        return self.sequence

    def returned(self, token, **metadata):
        need(type(token) is int and token in self.pending and not set(metadata) & {'kind', 'counters', 'token'}, 'One unmatched reserved operation required')
        pending = self.pending[token]
        _, returned, _ = OPERATIONS[pending['kind']]
        counters = dict(self.counters)
        counters[returned] += 1
        self.persist({**pending, **metadata, 'kind': pending['kind'] + '_returned', 'token': token, 'counters': counters})
        self.counters = counters
        del self.pending[token]

    def snapshot(self):
        return {'counters': dict(self.counters),
                'unknown_reservations': [{'token': token, **row} for token, row in sorted(self.pending.items())],
                'operation_outcome_unknown': bool(self.pending)}


def supervisor_context(sha, *, environ, parent_pid):
    need(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha)
         and environ.get('FORENSIC_U64_CLEANANCHORS_SUPERVISED') == '1'
         and environ.get('FORENSIC_U64_CLEANANCHORS_PROTOCOL') == sha
         and environ.get('FORENSIC_U64_CLEANANCHORS_SUPERVISOR_PID') == str(parent_pid),
         'Exact supplied seal and actual finite supervisor parent required')


def learning_rate(update):
    need(type(update) is int and 1 <= update <= 16, 'New finite update1–16 required')
    return 0.0001 * min(1.0, update / 4.0)


def run(root, protocol, expected_sha):
    from . import packet, contract
    root = Path(root).absolute()
    began = time.monotonic()
    store = ledger = candidate = optimizer = torch = np = None
    phase = 'static_guards'
    failed_phase = None
    active_update = 0
    active_case = None
    caller_rng = None
    last_checkpoint = last_complete_state = None
    frozen_initial = None
    full_states = 0
    old_term = None
    resources_ready = False

    def event(value):
        store.append('events.jsonl', {'UTC': datetime.now(timezone.utc).isoformat(),
                     'elapsed_seconds': time.monotonic()-began, 'phase': phase, **value})

    def status(label, *, complete=False, quality_stop=False):
        if store is not None:
            snapshot = ledger.snapshot() if ledger else {'counters': dict.fromkeys(COUNTER_KEYS, 0),
                         'unknown_reservations': [], 'operation_outcome_unknown': False}
            store.status({'complete': complete, 'protocol_sha256': expected_sha,
                'status': 'FINITE_U64_CONTINUATION_NOT_QUALIFIED', 'label': label, 'phase': failed_phase or phase,
                'active_update': active_update, 'active_case': active_case,
                'elapsed_seconds': time.monotonic()-began, 'last_checkpoint': last_checkpoint,
                'last_complete_state': last_complete_state, 'quality_stop': quality_stop,
                'model_qualified': False, **snapshot})

    def checked(binding):
        need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact file binding required')
        path = packet.regular(root / binding['path'])
        need(packet.pin(root, path) == binding, 'File differs from its exact binding')
        return path

    def resource(label):
        elapsed = time.monotonic()-began
        free = shutil.disk_usage(root).free
        rss = helper.linux_current_rss_bytes(Path('/proc/self/statm').read_text(), int(os.sysconf('SC_PAGE_SIZE')))
        peak = int(linux_resource.getrusage(linux_resource.RUSAGE_SELF).ru_maxrss)*1024
        cuda = peak_cuda = 0
        if torch is not None and torch.cuda.is_initialized():
            cuda = int(torch.cuda.memory_reserved(0))
            peak_cuda = int(torch.cuda.max_memory_reserved(0))
        row = {'label': label, 'elapsed_seconds': elapsed, 'free_disk_bytes': free,
               'current_RSS_bytes': rss, 'peak_RSS_bytes': peak,
               'CUDA_reserved_bytes': cuda, 'CUDA_peak_reserved_bytes': peak_cuda,
               'output_bytes': store.bytes if store else 0}
        if resources_ready:
            store.append('resources.jsonl', row)
        limits = contract.LIMITS
        need(elapsed < limits['worker_seconds'], 'Finite1200s scientific stop')
        need(free >= limits['continuing_free_disk_bytes'], 'Continuing4GiB disk stop')
        need(max(rss, peak) <= limits['maximum_RSS_bytes'], '10GiB RSS stop')
        need(max(cuda, peak_cuda) <= limits['maximum_CUDA_reserved_bytes'], '18GiB CUDA reserved stop')
        need(store is None or store.bytes <= limits['maximum_output_bytes'], '2GiB output stop')

    @contextmanager
    def timed(seconds, name):
        need(seconds > 0 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Fresh positive bounded timer required')
        old = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError('Finite stop: '+name)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    @contextmanager
    def bounded_phase(name):
        nonlocal phase, failed_phase
        previous = phase
        phase = name
        start = time.monotonic()
        resource('before')
        seconds = min(180.0, 1200.0-(time.monotonic()-began))
        try:
            with timed(seconds, name):
                yield
                resource('after')
                event({'kind': 'phase_returned', 'phase_seconds': time.monotonic()-start, 'counters': dict(ledger.counters)})
        except BaseException:
            failed_phase = name
            raise
        finally:
            phase = previous

    try:
        p = packet.verify_static(root, expected_sha)
        need(p == protocol and p['continuation'] == contract.prospective_contract(), 'Exact sealed prospective contract required')
        need(Path(__file__).resolve() == root / PACKAGE / 'worker.py', 'Use own staged worker')
        facts = packet.require_vm(root)
        supervisor_context(expected_sha, environ=os.environ, parent_pid=os.getppid())
        need((root / OUTPUT).exists() is False, 'Prior complete or partial run immutable')
        helper_path = packet.regular(root / HELPER)
        need((helper_path.stat().st_size, packet.sha256(helper_path)) == HELPER_PIN, 'Exact reviewed inert storage helper required')
        from research.codeformer_u96_spatial_feasibility_v1 import worker as helper
        need(Path(helper.__file__).resolve() == helper_path, 'Active storage helper origin differs')
        import resource as linux_resource
        parent_bytes = checked(p['parent_protocol']).read_bytes()
        audit_bytes = checked(p['historical_pixel_audit']).read_bytes()
        selection = contract.select_parent_cases(parent_bytes)
        plan = contract.operation_plan(selection)
        rows = {r['id']: r for r in p['cases']+p['guards']}
        store = helper.DurableStore(root, root / OUTPUT, contract.LIMITS['maximum_output_bytes'], packet.pin)
        store.json('started.json', {'complete': False, 'protocol_sha256': expected_sha, 'host': facts, 'model_qualified': False})
        store.write('events.jsonl', b'')
        store.write('resources.jsonl', b'')
        ledger = OperationLedger(event)
        resources_ready = True
        old_term = signal.getsignal(signal.SIGTERM)
        def stopped(signum, frame):
            raise TimeoutError('External finite SIGTERM; preserve actual prefix and unknown outcomes')
        signal.signal(signal.SIGTERM, stopped)
        store.json('operation_plan.json', {'complete': True, 'calls': plan, 'selection': selection,
                                         'optimizer_maximum_new_updates': 16, 'model_qualified': False})
        event({'kind': 'static_VM_supervisor_guards_passed_before_Torch', 'Torch_imported': False, 'counters': dict(ledger.counters)})
        with bounded_phase('idle_GPU_before_framework'):
            probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                   text=True, capture_output=True, timeout=10, check=True)
            need(not probe.stdout.strip(), 'Competing GPU workload; do not construct')
        with bounded_phase('dependencies_and_FP32_policy'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            need(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8'), 'Deterministic cuBLAS config required')
            import torch as torch_module
            import numpy as numpy_module
            from PIL import Image
            from torch.nn import functional as F
            from . import model, objective, metrics
            torch, np = torch_module, numpy_module
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1
                 and torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one actual L4 required')
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
            need(not torch.is_inference_mode_enabled() and torch.get_default_dtype() == torch.float32, 'Ordinary FP32 autograd required')
            caller_rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
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
            for name, module in (('model', model), ('objective', objective), ('metrics', metrics), ('contract', contract)):
                relative = PACKAGE+'/'+name+'.py'
                need(Path(module.__file__).resolve() == root / relative
                     and packet.sha256(root / relative) == p['files'][relative]['sha256'], 'Loaded sealed component differs')

        def arrays(state):
            return {name: value.detach().cpu().contiguous().numpy().copy() for name, value in state.items()}

        def canonical(state):
            return helper.canonical_arrays(arrays(state), np=np)[0]

        def save_array(name, value):
            a = value.detach().cpu().contiguous().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value)
            need(a.dtype.kind in ('f', 'i', 'u') and np.isfinite(a).all(), 'Finite numerical array required')
            out = io.BytesIO()
            np.save(out, a, allow_pickle=False)
            return store.write(name, out.getvalue())

        def load_raw(binding):
            with checked(binding).open('rb') as stream:
                a = np.load(stream, allow_pickle=False)
            need(a.dtype == np.float32 and a.shape == (256, 256, 3) and np.isfinite(a).all()
                 and a.min() >= 0 and a.max() <= 1, 'Exact delivered FP32 HWC256 cache required')
            return a

        def rgb(binding):
            with Image.open(checked(binding)) as im:
                need(im.mode == 'RGB' and im.size == (256, 256) and getattr(im, 'n_frames', 1) == 1, 'Exact RGB256 file required')
                return np.asarray(im).copy()

        def tensor_record(tensor):
            a = tensor.detach().cpu().contiguous().numpy()
            return {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes_sha256': hashlib.sha256(a.tobytes()).hexdigest()}

        def rng_record():
            ns = np.random.get_state()
            return {'Python': repr(random.getstate()),
                    'NumPy': {'algorithm': ns[0], 'keys_hex': ns[1].tobytes().hex(), 'position': int(ns[2]),
                              'has_gauss': int(ns[3]), 'cached_gaussian': float(ns[4])},
                    'CPU_hex': torch.get_rng_state().cpu().numpy().tobytes().hex(),
                    'CUDA_hex': [v.cpu().numpy().tobytes().hex() for v in torch.cuda.get_rng_state_all()]}

        with bounded_phase('exact_u64_initialization_no_forward'):
            r2_path, u96_path = (checked(p['model'][k]) for k in ('r2', 'u96'))
            r2 = torch.load(r2_path, map_location='cpu', weights_only=True)
            u96 = torch.load(u96_path, map_location='cpu', weights_only=True)
            need(type(r2) is dict and set(r2) == {'params_ema'} and type(u96) is dict
                 and set(u96) == {'format', 'adapter_state'}
                 and u96['format'] == 'codeformer-spatial-adapter-inference-v1', 'Exact initializer formats required')
            need(canonical(r2['params_ema']) == model.R2_CANONICAL_SHA256
                 and canonical(u96['adapter_state']) == model.U96_CANONICAL_SHA256, 'Exact canonical frozen initializers required')
            from third_party.codeformer.codeformer_arch import CodeFormer
            for relative, digest in model.ARCHITECTURE_SOURCES.items():
                module = sys.modules.get(relative[:-3].replace('/', '.'))
                need(module is not None and Path(module.__file__).resolve() == root / relative
                     and packet.sha256(root / relative) == digest, 'Exact architecture import origin required')
            before = rng_record()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261023)
                base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(model.SITES))
                base.load_state_dict(r2['params_ema'], strict=True)
                base = base.float().eval().requires_grad_(False).to(device)
            need(before == rng_record(), 'Base construction must preserve caller RNG')
            candidate = model.build_restorer(base, u96['adapter_state'], r2_artifact=r2_path, u96_artifact=u96_path,
                head_artifact=checked(p['u64_head']), manual_vm_root=root, device='cuda:0',
                acknowledgement=model.MANUAL_ACKNOWLEDGEMENT,
                prototype_source_sha256=p['files'][PACKAGE+'/model.py']['sha256']).eval()
            need(before == rng_record(), 'Exact terminal initialization must preserve caller RNG')
            named = dict(candidate.named_parameters())
            head_items = [(name, named[name]) for name in sorted(model.expected_head_shapes())]
            need(len(named) == 575 and sum(v.numel() for _, v in head_items) == 168960,
                 'Complete575/36-head168960 census required')
            frozen_initial = arrays({n: v for n, v in named.items() if not n.startswith('heads.')})
            frozen_sha = helper.canonical_arrays(frozen_initial, np=np)[0]
            del r2, u96
            store.json('preflight.json', {'complete': True, 'protocol_sha256': expected_sha,
                'initializer': candidate.u64_initializer_receipt, 'frozen539_canonical_sha256': frozen_sha,
                'torch': torch.__version__, 'GPU': torch.cuda.get_device_name(0), 'limits': contract.LIMITS,
                'model_qualified': False})
            run_rng = rng_record()

        def assert_frozen():
            candidate.validate_invariants(check_frozen_digests=True)
            now = arrays({n: v for n, v in candidate.named_parameters() if not n.startswith('heads.')})
            need(len(now) == 539 and helper.canonical_arrays(now, np=np)[0] == frozen_sha
                 and all(np.array_equal(now[n], frozen_initial[n]) for n in now), 'Every frozen539 value must stay exact')

        def full_state(label, *, emergency=False):
            nonlocal full_states, last_complete_state
            need(full_states < 2, 'At most two full575 state payloads')
            assert_frozen()
            values = arrays(candidate.capture_complete_state_for_audit())
            need(len(values) == 575, 'Full575 values required')
            payload, schema = helper.npz_state_payload(values, np=np)
            need(len(payload) <= 384*1024**2 and schema['canonical_sha256'] == canonical(candidate.state_dict()),
                 'Bounded complete live/captured state required')
            full_states += 1  # charge a partial write before attempting it
            binding = store.write('states/'+label+'/full_state.npz', payload, emergency=emergency)
            with np.load(checked(binding), allow_pickle=False) as durable:
                decoded = {name: durable[name] for name in durable.files}
            need(helper.canonical_arrays(decoded, np=np)[0] == schema['canonical_sha256']
                 and set(decoded) == set(values) and all(np.array_equal(decoded[n], values[n]) for n in values),
                 'Durable575 state differs from actual live/captured values')
            witness = {'complete': True, 'file': binding, 'tensor_count': 575,
                       'canonical_sha256': schema['canonical_sha256'], 'frozen539_canonical_sha256': frozen_sha,
                       'arrays': schema['arrays'], 'update': active_update, 'model_qualified': False}
            last_complete_state = store.json('states/'+label+'/state.json', witness, emergency=emergency)

        controls = {row['id']: row for row in p['controls']}
        parity = {row['id']: row for row in p['u64_parity']}
        bank_ids = {r['id'] for r in selection['monitors']+selection['guards']}
        need(set(controls) == bank_ids and set(parity) == {r['id'] for r in selection['monitors'][:4]}, 'Exact56 controls/four parity roles required')
        frozen_records = {}
        with bounded_phase('saved_U96_control_integrity_zero_neural'):
            for key, control in controls.items():
                a, q = load_raw(control['raw']), rgb(control['png'])
                need(np.array_equal(q, (np.clip(a, 0, 1)*255).astype(np.uint8)), 'Cached U96 RAW/floorPNG mismatch')
                row = rows[key]
                target = None if row.get('target') is None else rgb(row['target']).astype(np.float32)/np.float32(255)
                teacher = row.get('behavior_teacher')
                kwargs = {} if teacher is None else {'teacher_raw': load_raw(teacher['raw']), 'teacher_png': rgb(teacher['png'])}
                frozen_records[key] = {**{n: row[n] for n in ('id', 'reference_id', 'role', 'split')},
                    **metrics.record_metrics(a, q, target, row, np=np, frozen_U96_raw=a, frozen_U96_png=q, **kwargs)}
                original_record = packet.strict_json(checked(control['record']).read_bytes())
                need(original_record['id'] == key and original_record['metrics'] == frozen_records[key]['metrics']
                     and original_record['regional_metrics'] == frozen_records[key]['regional_metrics'], 'Fresh cached metrics differ from original U96 record')
            store.json('frozen_U96_control_metrics.json', {'complete': True, 'records': frozen_records, 'model_qualified': False})
        with bounded_phase('complete_initial_u64_state'):
            full_state('initial_u64')

        def input_values(row):
            image = rgb(row['input'])
            need(hashlib.sha256(image.tobytes()).hexdigest() == row['input_RGB8_sha256'], 'Admitted input RGB differs')
            sample = torch.from_numpy(image.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)[None]/255.0
            sample = F.interpolate(sample, (512, 512), mode='bilinear', align_corners=False)*2-1
            target = None if row.get('target') is None else rgb(row['target'])
            if target is not None:
                need(hashlib.sha256(target.tobytes()).hexdigest() == row['target_RGB8_sha256'], 'Admitted clean target RGB differs')
                target = torch.from_numpy(target.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)[None]/255.0
            return sample, target

        def full_call(sample, call):
            nonlocal active_case
            active_case = call['id']
            need(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Strict deterministic forward required')
            category = {'initial_u64_quality_bank': 'initial_quality_bank_forwards',
                        'deterministic_same_u64_repeat': 'deterministic_repeat_forwards',
                        'new_quality_snapshot': 'candidate_snapshot_forwards_after_initializer',
                        'paired_TRAIN_optimization': 'candidate_training_forwards'}[call['phase']]
            resource('before_forward')
            token = ledger.reserve('forward', id=call['id'], category=category, plan_slot=call['slot'], update=active_update)
            status('forward_reserved')
            with torch.autocast(device_type='cuda', enabled=False):
                result = candidate(sample)
            torch.cuda.synchronize(0)
            ledger.returned(token, synchronized_return=True)
            status('forward_returned')
            valid = isinstance(result, tuple) and len(result) == 3 and all(
                isinstance(v, torch.Tensor) and v.dtype == torch.float32 and tuple(v.shape) == shape
                and v.device == device and torch.isfinite(v).all().item()
                for v, shape in zip(result, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))))
            if not valid:
                retained = []
                for i, value in enumerate(result if isinstance(result, (list, tuple)) else (result,)):
                    if isinstance(value, torch.Tensor):
                        raw_bits = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
                        retained.append({'index': i, 'dtype': str(value.dtype), 'shape': list(value.shape),
                            'bits': save_array('invalid_returns/'+str(call['slot'])+'/'+str(i)+'.npy', raw_bits)})
                store.json('invalid_returns/'+str(call['slot'])+'/record.json', {'accepted_as_valid': False, 'values': retained})
                raise ValueError('Invalid actual full return retained; do not retry')
            need(candidate.last_base_calls == 1 and tuple(candidate.last_forward_sites) == tuple(model.SITES), 'One complete four-site owned forward required')
            resource('after_forward')
            return result

        def saved_record(row, triple, call, target):
            stem = 'calls/'+str(call['slot']).zfill(4)
            unclipped = F.interpolate((triple[0]+1)/2, (256, 256), mode='bilinear', align_corners=False)
            # Preserve the actual u64 delivery order: clamp native512 first,
            # then resize. Resize-before-clamp changes boundary/overflow pixels.
            raw = F.interpolate(((triple[0]+1)/2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
            a = raw.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
            q = (np.clip(a, 0, 1)*255).astype(np.uint8)
            image = io.BytesIO()
            Image.fromarray(q).save(image, format='PNG')
            png_payload = image.getvalue()
            need(len(png_payload) <= 256*1024, 'PNG per-image cap')
            outputs = {'raw': save_array(stem+'/raw.npy', a),
                       'unclipped': save_array(stem+'/unclipped.npy', unclipped),
                       'png': store.write(stem+'/result.png', png_payload)}
            with Image.open(checked(outputs['png'])) as decoded:
                need(np.array_equal(np.asarray(decoded), q), 'Durable PNG pixels differ')
            stock = {name: tensor_record(v) for name, v in zip(('native', 'logits', 'LQ'), triple)}
            if row['id'] in parity and call['phase'] in ('initial_u64_quality_bank', 'deterministic_same_u64_repeat'):
                stock = {name: {**stock[name], 'file': save_array(stem+'/'+name+'.npy', v)}
                         for name, v in zip(('native', 'logits', 'LQ'), triple)}
            # Historical scalar metrics normalize target RGB8 with NumPy FP32.
            # Keep that arithmetic independent of CUDA objective normalization.
            target_array = (None if row.get('target') is None else
                            rgb(row['target']).astype(np.float32)/np.float32(255))
            kwargs = {}
            teacher = row.get('behavior_teacher')
            if teacher is not None:
                kwargs.update(teacher_raw=load_raw(teacher['raw']), teacher_png=rgb(teacher['png']))
            if row['id'] in controls:
                kwargs.update(frozen_U96_raw=load_raw(controls[row['id']]['raw']), frozen_U96_png=rgb(controls[row['id']]['png']))
                measured = metrics.record_metrics(a, q, target_array, row, np=np, **kwargs)
            else:
                measured = metrics.record_training_metrics(a, q, target_array, row, np=np)
            record = {**{name: row[name] for name in ('id', 'reference_id', 'role', 'split')},
                'call': call, 'update': active_update, 'input': row['input'], 'target': row.get('target'),
                'outputs': outputs, 'stock': stock, **measured, 'model_qualified': False}
            record['receipt'] = store.json(stem+'/record.json', record)
            return record, raw

        initial_bank = {}
        current_bank = {}
        initial_gate = None
        initial_calls = [c for c in plan if c['new_update'] == 0]
        with bounded_phase('initial_u64_quality_bank_and_four_repeats'):
            for call in initial_calls:
                row = rows[call['id']]
                sample, target = input_values(row)
                with torch.no_grad():
                    triple = full_call(sample, call)
                    record, raw = saved_record(row, triple, call, target)
                if call['phase'] == 'initial_u64_quality_bank':
                    initial_bank[row['id']] = record
                    if row['id'] in parity:
                        witness = parity[row['id']]
                        need(np.array_equal(load_raw(record['outputs']['raw']), load_raw(witness['raw']))
                             and np.array_equal(rgb(record['outputs']['png']), rgb(witness['png'])), 'Actual initializer differs from saved terminal-u64 RAW/PNG')
                else:
                    baseline = initial_bank[row['id']]
                    for field in ('raw', 'unclipped'):
                        need(packet.sha256(checked(record['outputs'][field])) == packet.sha256(checked(baseline['outputs'][field])), 'Same-u64 deterministic array repeat differs')
                    need(np.array_equal(rgb(record['outputs']['png']), rgb(baseline['outputs']['png'])), 'Same-u64 deterministic PNG repeat differs')
                    for name in ('native', 'logits', 'LQ'):
                        need(record['stock'][name] == {**baseline['stock'][name], 'file': record['stock'][name]['file']}, 'Same-u64 native/logits/LQ repeat differs')
                del sample, target, triple, raw
            initial_gate = metrics.evaluate_bank(initial_bank, initial_bank, frozen_records, parent_bytes,
                                                 phase='initializer', saved_audit_bytes=audit_bytes)
            store.json('banks/u0000/bank.json', {'complete': True, 'records': initial_bank, 'gate': initial_gate})
            if not initial_gate['pass']:
                raise QualityStop('Exact u64 initializer failed its inherited TRAIN screen; preserve stop')

        @contextmanager
        def backward_warning_scope():
            need(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Strict policy before backward required')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    yield
                finally:
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    warning_rows = [{'category': x.category.__name__, 'message': str(x.message)[:12000]} for x in caught]
                    unexpected = [x for x in warning_rows if ('deterministic' in x['message'].lower() or 'nondeterminism' in x['message'].lower())
                                  and re.match(r'^upsample_bilinear2d_backward_out_cuda\b', x['message']) is None]
                    event({'kind': 'backward_warning_scope', 'warnings': warning_rows, 'unexpected': unexpected,
                           'strict_flags_restored': True, 'bitwise_CUDA_gradient_repeatability_claimed': False,
                           'counters': dict(ledger.counters)})
                    need(not unexpected, 'Unexpected nondeterministic backward; preserve stop')

        def cpu_tree(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if type(value) is dict:
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu_tree(v) for v in value)
            return value

        def saved_tree_equal(left, right):
            if isinstance(left, torch.Tensor):
                return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
            if type(left) is not type(right):
                return False
            if type(left) is dict:
                return set(left) == set(right) and all(saved_tree_equal(left[k], right[k]) for k in left)
            if isinstance(left, (tuple, list)):
                return len(left) == len(right) and all(saved_tree_equal(a, b) for a, b in zip(left, right))
            return left == right

        def checkpoint(update, *, emergency=False):
            nonlocal last_checkpoint
            assert_frozen()
            heads = {name: value for name, value in head_items}
            head_payload, schema = helper.npz_state_payload(arrays(heads), np=np)
            folder = 'checkpoints/u'+str(update).zfill(4)
            head_file = store.write(folder+'/heads.npz', head_payload, emergency=emergency)
            with np.load(checked(head_file), allow_pickle=False) as durable:
                saved_heads = {name: durable[name] for name in durable.files}
            need(helper.canonical_arrays(saved_heads, np=np)[0] == schema['canonical_sha256']
                 and all(np.array_equal(saved_heads[n], a) for n, a in arrays(heads).items()),
                 'Actual durable36 head values differ from committed model')
            snapshot = {'format': 'u64-cleananchors-closed-finite-state-v1', 'protocol_sha256': expected_sha,
                        'update': update, 'automatic_resume_permitted': False, 'optimizer': cpu_tree(optimizer.state_dict()),
                        'optimizer_parameter_names': [name for name, _ in head_items], 'RNG': rng_record(),
                        'sampler_next_new_update': update+1 if update < 16 else None,
                        'initial_head_canonical_sha256': contract.HEAD_CANONICAL_SHA256,
                        'frozen539_canonical_sha256': frozen_sha, 'counters': dict(ledger.counters), 'model_qualified': False}
            stream = io.BytesIO()
            torch.save(snapshot, stream)
            payload = stream.getvalue()
            need(len(head_payload)+len(payload) <= 4*1024**2, 'Head/Adam/RNG checkpoint4MiB cap')
            resume = store.write(folder+'/resume.pt', payload, emergency=emergency)
            restored = torch.load(checked(resume), map_location='cpu', weights_only=True)
            need(saved_tree_equal(snapshot, restored), 'Durable Adam/RNG/sampler readback differs from committed snapshot')
            last_checkpoint = store.json(folder+'/checkpoint.json', {'complete': True, 'update': update,
                'heads': head_file, 'head_canonical_sha256': schema['canonical_sha256'],
                'resume': resume, 'frozen539_canonical_sha256': frozen_sha, 'counters': dict(ledger.counters),
                'model_qualified': False}, emergency=emergency)
            status('checkpoint_committed')

        with bounded_phase('one_fresh_Adam_and_initial_checkpoint'):
            token = ledger.reserve('optimizer_construction')
            status('optimizer_construction_reserved')
            options = dict(contract.prospective_contract()['optimizer'])
            kwargs = {k: options[k] for k in ('eps', 'weight_decay', 'foreach', 'fused', 'capturable', 'differentiable', 'amsgrad', 'maximize')}
            optimizer = torch.optim.Adam([v for _, v in head_items], lr=0.0001, betas=(0.9, 0.99), **kwargs)
            ledger.returned(token)
            need(not optimizer.state, 'Fresh empty Adam moments required for changed objective')
            checkpoint(0)
        for update in range(1, 17):
            active_update = update
            with bounded_phase('new_Adam_update_'+str(update)):
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(update)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                micro = []
                calls = [c for c in plan if c['new_update'] == update and c['backward']]
                for call in calls:
                    row = rows[call['id']]
                    need(row['cohort'] == 'optimize' and row['optimization_permitted'] is True and row['target'] is not None, 'Admitted paired TRAIN targets only')
                    sample, target = input_values(row)
                    with torch.enable_grad(), torch.autocast(device_type='cuda', enabled=False):
                        triple = full_call(sample, call)
                        record, raw = saved_record(row, triple, call, target)
                        value, terms = objective.connected_objective(raw, target, row['role'], torch=torch)
                        scalar = {'complete': True, 'new_update': update, 'call_slot': call['slot'], 'id': row['id'],
                            'role': row['role'], 'LR': lr, 'objective': float(value.detach().item()),
                            'divided_objective': float((value/4).detach().item()),
                            'terms': {k: float(v.detach().item()) for k, v in terms.items()},
                            'output_record': record['receipt'], 'head_before_commit_sha256': canonical(dict(head_items)),
                            'photographic_target': row['target'], 'model_qualified': False}
                        micro.append(store.json('calls/'+str(call['slot']).zfill(4)+'/objective.json', scalar))
                        token = ledger.reserve('backward', id=row['id'], update=update, plan_slot=call['slot'])
                        status('backward_reserved')
                        with backward_warning_scope():
                            (value/4).backward()
                            torch.cuda.synchronize(0)
                            ledger.returned(token, synchronized_return=True)
                            status('backward_returned')
                    need(all(v.grad is None for n, v in candidate.named_parameters() if not n.startswith('heads.')), 'Frozen539 gradients must stay None')
                    del sample, target, triple, raw, value, terms, record
                need(all(v.grad is not None and v.grad.dtype == torch.float32 and torch.isfinite(v.grad).all().item() for _, v in head_items), 'All36 finite accumulated gradients required')
                need(any(torch.count_nonzero(v.grad).item() for _, v in head_items), 'All-zero Adam update refused')
                before_heads = arrays(dict(head_items))
                prefix = 'updates/u'+str(update).zfill(4)
                before_payload, _ = helper.npz_state_payload(arrays({n: v.grad for n, v in head_items}), np=np)
                preclip = store.write(prefix+'/preclip_gradients.npz', before_payload)
                norm = torch.nn.utils.clip_grad_norm_([v for _, v in head_items], max_norm=1.0, error_if_nonfinite=True)
                after_payload, _ = helper.npz_state_payload(arrays({n: v.grad for n, v in head_items}), np=np)
                need(len(before_payload)+len(after_payload) <= 2*1024**2, 'Gradient bundle2MiB cap')
                postclip = store.write(prefix+'/postclip_gradients.npz', after_payload)
                token = ledger.reserve('optimizer_update', update=update)
                status('optimizer_update_reserved')
                optimizer.step()
                torch.cuda.synchronize(0)
                ledger.returned(token, synchronized_return=True)
                status('optimizer_update_returned')
                need(all(torch.isfinite(v).all().item() for _, v in head_items), 'Finite updated head values required')
                optimizer.zero_grad(set_to_none=True)
                assert_frozen()
                now = arrays(dict(head_items))
                need(any(not np.array_equal(now[n], before_heads[n]) for n in now), 'Adam commit changed no head value')
                store.json(prefix+'/commit.json', {'complete': True, 'new_update': update, 'LR': lr,
                    'microbatches': micro, 'preclip_gradients': preclip, 'postclip_gradients': postclip,
                    'preclip_norm': float(norm.detach().item()), 'clip_coefficient': min(1.0, 1.0/(float(norm.detach().item())+1e-6)),
                    'head_before_sha256': helper.canonical_arrays(before_heads, np=np)[0],
                    'head_after_sha256': helper.canonical_arrays(now, np=np)[0], 'frozen539_exact': True,
                    'counters': dict(ledger.counters), 'model_qualified': False})
                checkpoint(update)
                print(json.dumps({'new_u64_update': update, 'maximum': 16, 'LR': lr}), flush=True)
            if update in (4, 8, 16):
                with bounded_phase('quality_bank_u'+str(update)):
                    current_bank = {}
                    for call in [c for c in plan if c['new_update'] == update and not c['backward']]:
                        row = rows[call['id']]
                        sample, target = input_values(row)
                        with torch.no_grad():
                            triple = full_call(sample, call)
                            record, raw = saved_record(row, triple, call, target)
                        current_bank[row['id']] = record
                        del sample, target, triple, raw
                    gate = metrics.evaluate_bank(current_bank, initial_bank, frozen_records, parent_bytes,
                                                 phase='u'+str(update), saved_audit_bytes=audit_bytes)
                    store.json('banks/u'+str(update).zfill(4)+'/bank.json', {'complete': True, 'records': current_bank, 'gate': gate})
                    assert_frozen()
                    print(json.dumps({'new_u64_snapshot': update, 'pass': gate['pass']}), flush=True)
                    if not gate['pass']:
                        raise QualityStop('Original preservation or u64-retention requirement failed; retain stop')
        active_case = None
        with bounded_phase('terminal_full_state_and_integrity'):
            full_state('terminal_u0016')
            need(rng_record() == run_rng, 'Fixed model/sampler operations changed caller RNG')
            snap = ledger.snapshot()
            need(not snap['operation_outcome_unknown'] and ledger.counters['full_forward_returns'] == 292
                 and ledger.counters['backward_returns'] == 64 and ledger.counters['optimizer_updates'] == 16,
                 'Complete292/64/16 known operation prefix required')
            store.audit_namespace()
            store.json('result.json', {'complete': True, 'status': 'FINITE_U64_TARGET_REFERENCE_SCREEN_PASSED_NOT_QUALIFIED',
                'protocol_sha256': expected_sha, 'counters': dict(ledger.counters), 'last_checkpoint': last_checkpoint,
                'last_complete_state': last_complete_state, 'model_qualified': False,
                'inherited_broader_failures_not_repaired_by_TRAIN_pass': True})
            event({'kind': 'run_result_saved', 'counters': dict(ledger.counters)})
            status('finite_screen_complete', complete=True)
            print(json.dumps({'complete': True, 'new_updates': 16, 'model_qualified': False}), flush=True)
        return 0
    except BaseException as error:
        if store is not None:
            fail = {'complete': False, 'quality_stop': isinstance(error, QualityStop),
                'phase': failed_phase or phase, 'error_type': type(error).__name__, 'error': str(error)[:4000],
                'traceback': traceback.format_exc()[-20000:], 'protocol_sha256': expected_sha,
                'active_update': active_update, 'active_case': active_case, 'last_checkpoint': last_checkpoint,
                'last_complete_state': last_complete_state, 'model_qualified': False,
                **(ledger.snapshot() if ledger else {})}
            try:
                budget = helper.failure_capture_budget(time.monotonic()-began)
                if candidate is not None and budget > 0:
                    with timed(budget, 'bounded_failure_state'):
                        try:
                            if full_states < 2:
                                need(frozen_initial is not None, 'Frozen baseline unavailable; retain current values as unvalidated')
                                full_state('failed_current_u'+str(active_update).zfill(4), emergency=True)
                        except TimeoutError:
                            # The one-shot alarm has already expired; do not
                            # start fallback work outside its finite budget.
                            raise
                        except BaseException as valid_capture_error:
                            fail['validated_state_capture_error'] = repr(valid_capture_error)[:2000]
                            if full_states < 2:
                                # Retain actual invalid/nonfinite/frozen-drift values
                                # without certifying them or making a new model call.
                                raw_state = arrays(candidate.state_dict())
                                need(len(raw_state) == 575 and all(v.dtype == np.float32 for v in raw_state.values()),
                                     'Invalid state retention still requires actual575 FP32 arrays')
                                payload = io.BytesIO()
                                np.savez(payload, **raw_state)
                                raw_payload = payload.getvalue()
                                need(len(raw_payload) <= 384*1024**2, 'Invalid state384MiB cap')
                                full_states += 1
                                fail['unvalidated_current575'] = store.write('failure_values/current575.npz', raw_payload, emergency=True)
                                fail['unvalidated_current575_accepted_as_valid'] = False
                        if optimizer is not None:
                            stream = io.BytesIO()
                            torch.save({'optimizer': cpu_tree(optimizer.state_dict()),
                                'heads': cpu_tree(dict(head_items)),
                                'actual_gradients': cpu_tree({n: v.grad for n, v in head_items}),
                                'RNG': rng_record(), 'new_update': active_update,
                                'operation_ledger': ledger.snapshot(), 'accepted_as_valid': False}, stream)
                            failure_payload = stream.getvalue()
                            need(len(failure_payload) <= 8*1024**2, 'Failure head/Adam/grad evidence8MiB cap')
                            fail['actual_partial_head_Adam_RNG'] = store.write('failure_values/partial_optimizer.pt', failure_payload, emergency=True)
                    fail['last_complete_state'] = last_complete_state
                fail['partial_files'] = sorted(store.partial_files)
            except BaseException as capture_error:
                fail['extra_state_capture_error'] = repr(capture_error)[:2000]
            try:
                with timed(min(5.0, 1255-(time.monotonic()-began)), 'failure_receipt'):
                    store.json('failure.json', fail, emergency=True)
                    event({'kind': 'failure_saved', 'counters': dict(ledger.counters)})
                    status('retained_stop', quality_stop=isinstance(error, QualityStop))
            except BaseException as receipt_error:
                print('Failure receipt error: '+repr(receipt_error), flush=True)
        traceback.print_exc()
        return 2 if isinstance(error, QualityStop) else 1
    finally:
        if torch is not None and caller_rng is not None:
            try:
                with timed(min(5.0, 1260-(time.monotonic()-began)), 'caller_RNG_restore'):
                    random.setstate(caller_rng['python'])
                    np.random.set_state(caller_rng['numpy'])
                    torch.set_rng_state(caller_rng['CPU'])
                    torch.cuda.set_rng_state_all(caller_rng['CUDA'])
                    need(torch.equal(torch.get_rng_state().cpu(), caller_rng['CPU'])
                         and all(torch.equal(a.cpu(), b) for a, b in zip(torch.cuda.get_rng_state_all(), caller_rng['CUDA'])), 'Caller RNG restore differs')
            except BaseException as restore_error:
                print('Bounded caller RNG restore failed: '+repr(restore_error), flush=True)
                # A previously saved finite result cannot convert an integrity
                # failure during finalization into supervisor exit0.
                raise RuntimeError('Caller RNG finalization failed; retain prior result as incomplete evidence') from restore_error
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
