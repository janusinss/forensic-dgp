"""Finite manual-L4 fusion adaptation, inert to import and transfer checking.

No neural framework is imported until --run passes exact manual root, transfer,
admission, finite-plan, idle and resource checks. No automatic resume, historical
run, retry, quality-gate bypass or app promotion exists. Returned Python is never
used by the independent local artifact auditor.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid

from . import data as d
from . import data_v2 as schedule_data
from . import model as m
from . import transport as t


OUTPUTS = t.OUTPUTS
FINAL_UPDATE = 151
TEACHER_SOURCES = 55
STAGES = {'initialization': 600, 'witness': 180, 'training': 1800,
          'snapshot': 600, 'preflight': 300, 'worker': 4800}
WITNESS_FORWARD_CAPS = {'parity_candidate_forwards': 1, 'parity_R2_forwards': 1,
    'filter_witness_forwards': 3, 'clear_witness_forwards': 2, 'dark_witness_forwards': 2}
GRADIENT_QUERY_NAMES = ('rows_complete_divided', 'rows_spatial_unweighted',
    'rows_adjacent_unweighted', 'local_mosaic_complete_divided',
    'local_blur_complete_divided', 'clear_original_complete_divided',
    'dark_R2_initial_complete_divided')


def require(value, message):
    if not value:
        raise ValueError(message)


def call_caps():
    ledger = t.operation_ledger(WITNESS_FORWARD_CAPS, teacher_sources=TEACHER_SOURCES)
    return {**ledger['by_counter'], 'backward_calls': FINAL_UPDATE * 4,
            'optimizer_step_attempts': FINAL_UPDATE, 'gradient_queries': 7}


def validate_plan(protocol):
    """Exact full intended finite experiment; never silently shrink admission."""
    require(protocol.get('vm_name') == t.VM_NAME and protocol.get('outputs') == OUTPUTS,
            'Exact new pilot namespace required')
    require(protocol.get('call_caps') == call_caps(), 'Exact3639-forward/604-backward/151-step/7-query plan required')
    require(protocol.get('evaluation_path') == 'evaluation_bundle.json', 'Exact frozen evaluation bundle path required')
    data = protocol.get('data')
    require(type(data) is dict and all(type(data.get(key)) is str for key in
        ('bundle_path', 'schedule_path', 'admission_path', 'dark_review_path',
         'holds_sha256', 'dark_review_sha256')), 'Complete frozen data/admission bindings required')
    for key in ('bundle_path', 'schedule_path', 'admission_path', 'dark_review_path'):
        t.canonical(data[key]); require(not data[key].startswith('outputs/'), 'Training metadata cannot be returned output')
    guards = protocol.get('guard_cases')
    require(type(guards) is list and len(guards) == 24 and
            {role: sum(row.get('role') == role for row in guards) for role in ('clear', 'dark', 'filter')} ==
            {'clear': 8, 'dark': 8, 'filter': 8}, 'Exact24 TRAIN guard probes required')
    require(len({(row['role'], row['id']) for row in guards}) == 24,
            'Distinct declared role-case guard probes required')
    witnesses = protocol.get('witness_cases')
    require(type(witnesses) is dict and set(witnesses) == {'clear', 'dark', 'rows', 'local_mosaic', 'local_blur'},
            'Exact prospective five input witness bindings required')
    require(protocol.get('snapshots') == [0, 50, FINAL_UPDATE] and protocol.get('guard_snapshots') == [0, 20],
            'Exact planned state/output schedule required')
    require(protocol.get('gradient_queries') == list(GRADIENT_QUERY_NAMES), 'Exact seven gradient-query purpose order required')
    require(protocol.get('automatic_resume') is False and protocol.get('automatic_training_launch') is False,
            'Manual finite no-resume boundary required')
    t.allowed_output_names(protocol)
    return {'complete': True, 'prospective': True, 'total_top_level_forwards': 3639,
            'gradient_queries': 7, 'backward_calls': 604, 'optimizer_step_attempts': FINAL_UPDATE,
            'snapshots': [0, 50, FINAL_UPDATE], 'guard_snapshots': [0, 20], 'model_qualified': False}


class Counters:
    """Reserve durable operation evidence BEFORE each actual call or attempt."""
    def __init__(self, caps, append):
        require(caps == call_caps(), 'Exact finite source counter caps required')
        self.caps = dict(caps); self.values = dict.fromkeys(caps, 0)
        self.append = append; self.sequence = 0

    def reserve(self, kind, **context):
        require(kind in self.caps and self.values[kind] < self.caps[kind], 'Finite operation cap exceeded: ' + str(kind))
        self.values[kind] += 1; self.sequence += 1
        row = {'event': 'reserved', 'sequence': self.sequence, 'kind': kind,
               'count': self.values[kind], 'context': context}
        self.append(row)
        return self.sequence

    def returned(self, reservation, **context):
        require(type(reservation) is int and 0 < reservation <= self.sequence,
                'Actual reserved operation required')
        self.append({'event': 'returned', 'reservation': reservation, 'context': context})


class Writer:
    """Exact fresh finite namespace, exclusive physical writes, retained partials."""
    def __init__(self, root, protocol):
        self.root = Path(root); self.allowed = t.allowed_output_names(protocol)
        self.bytes = 0; self.files = 0; self.append_sizes = {}; self.started = time.monotonic()
        for name, path in t.inventory(self.root).items():
            self.bytes += path.stat().st_size; self.files += 1
        require(self.bytes <= t.LIMITS['payload_bytes'] and self.files <= t.LIMITS['regular_files'],
                'Installed static payload already exceeds full return envelope')

    def path(self, name, *, temporary=False):
        t.canonical(name)
        require(name in self.allowed or (temporary and t.checkpoint_kind(name) == 'temporary'),
                'Write outside exact sealed artifact plan: ' + name)
        return t.no_links(self.root / name)

    def reserve(self, name, amount, *, temporary=False, new_file=True):
        path = self.path(name, temporary=temporary)
        require(type(amount) is int and amount >= 0 and
                self.bytes + amount <= t.LIMITS['payload_bytes'] and
                self.files + int(new_file) <= t.LIMITS['regular_files'], 'Whole output envelope exceeded')
        require(shutil.disk_usage(self.root).free >= t.LIMITS['reserve_free_bytes'] + amount,
                'Physical write would cross4GiB disk reserve')
        return path

    def write(self, name, payload, *, temporary=False):
        require(type(payload) is bytes, 'Exact inert byte payload required')
        path = self.reserve(name, len(payload), temporary=temporary)
        path.parent.mkdir(parents=True, exist_ok=True)
        opened = False
        try:
            with path.open('xb') as stream:
                opened = True
                require(stream.write(payload) == len(payload), 'Short exclusive artifact write')
                stream.flush(); os.fsync(stream.fileno())
        finally:
            if opened:
                self.bytes += path.stat().st_size; self.files += 1
        return {'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def json(self, name, value):
        payload = t.json_bytes(value)
        require(len(payload) <= t.LIMITS['one_json_bytes'], 'Individual JSON cap exceeded')
        return self.write(name, payload)

    def append(self, name, value):
        payload = t.json_bytes(value)
        fresh = name not in self.append_sizes
        path = self.reserve(name, len(payload), new_file=fresh)
        require(self.append_sizes.get(name, 0) + len(payload) <= 16 * t.MiB,
                'Durable operation journal cap exceeded')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb' if fresh else 'ab') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        self.append_sizes[name] = self.append_sizes.get(name, 0) + len(payload)
        self.bytes += len(payload); self.files += int(fresh)

    def array(self, name, array, np):
        require(name.endswith('.npy') and not array.dtype.hasobject and np.isfinite(array).all(),
                'Finite inert non-object NPY required')
        buffer = io.BytesIO(); np.save(buffer, array, allow_pickle=False)
        require(buffer.tell() <= t.LIMITS['array_bytes'], 'Individual retained array cap exceeded')
        return self.write(name, buffer.getvalue())

    def png(self, name, rgb, np, Image):
        require(name.endswith('.png') and rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3), 'Exact RGB8 PNG256 required')
        buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, format='PNG')
        require(buffer.tell() <= t.LIMITS['png256_bytes'], 'Individual PNG256 cap exceeded')
        return self.write(name, buffer.getvalue())


@contextmanager
def stage_deadline(seconds):
    """Nested Linux wall alarm; outer supervisor remains a separate hard cap."""
    require(type(seconds) is int and seconds > 0, 'Positive finite stage seconds required')
    require(hasattr(signal, 'setitimer'), 'Manual Linux wall alarm required')
    old_handler = signal.getsignal(signal.SIGALRM); old_timer = signal.getitimer(signal.ITIMER_REAL)
    start = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError('Finite stage wall limit exceeded: ' + str(seconds))
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, old_timer[0]) if old_timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, old_timer[0] - (time.monotonic() - start)), old_timer[1])


def rss_bytes():
    for row in Path('/proc/self/status').read_text(encoding='ascii').splitlines():
        if row.startswith('VmRSS:'):
            return int(row.split()[1]) * 1024
    raise ValueError('Current process RSS unavailable')


def peak_rss_bytes():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def competing_process(arguments):
    """Pure activity classification, including Python module entry points.

    Read-only verify/export/audit actions stay disjoint. No tmux-session name is
    evidence of a running trainer. Unparseable commands are explicitly unknown.
    """
    require(type(arguments) is str and arguments, 'Full process arguments required')
    try:
        words = shlex.split(arguments)
    except ValueError:
        return {'competing': True, 'reason': 'unclassifiable full process command', 'words': None}
    if not words:
        return {'competing': False, 'reason': 'empty process arguments', 'words': []}
    leaves = [Path(word).name.lower() for word in words]
    python = any(name.startswith('python') for name in leaves[:2])
    activity = any(word in words for word in ('--run', '--train', '--fit', '--epochs', '--max-updates'))
    readonly = any(word in words for word in ('--verify-transfer', '--verify', '--export', '--audit', '--help'))
    if python and readonly and not activity:
        return {'competing': False, 'reason': 'explicit read-only Python action', 'words': words}
    module = words[words.index('-m') + 1] if '-m' in words and words.index('-m') + 1 < len(words) else ''
    files = [name for name in leaves if name.endswith('.py')]
    training_name = any(name.startswith(('train_', 'training')) or 'training' in name or
                        name.startswith('supervise_') for name in files)
    module_activity = python and bool(module) and activity
    inline_training = python and '-c' in words and any(token in arguments for token in ('.backward(', '.step(', 'optimizer.step'))
    shell_training = any(name in ('bash', 'sh', 'zsh') for name in leaves[:1]) and any(
        name.endswith('.sh') and name.startswith('run_') and 'export' not in name and 'audit' not in name
        for name in leaves[1:])
    competing = (python and (training_name or activity)) or module_activity or inline_training or shell_training
    return {'competing': competing, 'reason': 'active trainer/module/run wrapper' if competing else 'disjoint process',
            'words': words}


def idle_inventory():
    """Inventory first; never kill or interrupt another workload."""
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    require(not result.stdout.strip(), 'Competing GPU workload; preserve it and do not launch')
    ps = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True, timeout=20)
    competing = []; inventory = []
    for row in ps.stdout.splitlines():
        fields = row.strip().split(None, 1)
        require(len(fields) == 2 and fields[0].isdigit(), 'Unclassifiable process inventory')
        if int(fields[0]) in (os.getpid(), os.getppid()):
            inventory.append({'pid': int(fields[0]), 'arguments': fields[1], 'classification': 'current worker/supervisor ancestry'})
            continue
        classification = competing_process(fields[1])
        inventory.append({'pid': int(fields[0]), 'arguments': fields[1], **classification})
        if classification['competing']:
            competing.append(fields[1])
    require(not competing, 'Competing trainer/supervisor; preserve it and do not launch')
    return {'complete': True, 'competing_GPU_processes': [], 'competing_training_processes': [], 'full_process_inventory': inventory,
            'inventory_first_only_no_processes_killed': True}


class Runtime:
    def __init__(self, root, writer, torch=None):
        self.root, self.writer, self.torch = Path(root), writer, torch
        self.started = time.monotonic(); self.samples = 0

    def check(self, context):
        elapsed = time.monotonic() - self.started; memory = rss_bytes(); peak_memory = peak_rss_bytes()
        free = shutil.disk_usage(self.root).free
        gpu = 0 if self.torch is None else self.torch.cuda.memory_reserved(0)
        row = {'event': 'resource_sample', 'context': context, 'seconds': elapsed,
               'RSS_bytes': memory, 'peak_RSS_bytes': peak_memory, 'disk_free_bytes': free, 'CUDA_reserved_bytes': gpu}
        self.writer.append(OUTPUTS + '/resource.jsonl', row); self.samples += 1
        require(elapsed <= STAGES['worker'] and memory <= 8 * t.GiB and peak_memory <= 8 * t.GiB and
                gpu <= 22 * t.GiB and free >= t.LIMITS['reserve_free_bytes'], 'Finite runtime/RSS/CUDA/disk bound exceeded')
        return row


def _local_metadata(root, protocol):
    """Only hash-pinned installed static metadata; no returned/dynamic code."""
    data = protocol['data']
    bundle = t.read_json(t.regular(root, data['bundle_path']))
    schedule = t.read_json(t.regular(root, data['schedule_path']))
    admission = t.read_json(t.regular(root, data['admission_path']))
    dark_review_path = t.regular(root, data['dark_review_path'])
    require(t.file_pin(dark_review_path)['sha256'] == data['dark_review_sha256'], 'Exact root dark review differs')
    schedule_data.require_admission(admission, data['holds_sha256'], data['dark_review_sha256'], schedule)
    allcases = bundle['case_by_id']
    require(type(allcases) is dict and len(allcases) == 411, 'Full301 filter plus55 clear plus55 dark cohort required')
    require(len([r for r in allcases.values() if r['profile'] == 'clear']) == TEACHER_SOURCES,
            'All55 clear source anchors required')
    require(all(r['split'] == 'TRAIN' for r in allcases.values()), 'Exact disjoint411 TRAIN role inputs required')
    for update in schedule['updates']:
        for slot in update['slots']:
            require(slot['id'] in allcases and allcases[slot['id']]['reference_id'] == slot['reference_id'],
                    'Frozen sampler case/reference join differs')
    evaluation = t.read_json(t.regular(root, protocol['evaluation_path']))['cases']
    require(type(evaluation) is list and len(evaluation) == len({r['id'] for r in evaluation}) == 478,
            'Exact406 DEV plus72 mixed-role evaluation slots required')
    require(sum(r.get('cohort') == 'historical_DEV406' for r in evaluation) == 406 and
            sum(r.get('cohort') == 'retained_filter72_mixed_roles' for r in evaluation) == 72,
            'Evaluation cohort roles/scope differ')
    for row in protocol['guard_cases']:
        require(row['id'] in allcases, 'TRAIN guard probe not explicitly admitted')
    for role, identifier in protocol['witness_cases'].items():
        require(identifier in allcases, 'Witness input not explicitly admitted: ' + role)
    return allcases, schedule, evaluation


def _input_tensor(root, row, torch, np):
    pixel = row.get('input_pixels', {})
    expected = pixel.get('RGB8_sha256', row.get('input_RGB8_sha256'))
    require(isinstance(expected, str), 'Exact input RGB8 digest required')
    rgb = d.rgb8(root, row['input'], expected)
    array = np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255))
    if pixel.get('float32_sha256') is not None:
        require(hashlib.sha256(array.tobytes()).hexdigest() == pixel['float32_sha256'], 'Input float32 geometry policy differs')
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to('cuda'), array


def _clean_tensor(root, row, torch, np):
    rgb = d.rgb8(root, row['target256'], row['target256_pixels']['RGB8_sha256'])
    array = np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255))
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to('cuda')


def _arrays(record, np):
    def hwc(value):
        return np.ascontiguousarray(value.detach().cpu().numpy()[0].transpose(1, 2, 0))
    return hwc(record.unclipped256), hwc(record.delivered256), hwc(m.png_tensor(record.delivered256))


def _evaluation_truth(root, row, np, Image):
    target = None
    pin = row.get('target256', row.get('target'))
    if pin is not None:
        expected = (row.get('target256_pixels') or row.get('target_pixels') or {}).get('RGB8_sha256', row.get('target_RGB8_sha256'))
        rgb = d.rgb8(root, pin, expected)
        target = np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255))
    support_pin = row.get('support')
    if support_pin is None:
        support = np.ones((256, 256), dtype=bool)
    else:
        with Image.open(io.BytesIO(d.verified_bytes(root, support_pin))) as image:
            require(image.mode == 'L' and image.size == (256, 256), 'Exact L support256 required')
            support = np.asarray(image).copy() == 255
    return target, support


def observed_canvas(raw, png, input_rgb8, support, np):
    """Historical prepared-canvas policy: exact input outside observed support."""
    require(raw.dtype == np.float32 and png.dtype == input_rgb8.dtype == np.uint8 and
            raw.shape == png.shape == input_rgb8.shape == (256, 256, 3) and
            support.shape == (256, 256) and support.dtype == np.bool_ and support.any(),
            'Exact prepared canvas/support geometry required')
    result_raw, result_png = raw.copy(), png.copy()
    result_raw[~support] = input_rgb8[~support].astype(np.float32) / np.float32(255)
    result_png[~support] = input_rgb8[~support]
    return result_raw, result_png


def _save_output(writer, prefix, row, record, metrics_module, root, np, Image):
    u, raw, png = _arrays(record, np)
    target, support = _evaluation_truth(root, row, np, Image)
    pixels = row.get('input_pixels', {})
    input_rgb8 = d.rgb8(root, row['input'], pixels.get('RGB8_sha256', row.get('input_RGB8_sha256')))
    raw, png = observed_canvas(raw, png, input_rgb8, support, np)
    evidence = row.get('evidence', 'paired_synthetic_TRAIN')
    if target is None and row.get('profile') == d.DARK_LINEAR_PROFILE:
        # This TRAIN behavior-retention probe has no optimizing clean target.
        # Null clean scores are not falsely labeled native-CCTV measurements.
        measured = {view: dict.fromkeys(('mae', 'mse', 'psnr', 'ssim', 'luminance_bias',
            'signed_luminance_bias', 'coarse_rgb_mae', 'adjacent_rgb_error'))
            for view in ('raw', 'native_png')}
        evidence = 'TRAIN_brightness_behavior_retention_no_clean_scores'
    else:
        measured = metrics_module.metrics(raw, png, target, support, evidence)
    pins = {'raw': writer.array(prefix + '.npy', raw, np),
            'png': writer.png(prefix + '.png', png, np, Image)}
    metadata = {'id': row['id'], 'reference_id': row.get('reference_id'), 'profile': row.get('profile'),
        'split': row.get('split'), 'cohort': row.get('cohort'), 'evidence': evidence,
        'source': row.get('source'),
        'input': row['input'], 'target': row.get('target256', row.get('target')), 'support': row.get('support'),
        'retained_case_metadata': row,
        'outputs': pins, 'metrics': measured, 'clipping': m.clipping_statistics(record),
        'U_readout_parity': record.U_readout_parity,
        'clipping_statistics_scope': 'native forward and unclipped U before observed-canvas compositing',
        'outside_observed_support_RAW_and_PNG_match_input_exact': True,
        'upstream': m.readout_fingerprints(record), 'RAW_PNG_separate': True,
        'unclipped_U_saved_for_training_or_witness_only': True, 'model_qualified': False}
    writer.json(prefix + '.json', metadata)
    return metadata


def planned_output_names(protocol, evaluation, case_by_id):
    """Pure complete worker artifact namespace, sealed before a manual launch.

    Supervisor names are separately supplied/validated by its own source. Partial
    checkpoint temporaries follow transport.checkpoint_kind's finite slots.
    """
    require(len(evaluation) == 478 and len(case_by_id) == 411, 'Full planned cohort required for artifact census')
    names = {OUTPUTS + '/' + name for name in ('runtime.json', 'model_initialization.json',
        'resource.jsonl', 'operations.jsonl', 'teacher/manifest.json', 'witness/receipt.json',
        'run_results.json', 'failure.json', 'stopped_checkpoint.pt', 'export_manifest.json')}
    for alias in ('original', 'parent', 'r2'):
        names.add(OUTPUTS + '/baseline/' + alias + '/model.json')
        for row in evaluation:
            names.update(OUTPUTS + '/baseline/' + alias + '/' + row['id'] + suffix
                         for suffix in ('.npy', '.png', '.json'))
    for update in (0, 50, FINAL_UPDATE):
        prefix = OUTPUTS + f'/snapshot_{update:04d}/'
        names.update((prefix + 'checkpoint.pt', prefix + 'gates.json'))
        for row in evaluation:
            names.update(prefix + row['id'] + suffix for suffix in ('.npy', '.png', '.json'))
    for update in (0, 20):
        prefix = OUTPUTS + f'/guard_{update:04d}/'
        names.update((prefix + 'bank.json', prefix + 'gates.json'))
        for row in protocol['guard_cases']:
            names.update(prefix + row['role'] + '/' + row['id'] + suffix
                         for suffix in ('.npy', '.png', '.json'))
    for row in case_by_id.values():
        role = 'clear' if row['profile'] == 'clear' else 'dark' if row['profile'] == d.DARK_LINEAR_PROFILE else None
        if role is not None:
            names.update(OUTPUTS + '/teacher/' + role + '/' + row['id'] + suffix
                         for suffix in ('_U.npy', '_RAW.npy', '_PNG.png'))
    for label in ('parity_candidate', 'parity_R2', 'rows', 'local_mosaic', 'local_blur',
                  'clear_candidate', 'clear_original', 'dark_candidate', 'dark_R2'):
        names.update(OUTPUTS + '/witness/' + label + suffix for suffix in ('_U.npy', '_RAW.npy'))
    names.update(OUTPUTS + '/witness/' + label + '_native.npy' for label in ('parity_candidate', 'parity_R2'))
    for update in range(1, FINAL_UPDATE + 1):
        names.update(OUTPUTS + f'/training/u{update:04d}_s{slot}.json' for slot in range(1, 5))
        names.add(OUTPUTS + f'/training/u{update:04d}_commit.json')
    for name in names:
        t.canonical(name)
    require(len(names) == len({name.casefold() for name in names}), 'Portable output namespace collision')
    return sorted(names)


def initial_output_parity(bank, frozen_R2):
    """Full478 current-runtime initial RAW/PNG parity, no extra model calls."""
    require(set(bank) == set(frozen_R2) and len(bank) == 478, 'Every478 initial output required for parity')
    different = {view: [key for key in sorted(bank) if
        bank[key]['outputs'][view]['sha256'] != frozen_R2[key]['outputs'][view]['sha256'] or
        bank[key]['outputs'][view]['bytes'] != frozen_R2[key]['outputs'][view]['bytes']]
        for view in ('raw', 'png')}
    require(not any(different.values()), 'Full478 initial candidate/frozen-R2 delivered RAW/PNG exact parity failed')
    return {'pass': True, 'cases': 478, 'RAW_and_PNG_files_exact': True,
            'upstream_readouts_also_checked_for_every_case': True,
            'current_runtime_only': True, 'numerical_tolerance': 0.}


def _forward(counter, runtime, key, net, input256, *, grad, context):
    torch = runtime.torch
    runtime.check('before ' + key)
    reservation = counter.reserve(key, **context)
    if grad:
        record = m.forward_candidate(net, input256)
    else:
        with torch.no_grad():
            record = m.forward_record(net, input256)
    torch.cuda.synchronize()
    counter.returned(reservation, finite=True)
    runtime.check('after ' + key)
    return record


def _query(counter, torch, names, params, scalar, label, *, retain_graph=False):
    reservation = counter.reserve('gradient_queries', purpose=label, tensors=72)
    gradients = torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=retain_graph,
                                    create_graph=False)
    counter.returned(reservation, completed=True)
    return dict(zip(names, gradients))


def _witness(writer, runtime, counter, protocol, cases, candidate, original, r2, np):
    torch = runtime.torch
    before = m.prerequisite_context(candidate, (original, r2))
    names = sorted(n for n, p in candidate.named_parameters() if p.requires_grad)
    named = dict(candidate.named_parameters()); params = [named[n] for n in names]
    results = {}; fingerprints = {}; calls = 0
    def obtain(label, key, net, identifier, grad):
        nonlocal calls
        tensor, _ = _input_tensor(runtime.root, cases[identifier], torch, np)
        record = _forward(counter, runtime, key, net, tensor, grad=grad, context={'witness': label, 'id': identifier})
        u, raw, _ = _arrays(record, np)
        writer.array(OUTPUTS + '/witness/' + label + '_U.npy', u, np)
        writer.array(OUTPUTS + '/witness/' + label + '_RAW.npy', raw, np)
        fingerprints[label] = {'id': identifier, 'upstream': m.readout_fingerprints(record),
                               'clipping': m.clipping_statistics(record), 'U_readout_parity': record.U_readout_parity}
        calls += 1
        return record
    def query(label, record, role, target, footprint=None, component=False):
        divided, terms, scalar = m.slot_terms(record, role=role, target256=target, footprint=footprint)
        whole = _query(counter, torch, names, params, divided, label, retain_graph=component)
        summary = m.summarize_gradients(names, list(whole.values()), np=np)
        m.validate_gradient_summary(summary, allow_all_zero=role == 'dark')
        results[label] = {'scalar': float(scalar.detach().item()),
            'divided_scalar': float(divided.detach().item()),
            'unweighted_terms': {k: float(v.detach().item()) for k, v in terms.items()}, 'gradients': summary}
        if component:
            gradients = {}
            for index, (term, value) in enumerate(terms.items()):
                gradients[term] = _query(counter, torch, names, params, value,
                    GRADIENT_QUERY_NAMES[index + 1], retain_graph=index == 0)
            results['component_sum'] = m.component_sum_witness(whole, gradients, role=role, np=np)
            del gradients
        del whole
        return results[label]
    with stage_deadline(STAGES['witness']):
        identifier = protocol['witness_cases']['clear']
        first = obtain('parity_candidate', 'parity_candidate_forwards', candidate, identifier, False)
        second = obtain('parity_R2', 'parity_R2_forwards', r2, identifier, False)
        results['initial_parity'] = m.validate_initial_parity(first, second)
        for label, record in (('parity_candidate', first), ('parity_R2', second)):
            writer.array(OUTPUTS + '/witness/' + label + '_native.npy',
                         np.ascontiguousarray(record.native.detach().cpu().numpy()), np)
        del first, second
        for label, query_name in (('rows', GRADIENT_QUERY_NAMES[0]), ('local_mosaic', GRADIENT_QUERY_NAMES[3]),
                                  ('local_blur', GRADIENT_QUERY_NAMES[4])):
            identifier = protocol['witness_cases'][label]; row = cases[identifier]
            record = obtain(label, 'filter_witness_forwards', candidate, identifier, True)
            query(query_name, record, 'filter', _clean_tensor(runtime.root, row, torch, np),
                  row.get('local_rectangle_xyxy_half_open'), component=label == 'rows')
            del record
        identifier = protocol['witness_cases']['clear']
        record = obtain('clear_candidate', 'clear_witness_forwards', candidate, identifier, True)
        teacher = obtain('clear_original', 'clear_witness_forwards', original, identifier, False)
        query(GRADIENT_QUERY_NAMES[5], record, 'clear', m.TeacherTarget(teacher.unclipped256.detach().clone(), 'original'))
        del record, teacher
        identifier = protocol['witness_cases']['dark']
        record = obtain('dark_candidate', 'dark_witness_forwards', candidate, identifier, True)
        teacher = obtain('dark_R2', 'dark_witness_forwards', r2, identifier, False)
        dark_result = query(GRADIENT_QUERY_NAMES[6], record, 'dark', m.TeacherTarget(teacher.unclipped256.detach().clone(), 'r2'))
        require(dark_result['scalar'] == 0., 'Initially identical same-input R2 anchor must be exactly zero')
        del record, teacher
    require(calls == 9 and counter.values['gradient_queries'] == 7, 'Exact9forward7query witness census differs')
    require(all(p.grad is None for p in candidate.parameters()), 'Witness query polluted grad fields')
    results['restoration'] = m.validate_prerequisite_restoration(before, candidate, (original, r2))
    results.update(complete=True, forward_calls=9, gradient_queries=7, optimizer_updates=0,
                   readouts=fingerprints, source_bound_gradients_not_independent_replay=True)
    writer.json(OUTPUTS + '/witness/receipt.json', results)
    return results


class BoundedCheckpointStream:
    """Finite per-write cap before bytes enter a Torch ZIP serialization stream."""
    def __init__(self, stream, writer, cap):
        require(type(cap) is int and cap > 0, 'Finite checkpoint byte cap required')
        self.stream, self.writer, self.cap, self.written = stream, writer, cap, 0

    def write(self, payload):
        amount = len(payload)
        require(self.written + amount <= self.cap and
                self.writer.bytes + self.written + amount <= t.LIMITS['payload_bytes'],
                'Checkpoint serialization byte cap reached before write')
        require(shutil.disk_usage(self.writer.root).free >= t.LIMITS['reserve_free_bytes'] + amount,
                'Checkpoint write would cross disk reserve')
        count = self.stream.write(payload)
        require(count == amount, 'Short checkpoint stream write')
        self.written += count
        return count

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.written


def checkpoint_metadata(protocol_sha, progress, counts):
    """Pure metadata portion bound inside every saved physical model state."""
    require(type(counts) is dict and set(counts) == set(call_caps()) and
            all(type(value) is int and 0 <= value <= call_caps()[key] for key,value in counts.items()),
            'Exact finite actual checkpoint counts required')
    return {'format': 'codeformer-filter-fusion-training-state-v1', 'protocol_sha256': protocol_sha,
        'progress': dict(progress), 'counts': dict(counts), 'counter_caps': call_caps(),
        'automatic_resume_safe': False, 'gradient_buffers_not_serialized': True,
        'partial_or_inflight_checkpoint_not_a_resumable_full_autograd_state': True, 'model_qualified': False}


def _checkpoint(writer, runtime, protocol_sha, candidate, optimizer, sampler, progress, name, counter, *, recovery=False):
    """One retained physical state. A failed temporary is never called complete."""
    torch = runtime.torch
    if not recovery:
        runtime.check('before checkpoint')
    cap = t.LIMITS['initial_checkpoint_bytes'] if name == t.CHECKPOINT_PATHS[0] else t.LIMITS['trained_checkpoint_bytes']
    temporary = name + '.tmp-' + uuid.uuid4().hex
    path = writer.reserve(temporary, cap, temporary=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**checkpoint_metadata(protocol_sha, progress, counter.values),
        'model': m.cpu_state(candidate), 'optimizer': None if optimizer is None else optimizer.state_dict(),
        'sampler': None if sampler is None else sampler.state(),
        'CPU_RNG': torch.get_rng_state().clone(), 'CUDA_RNG': torch.cuda.get_rng_state(0).clone()}
    try:
        with path.open('xb') as stream:
            bounded = BoundedCheckpointStream(stream, writer, cap)
            torch.save(state, bounded); bounded.flush(); os.fsync(stream.fileno())
        require(path.stat().st_size <= cap, 'Physical checkpoint serialization cap exceeded')
        final = writer.path(name)
        require(not final.exists(), 'Preserve earlier checkpoint; no overwrite')
        path.rename(final)
        writer.bytes += final.stat().st_size; writer.files += 1
        pin = {'path': name, **t.file_pin(final)}
        return pin, True
    except BaseException:
        if path.exists():
            size = path.stat().st_size; writer.bytes += size; writer.files += 1
            if progress.get('state_checkpoint') is not None:
                progress['prior_checkpoint_retained_not_current_state'] = progress['state_checkpoint']
            progress['state_checkpoint'] = {'path': temporary, **t.file_pin(path)}
            progress['state_checkpoint_complete'] = False
            progress['checkpoint_incomplete'] = True
            progress['automatic_resume_safe'] = False
        raise
    finally:
        del state


def _metadata_value(value):
    """Finite loss readouts use explicit detached scalars, never tensors in JSON."""
    return {name: float(term.detach().item()) for name, term in value.items()}


def needs_stopped_checkpoint(progress):
    """A final151 quality stop reuses only its actual completed physical state."""
    pin = progress.get('state_checkpoint')
    if (type(pin) is dict and t.checkpoint_kind(pin.get('path', '')) == 'temporary' and
            progress.get('state_checkpoint_complete') is False):
        # Preserve the one partial stream, without creating a second possible
        # partial448MiB stream. Neither it nor a prior state is called current.
        return False
    return not (progress.get('completed_training_updates') == FINAL_UPDATE and
                progress.get('partial_microbatches') == 0 and progress.get('optimizer_inflight') is False and
                progress.get('state_checkpoint_complete') is True and type(pin) is dict and
                pin.get('path') == OUTPUTS + f'/snapshot_{FINAL_UPDATE:04d}/checkpoint.pt')


def _train_step(writer, runtime, counter, update, cases, candidate, optimizer, teachers, frozen_digest, np):
    torch = runtime.torch; optimizer.zero_grad(set_to_none=True)
    progress = update['progress']; committed = update['preview']; losses = []
    lr = m.learning_rate(committed['update'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    for slot in committed['slots']:
        row = cases[slot['id']]; role = 'filter' if slot['role'] in ('filter_a', 'filter_b') else slot['role']
        input256, _ = _input_tensor(runtime.root, row, torch, np)
        record = _forward(counter, runtime, 'candidate_training_forwards', candidate, input256,
            grad=True, context={'update': committed['update'], 'slot': slot['slot'], 'role': role, 'id': row['id']})
        target = _clean_tensor(runtime.root, row, torch, np) if role == 'filter' else teachers[(role, row['id'])]['target']
        divided, terms, scalar = m.slot_terms(record, role=role, target256=target,
            footprint=row.get('local_rectangle_xyxy_half_open') if role == 'filter' else None)
        metadata = {'update': committed['update'], 'slot': slot['slot'], 'role': role, 'id': row['id'],
            'LR': lr, 'scalar': float(scalar.detach().item()), 'divided_scalar': float(divided.detach().item()),
            'unweighted_terms': _metadata_value(terms), 'clipping': m.clipping_statistics(record),
            'U_readout_parity': record.U_readout_parity,
            'upstream': m.readout_fingerprints(record), 'input': row['input'],
            'target': row.get('target256') if role == 'filter' else teachers[(role, row['id'])]['pin'],
            'footprint': row.get('local_rectangle_xyxy_half_open') if role == 'filter' else None,
            'padding_duplicate': slot['padding_duplicate']}
        reservation = counter.reserve('backward_calls', update=committed['update'], slot=slot['slot'])
        divided.backward(); torch.cuda.synchronize()
        counter.returned(reservation, completed=True)
        progress['partial_microbatches'] += 1
        writer.json(OUTPUTS + f'/training/u{committed["update"]:04d}_s{slot["slot"]}.json', metadata)
        losses.append(metadata); del record, divided, terms, scalar, target, input256
    named = dict(candidate.named_parameters()); names = sorted(n for n,p in named.items() if p.requires_grad)
    gradients = [named[n].grad for n in names]
    summary = m.summarize_gradients(names, gradients, np=np)
    m.validate_gradient_summary(summary)
    require(all(p.grad is None for n,p in named.items() if not p.requires_grad), 'Frozen candidate acquired a gradient')
    old = {name: named[name].detach().clone() for name in names}
    reservation = counter.reserve('optimizer_step_attempts', update=committed['update'])
    progress['optimizer_inflight'] = True
    optimizer.step(); torch.cuda.synchronize()
    counter.returned(reservation, returned=True)
    require(all(torch.isfinite(p).all().item() for p in candidate.parameters()), 'Nonfinite post-Adam model state')
    require(m.frozen_state_digest(candidate) == frozen_digest, 'One of443 frozen candidate tensors changed')
    changes = m.summarize_gradients(names, [named[n].detach() - old[n] for n in names], np=np)
    require(all(math.isfinite(r['l2']) for r in changes['sites'].values()), 'Nonfinite site step norm')
    del old
    optimizer.zero_grad(set_to_none=True)
    update['sampler'].commit(committed)
    progress['completed_training_updates'] += 1; progress['partial_microbatches'] = 0; progress['optimizer_inflight'] = False
    writer.json(OUTPUTS + f'/training/u{committed["update"]:04d}_commit.json',
        {'complete': True, 'update': committed['update'], 'LR': lr, 'gradients': summary, 'step_changes': changes,
         'sampler_state_before_sha256': committed['state_before_sha256'],
         'sampler_state_after_sha256': committed['state_after_sha256'], 'slot_scalars': [r['scalar'] for r in losses]})


def run(root, protocol_sha):
    """One finite manually invoked run. Exceptions retain actual failed state."""
    root = Path(root).absolute()
    require(platform.system() == 'Linux' and root.resolve(strict=True) ==
            (Path.home() / 'forensic-dgp' / t.VM_NAME).resolve(strict=True) and bool(os.environ.get('TMUX')),
            'Manual exact Linux-home L4 tmux root required')
    require(not (root / OUTPUTS).exists(), 'Preserve prior/partial run; no automatic resume or retry')
    with stage_deadline(STAGES['preflight']):
        verified = t.verify_transfer(root, protocol_sha); protocol = verified['protocol']; validate_plan(protocol)
        cases, schedule, evaluation = _local_metadata(root, protocol)
        require(shutil.disk_usage(root).free >= t.LIMITS['initial_free_bytes'], 'At least20GiB post-install free space required')
        idle = idle_inventory()
    lock = t.no_links(root / t.LOCK_NAME)
    with lock.open('xb') as stream:
        stream.write(t.json_bytes({'pid': os.getpid(), 'protocol_sha256': protocol_sha, 'manual_run': True})); stream.flush(); os.fsync(stream.fileno())
    writer = Writer(root, protocol); runtime = Runtime(root, writer)
    counter = Counters(protocol['call_caps'], lambda row: writer.append(OUTPUTS + '/operations.jsonl', row))
    progress = {'completed_training_updates': 0, 'partial_microbatches': 0, 'optimizer_inflight': False,
                'state_checkpoint': None, 'state_checkpoint_complete': None}
    candidate = original = r2 = optimizer = sampler = None
    try:
        with stage_deadline(STAGES['worker']):
            runtime.check('before Torch import')
            with stage_deadline(STAGES['initialization']):
                require('torch' not in sys.modules, 'Fresh worker must set deterministic CUDA environment before Torch import')
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                torch = m.activate_runtime(root, protocol_sha); runtime.torch = torch
                import numpy as np
                from PIL import Image
                from . import metrics as metrics_module
                torch.set_num_threads(4); torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
                torch.manual_seed(d.SEED); torch.cuda.manual_seed_all(d.SEED)
                writer.json(OUTPUTS + '/runtime.json', {'complete': True, 'idle': idle, 'torch': torch.__version__,
                    'numpy': np.__version__, 'GPU': torch.cuda.get_device_name(0),
                    'FP32': True, 'TF32': False, 'deterministic_algorithms': True,
                    'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                    'manual_only': True})
            baselines = {}
            for alias in ('original', 'parent', 'r2'):
                with stage_deadline(STAGES['snapshot']):
                    net, receipt = m.load_frozen_baseline(alias)
                    writer.json(OUTPUTS + '/baseline/' + alias + '/model.json', receipt)
                    bank = {}
                    for index, row in enumerate(evaluation, 1):
                        tensor, _ = _input_tensor(root, row, torch, np)
                        record = _forward(counter, runtime, {'original': 'original_baseline_forwards', 'parent': 'parent_baseline_forwards',
                            'r2': 'R2_baseline_forwards'}[alias], net, tensor, grad=False, context={'id': row['id'], 'alias': alias})
                        bank[row['id']] = _save_output(writer, OUTPUTS + '/baseline/' + alias + '/' + row['id'], row,
                            record, metrics_module, root, np, Image)
                        del record, tensor
                        if index % 50 == 0 or index == len(evaluation):
                            print(json.dumps({'baseline': alias, 'outputs': index, 'of': len(evaluation)}), flush=True)
                    baselines[alias] = bank
                    del net; torch.cuda.empty_cache()
            with stage_deadline(STAGES['initialization']):
                candidate, original, r2, census, receipt = m.load_models()
                frozen_digest = m.frozen_state_digest(candidate)
                writer.json(OUTPUTS + '/model_initialization.json', receipt)
                teachers = {}
                for role, net, key, kind in (('clear', original, 'original_clear_teacher_forwards', 'original'),
                                              ('dark', r2, 'R2_dark_teacher_forwards', 'r2')):
                    selected = sorted((r for r in cases.values() if (r['profile'] == 'clear') == (role == 'clear') and
                                       (role == 'clear' or r['id'] in {s['id'] for u in schedule['updates'] for s in u['slots'] if s['role'] == 'dark'})),
                                      key=lambda r: r['id'])
                    require(len(selected) == TEACHER_SOURCES, 'All55 unique role behavior teacher targets required')
                    for index, row in enumerate(selected, 1):
                        tensor, _ = _input_tensor(root, row, torch, np)
                        record = _forward(counter, runtime, key, net, tensor, grad=False, context={'id': row['id'], 'role': role})
                        u, raw, png = _arrays(record, np)
                        stem = OUTPUTS + '/teacher/' + role + '/' + row['id']
                        pin = writer.array(stem + '_U.npy', u, np)
                        rawpin = writer.array(stem + '_RAW.npy', raw, np)
                        pngpin = writer.png(stem + '_PNG.png', png, np, Image)
                        teachers[(role, row['id'])] = {'target': m.TeacherTarget(record.unclipped256.detach().clone(), kind),
                            'pin': pin, 'raw': raw, 'png': png, 'raw_pin': rawpin, 'png_pin': pngpin,
                            'U_readout_parity': record.U_readout_parity}
                        del record, tensor
                        if index % 10 == 0 or index == len(selected):
                            print(json.dumps({'teacher': role, 'targets': index, 'of': len(selected)}), flush=True)
                writer.json(OUTPUTS + '/teacher/manifest.json', {'complete': True,
                    'targets': [{'role': role, 'id': identifier, 'U': value['pin'],
                        'RAW': value['raw_pin'], 'PNG': value['png_pin'], 'U_readout_parity': value['U_readout_parity']}
                        for (role, identifier), value in sorted(teachers.items())],
                    'unique_teacher_forwards': 2 * TEACHER_SOURCES, 'reuse_does_not_add_neural_calls': True,
                    'clamp_before_resize_evidence_not_derived_from_U': True})
            _witness(writer, runtime, counter, protocol, cases, candidate, original, r2, np)
            optimizer = torch.optim.Adam(m.optimizer_groups(candidate), betas=(.9, .99), eps=1e-8, weight_decay=0.)
            require(len(optimizer.state) == 0 and len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]['params']) == 72,
                    'Fresh empty exact72-tensor Adam required')
            sampler = schedule_data.FiniteSampler(schedule)
            initial_bank = None; initial_guards = None
            def snapshot(update):
                nonlocal initial_bank
                with stage_deadline(STAGES['snapshot']):
                    name = OUTPUTS + f'/snapshot_{update:04d}/checkpoint.pt'
                    pin, complete = _checkpoint(writer, runtime, protocol_sha, candidate, optimizer, sampler, progress, name, counter)
                    progress.update(state_checkpoint=pin, state_checkpoint_complete=complete)
                    bank = {}
                    for index, row in enumerate(evaluation, 1):
                        tensor, _ = _input_tensor(root, row, torch, np)
                        record = _forward(counter, runtime, 'candidate_snapshot_forwards', candidate, tensor,
                            grad=False, context={'snapshot': update, 'id': row['id']})
                        bank[row['id']] = _save_output(writer, OUTPUTS + f'/snapshot_{update:04d}/' + row['id'], row,
                            record, metrics_module, root, np, Image)
                        require(m.readout_fingerprints(record) == baselines['r2'][row['id']]['upstream'],
                                'Fixed upstream logits/LQ/tokens changed at candidate snapshot')
                        del record, tensor
                        if index % 50 == 0 or index == len(evaluation):
                            print(json.dumps({'snapshot': update, 'outputs': index, 'of': len(evaluation)}), flush=True)
                    if update == 0:
                        initial_bank = bank
                        initialization_parity = initial_output_parity(bank, baselines['r2'])
                    dev = {k:r for k,r in bank.items() if r.get('cohort') == 'historical_DEV406'}
                    dev_control = {arm: {k:r for k,r in b.items() if r.get('cohort') == 'historical_DEV406'}
                                   for arm,b in baselines.items()}
                    initial_dev = {k:r for k,r in initial_bank.items() if r.get('cohort') == 'historical_DEV406'}
                    gates = {'preservation': metrics_module.preservation(dev, dev_control['original']),
                        'parent_dark_retention': metrics_module.dark_retention(dev, dev_control['parent']),
                        'R2_dark_retention': metrics_module.dark_retention(dev, dev_control['r2']),
                        'interim': metrics_module.interim(dev, dev_control['original'], initial_dev),
                        'filters': metrics_module.filter_summary(list(bank.values()))}
                    if update == 0:
                        gates['initial_R2_full478_parity'] = initialization_parity
                    writer.json(OUTPUTS + f'/snapshot_{update:04d}/gates.json', gates)
                    if update:
                        require(gates['interim']['pass'] and gates['R2_dark_retention']['pass'],
                                'New/worsened preservation or R2 dark retention failure; retain actual stop')
                    return bank, gates
            def guards(update):
                nonlocal initial_guards
                bank = {}
                for guard in protocol['guard_cases']:
                    row = cases[guard['id']]
                    tensor, _ = _input_tensor(root, row, torch, np)
                    record = _forward(counter, runtime, 'candidate_TRAIN_guard_forwards', candidate, tensor,
                        grad=False, context={'TRAIN_guard': update, 'id': row['id'], 'role': guard['role']})
                    measured = _save_output(writer, OUTPUTS + f'/guard_{update:04d}/' + guard['role'] + '/' + row['id'], row,
                        record, metrics_module, root, np, Image)
                    measured['guard_role'] = measured['role'] = guard['role']
                    if guard['role'] in ('clear', 'dark'):
                        teacher = teachers[(guard['role'], row['id'])]
                        _, raw, png = _arrays(record, np)
                        measured['behavior_distance'] = {
                            'raw': float(np.abs(raw.astype(np.float64) - teacher['raw'].astype(np.float64)).mean()),
                            'native_png': float(np.abs((png.astype(np.float32) / np.float32(255)).astype(np.float64) -
                                                      (teacher['png'].astype(np.float32) / np.float32(255)).astype(np.float64)).mean())}
                        measured['behavior_teacher'] = {'kind': 'original' if guard['role'] == 'clear' else 'r2',
                            'RAW': teacher['raw_pin'], 'PNG': teacher['png_pin'], 'U': teacher['pin']}
                    bank[row['id']] = measured
                    del tensor, record
                writer.json(OUTPUTS + f'/guard_{update:04d}/bank.json', {'complete': True,
                    'cases': list(bank.values()), 'TRAIN_probes_not_final_or_holdout': True,
                    'behavior_teacher_reference_arrays_retained': True})
                if update == 0:
                    initial_guards = bank
                gate = metrics_module.guard_interim(bank, initial_guards)
                writer.json(OUTPUTS + f'/guard_{update:04d}/gates.json', gate)
                require(update == 0 or gate['pass'], 'TRAIN guard preservation/retention requirement failed; retain actual stop')
            snapshot(0); guards(0)
            with stage_deadline(STAGES['training']):
                for _ in range(FINAL_UPDATE):
                    preview = sampler.preview()
                    _train_step(writer, runtime, counter, {'preview': preview, 'sampler': sampler, 'progress': progress},
                                cases, candidate, optimizer, teachers, frozen_digest, np)
                    current = progress['completed_training_updates']
                    print(json.dumps({'update': current, 'of': FINAL_UPDATE, 'LR': m.learning_rate(current)}), flush=True)
                    if current == 20:
                        guards(current)
                    if current in (50, FINAL_UPDATE):
                        _, final_gates = snapshot(current)
            require(final_gates['preservation']['pass'] and final_gates['parent_dark_retention']['pass'],
                    'Final original preservation or parent dark-retention quality requirement failed; retain actual final state')
            require(counter.values == protocol['call_caps'] and sampler.state()['committed_updates'] == FINAL_UPDATE,
                    'Completed finite run operation/sampler census differs')
            writer.json(OUTPUTS + '/run_results.json', {'complete': True, 'protocol_sha256': protocol_sha,
                **progress, 'counts': counter.values, 'final_gates': final_gates, 'model_qualified': False,
                'training_completion_not_acceptance': True, 'automatic_app_promotion': False})
    except BaseException as error:
        if runtime.torch is not None and candidate is not None:
            try:
                # Reuse an actual completed final151 state, or retain one
                # incomplete temporary honestly without a second serialization.
                if needs_stopped_checkpoint(progress):
                    with stage_deadline(20):
                        pin, complete = _checkpoint(writer, runtime, protocol_sha, candidate, optimizer, sampler,
                                                    progress, OUTPUTS + '/stopped_checkpoint.pt', counter, recovery=True)
                    progress.update(state_checkpoint=pin, state_checkpoint_complete=complete)
            except BaseException as checkpoint_error:
                progress['checkpoint_failure'] = {'type': type(checkpoint_error).__name__, 'message': str(checkpoint_error)}
                if progress.get('state_checkpoint') and not t.checkpoint_kind(progress['state_checkpoint']['path']) == 'temporary':
                    progress['prior_checkpoint_retained_not_current_state'] = progress['state_checkpoint']
                    progress['state_checkpoint'] = None; progress['state_checkpoint_complete'] = False
        writer.json(OUTPUTS + '/failure.json', {'complete': True, 'failure': True, 'protocol_sha256': protocol_sha,
            **progress, 'counts': counter.values, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'training_success_not_implied': True, 'model_qualified': False})
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--verify-transfer', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.verify_transfer:
        receipt = t.verify_transfer(args.root, args.protocol_sha); validate_plan(receipt['protocol'])
        _local_metadata(args.root, receipt['protocol'])
        print(json.dumps({k: v for k,v in receipt.items() if k != 'protocol'}), flush=True)
    elif args.export:
        print(json.dumps(t.export_results(args.root, args.protocol_sha)), flush=True)
    else:
        run(args.root, args.protocol_sha)


if __name__ == '__main__':
    main()
