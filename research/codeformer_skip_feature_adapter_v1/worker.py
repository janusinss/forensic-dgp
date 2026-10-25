"""Finite manual-L4 clean-skip residual-adapter pilot; import never trains.

All neural/gradient/optimizer work is reachable only through the exact manually
invoked Linux/L4/tmux --run path after frozen transfer/data/resource checks.
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
import random
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
from . import model as m
from . import transport as t

OUTPUTS = 'outputs/skip_feature_adapter_v1'
FINAL_UPDATE = 110
TEACHER_SOURCES = 55
STAGES = {'preflight':300,'initialization':900,'witness':180,'training':2400,'snapshot':900,'worker':4800}
SITES = ('32','64','128','256')


def call_caps():
    return {'original_baseline_forwards':478,'parent_baseline_forwards':478,'R2_baseline_forwards':478,
        'R2_clean_feature_teacher_forwards':55,'original_clear_teacher_forwards':55,'R2_dark_teacher_forwards':55,
        'candidate_snapshot_forwards':1434,'candidate_training_forwards':440,'candidate_TRAIN_guard_forwards':96,
        'candidate_witness_forwards':3,'R2_auxiliary_forwards':18,'candidate_auxiliary_forwards':54,
        'backward_calls':440,'optimizer_step_attempts':110,'gradient_queries':4}


def require(value, message):
    if not value:
        raise ValueError(message)


class Counters:
    """Reserve durable operation evidence BEFORE each actual call or attempt."""
    def __init__(self, caps, append):
        require(caps == call_caps(), 'Exact finite source counter caps required')
        self.caps = dict(caps); self.values = dict.fromkeys(caps, 0)
        self.append = append; self.sequence = 0; self.pending = {}
        self.returned_values = dict.fromkeys(caps, 0)

    def reserve(self, kind, **context):
        require(kind in self.caps and self.values[kind] < self.caps[kind], 'Finite operation cap exceeded: ' + str(kind))
        self.values[kind] += 1; self.sequence += 1
        row = {'event': 'reserved', 'sequence': self.sequence, 'kind': kind,
               'count': self.values[kind], 'context': context}
        self.pending[self.sequence] = kind
        self.append(row)
        return self.sequence

    def returned(self, reservation, **context):
        require(type(reservation) is int and reservation in self.pending,
                'Actual reserved operation required')
        self.returned_values[self.pending[reservation]] += 1
        self.append({'event': 'returned', 'reservation': reservation,
                     'kind': self.pending[reservation], 'context': context})
        del self.pending[reservation]


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
        require(elapsed <= STAGES['worker'] and memory <= 10 * t.GiB and peak_memory <= 10 * t.GiB and
                gpu <= 18 * t.GiB and free >= t.LIMITS['reserve_free_bytes'], 'Finite4800s/10GiB RSS/18GiB CUDA/disk bound exceeded')
        return row


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
    reservation = counter.reserve('gradient_queries', purpose=label, tensors=24)
    gradients = torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=retain_graph,
                                    create_graph=False)
    counter.returned(reservation, completed=True)
    return dict(zip(names, gradients))


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


def _metadata_value(value):
    """Finite loss readouts use explicit detached scalars, never tensors in JSON."""
    return {name: float(term.detach().item()) for name, term in value.items()}


def validate_plan(protocol):
    require(protocol.get('format') == 'codeformer-skip-feature-adapter-finite-protocol-v1' and
            protocol.get('vm_name') == 'cctv_codeformer_skip_feature_adapter_v1_vm' and
            protocol.get('outputs') == OUTPUTS, 'Exact new finite adapter namespace required')
    require(protocol.get('call_caps') == call_caps() and
            sum(v for k,v in call_caps().items() if k.endswith('_forwards')) == 3644,
            'Exact3644-forward/440-backward/110-step/4-query ledger required')
    require(protocol.get('max_updates') == 110 and protocol.get('snapshots') == [0,55,110] and
            protocol.get('guard_snapshots') == [0,20,55,110], 'Finite state/readout schedule differs')
    require(protocol.get('automatic_resume') is False and protocol.get('automatic_training_launch') is False,
            'Manual finite no-resume execution required')
    guards = protocol.get('guard_cases'); witnesses = protocol.get('witness_cases')
    require(type(guards) is list and len(guards) == 24 and
            {role:sum(r.get('role') == role for r in guards) for role in ('filter','clear','dark')} ==
            {'filter':8,'clear':8,'dark':8}, 'Exact24 first-eight-source TRAIN guards required')
    require(type(witnesses) is dict and set(witnesses) == {'filter','clear','dark'}, 'Three declared role witnesses required')
    require(len({(g['role'],g['id']) for g in guards}) == 24, 'Distinct guard role/ID joins required')
    return {'complete':True,'planned_top_level_forwards':3644,'optimizer_updates':110,
            'gradient_queries':4,'backward_calls':440,'training_completion_not_acceptance':True}


def _local_metadata(root, protocol):
    cases, schedule, evaluation, auxiliary = d._local_metadata(root, protocol)
    require(type(cases) is dict and len(cases) == 165 and all(r['split'] == 'TRAIN' for r in cases.values()),
            'Exactly165 admitted TRAIN role observations required')
    require(len(schedule['anchor_source_order']) == len(set(schedule['anchor_source_order'])) == 55 and
            len(schedule['updates']) == 110 and [u['update'] for u in schedule['updates']] == list(range(1,111)),
            'Exact55-source110-update deterministic schedule required')
    for update in schedule['updates']:
        require([s['slot'] for s in update['slots']] == [1,2,3,4] and
                [s['role'] for s in update['slots']] == ['filter','filter','clear','dark'],
                'Two blur/one clear/one dark accumulation order required')
        require(all(s['id'] in cases and cases[s['id']]['reference_id'] == s['reference_id'] for s in update['slots']),
                'Complete schedule observation/source joins required')
    require(len(evaluation) == len({r['id'] for r in evaluation}) == 478 and
            sum(r.get('cohort') == 'historical_DEV406' for r in evaluation) == 406 and
            sum(r.get('cohort') == 'retained_filter72_mixed_roles' for r in evaluation) == 72,
            'All406 historical DEV and72 separately labeled filter slots required')
    require(len(auxiliary) == len({r['id'] for r in auxiliary}) == 18,
            'Exactly18 frozen photographic auxiliary cases required')
    for row in protocol['guard_cases']:
        require(row['id'] in cases, 'Guard observation not in explicitly admitted TRAIN bundle')
    require(all(identifier in cases for identifier in protocol['witness_cases'].values()), 'Witness not in TRAIN bundle')
    return cases,schedule,evaluation,auxiliary




def planned_output_names(protocol,evaluation,cases,auxiliary=None):
    auxiliary = auxiliary or []
    require(len(evaluation)==478 and len(cases)==165 and len(auxiliary)==18,'Complete finite artifact cohort required')
    names={OUTPUTS+'/'+n for n in ('runtime.json','model_initialization.json','operations.jsonl','resource.jsonl',
        'teacher/manifest.json','witness/receipt.json','run_results.json','failure.json','stopped_checkpoint.pt','export_manifest.json')}
    for alias in ('original','parent','r2'):
        names.add(OUTPUTS+'/baseline/'+alias+'/model.json')
        for row in evaluation:
            names.update(OUTPUTS+'/baseline/'+alias+'/'+row['id']+s for s in ('.npy','.png','.json'))
    for update in (0,55,110):
        stem=OUTPUTS+f'/snapshot_{update:04d}/';names.update((stem+'checkpoint.pt',stem+'gates.json'))
        for row in evaluation:
            names.update(stem+row['id']+s for s in ('.npy','.png','.json'))
    for alias in ('r2','candidate_0000','candidate_0055','candidate_0110'):
        for row in auxiliary:
            names.update(OUTPUTS+'/auxiliary/'+alias+'/'+row['id']+s for s in ('.npy','.png','.json'))
        names.add(OUTPUTS+'/auxiliary/'+alias+'/bank.json')
    for update in (0,20,55,110):
        stem=OUTPUTS+f'/guard_{update:04d}/';names.update((stem+'bank.json',stem+'gates.json'))
        for guard in protocol['guard_cases']:
            names.update(stem+guard['role']+'/'+guard['id']+s for s in ('.npy','.png','.json'))
    refs=sorted({r['reference_id'] for r in cases.values()})
    require(len(refs)==55,'All55 clean-feature teacher sources required')
    for ref in refs:
        names.add(OUTPUTS+'/teacher/features/'+ref+'/scales.json')
        names.update(OUTPUTS+'/teacher/features/'+ref+'/encoder'+site+'.npy' for site in SITES)
    for row in cases.values():
        if row['profile']=='clear' or row['profile']==d.DARK_LINEAR_PROFILE:
            role='clear' if row['profile']=='clear' else 'dark'
            names.update(OUTPUTS+'/teacher/'+role+'/'+row['id']+s for s in ('_U.npy','_RAW.npy','_PNG.png'))
    for role in ('filter','clear','dark'):
        names.update(OUTPUTS+'/witness/'+role+s for s in ('_U.npy','_RAW.npy'))
    for update in range(1,111):
        names.update(OUTPUTS+f'/training/u{update:04d}_s{slot}.json' for slot in (1,2,3,4))
        names.add(OUTPUTS+f'/training/u{update:04d}_commit.json')
    require(len(names)==len({n.casefold() for n in names}),'Portable artifact name collision')
    for name in names:t.canonical(name)
    return sorted(names)


def _checkpoint(writer,runtime,protocol_sha,candidate,optimizer,sampler,progress,name,counter,*,recovery=False):
    if not recovery:runtime.check('before checkpoint')
    cap=t.LIMITS['trained_checkpoint_bytes'];temporary=name+'.tmp-'+uuid.uuid4().hex
    path=writer.reserve(temporary,cap,temporary=True);path.parent.mkdir(parents=True,exist_ok=True)
    torch=runtime.torch
    import numpy as np
    numpy_rng=np.random.get_state()
    state={'format':'codeformer-skip-feature-adapter-training-state-v1','protocol_sha256':protocol_sha,
        'model':m.cpu_state(candidate),'adapter_state':m.adapter_state(candidate),
        'frozen_base_tensor_sha256':m.frozen_state_digest(candidate),
        'optimizer':None if optimizer is None else optimizer.state_dict(),
        'sampler':None if sampler is None else sampler.state(),'progress':dict(progress),
        'counts':dict(counter.values),'call_caps':call_caps(),
        'actual_return_counts':dict(counter.returned_values),'pending_reservations':dict(counter.pending),
        'CPU_RNG':torch.get_rng_state().clone(),'CUDA_RNG':torch.cuda.get_rng_state(0).clone(),
        'NumPy_RNG':{'algorithm':numpy_rng[0],'state':torch.from_numpy(numpy_rng[1].astype(np.int64)),
            'position':int(numpy_rng[2]),'has_gauss':int(numpy_rng[3]),'cached_gaussian':float(numpy_rng[4])},
        'python_RNG':random.getstate(),'automatic_resume_safe':False,'model_qualified':False,
        'gradient_buffers_not_serialized':True,'partial_state_is_not_resumable_full_autograd_state':True}
    try:
        with path.open('xb') as stream:
            bounded=BoundedCheckpointStream(stream,writer,cap)
            torch.save(state,bounded);bounded.flush();os.fsync(stream.fileno())
        final=writer.path(name);require(not final.exists(),'Preserve prior checkpoint')
        path.rename(final);writer.bytes+=final.stat().st_size;writer.files+=1
        return {'path':name,**t.file_pin(final)},True
    except BaseException:
        if path.exists():
            writer.bytes+=path.stat().st_size;writer.files+=1
            progress['prior_checkpoint_retained_not_current_state']=progress.get('state_checkpoint')
            progress.update(state_checkpoint={'path':temporary,**t.file_pin(path)},state_checkpoint_complete=False,
                checkpoint_incomplete=True,automatic_resume_safe=False)
        raise
    finally:del state


def needs_stopped_checkpoint(progress):
    state=progress.get('state_checkpoint')
    if state and '.tmp-' in state['path'] and progress.get('state_checkpoint_complete') is False:return False
    return not (state and progress.get('state_checkpoint_complete') is True and
        progress['partial_microbatches']==0 and not progress['optimizer_inflight'] and
        state['path']==OUTPUTS+f'/snapshot_{progress["completed_training_updates"]:04d}/checkpoint.pt')


def _feature_targets(root, entry, torch, np):
    """One source at a time; never keep the 2.4GiB bank on the GPU."""
    result={}
    for site,value in entry['features'].items():
        path=t.regular(root,value['path']);require(t.file_pin(path)=={k:value[k] for k in ('bytes','sha256')},
            'Frozen same-source clean feature file changed')
        require(path.stat().st_size<=40*t.MiB,'Clean feature NPY cap exceeded')
        array=np.load(path,allow_pickle=False)
        channels={'32':256,'64':256,'128':128,'256':128}[site]
        require(array.shape==(1,channels,int(site),int(site)) and array.dtype==np.float32 and np.isfinite(array).all(),
            'Exact finite clean feature target schema required')
        result[site]=torch.from_numpy(np.ascontiguousarray(array))
    require(set(result)==set(SITES),'Four same-source clean feature targets required')
    return result,entry['scales']


def _witness(writer,runtime,counter,protocol,cases,candidate,original,r2,teachers,feature_bank,np):
    """Three forwards/four queries, including isolated image-loss connectivity."""
    torch=runtime.torch
    before=m.prerequisite_context(candidate,(original,r2))
    python_rng=random.getstate();np_rng=np.random.get_state()
    names=sorted(n for n,p in candidate.named_parameters() if p.requires_grad)
    require(len(names)==24,'Exact24 new adapter tensors required')
    named=dict(candidate.named_parameters());params=[named[n] for n in names]
    reports={}
    with stage_deadline(STAGES['witness']):
        for role in ('filter','clear','dark'):
            row=cases[protocol['witness_cases'][role]]
            value,_=_input_tensor(runtime.root,row,torch,np)
            record=_forward(counter,runtime,'candidate_witness_forwards',candidate,value,grad=True,
                context={'witness_role':role,'id':row['id']})
            clean=scales=None
            if role=='filter':
                clean,scales=_feature_targets(runtime.root,feature_bank[row['reference_id']],torch,np)
                target=_clean_tensor(runtime.root,row,torch,np)
            else:target=teachers[role,row['id']]['target']
            divided,terms,scalar=m.slot_terms(record,role='blur' if role=='filter' else role,
                target256=target,clean_features=clean,feature_scales=scales)
            gradients=_query(counter,torch,names,params,divided,role+'_total_divided',retain_graph=role=='filter')
            summary=m.summarize_gradients(names,list(gradients.values()),np=np)
            m.validate_gradient_summary(summary,allow_all_zero=role!='filter')
            result={'id':row['id'],'scalar':float(scalar.detach().item()),
                'divided_scalar':float(divided.detach().item()),'terms':_metadata_value(terms),'total_gradient':summary,
                'upstream':m.readout_fingerprints(record),'U_readout_parity':record.U_readout_parity,
                'feature_scales':scales,'ordinary_graph_not_inference_mode':True}
            del gradients
            if role=='filter':
                image=(terms['spatial_rgb_l1']+.25*terms['adjacent_residual_l1'])/4.
                gradients=_query(counter,torch,names,params,image,'filter_image_only_divided',retain_graph=False)
                image_summary=m.summarize_gradients(names,list(gradients.values()),np=np)
                m.validate_gradient_summary(image_summary,allow_all_zero=False)
                result['image_only_gradient']=image_summary
                result['image_only_divided_scalar']=float(image.detach().item())
                result['separate_image_path_empirically_connected_all4_last_projections']=True
                del gradients,image,image_summary
            elif role=='dark':
                require(result['scalar']==0. and all(s['l2']==0. for s in summary['sites'].values()),
                    'Initially identical frozen R2 dark behavior objective/gradients must be exactly zero')
            u,raw,_=_arrays(record,np)
            writer.array(OUTPUTS+'/witness/'+role+'_U.npy',u,np)
            writer.array(OUTPUTS+'/witness/'+role+'_RAW.npy',raw,np)
            reports[role]=result
            del record,divided,terms,scalar,target,clean,scales,value,summary
    require(counter.values['candidate_witness_forwards']==3 and counter.values['gradient_queries']==4,
        'Exactly3 forward4 query witness census required')
    require(random.getstate()==python_rng and all(p.grad is None for p in candidate.parameters()),
        'Witness changed Python RNG or parameter gradient fields')
    after_np=np.random.get_state()
    require(np_rng[0]==after_np[0] and np.array_equal(np_rng[1],after_np[1]) and np_rng[2:]==after_np[2:],
        'Witness changed NumPy RNG state')
    restored=m.validate_prerequisite_restoration(before,candidate,(original,r2))
    writer.json(OUTPUTS+'/witness/receipt.json',{'complete':True,'roles':reports,'restoration':restored,
        'candidate_forwards':3,'gradient_queries':4,'optimizer_updates':0,
        'image_only_query_separate_from_combined_feature_image_query':True,
        'zero_earlier_layers_from_zero_final_projection_expected':True,
        'source_bound_gradient_summaries_not_independent_replay':True})


def _train_step(writer,runtime,counter,preview,cases,candidate,optimizer,teachers,feature_bank,frozen_digest,sampler,progress,np):
    torch=runtime.torch;optimizer.zero_grad(set_to_none=True)
    update=preview['update'];lr=m.learning_rate(update)
    for group in optimizer.param_groups:group['lr']=lr
    losses=[]
    for slot in preview['slots']:
        role=slot['role'];row=cases[slot['id']]
        value,_=_input_tensor(runtime.root,row,torch,np)
        record=_forward(counter,runtime,'candidate_training_forwards',candidate,value,grad=True,
            context={'update':update,'slot':slot['slot'],'role':role,'id':row['id']})
        clean=scales=None
        if role=='filter':
            target=_clean_tensor(runtime.root,row,torch,np)
            clean,scales=_feature_targets(runtime.root,feature_bank[row['reference_id']],torch,np)
        else:target=teachers[role,row['id']]['target']
        divided,terms,scalar=m.slot_terms(record,role='blur' if role=='filter' else role,
            target256=target,clean_features=clean,feature_scales=scales)
        metadata={'update':update,'slot':slot['slot'],'id':row['id'],'reference_id':row['reference_id'],
            'role':role,'LR':lr,'scalar':float(scalar.detach().item()),'divided_scalar':float(divided.detach().item()),
            'terms':_metadata_value(terms),'feature_scales':scales,
            'clean_feature_files':feature_bank[row['reference_id']]['features'] if role=='filter' else None,
            'behavior_teacher':teachers[role,row['id']]['U'] if role!='filter' else None,
            'clipping':m.clipping_statistics(record),'U_readout_parity':record.U_readout_parity,
            'upstream':m.readout_fingerprints(record),'input':row['input'],
            'photo_target':row.get('target256') if role=='filter' else None}
        reservation=counter.reserve('backward_calls',update=update,slot=slot['slot'],role=role)
        divided.backward();torch.cuda.synchronize();counter.returned(reservation,completed=True)
        progress['partial_microbatches']+=1
        writer.json(OUTPUTS+f'/training/u{update:04d}_s{slot["slot"]}.json',metadata)
        losses.append(metadata)
        del record,divided,terms,scalar,target,clean,scales,value
    named=dict(candidate.named_parameters());names=sorted(n for n,p in named.items() if p.requires_grad)
    grads=[named[n].grad for n in names]
    summary=m.summarize_gradients(names,grads,np=np);m.validate_gradient_summary(summary,allow_all_zero=True)
    require(any(s['l2']>0 for s in summary['sites'].values()),'All accumulated adapter gradients are zero')
    require(all(p.grad is None for n,p in named.items() if not p.requires_grad),'Frozen R2 acquired gradient')
    clipped=torch.nn.utils.clip_grad_norm_([named[n] for n in names],max_norm=1.,error_if_nonfinite=True)
    before={name:named[name].detach().clone() for name in names}
    reservation=counter.reserve('optimizer_step_attempts',update=update)
    progress['optimizer_inflight']=True
    optimizer.step();torch.cuda.synchronize();counter.returned(reservation,completed=True)
    require(all(torch.isfinite(p).all().item() for p in candidate.parameters()),'Nonfinite post-Adam state')
    require(m.frozen_state_digest(candidate)==frozen_digest,'One of515 frozen base tensors changed')
    changes=m.summarize_gradients(names,[named[n].detach()-before[n] for n in names],np=np)
    optimizer.zero_grad(set_to_none=True);sampler.commit(preview)
    progress['completed_training_updates']+=1;progress['partial_microbatches']=0;progress['optimizer_inflight']=False
    writer.json(OUTPUTS+f'/training/u{update:04d}_commit.json',{'complete':True,'update':update,'LR':lr,
        'gradient_clip_max_norm':1.,'preclip_global_norm':float(clipped.detach().item()),'gradients':summary,
        'step_changes':changes,'sampler':sampler.state(),'slot_scalars':[x['scalar'] for x in losses],
        'all515_base_values_unchanged':True})


def run(root,protocol_sha):
    """One new manually invoked finite pilot; every stop retains actual state."""
    root=Path(root).absolute()
    require(platform.system()=='Linux' and bool(os.environ.get('TMUX')) and
        root.resolve(strict=True)==(Path.home()/'forensic-dgp'/'cctv_codeformer_skip_feature_adapter_v1_vm').resolve(strict=True),
        'Manual exact Linux-home NVIDIA L4 tmux root required')
    require(not (root/OUTPUTS).exists(),'Preserve prior or partial pilot; no retry/resume')
    with stage_deadline(STAGES['preflight']):
        verified=t.verify_transfer(root,protocol_sha)
        protocol=verified['protocol'] if 'protocol' in verified else verified
        validate_plan(protocol)
        cases,schedule,evaluation,auxiliary=_local_metadata(root,protocol)
        require(shutil.disk_usage(root).free>=t.LIMITS['initial_free_bytes'],
            'At least24GiB post-install free space required before any model call')
        idle=idle_inventory()
    lock=t.no_links(root/t.LOCK_NAME)
    with lock.open('xb') as stream:
        stream.write(t.json_bytes({'pid':os.getpid(),'protocol_sha256':protocol_sha,'manual_run':True}))
        stream.flush();os.fsync(stream.fileno())
    writer=Writer(root,protocol);runtime=Runtime(root,writer)
    counter=Counters(protocol['call_caps'],lambda row:writer.append(OUTPUTS+'/operations.jsonl',row))
    progress={'completed_training_updates':0,'partial_microbatches':0,'optimizer_inflight':False,
        'state_checkpoint':None,'state_checkpoint_complete':None,'automatic_resume_safe':False}
    candidate=original=r2=optimizer=sampler=None
    try:
        with stage_deadline(STAGES['worker']):
            runtime.check('before Torch import')
            with stage_deadline(STAGES['initialization']):
                require('torch' not in sys.modules,'Fresh worker must set deterministic CUDA environment before import')
                os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                torch=m.activate_runtime(root,protocol_sha);runtime.torch=torch
                import numpy as np
                from PIL import Image
                from . import metrics as metrics_module
                torch.set_num_threads(4);torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
                random.seed(d.SEED);np.random.seed(d.SEED);torch.manual_seed(d.SEED);torch.cuda.manual_seed_all(d.SEED)
                writer.json(OUTPUTS+'/runtime.json',{'complete':True,'idle':idle,'torch':torch.__version__,
                    'numpy':np.__version__,'GPU':torch.cuda.get_device_name(0),'FP32':True,'TF32':False,
                    'seed':d.SEED,'deterministic_algorithms':True,'manual_only':True,
                    'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']})
            baselines={};r2_auxiliary={}
            for alias in ('original','parent','r2'):
                with stage_deadline(STAGES['snapshot']):
                    net,receipt=m.load_frozen_baseline(alias)
                    writer.json(OUTPUTS+'/baseline/'+alias+'/model.json',receipt)
                    bank={}
                    for index,row in enumerate(evaluation,1):
                        value,_=_input_tensor(root,row,torch,np)
                        record=_forward(counter,runtime,{'original':'original_baseline_forwards','parent':'parent_baseline_forwards',
                            'r2':'R2_baseline_forwards'}[alias],net,value,grad=False,context={'alias':alias,'id':row['id']})
                        bank[row['id']]=_save_output(writer,OUTPUTS+'/baseline/'+alias+'/'+row['id'],row,
                            record,metrics_module,root,np,Image)
                        del record,value
                        if index%50==0 or index==478:print(json.dumps({'baseline':alias,'outputs':index,'of':478}),flush=True)
                    if alias=='r2':
                        for row in auxiliary:
                            value,_=_input_tensor(root,row,torch,np)
                            record=_forward(counter,runtime,'R2_auxiliary_forwards',net,value,grad=False,
                                context={'auxiliary':'r2','id':row['id']})
                            r2_auxiliary[row['id']]=_save_output(writer,OUTPUTS+'/auxiliary/r2/'+row['id'],row,
                                record,metrics_module,root,np,Image)
                            del record,value
                        writer.json(OUTPUTS+'/auxiliary/r2/bank.json',{'complete':True,'cases':list(r2_auxiliary.values()),
                            'TRAIN_diagnostic_only_not_optimization':True,'not18_independent_identities':True})
                    require(m.state_digest(net)==receipt['canonical515_sha256'] and all(p.grad is None for p in net.parameters()),
                        'Frozen baseline changed during actual readouts')
                    baselines[alias]=bank
                    del net;torch.cuda.empty_cache()
            with stage_deadline(STAGES['initialization']):
                r2,r2_receipt=m.load_frozen_baseline('r2')
                original,original_receipt=m.load_frozen_baseline('original')
                candidate,census,initial_receipt=m.make_candidate(r2)
                require(len([p for p in candidate.parameters() if p.requires_grad])==24,
                    'Exactly24 new adapter tensors trainable')
                frozen_digest=m.frozen_state_digest(candidate)
                require(frozen_digest==m.R2_DIGEST,'Candidate retained base differs from R2')
                writer.json(OUTPUTS+'/model_initialization.json',{'candidate':initial_receipt,'census':census,
                    'r2':r2_receipt,'original':original_receipt,'all515_original_tensors_frozen':True})
                teachers={};feature_bank={}
                refs=schedule['anchor_source_order']
                clear_by_ref={r['reference_id']:r for r in cases.values() if r['profile']=='clear'}
                dark_by_ref={r['reference_id']:r for r in cases.values() if r['profile']==d.DARK_LINEAR_PROFILE}
                require(set(clear_by_ref)==set(dark_by_ref)==set(refs),'Complete55 paired teacher role sources required')
                for rid in refs:
                    row=clear_by_ref[rid];value,_=_input_tensor(root,row,torch,np)
                    record=_forward(counter,runtime,'R2_clean_feature_teacher_forwards',r2,value,grad=False,
                        context={'reference_id':rid,'id':row['id'],'target_kind':'same_R2_clear256_features'})
                    features=m.teacher_feature_cpu(record);scales=m.feature_scales_from_clean(features)
                    feature_bank[rid]={'features':{},'scales':scales,'clear_input':row['input'],
                        'target_policy':'same frozen R2 clear256 prepared512 observed skip features'}
                    for site,feature in features.items():
                        name=OUTPUTS+'/teacher/features/'+rid+'/encoder'+site+'.npy'
                        feature_bank[rid]['features'][site]=writer.array(name,np.ascontiguousarray(feature.numpy()),np)
                    writer.json(OUTPUTS+'/teacher/features/'+rid+'/scales.json',{'reference_id':rid,'scales':scales,
                        'algorithm':'max(1e-6,float64 RMS of frozen same-R2 clear feature)','input':row['input']})
                    del record,value,features
                for role,net,key,kind,by_ref in (('clear',original,'original_clear_teacher_forwards','original',clear_by_ref),
                                               ('dark',r2,'R2_dark_teacher_forwards','r2',dark_by_ref)):
                    for rid in refs:
                        row=by_ref[rid];value,_=_input_tensor(root,row,torch,np)
                        record=_forward(counter,runtime,key,net,value,grad=False,context={'role':role,'id':row['id']})
                        u,raw,png=_arrays(record,np);prefix=OUTPUTS+'/teacher/'+role+'/'+row['id']
                        upin=writer.array(prefix+'_U.npy',u,np);rawpin=writer.array(prefix+'_RAW.npy',raw,np)
                        pngpin=writer.png(prefix+'_PNG.png',png,np,Image)
                        teachers[role,row['id']]={'target':m.TeacherTarget(record.unclipped256.detach().cpu().clone(),kind),
                            'U':upin,'RAW':rawpin,'PNG':pngpin,'raw':raw,'png':png,'kind':kind}
                        del record,value
                writer.json(OUTPUTS+'/teacher/manifest.json',{'complete':True,'features':feature_bank,
                    'behavior':[{'role':role,'id':identifier,**{k:v[k] for k in ('U','RAW','PNG','kind')}}
                        for (role,identifier),v in sorted(teachers.items())],
                    'unique_feature_teacher_forwards':55,'unique_behavior_teacher_forwards':110,
                    'feature_targets_read_CPU_one_source_at_a_time':True,'no_new_HQ_prior_teacher':True})
            _witness(writer,runtime,counter,protocol,cases,candidate,original,r2,teachers,feature_bank,np)
            optimizer=torch.optim.Adam(m.optimizer_groups(candidate),betas=(.9,.99),eps=1e-8,weight_decay=0.)
            require(len(optimizer.state)==0 and len(optimizer.param_groups)==1 and
                len(optimizer.param_groups[0]['params'])==24,'Fresh empty24-tensor Adam required')
            sampler=d.FiniteSampler(schedule)
            initial_bank=None;initial_guards=None

            def auxiliary_snapshot(update):
                bank={};alias=f'candidate_{update:04d}'
                for row in auxiliary:
                    value,_=_input_tensor(root,row,torch,np)
                    record=_forward(counter,runtime,'candidate_auxiliary_forwards',candidate,value,grad=False,
                        context={'auxiliary_snapshot':update,'id':row['id']})
                    measured=_save_output(writer,OUTPUTS+'/auxiliary/'+alias+'/'+row['id'],row,record,metrics_module,root,np,Image)
                    require(measured['upstream']==r2_auxiliary[row['id']]['upstream'],'Auxiliary upstream R2 changed')
                    if update==0:
                        require(all(measured['outputs'][v]['sha256']==r2_auxiliary[row['id']]['outputs'][v]['sha256']
                            for v in ('raw','png')),'Auxiliary initial R2 RAW/PNG parity failed')
                    bank[row['id']]=measured;del record,value
                writer.json(OUTPUTS+'/auxiliary/'+alias+'/bank.json',{'complete':True,'cases':list(bank.values()),
                    'TRAIN_diagnostic_only_not_optimization':True,'not18_independent_identities':True})

            def snapshot(update):
                nonlocal initial_bank
                with stage_deadline(STAGES['snapshot']):
                    require(m.frozen_state_digest(candidate)==frozen_digest,'Frozen515 base changed before snapshot')
                    name=OUTPUTS+f'/snapshot_{update:04d}/checkpoint.pt'
                    value,complete=_checkpoint(writer,runtime,protocol_sha,candidate,optimizer,sampler,progress,name,counter)
                    progress.update(state_checkpoint=value,state_checkpoint_complete=complete)
                    bank={}
                    for index,row in enumerate(evaluation,1):
                        tensor,_=_input_tensor(root,row,torch,np)
                        record=_forward(counter,runtime,'candidate_snapshot_forwards',candidate,tensor,grad=False,
                            context={'snapshot':update,'id':row['id']})
                        measured=_save_output(writer,OUTPUTS+f'/snapshot_{update:04d}/'+row['id'],row,record,metrics_module,root,np,Image)
                        require(measured['upstream']==baselines['r2'][row['id']]['upstream'],
                            'Fixed observed logits/LQ/codes/postAdaIN changed')
                        bank[row['id']]=measured;del record,tensor
                        if index%50==0 or index==478:print(json.dumps({'snapshot':update,'outputs':index,'of':478}),flush=True)
                    if update==0:
                        initial_bank=bank;initial_parity=initial_output_parity(bank,baselines['r2'])
                    dev={k:v for k,v in bank.items() if v.get('cohort')=='historical_DEV406'}
                    controls={alias:{k:v for k,v in b.items() if v.get('cohort')=='historical_DEV406'} for alias,b in baselines.items()}
                    initial_dev={k:v for k,v in initial_bank.items() if v.get('cohort')=='historical_DEV406'}
                    gates={'preservation':metrics_module.preservation(dev,controls['original']),
                        'parent_dark_retention':metrics_module.dark_retention(dev,controls['parent']),
                        'R2_dark_retention':metrics_module.dark_retention(dev,controls['r2']),
                        'interim':metrics_module.interim(dev,controls['original'],initial_dev),
                        'filters':metrics_module.filter_summary(list(bank.values())),
                        'inherited_initial_failures_not_qualified':update==0}
                    if update==0:gates['initial_R2_full478_parity']=initial_parity
                    writer.json(OUTPUTS+f'/snapshot_{update:04d}/gates.json',gates)
                    auxiliary_snapshot(update)
                    require(m.frozen_state_digest(candidate)==frozen_digest and
                        all(not part.training for part in candidate.modules()),'Base/modes changed during snapshot')
                    if update:
                        require(gates['interim']['pass'] and gates['R2_dark_retention']['pass'] and
                            gates['parent_dark_retention']['pass'],
                            'New/worsened preservation or parent/R2 dark retention requirement failed; retain stop')
                    return bank,gates

            def guards(update):
                nonlocal initial_guards
                bank={}
                for guard in protocol['guard_cases']:
                    row=cases[guard['id']];tensor,_=_input_tensor(root,row,torch,np)
                    record=_forward(counter,runtime,'candidate_TRAIN_guard_forwards',candidate,tensor,grad=False,
                        context={'TRAIN_guard':update,'role':guard['role'],'id':row['id']})
                    measured=_save_output(writer,OUTPUTS+f'/guard_{update:04d}/'+guard['role']+'/'+row['id'],row,
                        record,metrics_module,root,np,Image)
                    measured['role']=measured['guard_role']=guard['role']
                    if guard['role'] in ('clear','dark'):
                        teacher=teachers[guard['role'],row['id']];_,raw,png=_arrays(record,np)
                        measured['behavior_distance']={
                            'raw':float(np.abs(raw.astype(np.float64)-teacher['raw'].astype(np.float64)).mean()),
                            'native_png':float(np.abs((png.astype(np.float32)/np.float32(255)).astype(np.float64)-
                                (teacher['png'].astype(np.float32)/np.float32(255)).astype(np.float64)).mean())}
                        measured['behavior_teacher']={k:teacher[k] for k in ('U','RAW','PNG','kind')}
                    bank[row['id']]=measured;del tensor,record
                writer.json(OUTPUTS+f'/guard_{update:04d}/bank.json',{'complete':True,'cases':list(bank.values()),
                    'TRAIN_replay_not_independent_DEV_or_FINAL':True})
                if update==0:initial_guards=bank
                gate=metrics_module.guard_interim(bank,initial_guards)
                writer.json(OUTPUTS+f'/guard_{update:04d}/gates.json',gate)
                require(update==0 or gate['pass'],'Repeated TRAIN guard preservation/retention requirement failed; retain stop')

            snapshot(0);guards(0)
            with stage_deadline(STAGES['training']):
                for _ in range(110):
                    preview=sampler.preview()
                    _train_step(writer,runtime,counter,preview,cases,candidate,optimizer,teachers,feature_bank,frozen_digest,sampler,progress,np)
                    current=progress['completed_training_updates']
                    print(json.dumps({'update':current,'of':110,'LR':m.learning_rate(current)}),flush=True)
                    if current in (55,110):_,final_gates=snapshot(current)
                    if current in (20,55,110):guards(current)
            require(final_gates['preservation']['pass'] and final_gates['parent_dark_retention']['pass'] and
                final_gates['R2_dark_retention']['pass'],'Final original preservation or both dark retention requirements failed')
            require(counter.values==call_caps() and counter.returned_values==call_caps() and not counter.pending and
                sampler.state()['completed_updates']==110,'Final finite operation/sampler census differs')
            writer.json(OUTPUTS+'/run_results.json',{'complete':True,'protocol_sha256':protocol_sha,**progress,
                'counts':counter.values,'actual_return_counts':counter.returned_values,
                'final_gates':final_gates,'model_qualified':False,
                'training_completion_not_acceptance':True,'automatic_app_promotion':False})
    except BaseException as error:
        if runtime.torch is not None and candidate is not None:
            try:
                if needs_stopped_checkpoint(progress):
                    with stage_deadline(45):
                        value,complete=_checkpoint(writer,runtime,protocol_sha,candidate,optimizer,sampler,progress,
                            OUTPUTS+'/stopped_checkpoint.pt',counter,recovery=True)
                    progress.update(state_checkpoint=value,state_checkpoint_complete=complete)
            except BaseException as checkpoint_error:
                progress['checkpoint_failure']={'type':type(checkpoint_error).__name__,'error':str(checkpoint_error)}
                if progress.get('state_checkpoint') and '.tmp-' not in progress['state_checkpoint']['path']:
                    progress['prior_checkpoint_retained_not_current_state']=progress['state_checkpoint']
                    progress.update(state_checkpoint=None,state_checkpoint_complete=False)
        writer.json(OUTPUTS+'/failure.json',{'complete':True,'failure':True,'protocol_sha256':protocol_sha,
            **progress,'counts':counter.values,'actual_return_counts':counter.returned_values,
            'pending_reservations':counter.pending,'error_type':type(error).__name__,
            'error':str(error),'traceback':traceback.format_exc(),'training_success_not_implied':True,'model_qualified':False})
        raise


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--verify-transfer',action='store_true');actions.add_argument('--run',action='store_true')
    actions.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.verify_transfer:
        verified=t.verify_transfer(args.root,args.protocol_sha)
        protocol=verified['protocol'] if 'protocol' in verified else verified
        plan=validate_plan(protocol);_local_metadata(args.root,protocol)
        print(json.dumps({'complete':True,'verified_transfer':True,'plan':plan,
            'neural_calls':0,'gradient_queries':0,'optimizer_updates':0}),flush=True)
    elif args.export:print(json.dumps(t.export_results(args.root,args.protocol_sha)),flush=True)
    else:run(args.root,args.protocol_sha)


if __name__=='__main__':main()
