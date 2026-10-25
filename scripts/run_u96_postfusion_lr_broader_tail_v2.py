"""Inert distinct655-slot tail after independently audited saved61; no retry.

Unchanged frozen inference/delivery/gates, with fixed16-slot disposable workers.
The original failed716 execution, its plan, all952 files and61 outputs stay exact.
"""
import argparse
import _thread
from collections import Counter
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid

import run_u96_postfusion_lr_broader_preservation_v1 as O

ROOT = O.ROOT
PREP = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_preparation_v2'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_v2'
PLAN = PREP / 'remaining655_tail_plan_v2.json'
AUDIT = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_partial_audit_v1'
ORIGINAL_PLAN_SHA = '33a95897ed91c7f59a0704de65410f6093dd54a35f13a91933113775d8703c34'
ORIGINAL_RUNNER_SHA = '9739c49f0a34ccc62f31968cf3b6d956bef08755d04d3cc9ac41518fc6d096ae'
PINS = {
    'partial_audit': ('independent_partial_audit_v1.json', '0ff0be619ea19cf0cca371c436fb97cecee3cff33e771f3f452e1287319f25aa'),
    'retained61': ('retained61_output_manifest_v1.json', 'a0a80651235a51891808b036e4deaeb7f93c467469b595f1b7d92561c5f84fc0'),
    'retained_worker': ('retained_worker_files_manifest_v1.json', 'fb50209ce4103c0b547343f68a4d94a4c71cf86c070a80bd00e3ef0aca04c824'),
    'resource_audit': ('independent_resource_journal_audit_v1.json', '675989fc1ef16238e86641725ccbc2e2b40d874db03bebd4191925168dd7b25f'),
    'root8_visual': ('root_partial_visual_review_v1.json', 'd82e6363f7f1bb35b7292e1f694ae829530e023303c93a41ae94da0863ad3f4a')}
LIMITS = {'new_attempts': 655, 'new_returns': 655, 'new_saved': 655,
    'per_arm': {O.ARMS[0]: 327, O.ARMS[1]: 328}, 'chunk_forward_slots': 16,
    'global_seconds': 13829, 'cleanup_seconds': 10, 'load_seconds': 240, 'forward_seconds': 120,
    'startup_RAM_bytes': 5 * 1024 ** 3 // 2, 'minimum_available_RAM_bytes': 512 * 1024 ** 2,
    'maximum_worker_RSS_bytes': 6 * 1024 ** 3, 'minimum_free_disk_bytes': 8 * 1024 ** 3,
    'cumulative_output_bytes': 6 * 1024 ** 3, 'retained_original_bytes': 381083149,
    'maximum_new_output_bytes': 6061367795, 'CPU_threads': 4}
OWN_TEST = 'tests/test_u96_postfusion_lr_broader_tail_v2.py'
need, pin, bound, load = O.need, O.pin, O.bound, O.load


def safe(path, directory=False):
    path = Path(path)
    need('..' not in path.parts, 'No parent traversal in tail evidence')
    path = path.absolute()
    need(path.resolve(strict=False).is_relative_to(ROOT), 'Tail evidence stays in workspace')
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                 'No linked/reparse tail evidence')
    if path.exists():
        info = path.stat()
        need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
             'Ordinary single-link tail evidence')
    return path.resolve(strict=False)


def json_file(path, cap=16 * 1024 ** 2):
    path = safe(path)
    before = path.stat()
    need(0 < before.st_size <= cap, 'Bounded tail JSON')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    after = path.stat()
    need(len(raw) <= cap and (before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Tail JSON changed while reading')
    return O.B.strict_json(raw)


def chunks(slots):
    need(type(slots) is list and len(slots) == 655, 'Exactly655 remaining slots')
    need([r['new_output_index'] for r in slots] == list(range(62, 717)), 'Exact global62..716 order')
    need(Counter(r['arm'] for r in slots) == Counter(LIMITS['per_arm']), 'Exact327/328 remaining arm counts')
    return [{'chunk': i // 16 + 1, 'slots': slots[i:i + 16]} for i in range(0, 655, 16)]


def dependency_contract():
    need('torch' not in sys.modules, 'All tail verification precedes Torch')
    need(pin(Path(O.__file__))['sha256'] == ORIGINAL_RUNNER_SHA, 'Original runner exact and unchanged')
    original = O.verify(ORIGINAL_PLAN_SHA)
    records = {}
    for key, (name, sha) in PINS.items():
        row = pin(AUDIT / name)
        need(row['sha256'] == sha, 'Exact independently completed partial evidence: ' + key)
        records[key] = row
    audit = load(bound(records['partial_audit']))
    retained = load(bound(records['retained61']))
    worker = load(bound(records['retained_worker']))
    resource = load(bound(records['resource_audit']))
    visual = load(bound(records['root8_visual']))
    need(audit['complete'] and audit['independent_partial_saved_audit_pass'] and audit['outputs_verified'] == 61
         and not audit['forward_outcome_unknown'] and audit['remaining_new_output_slots'] == 655
         and audit['retained61_output_manifest'] == records['retained61']
         and audit['retained_worker_manifest'] == records['retained_worker'], 'Exact known61 saved; no unknown retry')
    need(retained['complete'] and retained['outputs_verified'] == 61 and retained['U96'] == 31 and retained['u64'] == 30
         and [r['slot'] for r in retained['rows']] == original['forward_slots'][:61], 'Carried61 exact original slot prefix')
    need(resource['complete'] and resource['terminal_counters']['forward_attempts'] == 61
         and resource['terminal_counters']['forward_returns'] == 61 and resource['terminal_counters']['saved_outputs'] == 61
         and not resource['unresolved_forward_outcome']
         and resource['sole_resource_violation']['failed_constraints'] == ['available_RAM'], 'Original sole RAM stop retained')
    need(visual['complete'] and visual.get('all8_current_partial_sheets_personally_reviewed') is True
         and visual.get('generated_output_cells_reviewed') == 61 and visual.get('case_rows_reviewed') == 31,
         'Root actual8 partial visual review required')
    # The independently sealed namespace closes every original file, including failed receipts.
    files = worker.get('files', worker.get('rows'))
    need(type(files) is list and len(files) == 952 and sum(r['bytes'] for r in files) == 381083149,
         'Exact952/381083149 retained original worker bytes')
    for row in files:
        bound(row)
    for row in retained['rows']:
        for entry in [row['record'], row['input'], row['target'], row['support'], *row['outputs'].values()]:
            bound(entry)
    original_supervisor = O.PREP / 'broader_supervisor_v1.json'
    old = load(original_supervisor)
    need(old['worker_exit'] == 1 and not old['timeout'] and not old['forward_outcome_unknown']
         and old['counters']['saved_outputs'] == 61 and old['seconds'] == 571.5643963,
         'Original failed execution terminal preserved')
    records.update({'original_plan': pin(O.PLAN), 'original_runner': pin(Path(O.__file__)),
                    'original_supervisor': pin(original_supervisor), 'original_failure': audit['failure'],
                    'source_data_model_closure': audit['source_data_model_pin_closure'],
                    'root8_visual': records['root8_visual']})
    bound(records['source_data_model_closure'])
    bound(records['original_failure'])
    return original, retained, records


def prepare(source_review):
    need(not PLAN.exists(), 'Fresh tail plan; never overwrite or reselect')
    original, retained, prerequisites = dependency_contract()
    sources = [pin(Path(__file__).resolve()), pin(ROOT / OWN_TEST)]
    review = pin(safe(source_review))
    O.reviewed_sources(load(bound(review)), {r['path'] for r in sources})
    tail = original['forward_slots'][61:]
    value = {'complete': True, 'format': 'broader-preservation-distinct-tail655-v2', 'status': 'SEALED_PREPARED_UNRUN',
        'original_plan': pin(O.PLAN), 'prerequisites': prerequisites, 'sources': sources,
        'source_review': review, 'retained61': prerequisites['retained61'], 'remaining_slots': tail,
        'chunks': chunks(tail), 'limits': LIMITS, 'carried_forward_counts': {'attempts': 61, 'returns': 61, 'saved': 61},
        'original_supervisor_seconds': 571.5643963, 'new_calls_maximum': 655, 'combined_calls_maximum': 716,
        'carried_native_output_slots': 64, 'excluded_native_rows': 16, 'planned_final_visual_sheets': 90,
        'no_automatic_retry_resume': True, 'optimization_permitted': False, 'FINAL_used': False,
        'model_qualified': False, 'app_default_changed': False, 'preparation_neural_calls': 0}
    PREP.mkdir(parents=True, exist_ok=True)
    with PLAN.open('xb') as stream:
        stream.write(O.B.json_bytes(value))
    return pin(PLAN)


def verify(expected_sha):
    need('torch' not in sys.modules, 'Tail verify is non-neural')
    need(pin(PLAN)['sha256'] == expected_sha, 'Exact explicit tail plan SHA')
    plan = load(PLAN)
    original, retained, prerequisites = dependency_contract()
    need(plan['complete'] is True and plan['format'] == 'broader-preservation-distinct-tail655-v2'
         and plan['limits'] == LIMITS and plan['prerequisites'] == prerequisites
         and plan['remaining_slots'] == original['forward_slots'][61:]
         and plan['chunks'] == chunks(original['forward_slots'][61:])
         and plan['retained61'] == prerequisites['retained61'] and plan['original_plan'] == pin(O.PLAN)
         and plan['no_automatic_retry_resume'] is True and plan['optimization_permitted'] is False
         and plan['FINAL_used'] is False and plan['model_qualified'] is False and plan['app_default_changed'] is False,
         'Unchanged disjoint655-slot source/data/model/resource contract')
    need(plan['sources'] == [pin(Path(__file__).resolve()), pin(ROOT / OWN_TEST)], 'Exact current tail source and tests')
    O.reviewed_sources(load(bound(plan['source_review'])), {r['path'] for r in plan['sources']})
    return plan, original, retained


class Writer:
    """Own new tree only, with the cumulative original+tail6GiB budget."""
    def __init__(self, directory, allowance):
        self.directory = safe(directory, True)
        self.allowance = allowance
        self.reserved_bytes = 0
        self.successful_bytes = 0
        self.successful_files = 0
        self.external_bytes = 0

    def write(self, path, raw):
        path = safe(path)
        need(path.is_relative_to(self.directory) and type(raw) is bytes
             and self.reserved_bytes + len(raw) <= self.allowance, 'Reserve finite owned tail bytes before writing')
        if self.directory == OUT.resolve(strict=False):
            need(self.external_bytes + self.reserved_bytes + len(raw) <= LIMITS['maximum_new_output_bytes'],
                 'Cumulative parent plus sealed-child byte ledger before write')
        path.parent.mkdir(parents=True, exist_ok=True)
        safe(path.parent, True)
        self.reserved_bytes += len(raw)
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.successful_bytes += len(raw); self.successful_files += 1
        return pin(path)

    def json(self, path, value):
        path = safe(path)
        need(not path.exists(), 'Fresh immutable atomic tail JSON')
        temporary = path.with_name(path.name + '.new')
        self.write(temporary, O.B.json_bytes(value))
        need(not path.exists(), 'Atomic tail JSON target remains fresh')
        os.replace(temporary, path)
        return pin(path)

    def synchronize_external(self):
        actual = tree_bytes(self.directory)
        need(actual >= self.successful_bytes, 'Actual tree contains all parent exclusive bytes')
        self.external_bytes = actual - self.successful_bytes
        need(actual <= LIMITS['maximum_new_output_bytes'], 'Actual cumulative new tree byte cap')


def tree_bytes(directory):
    directory = safe(directory, True)
    total = 0
    for path in directory.rglob('*'):
        safe(path, path.is_dir())
        if path.is_file():
            total += path.stat().st_size
    return total


def blank_counts():
    return {'attempts': 0, 'returns': 0, 'saved': 0,
            'per_arm': {arm: {'attempts': 0, 'returns': 0, 'saved': 0} for arm in O.ARMS}}


def counts_valid(counts, assigned):
    need(type(counts) is dict and set(counts) == {'attempts', 'returns', 'saved', 'per_arm'}, 'Exact finite tail counters')
    for key in ('attempts', 'returns', 'saved'):
        need(type(counts[key]) is int and 0 <= counts[key] <= len(assigned), 'Finite integer tail counters')
    need(counts['saved'] <= counts['returns'] <= counts['attempts'], 'Saved<=returned<=attempted')
    need(type(counts['per_arm']) is dict and set(counts['per_arm']) == set(O.ARMS), 'Exact two arm counters')
    for arm in O.ARMS:
        row = counts['per_arm'][arm]
        need(type(row) is dict and set(row) == {'attempts', 'returns', 'saved'}
             and all(type(row[k]) is int and 0 <= row[k] <= sum(r['arm'] == arm for r in assigned)
                     for k in ('attempts', 'returns', 'saved'))
             and row['saved'] <= row['returns'] <= row['attempts'], 'Finite per-arm counters')
    need(all(sum(counts['per_arm'][a][k] for a in O.ARMS) == counts[k] for k in ('attempts', 'returns', 'saved')),
         'Per-arm counters exactly sum')
    return counts


def combine_counts(rows):
    answer = blank_counts()
    for counts in rows:
        for key in ('attempts', 'returns', 'saved'):
            answer[key] += counts[key]
            for arm in O.ARMS:
                answer['per_arm'][arm][key] += counts['per_arm'][arm][key]
    return answer


def worker_python():
    executable = sys._base_executable if os.name == 'nt' else sys.executable
    environment = dict(os.environ)
    if os.name == 'nt':
        environment['__PYVENV_LAUNCHER__'] = sys.executable
    return executable, environment


def owned_reap(child, deadline, orderly=True):
    """Only this exact actual-interpreter Popen handle; one total10s grace."""
    forced = False
    try:
        child.wait(timeout=max(0., min(5., deadline - time.monotonic())) if orderly else 0)
    except subprocess.TimeoutExpired:
        forced = True
        try:
            child.terminate()
        except OSError:
            pass  # A concurrent exit or termination error still proceeds to exact-handle wait/kill.
        try:
            child.wait(timeout=max(0., min(2.5, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            try:
                child.kill()
            except OSError:
                pass
            child.wait(timeout=max(0., deadline - time.monotonic()))
    need(child.poll() is not None, 'Actual owned worker exit must be confirmed')
    return forced


def input_arrays(spec, writer, folder, np, Image):
    with Image.open(bound(spec['input'])) as image:
        need(image.mode == 'RGB' and image.size == (256, 256), 'ExactRGB256 observed input')
        canvas = np.asarray(image).copy()
    if spec['support']:
        with Image.open(bound(spec['support'])) as image:
            need(image.mode == 'L' and image.size == (256, 256), 'ExactL256 support')
            observed = np.asarray(image).copy() > 0
    else:
        observed = np.ones((256, 256), dtype=np.bool_)
    with Image.open(bound(spec['target'])) as image:
        need(image.mode == 'RGB' and image.size == (256, 256), 'ExactRGB256 paired target')
        truth = np.asarray(image).astype(np.float32) / np.float32(255)
    need(O.B.digest(canvas.tobytes()) == spec['input_pixels']['RGB8_sha256'], 'Exact input pixels')
    stream = io.BytesIO(); Image.fromarray(observed.astype(np.uint8) * 255).save(stream, format='PNG')
    inputs = {'original': writer.write(folder / 'input.png', bound(spec['input']).read_bytes()),
              'target': writer.write(folder / 'target.png', bound(spec['target']).read_bytes()),
              'support': writer.write(folder / 'observed_support.png', stream.getvalue())}
    return canvas, observed, truth, inputs


def frozen_models(original, torch):
    source = importlib.import_module('research.codeformer_u96_postfusion_lr_inference_v1')
    legacy = importlib.import_module('research_spatial_restoration_v1')
    from third_party.codeformer.codeformer_arch import CodeFormer
    candidate = source.load_frozen_cpu_restorer(bound(original['model']['r2']), bound(original['model']['u96']),
        bound(original['head_file']), inference_source_sha256=O.B.INFERENCE_SHA)
    with torch.no_grad(), torch.device('cpu'), torch.random.fork_rng(devices=[]):
        base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(source.SITES))
        r2 = torch.load(bound(original['model']['r2']), map_location='cpu', weights_only=True, mmap=True)
        base.load_state_dict(r2['params_ema'], strict=True)
        incumbent = legacy.inference_class(torch)(base).float().eval().requires_grad_(False)
        u96 = torch.load(bound(original['model']['u96']), map_location='cpu', weights_only=True, mmap=True)
        incumbent.adapters.load_state_dict({k.removeprefix('adapters.'): v for k, v in u96['adapter_state'].items()}, strict=True)
        del r2, u96
    need(incumbent.base is not candidate.base, 'Distinct owned frozen model bases')
    def invariants():
        need(len(incumbent.state_dict()) == 539 and all(not m.training for m in incumbent.modules()) and
             all(v.device.type == 'cpu' and v.dtype == torch.float32 and not v.requires_grad and v.grad is None
                 for v in incumbent.parameters()), 'Frozen539 incumbent CPUFP32')
        need(legacy.base.state_sha(incumbent.base.state_dict(), torch) == source.R2_STATE and
             legacy.base.state_sha({'adapters.' + k: v for k, v in incumbent.adapters.state_dict().items()}, torch) == source.U96_STATE,
             'Exact retained515+24 canonical U96')
        need(all(not m._forward_hooks and not m._forward_pre_hooks for m in incumbent.modules()), 'Clean incumbent hooks')
        candidate.validate_invariants()
    invariants()
    return dict(zip(O.ARMS, (incumbent, candidate))), invariants, source


def metric_module():
    metric_spec = importlib.util.spec_from_file_location('_owned_broader_tail_metrics_v2', ROOT / O.METRICS_PATH)
    M = importlib.util.module_from_spec(metric_spec); metric_spec.loader.exec_module(M)
    M.sum = O.ordered_sum
    return M


def chunk_worker(expected_sha, chunk_index, token):
    need(type(chunk_index) is int and 1 <= chunk_index <= 41 and type(token) is str and len(token) == 32
         and all(c in '0123456789abcdef' for c in token), 'Fixed owned chunk index and generated token')
    folder = safe(OUT / 'chunks' / ('%03d' % chunk_index), True)
    startup = json_file(folder / 'startup.json', 16384)
    need(type(startup) is dict and set(startup) == {'chunk', 'token', 'plan_sha256', 'remaining_seconds',
                                                 'deadline_monotonic', 'byte_allowance'}
         and type(startup['chunk']) is int and startup['chunk'] == chunk_index and startup['token'] == token and startup['plan_sha256'] == expected_sha
         and type(startup['remaining_seconds']) in (int, float) and 0 < startup['remaining_seconds'] <= LIMITS['global_seconds']
         and type(startup['deadline_monotonic']) in (int, float)
         and 0 < startup['deadline_monotonic'] - time.monotonic() <= LIMITS['global_seconds']
         and type(startup['byte_allowance']) is int and 0 < startup['byte_allowance'] <= LIMITS['maximum_new_output_bytes'],
         'Bounded exact owned chunk startup')
    writer = Writer(folder, startup['byte_allowance'])
    writer.json(folder / 'identity.json', {'token': token, 'chunk': chunk_index, 'worker_pid': os.getpid(),
                'python_prefix': sys.prefix, 'python_executable': sys.executable})
    deadline = startup['deadline_monotonic']  # Same OS monotonic clock; import/startup cannot reset the global budget.
    command_path = folder / 'command.json'
    while not command_path.exists():
        need(time.monotonic() < deadline, 'Chunk command deadline exceeded')
        time.sleep(.1)
    command = json_file(command_path, 16384)
    need(type(command) is dict and type(command.get('chunk')) is int
         and command == {'token': token, 'chunk': chunk_index, 'plan_sha256': expected_sha},
         'Exact verified-identity chunk command')
    plan, original, retained = verify(expected_sha)
    assigned = plan['chunks'][chunk_index - 1]['slots']
    counters = blank_counts(); active = None; phase = 'verified_before_Torch'
    journal = O.Journal(writer, folder / 'journal'); resources = O.Journal(writer, folder / 'resources')
    def status(label):
        return journal.append({'token': token, 'chunk': chunk_index, 'label': label, 'phase': phase, 'slot': active,
            'counters': counters, 'forward_outcome_unknown': counters['attempts'] != counters['returns'],
            'model_qualified': False, 'new_calls_maximum': len(assigned)})
    def resource(label):
        import psutil
        row = {'label': label, 'worker_RSS_bytes': psutil.Process().memory_info().rss,
               'available_RAM_bytes': psutil.virtual_memory().available, 'free_disk_bytes': shutil.disk_usage(ROOT).free,
               'remaining_seconds': deadline - time.monotonic(), 'reserved_output_bytes': writer.reserved_bytes}
        resources.append(row)
        minimum_RAM = LIMITS['startup_RAM_bytes'] if label in ('before_Torch', 'before_model_load') else LIMITS['minimum_available_RAM_bytes']
        need(row['worker_RSS_bytes'] <= LIMITS['maximum_worker_RSS_bytes']
             and row['available_RAM_bytes'] >= minimum_RAM
             and row['free_disk_bytes'] >= LIMITS['minimum_free_disk_bytes'] and row['remaining_seconds'] > 0,
             'Unchanged finite chunk resource cap')
    def watchdog(seconds):
        remaining = deadline - time.monotonic()
        need(remaining > 0, 'Global remaining deadline exhausted')
        timer = threading.Timer(min(seconds, remaining), lambda: os._exit(124)); timer.daemon = True; timer.start()
        return timer
    records = []
    try:
        status('verified_before_Torch'); resource('before_Torch')
        import torch
        import numpy as np
        from PIL import Image
        from torch.nn import functional as F
        torch.set_num_threads(4); need(torch.get_default_dtype() == torch.float32, 'CPU FP32 default')
        need(ROOT.resolve() == Path.cwd().resolve(), 'Validated ROOT before retained factory imports')
        sys.path.insert(0, str(ROOT))
        phase = 'load_two_fresh_owned_frozen_CPU_models'; status('before_model_load'); resource('before_model_load')
        timer = watchdog(LIMITS['load_seconds'])
        try:
            models, invariants, source = frozen_models(original, torch)
        finally:
            timer.cancel()
        M = metric_module()
        def array(path, value):
            need(value.dtype == np.float32 and np.isfinite(value).all(), 'Finite FP32 saved tensor')
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); return writer.write(path, stream.getvalue())
        def png(path, value):
            stream = io.BytesIO(); Image.fromarray(value).save(stream, format='PNG'); return writer.write(path, stream.getvalue())
        for active in assigned:
            spec = original['paired_cases'][active['paired_case_index']]
            arm, arm_index, cid = active['arm'], active['arm_index'], active['case_id']
            need(spec['id'] == cid and arm == O.ARMS[arm_index], 'Exact original paired slot identity')
            slotfolder = folder / 'slots' / ('%06d' % active['new_output_index'])
            canvas, observed, truth, inputs = input_arrays(spec, writer, slotfolder, np, Image)
            tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1).copy())[None] / 255
            sample = F.interpolate(tensor, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            phase = 'before_forward'; resource(arm + '_before'); status('before_reservation')
            need(counters['attempts'] < len(assigned) <= 16, 'Finite fixed chunk forward reservations')
            counters['attempts'] += 1; counters['per_arm'][arm]['attempts'] += 1
            phase = 'forward_reserved'; status('forward_reserved')
            timer = watchdog(LIMITS['forward_seconds'])
            try:
                with torch.no_grad(): triple = models[arm](sample, w=1., detach_16=True, adain=True)
            finally:
                timer.cancel()
            counters['returns'] += 1; counters['per_arm'][arm]['returns'] += 1
            phase = 'forward_returned'; status('forward_returned')
            need(isinstance(triple, tuple) and len(triple) == 3, 'Stock retained output tuple')
            for value, shape in zip(triple, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))):
                need(value.shape == shape and value.dtype == torch.float32 and value.device.type == 'cpu'
                     and not value.requires_grad and torch.isfinite(value).all().item(), 'Finite exactCPU output shape')
            with torch.no_grad():
                native, logits, lq = triple
                unclipped = F.interpolate((native + 1) / 2, (256, 256), mode='bilinear', align_corners=False)
                raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
            raw_array = raw[0].permute(1, 2, 0).contiguous().numpy().copy()
            raw_array[~observed] = canvas.astype(np.float32)[~observed] / 255
            pixels = np.floor(raw_array * np.float32(255)).astype(np.uint8)
            output = {'native512': array(slotfolder / 'native512.npy', native.numpy()),
                'logits': array(slotfolder / 'logits.npy', logits.numpy()), 'LQ': array(slotfolder / 'LQ.npy', lq.numpy()),
                'unclipped': array(slotfolder / 'unclipped256.npy', unclipped[0].permute(1, 2, 0).contiguous().numpy()),
                'raw': array(slotfolder / 'raw.npy', raw_array), 'png': png(slotfolder / 'output.png', pixels),
                'metrics': M.metrics(raw_array, pixels, truth, observed, 'paired_synthetic'),
                'provenance': {'arm': arm, 'R2_file_sha256': O.B.R2_SHA, 'U96_file_sha256': O.B.U96_SHA,
                    'head_file_sha256': O.B.HEAD_SHA if arm_index else None,
                    'head_canonical_sha256': O.B.HEAD_STATE if arm_index else None,
                    'w': 1., 'detach16': True, 'AdaIN': True, 'CPU_FP32': True, 'CPU_threads': 4,
                    'alignment': 'none', 'display_processing': 'none', 'context_blur_or_noise': False,
                    'source_candidate_training_updates': 64 if arm_index else None,
                    'outside_support_exact_observed_pixels': True, 'model_qualified': False,
                    'forward_sites': list(models[arm].last_forward_sites)}}
            invariants(); need(models[arm].last_forward_sites == source.SITES, 'All4 orderedfeature sites')
            record = writer.json(slotfolder / 'record.json', {'complete': True, 'case_id': cid, **output,
                'inputs': inputs, 'counters': counters, 'slot': active, 'original_plan_sha256': ORIGINAL_PLAN_SHA,
                'tail_plan_sha256': expected_sha, 'chunk': chunk_index, 'worker_pid': os.getpid()})
            records.append({'slot': active, 'record': record, 'inputs': inputs, 'output': output})
            counters['saved'] += 1; counters['per_arm'][arm]['saved'] += 1
            phase = 'output_fully_saved'; status('output_fully_saved'); resource(arm + '_saved')
            del triple, native, logits, lq, raw, unclipped, raw_array, pixels, sample, tensor
        invariants(); resource('final'); counts_valid(counters, assigned)
        need(counters['attempts'] == counters['returns'] == counters['saved'] == len(assigned), 'Exact assigned chunk complete')
        writer.json(folder / 'terminal.json', {'complete': True, 'clean_exit': True, 'token': token,
                    'chunk': chunk_index, 'worker_pid': os.getpid(), 'counters': counters, 'records': records,
                    'tail_plan_sha256': expected_sha, 'forward_outcome_unknown': False, 'model_qualified': False})
        return 0
    except BaseException as error:
        writer.json(folder / 'failure.json', {'complete': True, 'token': token, 'chunk': chunk_index,
            'worker_pid': os.getpid(),
            'phase': phase, 'slot': active, 'counters': counters, 'forward_outcome_unknown': counters['attempts'] != counters['returns'],
            'error': type(error).__name__, 'message': str(error)[:4096], 'traceback': traceback.format_exc()[-16384:],
            'model_qualified': False, 'automatic_retry_resume': False})
        raise


def carried_records(retained):
    answer = []
    for row in retained['rows']:
        record = load(bound(row['record']))
        output = {key: record[key] for key in (*O.OUTPUT_KEYS, 'metrics', 'provenance')}
        answer.append({'slot': row['slot'], 'record': row['record'],
                       'inputs': {'original': row['input'], 'target': row['target'], 'support': row['support']},
                       'output': output})
    return answer


def aggregate(original, retained, new_records, new_counts, writer, expected_sha):
    need(len(new_records) == 655 and [r['slot'] for r in new_records] == original['forward_slots'][61:], 'Exactly655 ordered new saved outputs')
    counts_valid(new_counts, original['forward_slots'][61:])
    need(new_counts['attempts'] == new_counts['returns'] == new_counts['saved'] == 655, 'Exact tail655 complete')
    all_records = carried_records(retained) + new_records
    need([r['slot'] for r in all_records] == original['forward_slots'], 'No duplicate, missing or regenerated original slot')
    banks = {a: {} for a in O.ARMS}; cases = []
    for index, spec in enumerate(original['paired_cases']):
        pair = all_records[2 * index:2 * index + 2]
        need(len(pair) == 2 and [r['slot']['arm'] for r in pair] == list(O.ARMS)
             and all(r['slot']['case_id'] == spec['id'] for r in pair), 'Exact two-arm merged case')
        for row in pair:
            bound(row['record'])
            for entry in [*row['inputs'].values(), *[row['output'][key] for key in O.OUTPUT_KEYS]]:
                bound(entry)
            arm = row['slot']['arm']
            banks[arm][spec['id']] = {**{k: spec[k] for k in ('id', 'evidence', 'source', 'profile')},
                                     'metrics': row['output']['metrics']}
        case = {'complete': True, 'id': spec['id'], 'evidence': spec['evidence'], 'source': spec['source'],
                'profile': spec['profile'], 'split': 'DEV', 'historical_spec': spec,
                'inputs': pair[0]['inputs'], 'outputs': {row['slot']['arm']: row['output'] for row in pair},
                'output_record_provenance': {row['slot']['arm']: row['record'] for row in pair},
                'input_readability_not_newly_qualified': True, 'model_qualified': False}
        cases.append(case)
    by_pair = {r['id']: r for r in cases}; native = {r['historical_id']: r for r in original['native_carry32']}
    excluded = {r['historical_id']: r for r in original['native_excluded16']}; membership = []
    for spec in original['historical_cases']:
        cid = spec['id']
        if cid in by_pair:
            case_pin = writer.json(OUT / 'aggregate_cases' / cid / 'case.json', by_pair[cid])
            membership.append({'id': cid, 'status': 'SOURCE_BOUND_ORIGINAL61_PLUS_TAIL655_PAIRED_OUTPUTS', 'case': case_pin})
            continue
        for arm in O.ARMS:
            banks[arm][cid] = {**{k: spec[k] for k in ('id', 'evidence', 'source', 'profile')}, 'metrics': O.blank_native_metrics()}
        if cid in native:
            membership.append({'id': cid, 'status': 'CARRIED_NATIVE_UNPAIRED_OUTPUTS',
                               'prior_case': native[cid], 'native_reference_metrics': None, 'new_forward_calls': 0})
        else:
            need(cid in excluded, 'Exact excluded native membership')
            membership.append({'id': cid, 'status': 'EXCLUDED_INPUT_ONLY_ZERO_CALL',
                               'input_only_review': excluded[cid]['input_only_review'], 'outputs': None,
                               'reference_metrics': None, 'new_forward_calls': 0})
    M = metric_module()
    gates = {arm: {'preservation_vs_original': M.preservation(banks[arm], original['original_teacher_bank']),
                   'dark_retention_vs_R2': M.dark_retention(banks[arm], original['R2_dark_reference_bank'])} for arm in O.ARMS}
    passed = all(row['pass'] for row in gates[O.ARMS[1]].values())
    quality_status = 'BROAD_PRESERVATION_GATE_PASS_NOT_QUALIFIED' if passed else 'QUALITY_STOP_NOT_QUALIFIED'
    combined = {'forward_attempts': 716, 'forward_returns': 716, 'saved_outputs': 716, 'completed_paired_cases': 358,
                'per_arm': {arm: {'attempts': 358, 'returns': 358, 'saved': 358} for arm in O.ARMS}}
    value = {'complete': True, 'quality_status': quality_status, 'tail_plan_sha256': expected_sha,
             'original_plan': pin(O.PLAN), 'new_tail_counters': new_counts, 'carried_paired_output_slots': 61,
             'counters': combined, 'cases': cases, 'historical_membership406': membership, 'banks406': banks, 'gates': gates,
             'paired_fresh_cases': 358, 'combined_frozen_forward_returns': 716, 'new_tail_forward_returns': 655,
             'native_carried_output_slots': 64, 'native_excluded_rows_zero_call': 16, 'represented_output_slots': 780,
             'teacher_metric_policy': original['teacher_metric_policy'], 'native_reference_metrics': None,
             'paired_and_native_evidence_separate': True, 'model_qualified': False, 'app_default_changed': False,
             'no_optimization': True, 'planned_paired_visual_sheets': original['planned_paired_visual_sheets'],
             'final90_root_visual_review': 'pending', 'independent_saved_pixel_audit': 'pending'}
    writer.json(OUT / 'comparison.json', value)
    return value


def read_last_chunk(folder, assigned):
    last = None; partial = []
    for path in sorted((folder / 'journal').glob('*.json')):
        try:
            row = json_file(path, 1024 ** 2)
            counts_valid(row['counters'], assigned)
            need(type(row['sequence']) is int and row['sequence'] == int(path.stem), 'Ordered finite chunk journal')
            last = row
        except (ValueError, OSError, KeyError):
            partial.append(pin(path))
    return last, partial


def verified_failure(folder, chunk, token, worker_pid):
    path = folder / 'failure.json'
    if not path.exists():
        return None
    try:
        value = json_file(path, 32768)
        counts_valid(value['counters'], chunk['slots'])
        need(value['complete'] is True and value['token'] == token and type(value['chunk']) is int
             and value['chunk'] == chunk['chunk'] and type(value['forward_outcome_unknown']) is bool
             and type(value['worker_pid']) is int and value['worker_pid'] == worker_pid
             and value['forward_outcome_unknown'] == (value['counters']['attempts'] != value['counters']['returns'])
             and value['model_qualified'] is False and value['automatic_retry_resume'] is False
             and type(value['phase']) is str and len(value['phase']) <= 128
             and (value['slot'] is None or value['slot'] in chunk['slots']), 'Exact bounded known failed-chunk receipt')
        return value
    except (ValueError, OSError, KeyError):
        return None


def run_chunk(plan, expected_sha, chunk, writer, deadline, prior_counts):
    import psutil
    need(time.monotonic() < deadline, 'Global13829second tail cap')
    preflight = {'chunk': chunk['chunk'], 'available_RAM_bytes': psutil.virtual_memory().available,
                 'free_disk_bytes': shutil.disk_usage(ROOT).free, 'remaining_seconds': deadline - time.monotonic()}
    writer.json(OUT / 'preflight' / ('%03d.json' % chunk['chunk']), preflight)
    need(preflight['available_RAM_bytes'] >= LIMITS['startup_RAM_bytes']
         and preflight['free_disk_bytes'] >= LIMITS['minimum_free_disk_bytes'], 'New chunk needs2.5GiB RAM/8GiB disk; no retry')
    folder = safe(OUT / 'chunks' / ('%03d' % chunk['chunk']), True)
    need(not folder.exists(), 'Fresh fixed chunk namespace; no rerun')
    folder.mkdir(parents=True)
    token = uuid.uuid4().hex
    remaining_bytes = LIMITS['maximum_new_output_bytes'] - tree_bytes(OUT)
    need(remaining_bytes > 65536, 'Remaining cumulative6GiB output allowance')
    writer.json(folder / 'startup.json', {'chunk': chunk['chunk'], 'token': token, 'plan_sha256': expected_sha,
                 'remaining_seconds': min(LIMITS['global_seconds'], deadline - time.monotonic()),
                 'deadline_monotonic': deadline,
                 'byte_allowance': remaining_bytes - 65536})
    executable, environment = worker_python()
    log = safe(folder / 'worker.log')
    child = None; error = None; terminal = None; cleanup_failed = False
    prior_tree_bytes = tree_bytes(OUT) - tree_bytes(folder)
    try:
        # DEVNULL makes native stdout/stderr bounded at zero bytes. Keep an explicit empty log artifact.
        with log.open('xb'):
            child = subprocess.Popen([executable, '-B', str(Path(__file__).resolve()), '--worker', '--plan-sha', expected_sha,
                                      '--chunk', str(chunk['chunk']), '--token', token], env=environment, cwd=str(ROOT),
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
        identity_path = folder / 'identity.json'
        while not identity_path.exists():
            need(child.poll() is None and time.monotonic() < deadline, 'Owned chunk identity unavailable before deadline')
            time.sleep(.1)
        identity = json_file(identity_path, 4096)
        need(type(identity.get('worker_pid')) is int and type(identity.get('chunk')) is int
             and identity == {'token': token, 'chunk': chunk['chunk'], 'worker_pid': child.pid,
                              'python_prefix': sys.prefix, 'python_executable': sys.executable}, 'Exact actual PID before chunk command')
        writer.json(folder / 'command.json', {'token': token, 'chunk': chunk['chunk'], 'plan_sha256': expected_sha})
        while child.poll() is None:
            need(time.monotonic() < deadline, 'Global13829second deadline; no chunk retry')
            need(prior_tree_bytes + tree_bytes(folder) <= LIMITS['maximum_new_output_bytes'], 'Cumulative tail bytes within remaining6GiB')
            time.sleep(.25)
        child.wait(timeout=0)
        terminal = json_file(folder / 'terminal.json', 2 * 1024 ** 2)
        counts_valid(terminal['counters'], chunk['slots'])
        need(child.returncode == 0 and terminal['complete'] is True and terminal['clean_exit'] is True
             and terminal['token'] == token and type(terminal['chunk']) is int and terminal['chunk'] == chunk['chunk']
             and type(terminal['worker_pid']) is int and terminal['worker_pid'] == child.pid
             and terminal['tail_plan_sha256'] == expected_sha and terminal['forward_outcome_unknown'] is False
             and terminal['counters']['attempts'] == terminal['counters']['returns'] == terminal['counters']['saved'] == len(chunk['slots'])
             and [r['slot'] for r in terminal['records']] == chunk['slots'], 'Exact clean fixed chunk receipt before next spawn')
        for row in terminal['records']:
            record = load(bound(row['record']))
            need(record['slot'] == row['slot'] and record['inputs'] == row['inputs']
                 and all(record[k] == row['output'][k] for k in (*O.OUTPUT_KEYS, 'metrics', 'provenance')),
                 'Saved chunk record/terminal exact correspondence')
            for entry in [*row['inputs'].values(), *[row['output'][key] for key in O.OUTPUT_KEYS]]:
                bound(entry)
        total = combine_counts([prior_counts, terminal['counters']])
        counts_valid(total, plan['remaining_slots'])
        writer.synchronize_external()
        return terminal
    except BaseException as caught:
        error = caught
        raise
    finally:
        if child is not None:
            try:
                owned_reap(child, time.monotonic() + LIMITS['cleanup_seconds'], orderly=error is None)
            except BaseException:
                cleanup_failed = True
            last, partial = read_last_chunk(folder, chunk['slots'])
            failure = verified_failure(folder, chunk, token, child.pid)
            observed = failure['counters'] if failure else last['counters'] if last else blank_counts()
            if error is not None or cleanup_failed:
                writer.synchronize_external()
                writer.json(OUT / 'chunk_stop.json', {'complete': True, 'chunk': chunk['chunk'], 'worker_pid': child.pid,
                    'actual_exit_confirmed': child.poll() is not None, 'cleanup_failed': cleanup_failed,
                    'last_durable_counters': observed, 'partial_journal_files': partial,
                    'forward_outcome_unknown': bool(cleanup_failed or partial or last is None or
                         observed['attempts'] != observed['returns'] or failure is None),
                    'unobserved_attempt_upper_bound': len(chunk['slots']), 'global_retry_permitted': False,
                    'error': type(error).__name__ if error else 'CleanupFailure', 'model_qualified': False})
            need(not cleanup_failed, 'Actual worker exit unconfirmed; tail latched closed')


def supervise_body(expected_sha):
    start = time.monotonic(); deadline = start + LIMITS['global_seconds']
    plan, original, retained = verify(expected_sha)
    need(not OUT.exists(), 'Fresh tail output namespace; automatic retry/resume prohibited')
    safe(OUT, True).mkdir()
    writer = Writer(OUT, LIMITS['maximum_new_output_bytes'])
    writer.json(OUT / 'started.json', {'complete': True, 'tail_plan': pin(PLAN), 'runner': pin(Path(__file__).resolve()),
                 'original_plan': pin(O.PLAN), 'new_forward_cap': 655, 'carried61_new_calls': 0,
                 'global_remaining_seconds': 13829, 'original_supervisor_seconds': 571.5643963,
                 'native_carried_output_slots': 64, 'native_excluded_rows': 16, 'model_qualified': False})
    records = []; counters = blank_counts(); completed_chunks = []
    try:
        for chunk in plan['chunks']:
            terminal = run_chunk(plan, expected_sha, chunk, writer, deadline, counters)
            counters = combine_counts([counters, terminal['counters']]); records.extend(terminal['records'])
            completed_chunks.append({'chunk': chunk['chunk'], 'terminal': pin(OUT / 'chunks' / ('%03d' % chunk['chunk']) / 'terminal.json')})
            writer.json(OUT / 'progress' / ('%03d.json' % chunk['chunk']), {'complete': True,
                        'completed_chunks': len(completed_chunks), 'new_counts': counters, 'carried61_new_calls': 0,
                        'combined_attempts': 61 + counters['attempts'], 'combined_returns': 61 + counters['returns'],
                        'combined_saved': 61 + counters['saved'], 'remaining_seconds': deadline - time.monotonic(),
                        'actual_previous_worker_exit_confirmed': True})
        need(time.monotonic() < deadline, 'Global cap before aggregate')
        # Revalidate every original952 file and all source/data/model pins without importing Torch in parent.
        verify(expected_sha)
        writer.synchronize_external()
        comparison = aggregate(original, retained, records, counters, writer, expected_sha)
        need(time.monotonic() < deadline and tree_bytes(OUT) <= LIMITS['maximum_new_output_bytes'], 'Final shared time/byte caps')
        outcome = {'complete': True, 'all716_measurements_complete': True, 'quality_status': comparison['quality_status'],
                   'new_tail_counts': counters, 'carried61_counts': {'attempts': 61, 'returns': 61, 'saved': 61},
                   'combined_counts': comparison['counters'], 'completed_chunks': completed_chunks,
                   'forward_outcome_unknown': False, 'seconds': time.monotonic() - start,
                   'original_supervisor_seconds': 571.5643963, 'model_qualified': False, 'app_default_changed': False,
                   'automatic_retry_resume': False, 'training_optimizer_VM_calls': 0,
                   'final90_root_visual_review': 'pending', 'independent_saved_pixel_audit': 'pending'}
        writer.json(OUT / 'supervisor.json', outcome)
        return 0 if comparison['quality_status'] == 'BROAD_PRESERVATION_GATE_PASS_NOT_QUALIFIED' else 2
    except BaseException as error:
        writer.synchronize_external()
        stop = json_file(OUT / 'chunk_stop.json') if (OUT / 'chunk_stop.json').exists() else None
        outcome = {'complete': True, 'status': 'LOCAL_TAIL_INFRASTRUCTURE_STOP_NOT_QUALIFIED',
                   'completed_chunk_counts': counters, 'completed_chunks': completed_chunks,
                   'current_chunk_stop': pin(OUT / 'chunk_stop.json') if stop else None,
                   'observed_total_new_counts': combine_counts([counters, stop['last_durable_counters']]) if stop else counters,
                   'forward_outcome_unknown': stop['forward_outcome_unknown'] if stop else False,
                   'unobserved_attempt_upper_bound': stop['unobserved_attempt_upper_bound'] if stop else 0,
                   'carried61_outputs_unchanged': True, 'all716_measurements_complete': False,
                   'quality_stop_recorded': False, 'finite_preservation_gate_pass_not_qualification': False,
                   'seconds': time.monotonic() - start, 'error': type(error).__name__, 'message': str(error)[:4096],
                   'automatic_retry_resume': False, 'model_qualified': False, 'app_default_changed': False,
                   'training_optimizer_VM_calls': 0}
        writer.json(OUT / 'supervisor.json', outcome)
        return 1


def supervise(expected_sha):
    # Interrupt the main supervisor even inside the extensive pure hash/gate loops.
    # BaseException unwinds run_chunk's exact-child finally before the failure receipt.
    need(threading.current_thread() is threading.main_thread(), 'Run the owned supervisor in its main thread')
    timer = threading.Timer(LIMITS['global_seconds'], _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    try:
        return supervise_body(expected_sha)
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-plan', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--worker', action='store_true')
    parser.add_argument('--plan-sha')
    parser.add_argument('--source-review')
    parser.add_argument('--chunk', type=int)
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.prepare_plan:
        need(args.source_review is not None, 'Independent final source review path required')
        print(O.B.json_bytes(prepare(args.source_review)).decode(), end=''); return 0
    need(type(args.plan_sha) is str and len(args.plan_sha) == 64 and all(c in '0123456789abcdef' for c in args.plan_sha),
         'Exact explicit tail plan SHA required')
    if args.verify_only:
        verify(args.plan_sha)
        print(O.B.json_bytes({'complete': True, 'remaining655': 655, 'chunks': 41, 'new_neural_calls': 0}).decode(), end=''); return 0
    if args.worker:
        return chunk_worker(args.plan_sha, args.chunk, args.token)
    return supervise(args.plan_sha)


if __name__ == '__main__':
    raise SystemExit(main())
