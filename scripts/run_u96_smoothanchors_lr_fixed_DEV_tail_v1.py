"""Distinct finite89 local inference tail; preserve original47 outputs exactly.

No training, automatic resume, overwritten status, changed model/input policy,
or app promotion. All status/resource receipts use exclusive numbered files.
The original failed48th reservation and unknown flag remain retained evidence.
"""
import argparse
import hashlib
import importlib
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

import run_u96_smoothanchors_lr_fixed_DEV_v1 as B

ROOT = B.ROOT
PREP = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_preparation'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_run_v1'
PLAN = PREP / 'tail_plan_v1.json'
ORIGINAL_PLAN_SHA = 'bb8a60f29630ddb21a2e9cea7e43b30c689b78f6474e6671a28c516e21bf1a6d'
ORIGINAL_SOURCE_SHA = 'd3afb3884b32aa4dee8bf4d6e4d52552af3df55f0864222cb7d7bb763a2a2b79'
PARTIAL_METADATA = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_metadata_audit/independent_partial_metadata_v1.json'
PARTIAL_METADATA_SHA = 'b8e99092904c974c2579123e8969befb711cbd51ee26da6cb443d183157e3eb7'
ARMS = B.ARMS
LIMITS = {**B.LIMITS, 'full_forward_attempts': 89, 'full_forward_returns': 89,
          'per_arm_forward_attempts': {'incumbent_U96': 44, 'postfusion_LR_u64': 45},
          'per_arm_forward_returns': {'incumbent_U96': 44, 'postfusion_LR_u64': 45}}


def need(ok, message):
    B.need(ok, message)


def pin(path):
    return B.pin(path)


def load(path):
    return B.load(path)


def bound(row):
    return B.bound(row)


def save(path, raw):
    need(path.resolve().is_relative_to(OUT) or path.resolve().is_relative_to(PREP), 'Owned tail output only')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve().is_relative_to(OUT):
        size = sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file())
        need(size + len(raw) <= LIMITS['maximum_output_bytes'], 'Reserve finite output bytes before write')
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return pin(path)


def save_json(path, value):
    return save(path, B.json_bytes(value))


def slot_order(original):
    return [{'case_id': row['id'], 'case_index': i, 'arm': arm, 'arm_index': j,
             'aggregate_output_index': i * 2 + j + 1}
            for i, row in enumerate(original['cases']) for j, arm in enumerate(ARMS)]


def original_evidence():
    original = B.verify_plan(ORIGINAL_PLAN_SHA)
    need(pin(ROOT / 'scripts/run_u96_smoothanchors_lr_fixed_DEV_v1.py')['sha256'] == ORIGINAL_SOURCE_SHA,
         'Original failed source immutable')
    need(pin(PARTIAL_METADATA)['sha256'] == PARTIAL_METADATA_SHA, 'Exact original partial metadata audit')
    partial = load(PARTIAL_METADATA)
    need(partial['complete'] and partial['independent_partial_metadata_audit_pass'] and
         partial['known_saved_output_records'] == 47 and partial['known_complete_case_records'] == 23 and
         partial['reserved_forward_outcome_unknown_retained'], 'Original47/23partial evidence')
    for row in partial['all_partial_files']: bound(row)
    need(len(partial['all_partial_files']) == 454, 'Exact454originalfiles retained')
    failure = load(bound(partial['failure']))
    pending = load(bound(partial['pending_status_exact_preserved']))
    status = load(bound(partial['status']))
    supervisor = load(bound(partial['supervisor']))
    need(failure['full_forward_attempts'] == pending['full_forward_attempts'] == 48 and
         failure['full_forward_returns'] == pending['full_forward_returns'] == 47 and
         failure['forward_outcome_unknown'] and pending['forward_outcome_unknown'] and
         status['full_forward_attempts'] == status['full_forward_returns'] == 47 and
         supervisor['worker_exit'] == 1 and supervisor['forward_outcome_unknown'], 'Original attempt/return/unknown truth')
    return original, partial


def prepare(args):
    original, partial = original_evidence()
    need(not PLAN.exists(), 'Fresh tail plan; no overwrite')
    admission_pin, review_pin = pin(Path(args.carried_admission).absolute()), pin(Path(args.source_review).absolute())
    admission, review = load(bound(admission_pin)), load(bound(review_pin))
    need(admission['complete'] and admission['carried47_pixel_admission_pass'] and
         admission['carried47_records_count'] == 47 and admission['completed_pair_cases'] == 23,
         'Independent actual fullpixeladmission for47carried outputs required')
    carried = admission['carried_outputs']; allslots = slot_order(original)
    need([(r['case_id'], r['arm']) for r in carried] ==
         [(s['case_id'], s['arm']) for s in allslots[:47]], 'Exact first47prefix carried, no selection')
    for row in carried:
        bound(row['record'])
        for item in row['inputs'].values(): bound(item)
        for key in ('native512', 'logits', 'LQ', 'unclipped', 'raw', 'png'): bound(row['outputs'][key])
    needed_sources = {Path(__file__).resolve().relative_to(ROOT).as_posix(),
                      'tests/test_u96_smoothanchors_lr_fixed_DEV_tail_v1.py'}
    need(review['complete'] and review['source_review_pass'] and
         needed_sources <= {row['path'] for row in review['sources']}, 'Independent current tail source review')
    for row in review['sources']: bound(row)
    plan = {'complete': True, 'format': 'fixed68-distinct89forward-tail-carry47-v1', 'status': 'SEALED_TAIL_UNRUN',
            'original_plan': pin(B.PLAN), 'original_failed_source': pin(ROOT / 'scripts/run_u96_smoothanchors_lr_fixed_DEV_v1.py'),
            'original_partial_metadata': pin(PARTIAL_METADATA), 'original454files': partial['all_partial_files'],
            'original_failure': partial['failure'], 'original_pending': partial['pending_status_exact_preserved'],
            'original_status': partial['status'], 'original_supervisor': partial['supervisor'],
            'original_unknown_reservation_retained': True, 'carried_admission': admission_pin,
            'tail_source_review': review_pin, 'tail_source': pin(Path(__file__).resolve()),
            'carried_outputs': carried, 'tail_slots': allslots[47:], 'all68_case_order': original['case_order'],
            'model': original['model'], 'head_file': original['head_file'], 'limits': LIMITS,
            'new_forward_slots': 89, 'carried_saved_output_slots': 47,
            'planned_aggregate_saved_returns': 136, 'planned_aggregate_reservations_including_failed_pre_call': 137,
            'geometry_delivery_and_model_policy': 'Exact original fixedDEVplan; no additionalblur/noise/alignment/sharpening or model/input changes.',
            'status_policy': 'Exclusive numbered append-only fsyncedJSONreceipts; no os.replace or mutable lateststatus; partialfiles retained.',
            'automatic_retry_resume': False, 'optimization_permitted': False, 'model_qualified': False,
            'app_default_changed': False, 'FINAL_identity_or_pixel_review': False, 'preparation_neural_calls': 0}
    need(len(plan['tail_slots']) == 89, 'Exactly89newslots')
    result = save_json(PLAN, plan)
    print(json.dumps({'complete': True, 'tail_plan': result, 'carried47': 47, 'new_forwards': 89, 'runtime_calls': 0}))


def verify(expected_sha):
    need(pin(PLAN)['sha256'] == expected_sha, 'Exact explicitly reviewed tail plan SHA')
    plan = load(PLAN); original, partial = original_evidence()
    need(plan['status'] == 'SEALED_TAIL_UNRUN' and plan['limits'] == LIMITS and
         plan['tail_slots'] == slot_order(original)[47:] and plan['all68_case_order'] == original['case_order'] and
         plan['model'] == original['model'] and plan['head_file'] == original['head_file'] and
         plan['original454files'] == partial['all_partial_files'] and plan['original_unknown_reservation_retained'],
         'Frozen tail model/order/data/evidence/resource contract')
    for key in ('original_plan', 'original_failed_source', 'original_partial_metadata', 'original_failure',
                'original_pending', 'original_status', 'original_supervisor', 'carried_admission', 'tail_source_review', 'tail_source'):
        bound(plan[key])
    need(plan['tail_source'] == pin(Path(__file__).resolve()), 'Current tail worker selfpin')
    return plan, original


class Journal:
    def __init__(self, directory):
        self.directory = directory; self.sequence = 0

    def append(self, value):
        self.sequence += 1
        return save_json(self.directory / f'{self.sequence:06d}.json', {'sequence': self.sequence, **value})


def worker(expected_sha):
    plan, original = verify(expected_sha)
    need(not OUT.exists(), 'Fresh exclusive tail namespace; no retry/resume')
    OUT.mkdir(); journal = Journal(OUT / 'journal'); resources = Journal(OUT / 'resources')
    start = time.monotonic(); phase = 'claimed_before_Torch'; active = None; last_status = None
    counters = {'forward_attempts': 0, 'forward_returns': 0, 'saved_outputs': 0,
                'per_arm': {a: {'attempts': 0, 'returns': 0, 'saved': 0} for a in ARMS},
                'carried_saved_outputs': 47, 'completed_tail_cases': 0, 'optimizer_updates': 0,
                'gradient_queries': 0, 'backward_calls': 0, 'optimizer_constructions': 0}
    def status(label):
        nonlocal last_status
        last_status = journal.append({'complete': label == 'complete', 'label': label, 'phase': phase,
            'slot': active, 'tail_counters': counters, 'tail_forward_outcome_unknown': counters['forward_attempts'] != counters['forward_returns'],
            'original_unknown_reservation_retained': True,
            'aggregate_returned_outputs': 47 + counters['forward_returns'],
            'aggregate_reservations_including_failed_pre_call': 48 + counters['forward_attempts'],
            'seconds': time.monotonic() - start, 'tail_plan_sha256': expected_sha,
            'model_qualified': False, 'no_automatic_retry_resume': True})
        return last_status
    def resource(label):
        import psutil
        value = {'label': label, 'seconds': time.monotonic() - start,
                 'worker_RSS_bytes': psutil.Process().memory_info().rss,
                 'available_RAM_bytes': psutil.virtual_memory().available,
                 'free_disk_bytes': shutil.disk_usage(ROOT).free,
                 'output_bytes': sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file())}
        resources.append(value)
        need(value['seconds'] <= LIMITS['internal_seconds'] and value['worker_RSS_bytes'] <= LIMITS['maximum_worker_RSS_bytes'] and
             value['available_RAM_bytes'] >= LIMITS['minimum_available_RAM_bytes'] and
             value['free_disk_bytes'] >= LIMITS['minimum_free_disk_bytes'] and
             value['output_bytes'] <= LIMITS['maximum_output_bytes'], 'Finite tail resource stop')
    def watchdog(seconds):
        timer = threading.Timer(seconds, lambda: os._exit(124)); timer.daemon = True; timer.start(); return timer
    try:
        status('claimed_before_Torch')
        save_json(OUT / 'started.json', {'complete': True, 'plan': pin(PLAN), 'runner': pin(Path(__file__).resolve()),
                    'carried47_admission': plan['carried_admission'], 'model_qualified': False})
        save_json(OUT / 'controls_zero_calls.json', {'complete': True, 'controls': original['zero_call_controls'], 'neural_calls': 0})
        need(ROOT.resolve() == Path.cwd().resolve() and (ROOT / 'third_party/codeformer/codeformer_arch.py').is_file(), 'Validated projectROOT before modelimports')
        sys.path.insert(0, str(ROOT))
        import torch
        import numpy as np
        from PIL import Image
        from torch.nn import functional as F
        torch.set_num_threads(4); need(torch.get_default_dtype() == torch.float32, 'CPU FP32 default')
        source = importlib.import_module('research.codeformer_u96_postfusion_lr_inference_v1')
        legacy = importlib.import_module('research_spatial_restoration_v1')
        from third_party.codeformer.codeformer_arch import CodeFormer
        phase = 'load_two_fresh_owned_frozen_CPU_models'; status('before_model_load'); resource('before_model_load')
        timer = watchdog(LIMITS['model_load_watchdog_seconds'])
        try:
            candidate = source.load_frozen_cpu_restorer(bound(plan['model']['r2']), bound(plan['model']['u96']),
                bound(plan['head_file']), inference_source_sha256=B.INFERENCE_SHA)
            with torch.no_grad(), torch.device('cpu'), torch.random.fork_rng(devices=[]):
                base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(source.SITES))
                r2 = torch.load(bound(plan['model']['r2']), map_location='cpu', weights_only=True, mmap=True)
                base.load_state_dict(r2['params_ema'], strict=True)
                incumbent = legacy.inference_class(torch)(base).float().eval().requires_grad_(False)
                u96 = torch.load(bound(plan['model']['u96']), map_location='cpu', weights_only=True, mmap=True)
                incumbent.adapters.load_state_dict({k.removeprefix('adapters.'): v for k, v in u96['adapter_state'].items()}, strict=True)
                del r2, u96
            need(incumbent.base is not candidate.base, 'Distinct owned bases')
        finally: timer.cancel()
        def invariants():
            need(len(incumbent.state_dict()) == 539 and all(not m.training for m in incumbent.modules()) and
                 all(v.device.type == 'cpu' and v.dtype == torch.float32 and not v.requires_grad and v.grad is None for v in incumbent.parameters()), 'Frozen incumbent539')
            need(legacy.base.state_sha(incumbent.base.state_dict(), torch) == source.R2_STATE and
                 legacy.base.state_sha({'adapters.' + k: v for k, v in incumbent.adapters.state_dict().items()}, torch) == source.U96_STATE,
                 'Exact retained U96 canonical515+24')
            need(all(not m._forward_hooks and not m._forward_pre_hooks for m in incumbent.modules()), 'Clean incumbent hooks')
            candidate.validate_invariants()
        invariants(); models = dict(zip(ARMS, (incumbent, candidate)))
        def png(path, value):
            stream = io.BytesIO(); Image.fromarray(value).save(stream, format='PNG'); return save(path, stream.getvalue())
        def array(path, value):
            need(value.dtype == np.float32 and np.isfinite(value).all(), 'Finite saved float32')
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); return save(path, stream.getvalue())
        cases = {}; carried = {}
        for row in plan['carried_outputs']:
            carried[(row['case_id'], row['arm'])] = load(bound(row['record']))
        for index, spec in enumerate(original['cases']):
            if index < 23:
                case_path = B.OUT / 'cases' / spec['id'] / 'case.json'
                cases[spec['id']] = load(case_path)
        current_case = None; case = None; sample = None
        for count, slot in enumerate(plan['tail_slots'], 1):
            active = slot; cid, arm = slot['case_id'], slot['arm']; spec = original['cases'][slot['case_index']]
            if current_case != cid:
                current_case = cid
                rgb, canvas, observed = B.prepare_pixels(spec)
                folder = OUT / 'cases' / cid; folder.mkdir(parents=True)
                if slot['case_index'] == 23:
                    oldfolder = B.OUT / 'cases' / cid
                    inputs = {key: pin(oldfolder / name) for key, name in [('original', 'original.png'), ('prepared', 'prepared.png'), ('support', 'observed_support.png'), ('target', 'target.png')]}
                else:
                    inputs = {'original': png(folder / 'original.png', rgb), 'prepared': png(folder / 'prepared.png', canvas),
                              'support': png(folder / 'observed_support.png', observed.astype(np.uint8) * 255)}
                    if spec['target']: inputs['target'] = save(folder / 'target.png', bound(spec['target']).read_bytes())
                case = {'id': cid, 'cohort': spec['cohort'], 'split': spec['split'], 'input_review': spec['input_review'],
                        'appearance_flag': spec.get('appearance_flag', False), 'geometry': spec['geometry'],
                        'inputs': inputs, 'paired_target': spec['target'], 'outputs': {},
                        'native_PSNR_SSIM_identity_metrics': None, 'model_qualified': False}
                if slot['case_index'] == 23:
                    case['outputs'][ARMS[0]] = {k: v for k, v in carried[(cid, ARMS[0])].items()
                                               if k not in ('complete', 'case_id', 'counters')}
                tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1).copy())[None] / 255
                sample = F.interpolate(tensor, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            phase = 'before_forward'; resource(arm + '_before'); status('before_reservation')
            need(counters['forward_attempts'] < 89 and counters['per_arm'][arm]['attempts'] < LIMITS['per_arm_forward_attempts'][arm], 'Bounded89slot reservation')
            counters['forward_attempts'] += 1; counters['per_arm'][arm]['attempts'] += 1
            phase = 'forward_reserved'; status('forward_reserved')
            timer = watchdog(LIMITS['individual_forward_watchdog_seconds'])
            try:
                with torch.no_grad(): triple = models[arm](sample, w=1., detach_16=True, adain=True)
            finally: timer.cancel()
            counters['forward_returns'] += 1; counters['per_arm'][arm]['returns'] += 1
            phase = 'forward_returned'; status('forward_returned')
            need(isinstance(triple, tuple) and len(triple) == 3, 'Stock tuple')
            for value, shape in zip(triple, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))):
                need(value.shape == shape and value.dtype == torch.float32 and value.device.type == 'cpu' and not value.requires_grad and torch.isfinite(value).all().item(), 'Finite exactCPUtuple')
            with torch.no_grad():
                native, logits, lq = triple
                unclipped = F.interpolate((native + 1) / 2, (256, 256), mode='bilinear', align_corners=False)
                raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
            raw_array = raw[0].permute(1, 2, 0).contiguous().numpy().copy()
            raw_array[~observed] = canvas.astype(np.float32)[~observed] / 255
            pixels = np.floor(raw_array * np.float32(255)).astype(np.uint8)
            armfolder = OUT / 'cases' / cid / arm; armfolder.mkdir()
            output = {'native512': array(armfolder / 'native512.npy', native.numpy()),
                      'logits': array(armfolder / 'logits.npy', logits.numpy()), 'LQ': array(armfolder / 'LQ.npy', lq.numpy()),
                      'unclipped': array(armfolder / 'unclipped256.npy', unclipped[0].permute(1, 2, 0).contiguous().numpy()),
                      'raw': array(armfolder / 'raw.npy', raw_array), 'png': png(armfolder / 'output.png', pixels),
                      'provenance': {'arm': arm, 'R2_file_sha256': B.R2_SHA, 'U96_file_sha256': B.U96_SHA,
                          'head_file_sha256': B.HEAD_SHA if arm == ARMS[1] else None,
                          'head_canonical_sha256': B.HEAD_STATE if arm == ARMS[1] else None,
                          'w': 1., 'detach16': True, 'AdaIN': True, 'CPU_FP32': True, 'display_processing': 'none',
                          'source_candidate_training_updates': 64 if arm == ARMS[1] else None, 'model_qualified': False,
                          'same_observed_geometry': True, 'outside_support_exact_observed_pixels': True,
                          'forward_sites': list(models[arm].last_forward_sites)}}
            invariants(); need(models[arm].last_forward_sites == source.SITES, 'Ordered all4feature sites')
            save_json(armfolder / 'record.json', {'complete': True, 'case_id': cid, **output,
                'tail_counters': counters, 'slot': slot, 'tail_plan_sha256': expected_sha})
            case['outputs'][arm] = output
            counters['saved_outputs'] += 1; counters['per_arm'][arm]['saved'] += 1
            phase = 'output_fully_saved'; status('output_fully_saved'); resource(arm + '_saved')
            if arm == ARMS[1]:
                need(set(case['outputs']) == set(ARMS), 'Complete pairedarm case')
                case['complete'] = True; save_json(OUT / 'cases' / cid / 'case.json', case); cases[cid] = case
                counters['completed_tail_cases'] += 1; phase = 'case_complete'; status('case_complete')
            print(json.dumps({'tail_slot': count, 'of': 89, 'case': cid, 'arm': arm, 'tail_returns': counters['forward_returns']}), flush=True)
            del triple, native, logits, lq, raw, unclipped
        need(counters['forward_attempts'] == counters['forward_returns'] == counters['saved_outputs'] == 89 and
             len(cases) == 68 and counters['completed_tail_cases'] == 45, 'Exactly89new+47carried results')
        phase = 'final_integrity'; invariants(); resource('final')
        for row in plan['original454files']: bound(row)
        for row in [plan['tail_source'], *original['sources'], plan['head_file'], *plan['model'].values()]: bound(row)
        comparison = {'complete': True, 'status': 'AGGREGATE_FIXED68_SAVED_OUTPUTS_PENDING_INDEPENDENT_AUDIT_VISUAL_REVIEW',
            'original_plan': plan['original_plan'], 'tail_plan': pin(PLAN), 'cases': [cases[cid] for cid in original['case_order']],
            'tail_counters': counters, 'carried_output_records': [r['record'] for r in plan['carried_outputs']],
            'new_saved_outputs': 89, 'carried_saved_outputs': 47, 'aggregate_returned_outputs': 136,
            'aggregate_reservations_including_failed_pre_call': 137, 'original_unknown_reservation_retained': True,
            'original_failure': plan['original_failure'], 'original_pending': plan['original_pending'],
            'seconds': time.monotonic() - start, 'no_optimization': True, 'app_default_changed': False,
            'paired_and_native_evidence_separate': True, 'model_qualified': False}
        save_json(OUT / 'comparison.json', comparison); phase = 'complete'; status('complete'); return 0
    except BaseException as error:
        save_json(OUT / 'failure.json', {'complete': True, 'status': 'TAIL_LOCAL_INFERENCE_STOP_NOT_QUALIFIED',
            'phase': phase, 'slot': active, 'tail_counters': counters, 'last_returned_status_receipt': last_status,
            'tail_forward_outcome_unknown': counters['forward_attempts'] != counters['forward_returns'],
            'original_unknown_reservation_retained': True, 'error': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'seconds': time.monotonic() - start,
            'automatic_retry_resume': False, 'model_qualified': False})
        raise


def journal_readback(directory):
    parsed = []; partial = []
    for path in sorted(directory.glob('*.json')):
        try:
            value = load(path)
            need(value['sequence'] == int(path.stem), 'Numbered journal receipt identity')
            parsed.append((pin(path), value))
        except (ValueError, OSError, KeyError):
            # An interrupted exclusive write may leave an empty file. Preserve
            # its exact zero bytes instead of applying the nonempty-source pin
            # policy or guessing which reserved operation completed.
            info = path.lstat()
            need(path.resolve().is_relative_to(OUT) and stat.S_ISREG(info.st_mode) and
                 not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024 and
                 0 <= info.st_size <= 1024 ** 2, 'Bounded ordinary partial journal file')
            raw = path.read_bytes()
            partial.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest()})
    return parsed, partial


def supervisor_outcome(code, timed_out, last, partial, failure, unreadable_failure):
    durable = last['tail_counters'] if last else None
    failed = failure['tail_counters'] if failure else None
    counters = failed if failed is not None else durable
    complete = bool(code == 0 and not timed_out and last and last['complete'] and
                    counters and counters['forward_attempts'] == counters['forward_returns'] ==
                    counters['saved_outputs'] == 89 and not failure and not unreadable_failure and not partial)
    unknown = bool(partial or unreadable_failure or counters is None or
                   (counters and counters['forward_attempts'] != counters['forward_returns']) or
                   (last and last['tail_forward_outcome_unknown']) or
                   (failure and failure['tail_forward_outcome_unknown']) or
                   (not complete and failure is None))
    return {'last_durable_tail_counters': durable, 'failure_tail_counters': failed,
            'tail_counters': counters, 'tail_forward_outcome_unknown': unknown,
            'complete89_tail_observed': complete}


def supervise(expected_sha):
    verify(expected_sha); need(not OUT.exists(), 'Fresh tail namespace')
    log, receipt = PREP / 'tail_worker_v1.log', PREP / 'tail_supervisor_v1.json'
    need(not log.exists() and not receipt.exists(), 'Fresh exclusive supervisor files')
    start = time.monotonic(); timed_out = False
    with log.open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--plan-sha', expected_sha],
                                   cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
        try: code = process.wait(timeout=LIMITS['external_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True; process.terminate()
            try: code = process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); code = process.wait(timeout=10)
    parsed, partial = journal_readback(OUT / 'journal')
    last_pin, last = parsed[-1] if parsed else (None, None)
    failure_path = OUT / 'failure.json'; failure = None; unreadable_failure = False
    failure_pin = None; failure_read_error = None
    if failure_path.exists():
        try:
            info = failure_path.lstat()
            need(failure_path.resolve().is_relative_to(OUT) and stat.S_ISREG(info.st_mode) and
                 not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024 and
                 0 <= info.st_size <= 1024 ** 2, 'Bounded owned failure receipt')
            raw = failure_path.read_bytes()  # One bounded read, never an implicit retry.
            failure_pin = {'path': failure_path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
                           'sha256': hashlib.sha256(raw).hexdigest()}
            failure = B.strict_json(raw)
        except (ValueError, OSError) as error:
            unreadable_failure = True
            failure_read_error = {'path': failure_path.relative_to(ROOT).as_posix(),
                                  'error': type(error).__name__, 'message': str(error),
                                  'unreadable_bytes_not_given_an_invented_hash': failure_pin is None}
    result = {'complete': True, 'worker_exit': code, 'timeout': timed_out, 'seconds': time.monotonic() - start,
              'tail_plan_sha256': expected_sha, 'last_durable_status': last_pin, 'partial_journal_files': partial,
              'failure_receipt': failure_pin, 'failure_receipt_unreadable': unreadable_failure,
              'failure_receipt_read_error': failure_read_error,
              **supervisor_outcome(code, timed_out, last, partial, failure, unreadable_failure),
              'original_unknown_reservation_retained': True, 'model_qualified': False, 'no_automatic_retry_resume': True}
    save_json(receipt, result); print(json.dumps(result)); return 0 if code == 0 and not timed_out else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-plan', action='store_true'); mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--run', action='store_true'); mode.add_argument('--worker', action='store_true')
    parser.add_argument('--plan-sha'); parser.add_argument('--carried-admission'); parser.add_argument('--source-review')
    args = parser.parse_args()
    if args.prepare_plan:
        need(args.carried_admission and args.source_review, 'Actual carried47pixeladmission and current tail peer review required')
        return prepare(args)
    need(type(args.plan_sha) is str and len(args.plan_sha) == 64, 'Explicit tail plan SHA')
    if args.verify_only:
        verify(args.plan_sha); print(json.dumps({'complete': True, 'carried47': 47, 'new89': 89, 'neural_calls': 0})); return 0
    return worker(args.plan_sha) if args.worker else supervise(args.plan_sha)


if __name__ == '__main__': raise SystemExit(main())
