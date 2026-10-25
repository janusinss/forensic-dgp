"""Independent CPU value/history audit of the exact failed fusion V1 return.

Run only after the strict closed import. No returned Python, model construction,
optimizer construction, neural forward, gradient, VM or app action is performed.
Torch only decodes freshly hash-checked tensors with weights_only=True on CPU.
Saved history consistency is not independent replay of live CUDA Adam/gradients.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_filter_fusion_return_review_20261019_v1'
DEST = ROOT / 'outputs/codeformer_filter_fusion_training_v1_return'
RUN = 'outputs/filter_fusion_training_v1/'
PROTOCOL = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
MANIFEST = '8c6463a7fad4501ce55b4be2a07af98ab48b077951533578b77218de68006173'
SEED = 20261018
SITES = ('32', '64', '128', '256')
DIGESTS = {
    'original': '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
    'parent': '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
    'r2': 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'}
CAPS = {'candidate_snapshot_forwards': 1434, 'candidate_training_forwards': 604,
    'original_clear_teacher_forwards': 55, 'R2_dark_teacher_forwards': 55,
    'R2_baseline_forwards': 478, 'original_baseline_forwards': 478,
    'parent_baseline_forwards': 478, 'candidate_TRAIN_guard_forwards': 48,
    'parity_candidate_forwards': 1, 'parity_R2_forwards': 1,
    'filter_witness_forwards': 3, 'clear_witness_forwards': 2,
    'dark_witness_forwards': 2, 'backward_calls': 604,
    'optimizer_step_attempts': 151, 'gradient_queries': 7}
QUERIES = ('rows_complete_divided', 'rows_spatial_unweighted',
    'rows_adjacent_unweighted', 'local_mosaic_complete_divided',
    'local_blur_complete_divided', 'clear_original_complete_divided',
    'dark_R2_initial_complete_divided')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def parse(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read(path, cap=4 * 1024**2):
    need(Path(path).stat().st_size <= cap, 'Metadata exceeds read cap')
    return parse(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            state = item.lstat()
            need(not stat.S_ISLNK(state.st_mode) and not getattr(state, 'st_file_attributes', 0) & 1024,
                'Linked/reparse evidence forbidden')
    state = path.lstat()
    need(stat.S_ISREG(state.st_mode) and state.st_nlink == 1 and
        path.resolve(strict=True).is_relative_to(ROOT), 'Exact singly-linked regular workspace evidence required')
    return path.resolve(strict=True)


class Evidence:
    def __init__(self, receipt):
        self.receipt_path = regular(receipt); self.receipt_sha = sha(self.receipt_path)
        value = read(self.receipt_path, 8 * 1024**2)
        need(value.get('complete') is True and value['archive_sha256'] == ARCHIVE and
            value['protocol_sha256'] == PROTOCOL and value['manifest_sha256'] == MANIFEST and
            value['archive_bytes'] == 4408715848 and value['regular_files'] == 9081 and
            value['payload_bytes'] == 4905597698, 'Exact closed failed-return import required')
        self.root = Path(value['import_root']).resolve(strict=True)
        need(self.root == DEST.resolve(strict=True), 'Exact imported namespace required')
        self.hashes, self.sizes = value['files_sha256'], value['files_bytes']
        need(set(self.hashes) == set(self.sizes) and len(self.hashes) == 9081 and
            sum(self.sizes.values()) == 4905597698, 'Complete imported member maps required')
        self.read_pins = {}
        need(self.hashes[RUN + 'export_manifest.json'] == MANIFEST, 'Manifest digest differs')
        manifest = self.json(RUN + 'export_manifest.json')
        expected = {k: {'bytes': self.sizes[k], 'sha256': v} for k, v in self.hashes.items()
                    if k != RUN + 'export_manifest.json'}
        need(manifest['files'] == expected, 'Manifest and full import maps differ')
        self.p = self.json('protocol.json')
        need(self.hashes['protocol.json'] == PROTOCOL and self.p['call_caps'] == CAPS and
            self.p['snapshots'] == [0, 50, 151] and self.p['guard_snapshots'] == [0, 20] and
            self.p['automatic_resume'] is False and self.p['automatic_training_launch'] is False,
            'Sealed finite manual protocol differs')
        for name, value in self.p['files_sha256'].items():
            need(self.hashes[name] == value and self.sizes[name] == self.p['files_bytes'][name],
                'Static source/provenance/input binding differs: ' + name)

    def file(self, name):
        need(isinstance(name, str) and name in self.hashes and '\\' not in name and ':' not in name,
            'Unbound or unsafe member')
        path = PurePosixPath(name)
        need(path.as_posix() == name and not path.is_absolute() and
            all(p not in ('', '.', '..') for p in path.parts), 'Noncanonical member path')
        actual = regular(self.root.joinpath(*path.parts))
        need(actual.is_relative_to(self.root), 'Evidence escaped imported root')
        before = actual.stat()
        need(before.st_size == self.sizes[name] and sha(actual) == self.hashes[name],
            'Changed imported bytes: ' + name)
        after = actual.stat()
        need((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hashing')
        self.read_pins[name] = {'bytes': before.st_size, 'sha256': self.hashes[name],
            'mtime_ns': before.st_mtime_ns, 'ino': before.st_ino}
        return actual

    def json(self, name):
        return read(self.file(name))

    def lines(self, name):
        path = self.file(name)
        need(path.stat().st_size <= 16 * 1024**2, 'JSONL cap exceeded')
        payload = path.read_bytes()
        need(payload.endswith(b'\n') and b'\r' not in payload, 'Complete LF JSONL required')
        return [parse(line) for line in payload.splitlines()]

    def pin(self, name):
        return {'path': name, 'bytes': self.sizes[name], 'sha256': self.hashes[name]}

    def close(self):
        need(sha(self.receipt_path) == self.receipt_sha, 'Import receipt changed during audit')
        for name, pin in self.read_pins.items():
            state = (self.root / name).stat()
            need((state.st_size, state.st_mtime_ns, state.st_ino) ==
                (pin['bytes'], pin['mtime_ns'], pin['ino']), 'Read evidence changed after hashing')
        return {'files_freshly_hashed_before_read': len(self.read_pins),
            'all_read_file_identities_rechecked': True, 'import_receipt_sha256': self.receipt_sha}


def sampler_state(schedule, update):
    return {'format': 'codeformer-filter-fusion-sampler-state-v2', 'seed': SEED,
        'schedule_sha256': digest(schedule), 'committed_updates': update,
        'committed_microbatches': 4 * update}


def expected_trace(p, schedule, evaluation, cases):
    plan = []
    def add(kind, **context):
        plan.append((kind, context))
    for alias, counter in (('original', 'original_baseline_forwards'),
        ('parent', 'parent_baseline_forwards'), ('r2', 'R2_baseline_forwards')):
        for row in evaluation:
            add(counter, id=row['id'], alias=alias)
    for role, counter in (('clear', 'original_clear_teacher_forwards'), ('dark', 'R2_dark_teacher_forwards')):
        selected = sorted({s['id'] for u in schedule['updates'] for s in u['slots'] if s['role'] == role})
        need(len(selected) == 55, 'All55 role teachers required')
        for identifier in selected:
            add(counter, id=identifier, role=role)
    w = p['witness_cases']
    add('parity_candidate_forwards', witness='parity_candidate', id=w['clear'])
    add('parity_R2_forwards', witness='parity_R2', id=w['clear'])
    for label, query in (('rows', QUERIES[0]), ('local_mosaic', QUERIES[3]), ('local_blur', QUERIES[4])):
        add('filter_witness_forwards', witness=label, id=w[label])
        add('gradient_queries', purpose=query, tensors=72)
        if label == 'rows':
            for query in QUERIES[1:3]:
                add('gradient_queries', purpose=query, tensors=72)
    for role, counter, teacher, query in (('clear', 'clear_witness_forwards', 'original', QUERIES[5]),
        ('dark', 'dark_witness_forwards', 'R2', QUERIES[6])):
        add(counter, witness=role + '_candidate', id=w[role])
        add(counter, witness=role + '_' + teacher, id=w[role])
        add('gradient_queries', purpose=query, tensors=72)
    def snapshot(update):
        for row in evaluation:
            add('candidate_snapshot_forwards', snapshot=update, id=row['id'])
    def guard(update):
        for row in p['guard_cases']:
            add('candidate_TRAIN_guard_forwards', TRAIN_guard=update, id=row['id'], role=row['role'])
    snapshot(0); guard(0)
    for update in schedule['updates'][:50]:
        need(update['update'] in range(1, 51), 'Exact50 committed schedule prefix required')
        for slot in update['slots']:
            role = 'filter' if slot['role'].startswith('filter_') else slot['role']
            need(slot['id'] in cases and cases[slot['id']]['reference_id'] == slot['reference_id'],
                'Training identity/schedule join differs')
            add('candidate_training_forwards', update=update['update'], slot=slot['slot'], role=role, id=slot['id'])
            add('backward_calls', update=update['update'], slot=slot['slot'])
        add('optimizer_step_attempts', update=update['update'])
        if update['update'] == 20:
            guard(20)
    snapshot(50)
    return plan


def gradient_summary(value, *, nonzero_sites=True):
    need(value['queried_tensors'] == 72 and set(value['sites']) == set(SITES) and
        len(value['parameters']) == 72, 'Exact72 gradient/step summary required')
    names = set(fusion_shapes())
    need(set(value['parameters']) == names, 'Gradient summary names differ')
    for name, row in value['parameters'].items():
        need(row['is_none'] is False and type(row['nonzero']) is bool and
            math.isfinite(row['l2']) and row['l2'] >= 0 and
            math.isfinite(row['maximum_absolute']) and row['maximum_absolute'] >= 0 and
            bool(row['maximum_absolute'] > 0) == row['nonzero'] and
            bool(row['l2'] > 0) == row['nonzero'] and re.fullmatch('[0-9a-f]{64}', row['sha256']),
            'Disconnected/nonfinite gradient or step readout: ' + name)
    for site, row in value['sites'].items():
        selected = [r for n, r in value['parameters'].items() if n.split('.')[1] == site]
        need(row['tensors'] == len(selected) == 18 and row['none'] == 0 and
            row['nonzero'] == sum(r['nonzero'] for r in selected) and
            math.isfinite(row['l2']) and row['l2'] >= 0 and
            math.isfinite(row['maximum_absolute']) and row['maximum_absolute'] >= 0,
            'Finite18tensor site summary required')
        need(not nonzero_sites or row['nonzero'] > 0 and row['l2'] > 0,
            'Every fusion site needs a nonzero actual improvement gradient')


def audit_history(e):
    p = e.p; schedule = e.json(p['data']['schedule_path'])
    need(schedule['max_updates'] == 151 and schedule['seed'] == SEED and len(schedule['updates']) == 151 and
        [r['update'] for r in schedule['updates']] == list(range(1, 152)) and
        all([s['role'] for s in r['slots']] == ['filter_a', 'filter_b', 'clear', 'dark'] and
            [s['slot'] for s in r['slots']] == [1, 2, 3, 4] for r in schedule['updates']),
        'Exact151update604slot sampler required')
    bundle = e.json(p['data']['bundle_path']); cases = bundle['case_by_id']
    need(len(cases) == 411 and all(r['split'] == 'TRAIN' for r in cases.values()), 'All411 TRAIN role cases required')
    eval_bundle = e.json(p['evaluation_path'])
    evaluation = eval_bundle['cases']
    need(len(evaluation) == len({r['id'] for r in evaluation}) == 478, 'Full478 evaluation ordering required')
    failure = e.json(RUN + 'failure.json')
    need(failure['complete'] is True and failure['failure'] is True and failure['protocol_sha256'] == PROTOCOL and
        failure['completed_training_updates'] == 50 and failure['partial_microbatches'] == 0 and
        failure['optimizer_inflight'] is False and failure['state_checkpoint_complete'] is True and
        failure['training_success_not_implied'] is True and failure['model_qualified'] is False and
        failure['error_type'] == 'ValueError' and failure['error'] ==
        'New/worsened preservation or R2 dark retention failure; retain actual stop' and
        'worker.py' in failure['traceback'] and 'snapshot(current)' in failure['traceback'],
        'Actual snapshot50 quality stop differs')
    need(RUN + 'run_results.json' not in e.hashes and RUN + 'snapshot_0151/checkpoint.pt' not in e.hashes,
        'Unexpected completed151 outcome')
    trace = expected_trace(p, schedule, evaluation, cases)
    operations = e.lines(RUN + 'operations.jsonl')
    need(len(operations) == 2 * len(trace), 'Every reservation needs exactly one completed return')
    counts = dict.fromkeys(CAPS, 0)
    for index, (kind, context) in enumerate(trace, 1):
        counts[kind] += 1
        reserved, returned = operations[2 * index - 2: 2 * index]
        need(reserved == {'event': 'reserved', 'sequence': index, 'kind': kind,
            'count': counts[kind], 'context': context}, 'Reserved operation order/context differs: ' + str(index))
        returned_context = {'finite': True} if kind.endswith('_forwards') else {
            'returned': True} if kind == 'optimizer_step_attempts' else {'completed': True}
        need(returned == {'event': 'returned', 'reservation': index, 'context': returned_context},
            'Missing/different completed operation return: ' + str(index))
    need(counts == failure['counts'] and all(counts[k] <= CAPS[k] for k in CAPS),
        'Durable trace/failure counters differ')
    teacher_manifest = e.json(RUN + 'teacher/manifest.json')
    need(teacher_manifest['complete'] is True and teacher_manifest['unique_teacher_forwards'] == 110 and
        teacher_manifest['reuse_does_not_add_neural_calls'] is True and
        teacher_manifest['clamp_before_resize_evidence_not_derived_from_U'] is True,
        'Finite cached behavior teacher manifest differs')
    teachers = {(r['role'], r['id']): r for r in teacher_manifest['targets']}
    need(len(teachers) == len(teacher_manifest['targets']) == 110 and
        Counter(role for role, _ in teachers) == {'clear': 55, 'dark': 55},
        'Exactly55 source-bound teachers per behavior role required')
    for row in teachers.values():
        for view in ('U', 'RAW', 'PNG'):
            need(row[view] == e.pin(row[view]['path']), 'Cached teacher exact byte pin differs')
    names = set(fusion_shapes()); loss_records = []; commit_records = []
    for update in schedule['updates'][:50]:
        number = update['update']; lr = 1e-5 * min(number, 10) / 10
        records = []
        for slot in update['slots']:
            row = e.json(RUN + f'training/u{number:04d}_s{slot["slot"]}.json')
            case = cases[slot['id']]
            role = 'filter' if slot['role'].startswith('filter_') else slot['role']
            need(row['update'] == number and row['slot'] == slot['slot'] and row['role'] == role and
                row['id'] == slot['id'] and row['LR'] == lr and row['input'] == case['input'] and
                row['padding_duplicate'] == slot['padding_duplicate'] and
                row['footprint'] == (case.get('local_rectangle_xyxy_half_open') if role == 'filter' else None) and
                row['target'] == (case['target256'] if role == 'filter' else teachers[(role, slot['id'])]['U']),
                'Actual slot/source/LR/padding/footprint differs')
            terms = row['unweighted_terms']; scalar = row['scalar']
            need(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in
                [scalar, row['divided_scalar'], *terms.values()]) and row['divided_scalar'] == scalar / 4,
                'Finite divided slot scalar required')
            expected_terms = {'spatial_rgb_l1', 'adjacent_residual_l1'} if role == 'filter' else {'teacher_rgb_l1'}
            need(set(terms) == expected_terms, 'Actual loss terms differ')
            weighted = terms['spatial_rgb_l1'] + .25 * terms['adjacent_residual_l1'] if role == 'filter' else 2 * terms['teacher_rgb_l1']
            # FP32 optimization adds these on device, then serializes scalars.
            # This numerical witness does not change any delivered quality gate.
            need(abs(scalar - weighted) <= 4 * 2**-23 * max(1., abs(weighted)),
                'Weighted optimizing loss coefficient witness differs')
            need(set(row['upstream']) == {'logits', 'lq', 'tokens'} and
                all(re.fullmatch('[0-9a-f]{64}', v) for v in row['upstream'].values()),
                'Source-bound upstream fingerprints required')
            parity = row['U_readout_parity']
            need(parity['pass'] is True and parity['maximum_absolute_difference'] <= parity['absolute_bound'] and
                parity['absolute_bound'] == 8 * 2**-23 * max(1., parity['maximum_absolute_native_RGB']) and
                parity['delivered_quality_gate_tolerance_changed'] is False,
                'Prospective deterministic U arithmetic witness failed')
            records.append(row); loss_records.append(row)
        commit = e.json(RUN + f'training/u{number:04d}_commit.json')
        need(commit['complete'] is True and commit['update'] == number and commit['LR'] == lr and
            commit['slot_scalars'] == [r['scalar'] for r in records] and
            commit['sampler_state_before_sha256'] == digest(sampler_state(schedule, number - 1)) and
            commit['sampler_state_after_sha256'] == digest(sampler_state(schedule, number)),
            'Actual50 commit/sampler/LR evidence differs')
        gradient_summary(commit['gradients']); gradient_summary(commit['step_changes'], nonzero_sites=False)
        need(set(commit['gradients']['parameters']) == names, 'Exact fusion commit names required')
        commit_records.append(commit)
    actual_training_files = {n for n in e.hashes if n.startswith(RUN + 'training/')}
    expected_training_files = {RUN + f'training/u{u:04d}_s{s}.json' for u in range(1, 51) for s in range(1, 5)} | {
        RUN + f'training/u{u:04d}_commit.json' for u in range(1, 51)}
    need(actual_training_files == expected_training_files, 'Unexpected/missing physical training slot/commit evidence')
    gates0 = e.json(RUN + 'snapshot_0000/gates.json'); gates50 = e.json(RUN + 'snapshot_0050/gates.json')
    need(gates0['initial_R2_full478_parity']['pass'] is True and
        not (gates50['interim']['pass'] and gates50['R2_dark_retention']['pass']),
        'Actual quality-stop gate record differs')
    need(gates0['preservation']['quality_gate_thresholds_changed'] is False and
        gates50['preservation']['quality_gate_thresholds_changed'] is False,
        'Original quality requirements changed')
    for at in ('0000', '0020'):
        need(e.json(RUN + f'guard_{at}/gates.json')['pass'] is True, 'Earlier repeated TRAIN guard did not pass')
    runtime = e.json(RUN + 'runtime.json')
    need(runtime['GPU'] == 'NVIDIA L4' and runtime['manual_only'] is True and runtime['FP32'] is True and
        runtime['TF32'] is False and runtime['deterministic_algorithms'] is True and
        runtime['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8', 'Manual L4 FP32 deterministic runtime differs')
    resources = e.lines(RUN + 'resource.jsonl'); previous = -1.
    resource_contexts = ['before Torch import']
    for kind, context in trace:
        if not kind.endswith('_forwards'):
            continue
        if kind == 'candidate_snapshot_forwards' and context['id'] == evaluation[0]['id']:
            resource_contexts.append('before checkpoint')
        resource_contexts.extend(('before ' + kind, 'after ' + kind))
    need([r['context'] for r in resources] == resource_contexts,
        'Missing/extra or reordered resource guard samples')
    for row in resources:
        need(row['event'] == 'resource_sample' and math.isfinite(row['seconds']) and
            previous <= row['seconds'] <= 4800 and 0 <= row['RSS_bytes'] <= 8 * 1024**3 and
            0 <= row['peak_RSS_bytes'] <= 8 * 1024**3 and
            0 <= row['CUDA_reserved_bytes'] <= 22 * 1024**3 and row['disk_free_bytes'] >= 4 * 1024**3,
            'Recorded finite runtime/RSS/GPU/disk guard failed')
        previous = row['seconds']
    # External final closure records exported bytes and is read independently of returned code.
    external = regular(ROOT / 'outputs/cctv-codeformer-filter-fusion-training-v1-supervisor-final.json')
    external_sha = sha(external); final = read(external)
    need(final['complete'] is True and final['protocol_sha256'] == PROTOCOL and
        final['archive'] == {'path': 'cctv-codeformer-filter-fusion-training-v1-results.tar.gz',
            'sha256': ARCHIVE, 'bytes': 4408715848} and len(final['stages']) == 3,
        'External supervisor/archive closure differs')
    for stage, flag, code, cap in zip(final['stages'], ('--verify-transfer', '--run', '--export'), (0, 1, 0), (300, 5300, 600)):
        need(stage['stage'] == flag and stage['exit_code'] == code and stage['external_cap_seconds'] == cap and
            stage['seconds'] <= cap and stage['outer_watchdog_expired'] is False and
            stage['worker_terminal'] is True and stage['child_process_group_terminal'] is True and
            stage['supervisor_exception'] is None, 'Bounded terminal supervisor stage differs')
    # Export finishes before final supervisor closure; finished.json is not an
    # archive member. Reconstruct the exact documented compact receipt instead.
    finished = {'complete': True, 'protocol_sha256': PROTOCOL, 'worker_terminal': True,
        'stages': final['stages'], 'model_qualified': False,
        'training_or_export_completion_not_model_acceptance': True}
    finished_payload = (json.dumps(finished, separators=(',', ':'), allow_nan=False) + '\n').encode()
    need(final['finished'] == {'path': 'outputs/supervision/finished.json',
        'sha256': hashlib.sha256(finished_payload).hexdigest(), 'bytes': len(finished_payload)},
        'External completed supervisor receipt does not reconstruct exactly')
    need(sha(external) == external_sha, 'External closure changed during read')
    witness = e.json(RUN + 'witness/receipt.json')
    need(witness['complete'] is True and witness['forward_calls'] == 9 and witness['gradient_queries'] == 7 and
        witness['optimizer_updates'] == 0 and witness['initial_parity']['pass'] is True and
        witness['component_sum']['pass'] is True and witness['restoration']['pass'] is True and
        witness['source_bound_gradients_not_independent_replay'] is True,
        'Initial gradient/parity/restoration witness differs')
    for query in (QUERIES[0], QUERIES[3], QUERIES[4], QUERIES[5], QUERIES[6]):
        gradient_summary(witness[query]['gradients'], nonzero_sites=query != QUERIES[6])
    need(witness[QUERIES[6]]['scalar'] == 0., 'Initially identical R2 dark anchor was not zero')
    return {'failure': failure, 'counts': counts, 'schedule': schedule, 'gates0': gates0, 'gates50': gates50,
        'report': {'durable_operation_reservations': len(trace), 'durable_operation_returns': len(trace),
            'all_reserved_operations_returned_in_exact_source_order': True, 'actual_counts': counts,
            'physical_training_slots': len(loss_records), 'committed_updates': len(commit_records),
            'learning_rate_first': commit_records[0]['LR'], 'learning_rate_update10': commit_records[9]['LR'],
            'learning_rate_last': commit_records[-1]['LR'], 'original_stop_retained': failure['error'],
            'resource_samples': len(resources), 'last_recorded_worker_seconds': previous,
            'peak_recorded_RSS_bytes': max(r['peak_RSS_bytes'] for r in resources),
            'peak_recorded_CUDA_reserved_bytes': max(r['CUDA_reserved_bytes'] for r in resources),
            'minimum_recorded_disk_free_bytes': min(r['disk_free_bytes'] for r in resources),
            'supervisor_stage_seconds': {s['stage']: s['seconds'] for s in final['stages']},
            'external_supervisor_sha256': external_sha, 'initial478_R2_parity_record_pass': True,
            'witness_metadata_pass': True, 'guard20_pass': True,
            'saved_gate_booleans': {name: gates50[name]['pass'] for name in
                ('preservation', 'parent_dark_retention', 'R2_dark_retention', 'interim')},
            'saved_pixel_gate_recomputation_scope': 'Separate independent full saved-pixel audit; this auditor binds authoritative gate bytes only.',
            'not_live_CUDA_gradient_or_Adam_replay': True}}


def fusion_shapes():
    result = {}
    for site, channels in zip(SITES, (256, 256, 128, 128)):
        shapes = {'encode_enc.norm1.weight': (2 * channels,), 'encode_enc.norm1.bias': (2 * channels,),
            'encode_enc.conv1.weight': (channels, 2 * channels, 3, 3), 'encode_enc.conv1.bias': (channels,),
            'encode_enc.norm2.weight': (channels,), 'encode_enc.norm2.bias': (channels,),
            'encode_enc.conv2.weight': (channels, channels, 3, 3), 'encode_enc.conv2.bias': (channels,),
            'encode_enc.conv_out.weight': (channels, 2 * channels, 1, 1), 'encode_enc.conv_out.bias': (channels,)}
        for branch in ('scale', 'shift'):
            for layer in (0, 2):
                shapes[f'{branch}.{layer}.weight'] = (channels, channels, 3, 3)
                shapes[f'{branch}.{layer}.bias'] = (channels,)
        result.update({f'fuse_convs_dict.{site}.{name}': shape for name, shape in shapes.items()})
    need(len(result) == 72 and sum(math.prod(s) for s in result.values()) == 10659584,
        'Independent exact72 fusion schema differs')
    return result


def tensor_digest(state, torch):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        result.update(len(header).to_bytes(8, 'big')); result.update(header)
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def tree_digest(value, torch):
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            header = json.dumps(['tensor', str(array.dtype), list(array.shape)], separators=(',', ':')).encode()
            result.update(len(header).to_bytes(8, 'big')); result.update(header)
            result.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict[')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
            result.update(b']')
        elif isinstance(item, (list, tuple)):
            result.update((type(item).__name__ + '[').encode())
            for child in item:
                visit(child)
            result.update(b']')
        else:
            payload = json.dumps([type(item).__name__, item], separators=(',', ':'), allow_nan=False).encode()
            result.update(len(payload).to_bytes(8, 'big')); result.update(payload)
    visit(value)
    return result.hexdigest()


def validate_model(state, reference, torch):
    need(type(state) is dict or isinstance(state, dict), 'Model state dictionary required')
    need(len(state) == 515 and set(state) == set(reference), 'Exact515 parameter schema required')
    shapes = fusion_shapes(); count = 0
    for name, value in state.items():
        need(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
            value.shape == reference[name].shape and not value.requires_grad and torch.isfinite(value).all().item(),
            'Finite515 FP32 CPU model schema differs: ' + name)
        if name in shapes:
            need(tuple(value.shape) == shapes[name], 'Exact fusion tensor shape differs')
        else:
            need(not name.startswith('fuse_convs_dict.'), 'Undeclared fusion tensor')
        count += value.numel() * value.element_size()
    need(count == 376450828, 'Exact515 FP32 model value bytes differ')
    return tensor_digest(state, torch)


def checkpoint_audit(value, r2, label, e, metadata, torch):
    update = 0 if label == 'update0' else 50
    need(set(value) == {'format', 'protocol_sha256', 'progress', 'counts', 'counter_caps',
        'automatic_resume_safe', 'gradient_buffers_not_serialized',
        'partial_or_inflight_checkpoint_not_a_resumable_full_autograd_state', 'model_qualified',
        'model', 'optimizer', 'sampler', 'CPU_RNG', 'CUDA_RNG'} and
        value['format'] == 'codeformer-filter-fusion-training-state-v1' and
        value['protocol_sha256'] == PROTOCOL and value['counter_caps'] == CAPS and
        value['automatic_resume_safe'] is False and value['gradient_buffers_not_serialized'] is True and
        value['partial_or_inflight_checkpoint_not_a_resumable_full_autograd_state'] is True and
        value['model_qualified'] is False, 'Saved checkpoint finite/safety schema differs')
    progress = value['progress']
    need(set(progress) == {'completed_training_updates', 'partial_microbatches', 'optimizer_inflight',
        'state_checkpoint', 'state_checkpoint_complete'} and
        progress['completed_training_updates'] == update and progress['partial_microbatches'] == 0 and
        progress['optimizer_inflight'] is False, 'Saved committed progress differs')
    previous = None if label == 'update0' else e.pin(RUN + (
        'snapshot_0000/checkpoint.pt' if label == 'update50' else 'snapshot_0050/checkpoint.pt'))
    need(progress['state_checkpoint'] == previous and
        progress['state_checkpoint_complete'] is (None if previous is None else True),
        'Checkpoint metadata must retain exact prior physical state at serialization')
    counts = metadata['counts'].copy()
    if label == 'update0':
        for key in ('candidate_snapshot_forwards', 'candidate_training_forwards',
            'candidate_TRAIN_guard_forwards', 'backward_calls', 'optimizer_step_attempts'):
            counts[key] = 0
    elif label == 'update50':
        counts['candidate_snapshot_forwards'] = 478
    need(value['counts'] == counts, 'Checkpoint-before-snapshot counter boundary differs')
    model = value['model']; model_sha = validate_model(model, r2, torch)
    shapes = fusion_shapes(); changed = []
    for name, tensor in model.items():
        exact = torch.equal(tensor, r2[name])
        need(name in shapes or exact, 'One of443 frozen R2 tensor values changed')
        need(update != 0 or exact, 'Snapshot0 is not exact R2')
        if not exact:
            changed.append(name)
    need((update == 0 and not changed) or (update == 50 and changed), 'Actual changed fusion values required')
    optimizer = value['optimizer']
    need(set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1,
        'Exact one-group fresh Adam state schema required')
    group = optimizer['param_groups'][0]; identifiers = group['params']
    names = [name for name in model if name in shapes]
    need(len(names) == 72 and identifiers == list(range(72)) and group['name'] == 'fusion' and
        group['base_lr'] == 1e-5 and group['lr'] == 1e-5 * min(max(update, 1), 10) / 10 and
        group['betas'] == (.9, .99) and group['eps'] == 1e-8 and group['weight_decay'] == 0.,
        'Exact optimizer72 membership/LR/Adam hyperparameters differ')
    for key in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay'):
        need(group.get(key, False) is False, 'Undeclared Adam behavior enabled: ' + key)
    need(group.get('foreach') in (False, None) and group.get('fused') in (False, None),
        'Undeclared fused/foreach Adam mode')
    need(set(optimizer['state']) == (set(identifiers) if update else set()), 'Missing/extra populated Adam states')
    steps = Counter()
    for identifier, name in zip(identifiers, names):
        if not update:
            continue
        row = optimizer['state'][identifier]
        need(set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Exact Adam moment schema differs')
        step = row['step']
        need(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.dtype == torch.float32 and
            step.ndim == 0 and step.item() == update, 'All72 Adam step scalars must equal50')
        steps[int(step.item())] += 1
        for field in ('exp_avg', 'exp_avg_sq'):
            moment = row[field]
            need(isinstance(moment, torch.Tensor) and moment.device.type == 'cpu' and moment.dtype == torch.float32 and
                moment.shape == model[name].shape and not moment.requires_grad and torch.isfinite(moment).all().item() and
                (field != 'exp_avg_sq' or moment.min().item() >= 0), 'Finite FP32 shape-exact Adam moments required')
    need(value['sampler'] == sampler_state(metadata['schedule'], update), 'Actual committed sampler prefix differs')
    for key in ('CPU_RNG', 'CUDA_RNG'):
        tensor = value[key]
        need(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and tensor.dtype == torch.uint8 and
            tensor.ndim == 1 and tensor.numel() > 0, 'Serialized RNG state geometry differs')
    committed = {k: value[k] for k in ('model', 'optimizer', 'sampler', 'CPU_RNG', 'CUDA_RNG')}
    return {'completed_updates': update, 'model_tensor_sha256': model_sha,
        'model_tensors': 515, 'trainable_fusion_tensors': 72, 'frozen_tensors': 443,
        'all443_frozen_tensors_exact': True, 'actual_changed_fusion_tensor_count': len(changed),
        'actual_changed_fusion_tensors': sorted(changed),
        'frozen443_tensor_sha256': tensor_digest({n:v for n,v in model.items() if n not in shapes}, torch),
        'Adam_saved_states': len(optimizer['state']), 'Adam_moment_tensors': 2 * len(optimizer['state']),
        'Adam_step_histogram': dict(steps), 'Adam_group': group,
        'committed_model_optimizer_sampler_RNG_sha256': tree_digest(committed, torch),
        'actual_checkpoint_counts': counts, 'sampler': value['sampler'],
        'automatic_resume_safe': False, 'model_qualified': False}


def audit(receipt, out):
    start = time.monotonic(); need(not out.exists(), 'Preserve previous state audit')
    e = Evidence(receipt); metadata = audit_history(e)
    checkpoint_names = {'update0': RUN + 'snapshot_0000/checkpoint.pt',
        'update50': RUN + 'snapshot_0050/checkpoint.pt', 'stopped': RUN + 'stopped_checkpoint.pt'}
    # Freshly hash every exact model-bearing file before importing Torch or deserializing any pickle.
    names = [*e.p['weight_assets'], *checkpoint_names.values()]
    paths = {name: e.file(name) for name in names}
    need(metadata['failure']['state_checkpoint'] == e.pin(checkpoint_names['stopped']),
        'Actual failure does not point to exact complete stopped checkpoint')
    need(not any(name.startswith(('research.codeformer_filter_fusion_training_v1', 'third_party.codeformer'))
        for name in sys.modules), 'Returned/local model module must not be imported for value audit')
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    immutable = {}; immutable_reports = {}
    for alias in ('original', 'parent', 'r2'):
        name = f'weights/codeformer_{alias}.pth'
        saved = torch.load(paths[name], map_location='cpu', weights_only=True, mmap=True)
        state = saved['model' if alias == 'parent' else 'params_ema']
        actual = validate_model(state, state, torch)
        need(actual == DIGESTS[alias], 'Actual sealed immutable515 tensor digest differs: ' + alias)
        immutable[alias] = state
        immutable_reports[alias] = {'file': e.pin(name), 'canonical515_tensor_sha256': actual}
    reports = {}
    for label, name in checkpoint_names.items():
        saved = torch.load(paths[name], map_location='cpu', weights_only=True, mmap=True)
        reports[label] = checkpoint_audit(saved, immutable['r2'], label, e, metadata, torch)
        reports[label]['file'] = e.pin(name)
        del saved
        print(json.dumps({'checkpoint': label, 'complete': True,
            'model_tensor_sha256': reports[label]['model_tensor_sha256'],
            'Adam_states': reports[label]['Adam_saved_states']}), flush=True)
    need(reports['update0']['model_tensor_sha256'] == DIGESTS['r2'] and
        reports['update50']['committed_model_optimizer_sampler_RNG_sha256'] ==
        reports['stopped']['committed_model_optimizer_sampler_RNG_sha256'],
        'Snapshot50/stopped model+Adam+sampler+RNG differ')
    closure = e.close()
    result = {'format': 'codeformer-filter-fusion-independent-state-audit-v1', 'complete': True,
        'audit_pass': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256': sha(__file__), 'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'import_receipt_sha256': e.receipt_sha, 'immutable_input_weights': immutable_reports,
        'checkpoint_readback': reports,
        'all443_frozen_tensors_exact': True,
        'actual_changed_fusion_tensor_count': reports['update50']['actual_changed_fusion_tensor_count'],
        'saved_checkpoint50_and_stopped_exact_committed_state': True,
        'committed_state_equality_fields': ['model', 'optimizer', 'sampler', 'CPU_RNG', 'CUDA_RNG'],
        'checkpoint_census_boundary_note': 'Snapshot50 is serialized before its478 output forwards; stopped checkpoint is after them. Complete physical state equality excludes deliberately different checkpoint counts/progress metadata.',
        'history': metadata['report'], 'actual_counts': metadata['counts'],
        'saved_gate_booleans': metadata['report']['saved_gate_booleans'],
        'closure': closure, 'seconds': time.monotonic() - start,
        'execution_boundary': {'returned_code_executed': False, 'model_instances': 0,
            'optimizer_constructions': 0, 'neural_forwards': 0, 'gradient_queries': 0,
            'optimizer_updates': 0, 'VM_connections': 0, 'app_changed': False,
            'CPU_weights_only_tensor_deserialization': True},
        'limitations': ['Consistent saved values/history/source joins do not independently replay live CUDA gradients or Adam updates.',
            'Full saved-pixel metric/gate/visual review is separate; state audit passing does not make the stopped model qualified.',
            'This receipt authorizes neither automatic training nor app/default promotion.'],
        'model_qualified': False, 'overall_goal_complete': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return {'complete': True, 'audit_pass': True, 'receipt': str(out), 'receipt_sha256': sha(out),
        'update50_model_tensor_sha256': reports['update50']['model_tensor_sha256'],
        'model_qualified': False, 'seconds': result['seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt', type=Path, default=REVIEW / 'return_import_v1.json')
    parser.add_argument('--out', type=Path, default=REVIEW / 'state_v1/state_audit_v1.json')
    args = parser.parse_args()
    print(json.dumps(audit(args.import_receipt, args.out)), flush=True)
