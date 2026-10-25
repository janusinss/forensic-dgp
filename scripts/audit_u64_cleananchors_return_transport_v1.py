"""Bounded independent return bytes/events audit; no checkpoint or array decoding.

Never extracts executables, imports returned sources, loads models or connects
to a VM. Full framing is read once with compressed and per-member hashes.
State arithmetic, pixels, scalar gates and visual judgments remain separate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
RUN = 'outputs/u64_cleananchors_continuation_v1/'
PREFIX = 'cctv-codeformer-u64-cleananchors-continuation-v1'
PROTOCOL_FIXED = (279879, '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419')
HELPER = ROOT / 'scripts/audit_u96_decoder_readout_transport_v1.py'
HELPER_FIXED = (20828, 'dc3d05237b0eb163579d4034ae5490d8ecfb6005bafba478f243f9f0c219dcfc')
MAX_BYTES = 2 * 1024**3
MAX_EXPANDED = MAX_BYTES + 64 * 1024**2
MAX_MEMBER = 384 * 1024**2
MAX_METADATA = 32 * 1024**2
MAX_RETAINED = 128 * 1024**2
MAX_FILES = 6000
MAX_SECONDS = 240
CATEGORIES = {'initial_quality_bank_forwards': 56, 'deterministic_repeat_forwards': 4,
              'candidate_snapshot_forwards_after_initializer': 168, 'candidate_training_forwards': 64}
OPERATIONS = {'forward': ('full_forward_attempts', 'full_forward_returns', 292),
              'backward': ('backward_attempts', 'backward_returns', 64),
              'optimizer_construction': ('optimizer_constructions', 'optimizer_construction_returns', 1),
              'optimizer_update': ('optimizer_update_attempts', 'optimizer_updates', 16)}
CAPS = {key: cap for attempt, returned, cap in OPERATIONS.values() for key in (attempt, returned)}
CAPS.update(CATEGORIES)
PHASE_CATEGORY = {'initial_u64_quality_bank': 'initial_quality_bank_forwards',
                  'deterministic_same_u64_repeat': 'deterministic_repeat_forwards',
                  'new_quality_snapshot': 'candidate_snapshot_forwards_after_initializer',
                  'paired_TRAIN_optimization': 'candidate_training_forwards'}
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
NEUTRAL_EVENTS = {'static_VM_supervisor_guards_passed_before_Torch', 'phase_returned',
                  'backward_warning_scope', 'failure_saved', 'run_result_saved'}
RUNTIME = {'install.json', 'install_failure.json', 'run.log', 'export.log',
           'supervisor.start.json', 'supervisor.worker.started.json', 'supervisor.worker.json',
           'supervisor.export.started.json', 'supervisor.export.json', 'supervisor.json',
           'supervisor_failure.json', 'export.json', 'export_failure.json'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    need(type(value) is str and 0 < len(value) <= 512 and value != '.' and '\\' not in value
         and ':' not in value, 'Canonical relative path required')
    path = PurePosixPath(value)
    need(not path.is_absolute() and path.as_posix() == value
         and all(part not in ('', '.', '..') for part in path.parts), 'Confined relative path required')
    return value


def member_relative(name):
    need(type(name) is str and name.startswith(STAGE + '/'), 'Exact return stage prefix required')
    return canonical(name[len(STAGE) + 1:])


def regular(path, maximum=MAX_BYTES):
    path = Path(path).absolute()
    need(path == path.resolve(strict=True) and path.is_relative_to(ROOT), 'Canonical workspace input required')
    for item in path.parents:
        state = item.lstat()
        need(stat.S_ISDIR(state.st_mode) and not stat.S_ISLNK(state.st_mode)
             and not getattr(state, 'st_file_attributes', 0) & 1024, 'Ordinary input ancestry required')
    state = path.lstat()
    need(stat.S_ISREG(state.st_mode) and not stat.S_ISLNK(state.st_mode) and state.st_nlink == 1
         and not getattr(state, 'st_file_attributes', 0) & 1024 and 0 < state.st_size <= maximum,
         'Bounded single-link ordinary input required')
    return path


def identity(path):
    state = path.stat()
    return state.st_ino, state.st_size, state.st_mtime_ns


def filepin(path):
    path = regular(path)
    before = identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    need(identity(path) == before, 'Input changed during hash')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': before[1], 'sha256': digest.hexdigest()}


def load_helper():
    pin = filepin(HELPER)
    need((pin['bytes'], pin['sha256']) == HELPER_FIXED, 'Exact reviewed R1 framing primitives required')
    spec = importlib.util.spec_from_file_location('_u64_return_R1_primitives', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helper()


def number(value, message, maximum=None):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0
         and (maximum is None or value <= maximum), message)
    return value


def counters(value):
    need(type(value) is dict and set(value) == set(CAPS)
         and all(type(value[key]) is int and 0 <= value[key] <= cap for key, cap in CAPS.items()),
         'Exact finite integer counter schema required')
    return value


def planned_calls(protocol, parent):
    """Independent stdlib construction from the sealed original TRAIN table."""
    for key in ('model', 'data', 'cases', 'guards', 'quality_policy', 'inherited_DEV_evidence'):
        need(protocol[key] == parent[key], 'Original parent field differs: ' + key)
    by = {(row['reference_id'], row['role']): row for row in parent['cases']}
    opt, mon = (parent['learning'][key] for key in ('optimization_reference_order', 'monitor_reference_order'))
    need(len(by) == 96 and len(opt) == len(set(opt)) == 16 and len(mon) == len(set(mon)) == 8
         and not set(opt) & set(mon) and tuple(parent['learning']['slot_roles']) == ROLES,
         'Exact original TRAIN photograph/role partition required')
    train = [by[(reference, role)] for reference in opt for role in ROLES]
    monitor = [by[(reference, role)] for reference in mon for role in ROLES]
    guards = parent['guards']
    need(len(guards) == 24 and all(row['split'] == 'TRAIN' for row in train + monitor + guards), 'TRAIN-only64/32/24 scope required')
    plan = []
    def append(row, phase, update, backward=False):
        plan.append({'slot': len(plan) + 1, 'id': row['id'], 'phase': phase, 'new_update': update,
                     'backward': backward, 'optimizer_application_after_this_forward': backward and row['role'] == ROLES[-1]})
    for row in monitor + guards:
        append(row, 'initial_u64_quality_bank', 0)
    for row in monitor[:4]:
        append(row, 'deterministic_same_u64_repeat', 0)
    for update in range(1, 17):
        for role, offset in zip(ROLES, (0, 4, 8, 12)):
            append(by[(opt[(update - 1 + offset) % 16], role)], 'paired_TRAIN_optimization', update, True)
        if update in (4, 8, 16):
            for row in monitor + guards:
                append(row, 'new_quality_snapshot', update)
    need(len(plan) == 292, 'Complete prospective292 plan required')
    return plan, {row['id']: row for row in train + monitor + guards}


def event_readback(rows, plan):
    """Replay durable unique tokens and each reservation against its exact call."""
    count, pending, sequence = dict.fromkeys(CAPS, 0), {}, 0
    kinds = Counter()
    last_time = 0.0
    forward_calls, returned_calls, backward_calls, commits = [], [], [], {}
    last_forward = None
    for row in rows:
        need(type(row) is dict and type(row.get('kind')) is str, 'Explicit event object/kind required')
        elapsed = number(row['elapsed_seconds'], 'Finite event elapsed time required', 1255)
        need(elapsed >= last_time, 'Monotone durable event time required')
        last_time = elapsed
        kind = row['kind']; kinds[kind] += 1
        operation, action = (kind.rsplit('_', 1) if '_' in kind else ('', ''))
        if operation in OPERATIONS and action in ('reserved', 'returned'):
            attempt, returned, cap = OPERATIONS[operation]
            token = row['token']
            need(type(token) is int and token > 0, 'Exact integer operation token required')
            if action == 'reserved':
                need(not pending and token == sequence + 1 and count[attempt] < cap,
                     'Consecutive nonoverlapping reserved token required')
                sequence = token
                record = {'kind': operation}
                if operation == 'forward':
                    call = plan[count[attempt]]
                    need(count['backward_attempts'] == count['backward_returns'] == count['candidate_training_forwards'],
                         'Every preceding training forward needs its known backward before the next forward')
                    expected_commits = call['new_update'] - (1 if call['backward'] else 0)
                    need(count['optimizer_updates'] == count['optimizer_update_attempts'] == expected_commits,
                         'Forward requires the exact preceding committed-update cursor')
                    need(count['optimizer_construction_returns'] == (0 if call['new_update'] == 0 else 1),
                         'Forward must have the correct fresh-Adam construction phase')
                    record.update({'id': call['id'], 'category': PHASE_CATEGORY[call['phase']],
                                   'plan_slot': call['slot'], 'update': call['new_update']})
                    count[record['category']] += 1
                    forward_calls.append(call)
                elif operation == 'backward':
                    need(last_forward is not None and last_forward['backward'] is True
                         and last_forward['slot'] not in backward_calls
                         and count['backward_attempts'] == count['candidate_training_forwards'] - 1
                         and count['optimizer_updates'] == last_forward['new_update'] - 1,
                         'One backward per admitted optimization forward at the current update required')
                    record.update({'id': last_forward['id'], 'plan_slot': last_forward['slot'], 'update': last_forward['new_update']})
                    backward_calls.append(last_forward['slot'])
                elif operation == 'optimizer_construction':
                    need(count['full_forward_returns'] == 60 and count['backward_attempts'] == 0,
                         'Fresh Adam must follow the60 initializer/repeat calls')
                else:
                    update = count['optimizer_updates'] + 1
                    need(count['optimizer_construction_returns'] == 1
                         and count['backward_returns'] == count['backward_attempts'] == count['candidate_training_forwards'] == update * 4
                         and last_forward is not None and last_forward['new_update'] == update
                         and last_forward['optimizer_application_after_this_forward'] is True,
                         'Four known backward returns before each consecutive update required')
                    record['update'] = update
                need(all(type(row.get(key)) is type(value) and row[key] == value for key, value in record.items() if key != 'kind'), 'Reserved metadata differs from independent operation plan')
                pending[token] = record
                count[attempt] += 1
            else:
                need(token in pending and pending[token]['kind'] == operation, 'Unreserved or repeated return refused')
                record = pending[token]
                need(all(type(row.get(key)) is type(value) and row[key] == value for key, value in record.items() if key != 'kind'), 'Returned metadata differs from reserved operation')
                if operation != 'optimizer_construction':
                    need(row.get('synchronized_return') is True, 'Synchronized return evidence required')
                count[returned] += 1
                del pending[token]
                if operation == 'forward':
                    last_forward = forward_calls[-1]
                    returned_calls.append(last_forward)
                elif operation in ('optimizer_construction', 'optimizer_update'):
                    commits[count['optimizer_updates']] = dict(count)
        else:
            need(kind in NEUTRAL_EVENTS, 'Unknown event kind cannot conceal an operation')
            if kind == 'phase_returned':
                number(row['phase_seconds'], 'Bounded phase receipt required', 180)
            if kind == 'backward_warning_scope':
                need(row.get('strict_flags_restored') is True and row.get('bitwise_CUDA_gradient_repeatability_claimed') is False,
                     'Scoped CUDA warning disclosure/strict restoration required')
                need(type(row.get('warnings')) is list and all(type(item) is dict and
                     type(item.get('message')) is str and type(item.get('category')) is str for item in row['warnings']),
                     'Actual scoped warning messages required')
                unexpected = [item for item in row['warnings'] if
                    ('deterministic' in item['message'].lower() or 'nondeterminism' in item['message'].lower())
                    and re.match(r'^upsample_bilinear2d_backward_out_cuda\b', item['message']) is None]
                need(row.get('unexpected') == unexpected, 'Unexpected nondeterministic warnings cannot be relabeled')
        counters(row['counters'])
        need(row['counters'] == count, 'Durable counters differ from independent token replay at ' + kind)
        if kind != 'failure_saved':
            need(elapsed <= 1200, 'Scientific event exceeds1200s')
    unknown = [{'token': token, **value} for token, value in sorted(pending.items())]
    return {'events': len(rows), 'counts_by_kind': dict(kinds), 'counters': count,
            'unknown_reservations': unknown, 'operation_outcome_unknown': bool(unknown),
            'last_elapsed_seconds': last_time, 'returned_calls': returned_calls,
            'checkpoint_counter_snapshots': commits}


def inspect_stream(raw, *, guard=lambda: None):
    """Hash every compressed/member byte; retain bounded JSON/JSONL/log only."""
    class Hashed:
        def __init__(self):
            self.digest, self.count = hashlib.sha256(), 0
        def read(self, size):
            guard(); value = raw.read(size); self.count += len(value); self.digest.update(value)
            need(self.count <= MAX_BYTES, 'Compressed return2GiB cap')
            return value
    hashed = Hashed()
    stream = H.SingleGzip(hashed, guard)
    index, metadata, retained = {}, {}, {}
    total = kept = last = 0
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            guard(); name = member_relative(member.name)
            need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.linkname
                 and 0 <= member.size <= MAX_MEMBER, 'Bounded ordinary regular member required')
            need(name not in index and len(index) < MAX_FILES and not name.startswith('weights/'), 'Unique bounded nonweight return namespace required')
            total += member.size
            need(total <= MAX_BYTES, 'Return payload2GiB cap')
            keep = name.endswith(('.json', '.jsonl', '.log'))
            if keep:
                kept += member.size
                need(member.size <= MAX_METADATA and kept <= MAX_RETAINED, 'Bounded retained metadata required')
            digest, size, pieces = hashlib.sha256(), 0, []
            source = archive.extractfile(member)
            for block in iter(lambda: source.read(1024**2), b''):
                guard(); digest.update(block); size += len(block)
                if keep:
                    pieces.append(block)
            need(size == member.size, 'Complete member extent required')
            index[name] = {'bytes': size, 'sha256': digest.hexdigest()}
            if keep:
                value = b''.join(pieces); retained[name] = value
                if name.endswith('.json'):
                    metadata[name] = H.strict_json(value)
                elif name.endswith('.jsonl'):
                    need(not value or value.endswith(b'\n'), 'Interrupted JSONL line remains incomplete evidence')
                    metadata[name] = [H.strict_json(line) for line in value.splitlines()]
            last = ((member.offset_data + member.size + 511) // 512) * 512
    while stream.read(65536):
        guard()
    padding = H.verify_tar_end(stream, last)
    return index, metadata, retained, {'single_gzip_CRC_ISIZE_complete': stream.done,
        'expanded_bytes': stream.expanded, 'tar_zero_EOF_padding_bytes': padding,
        'payload_bytes': total, 'compressed_bytes': hashed.count, 'compressed_sha256': hashed.digest.hexdigest()}


def direct_joins(metadata, index):
    joins = 0
    def visit(value):
        nonlocal joins
        if type(value) is dict:
            if set(value) == {'path', 'bytes', 'sha256'}:
                name = canonical(value['path'])
                need(type(value['bytes']) is int and value['bytes'] >= 0
                     and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Exact saved file pin required')
                need(index.get(name) == {key: value[key] for key in ('bytes', 'sha256')}, 'Saved runtime binding missing/different: ' + name)
                joins += 1
            else:
                for child in value.values():
                    visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)
    for name, value in metadata.items():
        if name.startswith(RUN) and name != RUN + 'operation_plan.json':
            visit(value)
    return joins


def validate_return(protocol, index, metadata, export):
    """Validate actual parsed archived records; claims do not establish pixels."""
    static = {name: value for name, value in protocol['files'].items() if not name.startswith('weights/')}
    for name, value in static.items():
        need(index.get(name) == value, 'Returned sealed static bytes differ: ' + name)
    for name in index:
        need(name in static or name == 'protocol.json' or name in RUNTIME or name.startswith(RUN), 'Unexpected return namespace: ' + name)
    need(export['payload_files'] == len(index) and export['payload_bytes'] == sum(row['bytes'] for row in index.values()), 'Downloaded export inventory totals differ')
    parent = metadata[protocol['parent_protocol']['path']]
    plan, cases = planned_calls(protocol, parent)
    saved_plan = metadata[RUN + 'operation_plan.json']
    need(saved_plan['complete'] is True and saved_plan['calls'] == plan
         and saved_plan['optimizer_maximum_new_updates'] == 16, 'Actual saved prospective plan differs')
    by = {(row['reference_id'], row['role']): row for row in parent['cases']}
    opt, mon = (parent['learning'][key] for key in ('optimization_reference_order', 'monitor_reference_order'))
    selection = {'optimization': [by[(reference, role)] for reference in opt for role in ROLES],
        'monitors': [by[(reference, role)] for reference in mon for role in ROLES], 'guards': parent['guards'],
        'optimization_reference_order': opt, 'monitor_reference_order': mon, 'independent_holdout': False,
        'photograph_separation_is_not_identity_or_pretraining_proof': True, 'DEV_native_wild_FINAL_optimized': False}
    need(saved_plan['selection'] == selection, 'Exact original64/32/24 selection and TRAIN qualification boundaries required')
    replay = event_readback(metadata[RUN + 'events.jsonl'], plan)
    for name in ('status.json', 'failure.json', 'result.json'):
        key = RUN + name
        if key in metadata:
            row = metadata[key]
            need(row['protocol_sha256'] == PROTOCOL_FIXED[1] and counters(row['counters']) == replay['counters']
                 and row.get('model_qualified') is False, 'Terminal protocol/counters/qualification differ: ' + name)
            if name != 'result.json':
                need(row['operation_outcome_unknown'] is replay['operation_outcome_unknown']
                     and row['unknown_reservations'] == replay['unknown_reservations'], 'Terminal unknown-operation ledger differs')
    status = metadata[RUN + 'status.json']
    worker = metadata['supervisor.worker.json']
    need(worker['complete'] is True and worker['phase'] == 'worker' and worker['spawn_returned'] is True
         and type(worker['exit_code']) is int and worker['cleanup']['child_reaped'] is True
         and worker['cleanup']['process_group_absent'] is True and not worker.get('orphan_cleanup_unknown', False), 'Known reaped worker/group cleanup required')
    number(worker['seconds'], 'Worker external cap/grace receipt required', 1280)
    need(export['worker_exit'] == worker['exit_code'] and export['confirmed_counters'] == replay['counters']
         and export['operation_outcome_unknown'] is replay['operation_outcome_unknown']
         and export['status_receipt'] == {'path': RUN + 'status.json', **index[RUN + 'status.json']}, 'Downloaded export/worker/status replay join differs')
    failure, result = metadata.get(RUN + 'failure.json'), metadata.get(RUN + 'result.json')
    need(export['result_present'] is (result is not None) and export['failure_present'] is (failure is not None)
         and export.get('failure_read_error') is None and export.get('worker_exit_receipt_error') is None,
         'Downloaded export outcome-presence/readback claims differ')
    need(export['journal_tail_reconciled'] is True
         and export['journal_bytes_observed'] == index[RUN + 'events.jsonl']['bytes']
         and export['journal_tail_kind'] == metadata[RUN + 'events.jsonl'][-1]['kind'],
         'Downloaded export journal-tail claims differ from actual full replay')
    quality_stop = worker['exit_code'] == 2
    if quality_stop:
        need(type(failure) is dict and failure['complete'] is False and failure['quality_stop'] is True and failure['error_type'] == 'QualityStop'
             and result is None and status['quality_stop'] is True and status['label'] == 'retained_stop'
             and status['complete'] is False and not replay['operation_outcome_unknown']
             and export['quality_stop_exit2'] is True and export['quality_stop_recorded'] is True,
             'Known quality stop must retain failure and cannot become successful result')
        update = failure['active_update']
        need(type(update) is int and update == replay['counters']['optimizer_updates']
             and status['active_update'] == update,
             'Quality-stop active update must equal the actual committed cursor')
        bank = metadata[RUN + 'banks/u' + str(update).zfill(4) + '/bank.json']
        need(update in (0, 4, 8, 16) and bank['complete'] is True and bank['gate']['pass'] is False,
             'Actual failed saved quality bank required')
        terminal = 'KNOWN_QUALITY_STOP_U' + str(update) + '_NOT_QUALIFIED'
    elif worker['exit_code'] == 0:
        need(type(result) is dict and result['complete'] is True and failure is None
             and replay['counters'] == CAPS and not replay['operation_outcome_unknown'], 'Complete finite screen counters/result required for exit0')
        terminal = 'SAVED_FINITE_TRAIN_SCREEN_PASS_NOT_QUALIFIED'
    else:
        need(type(failure) is dict or worker['timeout'] is True, 'Retain error/timeout evidence')
        terminal = 'KNOWN_WORKER_ERROR_OR_TIMEOUT_NOT_QUALIFIED'
    returned = replay.pop('returned_calls')
    records = []
    for call in returned:
        name = RUN + 'calls/' + str(call['slot']).zfill(4) + '/record.json'
        if name not in metadata:
            need(failure is not None and call == returned[-1], 'Only final failed return may lack accepted output record')
            continue
        row, source = metadata[name], cases[call['id']]
        need(row['call'] == call and row['update'] == call['new_update']
             and all(row[key] == source[key] for key in ('id', 'reference_id', 'role', 'split', 'input'))
             and row['target'] == source.get('target') and row['model_qualified'] is False,
             'Actual output record differs from exact admitted call')
        need(set(row['outputs']) == {'raw', 'unclipped', 'png'}, 'Three actual output bindings required')
        records.append(name)
    actual_records = {name for name in index if re.fullmatch(re.escape(RUN) + r'calls/\d{4}/record\.json', name)}
    need(actual_records == set(records), 'No extra or unreturned generated-call records permitted')
    for name, bank in metadata.items():
        match = re.fullmatch(re.escape(RUN) + r'banks/u(0000|0004|0008|0016)/bank\.json', name)
        if match is None:
            continue
        update = int(match.group(1))
        calls = [call for call in returned if call['new_update'] == update and
                 call['phase'] == ('initial_u64_quality_bank' if update == 0 else 'new_quality_snapshot')]
        need(len(calls) == 56 and bank['complete'] is True
             and set(bank['records']) == {call['id'] for call in calls}, 'Exact completed56-case saved bank required')
        for call in calls:
            record_name = RUN + 'calls/' + str(call['slot']).zfill(4) + '/record.json'
            need(bank['records'][call['id']] == {**metadata[record_name], 'receipt': {'path': record_name, **index[record_name]}},
                 'Bank must contain actual bound generated record bytes')
    if (quality_stop or worker['exit_code'] == 0) and replay['counters']['optimizer_construction_returns']:
        updates = replay['counters']['optimizer_updates']
        for update in range(updates + 1):
            checkpoint = metadata[RUN + 'checkpoints/u' + str(update).zfill(4) + '/checkpoint.json']
            need(checkpoint['complete'] is True and checkpoint['update'] == update
                 and checkpoint['model_qualified'] is False, 'Every committed checkpoint receipt includingu0 required')
            counters(checkpoint['counters'])
            need(checkpoint['counters'] == replay['checkpoint_counter_snapshots'][update], 'Checkpoint counters differ from actual committed-event prefix')
            if update:
                commit = metadata[RUN + 'updates/u' + str(update).zfill(4) + '/commit.json']
                need(commit['complete'] is True and commit['new_update'] == update
                     and commit['LR'] == 0.0001 * min(1.0, update / 4.0)
                     and len(commit['microbatches']) == 4 and commit['frozen539_exact'] is True,
                     'Saved fresh-Adam commit/microbatch metadata required')
    joins = direct_joins(metadata, index)
    resources = metadata[RUN + 'resources.jsonl']
    need(bool(resources), 'Actual resource samples required')
    last = 0.0
    for row in resources:
        elapsed = number(row['elapsed_seconds'], 'Finite sampled resource time required')
        need(elapsed >= last, 'Monotone resource sample time required'); last = elapsed
        for key in ('free_disk_bytes', 'current_RSS_bytes', 'peak_RSS_bytes', 'CUDA_reserved_bytes', 'CUDA_peak_reserved_bytes', 'output_bytes'):
            need(type(row[key]) is int and row[key] >= 0, 'Exact nonnegative sampled byte count required')
    maxima = {key: max(row[key] for row in resources) for key in ('current_RSS_bytes', 'peak_RSS_bytes', 'CUDA_reserved_bytes', 'CUDA_peak_reserved_bytes', 'output_bytes')}
    respected = (maxima['peak_RSS_bytes'] <= protocol['limits']['maximum_RSS_bytes']
        and maxima['CUDA_peak_reserved_bytes'] <= protocol['limits']['maximum_CUDA_reserved_bytes']
        and maxima['output_bytes'] <= MAX_BYTES
        and min(row['free_disk_bytes'] for row in resources) >= protocol['limits']['continuing_free_disk_bytes']
        and last <= 1200)
    need(not quality_stop or respected, 'Quality-stop receipt conflicts with sampled resource cap violation')
    need(not any('status.pending' in name for name in index) or replay['operation_outcome_unknown'], 'Interrupted status cannot become known operation outcome')
    return {'terminal_status': terminal, 'worker_exit': worker['exit_code'], 'events': replay,
        'generated_records': len(records), 'saved_direct_bindings_checked': joins,
        'sampled_resource_limits_respected': respected, 'resource_samples': len(resources),
        'sampled_memory_output_maxima': maxima, 'minimum_sampled_free_disk_bytes': min(row['free_disk_bytes'] for row in resources),
        'saved_scalar_gates_independently_recomputed': False, 'post_export_supervisor_receipts_required_in_archive': False}


def audit(archive, expected_sha, out):
    start = time.monotonic()
    need('torch' not in sys.modules and 'numpy' not in sys.modules, 'Saved bytes/JSON only')
    need(not out.exists(), 'Preserve prior return-audit namespace')
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha), 'Explicit externally verified archive SHA required')
    archive = regular(archive); before = identity(archive)
    checksum, export_path = Path(str(archive) + '.sha256'), archive.with_name(PREFIX + '-export.json')
    protocol_path = ROOT / 'outputs' / STAGE / 'protocol.json'
    inputs = {name: filepin(path) for name, path in (('checksum', checksum), ('export', export_path), ('protocol', protocol_path), ('framing_helper', HELPER))}
    raw_protocol = protocol_path.read_bytes()
    need((len(raw_protocol), sha(raw_protocol)) == PROTOCOL_FIXED, 'Exact source-approved execution protocol required')
    need(checksum.read_bytes() == (expected_sha + '  ' + archive.name + '\n').encode('ascii'), 'Exact LF downloaded checksum required')
    export, protocol = H.strict_json(export_path.read_bytes()), H.strict_json(raw_protocol)
    need(export['complete'] is True and export['archive_sha256'] == expected_sha and export['bytes'] == before[1]
         and export['protocol_sha256'] == PROTOCOL_FIXED[1] and export['model_qualified'] is False, 'Downloaded export/archive/protocol join differs')
    def guard():
        need(time.monotonic() - start < MAX_SECONDS, 'Independent return-audit240s stop')
    with archive.open('rb') as stream:
        index, metadata, retained, framing = inspect_stream(stream, guard=guard)
    need(framing['compressed_bytes'] == before[1] and framing['compressed_sha256'] == expected_sha
         and identity(archive) == before, 'Single-pass exact archive hash/identity required')
    need(index['protocol.json'] == {'bytes': len(raw_protocol), 'sha256': sha(raw_protocol)}
         and retained['protocol.json'] == raw_protocol, 'Actual returned protocol bytes differ')
    result = validate_return(protocol, index, metadata, export)
    guard()
    need(inputs == {name: filepin(path) for name, path in (('checksum', checksum), ('export', export_path), ('protocol', protocol_path), ('framing_helper', HELPER))}, 'Small input pins changed during audit')
    out.mkdir()
    for name, raw in retained.items():
        target = out / 'returned_metadata' / canonical(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    archive_pin = {'path': archive.relative_to(ROOT).as_posix(), 'bytes': before[1], 'sha256': expected_sha}
    index_doc = {'complete': True, 'archive': archive_pin, 'protocol': inputs['protocol'], 'files': index, 'framing': framing}
    with (out / 'archive_index_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(index_doc, stream, indent=2, allow_nan=False); stream.write('\n')
    result.update({'complete': True, 'transport_and_event_audit_pass': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'archive': archive_pin, **inputs, 'payload_files': len(index), 'framing': framing,
        'auditor': filepin(Path(__file__).resolve()), 'seconds': time.monotonic() - start,
        'actual_optimizer_states_and_pixels_independently_recomputed_here': False,
        'all_visual_sheets_reviewed': False, 'Torch_imported': False, 'NumPy_imported': False,
        'new_model_neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False})
    with (out / 'independent_transport_event_audit_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'outputs' / (PREFIX + '-results.tar.gz'))
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/codeformer_u64_cleananchors_continuation_v1_return_transport_audit')
    args = parser.parse_args()
    result = audit(args.archive, args.archive_sha, args.out)
    print(json.dumps({key: result[key] for key in ('complete', 'transport_and_event_audit_pass', 'terminal_status', 'worker_exit', 'payload_files', 'events')}))


if __name__ == '__main__':
    main()
