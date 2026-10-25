"""Durable bookkeeping only; no worker, framework, model or optimizer execution."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import uuid

from . import contract as C


FORMAT = 'conditioned-sft-execution-ledger-v1'
PHASE_CATEGORY = {
    'initial_quality': 'initializer_forwards',
    'initial_optimization_readout': 'initializer_forwards',
    'same_state_deterministic_repeat': 'deterministic_repeat_forwards',
    'photographic_optimization': 'training_forwards',
    'quality_snapshot': 'later_quality_forwards',
    'terminal_optimization_readout': 'terminal_optimization_readout_forwards',
}
MAX_LINE_BYTES = 16384
MAX_JOURNAL_BYTES = 64 * 1024 ** 2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _reason(value):
    need(type(value) is str and 0 < len(value.strip()) <= 512,
         'Explicit bounded stop reason required')


def _sha(value):
    need(type(value) is str and len(value) == 64 and
         all(c in '0123456789abcdef' for c in value), 'Exact lowercase SHA256 receipt pin required')


class _Machine:
    """Deterministic state transitions shared by live recording and read-only audit."""

    def __init__(self, selection):
        self.plan = C.operation_plan(selection)
        self.limits = copy.deepcopy(C.LIMITS)
        self.plan_sha256 = _digest(self.plan)
        need(len(self.plan) == 1736 and [r['slot'] for r in self.plan] == list(range(1, 1737)),
             'Exact contract1736 ordered unique forward slots required')
        self.initial_slots = {r['slot'] for r in self.plan if r['new_update'] == 0}
        need(len(self.initial_slots) == 440, 'Both complete440 initializer/parity/repeat slots required')
        self.cursor = 0
        self.status = 'OPEN'
        self.pending = None
        self.backward_due = None
        self.update_due = None
        self.batch = []
        self.returned_slots = set()
        self.skipped_slots = set()
        self.stopped_arms = {}
        self.constructed = set()
        self.commits = dict.fromkeys(C.ARMS, 0)
        self.quality_boundary = dict.fromkeys(C.ARMS, None)
        self.counters = {key: 0 for key in (
            'full_forward_attempts', 'full_forward_returns', *set(PHASE_CATEGORY.values()),
            'backward_attempts', 'backward_returns', 'gradient_queries',
            'optimizer_construction_attempts', 'optimizer_constructions',
            'optimizer_update_attempts', 'optimizer_updates')}
        self.per_arm = {arm: dict(self.counters) for arm in C.ARMS}
        self.category_returns = {key: 0 for key in set(PHASE_CATEGORY.values())}
        self.fatal_reason = None
        self.quality_last = {
            (arm, update): max(r['slot'] for r in self.plan if r['arm'] == arm and
                               r['phase'] == 'quality_snapshot' and r['new_update'] == update)
            for arm in C.ARMS for update in C.SNAPSHOTS[1:]}

    def _open(self):
        need(self.status == 'OPEN', 'Closed/fatal/journal-failed ledger cannot execute or resume')

    def _idle(self):
        self._open()
        need(self.pending is None, 'Resolve the existing reservation before any new operation')

    def _cap(self, counter, arm, *, limit=None):
        key = limit or counter
        maximum = self.limits[key]
        need(self.counters[counter] < maximum and self.per_arm[arm][counter] < maximum // 2,
             'Global or per-arm allocation exhausted: ' + counter)

    def _add(self, counter, arm):
        self.counters[counter] += 1
        self.per_arm[arm][counter] += 1

    def next_forward(self):
        return copy.deepcopy(self.plan[self.cursor]) if self.cursor < len(self.plan) else None

    def _advance_skips(self):
        while self.cursor < len(self.plan) and self.plan[self.cursor]['slot'] in self.skipped_slots:
            self.cursor += 1

    def reserve(self, kind, operation, token):
        self._idle()
        need(type(operation) is dict and type(token) is str, 'Exact reservation object/token required')
        need(kind in ('forward', 'backward', 'optimizer_construction', 'optimizer_update'),
             'Only precommitted bounded operation kinds admitted')
        arm = operation.get('arm')
        need(type(arm) is str and arm in C.ARMS and arm not in self.stopped_arms,
             'Explicit live arm required')
        if kind == 'forward':
            row = self.next_forward()
            need(row is not None and _json(operation) == _json(row),
                 'Exact next slot/arm/bank/case_key/phase/order required')
            need(self.backward_due is None and self.update_due is None,
                 'Complete the role backward or five-role commit before another forward')
            category = PHASE_CATEGORY[row['phase']]
            self._cap('full_forward_attempts', arm)
            self._cap(category, arm)
            if row['backward']:
                need(self.initial_slots <= self.returned_slots and arm in self.constructed,
                     'All440 initial returns and fresh arm optimizer required before training')
                need(row['new_update'] == self.commits[arm] + 1 and
                     row['id'].rsplit('__', 1)[1] == C.ROLES[len(self.batch)],
                     'One exact five-role sequence required per uncommitted photograph')
                if self.batch:
                    first = self.plan[self.batch[0] - 1]
                    need(row['id'].rsplit('__', 1)[0] == first['id'].rsplit('__', 1)[0] and
                         row['new_update'] == first['new_update'] and row['arm'] == first['arm'],
                         'All five roles must belong to the same arm/update/photograph')
            self._add('full_forward_attempts', arm)
            self._add(category, arm)
            if row['backward']:
                self.quality_boundary[arm] = None
        elif kind == 'backward':
            need(self.backward_due is not None and _json(operation) == _json(self.backward_due),
                 'Backward only for the immediately returned admitted training slot')
            need(operation['bank'] == 'new_optimization' and operation['backward'] is True,
                 'Monitor/guard/readout backward refused')
            self._cap('backward_attempts', arm)
            self._add('backward_attempts', arm)
        elif kind == 'optimizer_construction':
            need(set(operation) == {'arm'} and self.initial_slots <= self.returned_slots and
                 arm not in self.constructed and self.backward_due is None and
                 self.update_due is None and not self.batch,
                 'Fresh optimizer once per arm after all440 initial returns required')
            row = self.next_forward()
            need(row is not None and row['arm'] == arm and row['backward'],
                 'Construct only the next active precommitted training arm optimizer')
            self._cap('optimizer_construction_attempts', arm, limit='optimizer_constructions')
            self._add('optimizer_construction_attempts', arm)
        else:
            need(set(operation) == {'arm', 'new_update'} and type(operation['new_update']) is int and
                 _json(self.update_due) == _json(operation) and self.backward_due is None and
                 len(self.batch) == len(C.ROLES) and arm in self.constructed,
                 'Commit only after all five ordered role forward/backward returns')
            self._cap('optimizer_update_attempts', arm)
            self._add('optimizer_update_attempts', arm)
        self.pending = {'kind': kind, 'operation': copy.deepcopy(operation), 'token': token}

    def check_return(self, token):
        self._open()
        need(self.pending is not None and type(token) is str and token == self.pending['token'],
             'One exact outstanding reservation token required before return')

    def returned(self, token):
        self.check_return(token)
        kind, operation = self.pending['kind'], self.pending['operation']
        arm = operation['arm']
        if kind == 'forward':
            self._add('full_forward_returns', arm)
            self.category_returns[PHASE_CATEGORY[operation['phase']]] += 1
            self.returned_slots.add(operation['slot'])
            self.cursor += 1
            self._advance_skips()
            if operation['backward']:
                self.backward_due = copy.deepcopy(operation)
            if self.initial_slots <= self.returned_slots and len(self.returned_slots) == 440:
                self.quality_boundary = dict.fromkeys(C.ARMS, 0)
            boundary = (arm, operation['new_update'])
            if self.quality_last.get(boundary) == operation['slot']:
                self.quality_boundary[arm] = operation['new_update']
        elif kind == 'backward':
            self._add('backward_returns', arm)
            self.batch.append(operation['slot'])
            self.backward_due = None
            if len(self.batch) == len(C.ROLES):
                need(operation['optimizer_application_after_this_forward'] is True,
                     'Only exact fifth role may expose a commit reservation')
                self.update_due = {'arm': arm, 'new_update': operation['new_update']}
        elif kind == 'optimizer_construction':
            self._add('optimizer_constructions', arm)
            self.constructed.add(arm)
        else:
            self._add('optimizer_updates', arm)
            self.commits[arm] += 1
            self.update_due = None
            self.batch = []
        self.pending = None

    def quality_payload(self, arm, reason, receipt_sha256):
        self._idle()
        _reason(reason)
        _sha(receipt_sha256)
        need(type(arm) is str and arm in C.ARMS and arm not in self.stopped_arms and
             self.initial_slots <= self.returned_slots and self.pending is None and
             self.backward_due is None and self.update_due is None and not self.batch and
             self.quality_boundary[arm] in C.SNAPSHOTS,
             'Quality stop only at complete checkpoint0/8/24/48 with no partial update')
        slots = [r['slot'] for r in self.plan[self.cursor:] if r['arm'] == arm]
        need(bool(slots), 'Quality stop must explicitly skip remaining arm operations')
        return {'event': 'quality_stop', 'arm': arm, 'reason': reason,
                'receipt_sha256': receipt_sha256, 'checkpoint': self.quality_boundary[arm],
                'skipped_count': len(slots), 'skipped_slots_sha256': _digest(slots)}

    def quality_stopped(self, payload):
        expected = self.quality_payload(payload.get('arm'), payload.get('reason'), payload.get('receipt_sha256'))
        need(_json(payload) == _json(expected), 'Exact order-preserving arm skip receipt required')
        arm = payload['arm']
        self.stopped_arms[arm] = copy.deepcopy(payload)
        self.skipped_slots.update(r['slot'] for r in self.plan[self.cursor:] if r['arm'] == arm)
        self._advance_skips()

    def fatal(self, reason):
        self._open()
        _reason(reason)
        self.status, self.fatal_reason = 'FATAL', reason

    def finish_check(self):
        self._idle()
        need(self.cursor == len(self.plan) and self.backward_due is None and
             self.update_due is None and not self.batch,
             'Finish requires every unskipped slot and every five-role commit returned')

    def summary(self):
        return copy.deepcopy({
            'status': self.status, 'plan_sha256': self.plan_sha256,
            'counters': self.counters, 'per_arm': self.per_arm,
            'category_forward_returns': self.category_returns,
            'returned_forward_slots': sorted(self.returned_slots),
            'skipped_forward_slots': sorted(self.skipped_slots),
            'next_forward': self.next_forward(), 'pending': self.pending,
            'backward_due': self.backward_due, 'optimizer_update_due': self.update_due,
            'committed_updates': self.commits, 'quality_stops': self.stopped_arms,
            'fatal_reason': self.fatal_reason, 'resume_permitted': False,
            'bookkeeping_complete': self.status == 'FINISHED',
            'images_gradients_quality_or_training_verified': False})


class ExecutionLedger:
    """Durable JSONL journal; never a resume interface."""

    def __init__(self, selection, persist=None, *, path=None):
        self._machine = _Machine(selection)
        self._lock = threading.Lock()
        need((callable(persist) and path is None) or (persist is None and path is not None),
             'One durable owned-store callback or create-exclusive path backend required')
        self._persist = persist
        self._fd = None
        self.path = None
        if path is not None:
            self.path = Path(path).absolute()
            need(self.path.parent.is_dir() and self.path.parent.resolve() == self.path.parent,
                 'Existing ordinary journal parent required')
            self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND |
                               getattr(os, 'O_BINARY', 0), 0o600)
        self._closed = False
        self._seq, self._previous, self._size = 0, '0' * 64, 0
        self.run_id = uuid.uuid4().hex
        try:
            self._record({'event': 'begin', 'format': FORMAT, 'run_id': self.run_id,
                          'plan_sha256': self._machine.plan_sha256, 'limits': self._machine.limits})
        except BaseException:
            self.close()
            raise

    @classmethod
    def for_path(cls, path, selection):
        return cls(selection, path=path)

    def _record(self, payload):
        need(not self._closed and self._machine.status != 'JOURNAL_FAILED',
             'Closed or failed journal cannot append')
        record = {'seq': self._seq + 1, 'previous_sha256': self._previous, 'payload': payload}
        record['sha256'] = _digest(record)
        encoded = _json(record) + b'\n'
        need(len(encoded) <= MAX_LINE_BYTES and self._size + len(encoded) <= MAX_JOURNAL_BYTES,
             'Bounded ledger journal size exhausted')
        try:
            if self._persist is not None:
                self._persist(encoded)
            else:
                live, named = os.fstat(self._fd), self.path.lstat()
                need(stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode) and
                     not getattr(named, 'st_file_attributes', 0) & 1024 and
                     live.st_ino == named.st_ino and live.st_dev == named.st_dev and
                     live.st_nlink == named.st_nlink == 1 and live.st_size == self._size,
                     'Journal identity/link/size changed before append')
                offset = 0
                while offset < len(encoded):
                    written = os.write(self._fd, encoded[offset:])
                    need(type(written) is int and written > 0, 'Journal write did not advance')
                    offset += written
                os.fsync(self._fd)
                need(os.fstat(self._fd).st_size == self._size + len(encoded), 'Journal size changed during append')
        except BaseException:
            self._machine.status = 'JOURNAL_FAILED'
            self._machine.fatal_reason = 'Journal write/fsync uncertain; no return credit and no retry/resume'
            raise
        self._seq, self._previous = record['seq'], record['sha256']
        self._size += len(encoded)

    def _reserve(self, kind, operation):
        with self._lock:
            need(not self._closed, 'Closed ledger cannot execute')
            token = self.run_id + ':' + str(self._seq + 1)
            self._machine.reserve(kind, operation, token)
            self._record({'event': 'reserve', 'kind': kind, 'operation': operation, 'token': token})
            return token

    def reserve_forward(self, operation):
        return self._reserve('forward', copy.deepcopy(operation))

    def reserve_backward(self, slot):
        need(type(slot) is int and 1 <= slot <= len(self._machine.plan), 'Explicit exact forward slot required')
        return self._reserve('backward', copy.deepcopy(self._machine.plan[slot - 1]))

    def reserve_optimizer_construction(self, arm):
        return self._reserve('optimizer_construction', {'arm': arm})

    def reserve_optimizer_update(self, arm, update):
        return self._reserve('optimizer_update', {'arm': arm, 'new_update': update})

    def returned(self, token):
        with self._lock:
            need(not self._closed, 'Closed ledger cannot return')
            self._machine.check_return(token)
            self._record({'event': 'return', 'token': token})
            self._machine.returned(token)

    def quality_stop(self, arm, reason, *, receipt_sha256):
        with self._lock:
            payload = self._machine.quality_payload(arm, reason, receipt_sha256)
            self._record(payload)
            self._machine.quality_stopped(payload)

    def fatal_stop(self, reason):
        with self._lock:
            self._machine._open()
            _reason(reason)
            self._record({'event': 'fatal_stop', 'reason': reason})
            self._machine.fatal(reason)

    def finish(self):
        with self._lock:
            self._machine.finish_check()
            self._record({'event': 'finish'})
            self._machine.status = 'FINISHED'

    def next_forward(self):
        with self._lock:
            return self._machine.next_forward()

    def summary(self):
        with self._lock:
            result = self._machine.summary()
            result.update(journal_records=self._seq, journal_sha256=self._previous,
                          journal_bytes=self._size, closed=self._closed)
            return result

    def close(self):
        if not self._closed:
            if self._fd is not None:
                os.close(self._fd)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def audit_journal(path, selection):
    """Strict replay from bytes or path; never resumes an incomplete trace."""
    machine = _Machine(selection)
    if type(path) is bytes:
        raw, path = path, None
    else:
        path = Path(path)
        info = path.lstat()
        need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 1024 and info.st_nlink == 1 and
             0 < info.st_size <= MAX_JOURNAL_BYTES, 'Bounded ordinary single-link journal required')
        raw = path.read_bytes()
        need(len(raw) == info.st_size, 'Changed journal record refused')
    need(0 < len(raw) <= MAX_JOURNAL_BYTES and raw.endswith(b'\n'), 'Partial/empty journal record refused')
    previous, run_id = '0' * 64, None
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate journal JSON key refused')
            result[key] = value
        return result
    for seq, line in enumerate(raw.splitlines(), 1):
        need(0 < len(line) <= MAX_LINE_BYTES, 'Bounded nonempty journal line required')
        record = json.loads(line, object_pairs_hook=pairs,
                            parse_constant=lambda value: need(False, 'Nonfinite JSON refused'))
        need(type(record) is dict and set(record) == {'seq', 'previous_sha256', 'payload', 'sha256'} and
             type(record['seq']) is int and record['seq'] == seq and record['previous_sha256'] == previous,
             'Exact journal sequence/hash-chain schema required')
        digest = record.pop('sha256')
        need(digest == _digest(record), 'Journal record hash changed')
        payload = record['payload']
        need(type(payload) is dict and type(payload.get('event')) is str, 'Exact journal payload required')
        event = payload['event']
        if seq == 1:
            need(set(payload) == {'event', 'format', 'run_id', 'plan_sha256', 'limits'} and
                 event == 'begin' and payload['format'] == FORMAT and
                 payload['plan_sha256'] == machine.plan_sha256 and
                 type(payload['limits']) is dict and set(payload['limits']) == set(machine.limits) and
                 all(type(value) is int for value in payload['limits'].values()) and
                 _json(payload['limits']) == _json(machine.limits) and
                 type(payload['run_id']) is str and len(payload['run_id']) == 32 and
                 all(c in '0123456789abcdef' for c in payload['run_id']),
                 'Exact contract/plan-bound first journal record required')
            run_id = payload['run_id']
        elif event == 'reserve':
            need(set(payload) == {'event', 'kind', 'operation', 'token'} and
                 payload['token'] == run_id + ':' + str(seq), 'Exact reservation journal schema required')
            machine.reserve(payload['kind'], payload['operation'], payload['token'])
        elif event == 'return':
            need(set(payload) == {'event', 'token'}, 'Exact return journal schema required')
            machine.returned(payload['token'])
        elif event == 'quality_stop':
            machine.quality_stopped(payload)
        elif event == 'fatal_stop':
            need(set(payload) == {'event', 'reason'}, 'Exact fatal-stop journal schema required')
            machine.fatal(payload['reason'])
        elif event == 'finish':
            need(set(payload) == {'event'}, 'Exact finish journal schema required')
            machine.finish_check()
            machine.status = 'FINISHED'
        else:
            raise ValueError('Unknown/out-of-order journal event refused')
        previous = digest
    if path is not None:
        need(path.stat().st_size == len(raw) and path.read_bytes() == raw, 'Journal changed during audit')
    result = machine.summary()
    result.update(journal_records=seq, journal_sha256=previous, journal_bytes=len(raw))
    return result


def _seconds(value):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
         'Finite nonnegative explicit monotonic seconds required')
    return float(value)


class TimeBudget:
    """Pure supplied-clock deadlines; active-arm time excludes inactive intervals."""

    def __init__(self, started_at):
        self.started_at = self.last_now = _seconds(started_at)
        self.limits = copy.deepcopy(C.LIMITS)
        self.active_arm = None
        self.active_since = None
        self.arm_seconds = dict.fromkeys(C.ARMS, 0.)
        self._failure_anchors = {}

    def _now(self, now):
        now = _seconds(now)
        need(now >= self.last_now, 'Monotonic time cannot move backwards')
        self.last_now = now
        return now

    def enter_arm(self, arm, now):
        now = self._now(now)
        need(type(arm) is str and arm in C.ARMS and self.active_arm is None,
             'One explicit active arm at a time required')
        self.active_arm, self.active_since = arm, now

    def leave_arm(self, now):
        now = self._now(now)
        need(self.active_arm is not None, 'An active arm interval is required')
        self.arm_seconds[self.active_arm] += now - self.active_since
        self.active_arm = self.active_since = None

    def deadline(self, now, *, phase_started_at, mode='work'):
        now, phase = self._now(now), _seconds(phase_started_at)
        need(self.started_at <= phase <= now, 'Phase must begin within the current monotonic run')
        limits = self.limits
        outer = self.started_at + limits['external_seconds']
        if mode == 'work':
            candidates = [self.started_at + limits['worker_seconds'], phase + limits['per_phase_seconds']]
            if self.active_arm is not None:
                used = self.arm_seconds[self.active_arm] + now - self.active_since
                candidates.append(now + limits['per_arm_seconds'] - used)
        elif mode == 'failure_capture':
            anchor = self._failure_anchors.setdefault(mode, phase)
            candidates = [anchor + limits['failure_capture_seconds'], outer -
                          limits['failure_receipt_seconds'] - limits['termination_grace_seconds']]
        elif mode == 'failure_receipt':
            anchor = self._failure_anchors.setdefault(mode, phase)
            candidates = [anchor + limits['failure_receipt_seconds'], outer - limits['termination_grace_seconds']]
        elif mode == 'termination_grace':
            anchor = self._failure_anchors.setdefault(mode, phase)
            candidates = [anchor + limits['termination_grace_seconds'], outer]
        else:
            raise ValueError('Only work/failure_capture/failure_receipt/termination_grace deadline modes admitted')
        deadline = min(candidates)
        return {'deadline': deadline, 'remaining_seconds': max(0., deadline - now),
                'expired': now >= deadline, 'mode': mode,
                'worker_deadline': self.started_at + limits['worker_seconds'],
                'external_deadline': outer, 'timeout_enforced_by_helper': False}


def export_deadlines(started_at):
    started = _seconds(started_at)
    return {'export_deadline': started + C.LIMITS['export_seconds'],
            'export_external_deadline': started + C.LIMITS['export_external_seconds'],
            'termination_grace_seconds': C.LIMITS['termination_grace_seconds'],
            'timeout_enforced_by_helper': False}
