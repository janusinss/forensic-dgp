"""Pure ledger, failure-injection and supplied-clock checks; no neural execution."""
import ast
import copy
from hashlib import sha256
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


class FrameworkBlocker(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'tensorflow', 'jax', 'jaxlib'} or fullname.endswith(('.model', '.worker')):
            self.attempted.append(fullname)
            raise AssertionError('Framework/model/worker forbidden in ledger tests: ' + fullname)
        return None


BLOCKER = FrameworkBlocker()
sys.meta_path.insert(0, BLOCKER)
from research.codeformer_matched_feature_learning_v1 import contract as C
from research.codeformer_matched_feature_learning_v1 import execution as E

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'


class Sink:
    """Acknowledged in-memory transport; does not claim filesystem durability."""
    def __init__(self):
        self.lines = []
    def __call__(self, line):
        self.lines.append(line)
    def raw(self):
        return b''.join(self.lines)


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = C.select_cases(
            (PREP / 'data_admission_v1.json').read_bytes(),
            (PREP / 'candidate_data_v1.json').read_bytes(),
            (ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/evidence/u64_parent_protocol.json').read_bytes())
        cls.plan = C.operation_plan(cls.selection)

    def ledger(self):
        sink = Sink()
        ledger = E.ExecutionLedger(self.selection, persist=sink)
        self.addCleanup(ledger.close)
        return ledger, sink

    def drive(self, ledger, last_slot=None, *, constructed=None):
        constructed = set() if constructed is None else constructed
        while (row := ledger.next_forward()) is not None and (last_slot is None or row['slot'] <= last_slot):
            if row['backward'] and row['arm'] not in constructed:
                ledger.returned(ledger.reserve_optimizer_construction(row['arm']))
                constructed.add(row['arm'])
            ledger.returned(ledger.reserve_forward(row))
            if row['backward']:
                ledger.returned(ledger.reserve_backward(row['slot']))
                if row['optimizer_application_after_this_forward']:
                    ledger.returned(ledger.reserve_optimizer_update(row['arm'], row['new_update']))
        return constructed

    def test_complete_exact_schedule_counters_and_strict_replay(self):
        ledger, sink = self.ledger()
        self.drive(ledger)
        ledger.finish()
        actual = ledger.summary()
        for counter in ('full_forward_attempts', 'full_forward_returns', 'training_forwards',
                        'initializer_forwards', 'deterministic_repeat_forwards', 'later_quality_forwards',
                        'terminal_optimization_readout_forwards', 'backward_attempts', 'backward_returns',
                        'optimizer_constructions', 'optimizer_update_attempts', 'optimizer_updates'):
            with self.subTest(counter=counter):
                self.assertEqual(actual['counters'][counter], C.LIMITS[counter])
                self.assertTrue(all(actual['per_arm'][arm][counter] == C.LIMITS[counter] // 2 for arm in C.ARMS))
        self.assertEqual(actual['committed_updates'], {'A': 48, 'B': 48})
        self.assertEqual(actual['counters']['gradient_queries'], 0)
        self.assertEqual(actual['returned_forward_slots'], list(range(1, 1737)))
        replay = E.audit_journal(sink.raw(), self.selection)
        for key in replay:
            self.assertEqual(replay[key], actual[key])
        self.assertTrue(replay['bookkeeping_complete'])
        self.assertFalse(replay['images_gradients_quality_or_training_verified'])
        self.assertFalse(replay['resume_permitted'])
        with self.assertRaises(ValueError):
            ledger.reserve_optimizer_update('B', 49)

    def test_initializer_all440_returns_precede_first_optimizer(self):
        ledger, _ = self.ledger()
        with self.assertRaisesRegex(ValueError, '440'):
            ledger.reserve_optimizer_construction('A')
        self.drive(ledger, 439)
        with self.assertRaisesRegex(ValueError, '440'):
            ledger.reserve_optimizer_construction('A')
        row = ledger.next_forward()
        token = ledger.reserve_forward(row)
        with self.assertRaisesRegex(ValueError, 'reservation'):
            ledger.reserve_optimizer_construction('A')
        ledger.returned(token)
        with self.assertRaises(ValueError):
            ledger.reserve_forward(ledger.next_forward())
        ledger.returned(ledger.reserve_optimizer_construction('A'))
        self.assertEqual(ledger.summary()['counters']['full_forward_returns'], 440)

    def test_exact_slot_bank_case_key_arm_and_types_cannot_be_forged(self):
        ledger, _ = self.ledger()
        row = ledger.next_forward()
        for field, value in [('slot', 2), ('slot', True), ('arm', 'B'), ('bank', 'new_monitors'),
                             ('case_key', row['id']), ('phase', 'photographic_optimization'), ('backward', True)]:
            bad = copy.deepcopy(row); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ledger.reserve_forward(bad)
        self.assertEqual(ledger.summary()['counters']['full_forward_attempts'], 0)
        ledger.returned(ledger.reserve_forward(row))
        with self.assertRaises(ValueError):
            ledger.reserve_forward(row)

    def test_pending_reservation_wrong_return_and_duplicate_return_refused(self):
        ledger, _ = self.ledger()
        row = ledger.next_forward()
        token = ledger.reserve_forward(row)
        for action in (lambda: ledger.reserve_forward(row), lambda: ledger.returned('invented'),
                       lambda: ledger.finish(), lambda: ledger.reserve_optimizer_construction('A')):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(ledger.summary()['counters']['full_forward_returns'], 0)
        ledger.returned(token)
        with self.assertRaises(ValueError):
            ledger.returned(token)

    def test_monitor_or_readout_backward_never_reserved(self):
        ledger, _ = self.ledger()
        ledger.returned(ledger.reserve_forward(ledger.next_forward()))
        with self.assertRaises(ValueError):
            ledger.reserve_backward(1)
        for slot in (True, 0, 1737, 1.0):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                ledger.reserve_backward(slot)
        self.assertEqual(ledger.summary()['counters']['backward_attempts'], 0)

    def test_five_roles_require_backward_each_and_only_then_one_commit(self):
        ledger, _ = self.ledger()
        self.drive(ledger, 440)
        ledger.returned(ledger.reserve_optimizer_construction('A'))
        for index in range(5):
            row = ledger.next_forward()
            self.assertEqual(row['id'].rsplit('__', 1)[1], C.ROLES[index])
            ledger.returned(ledger.reserve_forward(row))
            with self.assertRaises(ValueError):
                ledger.reserve_forward(ledger.next_forward())
            with self.assertRaises(ValueError):
                ledger.reserve_optimizer_update('A', 1)
            ledger.returned(ledger.reserve_backward(row['slot']))
            if index < 4:
                with self.assertRaises(ValueError):
                    ledger.reserve_optimizer_update('A', 1)
        for arm, update in [('B', 1), ('A', 2), ('A', True)]:
            with self.subTest(arm=arm, update=update), self.assertRaises(ValueError):
                ledger.reserve_optimizer_update(arm, update)
        with self.assertRaises(ValueError):
            ledger.reserve_forward(ledger.next_forward())
        ledger.returned(ledger.reserve_optimizer_update('A', 1))
        with self.assertRaises(ValueError):
            ledger.reserve_optimizer_update('A', 1)
        self.assertEqual(ledger.summary()['committed_updates'], {'A': 1, 'B': 0})

    def test_no_second_optimizer_or_wrong_active_arm_constructor(self):
        ledger, _ = self.ledger()
        self.drive(ledger, 440)
        with self.assertRaises(ValueError):
            ledger.reserve_optimizer_construction('B')
        ledger.returned(ledger.reserve_optimizer_construction('A'))
        with self.assertRaises(ValueError):
            ledger.reserve_optimizer_construction('A')
        self.assertEqual(ledger.summary()['counters']['optimizer_construction_attempts'], 1)

    def test_quality_stop_A_skips_only_remaining_A_ordered_subsequence(self):
        ledger, sink = self.ledger()
        self.drive(ledger, 440)
        ledger.quality_stop('A', 'Bound initial gate receipt failed', receipt_sha256='a' * 64)
        self.drive(ledger)
        ledger.finish()
        result = E.audit_journal(sink.raw(), self.selection)
        self.assertEqual(result['committed_updates'], {'A': 0, 'B': 48})
        self.assertEqual(result['counters']['optimizer_constructions'], 1)
        self.assertEqual(result['returned_forward_slots'], sorted(result['returned_forward_slots']))
        skipped = result['skipped_forward_slots']
        self.assertTrue(all(self.plan[slot - 1]['arm'] == 'A' for slot in skipped))
        self.assertEqual(len(skipped), 648)
        self.assertEqual(set(skipped) | set(result['returned_forward_slots']), set(range(1, 1737)))

    def test_quality_stop_B_before_A_training_preserves_A_and_skips_B_later(self):
        ledger, sink = self.ledger()
        self.drive(ledger, 440)
        ledger.quality_stop('B', 'Initial B quality gate receipt failed', receipt_sha256='b' * 64)
        self.drive(ledger)
        ledger.finish()
        result = E.audit_journal(sink.raw(), self.selection)
        self.assertEqual(result['committed_updates'], {'A': 48, 'B': 0})
        self.assertEqual(result['counters']['full_forward_returns'], 1088)
        self.assertIsNone(result['next_forward'])

    def test_quality_stop_after_checkpoint8_retains_prefix_and_other_arm(self):
        ledger, sink = self.ledger()
        constructed = self.drive(ledger, 576)
        ledger.quality_stop('A', 'Complete checkpoint8 gate failed', receipt_sha256='c' * 64)
        self.drive(ledger, constructed=constructed)
        ledger.finish()
        result = E.audit_journal(sink.raw(), self.selection)
        self.assertEqual(result['committed_updates'], {'A': 8, 'B': 48})
        self.assertEqual(result['quality_stops']['A']['checkpoint'], 8)

    def test_quality_stop_requires_complete_checkpoint_receipt_no_partial_update(self):
        ledger, _ = self.ledger()
        with self.assertRaises(ValueError):
            ledger.quality_stop('A', 'Too early', receipt_sha256='d' * 64)
        self.drive(ledger, 440)
        for pin in ('', 'G' * 64, '0' * 63):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                ledger.quality_stop('A', 'Bound stop', receipt_sha256=pin)
        self.drive(ledger, 445)
        with self.assertRaises(ValueError):
            ledger.quality_stop('A', 'Update1 is no checkpoint', receipt_sha256='d' * 64)

    def test_fatal_stop_keeps_unknown_pending_attempt_and_cannot_resume(self):
        ledger, sink = self.ledger()
        row = ledger.next_forward()
        token = ledger.reserve_forward(row)
        ledger.fatal_stop('Unknown partial operation outcome')
        result = E.audit_journal(sink.raw(), self.selection)
        self.assertEqual(result['status'], 'FATAL')
        self.assertEqual(result['counters']['full_forward_attempts'], 1)
        self.assertEqual(result['counters']['full_forward_returns'], 0)
        self.assertEqual(result['pending']['token'], token)
        for action in (lambda: ledger.reserve_forward(row), lambda: ledger.returned(token),
                       lambda: ledger.finish(), lambda: ledger.fatal_stop('Retry')):
            with self.assertRaises(ValueError):
                action()

    def test_reservation_sink_failure_counts_conservatively_and_no_permit(self):
        ledger, _ = self.ledger()
        with mock.patch.object(ledger, '_persist', side_effect=OSError('injected append failure')):
            with self.assertRaises(OSError):
                ledger.reserve_forward(ledger.next_forward())
        result = ledger.summary()
        self.assertEqual(result['status'], 'JOURNAL_FAILED')
        self.assertEqual(result['counters']['full_forward_attempts'], 1)
        self.assertEqual(result['counters']['full_forward_returns'], 0)
        self.assertIsNotNone(result['pending'])
        with self.assertRaises(ValueError):
            ledger.returned(result['pending']['token'])

    def test_return_sink_failure_does_not_credit_return_or_clear_pending(self):
        ledger, sink = self.ledger()
        token = ledger.reserve_forward(ledger.next_forward())
        with mock.patch.object(ledger, '_persist', side_effect=OSError('injected return append failure')):
            with self.assertRaises(OSError):
                ledger.returned(token)
        self.assertEqual(ledger.summary()['counters']['full_forward_returns'], 0)
        self.assertEqual(ledger.summary()['pending']['token'], token)
        replay = E.audit_journal(sink.raw(), self.selection)
        self.assertEqual(replay['counters']['full_forward_returns'], 0)
        self.assertFalse(replay['bookkeeping_complete'])

    def test_owned_store_callback_creates_no_unregistered_file(self):
        owned = {'operations.jsonl': b''}
        appends = []
        def persist(line):
            self.assertEqual(set(owned), {'operations.jsonl'})
            self.assertIs(type(line), bytes)
            owned['operations.jsonl'] += line
            appends.append('operations.jsonl')
        with mock.patch.object(E.os, 'open', side_effect=AssertionError('Unowned file open')):
            ledger = E.ExecutionLedger(self.selection, persist=persist)
            ledger.returned(ledger.reserve_forward(ledger.next_forward()))
            ledger.close()
        self.assertEqual(len(appends), 3)
        self.assertEqual(E.audit_journal(owned['operations.jsonl'], self.selection)['counters']['full_forward_returns'], 1)

    def test_real_path_journal_fsync_and_create_exclusive_no_resume(self):
        with tempfile.TemporaryDirectory(prefix='matched_ledger_', dir=ROOT / 'scratch') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to((ROOT / 'scratch').resolve()))
            path = Path(directory) / 'ledger.jsonl'
            with mock.patch.object(E.os, 'fsync', wraps=E.os.fsync) as synced:
                with E.ExecutionLedger.for_path(path, self.selection) as ledger:
                    ledger.returned(ledger.reserve_forward(ledger.next_forward()))
                self.assertEqual(synced.call_count, 3)
            self.assertEqual(E.audit_journal(path, self.selection)['counters']['full_forward_returns'], 1)
            with self.assertRaises(FileExistsError):
                E.ExecutionLedger.for_path(path, self.selection)

    def test_real_fsync_return_failure_keeps_zero_credit(self):
        with tempfile.TemporaryDirectory(prefix='matched_ledger_', dir=ROOT / 'scratch') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to((ROOT / 'scratch').resolve()))
            with E.ExecutionLedger.for_path(Path(directory) / 'ledger.jsonl', self.selection) as ledger:
                token = ledger.reserve_forward(ledger.next_forward())
                with mock.patch.object(E.os, 'fsync', side_effect=OSError('injected fsync')):
                    with self.assertRaises(OSError):
                        ledger.returned(token)
                self.assertEqual(ledger.summary()['status'], 'JOURNAL_FAILED')
                self.assertEqual(ledger.summary()['counters']['full_forward_returns'], 0)

    def test_truncated_mutated_duplicate_or_reordered_journal_refused(self):
        ledger, sink = self.ledger()
        ledger.returned(ledger.reserve_forward(ledger.next_forward()))
        originals = sink.lines
        bad = [sink.raw()[:-1], sink.raw()[:-30], b'', originals[0] + originals[2] + originals[1],
               sink.raw().replace(b'initial_quality', b'initial_qualitX', 1),
               originals[0].replace(b'"seq":1', b'"seq":1,"seq":1') + b''.join(originals[1:])]
        for index, raw in enumerate(bad):
            with self.subTest(index=index), self.assertRaises((ValueError, UnicodeError)):
                E.audit_journal(raw, self.selection)

    def test_rehashed_false_return_or_out_of_order_slot_still_refused(self):
        ledger, sink = self.ledger()
        ledger.returned(ledger.reserve_forward(ledger.next_forward()))
        def altered(change):
            records = [json.loads(line) for line in sink.lines]
            change(records)
            previous = '0' * 64
            encoded = []
            for seq, record in enumerate(records, 1):
                record.pop('sha256', None)
                record['seq'], record['previous_sha256'] = seq, previous
                record['sha256'] = E._digest(record)
                previous = record['sha256']
                encoded.append(E._json(record) + b'\n')
            return b''.join(encoded)
        for change in (lambda rs: rs[2]['payload'].update(token='fabricated'),
                       lambda rs: rs[1]['payload']['operation'].update(slot=2),
                       lambda rs: rs[0]['payload']['limits'].update(backward_attempts=481)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                E.audit_journal(altered(change), self.selection)

    def test_summary_mutation_cannot_change_ledger_or_plan(self):
        ledger, _ = self.ledger()
        summary = ledger.summary()
        summary['counters']['full_forward_attempts'] = 999
        summary['next_forward']['bank'] = 'mutated'
        row = ledger.next_forward(); row['slot'] = 99
        self.assertEqual(ledger.summary()['counters']['full_forward_attempts'], 0)
        self.assertEqual(ledger.next_forward(), self.plan[0])

    def test_rehashed_begin_limits_require_exact_integer_types_and_keys(self):
        ledger, sink = self.ledger()
        original = json.loads(sink.lines[0])
        replacements = [lambda d: d.update(gradient_queries=False),
                        lambda d: d.update(worker_seconds=5400.0),
                        lambda d: d.update(optimizer_constructions=2.0),
                        lambda d: d.update(extra=0),
                        lambda d: d.pop('per_phase_seconds')]
        for change in replacements:
            record = copy.deepcopy(original)
            record.pop('sha256')
            change(record['payload']['limits'])
            record['sha256'] = E._digest(record)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'first journal'):
                E.audit_journal(E._json(record) + b'\n', self.selection)

    def test_rehashed_quality_checkpoint_bool_cannot_replace_integer_zero(self):
        ledger, sink = self.ledger()
        self.drive(ledger, 440)
        ledger.quality_stop('A', 'Initial gate failed', receipt_sha256='e' * 64)
        record = json.loads(sink.lines[-1])
        self.assertEqual(record['payload']['checkpoint'], 0)
        record.pop('sha256')
        record['payload']['checkpoint'] = False
        record['sha256'] = E._digest(record)
        raw = b''.join(sink.lines[:-1]) + E._json(record) + b'\n'
        with self.assertRaisesRegex(ValueError, 'skip receipt'):
            E.audit_journal(raw, self.selection)

    def test_no_neural_execution_in_source_or_imports_and_limits_exact(self):
        source = Path(E.__file__).read_text()
        tree = ast.parse(source, feature_version=(3, 10))
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse({'backward', 'step', 'cuda', 'forward', 'sleep'} & calls)
        self.assertNotIn('1260', source)
        ledger, _ = self.ledger()
        self.assertEqual(json.loads(ledger._persist.lines[0])['payload']['limits'], C.LIMITS)
        self.assertEqual(BLOCKER.attempted, [])
        self.assertFalse({'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} & {name.split('.')[0] for name in sys.modules})


class TimeBudgetTests(unittest.TestCase):
    def test_absolute5400_worker5460_external_and300_phase_deadlines(self):
        budget = E.TimeBudget(100.)
        result = budget.deadline(200., phase_started_at=100.)
        self.assertEqual((result['worker_deadline'], result['external_deadline'], result['deadline']), (5500., 5560., 400.))
        self.assertFalse(result['timeout_enforced_by_helper'])
        result = budget.deadline(5499., phase_started_at=5490.)
        self.assertEqual(result['deadline'], 5500.)
        self.assertTrue(budget.deadline(5500., phase_started_at=5490.)['expired'])

    def test_active_arm2700_accumulates_only_active_intervals(self):
        budget = E.TimeBudget(0.)
        budget.enter_arm('A', 0.)
        budget.leave_arm(1000.)
        budget.enter_arm('B', 1000.)
        budget.leave_arm(2000.)
        budget.enter_arm('A', 2000.)
        result = budget.deadline(3690., phase_started_at=3690.)
        self.assertEqual(result['deadline'], 3700.)
        self.assertEqual(result['remaining_seconds'], 10.)
        self.assertTrue(budget.deadline(3700., phase_started_at=3690.)['expired'])

    def test_failure45_receipt5_and_grace20_respect_outer5460(self):
        budget = E.TimeBudget(0.)
        capture = budget.deadline(5400., phase_started_at=5400., mode='failure_capture')
        self.assertEqual(capture['deadline'], 5435.)
        receipt = budget.deadline(5435., phase_started_at=5435., mode='failure_receipt')
        self.assertEqual(receipt['deadline'], 5440.)
        grace = budget.deadline(5440., phase_started_at=5440., mode='termination_grace')
        self.assertEqual(grace['deadline'], 5460.)
        early = E.TimeBudget(0.).deadline(100., phase_started_at=100., mode='failure_capture')
        self.assertEqual(early['deadline'], 145.)

    def test_export_separate600_and660_external(self):
        self.assertEqual(E.export_deadlines(1000.), {'export_deadline': 1600.,
                         'export_external_deadline': 1660., 'termination_grace_seconds': 20,
                         'timeout_enforced_by_helper': False})

    def test_failure_entry_anchors_cannot_be_renewed_by_later_queries(self):
        for mode, duration in (('failure_capture', 45), ('failure_receipt', 5),
                               ('termination_grace', 20)):
            with self.subTest(mode=mode):
                budget = E.TimeBudget(0.)
                first = budget.deadline(0., phase_started_at=0., mode=mode)
                self.assertEqual(first['deadline'], float(duration))
                renewed = budget.deadline(duration - 1., phase_started_at=duration - 1., mode=mode)
                self.assertEqual(renewed['deadline'], first['deadline'])
                self.assertEqual(renewed['remaining_seconds'], 1.)
                expired = budget.deadline(float(duration), phase_started_at=float(duration), mode=mode)
                self.assertEqual(expired['deadline'], first['deadline'])
                self.assertTrue(expired['expired'])
                still_expired = budget.deadline(duration + 100., phase_started_at=duration + 100., mode=mode)
                self.assertEqual(still_expired['remaining_seconds'], 0.)
                self.assertEqual(still_expired['deadline'], first['deadline'])

    def test_invalid_nonfinite_boolean_negative_time_overlap_and_backwards_refused(self):
        for invalid in (True, -1, float('nan'), float('inf'), '1'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                E.TimeBudget(invalid)
        budget = E.TimeBudget(10.)
        budget.enter_arm('A', 10.)
        with self.assertRaises(ValueError):
            budget.enter_arm('B', 11.)
        with self.assertRaises(ValueError):
            budget.leave_arm(9.)
        with self.assertRaises(ValueError):
            budget.deadline(12., phase_started_at=13.)
        with self.assertRaises(ValueError):
            budget.deadline(12., phase_started_at=10., mode='resume')


if __name__ == '__main__':
    unittest.main()
