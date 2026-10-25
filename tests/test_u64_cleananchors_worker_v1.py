"""Pure worker ledger/contracts/AST/failure helpers; never call worker.run or Torch."""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from research.codeformer_u64_cleananchors_continuation_v1 import worker as W
from research.codeformer_u64_cleananchors_continuation_v1 import contract as C
from research.codeformer_u96_spatial_feasibility_v1 import worker as H

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_u64_cleananchors_continuation_v1/worker.py'


class DurableOperationLedger(unittest.TestCase):
    def test_reservation_is_durable_before_caller_operation(self):
        events = []
        ledger = W.OperationLedger(events.append)
        token = ledger.reserve('forward', category='initial_quality_bank_forwards', id='fixture')
        self.assertEqual(events[0]['kind'], 'forward_reserved')
        self.assertEqual(events[0]['token'], token)
        self.assertEqual(events[0]['counters']['full_forward_attempts'], 1)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])
        ledger.returned(token, synchronized_return=True)
        self.assertEqual(events[-1]['kind'], 'forward_returned')
        self.assertEqual(events[-1]['counters']['full_forward_returns'], 1)
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])

    def test_failed_reservation_write_leaves_unknown_and_no_return(self):
        def failed(_):
            raise OSError('fixture fsync failure')
        ledger = W.OperationLedger(failed)
        with self.assertRaises(OSError):
            ledger.reserve('optimizer_update', update=1)
        self.assertEqual(ledger.counters['optimizer_update_attempts'], 1)
        self.assertEqual(ledger.counters['optimizer_updates'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_failed_return_write_never_claims_known_return(self):
        events = []
        def writer(row):
            if row['kind'].endswith('_returned'):
                raise OSError('fixture durable return failure')
            events.append(row)
        ledger = W.OperationLedger(writer)
        token = ledger.reserve('backward', id='fixture')
        with self.assertRaises(OSError):
            ledger.returned(token)
        self.assertEqual(ledger.counters['backward_returns'], 0)
        self.assertEqual(len(ledger.snapshot()['unknown_reservations']), 1)

    def test_return_requires_exact_unmatched_integer_token(self):
        ledger = W.OperationLedger(lambda _: None)
        token = ledger.reserve('optimizer_construction')
        for bad in (True, '1', 0, 2):
            with self.subTest(token=bad), self.assertRaises(ValueError):
                ledger.returned(bad)
        ledger.returned(token)
        with self.assertRaises(ValueError):
            ledger.returned(token)

    def test_reserved_metadata_cannot_replace_event_identity(self):
        ledger = W.OperationLedger(lambda _: None)
        for field in ('token', 'counters'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ledger.reserve('backward', **{field: {}})
        token = ledger.reserve('backward')
        with self.assertRaises(ValueError):
            ledger.returned(token, counters={})
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_every_forward_category_and_global_292_cap(self):
        events = []
        ledger = W.OperationLedger(events.append)
        for category, cap in W.CATEGORIES.items():
            for _ in range(cap):
                ledger.returned(ledger.reserve('forward', category=category))
            with self.subTest(category=category), self.assertRaises(ValueError):
                ledger.reserve('forward', category=category)
        self.assertEqual(ledger.counters['full_forward_attempts'], 292)
        self.assertEqual(ledger.counters['full_forward_returns'], 292)
        self.assertEqual(sum(ledger.counters[k] for k in W.CATEGORIES), 292)
        self.assertFalse(ledger.pending)
        self.assertEqual(len(events), 584)

    def test_other_operation_caps_64_backward_1_Adam_16_updates(self):
        ledger = W.OperationLedger(lambda _: None)
        for kind, attempts, returns, cap in (
            ('backward', 'backward_attempts', 'backward_returns', 64),
            ('optimizer_construction', 'optimizer_constructions', 'optimizer_construction_returns', 1),
            ('optimizer_update', 'optimizer_update_attempts', 'optimizer_updates', 16),
        ):
            for _ in range(cap):
                ledger.returned(ledger.reserve(kind))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ledger.reserve(kind)
            self.assertEqual(ledger.counters[attempts], cap)
            self.assertEqual(ledger.counters[returns], cap)

    def test_unknown_operations_and_forward_category_refused(self):
        for kind, metadata in [('gradient', {}), ('warmup', {}), ('forward', {}),
                               ('forward', {'category': 'reference_forwards'})]:
            with self.subTest(kind=kind, metadata=metadata), self.assertRaises(ValueError):
                W.OperationLedger(lambda _: None).reserve(kind, **metadata)


class ScheduleAndSource(unittest.TestCase):
    def test_learning_rate_is_fresh_four_update_warmup(self):
        self.assertEqual([W.learning_rate(i) for i in range(1, 5)],
                         [0.000025, 0.00005, 0.00007500000000000001, 0.0001])
        self.assertTrue(all(W.learning_rate(i) == 0.0001 for i in range(4, 17)))
        for bad in (0, 17, True, 1.0, '1'):
            with self.subTest(update=bad), self.assertRaises(ValueError):
                W.learning_rate(bad)

    def test_actual_contract_plan_matches_all_worker_categories(self):
        parent = (ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json').read_bytes()
        selection = C.select_parent_cases(parent)
        plan = C.operation_plan(selection)
        phases = {
            'initial_u64_quality_bank': 'initial_quality_bank_forwards',
            'deterministic_same_u64_repeat': 'deterministic_repeat_forwards',
            'new_quality_snapshot': 'candidate_snapshot_forwards_after_initializer',
            'paired_TRAIN_optimization': 'candidate_training_forwards',
        }
        counts = {category: sum(phases[row['phase']] == category for row in plan) for category in W.CATEGORIES}
        self.assertEqual(counts, W.CATEGORIES)
        self.assertEqual(len(plan), 292)
        self.assertEqual(sum(row['backward'] for row in plan), 64)
        self.assertEqual(sum(row['optimizer_application_after_this_forward'] for row in plan), 16)
        self.assertEqual([row['id'] for row in plan if row['phase'] == 'deterministic_same_u64_repeat'],
                         [row['id'] for row in selection['monitors'][:4]])

    def test_supervisor_binds_exact_protocol_and_actual_parent(self):
        sha = 'a' * 64
        env = {'FORENSIC_U64_CLEANANCHORS_SUPERVISED': '1',
               'FORENSIC_U64_CLEANANCHORS_PROTOCOL': sha,
               'FORENSIC_U64_CLEANANCHORS_SUPERVISOR_PID': '123'}
        W.supervisor_context(sha, environ=env, parent_pid=123)
        for changed in ({}, {**env, 'FORENSIC_U64_CLEANANCHORS_SUPERVISOR_PID': '124'},
                        {**env, 'FORENSIC_U64_CLEANANCHORS_PROTOCOL': 'b' * 64}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                W.supervisor_context(sha, environ=changed, parent_pid=123)

    def test_old_helper_exact_pin_and_failure_budget_are_inert(self):
        path = ROOT / W.HELPER
        self.assertEqual((path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()), W.HELPER_PIN)
        self.assertEqual(H.failure_capture_budget(1200), 30)
        self.assertEqual(H.failure_capture_budget(1250), 0)
        self.assertEqual(H.failure_capture_budget(1260), 0)
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'helper'
                             and n.func.attr == 'run' for n in ast.walk(tree)))

    def test_import_and_source_have_no_actual_execution_side_effect(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('PIL', sys.modules)
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn('torch', [alias.name for alias in node.names])
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        guard = next(node.lineno for node in ast.walk(run) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'supervisor_context')
        imported = next(node.lineno for node in ast.walk(run) if isinstance(node, ast.Import)
                        and any(alias.name == 'torch' for alias in node.names))
        self.assertLess(guard, imported)

    def test_delivery_clamps_native_before_bilinear_resize(self):
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        saved = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'saved_record')
        raw = next(node.value for node in ast.walk(saved) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'raw' for t in node.targets))
        self.assertIsInstance(raw, ast.Call)
        self.assertIsInstance(raw.func, ast.Attribute)
        self.assertEqual(raw.func.attr, 'interpolate')
        self.assertIsInstance(raw.args[0], ast.Call)
        self.assertEqual(raw.args[0].func.attr, 'clamp')
        # One align_corners=False 2x downsample uses the four neighboring centers.
        native_rgb = np.asarray([[-.2, .2], [.2, .2]], dtype=np.float32)
        before_resize = np.clip(native_rgb, 0, 1).mean(dtype=np.float32)
        after_resize = np.clip(native_rgb.mean(dtype=np.float32), 0, 1)
        self.assertAlmostEqual(float(before_resize), .15, places=7)
        self.assertAlmostEqual(float(after_resize), .10, places=7)
        self.assertNotEqual(before_resize, after_resize)

    def test_failure_capture_alarm_is_reraised_before_fallback(self):
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
                  and any(h.name == 'valid_capture_error' for h in node.handlers)]
        self.assertEqual(len(blocks), 1)
        handlers = blocks[0].handlers
        timeout = next(i for i, h in enumerate(handlers) if isinstance(h.type, ast.Name) and h.type.id == 'TimeoutError')
        fallback = next(i for i, h in enumerate(handlers) if h.name == 'valid_capture_error')
        self.assertLess(timeout, fallback)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in handlers[timeout].body))

    def test_full_state_rejoins_actual_durable_npz_not_only_memory_payload(self):
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        full_state = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'full_state')
        loads = [node for node in ast.walk(full_state) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == 'np' and node.func.attr == 'load']
        self.assertEqual(len(loads), 1)
        self.assertTrue(any(k.arg == 'allow_pickle' and isinstance(k.value, ast.Constant) and k.value.value is False
                            for k in loads[0].keywords))
        self.assertIsInstance(loads[0].args[0], ast.Call)
        self.assertEqual(loads[0].args[0].func.id, 'checked')
        source = ast.get_source_segment(SOURCE.read_text(encoding='utf-8'), full_state)
        self.assertIn('np.array_equal(decoded[n], values[n])', source)
        self.assertIn("schema['canonical_sha256']", source)

    def test_extracted_timer_fixture_rejects_expired_and_existing_alarm(self):
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        timed_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'timed')
        calls = []
        current = {'handler': 'prior', 'timer': (0.0, 0.0)}
        def set_handler(_, value):
            calls.append(('handler', value)); current['handler'] = value
        def set_timer(_, value):
            calls.append(('timer', value))
        fake = SimpleNamespace(SIGALRM=1, ITIMER_REAL=2,
            getsignal=lambda _: current['handler'], signal=set_handler,
            getitimer=lambda _: current['timer'], setitimer=set_timer)
        namespace = {'contextmanager': contextmanager, 'need': W.need, 'signal': fake}
        module = ast.Module(body=[timed_node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<extracted pure timer fixture>', 'exec'), namespace)
        timed = namespace['timed']
        for seconds in (0, -1):
            with self.assertRaises(ValueError):
                with timed(seconds, 'fixture'):
                    self.fail('Expired budget must not execute body')
        self.assertEqual(calls, [])
        with timed(3, 'fixture'):
            self.assertEqual(calls[1], ('timer', 3))
        self.assertEqual(calls[-2:], [('timer', 0), ('handler', 'prior')])
        current['timer'] = (1.0, 0.0)
        before = len(calls)
        with self.assertRaises(ValueError):
            with timed(1, 'fixture'):
                self.fail('Existing timer must not be replaced')
        self.assertEqual(len(calls), before)


if __name__ == '__main__':
    unittest.main()
