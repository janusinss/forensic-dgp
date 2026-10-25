"""Pure tail-order and append-only durability fixtures; no inference."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_u96_smoothanchors_lr_fixed_DEV_tail_v1 as T


class TailPureFixtures(unittest.TestCase):
    def test_47carried_and89tail_exact_unselected_order(self):
        cases = [{'id': f'case{i:02d}'} for i in range(68)]
        slots = T.slot_order({'cases': cases})
        self.assertEqual(len(slots), 136)
        self.assertEqual(slots[46]['case_id'], 'case23')
        self.assertEqual(slots[46]['arm'], 'incumbent_U96')
        self.assertEqual(slots[47], {'case_id': 'case23', 'case_index': 23, 'arm': 'postfusion_LR_u64',
                                    'arm_index': 1, 'aggregate_output_index': 48})
        self.assertEqual(len(slots[47:]), 89)
        self.assertEqual(sum(s['arm'] == 'incumbent_U96' for s in slots[47:]), 44)
        self.assertEqual(sum(s['arm'] == 'postfusion_LR_u64' for s in slots[47:]), 45)
        self.assertEqual(48 + 89, 137)
        self.assertEqual(47 + 89, 136)

    def test_appendonly_numbered_status_and_existing_collision_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            out = Path(directory)
            with patch.object(T, 'OUT', out):
                journal = T.Journal(out / 'journal')
                first = journal.append({'phase': 'reserved', 'tail_counters': {'attempts': 1, 'returns': 0}})
                before = (ROOT / first['path']).read_bytes()
                second = journal.append({'phase': 'returned', 'tail_counters': {'attempts': 1, 'returns': 1}})
                self.assertNotEqual(first['path'], second['path'])
                self.assertEqual((ROOT / first['path']).read_bytes(), before)
                conflicting = out / 'journal/000003.json'; conflicting.write_bytes(b'original partial')
                with self.assertRaises(FileExistsError): journal.append({'phase': 'must not overwrite'})
                self.assertEqual(conflicting.read_bytes(), b'original partial')

    def test_empty_and_malformed_journal_parts_preserved_without_unknown_reset(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            out = Path(directory)
            with patch.object(T, 'OUT', out):
                journal = T.Journal(out / 'journal')
                journal.append({'phase': 'reserved', 'tail_forward_outcome_unknown': True})
                (out / 'journal/000002.json').write_bytes(b'')
                (out / 'journal/000003.json').write_bytes(b'{partial')
                parsed, partial = T.journal_readback(out / 'journal')
                self.assertEqual(len(parsed), 1)
                self.assertTrue(parsed[-1][1]['tail_forward_outcome_unknown'])
                self.assertEqual([r['bytes'] for r in partial], [0, 8])
                self.assertEqual((out / 'journal/000002.json').read_bytes(), b'')
                self.assertEqual((out / 'journal/000003.json').read_bytes(), b'{partial')

    def test_inert_import_help_blocks_heavy_numerical_modules(self):
        code = """import importlib,sys
sys.path.insert(0,sys.argv[1])
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in{'torch','numpy','PIL','psutil','fastapi'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
m=importlib.import_module('run_u96_smoothanchors_lr_fixed_DEV_tail_v1')
sys.argv=[m.__file__,'--help']
try:m.main()
except SystemExit as e:assert e.code==0
assert 'torch'not in sys.modules
"""
        run = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT / 'scripts')], capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_no_replace_retry_training_calls_or_changed_resource_caps(self):
        source = Path(T.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 10))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('replace', calls)
        self.assertFalse({'backward', 'step', 'zero_grad'} & set(calls))
        self.assertEqual(T.LIMITS['internal_seconds'], 2400)
        self.assertEqual(T.LIMITS['external_seconds'], 2460)
        self.assertEqual(T.LIMITS['full_forward_attempts'], 89)
        self.assertEqual(T.LIMITS['maximum_worker_RSS_bytes'], T.B.LIMITS['maximum_worker_RSS_bytes'])
        self.assertEqual(T.LIMITS['maximum_output_bytes'], T.B.LIMITS['maximum_output_bytes'])
        self.assertNotIn('torch', sys.modules)

    def test_prepare_without_actual_admission_and_review_fails_before_io(self):
        run = subprocess.run([sys.executable, '-B', str(Path(T.__file__)), '--prepare-plan'], capture_output=True, text=True, timeout=15)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('Actual carried47pixeladmission and current tail peer review required', run.stderr)

    def test_failed_reservation_before_journal_file_keeps_failure_counter_unknown(self):
        before = {'forward_attempts': 0, 'forward_returns': 0, 'saved_outputs': 0}
        reserved = {'forward_attempts': 1, 'forward_returns': 0, 'saved_outputs': 0}
        last = {'complete': False, 'tail_counters': before, 'tail_forward_outcome_unknown': False}
        failure = {'tail_counters': reserved, 'tail_forward_outcome_unknown': True}
        result = T.supervisor_outcome(1, False, last, [], failure, False)
        self.assertTrue(result['tail_forward_outcome_unknown'])
        self.assertEqual(result['last_durable_tail_counters'], before)
        self.assertEqual(result['failure_tail_counters'], reserved)
        self.assertEqual(result['tail_counters'], reserved)
        self.assertFalse(result['complete89_tail_observed'])

    def test_missing_failure_or_interrupted_exit_is_conservatively_unknown(self):
        counters = {'forward_attempts': 0, 'forward_returns': 0, 'saved_outputs': 0}
        last = {'complete': False, 'tail_counters': counters, 'tail_forward_outcome_unknown': False}
        self.assertTrue(T.supervisor_outcome(1, False, last, [], None, False)['tail_forward_outcome_unknown'])
        self.assertTrue(T.supervisor_outcome(1, False, last, [], None, True)['tail_forward_outcome_unknown'])
        final = {'complete': True, 'tail_counters': {'forward_attempts': 89, 'forward_returns': 89, 'saved_outputs': 89}, 'tail_forward_outcome_unknown': False}
        success = T.supervisor_outcome(0, False, final, [], None, False)
        self.assertTrue(success['complete89_tail_observed'])
        self.assertFalse(success['tail_forward_outcome_unknown'])

    def test_actual_supervisor_reconciles_failure_after_journal_creation_denied(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            parent = Path(directory); prep = parent / 'prep'; prep.mkdir(); out = parent / 'owned-worker'
            before = {'forward_attempts': 0, 'forward_returns': 0, 'saved_outputs': 0}
            reserved = {'forward_attempts': 1, 'forward_returns': 0, 'saved_outputs': 0}
            class NonNeuralFakeChild:
                def __init__(self, *args, **kwargs):
                    out.mkdir()
                    T.Journal(out / 'journal').append({'complete': False, 'tail_counters': before,
                        'tail_forward_outcome_unknown': False, 'label': 'before_reservation'})
                    T.save_json(out / 'failure.json', {'complete': True, 'tail_counters': reserved,
                        'tail_forward_outcome_unknown': True, 'error': 'synthetic permission denial before journal file'})
                def wait(self, timeout): return 1
            with patch.object(T, 'PREP', prep), patch.object(T, 'OUT', out), patch.object(T, 'verify', return_value=(None, None)), patch.object(T.subprocess, 'Popen', NonNeuralFakeChild):
                self.assertEqual(T.supervise('0' * 64), 2)
                receipt = T.load(prep / 'tail_supervisor_v1.json')
                self.assertTrue(receipt['tail_forward_outcome_unknown'])
                self.assertEqual(receipt['last_durable_tail_counters'], before)
                self.assertEqual(receipt['failure_tail_counters'], reserved)
                self.assertEqual(receipt['tail_counters'], reserved)
                self.assertEqual(receipt['partial_journal_files'], [])
                self.assertTrue(receipt['original_unknown_reservation_retained'])

    def test_actual_supervisor_unreadable_failure_publishes_unknown_without_reread(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            parent = Path(directory); prep = parent / 'prep'; prep.mkdir(); out = parent / 'owned-worker'
            class NonNeuralFakeChild:
                def __init__(self, *args, **kwargs):
                    out.mkdir(); (out / 'failure.json').write_bytes(b'original unreadable evidence')
                def wait(self, timeout): return 1
            original = Path.read_bytes; denied = []
            def guarded_read(path):
                if path == out / 'failure.json':
                    denied.append(path); raise PermissionError('synthetic unreadable owned failure')
                return original(path)
            with patch.object(T, 'PREP', prep), patch.object(T, 'OUT', out), patch.object(T, 'verify', return_value=(None, None)), patch.object(T.subprocess, 'Popen', NonNeuralFakeChild), patch.object(Path, 'read_bytes', guarded_read):
                self.assertEqual(T.supervise('0' * 64), 2)
                receipt = T.load(prep / 'tail_supervisor_v1.json')
                self.assertTrue(receipt['tail_forward_outcome_unknown'])
                self.assertTrue(receipt['failure_receipt_unreadable'])
                self.assertIsNone(receipt['failure_receipt'])
                self.assertTrue(receipt['failure_receipt_read_error']['unreadable_bytes_not_given_an_invented_hash'])
                self.assertEqual(len(denied), 1)

    def test_actual_carried47_admission_has_exact_prepare_schema_and_record_joins(self):
        admission = T.load(ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_saved_audit_v1/carried47_pixel_admission_v1.json')
        original = T.load(T.B.PLAN)
        self.assertTrue(admission['carried47_pixel_admission_pass'])
        rows = admission['carried_outputs']
        self.assertEqual([(r['case_id'], r['arm']) for r in rows],
                         [(s['case_id'], s['arm']) for s in T.slot_order(original)[:47]])
        for row in rows:
            self.assertEqual(set(row['inputs']), {'original', 'prepared', 'support', 'target'})
            self.assertEqual(set(row['outputs']), {'native512', 'logits', 'LQ', 'unclipped', 'raw', 'png'})
            record = T.load(T.bound(row['record']))
            self.assertEqual(record['case_id'], row['case_id'])
            self.assertEqual(record['provenance']['arm'], row['arm'])
            self.assertTrue(all(record[key] == value for key, value in row['outputs'].items()))


if __name__ == '__main__': unittest.main()
