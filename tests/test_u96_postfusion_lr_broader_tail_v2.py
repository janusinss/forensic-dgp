"""Finite pure/owned-subprocess tail tests; no Torch or frozen artifact reads."""
import ast
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_u96_postfusion_lr_broader_tail_v2 as T

ORIGINAL = T.load(T.O.PLAN)  # Metadata only; never invoke dependency/model validation in tests.
RETAINED = T.load(T.AUDIT / 'retained61_output_manifest_v1.json')
REAL_POPEN = subprocess.Popen
TEST_ROOT = ROOT / 'scratch' / ('broader-tail-source-tests-' + uuid.uuid4().hex)
FIXTURE = '''import json,os,sys,time
from pathlib import Path
folder=Path(os.environ['TAIL_TEST_FOLDER']);mode=os.environ['TAIL_TEST_MODE']
startup=json.loads((folder/'startup.json').read_text())
def write(name,value):
 with (folder/(name+'.new')).open('x') as stream:json.dump(value,stream)
 os.replace(folder/(name+'.new'),folder/name)
write('identity.json',{'token':startup['token'],'chunk':startup['chunk'],'worker_pid':os.getpid()+(1 if mode=='identity' else 0),'python_prefix':sys.prefix,'python_executable':sys.executable})
while not (folder/'command.json').exists():time.sleep(.01)
if mode=='hang':time.sleep(60)
if mode=='crash':raise SystemExit(7)
payload=json.loads(Path(os.environ['TAIL_TEST_PAYLOAD']).read_text())
payload['token']=startup['token'];payload['worker_pid']=os.getpid()
if mode=='bool':payload['counters']['attempts']=True
write('terminal.json',payload)
'''


class TailTests(unittest.TestCase):
    def setUp(self):
        self.root = TEST_ROOT / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        self.assertNotIn('torch', sys.modules)

    def fixture_file(self, name, raw=b'finite source-only fixture'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return T.pin(path)

    def test_exact_disjoint655_fixed41chunks(self):
        remaining = ORIGINAL['forward_slots'][61:]
        result = T.chunks(remaining)
        self.assertEqual(len(result), 41)
        self.assertEqual([len(row['slots']) for row in result], [16] * 40 + [15])
        self.assertEqual([slot for row in result for slot in row['slots']], remaining)
        self.assertEqual(remaining[0]['new_output_index'], 62)
        self.assertEqual(remaining[0]['arm'], 'postfusion_LR_u64')
        self.assertEqual(remaining[0]['case_id'], 'validation_va_ffhq_59814_clear')
        self.assertEqual(remaining[-1]['new_output_index'], 716)
        self.assertEqual(Counter(row['arm'] for row in remaining), Counter(T.LIMITS['per_arm']))
        self.assertEqual(RETAINED['rows'][-1]['slot'], ORIGINAL['forward_slots'][60])
        self.assertFalse({r['new_output_index'] for r in remaining} & set(range(1, 62)))

    def test_duplicate_missing_reordered_slots_refused(self):
        good = ORIGINAL['forward_slots'][61:]
        for slots in (good[:-1], good[1:] + [good[-1]], [good[1], good[0], *good[2:]]):
            with self.assertRaises(ValueError):
                T.chunks(slots)

    def test_limits_are_global_cumulative_and_unrelaxed(self):
        self.assertEqual(T.LIMITS['maximum_new_output_bytes'] + 381083149, 6 * 1024 ** 3)
        self.assertEqual(T.LIMITS['global_seconds'], 13829)
        self.assertEqual(T.LIMITS['cleanup_seconds'], 10)
        self.assertEqual(T.LIMITS['minimum_available_RAM_bytes'], T.O.LIMITS['minimum_available_RAM_bytes'])
        self.assertEqual(T.LIMITS['maximum_worker_RSS_bytes'], T.O.LIMITS['maximum_worker_RSS_bytes'])
        self.assertEqual(T.LIMITS['forward_seconds'], T.O.LIMITS['individual_forward_watchdog_seconds'])
        self.assertEqual(T.LIMITS['load_seconds'], T.O.LIMITS['model_load_watchdog_seconds'])

    def test_global_watchdog_interrupts_pure_heavy_loop(self):
        def heavy(unused):
            limit = time.monotonic() + 2
            while time.monotonic() < limit:
                pass
        started = time.monotonic()
        with patch.object(T, 'LIMITS', {**T.LIMITS, 'global_seconds': .03}), \
             patch.object(T, 'supervise_body', side_effect=heavy):
            with self.assertRaises(KeyboardInterrupt):
                T.supervise('a' * 64)
        self.assertLess(time.monotonic() - started, .5)

    def test_prepare_verify_roundtrip_is_inert_and_source_bound(self):
        prep = self.root / 'prep'; prep.mkdir()
        review = self.root / 'review.json'
        review.write_bytes(T.O.B.json_bytes({'complete': True, 'source_review_pass': True,
            'sources': [T.pin(Path(T.__file__).resolve()), T.pin(Path(__file__).resolve())]}))
        proof = self.fixture_file('proof')
        with patch.object(T, 'PREP', prep), patch.object(T, 'PLAN', prep / 'plan.json'), \
             patch.object(T, 'dependency_contract', return_value=(ORIGINAL, RETAINED, {'retained61': proof})), \
             patch.object(T.subprocess, 'Popen') as spawn:
            receipt = T.prepare(review)
            plan, original, retained = T.verify(receipt['sha256'])
            self.assertEqual(len(plan['remaining_slots']), 655)
            self.assertEqual(len(plan['chunks']), 41)
            self.assertEqual(plan['status'], 'SEALED_PREPARED_UNRUN')
            with self.assertRaises(ValueError):
                T.prepare(review)
            spawn.assert_not_called()

    def test_finite_strict_counter_contract(self):
        slots = ORIGINAL['forward_slots'][61:77]
        counts = T.blank_counts()
        self.assertEqual(T.counts_valid(counts, slots), counts)
        for key, value in (('attempts', True), ('attempts', 17), ('returns', 1), ('saved', -1)):
            wrong = T.blank_counts(); wrong[key] = value
            with self.assertRaises(ValueError):
                T.counts_valid(wrong, slots)

    def test_exclusive_writer_byte_reservation_and_path(self):
        writer = T.Writer(self.root, 4)
        writer.write(self.root / 'one', b'1234')
        with self.assertRaises(ValueError):
            writer.write(self.root / 'two', b'5')
        with self.assertRaises(ValueError):
            T.safe(self.root / '..' / 'escape')
        with self.assertRaises((FileExistsError, ValueError)):
            writer.write(self.root / 'one', b'')
        self.assertFalse((self.root / 'two').exists())

    def test_atomic_JSON_publication_has_no_partial_ready_file(self):
        writer = T.Writer(self.root, 1024)
        original_replace = os.replace
        def replace(source, target):
            self.assertFalse(Path(target).exists())
            self.assertEqual(T.json_file(source), {'identity': 123})
            return original_replace(source, target)
        with patch.object(T.os, 'replace', side_effect=replace):
            record = writer.json(self.root / 'identity.json', {'identity': 123})
        self.assertEqual(T.json_file(T.bound(record)), {'identity': 123})
        self.assertFalse((self.root / 'identity.json.new').exists())

    def test_owned_reap_attempts_kill_after_terminate_error(self):
        class Child:
            returncode = None
            killed = False
            waits = 0
            def wait(self, timeout):
                self.waits += 1
                if not self.killed:
                    raise subprocess.TimeoutExpired('finite fixture', timeout)
                self.returncode = -1
                return -1
            def terminate(self):
                raise OSError('Simulated exact-handle terminate failure')
            def kill(self):
                self.killed = True
            def poll(self):
                return self.returncode
        child = Child()
        self.assertTrue(T.owned_reap(child, time.monotonic() + 1, orderly=False))
        self.assertTrue(child.killed)
        self.assertIsNotNone(child.poll())

    def test_nonfinite_duplicate_and_bool_JSON(self):
        for index, text in enumerate(('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}')):
            path = self.root / str(index); path.write_text(text)
            with self.assertRaises(ValueError):
                T.json_file(path)

    def test_failure_receipt_requires_exact_PID_token_and_counter_truth(self):
        chunk = T.chunks(ORIGINAL['forward_slots'][61:])[0]
        folder = self.root / 'failed'; folder.mkdir()
        path = folder / 'failure.json'
        value = {'complete': True, 'token': 'a' * 32, 'chunk': 1, 'worker_pid': 1234,
                 'forward_outcome_unknown': False, 'counters': T.blank_counts(), 'model_qualified': False,
                 'automatic_retry_resume': False, 'phase': 'before_Torch', 'slot': None}
        path.write_text(json.dumps(value))
        self.assertIsNotNone(T.verified_failure(folder, chunk, 'a' * 32, 1234))
        for field, wrong in (('token', 'b' * 32), ('worker_pid', True), ('chunk', True), ('forward_outcome_unknown', True)):
            path.write_text(json.dumps({**value, field: wrong}))
            self.assertIsNone(T.verified_failure(folder, chunk, 'a' * 32, 1234))
        path.write_text('{"complete":true}')
        self.assertIsNone(T.verified_failure(folder, chunk, 'a' * 32, 1234))

    def test_shared_deadline_and_RAM_guard_before_spawn(self):
        out = self.root / 'out'; out.mkdir()
        writer = T.Writer(out, 1024 ** 2)
        chunk = T.chunks(ORIGINAL['forward_slots'][61:])[0]
        with patch.object(T, 'OUT', out), patch.object(T.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                T.run_chunk({'remaining_slots': ORIGINAL['forward_slots'][61:]}, 'a' * 64,
                            chunk, writer, time.monotonic() - 1, T.blank_counts())
            spawn.assert_not_called()
            import psutil
            with patch.object(psutil, 'virtual_memory', return_value=type('Memory', (), {'available': 1})()):
                with self.assertRaises(ValueError):
                    T.run_chunk({'remaining_slots': ORIGINAL['forward_slots'][61:]}, 'a' * 64,
                                chunk, writer, time.monotonic() + 5, T.blank_counts())
            spawn.assert_not_called()

    def test_actual_chunk_worker_preTorch_guard_requires2p5GiB(self):
        import psutil
        out = self.root / 'child-guard'; folder = out / 'chunks' / '001'; folder.mkdir(parents=True)
        token = 'a' * 32
        writer = T.Writer(folder, 1024 ** 2)
        writer.json(folder / 'startup.json', {'chunk': 1, 'token': token, 'plan_sha256': 'b' * 64,
                    'remaining_seconds': 10, 'deadline_monotonic': time.monotonic() + 10, 'byte_allowance': 1024 ** 2})
        writer.json(folder / 'command.json', {'chunk': 1, 'token': token, 'plan_sha256': 'b' * 64})
        with patch.object(T, 'OUT', out), patch.object(T, 'verify', return_value=(
                {'chunks': T.chunks(ORIGINAL['forward_slots'][61:])}, ORIGINAL, RETAINED)), \
             patch.object(psutil, 'virtual_memory', return_value=type('Memory', (), {'available': 2 * 1024 ** 3})()):
            with self.assertRaisesRegex(ValueError, 'resource cap'):
                T.chunk_worker('b' * 64, 1, token)
        failure = T.json_file(folder / 'failure.json')
        self.assertEqual(failure['counters']['attempts'], 0)
        self.assertEqual(failure['counters']['returns'], 0)
        self.assertFalse(failure['forward_outcome_unknown'])
        self.assertFalse((folder / 'slots').exists())
        self.assertNotIn('torch', sys.modules)

    def counter_for(self, slots):
        counts = T.blank_counts()
        for slot in slots:
            for key in ('attempts', 'returns', 'saved'):
                counts[key] += 1
                counts['per_arm'][slot['arm']][key] += 1
        return counts

    def slot_records(self, slots):
        artifact = self.fixture_file('artifact.bin')
        records = []
        for slot in slots:
            output = {key: artifact for key in T.O.OUTPUT_KEYS}
            output.update({'metrics': {'raw': {'mae': 0.}, 'native_png': {'mae': 0.}}, 'provenance': {'arm': slot['arm']}})
            inputs = {key: artifact for key in ('original', 'target', 'support')}
            record = {'slot': slot, 'inputs': inputs, **output}
            receipt = self.fixture_file('record-%d.json' % slot['new_output_index'], T.O.B.json_bytes(record))
            records.append({'slot': slot, 'inputs': inputs, 'record': receipt, 'output': output})
        return records

    def run_fixture_chunk(self, mode):
        out = self.root / ('out-' + mode); out.mkdir()
        fixture = self.root / ('worker-' + mode + '.py'); fixture.write_text(FIXTURE)
        chunk = T.chunks(ORIGINAL['forward_slots'][61:])[0]
        records = self.slot_records(chunk['slots'])
        payload = {'complete': True, 'clean_exit': True, 'chunk': 1, 'tail_plan_sha256': 'a' * 64,
                   'forward_outcome_unknown': False, 'counters': self.counter_for(chunk['slots']), 'records': records}
        payload_path = self.root / ('payload-' + mode + '.json'); payload_path.write_text(json.dumps(payload))
        def spawn(args, **kwargs):
            self.assertEqual(args[0], T.worker_python()[0]); self.assertFalse(kwargs['shell'])
            env = dict(kwargs.pop('env'), TAIL_TEST_FOLDER=str(out / 'chunks' / '001'), TAIL_TEST_MODE=mode,
                       TAIL_TEST_PAYLOAD=str(payload_path))
            process = REAL_POPEN([args[0], '-B', str(fixture)], env=env, **kwargs)
            self.processes.append(process)
            return process
        limits = {**T.LIMITS, 'cleanup_seconds': 1}
        import psutil
        with patch.object(T, 'OUT', out), patch.object(T, 'LIMITS', limits), patch.object(T.subprocess, 'Popen', side_effect=spawn), \
             patch.object(psutil, 'virtual_memory', return_value=type('Memory', (), {'available': 8 * 1024 ** 3})()), \
             patch.object(T.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 16 * 1024 ** 3})()):
            writer = T.Writer(out, T.LIMITS['maximum_new_output_bytes'])
            if mode == 'success':
                terminal = T.run_chunk({'remaining_slots': ORIGINAL['forward_slots'][61:]}, 'a' * 64,
                                      chunk, writer, time.monotonic() + 8, T.blank_counts())
                self.assertEqual(terminal['worker_pid'], self.processes[-1].pid)
                self.assertEqual(terminal['counters']['saved'], 16)
            else:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    T.run_chunk({'remaining_slots': ORIGINAL['forward_slots'][61:]}, 'a' * 64,
                                chunk, writer, time.monotonic() + (1 if mode == 'hang' else 8), T.blank_counts())
                stop = T.json_file(out / 'chunk_stop.json')
                self.assertTrue(stop['forward_outcome_unknown'])
                self.assertTrue(stop['actual_exit_confirmed'])
                self.assertEqual(stop['unobserved_attempt_upper_bound'], 16)
                if mode == 'identity':
                    self.assertFalse((out / 'chunks' / '001' / 'command.json').exists())
            self.assertIsNotNone(self.processes[-1].poll())
            import psutil
            self.assertFalse(psutil.pid_exists(self.processes[-1].pid))
            self.assertEqual(len(self.processes), 1)

    def test_real_direct_owned_child_clean_chunk(self):
        self.run_fixture_chunk('success')

    def test_real_direct_owned_child_identity_mismatch_no_command(self):
        self.run_fixture_chunk('identity')

    def test_real_direct_owned_child_timeout_no_retry(self):
        self.run_fixture_chunk('hang')

    def test_real_direct_owned_child_crash_unknown(self):
        self.run_fixture_chunk('crash')

    def test_real_direct_owned_child_bool_counter_refused(self):
        self.run_fixture_chunk('bool')

    def test_split_case_aggregate716_membership406_without_new_carried_calls(self):
        records = self.slot_records(ORIGINAL['forward_slots'])
        out = self.root / 'aggregate'; out.mkdir()
        metric_calls = []
        class Metrics:
            @staticmethod
            def preservation(bank, teachers):
                metric_calls.append(('preservation', bank, teachers)); return {'pass': False}
            @staticmethod
            def dark_retention(bank, teachers):
                metric_calls.append(('dark', bank, teachers)); return {'pass': True}
        with patch.object(T, 'OUT', out), patch.object(T, 'carried_records', return_value=records[:61]), \
             patch.object(T, 'metric_module', return_value=Metrics):
            value = T.aggregate(ORIGINAL, {}, records[61:], self.counter_for(ORIGINAL['forward_slots'][61:]),
                                T.Writer(out, T.LIMITS['maximum_new_output_bytes']), 'a' * 64)
        self.assertEqual(value['counters']['saved_outputs'], 716)
        self.assertEqual(value['new_tail_forward_returns'], 655)
        self.assertEqual(value['carried_paired_output_slots'], 61)
        self.assertEqual(len(value['cases']), 358)
        self.assertEqual(len(value['historical_membership406']), 406)
        self.assertEqual(len(value['planned_paired_visual_sheets']), 90)
        self.assertEqual(value['native_carried_output_slots'], 64)
        self.assertEqual(value['native_excluded_rows_zero_call'], 16)
        split = value['cases'][30]
        self.assertEqual(split['output_record_provenance'][T.O.ARMS[0]], records[60]['record'])
        self.assertEqual(split['output_record_provenance'][T.O.ARMS[1]], records[61]['record'])
        self.assertEqual(len(metric_calls), 4)
        self.assertEqual(value['quality_status'], 'QUALITY_STOP_NOT_QUALIFIED')
        self.assertFalse(value['model_qualified'])
        self.assertEqual(sum(row['status'] == 'EXCLUDED_INPUT_ONLY_ZERO_CALL' for row in value['historical_membership406']), 16)

    def test_exact_inference_delivery_expressions_and_historical_sum(self):
        old = ast.parse(Path(T.O.__file__).read_bytes()); new = ast.parse(Path(T.__file__).read_bytes())
        def expressions(tree):
            return {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.Call))}
        old_expressions, new_expressions = expressions(old), expressions(new)
        for expression in (
            'tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1).copy())[None] / 255',
            "sample = F.interpolate(tensor, (512, 512), mode='bilinear', align_corners=False) * 2 - 1",
            'models[arm](sample, w=1.0, detach_16=True, adain=True)',
            "unclipped = F.interpolate((native + 1) / 2, (256, 256), mode='bilinear', align_corners=False)",
            "raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)",
            'raw_array[~observed] = canvas.astype(np.float32)[~observed] / 255',
            'pixels = np.floor(raw_array * np.float32(255)).astype(np.uint8)',
            "M.metrics(raw_array, pixels, truth, observed, 'paired_synthetic')"):
            self.assertIn(expression, old_expressions)
            self.assertIn(expression, new_expressions)
        old_text = ast.unparse(old); new_text = ast.unparse(new)
        for required in ("isinstance(triple, tuple) and len(triple) == 3",
                         "value.shape == shape and value.dtype == torch.float32 and (value.device.type == 'cpu')",
                         "torch.isfinite(value).all().item()"):
            self.assertIn(required, old_text)
            self.assertIn(required, new_text)
        M = T.metric_module()
        self.assertIs(M.sum, T.O.ordered_sum)
        self.assertEqual(M.sum([1e16, 1., -1e16]), 0.)
        self.assertEqual(T.pin(Path(T.O.__file__))['sha256'], T.ORIGINAL_RUNNER_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
