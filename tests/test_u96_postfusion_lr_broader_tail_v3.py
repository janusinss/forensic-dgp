"""Pure metadata and owned fake-subprocess continuation tests; never import Torch."""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_u96_postfusion_lr_broader_tail_v3 as T

ORIGINAL = T.load(T.O.PLAN)  # Frozen metadata only.
TEST_ROOT = ROOT / 'scratch' / ('broader-tail511-source-tests-' + uuid.uuid4().hex)
REAL_POPEN = subprocess.Popen


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.folder = TEST_ROOT / uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        self.processes = []

    def tearDown(self):
        for child in self.processes:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        self.assertNotIn('torch', sys.modules)

    def test_exact_remaining511_order_counts_and_disjointness(self):
        slots = ORIGINAL['forward_slots'][205:]
        chunks = T.chunks(slots)
        self.assertEqual(len(chunks), 32)
        self.assertEqual([len(c['slots']) for c in chunks], [16] * 31 + [15])
        self.assertEqual([r for c in chunks for r in c['slots']], slots)
        self.assertEqual(Counter(r['arm'] for r in slots), Counter(T.LIMITS['per_arm']))
        self.assertEqual(slots[0]['new_output_index'], 206)
        self.assertEqual(slots[0]['arm'], T.O.ARMS[1])
        self.assertEqual(slots[-1]['new_output_index'], 716)
        self.assertFalse({r['new_output_index'] for r in slots} & set(range(1, 206)))

    def test_missing_duplicate_reordered_or_carried_slots_rejected(self):
        slots = ORIGINAL['forward_slots'][205:]
        for wrong in (slots[:-1], slots[1:] + slots[-1:], [slots[1], slots[0], *slots[2:]],
                      ORIGINAL['forward_slots'][204:715]):
            with self.assertRaises(ValueError):
                T.chunks(wrong)

    def test_shared_bytes_and_stricter_absolute_clock(self):
        self.assertEqual(T.LIMITS['maximum_new_output_bytes'] + 381083149 + 911108033 + T.LAUNCH_BYTES_RESERVE,
                         6 * 1024 ** 3)
        self.assertEqual(T.ABSOLUTE_DEADLINE, T.V2_ABSOLUTE_DEADLINE - 1.)
        self.assertLessEqual(T.ORIGINAL_SECONDS + T.LIMITS['global_seconds'], 14400)
        for key in ('startup_RAM_bytes', 'minimum_available_RAM_bytes', 'maximum_worker_RSS_bytes',
                    'minimum_free_disk_bytes', 'load_seconds', 'forward_seconds', 'CPU_threads', 'cleanup_seconds'):
            self.assertEqual(T.LIMITS[key], T.BASE_LIMITS[key])

    def clock_patches(self, now=None, boot=None, clock_impl='QueryPerformanceCounter()'):
        observation = {'complete': True, 'Windows_CIM_boot_epoch': T.BOOT_EPOCH,
                       'Windows_CIM_LastBootUpTime_UTC': T.BOOT_MARKER,
                       'original_absolute_deadline': T.V2_ABSOLUTE_DEADLINE,
                       'conservative_stricter_absolute_deadline': T.ABSOLUTE_DEADLINE,
                       'same_absolute_deadline_monotonic': T.V2_ABSOLUTE_DEADLINE,
                       'monotonic_seconds': 6409.5553091, 'prior_clock': {}}
        return (patch.object(T, 'windows_boot_marker', return_value=T.BOOT_MARKER if boot is None else str(boot)),
                patch.object(T.time, 'monotonic', return_value=6500. if now is None else now),
                patch.object(T.time, 'get_clock_info', return_value=SimpleNamespace(monotonic=True, adjustable=False,
                                                                                   implementation=clock_impl)),
                patch.object(T, 'bound', return_value=self.folder / 'clock.json'),
                patch.object(T, 'load', return_value=observation))

    def test_clock_uses_same_deadline_with_no_idle_refund(self):
        a, b, c, d, e = self.clock_patches(now=7000.)
        with a, b, c, d, e:
            self.assertEqual(T.clock_remaining(), T.ABSOLUTE_DEADLINE - 7000.)
        a, b, c, d, e = self.clock_patches(now=8000.)
        with a, b, c, d, e:
            self.assertEqual(T.clock_remaining(), T.ABSOLUTE_DEADLINE - 8000.)

    def test_expired_clock_reboot_other_clock_or_reversed_monotonic_rejected(self):
        for kwargs in ({'now': T.ABSOLUTE_DEADLINE}, {'boot': T.BOOT_EPOCH + 1},
                       {'clock_impl': 'otherclock'}, {'now': 1000.}):
            a, b, c, d, e = self.clock_patches(**kwargs)
            with a, b, c, d, e, self.assertRaises(ValueError):
                T.clock_remaining()

    def test_namespace_closes_empty_directories_and_no_new_files(self):
        old = self.folder / 'old'; (old / 'empty010').mkdir(parents=True)
        data = old / 'saved'; data.write_bytes(b'exact')
        files = [T.pin(data)]; dirs = [(old / 'empty010').relative_to(ROOT).as_posix()]
        T.namespace_equal(old, files, dirs)
        (old / 'new').write_bytes(b'new')
        with self.assertRaises(ValueError):
            T.namespace_equal(old, files, dirs)

    def test_changed_empty_directory_is_rejected(self):
        old = self.folder / 'old'; (old / 'empty010').mkdir(parents=True)
        dirs = [(old / 'empty010').relative_to(ROOT).as_posix()]
        T.namespace_equal(old, [], dirs)
        (old / 'empty010').rmdir()
        with self.assertRaises(ValueError):
            T.namespace_equal(old, [], dirs)

    def test_namespace_uses_identical_string_sort_for_mixed_case_names(self):
        old = self.folder / 'old'; old.mkdir()
        for name in ('input.png', 'LQ.npy'):
            (old / name).write_bytes(b'exact')
        rows = [T.pin(old / name) for name in ('LQ.npy', 'input.png')]
        T.namespace_equal(old, rows, [])

    def test_only_nine_exact_intentional_empty_logs_accepted(self):
        old = self.folder / 'v2'
        empty_sha = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        with patch.object(T, 'V2_OUT', old):
            for index in range(1, 10):
                path = old / 'chunks' / ('%03d' % index) / 'worker.log'
                path.parent.mkdir(parents=True); path.write_bytes(b'')
                row = {'path': path.relative_to(ROOT).as_posix(), 'bytes': 0, 'sha256': empty_sha}
                self.assertEqual(T.retained_file(row), path.resolve())
                for wrong in ({**row, 'sha256': '0'*64}, {**row, 'bytes': False}, {**row, 'extra': True}):
                    with self.assertRaises(ValueError): T.retained_file(wrong)
            other = old / 'chunks/010/worker.log'; other.parent.mkdir(); other.write_bytes(b'')
            with self.assertRaises(ValueError):
                T.retained_file({'path': other.relative_to(ROOT).as_posix(), 'bytes': 0, 'sha256': empty_sha})

    def test_nonempty_pins_still_use_exact_existing_binder(self):
        row = {'path': 'fixture', 'bytes': 4, 'sha256': 'a'*64}
        with patch.object(T, 'bound', return_value=self.folder) as binder:
            self.assertEqual(T.retained_file(row), self.folder)
            binder.assert_called_once_with(row)

    def test_named_zero_log_now_nonempty_is_refused(self):
        old = self.folder / 'v2'; path = old / 'chunks/001/worker.log'
        path.parent.mkdir(parents=True); path.write_bytes(b'changed')
        with patch.object(T, 'V2_OUT', old), self.assertRaises(ValueError):
            T.retained_file({'path': path.relative_to(ROOT).as_posix(), 'bytes': 0,
                'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'})

    def test_named_empty_log_hardlink_and_reparse_refused(self):
        old = self.folder / 'v2'; path = old / 'chunks/001/worker.log'
        path.parent.mkdir(parents=True); original = self.folder / 'empty_source'; original.write_bytes(b'')
        os.link(original, path)
        row = {'path': path.relative_to(ROOT).as_posix(), 'bytes': 0,
               'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
        with patch.object(T, 'V2_OUT', old), self.assertRaises(ValueError): T.retained_file(row)
        path.unlink(); path.write_bytes(b'')
        real_lstat = Path.lstat
        def reparse(this):
            value = real_lstat(this)
            if this == path:
                return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=1024)
            return value
        with patch.object(T, 'V2_OUT', old), patch.object(Path, 'lstat', reparse), self.assertRaises(ValueError):
            T.retained_file(row)

    def test_prepare_requires_independent205_before_plan(self):
        with patch.object(T, 'PLAN', self.folder / 'plan.json'), patch.object(T, 'pin', return_value={'sha256': 'a'*64}), \
             patch.object(T, 'dependencies', side_effect=ValueError('independent205 not ready')), \
             patch.object(T.subprocess if hasattr(T, 'subprocess') else T.V.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'independent205'):
                T.prepare(self.folder / 'review.json', self.folder / 'audit.json', 'a'*64)
            self.assertFalse((self.folder / 'plan.json').exists()); spawn.assert_not_called()

    def test_core_binding_preserves_exact_functions_and_worker_dispatch(self):
        originals = {key: getattr(T.V, key) for key in ('OUT', 'LIMITS', 'verify', '__file__')}
        worker, runner, models, metric, delivery = T.V.chunk_worker, T.V.run_chunk, T.V.frozen_models, T.V.metric_module, T.V.input_arrays
        try:
            T.bind_core()
            self.assertIs(T.V.chunk_worker, worker); self.assertIs(T.V.run_chunk, runner)
            self.assertIs(T.V.frozen_models, models); self.assertIs(T.V.metric_module, metric); self.assertIs(T.V.input_arrays, delivery)
            self.assertEqual(T.V.__file__, str(T.SOURCE)); self.assertIs(T.V.verify, T.verify)
            self.assertEqual(T.V.OUT, T.OUT); self.assertEqual(T.V.LIMITS, T.LIMITS)
        finally:
            for key, value in originals.items():
                setattr(T.V, key, value)

    def test_launch_requires_actual_pid_root_approval_before_body(self):
        token = uuid.uuid4().hex; prep = self.folder / 'prep'; errors = []
        def approve():
            path = prep / 'runtime_launch' / token / 'identity.json'
            end = time.monotonic() + 3
            while not path.exists() and time.monotonic() < end:
                time.sleep(.01)
            try:
                identity = json.loads(path.read_text())
                approval = {'token': token, 'supervisor_pid': os.getpid(), 'plan_sha256': 'a'*64,
                            'actual_process_identity_verified': True}
                self.assertEqual(identity['supervisor_pid'], os.getpid())
                self.assertEqual(identity['new_neural_calls'], 0)
                (path.parent / 'root_approval.json').write_text(json.dumps(approval))
            except BaseException as error:
                errors.append(error)
        with patch.object(T, 'PREP', prep), patch.object(T, 'clock_remaining', return_value=100):
            thread = threading.Thread(target=approve); thread.start()
            proof = T.launch_handshake('a'*64, token); thread.join(3)
            self.assertFalse(thread.is_alive()); self.assertFalse(errors)
            self.assertEqual(proof['reserved_bytes'], 16384)
            with self.assertRaises(ValueError):
                T.launch_handshake('a'*64, token)

    def test_wrong_actual_pid_approval_rejected(self):
        token = uuid.uuid4().hex; prep = self.folder / 'prep'
        def wrong():
            folder = prep / 'runtime_launch' / token
            while not (folder / 'identity.json').exists():
                time.sleep(.01)
            (folder / 'root_approval.json').write_text(json.dumps({'token': token, 'supervisor_pid': os.getpid()+1,
                'plan_sha256': 'a'*64, 'actual_process_identity_verified': True}))
        with patch.object(T, 'PREP', prep), patch.object(T, 'clock_remaining', return_value=100):
            thread = threading.Thread(target=wrong); thread.start()
            with self.assertRaises(ValueError):
                T.launch_handshake('a'*64, token)
            thread.join(3); self.assertFalse(thread.is_alive())

    def test_global_deadline_covers_handshake_before_body(self):
        def hang(*unused):
            end = time.monotonic() + 1
            while time.monotonic() < end:
                pass
        start = time.monotonic()
        with patch.object(T, 'clock_remaining', return_value=.03), patch.object(T, 'launch_handshake', side_effect=hang), \
             patch.object(T, 'supervise_body') as body:
            with self.assertRaises(KeyboardInterrupt):
                T.supervise('a'*64, 'b'*32)
            body.assert_not_called()
        self.assertLess(time.monotonic()-start, .5)

    def test_aggregate_emits_only_correct511_lineage_once(self):
        carried = {'rows': [{'slot': r} for r in ORIGINAL['forward_slots'][:205]]}
        new = [{'slot': r} for r in ORIGINAL['forward_slots'][205:]]
        counts = T.V.blank_counts()
        for r in new:
            for k in ('attempts', 'returns', 'saved'):
                counts[k] += 1; counts['per_arm'][r['slot']['arm']][k] += 1
        class Sink:
            def __init__(self): self.rows = []
            def json(self, path, value): self.rows.append((path, value))
        def old_aggregate(original, retained, rows, merged_counts, writer, sha):
            self.assertEqual(len(rows), 655); self.assertEqual(merged_counts['saved'], 655)
            writer.json(T.OUT / 'comparison.json', {'counters': {'saved_outputs': 716},
                'quality_status': 'QUALITY_STOP_NOT_QUALIFIED', 'new_tail_forward_returns': 655,
                'paired_fresh_cases': 358,
                'carried_paired_output_slots': 61,
                'historical_membership406': [{'status': 'SOURCE_BOUND_ORIGINAL61_PLUS_TAIL655_PAIRED_OUTPUTS'}]})
        sink = Sink()
        with patch.object(T.V, 'aggregate', side_effect=old_aggregate), \
             patch.object(T, 'load', return_value={'prerequisites': {'partial205_audit': {'sha256': 'b'*64}}}):
            result = T.aggregate(ORIGINAL, {}, carried, new, counts, sink, 'a'*64)
        self.assertEqual(len(sink.rows), 1)
        self.assertEqual(result['new_tail_forward_returns'], 511)
        self.assertEqual(result['carried_paired_output_slots'], 205)
        self.assertEqual(result['carried_v2_slots'], 144)
        self.assertNotIn('paired_fresh_cases', result)
        self.assertEqual(result['paired_cases'], 358)
        self.assertEqual(result['quality_status'], 'QUALITY_STOP_NOT_QUALIFIED')
        self.assertEqual(result['historical_membership406'][0]['status'],
                         'SOURCE_BOUND_ORIGINAL61_PLUS_V2_144_PLUS_V3_511_PAIRED_OUTPUTS')

    def test_aggregate_failure_emits_no_comparison(self):
        class Sink:
            def json(self, *unused):
                self.fail('No aggregate comparison should be emitted')
        counts = T.V.blank_counts()
        with self.assertRaises(ValueError):
            T.aggregate(ORIGINAL, {}, {'rows': []}, [], counts, Sink(), 'a'*64)

    def test_changed_core_source_rejected_before_binding(self):
        prior = T.V.OUT
        with patch.object(T, 'pin', return_value={'sha256': '0'*64}), self.assertRaises(ValueError):
            T.bind_core()
        self.assertEqual(T.V.OUT, prior)

    def test_wrong_bool_or_extra_launch_files_rejected(self):
        for mode in ('bool', 'extra'):
            token = uuid.uuid4().hex; prep = self.folder / mode
            def approve():
                folder = prep / 'runtime_launch' / token
                while not (folder / 'identity.json').exists():
                    time.sleep(.01)
                approval = {'token': token, 'supervisor_pid': True if mode == 'bool' else os.getpid(),
                            'plan_sha256': 'a'*64, 'actual_process_identity_verified': True}
                if mode == 'extra':
                    (folder / 'unexpected').write_text('no')
                (folder / 'root_approval.json').write_text(json.dumps(approval))
            with patch.object(T, 'PREP', prep), patch.object(T, 'clock_remaining', return_value=100):
                thread = threading.Thread(target=approve); thread.start()
                with self.assertRaises(ValueError):
                    T.launch_handshake('a'*64, token)
                thread.join(3); self.assertFalse(thread.is_alive())

    def test_actual_owned_fake_worker_uses_v3_entrypoint_and_exits(self):
        import psutil
        out = self.folder / 'out'; out.mkdir()
        slots = ORIGINAL['forward_slots'][205:221]
        payload = {'complete': True, 'clean_exit': True, 'chunk': 1, 'tail_plan_sha256': 'a'*64,
                   'forward_outcome_unknown': False, 'counters': T.V.blank_counts(), 'records': []}
        dummy = self.folder / 'dummy'; dummy.write_bytes(b'not a model')
        artifact = T.pin(dummy)
        for i, slot in enumerate(slots):
            inputs = {'original': artifact, 'target': artifact, 'support': artifact}
            output = {**{k: artifact for k in T.O.OUTPUT_KEYS}, 'metrics': {'fixture': True}, 'provenance': {'fixture': True}}
            record = self.folder / ('record%d.json' % i)
            record.write_bytes(T.O.B.json_bytes({'slot': slot, 'inputs': inputs, **output}))
            payload['records'].append({'slot': slot, 'record': T.pin(record), 'inputs': inputs, 'output': output})
            for key in ('attempts', 'returns', 'saved'):
                payload['counters'][key] += 1; payload['counters']['per_arm'][slot['arm']][key] += 1
        payload_path = self.folder / 'payload.json'; payload_path.write_bytes(T.O.B.json_bytes(payload))
        code = """import json,os,sys,time
from pathlib import Path
folder=Path(os.environ['FINITE_FAKE_FOLDER']);start=json.loads((folder/'startup.json').read_text())
def write(name,row):
 p=folder/(name+'.new');p.write_text(json.dumps(row));os.replace(p,folder/name)
write('identity.json',{'token':start['token'],'chunk':1,'worker_pid':os.getpid(),'python_prefix':sys.prefix,'python_executable':sys.executable})
while not (folder/'command.json').exists():time.sleep(.01)
row=json.loads(Path(os.environ['FINITE_FAKE_PAYLOAD']).read_text());row['token']=start['token'];row['worker_pid']=os.getpid()
write('terminal.json',row)
"""
        def spawn(command, **kwargs):
            self.assertEqual(command[2], str(T.SOURCE)); self.assertIn('--worker', command)
            environment = dict(kwargs['env']); environment.update(FINITE_FAKE_FOLDER=str(out / 'chunks/001'),
                                                                  FINITE_FAKE_PAYLOAD=str(payload_path))
            kwargs['env'] = environment
            child = REAL_POPEN([command[0], '-B', '-c', code], **kwargs)
            self.processes.append(child); return child
        with patch.object(T.V, 'OUT', out), patch.object(T.V, 'LIMITS', T.LIMITS), \
             patch.object(T.V, '__file__', str(T.SOURCE)), patch.object(T.V.subprocess, 'Popen', side_effect=spawn), \
             patch.object(psutil, 'virtual_memory', return_value=SimpleNamespace(available=8*1024**3)):
            writer = T.V.Writer(out, T.LIMITS['maximum_new_output_bytes'])
            terminal = T.V.run_chunk({'remaining_slots': ORIGINAL['forward_slots'][205:]}, 'a'*64,
                                    {'chunk': 1, 'slots': slots}, writer, time.monotonic()+15, T.V.blank_counts())
        self.assertEqual(terminal['counters']['saved'], 16)
        self.assertEqual(self.processes[0].poll(), 0)
        self.assertTrue((out / 'chunks/001/command.json').exists())


if __name__ == '__main__':
    unittest.main()
