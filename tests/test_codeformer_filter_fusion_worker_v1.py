"""Finite orchestration/namespace fixtures only; no Torch, NN or VM execution."""
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import worker as w
from research.codeformer_filter_fusion_training_v1 import transport as t


def plan():
    return {'vm_name': t.VM_NAME, 'outputs': t.OUTPUTS, 'call_caps': w.call_caps(),
        'evaluation_path': 'evaluation_bundle.json',
        'data': {'bundle_path': 'data_bundle.json', 'schedule_path': 'schedule.json',
                 'admission_path': 'admission.json', 'dark_review_path': 'provenance/dark_review.json',
                 'holds_sha256': 'a'*64, 'dark_review_sha256': 'b'*64},
        'guard_cases': [{'role': role, 'id': role+str(index)}
                        for role in ('clear', 'dark', 'filter') for index in range(8)],
        'witness_cases': {key: key for key in ('clear', 'dark', 'rows', 'local_mosaic', 'local_blur')},
        'snapshots': [0, 50, 151], 'guard_snapshots': [0, 20],
        'gradient_queries': list(w.GRADIENT_QUERY_NAMES), 'automatic_resume': False,
        'automatic_training_launch': False, 'allowed_output_names': [t.OUTPUTS + '/example.json']}


class WorkerContracts(unittest.TestCase):
    def test_import_has_no_neural_framework(self):
        code = "import sys;sys.path.insert(0,%r);from research.codeformer_filter_fusion_training_v1 import worker;assert 'torch' not in sys.modules" % str(ROOT)
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_operation_census_not_free_diagnostic_calls(self):
        caps = w.call_caps()
        self.assertEqual(sum(value for key,value in caps.items() if key.endswith('_forwards')), 3639)
        self.assertEqual(caps['gradient_queries'], 7)
        self.assertEqual(caps['backward_calls'], 604)
        self.assertEqual(caps['optimizer_step_attempts'], 151)
        self.assertEqual(sum(caps.values()), 4401)
        self.assertEqual(len(w.GRADIENT_QUERY_NAMES), len(set(w.GRADIENT_QUERY_NAMES)))

    def test_full_plan_admitted_no_smaller_passing_substitute(self):
        self.assertEqual(w.validate_plan(plan())['total_top_level_forwards'], 3639)
        for change in ('short_epochs', 'fewer_calls', 'resume', 'remove_guard', 'new_query'):
            p = plan()
            if change == 'short_epochs': p['snapshots'] = [0, 50]
            if change == 'fewer_calls': p['call_caps']['candidate_training_forwards'] = 4
            if change == 'resume': p['automatic_resume'] = True
            if change == 'remove_guard': p['guard_cases'].pop()
            if change == 'new_query': p['gradient_queries'].append('extra_query')
            with self.subTest(change=change), self.assertRaises(ValueError):
                w.validate_plan(p)

    def test_output_traversal_rejected_before_runtime(self):
        p = plan(); p['data']['bundle_path'] = '../outside.json'
        with self.assertRaises(ValueError):
            w.validate_plan(p)

    def test_reservation_is_recorded_before_return_and_cap_preserved(self):
        rows = []; counters = w.Counters(w.call_caps(), rows.append)
        reservation = counters.reserve('parity_candidate_forwards', id='fixed')
        self.assertEqual(rows[0]['event'], 'reserved')
        self.assertEqual(counters.values['parity_candidate_forwards'], 1)
        counters.returned(reservation, finite=True)
        self.assertEqual(rows[1]['event'], 'returned')
        with self.assertRaises(ValueError):
            counters.reserve('parity_candidate_forwards')
        self.assertEqual(counters.values['parity_candidate_forwards'], 1)

    def test_failed_journal_cannot_execute_a_reserved_operation(self):
        def fail(row): raise OSError('inert forced journal failure')
        counters = w.Counters(w.call_caps(), fail)
        with self.assertRaises(OSError):
            counters.reserve('gradient_queries')
        self.assertEqual(counters.values['gradient_queries'], 1)

    def test_unknown_counter_is_not_contingency(self):
        counters = w.Counters(w.call_caps(), lambda row: None)
        with self.assertRaises(ValueError):
            counters.reserve('one_more_forward')

    def test_process_guard_detects_module_and_training_entrypoints_before_GPU_context(self):
        for command in ('python -m research.other.worker --run', '/some/.venv/bin/python training.py',
                        'python scripts/train_cctv.py', 'python main.py --epochs 10',
                        'bash scripts/run_training.sh'):
            with self.subTest(command=command):
                self.assertTrue(w.competing_process(command)['competing'])
        for command in ('python -m research.other.worker --verify-transfer', 'python worker.py --export',
                        'python scripts/audit_saved.py', 'tmux new-session -A -s training', 'bash'):
            with self.subTest(command=command):
                self.assertFalse(w.competing_process(command)['competing'])

    def writer(self):
        scratch = (ROOT/'scratch').resolve(); scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='filter_fusion_worker_', dir=scratch)
        self.addCleanup(temp.cleanup); root = Path(temp.name)
        self.assertTrue(root.resolve().is_relative_to(scratch))
        (root/'source.txt').write_bytes(b'pinned source')
        p = {'allowed_output_names': [t.OUTPUTS+'/one.json', t.OUTPUTS+'/operations.jsonl']}
        return root, w.Writer(root, p)

    def test_writer_is_exclusive_and_exact_namespace(self):
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*t.GiB)):
            root, writer = self.writer()
            result = writer.json(t.OUTPUTS+'/one.json', {'finite': 1})
            self.assertEqual(json.loads((root/result['path']).read_bytes()), {'finite': 1})
            before = (writer.bytes, writer.files)
            with self.assertRaises(FileExistsError):
                writer.json(t.OUTPUTS+'/one.json', {'changed': 2})
            self.assertEqual(before, (writer.bytes, writer.files))
            with self.assertRaises(ValueError):
                writer.json(t.OUTPUTS+'/arbitrary.json', {})

    def test_append_is_durable_finite_and_physical_count_once(self):
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*t.GiB)):
            root, writer = self.writer(); initial = writer.files
            writer.append(t.OUTPUTS+'/operations.jsonl', {'sequence': 1})
            writer.append(t.OUTPUTS+'/operations.jsonl', {'sequence': 2})
            self.assertEqual(writer.files, initial+1)
            self.assertEqual([json.loads(row)['sequence'] for row in
                (root/t.OUTPUTS/'operations.jsonl').read_bytes().splitlines()], [1, 2])

    def test_disk_reserve_prevents_write_before_creation(self):
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*t.GiB)):
            root, writer = self.writer()
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=t.LIMITS['reserve_free_bytes'])):
            with self.assertRaises(ValueError):
                writer.json(t.OUTPUTS+'/one.json', {'a': 1})
        self.assertFalse((root/t.OUTPUTS/'one.json').exists())

    def test_static_neural_and_optimizer_import_is_guarded(self):
        path = ROOT/'research/codeformer_filter_fusion_training_v1/worker.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == 'torch' for alias in node.names)
                             for node in top_imports))
        source = path.read_text(encoding='utf-8')
        functions = {node.name: ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertLess(functions['run'].index('_local_metadata(root, protocol)'), functions['run'].index('m.activate_runtime'))
        self.assertLess(functions['run'].index("os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'"),
                        functions['run'].index('m.activate_runtime'))
        self.assertIn("require('torch' not in sys.modules", functions['run'])
        self.assertLess(functions['run'].index('_witness('), functions['run'].index('torch.optim.Adam('))
        self.assertIn('len(optimizer.state) == 0', functions['run'])
        self.assertIn('automatic_resume_safe', functions['checkpoint_metadata'])
        self.assertIn('gradient_buffers_not_serialized', functions['checkpoint_metadata'])

    def test_static_whole_slot_backwards_and_optimizer_commit_order(self):
        source = (ROOT/'research/codeformer_filter_fusion_training_v1/worker.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        step = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_train_step')
        text = ast.get_source_segment(source, step)
        self.assertLess(text.index("counter.reserve('backward_calls'"), text.index('divided.backward()'))
        self.assertLess(text.index("counter.reserve('optimizer_step_attempts'"), text.index('optimizer.step()'))
        self.assertLess(text.index('optimizer.step()'), text.index("update['sampler'].commit(committed)"))
        self.assertIn("progress['optimizer_inflight'] = True", text)
        self.assertNotIn('clip_grad', text)
        self.assertNotIn('autocast', text)

    def test_final_quality_failure_reuses_only_actual_final_physical_state(self):
        progress = {'completed_training_updates': 151, 'partial_microbatches': 0,
                    'optimizer_inflight': False, 'state_checkpoint_complete': True,
                    'state_checkpoint': {'path': t.OUTPUTS+'/snapshot_0151/checkpoint.pt'}}
        self.assertFalse(w.needs_stopped_checkpoint(progress))
        for change in ('earlier', 'partial', 'inflight', 'not_complete'):
            value = deepcopy(progress)
            if change == 'earlier': value['state_checkpoint']['path'] = t.OUTPUTS+'/snapshot_0050/checkpoint.pt'
            if change == 'partial': value['partial_microbatches'] = 1
            if change == 'inflight': value['optimizer_inflight'] = True
            if change == 'not_complete': value['state_checkpoint_complete'] = False
            with self.subTest(change=change):
                self.assertTrue(w.needs_stopped_checkpoint(value))

    def test_interrupted_physical_temporary_is_retained_without_second_serialization(self):
        progress={'completed_training_updates':50,'partial_microbatches':0,'optimizer_inflight':False,
            'state_checkpoint_complete':False,'checkpoint_incomplete':True,'automatic_resume_safe':False,
            'state_checkpoint':{'path':t.OUTPUTS+'/snapshot_0050/checkpoint.pt.tmp-'+'a'*32}}
        self.assertFalse(w.needs_stopped_checkpoint(progress))
        progress['state_checkpoint_complete']=True
        self.assertTrue(w.needs_stopped_checkpoint(progress))

    def test_checkpoint_metadata_keeps_actual_reserved_counts_and_partial_status(self):
        progress = {'completed_training_updates': 3, 'partial_microbatches': 2, 'optimizer_inflight': False}
        counts = dict.fromkeys(w.call_caps(), 0)
        counts['backward_calls'] = 14; counts['optimizer_step_attempts'] = 3; counts['gradient_queries'] = 7
        value = w.checkpoint_metadata('a'*64, progress, counts)
        self.assertEqual(value['counts'], counts)
        self.assertEqual(value['progress'], progress)
        self.assertFalse(value['automatic_resume_safe'])
        self.assertTrue(value['gradient_buffers_not_serialized'])
        counts['gradient_queries'] = 8
        self.assertEqual(value['counts']['gradient_queries'], 7)
        with self.assertRaises(ValueError):
            w.checkpoint_metadata('a'*64, progress, counts)

    def test_checkpoint_stream_rejects_cap_before_excess_bytes_written(self):
        import io
        target = io.BytesIO(); writer = SimpleNamespace(root=ROOT, bytes=0)
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*t.GiB)):
            bounded = w.BoundedCheckpointStream(target, writer, 5)
            self.assertEqual(bounded.write(b'abc'), 3)
            with self.assertRaises(ValueError):
                bounded.write(b'def')
            self.assertEqual(target.getvalue(), b'abc')
            self.assertEqual(bounded.tell(), 3)

    def test_checkpoint_stream_refuses_disk_crossing_and_short_write(self):
        import io
        writer = SimpleNamespace(root=ROOT, bytes=0); target = io.BytesIO()
        bounded = w.BoundedCheckpointStream(target, writer, 10)
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=t.LIMITS['reserve_free_bytes'])):
            with self.assertRaises(ValueError): bounded.write(b'a')
        self.assertEqual(target.getvalue(), b'')
        class Short:
            def write(self, payload): return len(payload)-1
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*t.GiB)):
            with self.assertRaises(ValueError): w.BoundedCheckpointStream(Short(), writer, 10).write(b'ab')

    def test_observed_canvas_restores_only_outside_pixels_exactly(self):
        import numpy as np
        raw=np.full((256,256,3),.8,np.float32);png=np.full_like(raw,204,dtype=np.uint8)
        input_rgb8=np.full_like(png,71); support=np.zeros((256,256),bool);support[20:200,40:220]=True
        r,p=w.observed_canvas(raw,png,input_rgb8,support,np)
        np.testing.assert_array_equal(r[support],raw[support]);np.testing.assert_array_equal(p[support],png[support])
        np.testing.assert_array_equal(r[~support],input_rgb8[~support].astype(np.float32)/np.float32(255))
        np.testing.assert_array_equal(p[~support],input_rgb8[~support])
        self.assertTrue(np.all(raw==.8));self.assertTrue(np.all(png==204))

    def test_planned_namespace_contains_all_full_snapshots_teacher_and_partial_state_slots(self):
        evaluation=[{'id':'case'+str(i)}for i in range(478)]
        rows={str(i):{'id':str(i),'profile':'clear' if i<55 else w.d.DARK_LINEAR_PROFILE if i<110 else 'gaussian_2p4'}for i in range(411)}
        names=w.planned_output_names(plan(),evaluation,rows)
        self.assertEqual(sum('/baseline/'in n and n.endswith('.npy')for n in names),1434)
        self.assertEqual(sum('/snapshot_'in n and n.endswith('.npy')for n in names),1434)
        self.assertEqual(sum('/teacher/'in n and n.endswith('_U.npy')for n in names),110)
        self.assertEqual(sum('/training/'in n and n.endswith('_commit.json')for n in names),151)
        self.assertIn(t.OUTPUTS+'/snapshot_0151/checkpoint.pt',names)
        self.assertIn(t.OUTPUTS+'/stopped_checkpoint.pt',names)
        self.assertIn(t.OUTPUTS+'/guard_0020/bank.json',names)
        self.assertEqual(len(names),len(set(names)))

    def test_all478_initial_delivered_outputs_must_match_R2(self):
        row={'outputs':{view:{'bytes':42,'sha256':'a'*64}for view in ('raw','png')}}
        bank={str(i):deepcopy(row)for i in range(478)}
        self.assertTrue(w.initial_output_parity(bank,deepcopy(bank))['pass'])
        for view in ('raw','png'):
            changed=deepcopy(bank);changed['477']['outputs'][view]['sha256']='b'*64
            with self.subTest(view=view),self.assertRaises(ValueError):w.initial_output_parity(bank,changed)

    def test_peak_RSS_limit_cannot_hide_a_released_intermediate_peak(self):
        events=[];writer=SimpleNamespace(append=lambda name,row:events.append(row))
        runtime=w.Runtime(ROOT,writer)
        with mock.patch.object(w,'rss_bytes',return_value=t.GiB),mock.patch.object(w,'peak_rss_bytes',return_value=9*t.GiB), \
             mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=100*t.GiB)):
            with self.assertRaises(ValueError):runtime.check('inert fixture')
        self.assertEqual(events[0]['RSS_bytes'],t.GiB)
        self.assertEqual(events[0]['peak_RSS_bytes'],9*t.GiB)


if __name__ == '__main__':
    unittest.main()
