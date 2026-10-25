"""Independent pure accounting/source/NumPy fixtures; never call worker.run.

Passing these tests establishes no actual neural query, GPU execution, model
quality, exact CUDA forward parity or successful manual VM diagnostic.
"""
import ast
import copy
import importlib.machinery
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

from research.codeformer_u64_png_surrogate_measurement_v1 import contract, worker as W

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(W.__file__)


class LedgerFixtures(unittest.TestCase):
    def test_72_known_forwards_256_queries_656_durable_events(self):
        rows = []
        ledger = W.OperationLedger(lambda row: rows.append(copy.deepcopy(row)))
        for category, count in W.CATEGORIES.items():
            for slot in range(count):
                token = ledger.reserve('forward', category=category, plan_slot=slot + 1)
                ledger.returned(token, synchronized_return=True, synthetic_only=True)
                if category == 'diagnostic_forwards':
                    for name in W.QUERIES:
                        token = ledger.reserve('gradient_query', query=name)
                        ledger.returned(token, synchronized_return=True, synthetic_only=True)
        self.assertEqual(ledger.counters, {'full_forward_attempts': 72, 'full_forward_returns': 72,
            'gradient_query_attempts': 256, 'gradient_query_returns': 256,
            'parity_forwards': 4, 'repeat_forwards': 4, 'diagnostic_forwards': 64})
        self.assertEqual(ledger.sequence, 328)
        self.assertEqual(len(rows), 656)
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])
        self.assertEqual(ledger.snapshot()['unknown_reservations'], [])
        for index in range(0, len(rows), 2):
            self.assertEqual(rows[index]['token'], rows[index + 1]['token'])
            self.assertTrue(rows[index]['kind'].endswith('_reserved'))
            self.assertTrue(rows[index + 1]['kind'].endswith('_returned'))

    def test_no_optimizer_parameter_application_backward_or_solver_operation(self):
        ledger = W.OperationLedger(lambda row: None)
        for kind in ('optimizer_construction', 'optimizer_update', 'application',
                     'parameter_application', 'backward', 'solver', 'unknown'):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    ledger.reserve(kind)
        self.assertEqual(ledger.counters, dict.fromkeys(W.COUNTER_KEYS, 0))

    def test_unknown_query_reservation_remains_until_durable_return(self):
        ledger = W.OperationLedger(lambda row: None)
        token = ledger.reserve('gradient_query', query='png_surrogate_mae')
        self.assertEqual(ledger.counters['gradient_query_attempts'], 1)
        self.assertEqual(ledger.counters['gradient_query_returns'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])
        with self.assertRaises(ValueError):
            ledger.reserve('forward', category='parity_forwards')
        ledger.returned(token, synchronized_return=True)
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])
        with self.assertRaises(ValueError):
            ledger.returned(token)

    def test_failed_durable_return_never_invents_completed_query(self):
        def persist(row):
            if row['kind'].endswith('_returned'):
                raise OSError('Synthetic durable-return failure')
        ledger = W.OperationLedger(persist)
        token = ledger.reserve('gradient_query', query='raw_mae')
        with self.assertRaises(OSError):
            ledger.returned(token)
        self.assertEqual(ledger.counters['gradient_query_returns'], 0)
        self.assertIn(token, ledger.pending)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_failed_durable_reservation_remains_conservatively_unknown(self):
        ledger = W.OperationLedger(lambda row: (_ for _ in ()).throw(OSError('Synthetic write failure')))
        with self.assertRaises(OSError):
            ledger.reserve('gradient_query', query='raw_adjacent')
        self.assertEqual(ledger.counters['gradient_query_attempts'], 1)
        self.assertEqual(ledger.counters['gradient_query_returns'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_category_total_caps_bad_tokens_and_reserved_metadata_refused(self):
        ledger = W.OperationLedger(lambda row: None)
        for _ in range(4):
            ledger.returned(ledger.reserve('forward', category='parity_forwards'))
        with self.assertRaises(ValueError):
            ledger.reserve('forward', category='parity_forwards')
        for _ in range(256):
            ledger.returned(ledger.reserve('gradient_query'))
        with self.assertRaises(ValueError):
            ledger.reserve('gradient_query')
        ledger = W.OperationLedger(lambda row: None)
        for metadata in ({'category': 'unknown'}, {}):
            with self.assertRaises(ValueError):
                ledger.reserve('forward', **metadata)
        for metadata in ({'token': 1}, {'counters': {}}):
            with self.assertRaises(ValueError):
                ledger.reserve('gradient_query', **metadata)
        token = ledger.reserve('gradient_query')
        for wrong in (True, '1', 0, token + 1):
            with self.assertRaises(ValueError):
                ledger.returned(wrong)
        with self.assertRaises(ValueError):
            ledger.returned(token, counters={})


class SourceBoundaryFixtures(unittest.TestCase):
    def test_fresh_worker_import_blocks_all_neural_frameworks(self):
        code = '''import importlib.abc, importlib, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path, target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','jax'}:
   raise RuntimeError('Neural framework forbidden:'+fullname)
  return None
sys.meta_path.insert(0,Block())
before={n for n in sys.modules if n.split('.')[0] in {'numpy','torch','tensorflow','jax'}}
w=importlib.import_module('research.codeformer_u64_png_surrogate_measurement_v1.worker')
after={n for n in sys.modules if n.split('.')[0] in {'numpy','torch','tensorflow','jax'}}
assert before==after
print('INERT_WORKER_IMPORT')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'INERT_WORKER_IMPORT')

    def test_actual_contract_plan_matches_worker_categories_and_queries(self):
        folder = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
        selection = contract.select_cases((folder / 'evidence/parent_protocol.json').read_bytes(),
                                          (folder / 'evidence/standard_protocol.json').read_bytes())
        plan = contract.operation_plan(selection)
        self.assertEqual(W.QUERIES, contract.QUERIES)
        self.assertEqual(len(plan), 72)
        self.assertEqual(sum(len(c['gradient_queries']) for c in plan), 256)
        for phase, category in W.PHASE_CATEGORY.items():
            self.assertEqual(sum(c['phase'] == phase for c in plan), W.CATEGORIES[category])
        self.assertEqual([c['id'] for c in plan[8:]], [r['id'] for r in selection['optimization']])

    def test_supervisor_requires_own_protocol_seal_and_actual_parent_pid(self):
        sha = 'a' * 64
        environment = {contract.ENV + 'SUPERVISED': '1', contract.ENV + 'PROTOCOL': sha,
                       contract.ENV + 'SUPERVISOR_PID': '42'}
        W.supervisor_context(sha, environ=environment, parent_pid=42)
        with self.assertRaises(ValueError):
            W.supervisor_context(sha, environ=environment, parent_pid=43)
        with self.assertRaises(ValueError):
            W.supervisor_context('b' * 64, environ=environment, parent_pid=42)
        for key in environment:
            bad = dict(environment)
            bad[key] = 'wrong'
            with self.assertRaises(ValueError):
                W.supervisor_context(sha, environ=bad, parent_pid=42)

    def test_regular_and_namespace_package_parent_import_paths(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as temporary:
            base = Path(temporary)
            parent, stage = base / 'parent', base / 'stage'
            (parent / 'research/historical').mkdir(parents=True)
            (stage / 'research/newpackage').mkdir(parents=True)
            (parent / 'research/historical/__init__.py').write_text('MARKER="parent"\n')
            (stage / 'research/newpackage/__init__.py').write_text('MARKER="stage"\n')
            class IterableNamespace:
                def __iter__(self):
                    return iter([str(stage / 'research')])
            for paths in ([str(stage / 'research')], IterableNamespace()):
                package, before = types.SimpleNamespace(__path__=paths), list(sys.path)
                try:
                    W.install_parent_import_paths(parent, package)
                    historical = importlib.machinery.PathFinder.find_spec('research.historical', package.__path__)
                    new = importlib.machinery.PathFinder.find_spec('research.newpackage', package.__path__)
                    self.assertEqual(Path(historical.origin), parent / 'research/historical/__init__.py')
                    self.assertEqual(Path(new.origin), stage / 'research/newpackage/__init__.py')
                    self.assertEqual(sys.path[0], str(parent))
                finally:
                    sys.path[:] = before

    def test_no_top_level_run_and_all_neural_calls_owned_by_guarded_run(self):
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        def within_run(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.FunctionDef) and node.name == 'run':
                    return True
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.startswith(('torch.', 'candidate.', 'model.')) or name == 'CodeFormer':
                    self.assertTrue(within_run(node), name)
            if isinstance(node, ast.Import):
                if any(a.name.split('.')[0] in ('torch', 'numpy') for a in node.names):
                    self.assertTrue(within_run(node))
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in ('torch', 'numpy'):
                self.assertTrue(within_run(node))
        top_calls = [n for top in tree.body if not isinstance(top, (ast.FunctionDef, ast.ClassDef))
                     for n in ast.walk(top) if isinstance(n, ast.Call)]
        self.assertFalse(top_calls)

    def test_four_queries_keep_graph_only_until_last_query_no_solver_or_apply(self):
        tree = ast.parse(SOURCE.read_bytes())
        attributes = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        for forbidden in ('optim', 'backward', 'step', 'copy_', 'solve_common_descent', 'solve_eight_pooled_gates'):
            self.assertNotIn(forbidden, attributes)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'torch.autograd.grad']
        self.assertEqual(len(calls), 1)
        keywords = {k.arg: k.value for k in calls[0].keywords}
        self.assertEqual(ast.unparse(keywords['retain_graph']), 'index < 3')
        self.assertIs(keywords['create_graph'].value, False)
        self.assertIs(keywords['allow_unused'].value, True)
        self.assertIs(keywords['materialize_grads'].value, False)
        self.assertEqual(W.QUERIES, ('raw_mae', 'raw_adjacent', 'png_surrogate_mae', 'png_surrogate_adjacent'))

    def test_saved_PNG_CPU_decoded_reference_and_exact_custom_forward_binding(self):
        source = SOURCE.read_text()
        for fragment in ("with Image.open(checked(root,outputs['png'])) as saved:",
            'decoded_RGB8=np.asarray(saved).copy()',
            'decoded_float=decoded_RGB8.astype(np.float32)/np.float32(255)',
            'PNGSurrogate.apply(delivered,reference)', 'torch.equal(png_values,reference)',
            'actual_png_float.tobytes()==decoded_float.tobytes()',
            "'true_delivered_PNG_derivative':False", "'PNG_biased_surrogate_gradient_only':True"):
            self.assertIn(fragment, source)
        self.assertLess(source.index('decoded_RGB8=np.asarray(saved).copy()'),
                        source.index('PNGSurrogate.apply(delivered,reference)'))
        self.assertLess(source.index('actual_png_float.tobytes()==decoded_float.tobytes()'),
                        source.index('for index,name in enumerate(QUERIES):'))

    def test_three_from_numpy_calls_use_contiguous_CHW_inputs(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        transpose = image.transpose(2, 0, 1)
        contiguous = transpose.copy()
        self.assertFalse(transpose.flags.c_contiguous)
        self.assertTrue(contiguous.flags.c_contiguous)
        self.assertEqual(contiguous.strides, (65536, 256, 1))
        tree = ast.parse(SOURCE.read_bytes())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'torch.from_numpy']
        self.assertEqual(len(calls), 3)
        for node in calls:
            self.assertIn('.transpose(2, 0, 1).copy()', ast.unparse(node.args[0]))
        decoded = image.astype(np.float32) / np.float32(255)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(decoded.transpose(2, 0, 1).copy().flags.c_contiguous)

    def test_existing_FP32_floor_renderer_fixture_differs_from_nearest_rounding(self):
        raw = np.array([0., .5, 1.], dtype=np.float32)
        actual = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
        nearest = np.rint(np.clip(raw, 0, 1) * 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, [0, 127, 255])
        np.testing.assert_array_equal(nearest, [0, 128, 255])
        decoded = actual.astype(np.float32) / np.float32(255)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertIn('quantized=(np.clip(raw,0,1)*255).astype(np.uint8)', SOURCE.read_text())

    def test_initial_closing575_exact_state_RNG_and_preframework_guards(self):
        source = SOURCE.read_text()
        for fragment in ('initial[n].tobytes()', "'all575_exact_initial_values':True",
                         "'all539_frozen_exact':True", "'gradient_fields_all_None':True",
                         "full_state('initial')", "full_state('closing')",
                         "need(rng_record()==run_rng", "'optimizer':None",
                         "'parameter_applications':0", "'optimizer_constructions':0",
                         'full_states_attempted<2', "'unused_names_explicit_zero':unused"):
            self.assertIn(fragment, source)
        framework = source.index('import torch as torch_module')
        for fragment in ('runtime.verify_static(root,parent,expected_sha)',
                         'runtime.require_vm(root,parent)',
                         "contract.LIMITS['initial_free_disk_bytes']",
                         'supervisor_context(expected_sha,environ=os.environ,parent_pid=os.getppid())'):
            self.assertLess(source.index(fragment), framework)


if __name__ == '__main__':
    unittest.main()
