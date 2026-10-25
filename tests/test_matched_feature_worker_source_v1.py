"""Independent worker AST and extracted pure-controller tests; never import worker.

Mock graph metadata, scalar norm arrays and clocks are not neural/runtime proof.
Only selected pure function bodies are compiled from the inspected source AST.
"""
import ast
import copy
from contextlib import contextmanager
from hashlib import sha256
import importlib.abc
import math
from pathlib import Path
import re
from types import SimpleNamespace
import sys
import unittest


class ForbiddenImportBlocker(importlib.abc.MetaPathFinder):
    attempted = []
    roots = {'torch', 'numpy', 'PIL', 'tensorflow', 'jax', 'jaxlib', 'keras'}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.roots or fullname.endswith(('.worker', '.model')):
            self.attempted.append(fullname)
            raise AssertionError('Independent source tests forbid neural/worker/model import: ' + fullname)
        return None


BLOCKER = ForbiddenImportBlocker()
sys.meta_path.insert(0, BLOCKER)
from research.codeformer_matched_feature_learning_v1 import contract as C
from research.codeformer_matched_feature_learning_v1 import execution as E

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'research/codeformer_matched_feature_learning_v1/worker.py'
SOURCE = WORKER.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE, feature_version=(3, 10))
RUN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
SITES = ('32', '64', '128', '256')


def function(name):
    return next(node for node in ast.walk(TREE) if isinstance(node, ast.FunctionDef) and node.name == name)


def extracted(name, **globals_):
    node = copy.deepcopy(function(name))
    namespace = {'math': math, 're': re, 'copy': copy, 'contextmanager': contextmanager}
    namespace.update(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER) + ':' + name, 'exec'), namespace)
    return namespace[name]


NEED = extracted('need')


class FakeScalarArray:
    """Small scalar-list norm fixture; no array framework or tensor operations."""
    def __init__(self, values):
        self.values = values
    def astype(self, dtype):
        return self
    def sum(self):
        return sum(self.values)


class WorkerSourceReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prep = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
        cls.selection = C.select_cases((prep / 'data_admission_v1.json').read_bytes(),
            (prep / 'candidate_data_v1.json').read_bytes(),
            (ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/evidence/u64_parent_protocol.json').read_bytes())
        cls.plan = C.operation_plan(cls.selection)

    def test_supervisor_metadata_requires_exact_parent_protocol_and_explicit_flag(self):
        check = extracted('supervisor_context', need=NEED)
        digest = 'a' * 64
        good = {'FORENSIC_MATCHED_FEATURE_SUPERVISED': '1',
                'FORENSIC_MATCHED_FEATURE_PROTOCOL': digest,
                'FORENSIC_MATCHED_FEATURE_SUPERVISOR_PID': '72'}
        self.assertIsNone(check(digest, environ=good, parent_pid=72))
        for name, value in [('FORENSIC_MATCHED_FEATURE_SUPERVISED', '0'),
                            ('FORENSIC_MATCHED_FEATURE_PROTOCOL', 'b' * 64),
                            ('FORENSIC_MATCHED_FEATURE_SUPERVISOR_PID', '73')]:
            changed = dict(good); changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                check(digest, environ=changed, parent_pid=72)
        for bad in ('A' * 64, 'a' * 63, None, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check(bad, environ=good, parent_pid=72)

    def test_arithmetic_close_is_finite_and_does_not_accept_bool(self):
        close = extracted('arithmetic_close', ARITHMETIC_ATOL=2e-7, ARITHMETIC_RTOL=2e-6)
        self.assertTrue(close(1., 1.))
        self.assertTrue(close(1. + 1e-6, 1.))
        self.assertFalse(close(1. + 3e-6, 1.))
        for bad in (True, float('nan'), float('inf'), None, '1'):
            self.assertFalse(close(bad, 1.))
            self.assertFalse(close(1., bad))

    def test_parity_checks_stock_RAW_PNG_unclipped_input_and_target_metadata(self):
        equal = extracted('parity_equal')
        row = {'stock': {'native512': 'n', 'logits': 'l', 'LQ': 'q'},
               'RAW_pixel_bytes_sha256': 'r', 'PNG_RGB8_sha256': 'p',
               'outputs': {'unclipped': {'sha256': 'u'}},
               'input': {'sha256': 'i'}, 'target': {'path': 'target', 'sha256': 't'}}
        self.assertTrue(equal(row, copy.deepcopy(row)))
        for change in (lambda v: v['stock'].update(logits='changed'),
                       lambda v: v.update(RAW_pixel_bytes_sha256='changed'),
                       lambda v: v.update(PNG_RGB8_sha256='changed'),
                       lambda v: v['outputs']['unclipped'].update(sha256='changed'),
                       lambda v: v['input'].update(sha256='changed'),
                       lambda v: v['target'].update(sha256='changed')):
            changed = copy.deepcopy(row); change(changed)
            self.assertFalse(equal(row, changed))

    @staticmethod
    def graph_metadata(arm):
        return {'status': 'WITNESSED', 'complete': True, 'grad_enabled': True, 'arm': arm,
                'sites': {site: {branch: {'pass': True, 'expected_present': arm == 'B',
                    'current_adapter_leaves_reached': 6 if arm == 'B' else 0,
                    'autograd_queries_or_backward_calls': 0}
                    for branch in ('stock_SFT', 'head_encoder')} for site in SITES}}

    def graph_checker(self, arm, data):
        return extracted('graph_guard', need=NEED, candidates={arm: SimpleNamespace(last_graph_witness=data)},
                         model=SimpleNamespace(SITES=SITES))

    def test_graph_guard_requires_all_eight_arm_specific_current_adapter_receipts(self):
        for arm in ('A', 'B'):
            data = self.graph_metadata(arm)
            result = self.graph_checker(arm, data)(arm)
            self.assertEqual(result, data)
            result['sites']['32']['stock_SFT']['pass'] = False
            self.assertTrue(data['sites']['32']['stock_SFT']['pass'])

    def test_graph_guard_refuses_unwitnessed_wrong_arm_missing_branch_or_partial_B(self):
        changes = [lambda d: d.update(status='UNWITNESSED_NO_GRAD', grad_enabled=False),
                   lambda d: d.update(complete=False), lambda d: d.update(arm='A'),
                   lambda d: d['sites'].pop('256'),
                   lambda d: d['sites']['64'].pop('head_encoder'),
                   lambda d: d['sites']['32']['stock_SFT'].update(current_adapter_leaves_reached=5),
                   lambda d: d['sites']['128']['head_encoder'].update(autograd_queries_or_backward_calls=1),
                   lambda d: d['sites']['256']['stock_SFT'].update(expected_present=False)]
        for index, change in enumerate(changes):
            data = self.graph_metadata('B'); change(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.graph_checker('B', data)('B')
        data = self.graph_metadata('A')
        data['sites']['32']['head_encoder']['current_adapter_leaves_reached'] = 1
        with self.assertRaises(ValueError):
            self.graph_checker('A', data)('A')

    def norm_checker(self):
        fake_np = SimpleNamespace(float64='mock-f64', square=lambda value: FakeScalarArray([v * v for v in value.values]))
        return extracted('site_norms', need=NEED, np=fake_np, model=SimpleNamespace(SITES=SITES))

    def test_site_gradient_norms_bind_exact_nine_head_and_six_adapter_tensors(self):
        values = {family + '.' + site + '.' + str(index): FakeScalarArray([1.])
                  for family, count in (('heads', 9), ('adapters', 6))
                  for site in SITES for index in range(count)}
        current = self.norm_checker()(values, 'B')
        self.assertEqual(len(current), 8)
        self.assertTrue(all(current['heads/' + site] == 3. for site in SITES))
        self.assertTrue(all(current['adapters/' + site] == math.sqrt(6) for site in SITES))
        self.assertEqual(len(self.norm_checker()(values, 'A')), 4)
        del values['adapters.128.0']
        with self.assertRaisesRegex(ValueError, 'scope'):
            self.norm_checker()(values, 'B')

    def test_site_gradient_norms_refuse_nonfinite_accumulation(self):
        values = {'heads.' + site + '.' + str(index): FakeScalarArray([1.])
                  for site in SITES for index in range(9)}
        values['heads.64.0'] = FakeScalarArray([float('inf')])
        with self.assertRaisesRegex(ValueError, 'Finite'):
            self.norm_checker()(values, 'A')

    def test_checkpoint_tree_scalar_schema_detects_nested_value_type_or_shape_changes(self):
        # FakeTensor is only a type marker; no fake or actual tensor operation runs.
        class FakeTensor:
            pass
        equal = extracted('tree_equal', torch=SimpleNamespace(Tensor=FakeTensor))
        current = {'optimizer': {'state': {0: {'step': 1., 'shape': [32, 256, 1, 1]}},
                    'param_groups': [{'lr': 1e-5}]}, 'arm': 'B', 'update': 1,
                   'sampler': {'next_update': 2}, 'RNG': {'CPU_hex': '123'}}
        self.assertTrue(equal(current, copy.deepcopy(current)))
        for change in (lambda d: d['optimizer']['state'][0].update(step=2.),
                       lambda d: d['optimizer']['state'][0].update(shape=[32, 128, 1, 1]),
                       lambda d: d.update(update=True), lambda d: d['RNG'].update(CPU_hex='changed'),
                       lambda d: d['optimizer'].pop('param_groups')):
            changed = copy.deepcopy(current); change(changed)
            self.assertFalse(equal(current, changed))

    def test_worker_imports_are_inert_and_pre_framework_guards_precede_Torch(self):
        imports = [node for node in TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            roots = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]]
            self.assertTrue(set(roots) <= sys.stdlib_module_names)
        framework = min(n.lineno for n in ast.walk(RUN) if isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names))
        guard_names = ('verify_static', 'require_vm', 'supervisor_context')
        for name in guard_names:
            positions = [n.lineno for n in ast.walk(RUN) if isinstance(n, ast.Call) and
                         ((isinstance(n.func, ast.Attribute) and n.func.attr == name) or
                          (isinstance(n.func, ast.Name) and n.func.id == name))]
            self.assertTrue(positions and min(positions) < framework)
        self.assertLess(SOURCE.index('Competing GPU workload; stop before construction'), SOURCE.index('import torch as torch_module'))
        self.assertFalse(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                             isinstance(n.value.func, ast.Name) and n.value.func.id == 'run' for n in TREE.body))

    def test_source_dependency_APIs_and_immutable_retained_pins_resolve(self):
        package = ROOT / 'research/codeformer_matched_feature_learning_v1'
        needed = {'model': ['build_restorer', 'expected_head_shapes', 'expected_adapter_shapes'],
                  'objective': ['connected_objective', 'objective_numpy'],
                  'records': ['record_inherited', 'record_new'],
                  'quality': ['evaluate_inherited', 'evaluate_new_bank'],
                  'execution': ['audit_journal'], 'packet': ['verify_static', 'require_vm', 'pin', 'regular', 'strict_json']}
        for module, names in needed.items():
            tree = ast.parse((package / (module + '.py')).read_text(), feature_version=(3, 10))
            actual = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            self.assertTrue(set(names) <= actual, (module, names))
        constants = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        for path_key, pin_key in (('HELPER', 'HELPER_PIN'), ('HEAD_DECODER', 'HEAD_DECODER_PIN')):
            payload = (ROOT / constants[path_key]).read_bytes()
            self.assertEqual((len(payload), sha256(payload).hexdigest()), constants[pin_key])

    def test_full_worker_loop_maps_to_exact1736_banked_slots_without_extra_neural_call(self):
        self.assertEqual(len(self.plan), 1736)
        self.assertEqual(sum(call['backward'] for call in self.plan), 480)
        self.assertEqual(sum(call['optimizer_application_after_this_forward'] for call in self.plan), 96)
        for arm in C.ARMS:
            rows = [call for call in self.plan if call['arm'] == arm]
            self.assertEqual(len(rows), 868)
            self.assertEqual(sum(call['new_update'] == 0 for call in rows), 220)
            self.assertEqual(sum(call['phase'] == 'terminal_optimization_readout' for call in rows), 120)
        calls = [n for n in ast.walk(RUN) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'candidates']
        self.assertEqual(len(calls), 1)
        self.assertTrue(function('full_call').lineno < calls[0].lineno < function('saved_record').lineno)
        self.assertIn("call['case_key']", ast.unparse(function('full_call')))

    def test_durable_reservation_return_wraps_only_existing_neural_and_update_calls(self):
        forward = ast.unparse(function('full_call'))
        self.assertLess(forward.index('ledger.reserve_forward(call)'), forward.index("candidates[call['arm']](sample)"))
        self.assertLess(forward.index('torch.cuda.synchronize(0)'), forward.index('ledger.returned(token)'))
        self.assertLess(SOURCE.index('ledger.reserve_optimizer_construction(arm)'), SOURCE.index('torch.optim.Adam('))
        self.assertLess(SOURCE.index('ledger.reserve_optimizer_update(arm, update)'), SOURCE.index('optimizer.step()'))
        self.assertIn("store.write(filename, b'')", SOURCE)
        self.assertIn("store.append('operations.jsonl', value)", SOURCE)
        self.assertIn('ledger = execution.ExecutionLedger(selection, persist=persist)', SOURCE)

    def test_all440_initial_returns_before_first_optimizer_and_distinct_arm_models(self):
        self.assertLess(SOURCE.index("snapshot()['counters']['full_forward_returns'] == 440"), SOURCE.index('torch.optim.Adam('))
        self.assertLess(SOURCE.index("initial_records[arm][call['bank']][row['id']] = record"), SOURCE.index('torch.optim.Adam('))
        self.assertIn("candidates['A'].base is not candidates['B'].base", SOURCE)
        self.assertIn("set(model.expected_adapter_shapes())", SOURCE)
        self.assertIn('rng_record() == before_rng', SOURCE)
        self.assertIn('model.FULL_STATE_CANONICAL_SHA256', SOURCE)

    def test_one_division_by5_on_sole_backward_and_no_teacher_training_target(self):
        backwards = [n for n in ast.walk(RUN) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'backward']
        self.assertEqual(len(backwards), 1)
        self.assertEqual(ast.unparse(backwards[0]), '(value / 5).backward()')
        self.assertLess(SOURCE.index("ledger.reserve_backward(call['slot'])"), SOURCE.index('(value / 5).backward()'))
        self.assertIn('objective.connected_objective(raw, target, role, torch=torch)', SOURCE)
        self.assertIn('objective.objective_numpy(raw_array, target_array, role, np=np)', SOURCE)
        self.assertIn('target_array.astype(np.float32) / np.float32(255)', SOURCE)
        self.assertNotIn('autograd.grad(', SOURCE)

    def test_gradients_finite_exact_schema_frozen_None_and_positive_first_site_norms(self):
        source = ast.unparse(function('gradient_arrays'))
        for text in ('grad.dtype == torch.float32', 'grad.device == device', 'grad.shape == value.shape', 'torch.isfinite(grad).all().item()'):
            self.assertIn(text, source)
        self.assertIn('all(norm > 0 for norm in norms.values()) if update == 1', SOURCE)
        self.assertIn('max_norm=contract.OPTIMIZER[', SOURCE)
        self.assertIn('error_if_nonfinite=True', SOURCE)
        self.assertIn('Frozen arm gradients must stay None', SOURCE)
        self.assertIn("'zero_individual_role_site_allowed': True", SOURCE)

    def test_pixel_records_are_readback_measured_and_old_new_banks_separate(self):
        saved = ast.unparse(function('saved_record'))
        for text in ('records.record_inherited', 'records.record_new', "rgb(outputs['png'])", "'RAW_pixel_bytes_sha256'", "'PNG_RGB8_sha256'"):
            self.assertIn(text, saved)
        self.assertIn("bank['old_quality']", ast.unparse(function('bank_gate')))
        self.assertIn("bank['new_monitors']", ast.unparse(function('bank_gate')))
        self.assertIn("for _ in range(96):", SOURCE)
        self.assertIn('update in contract.SNAPSHOTS[1:]', SOURCE)
        self.assertIn("with torch.no_grad():", SOURCE)
        self.assertNotIn('arithmetic_close', ast.unparse(function('bank_gate')))

    def test_checkpoint_retains_actual_values_Adam_RNG_sampler_and_frozen_bindings(self):
        checkpoint = ast.unparse(function('checkpoint'))
        for text in ('assert_frozen(arm)', "optimizers[arm].state_dict()", "'optimizer_parameter_names'",
                     "'sampler_next_update'", "'sampler_next_plan_call'", "'RNG': rng_record()", 'tree_equal(state, restored)',
                     'torch.load(checked(resume), map_location=\'cpu\', weights_only=True)'):
            self.assertIn(text, checkpoint)
        self.assertIn("'before_canonical_sha256'", SOURCE)
        self.assertIn("'actual_after_step'", SOURCE)
        self.assertLess(SOURCE.index("after_binding = store.write(prefix + '/actual_after_step.npz'"), SOURCE.index("need(displacement['pass']"))
        self.assertIn('full_state_count < 3', SOURCE)
        self.assertIn('value.grad is None and np.array_equal(', SOURCE)

    def test_fatal_unknown_state_is_retained_without_resume_and_failure_budgets_are_finite(self):
        for text in ('ledger.fatal_stop(', "mode='failure_capture'", "mode='failure_receipt'",
                     "'unvalidated_current575_accepted_as_valid'] = False", "'accepted_as_valid': False",
                     "'partial_files'] = sorted(store.partial_files)", 'emergency=True'):
            self.assertIn(text, SOURCE)
        self.assertNotIn('while True', SOURCE)
        self.assertIn('Prior complete or partial matched run immutable', SOURCE)
        self.assertIn("'operation_outcome_unknown': state.get('pending') is not None", SOURCE)

    def test_bounded_phase_owns_one_anchor_and_sigalarm_is_restored_on_exception(self):
        bounded = function('bounded_phase')
        monotonic = [n for n in ast.walk(bounded) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == 'time.monotonic']
        self.assertEqual(len(monotonic), 1)
        self.assertIn('phase_started_at=start', ast.unparse(bounded))
        signal_calls = [ast.unparse(n) for n in ast.walk(function('timed')) if isinstance(n, ast.Call)]
        self.assertIn('signal.setitimer(signal.ITIMER_REAL, 0)', signal_calls)
        self.assertIn('signal.signal(signal.SIGALRM, old)', signal_calls)
        self.assertIn("limits['worker_seconds']", SOURCE)
        self.assertIn("limits['per_arm_seconds']", SOURCE)
        self.assertNotIn('1260', SOURCE)

    def test_extracted_timer_restores_prior_handler_after_mock_phase_failure(self):
        actions = []
        class MockSignal:
            ITIMER_REAL = 0
            SIGALRM = 14
            def getitimer(self, kind):
                return (0.0, 0.0)
            def getsignal(self, kind):
                return 'prior-handler'
            def signal(self, kind, handler):
                actions.append(('handler', kind, handler))
            def setitimer(self, kind, seconds):
                actions.append(('timer', kind, seconds))
        timer = extracted('timed', need=NEED, signal=MockSignal())
        with self.assertRaisesRegex(RuntimeError, 'mock phase failure'):
            with timer(7.5, 'mock_only'):
                raise RuntimeError('mock phase failure')
        self.assertEqual([item for item in actions if item[0] == 'timer'], [('timer', 0, 7.5), ('timer', 0, 0)])
        self.assertEqual(actions[-1], ('handler', 14, 'prior-handler'))
        actions.clear()
        with self.assertRaises(ValueError):
            with timer(0, 'zero-remaining'):
                self.fail('Expired phase must not enter')
        self.assertEqual(actions, [])

    def test_extracted_bounded_phase_retains_one_start_and_restores_failed_context(self):
        # Nonlocals are redirected only in the compiled test copy; worker source is untouched.
        node = copy.deepcopy(function('bounded_phase'))
        class TestScope(ast.NodeTransformer):
            def visit_Nonlocal(self, value):
                return ast.copy_location(ast.Global(names=value.names), value)
        node = ast.fix_missing_locations(TestScope().visit(node))
        times = iter((100., 101.))
        entered, deadline_calls = [], []
        @contextmanager
        def timer(seconds, label):
            entered.append((seconds, label))
            yield
        class Budget:
            def deadline(self, now, *, phase_started_at):
                deadline_calls.append((now, phase_started_at))
                return {'remaining_seconds': 299.}
        namespace = {'contextmanager': contextmanager, 'phase': 'prior_phase',
                     'failed_phase': None, 'failed_arm': None, 'active_arm': 'B',
                     'time': SimpleNamespace(monotonic=lambda: next(times)), 'budget': Budget(),
                     'resource': lambda label: None, 'timed': timer, 'event': lambda value: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER) + ':bounded_phase_fixture', 'exec'), namespace)
        with self.assertRaisesRegex(RuntimeError, 'mock bounded failure'):
            with namespace['bounded_phase']('update_B_1'):
                self.assertEqual(namespace['phase'], 'update_B_1')
                raise RuntimeError('mock bounded failure')
        self.assertEqual(deadline_calls, [(101., 100.)])
        self.assertEqual(entered, [(299., 'update_B_1')])
        self.assertEqual((namespace['phase'], namespace['failed_phase'], namespace['failed_arm']),
                         ('prior_phase', 'update_B_1', 'B'))

    def test_terminal_journal_replay_and_RNG_restore_failure_invalidate_exit(self):
        self.assertIn('ledger.finish()', SOURCE)
        self.assertIn('execution.audit_journal(', SOURCE)
        self.assertIn("replay['counters'] == snapshot()['counters']", SOURCE)
        self.assertIn('store.audit_namespace()', SOURCE)
        self.assertIn("'prior_result_invalidated': True", SOURCE)
        self.assertIn("'rng_finalization_failure.json'", SOURCE)
        self.assertIn('torch.set_rng_state', SOURCE)
        self.assertIn('torch.cuda.set_rng_state_all', SOURCE)
        self.assertIn('return proposed_exit', SOURCE)

    def test_no_worker_model_or_numerical_framework_import_or_runtime_calls(self):
        self.assertEqual(BLOCKER.attempted, [])
        self.assertFalse(BLOCKER.roots & {name.split('.')[0] for name in sys.modules})
        self.assertNotIn('research.codeformer_matched_feature_learning_v1.worker', sys.modules)
        self.assertNotIn('research.codeformer_matched_feature_learning_v1.model', sys.modules)


if __name__ == '__main__':
    unittest.main()
