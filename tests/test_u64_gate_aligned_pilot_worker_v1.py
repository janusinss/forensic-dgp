"""Independent synthetic contract/journal tests; never call the neural worker.

These fixtures validate accounting and source guards only. They establish no
GPU execution, model usefulness, finite-step result or VM readiness.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from research.codeformer_u64_gate_aligned_pilot_v1 import contract, worker

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'


def source_contract():
    parent = (STAGE / 'evidence/parent_protocol.json').read_bytes()
    standard = (STAGE / 'evidence/standard_protocol.json').read_bytes()
    gradients = (STAGE / 'protocol.json').read_bytes()
    return parent, standard, gradients


class WorkerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent, cls.standard, cls.gradients = source_contract()
        cls.selection = contract.select_cases(cls.parent, cls.standard)
        cls.plan = contract.operation_plan(cls.selection)

    def test_import_is_inert_with_neural_framework_blocker(self):
        code = '''import importlib.abc, importlib, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path, target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','jax'}:
   raise RuntimeError('Forbidden neural import:'+fullname)
  return None
sys.meta_path.insert(0,Block())
for suffix in ('contract','arithmetic','quality','worker','runtime'):
 importlib.import_module('research.codeformer_u64_gate_aligned_pilot_v1.'+suffix)
assert not any(n.split('.')[0] in {'torch','tensorflow','jax'} for n in sys.modules)
print('INERT_IMPORT_NO_NEURAL_FRAMEWORK')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'INERT_IMPORT_NO_NEURAL_FRAMEWORK')

    def test_actual_protocol_selection_and_exact_operation_budget(self):
        self.assertEqual(hashlib.sha256(self.parent).hexdigest(), contract.PARENT_SHA256)
        self.assertEqual(hashlib.sha256(self.standard).hexdigest(), contract.STANDARDIZED_SHA256)
        self.assertEqual(hashlib.sha256(self.gradients).hexdigest(), contract.GRADIENT_PROTOCOL_SHA256)
        self.assertEqual(len(self.plan), 544)
        self.assertEqual(sum(c['kind'] == 'forward' for c in self.plan), 540)
        self.assertEqual(sum(c['kind'] == 'parameter_application' for c in self.plan), 4)
        self.assertEqual(sum(len(c['gradient_queries']) for c in self.plan), 512)
        self.assertEqual([c['slot'] for c in self.plan], list(range(1, 545)))
        for category, expected in {'initial_quality_forwards': 56,
                'initial_repeat_forwards': 4, 'optimization_forwards': 256,
                'quality_forwards': 224}.items():
            self.assertEqual(sum(c.get('category') == category for c in self.plan), expected)
        self.assertEqual((len(self.selection['optimization']), len(self.selection['monitors']),
                          len(self.selection['guards'])), (64, 32, 24))

    def test_gradient_scope_and_each_application_after_all128_current_queries(self):
        optimizing = {r['id']: r for r in self.selection['optimization']}
        quality_ids = {r['id'] for r in self.selection['monitors'] + self.selection['guards']}
        self.assertFalse(set(optimizing) & quality_ids)
        query_counts, queried_ids, applied = {}, {}, []
        for call in self.plan:
            update = call['new_update']
            if call['gradient_queries']:
                self.assertEqual(call['kind'], 'forward')
                self.assertEqual(call['category'], 'optimization_forwards')
                self.assertEqual(call['gradient_queries'], list(contract.QUERIES))
                self.assertIn(call['id'], optimizing)
                self.assertEqual(optimizing[call['id']]['split'], 'TRAIN')
                self.assertTrue(optimizing[call['id']]['optimization_permitted'])
                query_counts[update] = query_counts.get(update, 0) + 2
                queried_ids.setdefault(update, []).append(call['id'])
            elif call['kind'] == 'parameter_application':
                self.assertEqual(query_counts[update], 128)
                self.assertEqual(queried_ids[update], [r['id'] for r in self.selection['optimization']])
                self.assertNotIn(update, applied)
                applied.append(update)
            elif call.get('category') == 'quality_forwards':
                self.assertIn(update, applied)
                self.assertIn(call['id'], quality_ids)
        self.assertEqual(applied, [1, 2, 3, 4])
        self.assertEqual(query_counts, dict.fromkeys(range(1, 5), 128))

    def test_original_quality_initialization_and_fixed_scalar_remain_bound(self):
        actual = contract.prospective_contract(self.parent, self.standard, self.gradients)
        original = json.loads(self.standard)
        self.assertEqual(actual['original_quality_policy_exact'], original['quality_policy'])
        self.assertEqual(actual['original_source_only_design']['quality']['inherited_legacy24_guard_definitions_exact'], original['guards'])
        self.assertEqual(actual['original_quality_policy_exact']['terminal64_adjacent_gain_fraction_minimum'], .01)
        self.assertTrue(actual['original_quality_policy_exact']['terminal64_dark_MAE_and_adjacent_nonregression'])
        self.assertEqual(actual['initializer']['all_tensors'], 575)
        self.assertEqual(actual['initializer']['frozen_R2_U96_tensors'], 539)
        self.assertEqual(actual['initializer']['trainable_head_tensors'], 36)
        self.assertEqual(actual['initializer']['trainable_head_coordinates'], 168960)
        self.assertEqual(actual['initializer']['full575_canonical_sha256'], contract.FULL_STATE_CANONICAL_SHA256)
        self.assertEqual(actual['initializer']['head36_canonical_sha256'], contract.HEAD_CANONICAL_SHA256)
        self.assertEqual(actual['step_scalar_safety_factor'], .99)
        self.assertFalse(actual['model_qualified'])
        self.assertFalse(actual['automatic_retry_resume'])
        self.assertEqual(actual['retained_full575_states'], 2)
        self.assertEqual(actual['retained_head_states'], 5)

    def test_bad_parent_or_standard_bytes_are_refused(self):
        for parent, standard in ((self.parent + b' ', self.standard),
                                 (self.parent, self.standard + b' ')):
            with self.assertRaises(ValueError):
                contract.select_cases(parent, standard)


class LedgerSyntheticTests(unittest.TestCase):
    def test_complete_plan_simulation_matches_every_quota_without_operations(self):
        parent, standard, _ = source_contract()
        plan = contract.operation_plan(contract.select_cases(parent, standard))
        journal = []
        ledger = worker.OperationLedger(lambda row: journal.append(copy.deepcopy(row)))
        for call in plan:
            fields = {'id': call['id'], 'plan_slot': call['slot'], 'new_update': call['new_update']}
            if call['kind'] == 'forward':
                token = ledger.reserve('forward', category=call['category'], **fields)
                ledger.returned(token, synthetic_return=True)
                for query in call['gradient_queries']:
                    query_token = ledger.reserve('gradient_query', query=query, **fields)
                    ledger.returned(query_token, synthetic_return=True)
            else:
                token = ledger.reserve('parameter_application', **fields)
                ledger.returned(token, synthetic_return=True)
        expected = {'full_forward_attempts': 540, 'full_forward_returns': 540,
            'gradient_query_attempts': 512, 'gradient_query_returns': 512,
            'parameter_application_attempts': 4, 'parameter_applications': 4,
            'initial_quality_forwards': 56, 'initial_repeat_forwards': 4,
            'optimization_forwards': 256, 'quality_forwards': 224}
        self.assertEqual(ledger.counters, expected)
        self.assertEqual(ledger.sequence, 1056)
        self.assertEqual(len(journal), 2112)
        self.assertEqual(ledger.snapshot()['unknown_reservations'], [])
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])
        self.assertTrue(all(journal[i]['token'] == journal[i + 1]['token']
                            for i in range(0, len(journal), 2)))
        for kind, metadata in (('forward', {'category': 'initial_quality_forwards'}),
                               ('gradient_query', {}), ('parameter_application', {})):
            with self.assertRaises(ValueError):
                ledger.reserve(kind, **metadata)

    def test_interrupted_parameter_commit_reservation_stays_unknown(self):
        ledger = worker.OperationLedger(lambda row: None)
        token = ledger.reserve('parameter_application', plan_slot=125, new_update=1)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['counters']['parameter_application_attempts'], 1)
        self.assertEqual(snapshot['counters']['parameter_applications'], 0)
        self.assertTrue(snapshot['operation_outcome_unknown'])
        self.assertEqual(snapshot['unknown_reservations'], [
            {'token': token, 'kind': 'parameter_application', 'plan_slot': 125, 'new_update': 1}])
        with self.assertRaises(ValueError):
            ledger.reserve('gradient_query', query='gate_mae')

    def test_durable_return_failure_stays_pending_and_does_not_count_return(self):
        rows = []
        def persist(row):
            if row['kind'].endswith('_returned'):
                raise OSError('synthetic fsync failure')
            rows.append(copy.deepcopy(row))
        ledger = worker.OperationLedger(persist)
        token = ledger.reserve('parameter_application', new_update=1)
        with self.assertRaises(OSError):
            ledger.returned(token, synchronized_return=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(ledger.counters['parameter_applications'], 0)
        self.assertIn(token, ledger.pending)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_durable_reservation_failure_is_still_conservatively_unknown(self):
        def persist(row):
            raise OSError('synthetic reservation fsync failure')
        ledger = worker.OperationLedger(persist)
        with self.assertRaises(OSError):
            ledger.reserve('gradient_query', query='gate_mae')
        self.assertEqual(ledger.counters['gradient_query_attempts'], 1)
        self.assertEqual(ledger.counters['gradient_query_returns'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_unknown_kind_category_reserved_metadata_and_bad_return_refused(self):
        ledger = worker.OperationLedger(lambda row: None)
        for kind, metadata in (('unknown', {}), ('forward', {'category': 'unknown'}),
                               ('forward', {}), ('gradient_query', {'token': 1}),
                               ('gradient_query', {'counters': {}})):
            with self.assertRaises(ValueError):
                ledger.reserve(kind, **metadata)
        self.assertEqual(ledger.counters, dict.fromkeys(worker.COUNTER_KEYS, 0))
        self.assertFalse(ledger.pending)
        token = ledger.reserve('gradient_query', query='gate_mae')
        for wrong in (0, token + 1, True, '1'):
            with self.assertRaises(ValueError):
                ledger.returned(wrong)
        for metadata in ({'kind': 'forged'}, {'token': token}, {'counters': {}}):
            with self.assertRaises((ValueError, TypeError)):
                ledger.returned(token, **metadata)
        self.assertEqual(ledger.counters['gradient_query_returns'], 0)
        self.assertIn(token, ledger.pending)


class ManualAndSourceGuardTests(unittest.TestCase):
    def test_supervisor_requires_exact_protocol_environment_and_actual_parent_pid(self):
        sha = 'a' * 64
        good = {contract.ENV + 'SUPERVISED': '1', contract.ENV + 'PROTOCOL': sha,
                contract.ENV + 'SUPERVISOR_PID': '12345'}
        worker.supervisor_context(sha, environ=good, parent_pid=12345)
        for key in good:
            bad = dict(good)
            bad[key] = 'wrong'
            with self.assertRaises(ValueError):
                worker.supervisor_context(sha, environ=bad, parent_pid=12345)
        with self.assertRaises(ValueError):
            worker.supervisor_context(sha, environ=good, parent_pid=12346)
        with self.assertRaises(ValueError):
            worker.supervisor_context('A' * 64, environ=good, parent_pid=12345)

    def test_all_neural_imports_construction_queries_and_copies_are_owned_by_run(self):
        source = Path(worker.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        def within_run(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'run':
                    return True
            return False
        imports, queries, copies, constructors = [], [], [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
                if any(name in {'torch', 'tensorflow', 'jax'} for name in names):
                    imports.append(node)
            elif isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in {'torch', 'tensorflow', 'jax'}:
                imports.append(node)
            if not isinstance(node, ast.Call):
                continue
            text = ast.unparse(node.func)
            if text == 'torch.autograd.grad':
                queries.append(node)
            if text.endswith('.copy_'):
                copies.append(node)
            if text in {'CodeFormer', 'model.build_restorer'}:
                constructors.append(node)
            if (text.startswith(('torch.', 'candidate.', 'model.')) or text in {'CodeFormer'}):
                self.assertTrue(within_run(node), 'Neural operation outside guarded run: ' + text)
        self.assertGreaterEqual(len(imports), 2)
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(copies), 1)
        self.assertEqual(len(constructors), 2)
        self.assertTrue(all(within_run(node) for node in imports + queries + copies + constructors))
        # Nothing at module scope invokes run or imports the model factory.
        self.assertFalse(any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'run'
                             for top in tree.body if not isinstance(top, (ast.FunctionDef, ast.ClassDef))
                             for node in ast.walk(top)))
        guarded = ['runtime.verify_static(', 'runtime.require_vm(', 'supervisor_context(expected_sha,',
                   "'static_VM_supervisor_guards_passed_before_Torch'", 'import torch as torch_module']
        offsets = [source.index(text) for text in guarded]
        self.assertEqual(offsets, sorted(offsets))


if __name__ == '__main__':
    unittest.main()
