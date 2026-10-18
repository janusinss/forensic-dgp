"""CPU/std-lib worker contract checks; never import a neural framework."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / 'scratch'
SOURCE = ROOT / 'research/codeformer_loss_response_v1/worker.py'
spec = importlib.util.spec_from_file_location('loss_response_worker_static_test', SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
contract_spec = importlib.util.spec_from_file_location('loss_response_contract_static_test',
    ROOT / 'research/codeformer_loss_response_v1/contract.py')
contract = importlib.util.module_from_spec(contract_spec)
contract_spec.loader.exec_module(contract)


class Vector:
    """Routing fixture only; this is not a neural or FP32 numerical replay."""
    def __init__(self, values):
        self.values = list(values)
    def copy(self):
        return Vector(self.values)


class ArrayRouting:
    @staticmethod
    def add(left, right, out):
        out.values[:] = [a + b for a, b in zip(left.values, right.values)]
    @staticmethod
    def zeros_like(value):
        return Vector([0] * len(value.values))


class FakeTorch:
    class Tensor:
        pass


class WorkerStaticChecks(unittest.TestCase):
    def fixture_directory(self):
        SCRATCH.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='loss-response-worker-', dir=SCRATCH)
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(SCRATCH.resolve()))
        return temporary

    def test_frozen_census_and_forward_query_arithmetic(self):
        self.assertEqual(worker.EXPECTED_CAPS, contract.COUNTS)
        self.assertEqual(list(worker.COMPONENTS), contract.COMPONENTS)
        self.assertEqual(list(worker.CONDITIONS), contract.BRANCHES)
        totals = worker.call_totals(worker.EXPECTED_CAPS)
        self.assertEqual(totals['neural_forwards'], 425)
        self.assertEqual(totals['gradient_queries'], 193)
        self.assertEqual(totals['Adam_step_attempt_reservations'], 18)
        self.assertEqual(totals['committed_training_updates'], 0)
        self.assertEqual(totals['backward_calls'], 0)
        self.assertEqual(len(worker.CONDITIONS), 10)

    def test_resource_projection_and_oversize_admission(self):
        planned = worker.projected_resources(855230072)
        self.assertLess(planned['projected_RAM_bytes'], contract.LIMITS['RSS_bytes'])
        self.assertLess(planned['projected_CUDA_bytes'], contract.LIMITS['CUDA_reserved_bytes'])
        self.assertGreater(worker.projected_resources(10 * 1024**3)['projected_RAM_bytes'],
                           contract.LIMITS['RSS_bytes'])
        with self.assertRaises(ValueError):
            worker.projected_resources(-1)

    def test_component_composition_and_leaveout_routing(self):
        names = ['encoder.x', 'idx_pred_layer.x', 'fuse_convs_dict.x']
        values = {component: {name: None for name in names} for component in worker.COMPONENTS}
        values['token_ce']['encoder.x'] = Vector([1, 2])
        values['latent_mse']['encoder.x'] = Vector([3, 4])
        values['token_ce']['idx_pred_layer.x'] = Vector([5, 6])
        values['behavior_token_kl']['idx_pred_layer.x'] = Vector([7, 8])
        values['coarse_rgb_l1']['fuse_convs_dict.x'] = Vector([9, 10])
        full = worker.compose_gradients(values, names, ArrayRouting)
        self.assertEqual(full['encoder.x'].values, [4, 6])
        self.assertEqual(full['idx_pred_layer.x'].values, [12, 14])
        dropped = worker.compose_gradients(values, names, ArrayRouting, omitted='token_ce')
        self.assertEqual(dropped['encoder.x'].values, [3, 4])
        self.assertEqual(dropped['idx_pred_layer.x'].values, [7, 8])
        momentum = worker.compose_gradients(values, names, ArrayRouting, momentum_only=True)
        self.assertTrue(all(value.values == [0, 0] for value in momentum.values()))
        self.assertEqual(values['token_ce']['encoder.x'].values, [1, 2])
        self.assertIsNone(values['pixel_l1_rgb']['encoder.x'])

    def test_none_is_not_an_implicit_zero_gradient(self):
        values = {component: {'encoder.x': None} for component in worker.COMPONENTS}
        with self.assertRaisesRegex(ValueError, 'Every disposable branch'):
            worker.compose_gradients(values, ['encoder.x'], ArrayRouting)

    def test_nonlinux_guard_fails_before_files_or_neural_imports(self):
        with patch.object(worker.platform, 'system', return_value='Windows'), \
                patch.object(worker.Path, 'home', side_effect=AssertionError('must not inspect home')):
            with self.assertRaisesRegex(ValueError, 'Linux L4 VM only'):
                worker.require_vm_path(ROOT)

    def test_new_evidence_is_exclusive_and_paths_stay_bounded(self):
        with self.fixture_directory() as temporary:
            base = Path(temporary)
            path = base / 'result.json'
            worker.atomic_json(path, {'complete': False})
            self.assertEqual(worker.read_json(path), {'complete': False})
            with self.assertRaisesRegex(ValueError, 'Retain existing evidence'):
                worker.atomic_json(path, {'complete': True})
            with self.assertRaises(ValueError):
                worker.bounded(base, '../outside.json', must_exist=False)
            with self.assertRaises(ValueError):
                worker.bounded(base, 'nested\\bad.json', must_exist=False)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        with self.fixture_directory() as temporary:
            path = Path(temporary) / 'bad.json'
            path.write_text('{"x": 1, "x": 2}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                worker.read_json(path)
            path.write_text('{"x": NaN}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                worker.read_json(path)

    def test_metadata_digest_is_order_stable_and_type_sensitive(self):
        self.assertEqual(worker.tree_digest({'b': [2, 3], 'a': 1}, FakeTorch),
                         worker.tree_digest({'a': 1, 'b': [2, 3]}, FakeTorch))
        self.assertNotEqual(worker.tree_digest([1, 2], FakeTorch),
                            worker.tree_digest((1, 2), FakeTorch))

    def test_ast_guards_forbidden_local_or_committed_operations(self):
        source = SOURCE.read_text(encoding='utf-8')
        module = ast.parse(source)
        top_imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(node) for node in top_imports))
        calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'backward' for node in calls))
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'save' and
                             isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' for node in calls))
        adam_steps = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'step']
        self.assertEqual(len(adam_steps), 1)
        run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        first = ast.unparse(run.body[0])
        self.assertIn('require_vm_path(root)', first)
        imports = [node for node in ast.walk(run) if isinstance(node, ast.Import) and
                   any(alias.name == 'torch' for alias in node.names)]
        self.assertEqual(len(imports), 1)
        self.assertGreater(imports[0].lineno, run.body[0].lineno)
        self.assertIn("parameters[name].grad = torch.from_numpy", source)
        self.assertIn("momentum_only_explicit_zero", source)
        self.assertIn("optimizer_inflight_at_failure", source)
        self.assertIn("exact_base_model_Adam_RNG_restored", source)


if __name__ == '__main__':
    unittest.main()
