"""Pure unit and AST tests for Conditioned SFT Learning V1; no Torch or model execution."""
import ast
from collections import Counter
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import unittest


FRAMEWORK_ROOTS = frozenset(('torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'jaxlib', 'keras'))


class FrameworkImportBlocker(importlib.abc.MetaPathFinder):
    """Fail immediately if any neural framework import is attempted during unit tests."""

    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FRAMEWORK_ROOTS:
            self.attempted.append(fullname)
            raise AssertionError('Neural framework import forbidden in unit tests: ' + fullname)
        return None


FRAMEWORK_BLOCKER = FrameworkImportBlocker()
sys.meta_path.insert(0, FRAMEWORK_BLOCKER)

import numpy as np

from research.codeformer_conditioned_sft_learning_v1 import (
    conditioning as cond,
    contract as C,
    entry as EN,
    execution as E,
    graph as G,
    install_weights as IW,
    objective as obj,
    packet as P,
    quality as Q,
    records as R,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_BASE = (
    ROOT / 'outputs'
    if (ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation').exists()
    else ROOT.parent / 'Testing' / 'outputs'
)
PREP = OUTPUTS_BASE / 'codeformer_matched_feature_learning_v1_preparation'
PARENT = OUTPUTS_BASE / 'cctv_codeformer_u64_cleananchors_continuation_v1_vm/evidence/u64_parent_protocol.json'


class Node:
    def __init__(self, *children):
        self.next_functions = tuple((child, 0) for child in children)


class AccumulateGrad(Node):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable


class TestConditionedSftLearningV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (
            (PREP / 'data_admission_v1.json').read_bytes(),
            (PREP / 'candidate_data_v1.json').read_bytes(),
            PARENT.read_bytes(),
        )
        cls.selection = C.select_cases(*cls.raw)
        cls.operations = C.operation_plan(cls.selection)

    @staticmethod
    def arrays(value=0.0, dtype=np.float64):
        return np.full((256, 256, 3), value, dtype=dtype)

    @staticmethod
    def sft_dict(value=0.0, dtype=np.float64):
        shapes = {'32': (1, 256, 32, 32), '64': (1, 256, 64, 64), '128': (1, 128, 128, 128), '256': (1, 64, 256, 256)}
        return {site: np.full(shape, value, dtype=dtype) for site, shape in shapes.items()}

    def test_pure_imports_load_zero_neural_frameworks(self):
        self.assertIn(FRAMEWORK_BLOCKER, sys.meta_path)
        self.assertEqual(FRAMEWORK_BLOCKER.attempted, [])
        for mod in (cond, C, EN, E, G, IW, obj, P, Q, R):
            self.assertIsNotNone(mod)
        for name in ('model.py', 'worker.py'):
            tree = ast.parse((ROOT / 'research/codeformer_conditioned_sft_learning_v1' / name).read_text(encoding='utf-8'))
            self.assertIsNotNone(tree)

    def test_conditioning_signals_separate_normal_and_dark_and_bound_gates(self):
        normal = self.arrays(0.75, dtype=np.float32)
        dark = self.arrays(0.25, dtype=np.float32)
        sig_normal = cond.compute_input_conditioning_numpy(normal, np=np)
        sig_dark = cond.compute_input_conditioning_numpy(dark, np=np)
        self.assertEqual(sig_normal['g_dark'], 0.0)
        self.assertEqual(sig_dark['g_dark'], 1.0)
        for sig in (sig_normal, sig_dark):
            for key in ('mean_luma', 'top1pct_luma', 'g_dark', 'laplacian_energy', 'central_laplacian_energy', 'effective_laplacian_energy', 'g_deg'):
                self.assertIn(key, sig)
            self.assertTrue(0.0 <= sig['g_dark'] <= 1.0)
            self.assertTrue(0.25 <= sig['g_deg'] <= 1.0)

    def test_spatial_residual_decoupling_zeros_block16_mean_drift(self):
        rng = np.random.default_rng(42)
        delta = rng.normal(loc=0.15, scale=0.05, size=(256, 256, 3)).astype(np.float64)
        proj = cond.block16_mean_numpy(delta, np=np)
        residual = delta - proj
        blk_means = residual.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(blk_means, np.zeros((16, 16, 3)), atol=1e-14)

    def test_conditioned_output_preserves_exact_prediction_at_zero_gamma(self):
        inp = self.arrays(0.5, dtype=np.float32)
        pred = self.arrays(0.8, dtype=np.float32)
        zero_gamma, gates = cond.condition_output_numpy(pred, inp, np=np, gamma=0.0)
        np.testing.assert_array_equal(zero_gamma, pred)
        self.assertIn('g_dark', gates)

    def test_conditioned_objective_equal_photograph_and_sft_zero_every_role(self):
        rng = np.random.default_rng(104)
        photograph = rng.random((256, 256, 3))
        sft = self.sft_dict(0.25)
        for role in obj.ROLES:
            with self.subTest(role=role):
                loss, terms, gates = obj.conditioned_objective_numpy(
                    photograph.copy(), photograph, photograph, role,
                    pred_sft_features=sft, target_sft_features=copy.deepcopy(sft), np=np)
                self.assertEqual(loss, 0.0)
                self.assertEqual(tuple(terms), obj.CONDITIONED_TERM_NAMES)
                self.assertTrue(all(value == 0.0 for value in terms.values()))
                self.assertIn('g_dark', gates)

    def test_conditioned_objective_clear_block16_weight_and_sft_distillation_weight(self):
        pred = self.arrays(0.5)
        targ = self.arrays(0.25)
        inp_normal = self.arrays(0.75)
        inp_dark = self.arrays(0.25)
        sft_pred = self.sft_dict(0.5)
        sft_targ = self.sft_dict(0.25)
        loss_norm, terms_norm, gates_norm = obj.conditioned_objective_numpy(
            pred, targ, inp_normal, 'clear',
            pred_sft_features=sft_pred, target_sft_features=sft_targ, np=np)
        self.assertEqual(gates_norm['g_dark'], 0.0)
        self.assertAlmostEqual(terms_norm['RGB_L1'], 0.25, places=12)
        self.assertAlmostEqual(terms_norm['block16_RGB_mean_residual_L1'], 0.25, places=12)
        self.assertAlmostEqual(terms_norm['sft_feature_distillation_L1'], 0.25, places=12)
        expected_norm = 0.25 + 0.10 * 0.25 + obj.LAMBDA_SFT * gates_norm['g_deg'] * 0.25
        self.assertAlmostEqual(loss_norm, expected_norm, places=12)

        loss_dark, terms_dark, gates_dark = obj.conditioned_objective_numpy(
            pred, targ, inp_dark, 'lowlight0p35_noise0p006',
            pred_sft_features=sft_pred, target_sft_features=sft_targ, np=np)
        self.assertEqual(gates_dark['g_dark'], 1.0)
        expected_dark = 0.25 + 0.35 * 0.25 + obj.LAMBDA_SFT * gates_dark['g_deg'] * 0.25
        self.assertAlmostEqual(loss_dark, expected_dark, places=12)
        self.assertEqual(terms_dark['sft_feature_distillation_L1'], 0.25)

    def test_role_gradient_projection_eliminates_negative_inner_product_with_clear(self):
        g_clear = {'w1': np.array([1.0, 0.0, 0.0]), 'w2': np.array([0.0, 2.0, 0.0])}
        g_conflict = {'w1': np.array([-2.0, 1.0, 0.0]), 'w2': np.array([0.0, -1.5, 3.0])}
        g_aligned = {'w1': np.array([0.5, 1.0, 0.0]), 'w2': np.array([0.0, 0.5, 1.0])}
        role_grads = {
            'clear': g_clear,
            'global_gaussian2p0': g_conflict,
            'local_gaussian2p0_soft': g_aligned,
            'compound_g1p2_lr64_jpeg70': g_conflict,
            'lowlight0p35_noise0p006': g_conflict,
        }
        combined, diag = obj.project_role_gradients_numpy(role_grads, np=np)
        self.assertEqual(diag['total_non_clear_tensor_pairs'], 8)
        self.assertEqual(diag['projected_conflict_pairs'], 6)
        self.assertGreaterEqual(diag['min_inner_product_with_clear_after_projection'], -1e-12)
        dot_combined_clear = sum(float(np.sum(combined[k] * g_clear[k])) for k in g_clear)
        self.assertGreater(dot_combined_clear, 0.0)

    def test_objective_ast_has_no_framework_imports_or_detach_or_division(self):
        tree = ast.parse(Path(obj.__file__).read_text(encoding='utf-8'), feature_version=(3, 10))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            if isinstance(node, ast.Import):
                self.assertFalse({a.name.split('.')[0] for a in node.names} & FRAMEWORK_ROOTS)
            elif node.module:
                self.assertNotIn(node.module.split('.')[0], FRAMEWORK_ROOTS)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        connected = functions['connected_conditioned_objective']
        attributes = {node.attr for node in ast.walk(connected) if isinstance(node, ast.Attribute)}
        self.assertFalse({'detach', 'round', 'floor', 'clamp', 'backward', 'autograd', 'no_grad'} & attributes)

    def test_contract_selection_and_schedule_preserve_exact_1736_480_96_counts(self):
        self.assertEqual(
            [len(self.selection[k]) for k in ('new_optimization', 'new_monitors', 'old_monitors', 'old_guards', 'old_quality')],
            [120, 40, 32, 24, 56],
        )
        self.assertEqual(len(self.operations), 1736)
        self.assertEqual(sum(row['backward'] for row in self.operations), 480)
        self.assertEqual(sum(row['optimizer_application_after_this_forward'] for row in self.operations), 96)
        self.assertEqual(Counter(row['arm'] for row in self.operations), {'A': 868, 'B': 868})
        desc = C.prospective_contract()
        self.assertEqual(desc['arm_A']['trainable_coordinates'], 168960)
        self.assertEqual(desc['arm_B']['trainable_coordinates'], 220288)
        self.assertEqual(desc['conditioning']['status'], 'PURE_INPUT_ONLY_EXPOSURE_DEGRADATION_CONDITIONING_V1')
        self.assertIn('g_clear', desc['objective']['role_gradient_conflict_projection'])
        self.assertEqual(desc['objective']['lambda_sft'], 0.08)

    def test_graph_reachability_positive_for_B_and_negative_for_A(self):
        params = {name: object() for name in G.ADAPTER_PARAMETER_NAMES}
        root = Node(*(AccumulateGrad(value) for value in params.values()))
        res_b = G.inspect_adapter_reachability(root, params, expected_present=True)
        self.assertTrue(res_b['pass'])
        self.assertEqual(res_b['current_adapter_leaves_reached'], 6)
        res_a = G.inspect_adapter_reachability(None, params, expected_present=False)
        self.assertTrue(res_a['pass'])
        self.assertEqual(res_a['current_adapter_leaves_reached'], 0)

    def test_execution_ledger_and_journal_replay(self):
        records = []
        ledger = E.ExecutionLedger(self.selection, persist=records.append)
        call0 = self.operations[0]
        token = ledger.reserve_forward(call0)
        ledger.returned(token)
        snap = ledger.summary()
        self.assertEqual(snap['counters']['full_forward_attempts'], 1)
        self.assertEqual(snap['counters']['full_forward_returns'], 1)
        raw_journal = b''.join(records)
        replayed = E.audit_journal(raw_journal, self.selection)
        self.assertEqual(replayed['counters']['full_forward_returns'], 1)

    def test_worker_source_zero_extra_forward_clean_sft_and_gradient_projection(self):
        worker_path = ROOT / 'research/codeformer_conditioned_sft_learning_v1/worker.py'
        source = worker_path.read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 10))
        run_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        calls = [
            n for n in ast.walk(run_node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'candidates'
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn('capture_sft_features=capture_sft_features', source)
        self.assertIn("clean_sft_targets[arm][row['reference_id']] = {", source)
        self.assertIn('objective.connected_conditioned_objective(', source)
        self.assertIn('objective.project_role_gradients_numpy(', source)
        self.assertIn("'role_gradient_conflict_projection': projection_diag", source)


if __name__ == '__main__':
    unittest.main()
