"""Source and scalar-algebra contracts; no Torch/model/gradient/VM execution."""
import ast
import builtins
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_path_diagnostic_v1/neural.py'
SPEC = importlib.util.spec_from_file_location('research.codeformer_path_diagnostic_v1.neural', SOURCE)
neural = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(neural)


class NeuralSourceAndAlgebraTests(unittest.TestCase):
    def test_import_and_python310_parse_require_no_heavy_dependency(self):
        tree = ast.parse(SOURCE.read_text(), feature_version=(3, 10))
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in top_imports:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            self.assertFalse(any(name and name.split('.')[0] in {'torch', 'numpy', 'PIL', 'cv2'} for name in names))
        original = builtins.__import__
        def reject_heavy(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'cv2'}:
                raise AssertionError('Heavy import during static module import')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=reject_heavy):
            copy = importlib.util.module_from_spec(SPEC)
            SPEC.loader.exec_module(copy)
        self.assertEqual(copy.MODES, ('E32', 'E32_grad', 'T32', 'TBF16'))

    def test_neural_run_rejects_windows_before_heavy_import(self):
        original = builtins.__import__
        def reject_heavy(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'cv2'}:
                raise AssertionError('Heavy import before Linux guard')
            return original(name, *args, **kwargs)
        with patch.object(neural.platform, 'system', return_value='Windows'):
            with patch('builtins.__import__', side_effect=reject_heavy):
                with self.assertRaisesRegex(ValueError, 'manual Linux L4 VM only'):
                    neural.run(ROOT, {})

    def test_manual_packet_and_tmux_guard_precedes_heavy_imports(self):
        tree = ast.parse(SOURCE.read_text(), feature_version=(3, 10))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        guard = next(node for node in run.body if isinstance(node, ast.Assign) and
                     isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                     node.value.func.id == 'manual_root')
        imports = [node for node in run.body if isinstance(node, (ast.Import, ast.ImportFrom)) and
                   ((isinstance(node, ast.Import) and any(alias.name in {'numpy', 'torch', 'PIL'} for alias in node.names)) or
                    (isinstance(node, ast.ImportFrom) and node.module in {'torch.nn', 'PIL'}))]
        self.assertTrue(imports)
        self.assertTrue(all(guard.lineno < node.lineno for node in imports))

    def test_gpu_budget_rejects_peak_even_after_current_memory_falls(self):
        limit = 22 * 1024 ** 3
        self.assertIsNone(neural.check_memory_peaks(1, limit, limit))
        with self.assertRaisesRegex(ValueError, 'current/peak'):
            neural.check_memory_peaks(1, limit + 1, limit)
        with self.assertRaisesRegex(ValueError, 'current/peak'):
            neural.check_memory_peaks(limit + 1, limit + 1, limit)
        for values in ((-1, 0, limit), (0, float(limit), limit), (False, 0, limit)):
            with self.assertRaisesRegex(ValueError, 'Invalid diagnostic memory'):
                neural.check_memory_peaks(*values)
        text = SOURCE.read_text()
        self.assertIn('check_memory_peaks(torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved(),', text)

    def test_gradients_use_saved_ordinary_behavior_targets_and_new_e32_is_descriptive(self):
        tree = ast.parse(SOURCE.read_text(), feature_version=(3, 10))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        saved_loader = next(node for node in run.body if isinstance(node, ast.FunctionDef) and
                            node.name == 'load_saved_behavior')
        loader_text = ast.get_source_segment(SOURCE.read_text(), saved_loader)
        self.assertIn("sha256(path) == supplied[name + '_sha256']", loader_text)
        self.assertIn('value.dtype == np.float32 and value.shape == shape', loader_text)
        self.assertIn('torch.from_numpy(np.ascontiguousarray(value).copy()).clone()', loader_text)
        self.assertIn('not torch.is_inference(tensors[name])', loader_text)
        self.assertNotIn('inference_mode(', loader_text)
        self.assertNotIn('no_grad(', loader_text)
        text = SOURCE.read_text()
        gradient_branch = text[text.index("if mode == 'TBF16' and"):text.index("if state_label == 'original' and case.get('behavior'):")]
        self.assertIn("anchor = saved_behaviors[case['id']]", gradient_branch)
        self.assertNotIn('original_behaviors', gradient_branch)
        self.assertIn("'gradient_targets_are_exact_saved_V2_anchors_not_new_E32': True", text)

    def test_exact_fixed_case_and_gradient_call_census(self):
        cases = [(update, slot) for update in (1, 24) for slot in range(1, 5)]
        base = 2 * len(cases) * len(neural.MODES)
        repeat = 2 * len(neural.REPEAT_POSITIONS) * len(neural.MODES)
        intervention = len(cases) * 3
        self.assertEqual((base, repeat, intervention, base + repeat + intervention), (64, 16, 24, 104))
        terms, lpips = [], 0
        for update, slot in sorted(neural.GRADIENT_POSITIONS):
            eligible = slot >= 3 or update == 24
            terms.append(len(neural.expected_terms(eligible, slot == 3)))
            lpips += int(eligible)
        self.assertEqual(terms, [3, 7, 5, 5, 7, 5])
        self.assertEqual(sum(terms) * 2, 64)
        self.assertEqual(lpips * 2, 10)

    def test_weighted_loss_component_retains_divide_four(self):
        self.assertEqual(neural.weighted_component(2., 'token_ce'), .25)
        self.assertEqual(neural.weighted_component(.2, 'coarse_rgb_l1'), .2)
        self.assertAlmostEqual(neural.weighted_component(.8, 'behavior_token_kl'), .02, places=15)
        with self.assertRaises(ValueError):
            neural.weighted_component(float('nan'), 'token_ce')
        with self.assertRaises(ValueError):
            neural.weighted_component(1., 'invented_term')

    def test_cosine_algebra_separates_absent_parallel_opposed_and_orthogonal(self):
        self.assertIsNone(neural.cosine_from_sufficient_statistics(0., 0., 1.))
        self.assertEqual(neural.cosine_from_sufficient_statistics(6., 4., 9.), 1.)
        self.assertEqual(neural.cosine_from_sufficient_statistics(-6., 4., 9.), -1.)
        self.assertEqual(neural.cosine_from_sufficient_statistics(0., 4., 9.), 0.)
        with self.assertRaises(ValueError):
            neural.cosine_from_sufficient_statistics(2., 1., 1.)
        with self.assertRaises(ValueError):
            neural.cosine_from_sufficient_statistics(0., -1., 1.)

    def test_scoped_intervention_restores_function_on_error(self):
        original = lambda x: x
        owner = SimpleNamespace(function=original)
        replacement = lambda x: x + 1
        with self.assertRaises(RuntimeError):
            with neural.temporary_attribute(owner, 'function', replacement):
                self.assertIs(owner.function, replacement)
                raise RuntimeError('owned algebra fixture')
        self.assertIs(owner.function, original)

    def test_no_optimizer_backward_accumulation_or_local_nn_test(self):
        text = SOURCE.read_text()
        tree = ast.parse(text)
        self.assertNotIn('.backward(', text)
        self.assertNotIn('torch.optim.', text)
        self.assertNotIn('optimizer.step(', text)
        self.assertIn('torch.autograd.grad(', text)
        self.assertIn('allow_unused=True', text)
        self.assertIn('retain_graph=(index != len(names) - 1)', text)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and node.func.attr == 'load' and
                 isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch']
        self.assertEqual(len(calls), 1)
        values = {node.arg: node.value for node in calls[0].keywords}
        self.assertTrue(ast.literal_eval(values['weights_only']))
        self.assertEqual(ast.literal_eval(values['map_location']), 'cpu')

    def test_actual_top1_is_captured_before_codebook_not_fp32_argmax(self):
        text = SOURCE.read_text()
        self.assertIn("temporary_attribute(net.quantize, 'get_codebook_feat', lookup)", text)
        self.assertIn("captured['selected'] = indices.detach().cpu().clone()", text)
        self.assertIn('native_probabilities = F.softmax(logits, dim=2)', text)
        self.assertIn('torch.topk(native_probabilities, 1, dim=2).indices', text)
        self.assertNotIn('argmax(', text)
        self.assertIn('selected_index_not_top2_first_due_to_ties', text)

    def test_hooks_and_adain_and_fusion_are_restored_after_interventions(self):
        text = SOURCE.read_text()
        self.assertIn('hook.remove()', text)
        self.assertIn('architecture.adaptive_instance_normalization = original_adain_function', text)
        self.assertIn('stopped.fuse_convs_dict.load_state_dict(stopped_fusion, strict=True)', text)
        self.assertIn("stage_hashes('after_all_interventions_restored')", text)
        self.assertIn('Intervention unexpectedly changed hard-selected code indices', text)
        self.assertIn('Measurement populated a parameter.grad field', text)

    def test_call_reservation_is_persisted_before_neural_invocation(self):
        text = SOURCE.read_text()
        self.assertLess(text.index("reserve('codeformer_forwards', context)"), text.index('native, logits, feature = net('))
        self.assertLess(text.index("reserve('gradient_queries'"), text.index('torch.autograd.grad('))
        self.assertIn("write_new_json(out / 'call_trace'", text)
        self.assertIn('require(counts == caps', text)


if __name__ == '__main__':
    unittest.main()
