import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v19_parity_diagnostic',
    ROOT / 'scripts/diagnose_cctv_dgp_input_selection_v19_parity.py')
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class DiagnosticTests(unittest.TestCase):
    def evidence(self):
        canonical = {'reference_PNG_equal': True, 'reference_raw_max_difference': 0.0}
        scalar = {'reference_PNG_equal': False, 'reference_raw_max_difference': 0.0}
        rows = [
            {'role': 'failed_development_case', 'routes': {'numpy_before_GPU': canonical.copy(), 'CUDA_scalar_division': scalar.copy()}},
            {'role': 'fixed_clear_preview', 'routes': {'numpy_before_GPU': canonical.copy(), 'CUDA_scalar_division': scalar.copy()}},
            {'role': 'training_parity_case', 'routes': {'numpy_before_GPU': scalar.copy(), 'CUDA_scalar_division': canonical.copy()}}]
        return rows, {'CUDA_matches_reciprocal_simulation': True, 'different_values_from_CPU_division': 126}, {
            'numpy_before_GPU': True, 'CUDA_scalar_division': True}

    def test_confirmation_requires_exact_baseline_and_stable_original_recipe(self):
        rows, scalar, stability = self.evidence()
        self.assertTrue(d.classify_normalization(rows, scalar, stability))
        for role, route, key, value in [
            (0, 'numpy_before_GPU', 'reference_PNG_equal', False),
            (0, 'CUDA_scalar_division', 'reference_PNG_equal', True),
            (1, 'numpy_before_GPU', 'reference_raw_max_difference', 2.01e-6),
            (2, 'CUDA_scalar_division', 'reference_PNG_equal', False)]:
            changed = copy.deepcopy(rows)
            changed[role]['routes'][route][key] = value
            self.assertFalse(d.classify_normalization(changed, scalar, stability))
        self.assertFalse(d.classify_normalization(rows, scalar, {**stability, 'CUDA_scalar_division': False}))
        self.assertFalse(d.classify_normalization(rows, {**scalar, 'different_values_from_CPU_division': 0}, stability))
        self.assertFalse(d.classify_normalization(rows, {**scalar, 'CUDA_matches_reciprocal_simulation': False}, stability))

    def test_partial_or_duplicate_role_evidence_cannot_confirm(self):
        rows, scalar, stability = self.evidence()
        with self.assertRaisesRegex(ValueError, 'Three fixed roles'):
            d.classify_normalization(rows[:2], scalar, stability)
        changed = copy.deepcopy(rows)
        changed[2]['role'] = 'failed_development_case'
        with self.assertRaises(ValueError):
            d.classify_normalization(changed, scalar, stability)

    def test_paths_and_unknown_tmux_state_are_rejected(self):
        for name in ['../outside', '/outside', 'C:/outside', 'a\\b', 'a//b', 'a/./b', '']:
            with self.assertRaises(ValueError):
                d.safe(ROOT / 'outputs', name)
        self.assertEqual(d.live_tmux_tasks('shell\tbash\t0\nlogs\ttail\t0'), [])
        self.assertEqual(d.live_tmux_tasks('busy\tpython\t0'), [{'session': 'busy', 'command': 'python'}])
        with self.assertRaises(ValueError):
            d.live_tmux_tasks('unrecognized')

    def test_no_training_calls_and_finite_process_group_stops(self):
        source = Path(SPEC.origin).read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 10))
        forbidden = {'backward', 'step', 'enable_vm_training', 'AdamW', 'Adam', 'SGD', 'grad'}
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(f, ast.Attribute) and f.attr in forbidden or
                             isinstance(f, ast.Name) and f.id in forbidden for f in calls))
        self.assertEqual((d.DESIGN['cases'], d.DESIGN['DGP_forwards'], d.DESIGN['other_network_forwards']), (3, 8, 0))
        self.assertEqual((d.DESIGN['worker_cap_seconds'], d.DESIGN['supervisor_cap_seconds'], d.DESIGN['export_cap_seconds']), (120, 240, 60))
        self.assertIn('start_new_session=True', source)
        self.assertIn('os.killpg(child.pid, signal.SIGINT)', source)
        self.assertIn('os.killpg(child.pid, signal.SIGKILL)', source)
        self.assertLess(source.index('require_vm(root)'), source.index('load_frozen_dgp_restorer(parent /'))


if __name__ == '__main__':
    unittest.main()
