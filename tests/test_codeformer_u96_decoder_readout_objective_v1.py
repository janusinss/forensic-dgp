"""Pure loss-contract regressions; no neural/model/gradient operations."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_u96_decoder_readout_v1/objective.py'
spec = importlib.util.spec_from_file_location('inert_readout_objective', SOURCE)
objective = importlib.util.module_from_spec(spec)
spec.loader.exec_module(objective)


class ObjectiveContractTests(unittest.TestCase):
    def test_all_ordinary_roles_use_clean_target_rgb_and_structure(self):
        for role in objective.ROLES[1:]:
            self.assertEqual(objective.combine_terms(role, {'image': 2., 'adjacent': 4.}), 3.)

    def test_local_regions_are_equal_weight_even_when_sizes_differ(self):
        self.assertEqual(objective.combine_terms(objective.ROLES[0], {
            'image_inside': 4., 'image_outside': 2.,
            'adjacent_inside': 8., 'adjacent_outside': 4.}), 4.5)

    def test_copied_behavior_targets_are_not_the_prospective_objective(self):
        contract = objective.objective_contract()
        self.assertFalse(contract['U96_behavior_target_used_for_optimization'])
        self.assertFalse(contract['clean_target_replaced_by_generated_teacher'])
        self.assertFalse(contract['quality_or_preservation_thresholds_changed'])

    def test_constant_rgb_error_has_zero_structure_error(self):
        target = np.zeros((256, 256, 3), dtype=np.float32)
        prediction = np.full_like(target, .25)
        for role in objective.ROLES:
            terms = objective.numpy_clean_terms(prediction, target, role, np=np)
            self.assertEqual(objective.combine_terms(role, terms), .25)
            for key, value in terms.items():
                if key.startswith('adjacent'):
                    self.assertEqual(value, 0.)

    def test_local_boundary_crossing_edges_are_excluded(self):
        target = np.zeros((256, 256, 3), dtype=np.float64)
        prediction = target.copy()
        prediction[64:192, 64:192] = 1.
        terms = objective.numpy_clean_terms(prediction, target, objective.ROLES[0], np=np)
        self.assertEqual(terms, {'image_inside': 1., 'image_outside': 0.,
                                'adjacent_inside': 0., 'adjacent_outside': 0.})
        self.assertEqual(objective.combine_terms(objective.ROLES[0], terms), .5)
        self.assertGreater(objective.numpy_clean_terms(prediction, target, 'clear', np=np)['adjacent'], 0.)

    def test_saved_array_reference_refuses_bad_shape_or_range(self):
        zero = np.zeros((256, 256, 3), dtype=np.float64)
        for bad in (zero[:128], zero + 2, zero + float('nan'), zero.astype(np.uint8)):
            with self.assertRaises(ValueError):
                objective.numpy_clean_terms(bad, zero, 'clear', np=np)

    def test_bad_roles_missing_extra_or_nonfinite_terms_refused(self):
        cases = [('guessed_dark', {'image': 0., 'adjacent': 0.}),
                 ('clear', {'image': 0.}),
                 ('clear', {'image': 0., 'adjacent': 0., 'anchor': 0.}),
                 ('clear', {'image': float('nan'), 'adjacent': 0.}),
                 ('clear', {'image': 0., 'adjacent': float('inf')}),
                 ('clear', {'image': -1., 'adjacent': 0.}),
                 ('clear', {'image': True, 'adjacent': 0.})]
        for role, terms in cases:
            with self.subTest(role=role, terms=terms), self.assertRaises(ValueError):
                objective.combine_terms(role, terms)

    def test_objective_import_is_inert_and_tensor_path_has_no_detach(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf8'))
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual([n.names[0].name for n in imports], ['math'])
        tensor_path = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
                           n.name == 'connected_clean_objective')
        attrs = {n.attr for n in ast.walk(tensor_path) if isinstance(n, ast.Attribute)}
        self.assertFalse({'detach', 'floor', 'no_grad', 'inference_mode', 'backward', 'step'} & attrs)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
