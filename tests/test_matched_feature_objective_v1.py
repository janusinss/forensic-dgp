"""NumPy/reference and source guards only; no Torch or autograd execution."""

import ast
import importlib.abc
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest


FRAMEWORK_ROOTS = frozenset(('torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'jaxlib', 'keras'))


class FrameworkImportBlocker(importlib.abc.MetaPathFinder):
    """Fail the local test process immediately if any neural import is attempted."""

    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FRAMEWORK_ROOTS:
            self.attempted.append(fullname)
            raise AssertionError('Neural framework import forbidden in local objective tests: ' + fullname)
        return None


FRAMEWORK_BLOCKER = FrameworkImportBlocker()
sys.meta_path.insert(0, FRAMEWORK_BLOCKER)

import numpy as np

from research.codeformer_matched_feature_learning_v1 import objective as obj


ROOT = Path(__file__).resolve().parents[1]


class MatchedPhotographicObjective(unittest.TestCase):
    @staticmethod
    def arrays(value=0.0, dtype=np.float64):
        return np.full((256, 256, 3), value, dtype=dtype)

    def calculate(self, prediction, target, role='clear'):
        return obj.objective_numpy(prediction, target, role, np=np)

    def test_roles_match_every_five_role_photograph_in_candidate_data(self):
        data = json.loads((ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation/candidate_data_v1.json').read_text())
        self.assertEqual(tuple(dict.fromkeys(row['role'] for row in data['cases'])), obj.ROLES)
        by_reference = {}
        for row in data['cases']:
            by_reference.setdefault(row['reference_id'], []).append(row['role'])
        self.assertEqual(len(by_reference), 32)
        self.assertTrue(all(tuple(roles) == obj.ROLES for roles in by_reference.values()))
        self.assertEqual(obj.ROLE_COUNT, len(obj.ROLES))

    def test_equal_nonuniform_photograph_zero_every_role(self):
        photograph = np.random.default_rng(104).random((256, 256, 3))
        for role in obj.ROLES:
            with self.subTest(role=role):
                loss, terms = self.calculate(photograph.copy(), photograph, role)
                self.assertEqual(loss, 0.0)
                self.assertEqual(tuple(terms), obj.TERM_NAMES)
                self.assertTrue(all(value == 0 for value in terms.values()))

    def test_constant_offset_has_analytic_RGB_and_coarse_loss_every_role(self):
        for role in obj.ROLES:
            with self.subTest(role=role):
                loss, terms = self.calculate(self.arrays(.5), self.arrays(.25), role)
                self.assertEqual(terms, {'RGB_L1': .25, 'adjacent_residual_L1': 0.,
                                         'block16_RGB_mean_residual_L1': .25})
                self.assertEqual(loss, .3125)

    def test_maximum_constant_offset_is_not_clipped_or_divided(self):
        loss, terms = self.calculate(self.arrays(1), self.arrays())
        self.assertEqual(terms['RGB_L1'], 1.)
        self.assertEqual(terms['block16_RGB_mean_residual_L1'], 1.)
        self.assertEqual(loss, 1.25)

    def test_internal_impulse_analytic_four_edges_and_one_block(self):
        prediction = self.arrays()
        prediction[40, 72, :] = 1.
        loss, terms = self.calculate(prediction, self.arrays())
        expected_rgb = 1 / (256 * 256)
        expected_edges = 4 / (2 * 256 * 255)
        self.assertAlmostEqual(terms['RGB_L1'], expected_rgb, places=15)
        self.assertAlmostEqual(terms['adjacent_residual_L1'], expected_edges, places=15)
        self.assertAlmostEqual(terms['block16_RGB_mean_residual_L1'], expected_rgb, places=15)
        self.assertAlmostEqual(loss, expected_rgb * 1.25 + .25 * expected_edges, places=15)

    def test_corner_impulse_has_two_edges_without_padding(self):
        prediction = self.arrays()
        prediction[0, 0, 0] = 1.
        _, terms = self.calculate(prediction, self.arrays())
        self.assertAlmostEqual(terms['adjacent_residual_L1'], 2 / (2 * 256 * 255 * 3), places=15)
        self.assertAlmostEqual(terms['RGB_L1'], 1 / (256 * 256 * 3), places=15)

    def test_alternating_signed_columns_cancel_only_within_coarse_RGB(self):
        target = self.arrays(.5)
        prediction = self.arrays(.25)
        prediction[:, 1::2] = .75
        loss, terms = self.calculate(prediction, target)
        self.assertEqual(terms, {'RGB_L1': .25, 'adjacent_residual_L1': .25,
                                'block16_RGB_mean_residual_L1': 0.})
        self.assertEqual(loss, .3125)

    def test_alternating_checkerboard_uses_both_edge_directions(self):
        yy, xx = np.indices((256, 256))
        prediction = np.repeat(((xx + yy) % 2)[:, :, None], 3, axis=2).astype(np.float64)
        loss, terms = self.calculate(prediction, self.arrays(.5))
        self.assertEqual(terms, {'RGB_L1': .5, 'adjacent_residual_L1': 1.,
                                'block16_RGB_mean_residual_L1': 0.})
        self.assertEqual(loss, .75)

    def test_edges_measure_residual_not_prediction_edges(self):
        yy, xx = np.indices((256, 256))
        target = np.repeat(((xx + yy) % 2 * .5)[:, :, None], 3, axis=2)
        loss, terms = self.calculate(target + .25, target)
        self.assertEqual(terms['adjacent_residual_L1'], 0.)
        self.assertEqual(loss, .3125)

    def test_coarse_preserves_RGB_channels_despite_zero_luminance_change(self):
        target = self.arrays(.5)
        prediction = target.copy()
        prediction[:, :, 0] += .125
        prediction[:, :, 1] -= .125 * .2126 / .7152
        _, terms = self.calculate(prediction, target)
        luminance_change = ((prediction - target) * np.asarray((.2126, .7152, .0722))).sum(axis=2)
        self.assertLess(float(np.abs(luminance_change).max()), 1e-15)
        self.assertGreater(terms['block16_RGB_mean_residual_L1'], .05)
        self.assertAlmostEqual(terms['block16_RGB_mean_residual_L1'], terms['RGB_L1'], places=12)

    def test_block16_means_do_not_cancel_across_distinct_blocks(self):
        target = self.arrays(.5)
        prediction = target.copy()
        prediction[:16, :16, :] = .75
        prediction[:16, 16:32, :] = .25
        _, terms = self.calculate(prediction, target)
        self.assertEqual(float((prediction - target).mean()), 0.)
        self.assertEqual(terms['block16_RGB_mean_residual_L1'], .5 / 256)

    def test_coarse_RGB_Jensen_bound_on_random_and_signed_fixtures(self):
        rng = np.random.default_rng(105)
        for index in range(6):
            prediction = rng.random((256, 256, 3))
            target = rng.random((256, 256, 3))
            _, terms = self.calculate(prediction, target)
            with self.subTest(index=index):
                self.assertLessEqual(terms['block16_RGB_mean_residual_L1'], terms['RGB_L1'] + 1e-15)
                self.assertLessEqual(terms['adjacent_residual_L1'], 2.)

    def test_local_role_uses_whole_image_and_cross_boundary_edges(self):
        prediction = self.arrays()
        prediction[64:192, 64:192, :] = 1.
        results = [self.calculate(prediction, self.arrays(), role) for role in obj.ROLES]
        self.assertTrue(all(result == results[0] for result in results))
        loss, terms = results[0]
        self.assertEqual(terms['RGB_L1'], .25)
        self.assertEqual(terms['block16_RGB_mean_residual_L1'], .25)
        self.assertAlmostEqual(terms['adjacent_residual_L1'], 512 / (2 * 256 * 255), places=15)
        self.assertGreater(loss, .3125)

    def test_five_role_external_division_matches_mean_once(self):
        values = [self.calculate(self.arrays(.5), self.arrays(.25), role)[0] for role in obj.ROLES]
        externally_divided = sum(value / obj.ROLE_COUNT for value in values)
        self.assertEqual(values, [.3125] * 5)
        self.assertEqual(externally_divided, .3125)

    def test_double_division_and_missing_division_fail_expected_microbatch_value(self):
        values = [self.calculate(self.arrays(.5), self.arrays(.25), role)[0] for role in obj.ROLES]
        expected = .3125
        missing = sum(values)
        twice = sum(value / obj.ROLE_COUNT / obj.ROLE_COUNT for value in values)
        self.assertEqual(missing, 5 * expected)
        self.assertAlmostEqual(twice, expected / 5, places=15)
        self.assertNotAlmostEqual(missing, expected, places=12)
        self.assertNotAlmostEqual(twice, expected, places=12)
        with self.assertRaises(TypeError):
            obj.objective_numpy(self.arrays(), self.arrays(), 'clear', np=np, divisor=5)

    def test_rejects_nonexact_or_unknown_roles(self):
        for role in ('CLEAR', 'clear ', 'local_gaussian_1p2', 'native', 'FINAL', None, True, 0, ['clear']):
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.calculate(self.arrays(), self.arrays(), role)

    def test_rejects_shapes_nonarrays_and_nonfloating_types_on_either_side(self):
        invalid = [None, [0.], np.zeros((1, 3, 256, 256)), np.zeros((256, 256)),
                   np.zeros((256, 255, 3)), self.arrays(dtype=np.uint8),
                   self.arrays(dtype=bool), self.arrays(dtype=np.complex128)]
        valid = self.arrays()
        for index, value in enumerate(invalid):
            for side in ('prediction', 'target'):
                with self.subTest(index=index, side=side), self.assertRaises(ValueError):
                    self.calculate(value if side == 'prediction' else valid,
                                   value if side == 'target' else valid)

    def test_rejects_nonfinite_or_out_of_range_single_elements_on_either_side(self):
        valid = self.arrays(.5)
        for value in (-.0001, 1.0001, float('nan'), float('inf'), -float('inf')):
            invalid = valid.copy()
            invalid[217, 111, 2] = value
            for side in ('prediction', 'target'):
                with self.subTest(value=value, side=side), self.assertRaises(ValueError):
                    self.calculate(invalid if side == 'prediction' else valid,
                                   invalid if side == 'target' else valid)

    def test_inputs_remain_immutable_and_float32_agrees_with_float64(self):
        rng = np.random.default_rng(106)
        prediction = rng.random((256, 256, 3), dtype=np.float32)
        target = rng.random((256, 256, 3), dtype=np.float32)
        originals = prediction.copy(), target.copy()
        prediction.flags.writeable = False
        target.flags.writeable = False
        actual = self.calculate(prediction, target)
        expected = self.calculate(prediction.astype(np.float64), target.astype(np.float64))
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(prediction, originals[0])
        np.testing.assert_array_equal(target, originals[1])

    def test_source_is_framework_free_and_connected_path_has_no_detach_or_hidden_division(self):
        tree = ast.parse(Path(obj.__file__).read_text(), feature_version=(3, 10))
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in ('objective_numpy', 'connected_objective'):
            self.assertFalse(any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                                 for node in ast.walk(functions[name])))
            self.assertEqual(tuple(inspect.signature(getattr(obj, name)).parameters),
                             ('prediction', 'target', 'role', 'np' if name == 'objective_numpy' else 'torch'))
        connected = functions['connected_objective']
        attributes = {node.attr for node in ast.walk(connected) if isinstance(node, ast.Attribute)}
        self.assertFalse({'detach', 'round', 'floor', 'clamp', 'backward', 'autograd', 'no_grad'} & attributes)
        self.assertIn('grad_fn', attributes)
        source = inspect.getsource(obj.connected_objective)
        self.assertIn('target.requires_grad or target.grad_fn is not None', source)
        self.assertIn('prediction.grad_fn is None', source)
        self.assertIn('mean(dim=(3, 5))', source)

    def test_contract_seals_common_unscaled_loss_and_redundant_coarse_term(self):
        contract = obj.formula_contract()
        self.assertTrue(contract['same_formula_for_both_routes_and_every_role'])
        self.assertTrue(contract['returned_objective_is_unscaled'])
        self.assertEqual(contract['microbatch_role_count'], 5)
        self.assertEqual(contract['coefficients'], dict(zip(obj.TERM_NAMES, (1., .25, .25))))
        self.assertIn('by five exactly once', contract['microbatch_loss_division'])
        self.assertIn('Jensen', contract['coarse_RGB_bound'])
        self.assertFalse(contract['separate_luminance_training_penalty'])
        self.assertFalse(contract['gradient_or_model_quality_verified'])
        self.assertFalse(contract['optimization_admitted'])

    def test_framework_blocker_retained_no_neural_module_loaded_or_attempted(self):
        self.assertIn(FRAMEWORK_BLOCKER, sys.meta_path)
        self.assertEqual(FRAMEWORK_BLOCKER.attempted, [])
        self.assertFalse(FRAMEWORK_ROOTS & {name.split('.')[0] for name in sys.modules})


if __name__ == '__main__':
    unittest.main()
