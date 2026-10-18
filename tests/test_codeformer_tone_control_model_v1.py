"""NumPy/AST/guard tests only; no Torch/model/autograd/optimizer execution."""
import ast
import builtins
import hashlib
import inspect
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.codeformer_tone_control_training_v1 import model as m
from research.codeformer_tone_control_training_v1 import data


ROOT = Path(__file__).resolve().parents[1]


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def fixture(clipped=False):
    mask = np.ones((2, 2), np.bool_)
    if clipped:
        mask[0, 1] = False
        base = np.broadcast_to(np.asarray([.95, .30, .10], np.float32), (2, 2, 3)).copy()
        return base, mask, float(np.float32(.6)), math.atanh(.4)
    return np.full((2, 2, 3), .4, np.float32), mask, float(np.float32(.45)), 0.


def reference_loss(base, mask, target, logit):
    delta = .25 * math.tanh(logit)
    corrected = np.clip(np.asarray(base, np.float64) + delta, 0, 1)
    y = np.asarray([.2126, .7152, .0722], np.float64)
    mean = float((corrected[mask] @ y).mean())
    return (mean - target)**2


class AnalyticControllerMath(unittest.TestCase):
    def test_both_actual_planned_fixtures_match_finite_difference(self):
        hidden = np.linspace(-.2, .8, 32)
        epsilon = 1e-6
        for clipping in (False, True):
            base, mask, target, logit = fixture(clipping)
            gradients, row = m.analytic_fixture_gradients(base, mask, target, logit, hidden)
            finite = (reference_loss(base, mask, target, logit + epsilon) -
                      reference_loss(base, mask, target, logit - epsilon)) / (2 * epsilon)
            self.assertAlmostEqual(float(gradients['output.bias'][0]), finite, places=10)
            np.testing.assert_allclose(gradients['output.weight'], finite * hidden.reshape(1, 32), rtol=1e-8, atol=1e-11)
            self.assertAlmostEqual(row['loss'], reference_loss(base, mask, target, logit), places=15)

    def test_hidden_layer_is_connected_zero_not_None_in_these_fixtures(self):
        for clipped in (False, True):
            gradients, _ = m.analytic_fixture_gradients(*fixture(clipped), np.ones(32))
            for name, shape in [('hidden.weight', (32, 256)), ('hidden.bias', (32,))]:
                self.assertIsNotNone(gradients[name])
                self.assertEqual(gradients[name].shape, shape)
                self.assertEqual(gradients[name].dtype, np.dtype('float64'))
                self.assertEqual(np.count_nonzero(gradients[name]), 0)

    def test_clipping_removes_only_red_channel_derivative_on_observed_pixels(self):
        base, mask, target, logit = fixture(True)
        _, row = m.analytic_fixture_gradients(base, mask, target, logit, np.ones(32))
        self.assertAlmostEqual(row['d_mean_d_delta'], .7152 + .0722, places=15)
        self.assertAlmostEqual(row['clipped_scalar_fraction'], 1 / 3, places=15)
        unclipped_wrong = 2 * (row['mean'] - target) * .25 * (1 - .4**2)
        self.assertGreater(abs(row['d_loss_d_logit'] - unclipped_wrong), .001)

    def test_unobserved_values_do_not_change_fixture_mean_or_derivative(self):
        base, mask, target, logit = fixture(True)
        first, scalar = m.analytic_fixture_gradients(base, mask, target, logit, np.ones(32))
        changed = base.copy(); changed[~mask] = [.4, .6, .7]
        second, other = m.analytic_fixture_gradients(changed, mask, target, logit, np.ones(32))
        self.assertEqual(scalar['mean'], other['mean'])
        for name in first: np.testing.assert_array_equal(first[name], second[name])

    def test_fixture_rejects_clamp_kink_and_invalid_support(self):
        base = np.full((2, 2, 3), .5)
        with self.assertRaisesRegex(ValueError, 'kinks'):
            m.analytic_fixture_gradients(base * 0, np.ones((2, 2), bool), .4, 0., np.ones(32))
        for mask in (np.zeros((2, 2), bool), np.ones((2, 2), np.uint8)):
            with self.subTest(dtype=str(mask.dtype)), self.assertRaises(ValueError):
                m.analytic_fixture_gradients(base, mask, .4, 0., np.ones(32))

    def test_native_dtype_and_feature_count_bounds_are_fixed(self):
        self.assertEqual(m.HEAD_PARAMETERS, 256 * 32 + 32 + 32 + 1)
        self.assertEqual(m.HEAD_PARAMETERS, 8257)
        self.assertEqual(m.OFFSET_SCALE, .25)
        self.assertEqual(m.GRADIENT_ATOL, 2e-7)
        self.assertEqual(m.GRADIENT_RTOL, 2e-5)


class SharedObservedFeatures(unittest.TestCase):
    def test_model_feature_wrapper_delegates_only_to_data(self):
        rgb = np.zeros((256, 256, 3), np.float32); mask = np.ones((256, 256), bool)
        marker = np.arange(256, dtype=np.float32)
        with patch.object(data, 'features', return_value=marker) as shared:
            self.assertIs(m.features_numpy(rgb, mask), marker)
        shared.assert_called_once_with(rgb, mask)

    def test_plane_order_and_support_normalized_values(self):
        rgb = np.full((256, 256, 3), [.25, .5, .75], np.float32)
        mask = np.ones((256, 256), bool)
        mask[:32, :16] = False
        features = m.features_numpy(rgb, mask).reshape(4, 8, 8)
        np.testing.assert_array_equal(features[:3], np.broadcast_to(np.array([.25, .5, .75], np.float32)[:, None, None], (3, 8, 8)))
        self.assertEqual(features[3, 0, 0], .5)
        self.assertEqual(features[3, 0, 1], 1.)
        self.assertEqual(features.dtype, np.float32)

    def test_empty_cell_and_protected_RGB_do_not_create_features(self):
        rgb = np.full((256, 256, 3), .3, np.float32)
        mask = np.ones((256, 256), bool); mask[:32, :32] = False
        first = m.features_numpy(rgb, mask)
        rgb[:32, :32] = [.9, .1, .5]
        second = m.features_numpy(rgb, mask)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.reshape(4, 8, 8)[:, 0, 0], np.zeros(4, np.float32))

    def test_shared_features_reject_invalid_type_range_mask(self):
        rgb = np.full((256, 256, 3), .3, np.float32); mask = np.ones((256, 256), bool)
        for image, support in [(rgb.astype(np.float64), mask), (rgb, mask.astype(np.uint8)),
                               (rgb, np.zeros_like(mask)), (rgb + 1, mask)]:
            with self.assertRaises(ValueError): m.features_numpy(image, support)


class GuardAndSourceBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(m.__file__).read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source, feature_version=(3, 10))

    def test_module_import_and_CPU_tests_never_import_Torch(self):
        topimports = []
        for node in self.tree.body:
            if isinstance(node, ast.Import): topimports.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom): topimports.append(node.module or '')
        self.assertTrue(all(not name.startswith(('torch', 'research.')) for name in topimports))
        torch_imports = [node for node in ast.walk(self.tree) if isinstance(node, ast.Import) and
                         any(alias.name == 'torch' for alias in node.names)]
        self.assertEqual(len(torch_imports), 1)
        activation = function(self.tree, 'activate_runtime')
        self.assertTrue(activation.lineno < torch_imports[0].lineno <= activation.end_lineno)
        self.assertNotIn('torch', sys.modules)

    def test_windows_refuses_before_Torch_or_filesystem_use(self):
        calls = []; original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                calls.append(name); raise AssertionError('Torch import forbidden in local fixtures')
            return original_import(name, *args, **kwargs)
        with patch.object(m.platform, 'system', return_value='Windows'), patch('builtins.__import__', side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, 'manual Linux L4'):
                m.activate_runtime('does-not-exist', '0' * 64)
        self.assertEqual(calls, [])

    def test_all_Torch_helpers_refuse_before_any_model_execution(self):
        self.assertIsNone(m._RUNTIME)
        for helper, args in [(m.create_head, ()), (m.load_frozen_parents, ('x', 'y')),
                             (m.ordinary_tensor, (None, 'rgb')), (m.mean_y, (None, None)),
                             (m.tensor_state_digest, ({},)), (m.capture_rng, ())]:
            with self.subTest(helper=helper.__name__), self.assertRaisesRegex(ValueError, 'Activate'):
                helper(*args)
        self.assertNotIn('torch', sys.modules)

    def test_head_source_initialization_RNG_restore_and_four_tensor_architecture(self):
        source = ast.unparse(function(self.tree, 'create_head'))
        for required in ['torch.nn.Linear(256, 32, bias=True)', 'torch.nn.Linear(32, 1, bias=True)',
                         'torch.nn.SiLU()', 'torch.nn.init.zeros_(self.output.weight)',
                         'torch.nn.init.zeros_(self.output.bias)', 'torch.random.fork_rng(devices=[0], enabled=True)',
                         'torch.manual_seed(HEAD_SEED)', 'OFFSET_SCALE * torch.tanh(logit)']:
            self.assertIn(required, source)
        self.assertIn('torch.equal(before_cpu, torch.get_rng_state())', source)

    def test_factory_source_hashes_before_load_and_freezes_every_parent(self):
        source = ast.unparse(function(self.tree, 'load_frozen_parents'))
        self.assertLess(source.index('sha256(path) == expected'), source.index('from research.codeformer_adaptation_v2'))
        self.assertIn("weights_only=True, mmap=True", source)
        self.assertIn("map_location='cpu'", source)
        self.assertIn("loaded.get('completed_updates') == 50", source)
        self.assertIn('base.load_state_dict(values, strict=True)', source)
        self.assertIn('requires_grad_(False)', source)
        self.assertNotIn('torch.optim', source)
        self.assertIn('ORIGINAL_MODEL_DIGEST', source)
        self.assertIn('STOPPED_MODEL_DIGEST', source)

    def test_canonical_state_digest_matches_historical_algorithm_source(self):
        source = ast.unparse(function(self.tree, 'tensor_state_digest'))
        self.assertIn("len(header).to_bytes(8, 'big')", source)
        self.assertIn('value.reshape(-1).view(torch.uint8).numpy().tobytes()', source)
        self.assertEqual(m.ORIGINAL_MODEL_DIGEST, '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd')
        self.assertEqual(m.STOPPED_MODEL_DIGEST, '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b')

    def test_training_and_prerequisites_share_actual_clamp_mean_loss_helpers(self):
        train = ast.unparse(function(self.tree, 'training_loss'))
        composite = ast.unparse(function(self.tree, 'composite_raw'))
        prereq = ast.unparse(function(self.tree, 'head_prerequisites'))
        self.assertIn('corrected, delta, predicted_mean = apply_head(', train)
        self.assertIn('squared_mean_loss(predicted_mean, target)', train)
        self.assertIn('corrected = composite_raw(base, fixture_input, observed, delta)', prereq)
        self.assertIn('predicted = mean_y(corrected, observed)', prereq)
        self.assertIn('squared_mean_loss(predicted, target)', prereq)
        self.assertIn('torch.where(observed, corrected, input_rgb)', composite)
        self.assertIn('.clamp(0, 1)', composite)
        self.assertNotIn('clean', inspect.signature(m.training_loss).parameters)

    def test_prerequisite_counts_arrays_and_exact_restoration_are_explicit(self):
        source = ast.unparse(function(self.tree, 'head_prerequisites'))
        self.assertEqual(source.count("reserve('controller_math_prerequisites')"), 1)
        self.assertEqual(source.count("reserve('head_autograd_queries')"), 1)
        self.assertEqual(source.count('head(features, return_details=True)'), 1)
        self.assertEqual(source.count('torch.autograd.grad('), 1)
        self.assertIn('for label, base_np, target_value, bias in fixtures:', source)
        self.assertIn('restore_rng(before_rng)', source)
        self.assertIn('head.load_state_dict(before_state, strict=True)', source)
        self.assertIn("write_json('prerequisites.json', report)", source)
        self.assertIn("'expected_connected_zero': name.startswith('hidden.')", source)
        self.assertIn('protected_exact', source)
        self.assertIn('parameter.grad = None', source)

    def test_zero_head_RAW_identity_scope_does_not_claim_all_pixels(self):
        self.assertIs(m.ZERO_HEAD_IDENTITY['full_frame_corrected_RAW_equals_unprotected_historical_RAW_claimed'], False)
        self.assertIn('observed', m.ZERO_HEAD_IDENTITY['corrected_RAW'])
        self.assertIn('input', m.ZERO_HEAD_IDENTITY['corrected_RAW'])
        self.assertIn('all pixels', m.ZERO_HEAD_IDENTITY['delivered_PNG'])


if __name__ == '__main__':
    unittest.main()
