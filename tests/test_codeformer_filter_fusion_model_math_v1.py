"""No Torch/NN execution: independent scalar oracles and static route guards."""
import ast
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_filter_fusion_training_v1/model.py'
SPEC = importlib.util.spec_from_file_location('filter_fusion_model_math_under_test', SOURCE)
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


class MathContracts(unittest.TestCase):
    def loss(self, p, q, role='filter', footprint=None):
        return MODEL.numpy_loss_terms(p, q, role=role, footprint=footprint, np=np)

    def test_import_never_imports_torch(self):
        code = "import runpy,sys;runpy.run_path(%r);assert 'torch' not in sys.modules" % str(SOURCE)
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_schema_not_prefix_only(self):
        shapes = MODEL.expected_fusion_shapes()
        self.assertEqual(len(shapes), 72)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), 10659584)
        self.assertEqual({site: sum(MODEL.fusion_site(name) == site for name in shapes)
                          for site in ('32', '64', '128', '256')}, dict.fromkeys(MODEL.SITES, 18))
        with self.assertRaises(ValueError):
            MODEL.fusion_site('fuse_convs_dict.32.unexpected.weight')

    def test_constant_residual_gradient_zero_not_image_gradient(self):
        target = np.arange(36).reshape(3, 3, 4) / 100
        observed = self.loss(target + .2, target)
        self.assertAlmostEqual(observed['unweighted_terms']['spatial_rgb_l1'], .2)
        self.assertAlmostEqual(observed['unweighted_terms']['adjacent_residual_l1'], 0.)
        self.assertAlmostEqual(observed['divided_slot_scalar'], .05)

    def test_direction_means_are_separate_with_nonsquare_dimensions(self):
        # Residual x+2y has horizontal1,vertical2, so adjacent=(1+2)/2.
        y, x = np.mgrid[:3, :5]
        p = np.repeat((x + 2*y)[None, :, :], 3, axis=0)
        result = self.loss(p, np.zeros_like(p))
        self.assertEqual(result['unweighted_terms']['adjacent_residual_l1'], 1.5)
        self.assertEqual(result['unweighted_terms']['spatial_rgb_l1'], 4.)
        self.assertEqual(result['slot_scalar'], 4.375)
        self.assertEqual(result['divided_slot_scalar'], 1.09375)

    def test_local_regions_are_balanced_not_whole_image_weighted(self):
        p = np.ones((3, 4, 4)); p[:, 1:2, 1:2] = 9
        result = self.loss(p, np.zeros_like(p), footprint=[1, 1, 2, 2])
        self.assertEqual(result['unweighted_terms']['spatial_rgb_l1'], 5.)
        self.assertNotEqual(result['unweighted_terms']['spatial_rgb_l1'], float(p.mean()))

    def test_boundary_adjacent_edges_not_dropped(self):
        p = np.zeros((3, 4, 4)); p[:, 1:3, 1:3] = 1
        result = self.loss(p, np.zeros_like(p), footprint=[1, 1, 3, 3])
        # Four boundary crossings /12 adjacencies in each direction.
        self.assertAlmostEqual(result['unweighted_terms']['adjacent_residual_l1'], 1/3)
        self.assertEqual(result['unweighted_terms']['spatial_rgb_l1'], .5)
        self.assertAlmostEqual(result['divided_slot_scalar'], (.5 + .25/3)/4)

    def test_rgb_and_batch_region_denominators(self):
        p = np.zeros((2, 3, 2, 4)); p[0, 0, 0, 0] = 12
        result = self.loss(p, np.zeros_like(p), footprint=[0, 0, 1, 1])
        #12 divided by2 batches*3 channels inside, then .5 =>1.
        self.assertEqual(result['unweighted_terms']['spatial_rgb_l1'], 1.)

    def test_clear_and_dark_coefficients_no_silent_clean_term(self):
        p = np.full((3, 2, 3), 1.5); q = np.full_like(p, -.5)
        for role in ('clear', 'dark'):
            value = self.loss(p, q, role)
            self.assertEqual(value['unweighted_terms'], {'teacher_rgb_l1': 2.})
            self.assertEqual(value['slot_scalar'], 4.)
            self.assertEqual(value['divided_slot_scalar'], 1.)

    def test_initial_dark_anchor_zero_is_valid(self):
        q = np.arange(24).reshape(3, 2, 4) / 10
        value = self.loss(q.copy(), q.copy(), 'dark')
        self.assertEqual(value['slot_scalar'], 0.)
        summary = {'queried_tensors': 72, 'sites': {s: {'tensors': 18, 'none': 0,
            'nonzero': 0, 'l2': 0., 'maximum_absolute': 0.} for s in MODEL.SITES}}
        self.assertTrue(MODEL.validate_gradient_summary(summary, allow_all_zero=True)['pass'])
        with self.assertRaises(ValueError):
            MODEL.validate_gradient_summary(summary)

    def test_each_of_four_sites_must_be_connected(self):
        summary = {'queried_tensors': 72, 'sites': {s: {'tensors': 18, 'none': 0,
            'nonzero': 18, 'l2': 1., 'maximum_absolute': .5} for s in MODEL.SITES}}
        summary['sites']['32']['none'] = 1
        with self.assertRaises(ValueError):
            MODEL.validate_gradient_summary(summary, allow_all_zero=True)

    def test_unclipped_loss_retains_out_of_range_error(self):
        p = np.full((3, 2, 2), 1.5); q = np.ones_like(p)
        self.assertEqual(self.loss(p, q)['unweighted_terms']['spatial_rgb_l1'], .5)
        self.assertEqual(self.loss(np.clip(p, 0, 1), q)['slot_scalar'], 0.)

    def test_clamp_before_resize_not_after(self):
        # At exact2:1 align_corners=False bilinear center each output averages
        # its2x2 footprint. Native RGB[-1,1;0,0] gives clamp-before .25,
        # whereas no clamp then resize gives0 and clamp-after also gives0.
        rgb = np.array([[-1., 1.], [0., 0.]], dtype=np.float32)
        self.assertEqual(float(np.clip(rgb, 0, 1).mean()), .25)
        self.assertEqual(float(rgb.mean()), 0.)
        self.assertNotEqual(float(np.clip(rgb, 0, 1).mean()), float(np.clip(rgb.mean(), 0, 1)))

    def test_floored_png_is_not_round_to_nearest(self):
        rgb = np.array([.5, 1., 0., 127.9/255], dtype=np.float32)
        self.assertEqual(np.floor(rgb * 255).astype(np.uint8).tolist(), [127, 255, 0, 127])

    def test_footprint_rejects_invalid_bool_empty_or_full(self):
        for fp in ([True, 0, 1, 1], [0, 0, 0, 1], [0, 0, 256, 256], [-1, 0, 1, 1]):
            with self.assertRaises(ValueError):
                MODEL.validate_footprint(fp)

    def test_teacher_role_rejects_footprint(self):
        with self.assertRaises(ValueError):
            self.loss(np.ones((3, 2, 2)), np.zeros((3, 2, 2)), 'dark', [0, 0, 1, 1])

    def test_nonfinite_or_nongraph_role_rejected(self):
        p = np.ones((3, 2, 2)); p[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.loss(p, np.zeros_like(p))
        with self.assertRaises(ValueError):
            MODEL.role_term_names('token_ce')

    def test_rate_schedule_no_epoch_or_unbounded_extension(self):
        self.assertAlmostEqual(MODEL.learning_rate(1), 1e-6, places=20)
        self.assertEqual(MODEL.learning_rate(10), 1e-5)
        self.assertEqual(MODEL.learning_rate(11), 1e-5)
        self.assertEqual(MODEL.learning_rate(151), 1e-5)
        for value in (0, 152, 154, 157, True, 1.0):
            with self.assertRaises(ValueError):
                MODEL.learning_rate(value)

    def test_scalar_finite_difference_preserves_slot_divisor(self):
        # Positive constant residual and positive uniform perturbation:
        # d(globalL1/4)/d(uniform scalar)=1/4; adjacency remains0.
        q = np.zeros((3, 3, 5)); p = np.full_like(q, .3); epsilon = 1e-6
        derivative = (self.loss(p+epsilon, q)['divided_slot_scalar'] -
                      self.loss(p-epsilon, q)['divided_slot_scalar']) / (2*epsilon)
        self.assertAlmostEqual(derivative, .25, places=9)

    def test_static_candidate_route_does_not_disable_autograd(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        candidate = ast.get_source_segment(SOURCE.read_text(encoding='utf-8'), functions['forward_candidate'])
        forward = ast.get_source_segment(SOURCE.read_text(encoding='utf-8'), functions['forward_record'])
        self.assertIn('torch.is_grad_enabled()', candidate)
        self.assertNotIn('torch.no_grad()', candidate + forward)
        self.assertNotIn('torch.inference_mode()', candidate + forward)
        self.assertIn('detach_16=True, adain=True', forward)
        teacher = ast.get_source_segment(SOURCE.read_text(encoding='utf-8'), functions['teacher_target'])
        self.assertIn('with torch.no_grad():', teacher)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        self.assertFalse(any(isinstance(target, ast.Attribute) and target.attr == 'adaptive_instance_normalization'
                             for node in assignments for target in node.targets))

    def test_static_delivery_policy_matches_native_clamp_before_resize(self):
        source = SOURCE.read_text(encoding='utf-8'); tree = ast.parse(source)
        functions = {node.name: ast.get_source_segment(source, node)
                     for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertLess(functions['delivered_rgb'].index('.clamp(0, 1)'),
                        functions['delivered_rgb'].index('functional.interpolate'))
        self.assertNotIn('.clamp(', functions['unclipped_rgb'])
        self.assertIn('.floor()', functions['png_tensor'])
        self.assertNotIn('.round(', functions['png_tensor'])
        self.assertIn('mode=\'bilinear\', align_corners=False', functions['prepare_input'])

    def test_fixed_upstream_fingerprints_must_all_match(self):
        before = {'logits': 'a', 'lq': 'b', 'tokens': 'c'}
        self.assertTrue(MODEL.validate_upstream_readouts(before, dict(before))['pass'])
        for name in before:
            after = dict(before); after[name] = 'changed'
            with self.assertRaises(ValueError):
                MODEL.validate_upstream_readouts(before, after)

    def test_disjoint_half_pair_U_matches_independent_bilinear_coordinates(self):
        native=(np.arange(3*8*10,dtype=np.float32).reshape(3,8,10)-np.float32(100))/np.float32(37)
        actual=MODEL.numpy_unclipped_readout(native,np=np)
        rgb=(native.astype(np.float64)+1)/2
        reference=np.empty((3,4,5),np.float64)
        for channel in range(3):
            for y in range(4):
                for x in range(5):
                    source_y=2*(y+.5)-.5;source_x=2*(x+.5)-.5
                    y0,x0=int(source_y),int(source_x)
                    fy,fx=source_y-y0,source_x-x0
                    reference[channel,y,x]=(1-fy)*((1-fx)*rgb[channel,y0,x0]+fx*rgb[channel,y0,x0+1])+ \
                        fy*((1-fx)*rgb[channel,y0+1,x0]+fx*rgb[channel,y0+1,x0+1])
        error=float(np.abs(actual.astype(np.float64)-reference).max())
        self.assertLessEqual(error,MODEL.readout_parity_bound(float(np.abs(rgb).max())))
        self.assertEqual(actual.dtype,np.float32)

    def test_every_native_pixel_has_one_output_coefficient_one_eighth(self):
        baseline=np.zeros((3,4,6),np.float32)
        reference=MODEL.numpy_unclipped_readout(baseline,np=np)
        for y in range(4):
            for x in range(6):
                source=baseline.copy();source[0,y,x]=1
                difference=MODEL.numpy_unclipped_readout(source,np=np)-reference
                expected=np.zeros_like(reference);expected[0,y//2,x//2]=np.float32(.125)
                np.testing.assert_array_equal(difference,expected)

    def test_U_readout_remains_unclipped_and_preserves_constant_fields(self):
        native=np.full((1,3,6,10),4,np.float32)
        expected=np.full((1,3,3,5),2.5,np.float32)
        np.testing.assert_array_equal(MODEL.numpy_unclipped_readout(native,np=np),expected)

    def test_readout_bound_is_prospective_scaled_arithmetic_not_quality_tolerance(self):
        bound=MODEL.readout_parity_bound(2.)
        self.assertEqual(bound,8*(2**-23)*2)
        self.assertTrue(MODEL.validate_readout_difference(bound,2.)['pass'])
        with self.assertRaises(ValueError):MODEL.validate_readout_difference(bound*1.01,2.)
        with self.assertRaises(ValueError):MODEL.validate_readout_difference(float('nan'),2.)
        self.assertFalse(MODEL.validate_readout_difference(0.,1.)['delivered_quality_gate_tolerance_changed'])

    def test_optimizing_U_has_no_interpolation_backward_or_custom_style_hook(self):
        source=SOURCE.read_text(encoding='utf-8');tree=ast.parse(source)
        functions={node.name:ast.get_source_segment(source,node)for node in tree.body if isinstance(node,ast.FunctionDef)}
        self.assertNotIn('interpolate',functions['unclipped_rgb'])
        self.assertNotIn('no_grad',functions['unclipped_rgb'])
        self.assertNotIn('.detach()',functions['unclipped_rgb'])
        self.assertIn('with torch.no_grad():',functions['unclipped_reference_parity'])
        self.assertIn('with torch.no_grad():',functions['delivered_rgb'])
        self.assertIn('unclipped_reference_parity(native, U)',functions['forward_record'])


if __name__ == '__main__':
    unittest.main()
