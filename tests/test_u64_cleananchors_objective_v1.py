import ast
import inspect
import unittest
import numpy as np
from research.codeformer_u64_cleananchors_continuation_v1 import objective as o


class PhotographicObjective(unittest.TestCase):
    def arrays(self, value=0.0):
        return np.full((256, 256, 3), value, dtype=np.float64)

    def test_equal_photograph_zero_every_role(self):
        p = np.random.default_rng(101).random((256, 256, 3))
        for role in o.ROLES:
            self.assertEqual(o.numpy_objective(p, p.copy(), role, np=np)[0], 0.0)

    def test_clear_does_not_copy_wrong_restored_teacher(self):
        prediction, photograph = self.arrays(0.3), self.arrays(0.1)
        previous_teacher_loss = 0.0  # prediction equals its old generated teacher
        current, _ = o.numpy_objective(prediction, photograph, 'clear', np=np)
        self.assertEqual(previous_teacher_loss, 0)
        self.assertAlmostEqual(current, 0.39, places=12)

    def test_clear_quadratic_branch(self):
        current, _ = o.numpy_objective(self.arrays(0.104), self.arrays(0.1), 'clear', np=np)
        self.assertAlmostEqual(current, 0.0016, places=12)

    def test_clear_loss_gradient_points_to_photograph(self):
        t = self.arrays(0.1)
        h = 1e-5
        loss = lambda v: o.numpy_objective(self.arrays(v), t, 'clear', np=np)[0]
        self.assertAlmostEqual((loss(0.3+h)-loss(0.3-h))/(2*h), 2.0, places=8)
        self.assertLess(loss(0.299), loss(0.3))

    def test_global_blur_criterion_unchanged(self):
        loss, terms = o.numpy_objective(self.arrays(0.3), self.arrays(0.1), o.ROLES[1], np=np)
        self.assertAlmostEqual(loss, 0.2, places=12)
        self.assertEqual(terms['target_adjacent_L1'], 0)

    def test_local_boundary_edges_are_excluded(self):
        t = self.arrays()
        p = t.copy()
        p[64:192, 64:192] = 0.2
        loss, terms = o.numpy_objective(p, t, o.ROLES[0], np=np)
        self.assertAlmostEqual(loss, 0.1, places=12)
        self.assertEqual(terms['inside_adjacent_L1'], 0)
        self.assertEqual(terms['outside_adjacent_L1'], 0)

    def test_local_regions_keep_equal_weight_despite_area(self):
        p = self.arrays(0.2)
        p[64:192, 64:192] = 0.4
        loss, _ = o.numpy_objective(p, self.arrays(), o.ROLES[0], np=np)
        self.assertAlmostEqual(loss, 0.3, places=12)

    def test_dark_reference_uses_photographic_luminance(self):
        loss, terms = o.numpy_objective(self.arrays(0.3), self.arrays(0.1), o.ROLES[3], np=np)
        self.assertAlmostEqual(loss, 0.59, places=12)
        self.assertAlmostEqual(terms['photographic_block16_luminance_anchor_L1'], 0.2, places=12)

    def test_dark_pooling_respects_block16(self):
        p, t = self.arrays(0.1), self.arrays(0.1)
        p[:16, :16] = 0.3
        _, terms = o.numpy_objective(p, t, o.ROLES[3], np=np)
        self.assertAlmostEqual(terms['photographic_block16_luminance_anchor_L1'], 0.2/256, places=12)

    def test_invalid_role_and_legacy_dark_target_refused(self):
        for role in ('dark', 'native', 'FINAL'):
            with self.assertRaises(ValueError):
                o.numpy_objective(self.arrays(), self.arrays(), role, np=np)
        with self.assertRaises(ValueError):
            o.numpy_objective(self.arrays(), None, o.ROLES[3], np=np)

    def test_invalid_shapes_dtype_range_and_nonfinite_refused(self):
        cases = [np.zeros((16, 16, 3)), np.zeros((256, 256, 3), dtype=np.uint8),
                 self.arrays(-0.1), self.arrays(1.1), self.arrays(np.nan), self.arrays(np.inf)]
        for value in cases:
            with self.assertRaises(ValueError):
                o.numpy_objective(value, self.arrays(), 'clear', np=np)
            with self.assertRaises(ValueError):
                o.numpy_objective(self.arrays(), value, 'clear', np=np)

    def test_input_arrays_immutable(self):
        p, t = self.arrays(0.3), self.arrays(0.1)
        p.flags.writeable = False
        t.flags.writeable = False
        for role in o.ROLES:
            o.numpy_objective(p, t, role, np=np)
        self.assertTrue(np.all(p == 0.3) and np.all(t == 0.1))

    def test_connected_source_has_no_detach_or_generated_teacher_argument(self):
        src = inspect.getsource(o.connected_objective)
        names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
        self.assertNotIn('detach', names)
        self.assertEqual(tuple(inspect.signature(o.connected_objective).parameters),
                         ('prediction', 'target', 'role', 'torch'))
        self.assertIn('grad_fn', names)

    def test_import_is_inert_and_python310_syntax(self):
        tree = ast.parse(inspect.getsource(o), feature_version=(3, 10))
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body))


if __name__ == '__main__':
    unittest.main()
