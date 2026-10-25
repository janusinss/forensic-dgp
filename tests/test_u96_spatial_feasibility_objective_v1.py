import ast
import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

from research.codeformer_u96_spatial_feasibility_v1 import objective as obj


class Components(unittest.TestCase):
    def test_constant_tone_has_rgb_but_zero_adjacent(self):
        target = np.full((256, 256, 3), .25, np.float32)
        prediction = np.full_like(target, .5)
        for role in obj.ROLES:
            result, _ = obj.numpy_components(prediction, target, role, np=np)
            self.assertEqual(result, {'rgb': .25, 'adjacent': 0.0})

    def test_region_boundary_is_not_a_facial_edge_gain(self):
        target = np.zeros((256, 256, 3), np.float32)
        prediction = target.copy()
        prediction[64:192, 64:192] = .5
        local, _ = obj.numpy_components(prediction, target, obj.ROLES[0], np=np)
        whole, _ = obj.numpy_components(prediction, target, obj.ROLES[1], np=np)
        self.assertEqual(local, {'rgb': .25, 'adjacent': 0.0})
        self.assertGreater(whole['adjacent'], 0)

    def test_alternating_columns_analytic_error(self):
        target = np.zeros((256, 256, 3), np.float64)
        prediction = target.copy()
        prediction[:, 1::2] = 1
        result, _ = obj.numpy_components(prediction, target, obj.ROLES[1], np=np)
        self.assertEqual(result, {'rgb': .5, 'adjacent': .5})

    def test_nonuniform_target_is_used_directly(self):
        y, x = np.indices((256, 256))
        target = np.repeat(((x + y) / 510.)[:, :, None], 3, axis=2)
        result, _ = obj.numpy_components(target.copy(), target, obj.ROLES[3], np=np)
        self.assertEqual(result, {'rgb': 0., 'adjacent': 0.})

    def test_local_is_equal_regions_not_area_weighted(self):
        result = obj.split_scalar_terms(obj.ROLES[0], {
            'image_inside': .8, 'image_outside': .2,
            'adjacent_inside': .4, 'adjacent_outside': .1})
        self.assertEqual(result, {'rgb': .5, 'adjacent': .25})

    def test_rejects_unknown_roles_shapes_types_and_ranges(self):
        valid = np.zeros((256, 256, 3), np.float32)
        invalid = [valid.astype(np.uint8), valid[:128], valid + float('nan'), valid - .1, valid + 1.1]
        for bad in invalid:
            with self.assertRaises(ValueError):
                obj.numpy_components(bad, valid, obj.ROLES[2], np=np)
        with self.assertRaises(ValueError):
            obj.numpy_components(valid, valid, 'native_CCTV', np=np)
        with self.assertRaises(ValueError):
            obj.split_scalar_terms(obj.ROLES[2], {'image': True, 'adjacent': 0.})

    def test_static_tensor_source_does_not_detach_or_floor_prediction(self):
        text = Path(obj.__file__).read_text()
        tree = ast.parse(text, feature_version=(3, 10))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                        node.name == 'connected_components')
        attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
        self.assertFalse({'detach', 'floor', 'round', 'no_grad'} & attributes)
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(len(obj.TASK_NAMES), 8)


if __name__ == '__main__':
    unittest.main()
