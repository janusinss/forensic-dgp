"""Independent saved-pixel auditor math/source/geometry regression fixtures."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_train_v2_audit',
    ROOT / 'scripts/audit_codeformer_filter_train_v2_pixels.py')
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)


def profile(name):
    return next(copy.deepcopy(p) for p in A.PROFILES if p['id'] == name)


class IndependentTrainingPixelFixtures(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((256, 256), dtype=np.uint8)
        self.image = np.stack((xx, yy, np.bitwise_xor(xx, yy)), axis=-1)

    def test_literal_profile_census_and_strong_tag(self):
        self.assertEqual(len(A.PROFILES), 8)
        self.assertEqual([p['id'] for p in A.PROFILES if p['strong_control']],
                         ['repeated_rows_4_blend0p75'])
        self.assertEqual(len(A.REVIEW_ORDER), 8)
        self.assertEqual(A.REVIEW_ORDER[-1], 'clear')

    def test_source_hash_rectangle_rule_and_fixed_size(self):
        source = hashlib.sha256(b'fixture-source').hexdigest()
        raw = hashlib.sha256(('codeformer-filter-train-v2/rectangle/' + source).encode()).digest()
        expected = [raw[0] % 9 * 16, raw[1] % 9 * 16]
        self.assertEqual(A.rectangle_for_source(source), expected + [expected[0] + 128, expected[1] + 128])
        for invalid in ('A' * 64, '0' * 63, '../file', None):
            with self.assertRaises(ValueError):
                A.rectangle_for_source(invalid)

    def test_invalid_rectangles_are_rejected(self):
        for rect in ([1, 0, 129, 128], [0, 0, 127, 128], [-16, 0, 112, 128],
                     [144, 0, 272, 128], [0., 0, 128, 128], [0, 0, 128]):
            with self.assertRaises(ValueError):
                A.validate_rectangle(rect)

    def test_clear_is_exact_copy_with_empty_footprint(self):
        result, mask = A.reconstruct(self.image, profile('clear'), [0, 128, 128, 256])
        self.assertTrue(np.array_equal(result, self.image))
        self.assertFalse(np.shares_memory(result, self.image))
        self.assertFalse(mask.any())

    def test_local_mosaic_uses_own_moving_origin(self):
        original = self.image.copy()
        rect = [16, 96, 144, 224]
        result, mask = A.reconstruct(self.image, profile('local_mosaic_8'), rect)
        expected = np.zeros((256, 256), bool); expected[96:224, 16:144] = True
        self.assertTrue(np.array_equal(mask > 0, expected))
        self.assertTrue(np.array_equal(result[~expected], original[~expected]))
        self.assertTrue(np.all(result[96:104, 16:24, 0] == 20))
        self.assertTrue(np.all(result[96:104, 16:24, 1] == 100))
        self.assertTrue(np.array_equal(self.image, original))
        self.assertEqual(A.H.RECT, (64, 64, 192, 192))

    def test_local_gaussian_filters_full_canvas_before_replacement(self):
        impulse = np.zeros((256, 256, 3), np.uint8); impulse[95, 64] = 255
        result, mask = A.reconstruct(impulse, profile('local_gaussian_3p0'), [64, 96, 192, 224])
        self.assertGreater(int(result[96, 64, 0]), 0)
        self.assertTrue(np.array_equal(result[mask == 0], impulse[mask == 0]))
        self.assertEqual(int(np.count_nonzero(mask)), 128 * 128)

    def test_repeated_rows_have_hand_computed_half_up_values(self):
        image = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 256, axis=1).repeat(3, axis=2)
        result, mask = A.reconstruct(image, profile('repeated_rows_4_blend0p75'), [0, 0, 128, 128])
        self.assertEqual(result[:8, 0, 0].tolist(), [0, 0, 1, 1, 4, 4, 5, 5])
        self.assertTrue(np.all(mask == 255))

    def test_all_literal_filters_preserve_constant_colour(self):
        image = np.empty((256, 256, 3), np.uint8); image[:] = [19, 127, 223]
        for p in A.PROFILES:
            with self.subTest(profile=p['id']):
                result, _ = A.reconstruct(image, p, [128, 128, 256, 256])
                self.assertTrue(np.array_equal(result, image))

    def test_gaussian_anisotropy_orientation(self):
        impulse = np.zeros((256, 256, 3), np.uint8); impulse[128, 128] = 255
        h, _ = A.reconstruct(impulse, profile('anisotropic_h5_v0p7'), [0, 0, 128, 128])
        v, _ = A.reconstruct(impulse, profile('anisotropic_h0p7_v5'), [0, 0, 128, 128])
        self.assertTrue(np.array_equal(h, v.transpose(1, 0, 2)))
        self.assertGreater(np.count_nonzero(h[128, :, 0]), np.count_nonzero(h[:, 128, 0]))

    def test_profile_parameter_change_is_rejected(self):
        changed = profile('local_mosaic_8'); changed['parameters']['cell'] = 16
        with self.assertRaises(ValueError):
            A.reconstruct(self.image, changed, [0, 0, 128, 128])

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":[1e999]}'):
            with self.assertRaises(ValueError):
                A.strict_json(raw)

    def test_namespace_rejects_empty_extra_directory(self):
        scratch = ROOT / 'scratch'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='independent-filter-namespace-', dir=scratch) as name:
            root = Path(name); (root / 'manifest.json').write_bytes(b'{}')
            A.exact_namespace(root, {})
            (root / 'unexpected').mkdir()
            with self.assertRaises(ValueError):
                A.exact_namespace(root, {})

    def test_bound_detects_same_length_byte_tamper(self):
        scratch = ROOT / 'scratch'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='independent-filter-hash-', dir=scratch) as name:
            p = Path(name) / 'bound'; p.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            self.assertEqual(A.H.bound(p, expected, 8), b'original')
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                A.H.bound(p, expected, 8)

    def test_no_neural_or_producer_imports(self):
        A.no_producer_or_neural()

    def test_frozen_criteria_has_no_automatic_admission(self):
        path = ROOT / 'outputs/codeformer_filter_train_v2_preparation/input_only_review_criteria_v1.json'
        value = A.H.strict_json(A.H.bound(path, A.CRITERIA_SHA))
        self.assertEqual(set(value['decisions']), {'readable', 'borderline', 'insufficient', 'out_of_scope'})
        self.assertFalse(value['optimization_permitted_by_data_generation'])
        self.assertTrue(value['frozen_before_new_inputs_viewed'])


if __name__ == '__main__':
    unittest.main()
