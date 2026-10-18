"""Independent math, corruption-footprint and source-binding CPU fixtures."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pair_prepare', ROOT / 'scripts/prepare_codeformer_filter_pairs_v1.py')
builder = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(builder)
from research.codeformer_filter_pairs_v1 import filters as f


def profile(name):
    return next(copy.deepcopy(p) for p in f.PROFILES if p['id'] == name)


class FilterPairFixtures(unittest.TestCase):
    def test_DEV_never_becomes_optimization_eligible_after_input_review(self):
        for strong in (False, True):
            record = builder.training_admission('DEV', strong)
            self.assertFalse(record['optimization_allowed_by_split'])
            self.assertFalse(record['optimization_permitted_now'])
            self.assertEqual(record['restoration_training_eligibility'], 'EXCLUDED_DEV')
        self.assertEqual(builder.training_admission('TRAIN', False)['restoration_training_eligibility'],
            'PENDING_INPUT_ONLY_REVIEW')
        self.assertEqual(builder.training_admission('TRAIN', True)['restoration_training_eligibility'],
            'EXCLUDED_INITIALLY_STRONG_CONTROL')

    def test_approved_profile_census_controls_and_orientations(self):
        self.assertEqual(len(f.PROFILES), 12)
        self.assertEqual({p['id'] for p in f.PROFILES if p['strong_control']},
            {'gaussian_6p0', 'repeated_rows_4_blend0p75', 'global_mosaic_16', 'local_gaussian_10p0', 'local_mosaic_16'})
        self.assertEqual(profile('anisotropic_h5_v0p7')['parameters'], {'sigma_x': 5.0, 'sigma_y': .7})
        self.assertEqual(profile('anisotropic_h0p7_v5')['parameters'], {'sigma_x': .7, 'sigma_y': 5.0})

    def test_identity_copies_pixels_and_has_empty_corruption_footprint(self):
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        result, footprint = f.apply_profile(image, profile('clear'))
        self.assertTrue(np.array_equal(result, image))
        self.assertFalse(np.shares_memory(result, image))
        self.assertFalse(footprint.any())

    def test_all_filters_preserve_a_constant_RGB_colour(self):
        image = np.empty((256, 256, 3), np.uint8); image[:] = [17, 128, 231]
        for spec in f.PROFILES:
            with self.subTest(profile=spec['id']):
                result, _ = f.apply_profile(image, spec)
                self.assertTrue(np.array_equal(result, image))

    def test_gaussian_reflected_corner_matches_independent_discrete_kernel(self):
        image = np.zeros((256, 256, 3), np.uint8); image[0, 0] = 255
        actual = f.gaussian(image, 1., 1.)
        coordinates = np.arange(-3, 4, dtype=np.float64)
        kernel = np.exp(-coordinates**2 / 2); kernel /= kernel.sum()
        # reflect repeats the edge sample: the corner receives w0+w1 per axis.
        expected = int(np.floor(255 * (kernel[3] + kernel[4])**2 + .5))
        self.assertEqual(int(actual[0, 0, 0]), expected)
        self.assertTrue(np.array_equal(actual[..., 0], actual[..., 1]))
        self.assertEqual(int(actual[-1, -1, 0]), 0)

    def test_anisotropic_impulse_spreads_along_declared_axis(self):
        image = np.zeros((256, 256, 3), np.uint8); image[128, 128] = 255
        ys, xs = np.indices((256, 256), dtype=np.float64)
        moments = []
        for name in ('anisotropic_h5_v0p7', 'anisotropic_h0p7_v5'):
            result, _ = f.apply_profile(image, profile(name)); mass = result[..., 0].astype(np.float64)
            moments.append(((mass * (xs - 128)**2).sum() / mass.sum(),
                (mass * (ys - 128)**2).sum() / mass.sum()))
        self.assertGreater(moments[0][0], moments[0][1] * 10)
        self.assertGreater(moments[1][1], moments[1][0] * 10)
        self.assertAlmostEqual(moments[0][0], moments[1][1], places=12)

    def test_mosaic_is_block_average_with_half_up_and_no_interpolation(self):
        yy, xx = np.indices((256, 256), dtype=np.uint8)
        image = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
        result, footprint = f.apply_profile(image, profile('global_mosaic_8'))
        self.assertTrue(np.all(result[:8, :8] == [4, 4, 0]))
        self.assertTrue(np.all(result[:8, 8:16] == [12, 4, 0]))
        self.assertTrue(np.all(footprint == 255))

    def test_repeated_row_blend_matches_hand_calculated_values(self):
        image = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 256, axis=1).repeat(3, axis=2)
        result, _ = f.apply_profile(image, profile('repeated_rows_4_blend0p75'))
        self.assertEqual(result[:8, 0, 0].tolist(), [0, 0, 1, 1, 4, 4, 5, 5])

    def test_local_corruption_never_changes_outside_fixed_rectangle(self):
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8); before = image.copy()
        expected = np.zeros((256, 256), bool); expected[64:192, 64:192] = True
        for spec in (p for p in f.PROFILES if p['operator'].startswith('local_')):
            with self.subTest(profile=spec['id']):
                result, footprint = f.apply_profile(image, spec)
                self.assertTrue(np.array_equal(footprint > 0, expected))
                self.assertEqual(int(np.count_nonzero(footprint)), 128 * 128)
                self.assertTrue(np.array_equal(result[~expected], image[~expected]))
                self.assertFalse(np.array_equal(result[expected], image[expected]))
        self.assertTrue(np.array_equal(image, before))

    def test_profile_parameter_tamper_is_not_silently_accepted(self):
        tampered = profile('gaussian_2p4'); tampered['parameters']['sigma_x'] = 2.5
        with self.assertRaisesRegex(ValueError, 'prospectively approved'):
            f.apply_profile(np.zeros((256, 256, 3), np.uint8), tampered)

    def test_source_hash_detects_same_length_mutation(self):
        with tempfile.TemporaryDirectory(prefix='filter-source-', dir=ROOT / 'scratch') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT / 'scratch'))
            path = Path(folder) / 'bound.bin'; path.write_bytes(b'original')
            digest = hashlib.sha256(b'original').hexdigest()
            self.assertEqual(builder.read_bound(path, digest, 8), b'original')
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                builder.read_bound(path, digest, 8)

    def test_writer_refuses_existing_namespace_and_traversal(self):
        with tempfile.TemporaryDirectory(prefix='filter-writer-', dir=ROOT / 'outputs') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT / 'outputs'))
            with self.assertRaisesRegex(ValueError, 'New output namespace'):
                builder.Writer(folder)
            writer = builder.Writer(Path(folder) / 'new')
            with self.assertRaisesRegex(ValueError, 'Canonical'):
                writer.put('../escaped.bin', b'bad')
            writer.put('evidence.bin', b'first')
            with self.assertRaisesRegex(ValueError, 'Finite unique'):
                writer.put('evidence.bin', b'second')
            self.assertEqual((writer.root / 'evidence.bin').read_bytes(), b'first')

    def test_direct512_reference_is_not_an_input256_upscale(self):
        yy, xx = np.indices((1024, 1024))
        pixels = ((xx * 23 + yy * 17 + (xx % 7) * (yy % 13)) % 256).astype(np.uint8)
        image = Image.fromarray(np.stack([pixels, pixels, pixels], axis=-1))
        direct512 = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS))
        from256 = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.LANCZOS))
        self.assertFalse(np.array_equal(direct512, from256))

    def test_input_only_sheet_retains_eight_exact_cells_and_clear_reviewed_last(self):
        references = [(rid, split) for split in ('TRAIN', 'DEV') for rid in builder.REFERENCE_IDS[split]]
        images = [(rid, split, np.full((256, 256, 3), n * 17, np.uint8))
            for n, (rid, split) in enumerate(references)]
        data, cells = builder.input_only_sheet('global_mosaic_16', images)
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (1024, 612)); saved = np.asarray(image)
        self.assertEqual(len(cells), 8)
        self.assertEqual(builder.INPUT_REVIEW_ORDER[-1], 'clear')
        self.assertEqual(builder.INPUT_REVIEW_ORDER[:5], ('global_mosaic_16', 'local_mosaic_16',
            'local_gaussian_10p0', 'gaussian_6p0', 'repeated_rows_4_blend0p75'))
        for index, cell in enumerate(cells):
            tile = saved[cell['y']:cell['y'] + 256, cell['x']:cell['x'] + 256]
            self.assertTrue(np.all(tile == index * 17))


if __name__ == '__main__':
    unittest.main()
