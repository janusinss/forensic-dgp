"""Meaningful native pixel/mask/identity fixtures; no model/server/browser calls."""
import builtins
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
original_import = builtins.__import__
TORCH_PREEXISTING = 'torch' in sys.modules


def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'): raise AssertionError('No Torch in pure brush tests')
    return original_import(name, *args, **kwargs)


with patch('builtins.__import__', blocked):
    import numpy as np
    from PIL import Image
    from research import protected_surroundings_brush_v1 as M


def encoded(array, format='PNG', **options):
    stream = io.BytesIO(); Image.fromarray(array).save(stream, format=format, **options); return stream.getvalue()


def pixels(raw):
    with Image.open(io.BytesIO(raw)) as image: return np.asarray(image).copy()


class NativeBrushFixtures(unittest.TestCase):
    def setUp(self):
        import_guard = patch('builtins.__import__', blocked)
        import_guard.start(); self.addCleanup(import_guard.stop)
        y, x = np.mgrid[:49, :65]
        self.rgb = np.stack([(x * 37 + y * 13) % 256, (x * 11 + y * 43) % 256, (x * 3 + y * 7) % 256], -1).astype(np.uint8)
        self.file = encoded(self.rgb)
        self.protect = np.zeros((49, 65), np.uint8); self.protect[15:30, 21:37] = 255
        self.apply = np.zeros_like(self.protect); self.apply[4:45, 3:60] = 255

    def preview(self, protect=None, apply=None, **params):
        return M.build_preview(self.file, encoded(self.protect if protect is None else protect),
                               encoded(self.apply if apply is None else apply), **params)

    def test_exact_protection_outsideApply_and_external_weights(self):
        response, payload = self.preview(guard=3, feather=5)
        guarded = pixels(payload['effective-Protect-native.png']) > 0
        effective = pixels(payload['effective-Apply-native.png']) > 0
        result = pixels(payload['derived-native.png'])
        alpha = np.load(io.BytesIO(payload['alpha-float32.npy']), allow_pickle=False)
        np.testing.assert_array_equal(result[guarded], self.rgb[guarded])
        np.testing.assert_array_equal(result[self.apply == 0], self.rgb[self.apply == 0])
        self.assertFalse(alpha[~effective].any()); self.assertTrue(np.any(result != self.rgb))
        self.assertEqual(alpha.dtype, np.float32); self.assertTrue(((alpha >= 0) & (alpha <= 1)).all())
        self.assertTrue(response['metadata']['filter_may_read_protected_context'])

    def test_overlap_Protect_wins_and_both_input_masks_remain_unchanged(self):
        apply = np.full_like(self.apply, 255); before = self.protect.copy(); a_before = apply.copy()
        _, payload = self.preview(apply=apply, guard=0, feather=1)
        result = pixels(payload['derived-native.png'])
        np.testing.assert_array_equal(result[self.protect > 0], self.rgb[self.protect > 0])
        np.testing.assert_array_equal(before, self.protect); np.testing.assert_array_equal(a_before, apply)

    def test_emptyApply_fullProtect_and_radiusZero_are_truthful_noops(self):
        zero, full = np.zeros_like(self.apply), np.full_like(self.apply, 255)
        for kwargs in ({'apply': zero}, {'protect': full, 'apply': full}, {'nominal_radius': 0}):
            response, payload = self.preview(**kwargs)
            self.assertTrue(response['metadata']['no_effect'])
            np.testing.assert_array_equal(pixels(payload['derived-native.png']), self.rgb)
            self.assertEqual(response['metadata']['changed_native_pixels'], 0)

    def test_emptyProtect_edge_islands_and_holes_never_expandApply(self):
        zero = np.zeros_like(self.apply); apply = zero.copy()
        apply[0:3, 0:3] = 255; apply[20:24, 30:35] = 255; apply[21, 32] = 0
        _, payload = self.preview(protect=zero, apply=apply, guard=8, feather=16)
        result = pixels(payload['derived-native.png']); alpha = np.load(io.BytesIO(payload['alpha-float32.npy']), allow_pickle=False)
        np.testing.assert_array_equal(result[apply == 0], self.rgb[apply == 0]); self.assertFalse(alpha[apply == 0].any())
        self.assertGreater(alpha[0, 0], 0); self.assertEqual(alpha[21, 32], 0)

    def test_rebuild_determinism_and_each_edit_invalidates_identity(self):
        first, payload = self.preview(); second, again = self.preview()
        self.assertEqual(first['preview_id'], second['preview_id']); self.assertEqual(payload, again)
        p = self.protect.copy(); p[0, 0] = 255
        a = self.apply.copy(); a[0, 0] = 255
        ids = {first['preview_id'], self.preview(protect=p)[0]['preview_id'], self.preview(apply=a)[0]['preview_id'],
               self.preview(nominal_radius=5)[0]['preview_id'], self.preview(guard=7)[0]['preview_id'], self.preview(feather=15)[0]['preview_id']}
        self.assertEqual(len(ids), 6)
        self.assertEqual(payload['uploaded-original.bin'], self.file)

    def test_strict_mask_schema_and_nonbinary_values_rejected(self):
        wrong = np.zeros((48, 65), np.uint8); nonbinary = self.protect.copy(); nonbinary[0, 0] = 1
        rgba = np.zeros((49, 65, 4), np.uint8)
        for raw in (encoded(wrong), encoded(nonbinary), encoded(rgba), encoded(self.protect, format='JPEG'), b'not PNG'):
            with self.assertRaises(ValueError): M.build_preview(self.file, raw, encoded(self.apply))
        with self.assertRaises(ValueError): M.build_preview(self.file, b'x' * (M.MAX_MASK_BYTES + 1), encoded(self.apply))

    def test_parameters_and_original_decode_limits_rejected_before_processing(self):
        for key, value in (('nominal_radius', float('nan')), ('nominal_radius', 13), ('guard', True),
                           ('guard', -1), ('feather', 0), ('feather', float('inf'))):
            with self.assertRaises(ValueError): self.preview(**{key: value})
        for raw in (b'', b'bad', b'x' * (M.MAX_UPLOAD_BYTES + 1), encoded(np.zeros((2, 1601, 3), np.uint8))):
            with self.assertRaises(ValueError): M.build_preview(raw, encoded(self.protect), encoded(self.apply))

    def test_EXIF_orientation_masks_follow_decodedRGB_not_file_geometry(self):
        image = Image.fromarray(np.arange(17 * 25 * 3, dtype=np.uint8).reshape(17, 25, 3))
        exif = Image.Exif(); exif[274] = 6; stream = io.BytesIO(); image.save(stream, format='JPEG', exif=exif)
        correct = encoded(np.zeros((25, 17), np.uint8))
        response, payload = M.build_preview(stream.getvalue(), correct, correct)
        self.assertEqual((response['width'], response['height']), (17, 25))
        self.assertEqual(pixels(payload['original-decoded-rgb.png']).shape, (25, 17, 3))
        with self.assertRaises(ValueError): M.build_preview(stream.getvalue(), encoded(np.zeros((17, 25), np.uint8)), correct)
        mask = Image.fromarray(np.zeros((25, 17), np.uint8)); bad = io.BytesIO(); mask.save(bad, format='PNG', exif=exif)
        with self.assertRaises(ValueError): M.build_preview(stream.getvalue(), bad.getvalue(), correct)

    def test_float64_rint_color_blend_and_exact256_interiors(self):
        response, payload = self.preview(guard=2, feather=4)
        alpha = np.load(io.BytesIO(payload['alpha-float32.npy']), allow_pickle=False)
        from PIL import ImageFilter
        blurred = np.asarray(Image.fromarray(self.rgb).filter(ImageFilter.GaussianBlur(radius=6 * 65 / 256)))
        expected = np.rint(self.rgb.astype(np.float64) * (1 - alpha[..., None].astype(np.float64))
                           + blurred.astype(np.float64) * alpha[..., None].astype(np.float64)).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(pixels(payload['derived-native.png']), expected)
        before, after = pixels(payload['prepared-original-256.png']), pixels(payload['prepared-derived-256.png'])
        for name in ('protected', 'outside_apply', 'unchanged_union'):
            interior = pixels(payload[name + '-interior-256.png']) > 0
            np.testing.assert_array_equal(before[interior], after[interior])
            self.assertEqual(int(interior.sum()), response['metadata']['conservative_256_exact_counts'][name])
        self.assertTrue(response['metadata']['prepared_boundary_pixels_not_claimed_exact'])

    def test_payloads_hashes_and_metadata_no_neural_claim(self):
        response, payload = self.preview()
        for name, row in response['metadata']['payload_pins'].items():
            self.assertEqual(row, {'bytes': len(payload[name]), 'sha256': M.sha(payload[name])})
        self.assertEqual(response['metadata']['preprocessing_neural_optimizer_training_VM_calls'], 0)
        self.assertFalse(response['metadata']['automatic_detection_or_adaptive_matching'])
        self.assertEqual('torch' in sys.modules, TORCH_PREEXISTING)


if __name__ == '__main__': unittest.main()
