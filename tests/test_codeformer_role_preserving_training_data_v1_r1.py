"""Synthetic NumPy/Pillow data boundary checks, with no Torch or model imports."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'role_preserving_data_CPU_fixture'
DIRECTORY = ROOT/'research/codeformer_role_preserving_training_v1_r1'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]
sys.modules[PACKAGE] = package
for suffix in ('contract', 'data'):
    name = PACKAGE+'.'+suffix
    spec = importlib.util.spec_from_file_location(name, DIRECTORY/(suffix+'.py'))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
DATA = sys.modules[PACKAGE+'.data']


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def references():
    rows = []
    for index in range(444):
        role = 'train' if index < 391 else 'validation'
        rows.append({'id': '%s_ffhq_%05d' % ('tr' if role == 'train' else 'val', index),
            'historical_role': role, 'decision': 'accept_clean_restoration',
            'native_size': [1024, 1024], 'existing_target_size': [256, 256],
            'suitable_as_hq_train_candidate': role == 'train',
            'source_sha256': hashlib.sha256(('source%d' % index).encode()).hexdigest(),
            'proposed_512_rgb_sha256': hashlib.sha256(('clean%d' % index).encode()).hexdigest(),
            'source_path': 'approved/%s/%05d.png' % (role, index)})
    return {'format': 'codeformer-adaptation-reference-manifest-v1', 'references': rows}


class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = ROOT/'scratch'
        cls.scratch.mkdir(exist_ok=True)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=self.scratch)

    def test_import_does_not_activate_neural_framework(self):
        self.assertFalse({'torch', 'tensorflow', 'jax'} & set(sys.modules))

    def test_input_quantized_RGB8_exact_f32_division(self):
        pixels = np.arange(3*8*8, dtype=np.uint8).reshape(8, 8, 3)
        result = DATA.input_float32(pixels)
        self.assertEqual(result.dtype, np.dtype('float32'))
        self.assertTrue(result.flags.c_contiguous)
        self.assertTrue(np.array_equal(result, pixels.astype(np.float32)/np.float32(255)))
        for invalid in (pixels.astype(np.float32), pixels[..., 0], pixels[..., :2]):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                DATA.input_float32(invalid)

    def test_numeric_exact_positive_and_hash_mutation(self):
        with self.temporary() as folder:
            path = Path(folder)/'indices.npy'
            expected = np.arange(256, dtype=np.int64).reshape(1, 256)
            np.save(path, expected, allow_pickle=False)
            result = DATA.load_numeric(path, dtype=np.dtype('int64'), shape=(1, 256), expected_sha256=digest(expected))
            self.assertTrue(np.array_equal(result, expected))
            with self.assertRaises(ValueError):
                DATA.load_numeric(path, dtype=np.dtype('int64'), shape=(1, 256), expected_sha256='0'*64)

    def test_numeric_wrong_dtype_shape_and_nonfinite_refused(self):
        with self.temporary() as folder:
            path = Path(folder)/'target.npy'
            bad_arrays = [np.zeros((1, 4), dtype=np.float64), np.zeros((4,), dtype=np.float32),
                          np.array([[0, 1, np.nan, 3]], dtype=np.float32),
                          np.array([[0, 1, np.inf, 3]], dtype=np.float32),
                          np.array([[0, 1, -np.inf, 3]], dtype=np.float32)]
            for value in bad_arrays:
                np.save(path, value, allow_pickle=False)
                with self.subTest(dtype=str(value.dtype), shape=value.shape), self.assertRaises(ValueError):
                    DATA.load_numeric(path, dtype=np.dtype('float32'), shape=(1, 4), expected_sha256=digest(value))

    def test_numeric_pickle_and_fortran_layout_refused(self):
        with self.temporary() as folder:
            path = Path(folder)/'target.npy'
            np.save(path, np.array([{'unsafe': 'object'}], dtype=object), allow_pickle=True)
            with self.assertRaises(ValueError):
                DATA.load_numeric(path, dtype=np.dtype('float32'), shape=(1,), expected_sha256='0'*64)
            value = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
            np.save(path, value, allow_pickle=False)
            with self.assertRaises(ValueError):
                DATA.load_numeric(path, dtype=np.dtype('float32'), shape=(3, 4), expected_sha256=digest(value))

    def test_reference_metadata_roles_and_source_overlap(self):
        document = references()
        self.assertEqual(len(DATA.validate_references(document)), 444)
        for key in ('source_path', 'source_sha256'):
            changed = deepcopy(document)
            changed['references'][-1][key] = changed['references'][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                DATA.validate_references(changed)
        changed = deepcopy(document)
        changed['references'][0]['historical_role'] = 'validation'
        with self.assertRaises(ValueError):
            DATA.validate_references(changed)

    def test_reference_approval_geometry_and_duplicate_id(self):
        document = references()
        for key, value in [('decision', 'reject'), ('native_size', [512, 512]),
                           ('suitable_as_hq_train_candidate', False)]:
            changed = deepcopy(document); changed['references'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                DATA.validate_references(changed)
        changed = deepcopy(document)
        changed['references'][1]['id'] = changed['references'][0]['id']
        with self.assertRaises(ValueError):
            DATA.validate_references(changed)

    def test_observed_mask_partial_exact_and_invalid(self):
        with self.temporary() as folder:
            path = Path(folder)/'observed.png'
            mask = np.zeros((256, 256), dtype=np.uint8); mask[32:224, 16:240] = 255
            Image.fromarray(mask).save(path)
            result = DATA.load_mask(path)
            self.assertEqual(result.dtype, np.dtype('bool'))
            self.assertTrue(np.array_equal(result, mask > 0))
            for invalid in (np.zeros((256, 256), dtype=np.uint8), np.full((256, 256), 127, dtype=np.uint8),
                            np.ones((128, 128), dtype=np.uint8)*255):
                Image.fromarray(invalid).save(path)
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    DATA.load_mask(path)

    def test_clean512_direct_source_parity_and_mutation(self):
        with self.temporary() as folder:
            path = Path(folder)/'source.png'
            axis = np.arange(1024, dtype=np.uint16)
            pixels = np.empty((1024, 1024, 3), dtype=np.uint8)
            pixels[..., 0] = axis[None, :] % 256
            pixels[..., 1] = axis[:, None] % 256
            pixels[..., 2] = (axis[:, None]+3*axis[None, :]) % 256
            Image.fromarray(pixels).save(path)
            expected_rgb8 = np.asarray(Image.fromarray(pixels).resize((512, 512), Image.Resampling.LANCZOS)).copy()
            expected_float = expected_rgb8.astype(np.float32)/np.float32(255)
            result = DATA.clean512(path, digest(expected_rgb8), digest(expected_float))
            self.assertTrue(np.array_equal(result, expected_float))
            for a, b in [('0'*64, digest(expected_float)), (digest(expected_rgb8), '0'*64)]:
                with self.assertRaises(ValueError):
                    DATA.clean512(path, a, b)

    def test_RGB_geometry_mode_is_not_implicitly_changed(self):
        with self.temporary() as folder:
            path = Path(folder)/'image.png'
            for invalid in (np.zeros((256, 256), dtype=np.uint8), np.zeros((128, 128, 3), dtype=np.uint8),
                            np.zeros((256, 256, 4), dtype=np.uint8)):
                Image.fromarray(invalid).save(path)
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    DATA.load_rgb(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
