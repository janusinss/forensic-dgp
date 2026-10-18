"""Synthetic CPU fixtures only; no real training images, model or VM calls."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid

import numpy as np
from PIL import Image

from research.codeformer_adaptation_v1 import data


class CodeFormerAdaptationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Python3.13's secure TemporaryDirectory ACL blocks this restricted
        # Windows token. Use an ordinary workspace fixture directory instead.
        cls.fixture_parent = (Path(__file__).resolve().parents[1]
            / 'outputs/codeformer_training_readiness_v1/data_review').resolve()
        cls.root = cls.fixture_parent / ('cpu_test_fixture_' + uuid.uuid4().hex)
        cls.root.mkdir()
        cls.addClassCleanup(cls._remove_fixture)
        rows = []
        y, x = np.indices((1024, 1024), dtype=np.int32)
        for number, role in ((1, 'train'), (2, 'validation')):
            rid = ('tr' if role == 'train' else 'va') + f'_ffhq_{number:05d}'
            relative = f'outputs/cctv_dgp_hq_cohort_v2/images1024/{number:05d}.png'
            path = cls.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb = np.stack(((x + number * 11) % 256, (y + number * 31) % 256,
                            ((x // 8 + y // 8) % 2) * 210 + number * 10), axis=-1).astype(np.uint8)
            Image.fromarray(rgb).save(path)
            with Image.open(path) as source:
                target = np.asarray(source.resize((512, 512), Image.Resampling.LANCZOS))
            rows.append({'id': rid, 'historical_role': role,
                'decision': 'accept_clean_restoration',
                'source': 'FFHQ official aligned 1024 counterpart',
                'source_path': relative, 'source_sha256': data.file_sha256(path),
                'native_size': [1024, 1024],
                'proposed_512_rgb_sha256': data.pixel_sha256(target),
                'attribution': {'author': 'Synthetic test fixture', 'photo_url': f'fixture://{number}'},
                'suitable_as_hq_train_candidate': role == 'train'})
        cls.rows = rows
        cls.manifest_dir = cls.root / 'audit'
        cls.manifest_dir.mkdir()
        cls.train_path = cls.manifest_dir / 'train.json'
        cls.ledger_path = cls.manifest_dir / 'ledger.json'
        cls.train_path.write_text(json.dumps({'candidates': rows[:1]}), encoding='utf-8')
        cls.ledger_path.write_text(json.dumps({'sources': rows}), encoding='utf-8')
        cls.specs = {
            'train': ('audit/train.json', data.file_sha256(cls.train_path), 1),
            'all': ('audit/ledger.json', data.file_sha256(cls.ledger_path), 2),
        }

    @classmethod
    def _remove_fixture(cls):
        resolved = cls.root.resolve()
        assert resolved.parent == cls.fixture_parent and resolved.name.startswith('cpu_test_fixture_')
        shutil.rmtree(resolved)

    def setUp(self):
        self.spec_patch = mock.patch.object(data, '_MANIFEST_SPECS', self.specs)
        self.validation_patch = mock.patch.object(data, '_VALIDATION_COUNT', 1)
        self.spec_patch.start()
        self.validation_patch.start()
        self.addCleanup(self.spec_patch.stop)
        self.addCleanup(self.validation_patch.stop)
        self.reference = data.load_reference_manifest(self.root, 'train')[0]

    def test_exact_determinism_independent_of_global_rng(self):
        first = data.make_pair(self.reference, 'compound', 123)
        np.random.seed(922)
        before = np.random.get_state()
        second = data.make_pair(self.reference, 'compound', 123)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(first.input256, second.input256))
        self.assertTrue(np.array_equal(first.target512, second.target512))
        self.assertEqual(first.record, second.record)
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_all_profiles_shape_dtype_finite_and_pixel_hashes(self):
        for profile in data.PROFILES:
            with self.subTest(profile=profile):
                pair = data.make_pair(self.reference, profile, 453)
                self.assertEqual(pair.input256.shape, (256, 256, 3))
                self.assertEqual(pair.target512.shape, (512, 512, 3))
                for array in (pair.input256, pair.target512):
                    self.assertEqual(array.dtype, np.float32)
                    self.assertTrue(array.flags.c_contiguous)
                    self.assertTrue(np.isfinite(array).all())
                    self.assertGreaterEqual(float(array.min()), 0)
                    self.assertLessEqual(float(array.max()), 1)
                self.assertEqual(data.pixel_sha256(pair.input256), pair.record['input256_float32_sha256'])
                self.assertEqual(data.pixel_sha256(pair.target512), pair.record['clean512_float32_sha256'])
                self.assertEqual(pair.parameters['image_eligible'], pair.parameters['long_edge'] >= 32)
                self.assertFalse(pair.record['geometry']['covering_simulation'])

    def test_seed_variation_changes_degraded_input_only(self):
        first = data.make_pair(self.reference, 'lowlight', 123)
        second = data.make_pair(self.reference, 'lowlight', 456)
        self.assertFalse(np.array_equal(first.input256, second.input256))
        self.assertTrue(np.array_equal(first.target512, second.target512))
        self.assertNotEqual(first.parameters, second.parameters)

    def test_clear_is_exact_declared_bilinear_input_and_lanczos_target(self):
        pair = data.make_pair(self.reference, 'clear', 31)
        with Image.open(self.reference.source_path) as source:
            expected_input = np.asarray(source.resize((256, 256), Image.Resampling.BILINEAR))
            expected_target = np.asarray(source.resize((512, 512), Image.Resampling.LANCZOS))
        self.assertTrue(np.array_equal(pair.input256, expected_input.astype(np.float32) / np.float32(255)))
        self.assertTrue(np.array_equal(pair.target512, expected_target.astype(np.float32) / np.float32(255)))
        self.assertTrue(np.array_equal(pair.target512, data.derive_clean_target(self.reference)))
        self.assertTrue(np.array_equal(pair.input256, data.make_pair(self.reference, 'clear', 99).input256))
        self.assertTrue(pair.parameters['image_eligible'])

    def test_historical_roles_remain_separate(self):
        train = data.load_reference_manifest(self.root, 'train')
        validation = data.load_reference_manifest(self.root, 'validation')
        self.assertTrue(all(r.role == 'train' for r in train))
        self.assertTrue(all(r.role == 'validation' for r in validation))
        self.assertFalse({r.id for r in train} & {r.id for r in validation})
        for role in ('reserved_evaluation', 'final', 'native', ''):
            with self.subTest(role=role), self.assertRaises(ValueError):
                data.load_reference_manifest(self.root, role)

    def test_source_mutation_refused_before_decode(self):
        path = self.reference.source_path
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\x00')
            with self.assertRaisesRegex(ValueError, 'Source SHA256 differs'):
                data.make_pair(self.reference, 'blur', 32)
        finally:
            path.write_bytes(original)

    def test_target_derivation_hash_mismatch_refused(self):
        wrong = data.Reference(**{**self.reference.__dict__, 'clean512_rgb8_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'target pixels differ'):
            data.derive_clean_target(wrong)

    def test_manifest_byte_mutation_refused(self):
        original = self.train_path.read_bytes()
        try:
            self.train_path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Manifest SHA256 differs'):
                data.load_reference_manifest(self.root, 'train')
        finally:
            self.train_path.write_bytes(original)

    def test_native_namespace_and_traversal_refused(self):
        for edits in ({'id': 'choke_dev_01', 'role': 'train'},
                      {'role': 'reserved_evaluation'},
                      {'source_relative_path': 'dataset/cctv_survface_raw/a.png'},
                      {'source_relative_path': 'data/sources/../../outside.png'}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                data.Reference(**{**self.reference.__dict__, **edits})

    def test_portable_manifest_requires_pin_and_relocates_sources(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            source = self.root / row['source_path']
            row['source_path'] = f"data/sources/{row['id']}.png"
            destination = self.root / row['source_path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        path = self.root / 'references.json'
        path.write_text(json.dumps({'format': 'codeformer-adaptation-reference-manifest-v1',
                                   'references': rows}), encoding='utf-8')
        pin = data.file_sha256(path)
        with self.assertRaisesRegex(ValueError, 'protocol-bound SHA256'):
            data.load_reference_manifest(self.root, 'train', path)
        train = data.load_reference_manifest(self.root, 'train', path, expected_manifest_sha256=pin)
        self.assertEqual(train[0].source_relative_path, f'data/sources/{train[0].id}.png')
        self.assertTrue(np.array_equal(data.derive_clean_target(train[0]), data.derive_clean_target(self.reference)))
        rows[0]['historical_role'] = 'validation'
        path.write_text(json.dumps({'format': 'codeformer-adaptation-reference-manifest-v1',
                                   'references': rows}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            data.load_reference_manifest(self.root, 'train', path,
                                         expected_manifest_sha256=data.file_sha256(path))

    def test_invalid_profile_or_seed_has_no_fallback(self):
        for profile, seed in (('unrecognized', 0), ('blur', -1), ('blur', 2**64),
                              ('blur', True), ('blur', 0.5)):
            with self.subTest(profile=profile, seed=seed), self.assertRaises(ValueError):
                data.make_pair(self.reference, profile, seed)

    def test_generated_min_edge_eligibility_is_explicit(self):
        params = [data.profile_parameters('blur', seed) for seed in range(100)]
        self.assertTrue(any(not p['image_eligible'] for p in params))
        self.assertTrue(any(p['image_eligible'] for p in params))
        self.assertTrue(all(p['image_eligible'] == (p['long_edge'] >= 32) for p in params))


if __name__ == '__main__':
    unittest.main()
