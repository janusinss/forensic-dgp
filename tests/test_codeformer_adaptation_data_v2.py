"""Meaningful synthetic CPU fixtures; no real face/model/gradient/VM calls."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from research.codeformer_adaptation_v1 import data as v1
from research.codeformer_adaptation_v2 import data


class PairedCameraV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_parent = (Path(__file__).resolve().parents[1]
            / 'outputs/codeformer_adaptation_v2_design_review').resolve()
        cls.fixture_parent.mkdir(parents=True, exist_ok=True)
        cls.root = cls.fixture_parent / ('cpu_data_fixture_' + uuid.uuid4().hex)
        cls.root.mkdir()
        cls.addClassCleanup(cls._remove_fixture)
        cls.refs = []
        y, x = np.indices((1024, 1024), dtype=np.int32)
        for number in (1, 2):
            rgb = np.stack(((x + 17 * number) % 256, (y + 31 * number) % 256,
                ((x // 4 + y // 8) % 2) * 150 + 40 + number), axis=-1).astype(np.uint8)
            relative = f'data/sources/tr_ffhq_{number:05d}.png'
            path = cls.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
            with Image.open(path) as image:
                target = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS))
            cls.refs.append(data.Reference(f'tr_ffhq_{number:05d}', 'train', cls.root,
                relative, data.file_sha256(path), data.pixel_sha256(target), 'a' * 64,
                json.dumps({'author': 'Synthetic fixture', 'photo_url': f'fixture://{number}'})))

    @classmethod
    def _remove_fixture(cls):
        target = cls.root.resolve()
        assert target.parent == cls.fixture_parent and target.name.startswith('cpu_data_fixture_')
        shutil.rmtree(target)

    def test_deterministic_pair_and_global_rng_unchanged(self):
        first = data.make_training_update(*self.refs, 13)
        np.random.seed(12345)
        before = np.random.get_state()
        second = data.make_training_update(*self.refs, 13)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])
        for a, b in zip(first, second):
            self.assertTrue(np.array_equal(a.input256, b.input256))
            self.assertTrue(np.array_equal(a.target512, b.target512))
            self.assertEqual(a.record, b.record)

    def test_shape_slots_eligibility_hashes_and_noise_linkage(self):
        for update in (1, 3, 5, 9, 13, 17, 21):
            with self.subTest(update=update):
                pairs = data.make_training_update(*self.refs, update)
                self.assertEqual([p.record['slot'] for p in pairs], [1, 2, 3, 4])
                self.assertEqual([p.parameters['behavior_anchor'] for p in pairs], [False, False, True, False])
                for pair in pairs:
                    self.assertEqual(pair.input256.shape, (256, 256, 3))
                    self.assertEqual(pair.target512.shape, (512, 512, 3))
                    for array in (pair.input256, pair.target512):
                        self.assertEqual(array.dtype, np.float32)
                        self.assertTrue(array.flags.c_contiguous)
                        self.assertTrue(np.isfinite(array).all())
                        self.assertGreaterEqual(float(array.min()), 0.)
                        self.assertLessEqual(float(array.max()), 1.)
                    self.assertEqual(data.pixel_sha256(pair.input256), pair.record['input256_float32_sha256'])
                    self.assertEqual(data.pixel_sha256(pair.target512), pair.record['clean512_float32_sha256'])
                    self.assertFalse(pair.record['geometry']['crop'])
                    self.assertFalse(pair.record['geometry']['covering_simulation'])
                    self.assertEqual(pair.parameters['image_eligible'], pair.parameters['long_edge'] >= 32)
                for normal, dark in ((pairs[0], pairs[1]), (pairs[2], pairs[3])):
                    self.assertEqual(normal.parameters['seeds'], dark.parameters['seeds'])
                    self.assertEqual(normal.parameters['noise_fields'], dark.parameters['noise_fields'])
                    self.assertEqual(normal.record['pair_id'], dark.record['pair_id'])
                    self.assertTrue(np.array_equal(normal.target512, dark.target512))
                    self.assertFalse(np.array_equal(normal.input256, dark.input256))

    def test_seed_key_is_exact_and_roles_update_have_independent_fields(self):
        p = data.camera_parameters('tr_ffhq_00001', 'A', 1)
        key = b'codeformer-adaptation-v2|20261015|0|tr_ffhq_00001|A|linear_noise'
        self.assertEqual(p['seeds']['linear_noise']['key'], key.decode())
        self.assertEqual(p['seeds']['linear_noise']['seed64'],
                         int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        fields = {v['seed64'] for v in p['seeds'].values()}
        self.assertEqual(len(fields), 3)
        self.assertNotEqual(p['seeds'], data.camera_parameters('tr_ffhq_00001', 'B', 1)['seeds'])
        self.assertNotEqual(p['seeds'], data.camera_parameters('tr_ffhq_00001', 'A', 2)['seeds'])

    def test_keyed_rgb_noise_unchanged_when_linear_parameters_change(self):
        p = data.camera_parameters(self.refs[1].id, 'B', 13)
        linear, rgb = data._noise_fields(p)
        changed = copy.deepcopy(p)
        changed['shot_coefficient'] = .003
        changed['read_sigma_linear'] = .01
        other_linear, other_rgb = data._noise_fields(changed)
        self.assertTrue(np.array_equal(rgb, other_rgb))
        self.assertTrue(np.array_equal(linear, other_linear))
        self.assertEqual(rgb.dtype, np.float64)
        self.assertFalse(np.array_equal(rgb, linear))

    def test_clear64_has_actual_resolution_loss_and_dark_exposure(self):
        clear256 = data.make_training_update(*self.refs, 1)[2]
        clear64, dark64 = data.make_training_update(*self.refs, 9)[2:]
        self.assertEqual(clear64.parameters['variant'], 'clear64')
        self.assertEqual(clear64.parameters['gaussian_sigma_256'], 0.)
        self.assertIsNone(clear64.parameters['jpeg_quality'])
        self.assertFalse(np.array_equal(clear256.input256, clear64.input256))
        def horizontal_variation(image):
            return float(np.abs(np.diff(image, axis=1)).mean())
        self.assertLess(horizontal_variation(clear64.input256), horizontal_variation(clear256.input256))
        self.assertLess(float(dark64.input256.mean()), float(clear64.input256.mean()) * .9)
        # Target stays the same original photograph, never degraded/upscaled GT.
        self.assertTrue(np.array_equal(clear64.target512, data.derive_clean_target(self.refs[1])))

    def test_clear_camera_does_not_call_v1_shortcuts(self):
        with (mock.patch.object(v1, '_degrade', side_effect=AssertionError('V1 camera forbidden')),
              mock.patch.object(v1, 'profile_parameters', side_effect=AssertionError('V1 recipe forbidden'))):
            result = data.make_training_update(*self.refs, 9)
        self.assertEqual(len(result), 4)

    def test_source_mutation_refused(self):
        path = self.refs[0].source_path
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Source SHA256 differs'):
                data.make_training_update(*self.refs, 1)
        finally:
            path.write_bytes(original)

    def test_target_hash_mutation_refused(self):
        bad = replace(self.refs[0], clean512_rgb8_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'target pixels differ'):
            data.make_training_update(bad, self.refs[1], 1)

    def test_validation_same_reference_invalid_updates_and_bad_roles_refused(self):
        validation = replace(self.refs[0], id='va_ffhq_00001', role='validation')
        for a, b in ((validation, self.refs[1]), (self.refs[0], self.refs[0])):
            with self.assertRaises(ValueError):
                data.make_training_update(a, b, 1)
        for update in (0, 601, -1, True, 1.5):
            with self.subTest(update=update), self.assertRaises(ValueError):
                data.make_training_update(*self.refs, update)
        for role in ('native', 'final', 'reserved_evaluation'):
            with self.subTest(role=role), self.assertRaises(ValueError):
                data.load_reference_manifest(self.root, role)

    def test_codec_failure_has_no_silent_fallback(self):
        with mock.patch.object(cv2, 'imencode', return_value=(False, None)):
            with self.assertRaisesRegex(ValueError, 'JPEG encode failed'):
                data.make_training_update(*self.refs, 1)


class TrainSamplerV2Tests(unittest.TestCase):
    def setUp(self):
        self.ids = [f'tr_ffhq_{n:05d}' for n in range(391)]

    def test_full_schedule_counts_factorial_cycle_and_eligibility(self):
        sampler = data.TrainSampler(self.ids)
        selected, cycle, eligibility = [], Counter(), Counter()
        for update in range(1, 601):
            before = sampler.state_dict()
            selection = sampler.preview_update()
            self.assertEqual(before, sampler.state_dict())
            self.assertEqual(selection, sampler.preview_update())
            self.assertEqual(selection.update, update)
            self.assertNotEqual(selection.id_a, selection.id_b)
            selected.extend((selection.id_a, selection.id_b))
            spec = data.cycle_spec(update)
            cycle[spec['a_edge'], spec['b_variant']] += 1
            eligibility['normal'] += 1 + int(spec['a_edge'] >= 32)
            eligibility['dark'] += 1 + int(spec['a_edge'] >= 32)
            sampler.commit(selection)
        self.assertEqual(set(selected), set(self.ids))
        self.assertEqual(Counter(Counter(selected).values()), {3: 364, 4: 27})
        self.assertEqual(len(cycle), 24)
        self.assertEqual(set(cycle.values()), {25})
        self.assertEqual(eligibility, {'normal': 900, 'dark': 900})
        for start in (0, 391, 782):
            self.assertEqual(set(selected[start:start + 391]), set(self.ids))
        state = sampler.state_dict()
        self.assertEqual(state['completed_updates'], 600)
        self.assertEqual(state['selection_count'], 1200)
        self.assertEqual(state['next_update'], 601)
        with self.assertRaises(ValueError):
            sampler.preview_update()

    def test_commit_rejects_stale_or_tampered_selection(self):
        sampler = data.TrainSampler(self.ids)
        selection = sampler.preview_update()
        before = sampler.state_dict()
        with self.assertRaises(ValueError):
            sampler.commit(replace(selection, id_a=selection.id_b))
        self.assertEqual(before, sampler.state_dict())
        sampler.commit(selection)
        with self.assertRaises(ValueError):
            sampler.commit(selection)
        returned = sampler.state_dict()
        returned['cursor'] = -1
        self.assertNotEqual(returned, sampler.state_dict())

    def test_forced_shuffle_boundary_rotation_preserves_full_permutation(self):
        sampler = data.TrainSampler(self.ids)
        state = sampler.state_dict()
        probe = np.random.Generator(np.random.PCG64(data.SEED))
        next_permutation = probe.permutation(self.ids).tolist()
        duplicate = next_permutation[0]
        state.update(permutation=[i for i in self.ids if i != duplicate] + [duplicate],
                     cursor=390, permutation_index=0)
        sampler._state = state  # Synthetic fault fixture forces the rare boundary.
        selection = sampler.preview_update()
        self.assertEqual(selection.id_a, duplicate)
        self.assertEqual(selection.id_b, next_permutation[1])
        self.assertEqual(selection.rotation_events[0]['left_rotation'], 1)
        sampler.commit(selection)
        committed = sampler.state_dict()
        self.assertEqual(set(committed['permutation']), set(self.ids))
        self.assertEqual(len(committed['permutation']), 391)
        self.assertEqual(committed['permutation'][-1], duplicate)

    def test_canonical_input_order_and_invalid_namespace(self):
        left, right = data.TrainSampler(self.ids), data.TrainSampler(list(reversed(self.ids)))
        self.assertEqual(left.preview_update().to_record(), right.preview_update().to_record())
        for ids in (self.ids[:-1], self.ids[:-1] + [self.ids[0]],
                    ['va_ffhq_00000'] + self.ids[1:]):
            with self.assertRaises(ValueError):
                data.TrainSampler(ids)
        with self.assertRaises(ValueError):
            data.TrainSampler(self.ids, seed=17)


if __name__ == '__main__':
    unittest.main()
