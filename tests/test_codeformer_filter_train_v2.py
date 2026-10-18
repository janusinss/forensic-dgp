"""Pure CPU math, cohort boundaries, immutable IO and exact-sheet fixtures."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import time
import stat
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('filter_train_v2', ROOT / 'scripts/prepare_codeformer_filter_train_v2.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def profile(name):
    return copy.deepcopy(next(p for p in m.PROFILES if p['id'] == name))


def catalog_fixture():
    counts = [1, 5, 20, 35, 60, 70, 80, 90, 30]
    rows = []
    for group, count in enumerate(counts):
        for _ in range(count):
            index = len(rows); rid = 'tr_ffhq_%05d' % index
            source = hashlib.sha256(('source' + rid).encode()).hexdigest()
            target = hashlib.sha256(('target' + rid).encode()).hexdigest()
            rows.append({'id': rid, 'historical_role': 'train', 'source_sha256': source,
                'existing_target_sha256': target, 'native_size': [1024, 1024], 'existing_target_size': [256, 256],
                'decision': 'accept_clean_restoration', 'use_for_optimization': False,
                'attribution': {'license_url': 'https://example.test/license/' + str(group // 3)},
                'official_category': 'category' + str(group % 3),
                'original_source_path': 'scratch/fake_sources/' + rid + '.png',
                'original_existing_target_path': 'scratch/fake_targets/' + rid + '.png'})
    for index in range(53):
        rid = 'va_ffhq_%05d' % index
        rows.append({'id': rid, 'historical_role': 'validation',
            'source_sha256': hashlib.sha256(('dev' + rid).encode()).hexdigest()})
    return {'references': rows}


def selection_fixture(catalog):
    selected, allocations = m.selected_ids(catalog['references'])
    by_id = {r['id']: r for r in catalog['references']}
    cohort = []
    for rid in selected:
        r = by_id[rid]
        cohort.append({'reference_id': rid, 'split': 'TRAIN', 'historical_role': 'train',
            'use_for_optimization': False, 'reference_manifest_row': copy.deepcopy(r),
            'source1024': {'path': r['original_source_path'], 'sha256': r['source_sha256']},
            'target256': {'path': r['original_existing_target_path'], 'sha256': r['existing_target_sha256']},
            'attribution': copy.deepcopy(r['attribution']),
            'local_rectangle_xyxy_half_open': m.local_rectangle(r['source_sha256'])})
    strata = [{'license_url': key[0], 'official_category': key[1],
        'population': sum(m.stratum(r) == key for r in catalog['references'] if r['historical_role'] == 'train'),
        'selected': value} for key, value in sorted(allocations.items())]
    return {'format': 'codeformer-filter-train-v2-prospective-source-cohort-v1', 'complete': True,
        'catalog_counts': {'TRAIN': 391, 'DEV': 53, 'total': 444}, 'strata': strata, 'cohort': cohort}


class FilterTrainV2Fixtures(unittest.TestCase):
    def test_exact8profiles_and_review_order_keep_clear_last(self):
        self.assertEqual(len(m.PROFILES), 8)
        self.assertEqual(tuple(p['id'] for p in m.PROFILES), m.PROFILE_IDS)
        self.assertEqual(m.INPUT_REVIEW_ORDER[-1], 'clear')
        self.assertEqual(m.INPUT_REVIEW_ORDER[:4], ('global_mosaic_8', 'local_mosaic_8',
            'local_gaussian_3p0', 'repeated_rows_4_blend0p75'))
        self.assertEqual({p['id'] for p in m.PROFILES if p['strong_control']}, {'repeated_rows_4_blend0p75'})

    def test_approved_profiles_match_frozen_root_contract_without_generation(self):
        path = ROOT / 'outputs/codeformer_filter_train_v2_preparation/approved_profiles_v1.json'
        self.assertEqual(m.read_json(path.read_bytes()), m.profiles_document())

    def test_strict_JSON_rejects_duplicates_nonfinite_and_invalid_encoding(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeDecodeError)):
                m.read_json(raw)

    def test_every_TRAIN_case_stays_pending_and_rows_stay_excluded(self):
        for pid in m.PROFILE_IDS:
            entry = m.training_admission(pid)
            self.assertTrue(entry['optimization_allowed_by_historical_split'])
            self.assertFalse(entry['optimization_permitted_now'])
            self.assertIn('PENDING', entry['restoration_training_eligibility'])
            self.assertEqual(entry['strong_control_excluded_initially'], pid == 'repeated_rows_4_blend0p75')

    def test_rectangle_hash_rule_matches_independent_byte_formula_and_varies(self):
        positions = set()
        for i in range(32):
            source_sha = hashlib.sha256(str(i).encode()).hexdigest()
            raw = hashlib.sha256(('codeformer-filter-train-v2/rectangle/' + source_sha).encode()).digest()
            x, y = raw[0] % 9 * 16, raw[1] % 9 * 16
            self.assertEqual(m.local_rectangle(source_sha), [x, y, x + 128, y + 128])
            positions.add((x, y))
        self.assertGreater(len(positions), 16)
        for invalid in ('bad', 'A' * 64, '0' * 63):
            with self.assertRaisesRegex(ValueError, 'Canonical'):
                m.local_rectangle(invalid)

    def test_moving_local_filters_preserve_exact_outside_and_prior_global_constant(self):
        yy, xx = np.indices((256, 256))
        image = np.stack([(xx * 37 + yy * 11) % 256, xx % 256, yy % 256], axis=-1).astype(np.uint8)
        original = image.copy(); old_rectangle = m.FILTERS.LOCAL_RECT
        for rect in ([0, 0, 128, 128], [128, 128, 256, 256], [32, 96, 160, 224]):
            x0, y0, x1, y1 = rect
            expected = np.zeros((256, 256), bool); expected[y0:y1, x0:x1] = True
            for pid in ('local_gaussian_3p0', 'local_mosaic_8'):
                result, footprint = m.apply_profile(image, profile(pid), rect)
                self.assertTrue(np.array_equal(footprint > 0, expected))
                self.assertEqual(np.count_nonzero(footprint), 16384)
                self.assertTrue(np.array_equal(result[~expected], image[~expected]))
                self.assertFalse(np.array_equal(result[expected], image[expected]))
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(m.FILTERS.LOCAL_RECT, old_rectangle)

    def test_local_gaussian_uses_full_canvas_neighbors_not_crop_boundary(self):
        image = np.zeros((256, 256, 3), np.uint8); image[:, 15] = 255
        result, _ = m.apply_profile(image, profile('local_gaussian_3p0'), [16, 16, 144, 144])
        self.assertGreater(int(result[32, 16, 0]), 0)
        self.assertEqual(int(result[32, 15, 0]), 255)

    def test_mosaic_and_repeated_rows_match_hand_calculated_values(self):
        yy, xx = np.indices((256, 256), dtype=np.uint8)
        image = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
        result, _ = m.apply_profile(image, profile('local_mosaic_8'), [32, 96, 160, 224])
        self.assertTrue(np.all(result[96:104, 32:40] == [36, 100, 0]))
        self.assertTrue(np.all(result[96:104, 40:48] == [44, 100, 0]))
        result, _ = m.apply_profile(image, profile('repeated_rows_4_blend0p75'), [0, 0, 128, 128])
        self.assertEqual(result[:8, 0, 1].tolist(), [0, 0, 1, 1, 4, 4, 5, 5])

    def test_constant_images_preserved_all_profiles_and_profiles_not_mutated(self):
        image = np.full((256, 256, 3), [17, 128, 231], np.uint8)
        before = copy.deepcopy(m.PROFILES)
        for spec in m.PROFILES:
            result, _ = m.apply_profile(image, spec, [0, 128, 128, 256])
            self.assertTrue(np.array_equal(result, image))
        self.assertEqual(before, m.PROFILES)
        bad = profile('gaussian_2p4'); bad['parameters']['sigma_x'] = 2.5
        with self.assertRaisesRegex(ValueError, 'approved'):
            m.apply_profile(image, bad, [0, 0, 128, 128])

    def test_nonlocal_filter_arithmetic_equals_bound_historical_implementation(self):
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        for spec in (p for p in m.PROFILES if not p['operator'].startswith('local_')):
            result, footprint = m.apply_profile(image, spec, [0, 0, 128, 128])
            old, old_footprint = m.FILTERS.apply_profile(image, spec)
            self.assertTrue(np.array_equal(result, old))
            self.assertTrue(np.array_equal(footprint, old_footprint))

    def test_hamilton_reserve1_allocation_and_membership_order_are_reproducible(self):
        catalog = catalog_fixture()
        ids, counts = m.selected_ids(catalog['references'])
        self.assertEqual(list(v for _, v in sorted(counts.items())), [1, 2, 4, 6, 9, 11, 12, 14, 5])
        self.assertEqual(len(ids), 64); self.assertEqual(ids, sorted(ids))
        shuffled = list(reversed(catalog['references']))
        self.assertEqual(m.selected_ids(shuffled), (ids, counts))
        self.assertEqual(m.validate_selection(selection_fixture(catalog), catalog)[0][0]['reference_id'], ids[0])

    def test_selection_rejects_DEV_role_attribution_rectangle_and_membership_tampering(self):
        catalog = catalog_fixture(); selection = selection_fixture(catalog)
        mutations = [lambda s: s['cohort'][0].update(split='DEV'),
            lambda s: s['cohort'][0].update(reference_id='tr_ffhq_99999'),
            lambda s: s['cohort'][0]['attribution'].update(license_url='different'),
            lambda s: s['cohort'][0].update(local_rectangle_xyxy_half_open=[1, 1, 129, 129]),
            lambda s: s['cohort'][0].update(use_for_optimization=True)]
        for mutate in mutations:
            tampered = copy.deepcopy(selection); mutate(tampered)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                m.validate_selection(tampered, catalog)

    def test_source_digest_and_singlelink_guard_reject_substitution(self):
        with tempfile.TemporaryDirectory(prefix='filter-train-v2-read-', dir=ROOT / 'scratch') as directory:
            path = Path(directory) / 'source.bin'; path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            self.assertEqual(m.read_bound(path, expected, 8), b'original')
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                m.read_bound(path, expected, 8)
            # Windows sandbox forbids creating a hardlink. Inject only the OS
            # stat response so the producer's non-single-link guard is exercised.
            info = SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=2)
            with mock.patch.object(m, 'safe_parents', return_value=path), \
                    mock.patch.object(m.Path, 'stat', return_value=info), \
                    self.assertRaisesRegex(ValueError, 'single-link'):
                m.read_bound(path, hashlib.sha256(b'changed!').hexdigest(), 8)

    def test_writer_exact_namespace_traversal_existing_and_foreign_file_guards(self):
        with tempfile.TemporaryDirectory(prefix='filter-train-v2-write-', dir=ROOT / 'outputs') as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'New exclusive'):
                m.Writer(root)
            writer = m.Writer(root / 'new')
            for bad in ('../bad', 'a//b', 'a/./b', '/absolute', 'C:bad', 'a\\b'):
                with self.subTest(path=bad), self.assertRaisesRegex(ValueError, 'Canonical'):
                    writer.put(bad, b'x')
            writer.put('nested/evidence.bin', b'first'); writer.close_namespace()
            with self.assertRaisesRegex(ValueError, 'Finite unique'):
                writer.put('nested/evidence.bin', b'second')
            (writer.root / 'foreign.bin').write_bytes(b'foreign')
            with self.assertRaisesRegex(ValueError, 'namespace closure'):
                writer.close_namespace()

    def test_writer_reserves_failure_receipt_despite_clock_stop(self):
        with tempfile.TemporaryDirectory(prefix='filter-train-v2-stop-', dir=ROOT / 'outputs') as directory:
            writer = m.Writer(Path(directory) / 'new', time.monotonic() - m.MAX_SECONDS - 1)
            with self.assertRaisesRegex(ValueError, '600-second'):
                writer.put('normal.bin', b'no')
            writer.put('failure.json', m.json_bytes({'complete': False}), failure=True)
            self.assertTrue((writer.root / 'failure.json').is_file())

    def test_input_sheet_has16_exact_cells_without_target_panels(self):
        inputs = [('tr_ffhq_%05d' % i, np.full((256, 256, 3), i * 13, np.uint8)) for i in range(16)]
        data, cells = m.input_only_sheet('global_mosaic_8', inputs, 2)
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (1024, 1184)); saved = np.asarray(image)
        self.assertEqual(len(cells), 16)
        for i, cell in enumerate(cells):
            self.assertEqual((cell['width'], cell['height']), (256, 256))
            tile = saved[cell['y']:cell['y'] + 256, cell['x']:cell['x'] + 256]
            self.assertTrue(np.all(tile == i * 13))

    def test_direct512_is_not_upscaled256_and_pixel_proofs_cover_float32(self):
        yy, xx = np.indices((1024, 1024)); pixels = ((xx * 23 + yy * 17 + (xx % 7) * (yy % 13)) % 256).astype(np.uint8)
        image = Image.fromarray(np.stack([pixels] * 3, axis=-1))
        direct = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS))
        indirect = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.LANCZOS))
        self.assertFalse(np.array_equal(direct, indirect))
        proof = m.pixel_record(direct)
        self.assertEqual(proof['float32_sha256'], hashlib.sha256((direct.astype(np.float32) / np.float32(255)).tobytes()).hexdigest())

    def test_no_neural_framework_is_imported_or_allowed(self):
        self.assertFalse(any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')))
        with mock.patch.dict(sys.modules, {'torch': object()}):
            with self.assertRaisesRegex(ValueError, 'No neural framework'):
                m.no_neural_framework()


if __name__ == '__main__':
    unittest.main()
