"""Pure CPU contract fixtures; no neural imports, camera generation or VM."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import data as d


def fixture():
    ids = ['tr_ffhq_03212'] + [f'tr_ffhq_{4000 + i:05d}' for i in range(56)]
    rows = []
    for profile, count in {'clear': 57, **d.PROFILE_COUNTS}.items():
        for rid in ids[:count]:
            rows.append({'id': rid + '__' + profile, 'reference_id': rid, 'profile': profile,
                         'candidate_for_finite_packet_preparation': True,
                         'root_input_decision': 'readable', 'root_source_clear_decision': 'readable',
                         'second_observer_global_decision': 'readable' if profile == 'global_mosaic_8' else None,
                         'hold_reasons': [], 'optimization_permitted_now': False})
    rows += [{'id': 'held_' + str(i), 'reference_id': ids[0], 'profile': 'held',
              'candidate_for_finite_packet_preparation': False,
              'optimization_permitted_now': False} for i in range(512 - len(rows))]
    holds = {'format': 'codeformer-filter-train-v2-conservative-candidate-holds-v1',
             'complete': True, 'all512_cases': rows}
    dark = [{'id': 'v9_' + rid + '_lowlight_lr32', 'reference_id': rid,
             'historical_role': 'train', 'profile': 'lowlight_lr32'} for rid in ids]
    return holds, dark


class Contract(unittest.TestCase):
    def test_linear_dark_formula_endpoints_and_midtones(self):
        import numpy as np
        samples = np.asarray([[[0, 255, 128], [10, 11, 64], [32, 192, 240]]], dtype=np.uint8)
        expected = []
        for value in samples.reshape(-1):
            srgb = int(value) / 255.
            linear = srgb / 12.92 if srgb <= .04045 else ((srgb + .055) / 1.055) ** 2.4
            scaled = linear * .35
            result = 12.92 * scaled if scaled <= .0031308 else 1.055 * scaled ** (1 / 2.4) - .055
            expected.append(int(result * 255 + .5))
        # Exactfloat64 evaluation of10/255 inverse/forward liesjustbelow3.5;
        # no tolerance, epsilon, decimal arithmetic or tie override is added.
        self.assertEqual(expected, [0, 160, 78, 3, 4, 36, 16, 119, 150])
        np.testing.assert_array_equal(d.linear_dark_proxy(samples), np.asarray(expected, np.uint8).reshape(samples.shape))

    def test_linear_dark_monotone_no_geometry_change(self):
        import numpy as np
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)
        before = ramp.copy()
        dark = d.linear_dark_proxy(ramp)
        self.assertEqual(dark.shape, ramp.shape)
        self.assertTrue(dark.flags.c_contiguous)
        self.assertTrue(np.all(np.diff(dark[:, 0, 0].astype(int)) >= 0))
        self.assertTrue(np.all(dark <= ramp))
        np.testing.assert_array_equal(ramp, before)

    def test_linear_dark_rejects_float(self):
        import numpy as np
        with self.assertRaises(ValueError):
            d.linear_dark_proxy(np.zeros((2, 2, 3), np.float32))

    def test_import_no_neural_runtime(self):
        self.assertNotIn('torch', sys.modules)

    def test_exact_candidate_census(self):
        h, _ = fixture()
        self.assertEqual(len(d.validate_holds(h)), 370)

    def test_historical_authorization_not_mutated(self):
        h, dark = fixture()
        before = copy.deepcopy(h)
        d.build_schedule(h, dark)
        self.assertEqual(h, before)
        h['all512_cases'][0]['optimization_permitted_now'] = True
        with self.assertRaises(ValueError):
            d.validate_holds(h)

    def test_held_case_cannot_enter_candidates(self):
        h, _ = fixture()
        h['all512_cases'][0]['hold_reasons'] = ['pose_pending']
        with self.assertRaises(ValueError):
            d.validate_holds(h)

    def test_second_observer_global_must_be_readable(self):
        h, _ = fixture()
        next(r for r in h['all512_cases'] if r['profile'] == 'global_mosaic_8')['second_observer_global_decision'] = 'borderline'
        with self.assertRaises(ValueError):
            d.validate_holds(h)

    def test_all512_preserved(self):
        h, _ = fixture()
        h['all512_cases'].pop()
        with self.assertRaises(ValueError):
            d.validate_holds(h)

    def test_complete_single_pass_and_exact_padding(self):
        h, dark = fixture()
        schedule = d.build_schedule(h, dark)
        filtered = [s for u in schedule['updates'] for s in u['slots'][:2]]
        counts = Counter(r['id'] for r in filtered)
        self.assertEqual(len(filtered), 314)
        self.assertEqual(len(counts), 313)
        self.assertEqual({k: v for k, v in counts.items() if v != 1}, {d.PADDING_ID: 2})
        self.assertEqual(sum(r['padding_duplicate'] for r in filtered), 1)

    def test_order_independent_and_deterministic(self):
        h, dark = fixture()
        a = d.build_schedule(h, dark)
        h['all512_cases'].reverse()
        self.assertEqual(a, d.build_schedule(h, list(reversed(dark))))

    def test_anchor_source_balance_and_pairing(self):
        h, dark = fixture()
        schedule = d.build_schedule(h, dark)
        counts = Counter(u['slots'][2]['reference_id'] for u in schedule['updates'])
        self.assertEqual(Counter(counts.values()), {3: 43, 2: 14})
        self.assertTrue(all(u['slots'][2]['reference_id'] == u['slots'][3]['reference_id']
                            for u in schedule['updates']))

    def test_dark_join_gap_not_replaced(self):
        h, dark = fixture()
        with self.assertRaises(ValueError):
            d.build_schedule(h, dark[:-1])
        dark[0]['historical_role'] = 'validation'
        with self.assertRaises(ValueError):
            d.build_schedule(h, dark)

    def test_preview_pure_commit_explicit(self):
        h, dark = fixture()
        sampler = d.FiniteSampler(d.build_schedule(h, dark))
        before = sampler.state()
        a, b = sampler.preview(), sampler.preview()
        self.assertEqual(a, b)
        self.assertEqual(before, sampler.state())
        sampler.commit(a)
        self.assertEqual(sampler.state()['committed_microbatches'], 4)
        with self.assertRaises(ValueError):
            sampler.commit(a)

    def test_changed_or_partial_preview_rejected(self):
        h, dark = fixture()
        sampler = d.FiniteSampler(d.build_schedule(h, dark))
        preview = sampler.preview()
        preview['slots'].pop()
        with self.assertRaises(ValueError):
            sampler.commit(preview)

    def test_finite_no_automatic_extension(self):
        h, dark = fixture()
        sampler = d.FiniteSampler(d.build_schedule(h, dark))
        for _ in range(157):
            sampler.commit(sampler.preview())
        self.assertEqual(sampler.state()['committed_microbatches'], 628)
        with self.assertRaises(ValueError):
            sampler.preview()

    def test_paths_reject_traversal_and_absolute(self):
        for bad in ('../x', 'a/../x', '/x', 'C:/x', 'a\\x', 'a//x', 'a/./x', ''):
            with self.assertRaises(ValueError, msg=bad):
                d.relative(bad)

    def test_exact_bytes_guard_and_hash_mismatch(self):
        scratch = (ROOT / 'scratch').resolve()
        self.assertTrue(scratch.is_relative_to(ROOT.resolve()))
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='filter-fusion-fixture-') as temp:
            root = Path(temp)
            self.assertTrue(root.resolve().is_relative_to(scratch))
            (root / 'x').write_bytes(b'data')
            pin = {'path': 'x', 'bytes': 4, 'sha256': hashlib.sha256(b'data').hexdigest()}
            self.assertEqual(d.verified_bytes(root, pin), b'data')
            pin['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                d.verified_bytes(root, pin)

    def test_root_reparse_guard(self):
        class Reparse:
            st_file_attributes = 0x400
        with patch.object(Path, 'lstat', return_value=Reparse()), patch.object(Path, 'is_symlink', return_value=False):
            with self.assertRaisesRegex(ValueError, 'root or ancestor'):
                d.verified_bytes(ROOT, {'path': 'x', 'bytes': 1, 'sha256': '0' * 64})

    def test_explicit_admission_required(self):
        h, dark = fixture()
        schedule = d.build_schedule(h, dark)
        with self.assertRaises(ValueError):
            d.require_admission({}, 'a', 'b', schedule)
        cases = {(s['role'], s['id']) for u in schedule['updates'] for s in u['slots']}
        admission = {'format': 'codeformer-filter-fusion-explicit-admission-v1', 'complete': True,
                     'root_approved': True, 'candidate_holds_sha256': 'a',
                     'dark_input_review_sha256': 'b', 'schedule_sha256': d.digest(schedule),
                     'admitted_role_cases': [{'role': role, 'id': case} for role, case in sorted(cases)],
                     'held512_ledger_unchanged': True, 'historical_DEV_FINAL_native_excluded': True,
                     'automatic_launch_authorized': False}
        self.assertTrue(d.require_admission(admission, 'a', 'b', schedule))
        admission['admitted_role_cases'].pop()
        with self.assertRaises(ValueError):
            d.require_admission(admission, 'a', 'b', schedule)


if __name__ == '__main__':
    unittest.main()
