"""Pure synthetic selector/FP32 fixtures; no Torch, actual model or VM calls."""
import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from research.codeformer_u64_png_checked_step_v1 import selection as S


def metadata():
    rows = []
    for i in range(16):
        reference = 'photo_' + str(i).zfill(2)
        target = {'path': 'targets/' + reference + '.png', 'bytes': 100, 'sha256': 'a' * 64}
        for role in S.ROLES:
            rows.append({'id': reference + '__' + role, 'reference_id': reference,
                'role': role, 'split': 'TRAIN', 'cohort': 'optimize', 'optimization_permitted': True,
                'metric_region': 'central128' if role == S.ROLES[0] else 'whole',
                'input': {'path': 'inputs/' + reference + '__' + role + '.png', 'bytes': 101, 'sha256': 'b' * 64},
                'target': target.copy()})
    return rows


def bank(meta, value=.1, eligible=False):
    result = {}
    for ref in meta:
        result[ref['id']] = copy.deepcopy(ref) | {'metrics': {
            view: {field: value for field in S.FIELDS} for view in S.VIEWS}}
        if eligible and ref['role'] in S.ROLES[:2]:
            result[ref['id']]['metrics']['native_png']['adjacent_rgb_error'] = value - .001
    return result


def trials(meta, eligibility=(False, True, True, True)):
    return [{'maximum_absolute_coordinate_bound': cap, 'bank': bank(meta, eligible=ok)}
            for cap, ok in zip(S.CANDIDATE_MAXABS, eligibility)]


class SelectorTests(unittest.TestCase):
    def test_fixed_order_all_four_checked_first_eligible(self):
        meta = metadata(); initial = bank(meta); candidates = trials(meta)
        result = S.select_candidate(initial, candidates, meta)
        self.assertEqual(result['selected_index'], 1)
        self.assertEqual(result['selected_maximum_absolute_coordinate_bound'], 1.5e-5)
        self.assertEqual(len(result['trial_results']), 4)
        self.assertTrue(result['all_four_trials_evaluated'])
        self.assertTrue(all(len(r['all16_nonincrease_checks']) == 16 for r in result['trial_results']))

    def test_no_candidate_and_equality_is_not_strict_blur_gain(self):
        meta = metadata()
        result = S.select_candidate(bank(meta), trials(meta, (False,) * 4), meta)
        self.assertIsNone(result['selected_index'])
        self.assertTrue(result['no_eligible_candidate'])

    def test_first_eligible_does_not_hide_invalid_later_trial(self):
        meta = metadata(); candidates = trials(meta, (True,) * 4)
        del candidates[-1]['bank'][meta[-1]['id']]
        with self.assertRaises(ValueError):
            S.select_candidate(bank(meta), candidates, meta)

    def test_all16_endpoints_zero_tolerance_and_both_blurs_required(self):
        meta = metadata(); candidates = trials(meta, (True,) * 4)
        key = next(r['id'] for r in meta if r['role'] == 'clear')
        for candidate in candidates:
            candidate['bank'][key]['metrics']['raw']['mae'] = np.nextafter(.1, np.inf)
        # A one-ULP case change can vanish in mean16; use an actual one-ULP mean
        # increase by setting the whole clear cohort to nextafter(.1,+inf).
        for candidate in candidates:
            for ref in meta:
                if ref['role'] == 'clear':
                    candidate['bank'][ref['id']]['metrics']['raw']['mae'] = np.nextafter(.1, np.inf)
        result = S.select_candidate(bank(meta), candidates, meta)
        self.assertIsNone(result['selected_index'])
        candidates = trials(meta, (True,) * 4)
        for candidate in candidates:
            for ref in meta:
                if ref['role'] == S.ROLES[1]:
                    candidate['bank'][ref['id']]['metrics']['native_png']['adjacent_rgb_error'] = .1
        self.assertIsNone(S.select_candidate(bank(meta), candidates, meta)['selected_index'])

    def test_monitor_guard_extra_missing_target_and_ROI_changes_rejected(self):
        for mutate in ('monitor', 'guard', 'roi', 'target'):
            meta = metadata()
            if mutate in ('monitor', 'guard'):
                meta[0]['cohort'] = mutate
                meta[0]['optimization_permitted'] = False
            elif mutate == 'roi':
                meta[0]['metric_region'] = 'whole'
            else:
                meta[0]['target'] = None
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):
                    S.select_candidate(bank(meta), trials(meta), meta)
        meta = metadata(); candidates = trials(meta)
        candidates[0]['bank']['extra_monitor'] = copy.deepcopy(candidates[0]['bank'][meta[0]['id']])
        with self.assertRaises(ValueError):
            S.select_candidate(bank(meta), candidates, meta)

    def test_trial_provenance_nonfinite_reordered_or_extra_scales_reject(self):
        meta = metadata()
        for scenario in ('target', 'nan', 'reverse', 'append'):
            candidates = trials(meta)
            if scenario == 'target':
                candidates[0]['bank'][meta[0]['id']]['target']['sha256'] = 'c' * 64
            elif scenario == 'nan':
                candidates[0]['bank'][meta[0]['id']]['metrics']['native_png']['mae'] = float('nan')
            elif scenario == 'reverse':
                candidates.reverse()
            else:
                candidates.append(copy.deepcopy(candidates[0]))
            with self.subTest(scenario=scenario):
                with self.assertRaises(ValueError):
                    S.select_candidate(bank(meta), candidates, meta)

    def test_inputs_unchanged_and_no_monitor_argument_or_worker_claim(self):
        meta = metadata(); original = bank(meta); candidates = trials(meta)
        before = json.dumps([meta, original, candidates], sort_keys=True)
        result = S.select_candidate(original, candidates, meta)
        self.assertEqual(before, json.dumps([meta, original, candidates], sort_keys=True))
        self.assertFalse(result['worker_or_nine_application_journal_implemented'])
        self.assertFalse(result['manual_packet_ready'])
        with self.assertRaises(TypeError):
            S.select_candidate(original, candidates, meta, monitor_bank={})

    def test_quantization_crossing_fixture_can_reverse_continuous_error_order(self):
        # Target code128: raw .5001 is closer to target than raw .4999, but floor
        # code127 is identical; crossing at128/255 can suddenly change PNG error.
        target = np.float32(128) / np.float32(255)
        raw = np.array([target - .00001, target + .00002], dtype=np.float32)
        png = (raw * np.float32(255)).astype(np.uint8).astype(np.float32) / np.float32(255)
        self.assertLess(abs(float(raw[0] - target)), abs(float(raw[1] - target)))
        self.assertGreater(abs(float(png[0] - target)), abs(float(png[1] - target)))


class CandidateArithmeticTests(unittest.TestCase):
    def test_four_independent_original_proposals_fixed_headroom_strict_dots(self):
        original = np.array([.1, -.2], dtype=np.float32)
        direction = np.array([-1., -2.], dtype=np.float64)
        gates = np.ones((8, 2), dtype=np.float64)
        before = [a.tobytes() for a in (original, direction, gates)]
        for cap in S.CANDIDATE_MAXABS:
            result = S.propose_from_original_fp32(original, direction, gates, cap)
            cert = result['certificate']
            self.assertEqual(cert['alpha'], .99 * cap / 2.)
            np.testing.assert_array_equal(result['candidate'],
                (original.astype(np.float64) + cert['alpha'] * direction).astype(np.float32))
            self.assertLessEqual(np.max(np.abs(result['actual_delta'])), cap)
            self.assertTrue(all(x < 0 for x in cert['all8_actual_RAW_dots']))
            self.assertFalse(np.shares_memory(result['candidate'], original))
            self.assertFalse(cert['parameter_applied'])
        self.assertEqual(before, [a.tobytes() for a in (original, direction, gates)])

    def test_large_unregistered_oldscale_and_rounded_bad_sign_reject(self):
        original = np.array([.1], dtype=np.float32)
        for cap in (1e-4, 7.5e-5, 5e-5, 2.5e-5, 1.25e-5):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    S.propose_from_original_fp32(original, np.array([-1.]), np.ones((8, 1)), cap)
        with self.assertRaises(ValueError):
            S.propose_from_original_fp32(np.array([1e8], np.float32), np.array([-1.]), np.ones((8, 1)), 2e-5)
        with self.assertRaises(ValueError):
            S.propose_from_original_fp32(original, np.array([1.]), np.ones((8, 1)), 2e-5)

    def test_import_under_neural_blocker_no_model_or_VM_entry(self):
        root = Path(__file__).resolve().parents[1]
        code = '''import importlib.abc,sys,importlib
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','jax'}:raise RuntimeError('Forbidden framework')
  return None
sys.meta_path.insert(0,Block())
m=importlib.import_module('research.codeformer_u64_png_checked_step_v1.selection')
assert not any(n.split('.')[0] in {'torch','tensorflow','jax'} for n in sys.modules)
assert not any(hasattr(m,n) for n in ('run','worker','launch','build_model'))
print('PURE_SELECTOR_ONLY')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'PURE_SELECTOR_ONLY')


if __name__ == '__main__':
    unittest.main()
