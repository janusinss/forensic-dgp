"""Pure inherited/terminal gate fixtures; no model, gradient, worker or VM."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from research.codeformer_u64_png_checked_step_v1 import quality


def means(value=.1):
    return {view: {role: dict.fromkeys(quality.FIELDS, value) for role in quality.ROLES}
            for view in quality.VIEWS}


def inherited(passed=True, terminal=True):
    return {'pass': passed, 'legacy': {'pass': passed, 'teachers_reset': False},
            'ordinary_monitor': {'pass': True},
            'original_terminal': {'pass': terminal, 'binding': False, 'unchanged_gain_fraction': .01},
            'model_qualified': False}


class PNGCheckedStepQualityTests(unittest.TestCase):
    def evaluate(self, current=None, original=None, prior=None, gate=None):
        return quality.evaluate_quality(means(.09) if current is None else current,
            means() if original is None else original, means() if prior is None else prior,
            True, inherited() if gate is None else gate)

    def test_exact_inherited_payload_pin_and_same_complete_result(self):
        source = quality.ROOT / quality.INHERITED_SOURCE
        raw = source.read_bytes()
        self.assertEqual(len(raw), quality.INHERITED_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), quality.INHERITED_SHA256)
        module = quality.inherited_quality()
        current, original, gate = means(.09), means(), inherited()
        expected = module.evaluate_quality(current, original, original, True, gate)
        self.assertEqual(self.evaluate(current, original, original, gate), expected)

    def test_success_keeps_16_endpoints_32_comparisons_and_strict_four(self):
        result = self.evaluate()
        self.assertTrue(result['pass'])
        self.assertEqual(len(result['role_mean_nonincrease_checks']), 32)
        self.assertEqual(len(result['terminal_clear_lowlight_strict_MAE_checks']), 4)
        endpoints = {(r['view'], r['role'], r['field']) for r in result['role_mean_nonincrease_checks']}
        self.assertEqual(len(endpoints), 16)
        self.assertTrue(result['inherited_original_terminal_binding']['required'])
        self.assertTrue(result['zero_added_tolerance'])
        self.assertFalse(result['model_qualified'])

    def test_unchanged_original_is_not_terminal_success(self):
        result = self.evaluate(means(), means(), means())
        self.assertTrue(all(r['pass'] for r in result['role_mean_nonincrease_checks']))
        self.assertFalse(any(r['pass'] for r in result['terminal_clear_lowlight_strict_MAE_checks']))
        self.assertFalse(result['pass'])
        self.assertTrue(result['quality_stop'])

    def test_each_of_16_endpoints_has_zero_tolerance(self):
        for view in quality.VIEWS:
            for role in quality.ROLES:
                for field in quality.FIELDS:
                    current = means(.09)
                    current[view][role][field] = .1 + 1e-12
                    with self.subTest(view=view, role=role, field=field):
                        result = self.evaluate(current)
                        self.assertFalse(result['pass'])
                        bad = [r for r in result['role_mean_nonincrease_checks'] if not r['pass']]
                        self.assertEqual(len(bad), 2)

    def test_original_and_prior_comparisons_are_both_binding(self):
        current, original, prior = means(.09), means(), means(.08)
        result = self.evaluate(current, original, prior)
        self.assertFalse(result['pass'])
        bad = [r for r in result['role_mean_nonincrease_checks'] if not r['pass']]
        self.assertEqual(len(bad), 16)
        self.assertEqual({r['reference'] for r in bad}, {'prior_state'})

    def test_inherited_case_teacher_or_guard_failure_cannot_be_overridden(self):
        gate = inherited(False, True)
        result = self.evaluate(gate=gate)
        self.assertTrue(result['additional_role_mean_checks_pass'])
        self.assertFalse(result['pass'])
        self.assertEqual(result['inherited_gate_result'], gate)
        self.assertIsNot(result['inherited_gate_result'], gate)

    def test_original_one_percent_terminal_failure_is_binding(self):
        result = self.evaluate(gate=inherited(True, False))
        self.assertTrue(result['additional_role_mean_checks_pass'])
        self.assertFalse(result['pass'])
        self.assertEqual(result['inherited_original_terminal_binding'], {'required': True, 'pass': False})

    def test_terminal_cannot_be_disabled_or_supplied_as_truthy_scalar(self):
        for terminal in (False, 0, 1, None, 'true'):
            with self.subTest(value=terminal), self.assertRaises(ValueError):
                quality.evaluate_quality(means(.09), means(), means(), terminal, inherited())

    def test_missing_or_nonboolean_terminal_result_fails_closed(self):
        for gate in ({'pass': True}, {'pass': True, 'original_terminal': {}},
                     {'pass': True, 'original_terminal': {'pass': 1}},
                     {'pass': 1, 'original_terminal': {'pass': True}}):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                self.evaluate(gate=gate)

    def test_last_endpoint_invalid_value_is_validated_before_result(self):
        for bad in (None, float('nan'), float('inf'), -.01, True, '0.09'):
            current = means(.09)
            current['native_png'][quality.ROLES[-1]]['adjacent_rgb_error'] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.evaluate(current)

    def test_late_quality_failure_has_no_fallback_or_retry(self):
        current = means(.09)
        current['native_png'][quality.ROLES[-1]]['mae'] = .1
        result = self.evaluate(current)
        self.assertFalse(result['pass'])
        self.assertFalse(result['earlier_state_fallback_or_retry_authorized'])
        self.assertTrue(result['quality_stop'])
        self.assertEqual(len(result['terminal_clear_lowlight_strict_MAE_checks']), 4)

    def test_missing_extra_role_view_field_fails_closed(self):
        cases = []
        value = means(.09); del value['raw']; cases.append(value)
        value = means(.09); value['unexpected'] = {}; cases.append(value)
        value = means(.09); del value['raw'][quality.ROLES[0]]; cases.append(value)
        value = means(.09); del value['native_png'][quality.ROLES[-1]]['adjacent_rgb_error']; cases.append(value)
        for current in cases:
            with self.subTest(current=current), self.assertRaises(ValueError):
                self.evaluate(current)

    def test_inputs_and_inherited_result_are_not_modified(self):
        current, original, prior, gate = means(.09), means(), means(), inherited()
        before = copy.deepcopy((current, original, prior, gate))
        result = self.evaluate(current, original, prior, gate)
        self.assertEqual((current, original, prior, gate), before)
        result['inherited_gate_result']['legacy']['pass'] = False
        self.assertTrue(gate['legacy']['pass'])

    def test_changed_payload_or_wrong_import_origin_is_rejected(self):
        with mock.patch.object(quality, 'INHERITED_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'unchanged inherited quality source bytes'):
                quality.inherited_quality()
        module = quality.inherited_quality()
        with mock.patch.object(module, '__file__', str(quality.ROOT / 'wrong_quality.py')):
            with self.assertRaisesRegex(ValueError, 'import origin differs'):
                quality.inherited_quality()

    def test_inert_import_blocks_neural_frameworks_and_old_worker(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'torch','numpy','tensorflow','jax','PIL'} or fullname.endswith('.worker'):
   raise RuntimeError('Forbidden import:'+fullname)
  return None
sys.meta_path.insert(0,Block())
from research.codeformer_u64_png_checked_step_v1 import quality
assert 'research.codeformer_u64_gate_aligned_pilot_v1.quality' not in sys.modules
module=quality.inherited_quality()
assert not any(n.split('.')[0] in {'torch','numpy','tensorflow','jax','PIL'} or n.endswith('.worker') for n in sys.modules)
print('INERT_EXACT_QUALITY_ONLY')
'''
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'INERT_EXACT_QUALITY_ONLY')


if __name__ == '__main__':
    unittest.main()
