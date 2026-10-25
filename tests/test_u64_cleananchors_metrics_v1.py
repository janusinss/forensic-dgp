"""Pure saved-bank and NumPy fixtures; no model, Torch, worker or VM execution."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from research.codeformer_u64_cleananchors_continuation_v1 import metrics as M


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json'
AUDIT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/saved_pixel_audit_v1.json'


class ContinuationMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_bytes = PARENT.read_bytes()
        cls.audit_bytes = AUDIT.read_bytes()
        cls.protocol = json.loads(cls.parent_bytes)
        cls.controls = M.historical_controls(cls.parent_bytes, cls.audit_bytes)
        cls.monitor_rows = [row for row in cls.protocol['cases'] if row['cohort'] == 'monitor']

    def setUp(self):
        self.initial = copy.deepcopy(self.controls['saved_u64'])
        self.reference = copy.deepcopy(self.controls['frozen_U96'])

    def evaluate(self, current=None, phase='u16', **changes):
        args = {'current': self.improving() if current is None else current,
                'initial_u64': self.initial, 'frozen_U96': self.reference,
                'parent_bytes': self.parent_bytes, 'phase': phase,
                'saved_audit_bytes': self.audit_bytes}
        args.update(changes)
        return M.evaluate_bank(**args)

    def improving(self):
        current = copy.deepcopy(self.initial)
        for row in self.monitor_rows:
            record = current[row['id']]
            for view in M.VIEWS:
                for region in M.REGIONS:
                    for field in M.FIELDS:
                        record['regional_metrics'][view][region][field] *= .999
                record['metrics'][view] = record['regional_metrics'][view]['whole'].copy()
        return current

    def set_role(self, bank, role, view, field, transform, region='whole'):
        for row in self.monitor_rows:
            if row['role'] != role:
                continue
            record = bank[row['id']]
            record['regional_metrics'][view][region][field] = transform(row['id'])
            if region == 'whole':
                record['metrics'][view][field] = record['regional_metrics'][view][region][field]

    def test_actual_saved_initial_bank_joins_all56_and_retains_original_gains(self):
        result = self.evaluate(copy.deepcopy(self.initial), phase='initializer')
        self.assertTrue(result['pass'])
        self.assertEqual(result['complete_quality_cases'], 56)
        self.assertTrue(result['initial_u64_full_metric_join'])
        self.assertTrue(result['frozen_U96_full_metric_join'])
        self.assertFalse(result['inherited_proposal_extras_applied'])
        self.assertFalse(result['historical_broader_failures_cleared'])
        self.assertFalse(result['model_qualified'])
        self.assertFalse(result['original_terminal']['binding'])

    def test_complete_improving_terminal_passes_original_and_additional_gates(self):
        result = self.evaluate()
        self.assertTrue(result['pass'])
        self.assertTrue(result['legacy']['pass'])
        self.assertTrue(result['ordinary_monitor']['pass'])
        self.assertTrue(result['original_terminal']['binding'])
        self.assertTrue(result['original_terminal']['pass'])
        self.assertTrue(result['additional_retention']['pass'])

    def test_unchanged_u64_is_not_a_terminal_learning_pass(self):
        result = self.evaluate(copy.deepcopy(self.initial))
        self.assertTrue(result['ordinary_monitor']['pass'])
        self.assertTrue(result['original_terminal']['pass'])
        self.assertFalse(result['additional_retention']['pass'])
        self.assertFalse(result['pass'])

    def test_all_metric_fields_of_initial_and_frozen_controls_must_match(self):
        for field in ('mae', 'adjacent_rgb_error'):
            changed = copy.deepcopy(self.initial)
            key = self.monitor_rows[0]['id']
            changed[key]['regional_metrics']['raw']['outside'][field] += 1e-9
            with self.assertRaisesRegex(ValueError, 'exact saved historical initial_u64'):
                self.evaluate(initial_u64=changed)
        changed = copy.deepcopy(self.reference)
        key = next(row['id'] for row in self.protocol['guards'] if row['role'] == 'dark')
        changed[key]['U96_illumination_distance']['native_png'] = 1e-9
        with self.assertRaisesRegex(ValueError, 'exact saved historical frozen_U96'):
            self.evaluate(frozen_U96=changed)

    def test_initializer_cannot_be_another_current_bank(self):
        with self.assertRaisesRegex(ValueError, 'Initializer phase'):
            self.evaluate(phase='initializer')

    def test_original_monitor_baseline_is_U96_and_legacy_teachers_are_not_rebased(self):
        result = self.evaluate()
        checks = result['ordinary_monitor']['groups']['clear']['views']['raw']['whole/mae']
        self.assertEqual(checks['initial'], {
            row['id']: self.reference[row['id']]['metrics']['raw']['mae']
            for row in self.monitor_rows if row['role'] == 'clear'})
        self.assertNotEqual(checks['initial'], {
            row['id']: self.initial[row['id']]['metrics']['raw']['mae']
            for row in self.monitor_rows if row['role'] == 'clear'})
        current = self.improving()
        key = next(row['id'] for row in self.protocol['guards'] if row['role'] == 'clear')
        current[key]['behavior_distance']['raw'] = .5
        result = self.evaluate(current)
        self.assertFalse(result['legacy']['pass'])
        self.assertFalse(result['pass'])
        self.assertFalse(result['legacy']['thresholds_reset_to_U96'])

    def test_all24_legacy_and32_monitor_rows_remain_mandatory(self):
        for key in (self.protocol['guards'][0]['id'], self.monitor_rows[0]['id']):
            current = self.improving()
            del current[key]
            with self.assertRaisesRegex(ValueError, 'Complete exact56'):
                self.evaluate(current)
        current = list(self.improving().values())
        current.append(copy.deepcopy(current[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate record ID'):
            self.evaluate(current)

    def test_terminal_dark_adjacent_absolute_U96_requirement_is_binding(self):
        current = self.improving()
        role = M.ROLES[-1]
        self.set_role(current, role, 'native_png', 'adjacent_rgb_error',
                      lambda key: self.reference[key]['metrics']['native_png']['adjacent_rgb_error'] + 1e-6)
        result = self.evaluate(current)
        self.assertTrue(result['ordinary_monitor']['pass'])
        self.assertTrue(result['additional_retention']['pass'])
        self.assertFalse(result['original_terminal']['pass'])
        self.assertFalse(result['pass'])

    def test_original_one_percent_cannot_hide_losing_saved_u64_gain(self):
        current = self.improving()
        role = M.ROLES[0]
        self.set_role(current, role, 'native_png', 'adjacent_rgb_error',
                      lambda key: self.initial[key]['regional_metrics']['native_png']['central128']['adjacent_rgb_error'] * 1.00001,
                      region='central128')
        result = self.evaluate(current)
        self.assertTrue(result['original_terminal']['pass'])
        self.assertFalse(result['additional_retention']['pass'])
        self.assertFalse(result['pass'])

    def test_u8_clear_PNG_equality_is_not_improvement_and_u4_has_original_guards(self):
        current = self.improving()
        self.set_role(current, 'clear', 'native_png', 'mae',
                      lambda key: self.initial[key]['metrics']['native_png']['mae'])
        result = self.evaluate(current, phase='u8')
        self.assertFalse(result['pass'])
        self.assertTrue(self.evaluate(copy.deepcopy(self.initial), phase='u4')['pass'])
        key = next(row['id'] for row in self.monitor_rows if row['role'] == 'clear')
        current[key]['U96_behavior_distance']['native_png'] = 1.0
        self.assertFalse(self.evaluate(current, phase='u4')['pass'])

    def test_saved_audit_parent_or_dependency_pin_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError, 'independently audited u64 pixel receipt'):
            M.historical_controls(self.parent_bytes, self.audit_bytes + b' ')
        with self.assertRaisesRegex(ValueError, 'retained standardized parent'):
            M.historical_controls(self.parent_bytes + b' ', self.audit_bytes)
        with mock.patch.object(M, 'INHERITED_METRICS_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'reviewed inherited metric source'):
                self.evaluate()

    def test_null_legacy_dark_and_nonfinite_metric_schemas_are_enforced(self):
        current = self.improving()
        key = next(row['id'] for row in self.protocol['guards'] if row['role'] == 'dark')
        current[key]['metrics']['raw']['mae'] = 0.0
        with self.assertRaisesRegex(ValueError, 'nulls must remain null'):
            self.evaluate(current)
        for bad in (True, float('nan'), float('inf'), -.1):
            current = self.improving()
            key = self.monitor_rows[0]['id']
            current[key]['U96_illumination_distance']['raw'] = bad
            with self.assertRaises(ValueError):
                self.evaluate(current)

    def test_ordered_eight_means_are_insertion_independent_and_correct_region(self):
        current = self.improving()
        direct = M.cohort_means(current, self.parent_bytes)
        reverse = M.cohort_means(dict(reversed(list(current.items()))), self.parent_bytes)
        self.assertEqual(direct, reverse)
        for role in M.ROLES:
            ids = sorted(row['id'] for row in self.monitor_rows if row['role'] == role)
            region = 'central128' if role == M.ROLES[0] else 'whole'
            for view in M.VIEWS:
                for field in M.FIELDS:
                    total = 0.0
                    for key in ids:
                        total = total + current[key]['regional_metrics'][view][region][field]
                    self.assertEqual(direct[view][role][field], total / 8)

    def test_record_metrics_requires_original_control_and_preserves_both_domains(self):
        raw = np.full((256, 256, 3), .25, np.float32)
        baseline = np.full_like(raw, .125)
        png = np.full((256, 256, 3), 63, np.uint8)
        base_png = np.full_like(png, 31)
        result = M.record_metrics(raw, png, baseline, {'role': 'clear'}, np=np,
                                  frozen_U96_raw=baseline, frozen_U96_png=base_png)
        self.assertEqual(result['U96_behavior_distance']['raw'], .125)
        self.assertNotEqual(result['U96_behavior_distance']['native_png'], .125)
        with self.assertRaisesRegex(ValueError, 'frozen U96 control arrays'):
            M.record_metrics(raw, png, baseline, {'role': 'clear'}, np=np,
                             frozen_U96_raw=None, frozen_U96_png=None)

    def test_training_records_do_not_invent_uncached_U96_movement(self):
        raw = np.full((256, 256, 3), .25, np.float32)
        target = np.full_like(raw, .125)
        png = np.full((256, 256, 3), 63, np.uint8)
        row = copy.deepcopy(next(r for r in self.protocol['cases']
                                 if r['cohort'] == 'optimize' and r['role'] == 'clear'))
        result = M.record_training_metrics(raw, png, target, row, np=np)
        self.assertEqual(result['metrics']['raw']['mae'], .125)
        self.assertFalse(result['behavior_movement_reference_available'])
        self.assertNotIn('U96_behavior_distance', result)
        self.assertNotIn('U96_illumination_distance', result)
        self.assertNotIn('behavior_distance', result)
        for field, value in (('cohort', 'monitor'), ('optimization_permitted', False),
                             ('split', 'DEV'), ('role', 'dark'), ('target', None),
                             ('behavior_teacher', {'unadmitted': True})):
            changed = copy.deepcopy(row)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, 'paired TRAIN optimization'):
                M.record_training_metrics(raw, png, target, changed, np=np)
        with self.assertRaisesRegex(ValueError, 'paired TRAIN optimization'):
            M.record_training_metrics(raw, png, None, row, np=np)

    def test_inert_syntax_does_not_import_Torch_or_call_worker(self):
        ast.parse(inspect.getsource(M), feature_version=(3, 10))
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(self.controls['cached_pixels_redecoded'])
        self.assertTrue(self.controls['scalar_evidence_only'])


if __name__ == '__main__':
    unittest.main()
