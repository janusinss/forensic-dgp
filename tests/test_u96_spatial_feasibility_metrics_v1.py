"""Saved-array/scalar gate regression tests; no model, Torch or VM execution."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

from research.codeformer_u96_spatial_feasibility_v1 import contract as C
from research.codeformer_u96_spatial_feasibility_v1 import metrics as M


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json'


class PixelMetrics(unittest.TestCase):
    def setUp(self):
        self.target = np.full((256, 256, 3), .25, np.float32)

    def test_constant_tone_and_null_target_are_separate(self):
        value = self.target + .125
        self.assertEqual(M.pixel_metrics(value, self.target, np=np),
                         {'mae': .125, 'adjacent_rgb_error': 0.})
        self.assertEqual(M.pixel_metrics(value, None, np=np), dict.fromkeys(M.FIELDS))
        self.assertIsNone(M.regional_metrics(value, None, np=np))

    def test_half_open_ROI_does_not_count_boundary_as_detail(self):
        value = self.target.copy()
        value[64:192, 64:192] += .125
        regions = M.regional_metrics(value, self.target, np=np)
        self.assertEqual(regions['central128'], {'mae': .125, 'adjacent_rgb_error': 0.})
        self.assertEqual(regions['outside'], {'mae': 0., 'adjacent_rgb_error': 0.})
        self.assertGreater(regions['whole']['adjacent_rgb_error'], 0.)

    def test_whole_adjacent_is_exact_old_mean_of_two_means(self):
        y, x = np.indices((256, 256))
        value = np.repeat((.1 + .25*x/255 + .5*y/255)[:, :, None], 3, axis=2)
        residual = value - self.target.astype(np.float64)
        expected = float(.5*(np.abs(residual[:, 1:]-residual[:, :-1]).mean()+
                             np.abs(residual[1:]-residual[:-1]).mean()))
        self.assertEqual(M.pixel_metrics(value, self.target, np=np)['adjacent_rgb_error'], expected)

    def test_illumination_averages_within_actual16pixel_blocks(self):
        value = self.target.copy()
        value[0:16, 0:16] += .125
        self.assertAlmostEqual(M.illumination_distance(value, self.target, np=np), .125/256, places=16)
        self.assertEqual(M.behavior_distance(value, self.target, np=np), .125/256)
        # Equal positive/negative intra-block changes cancel in the illumination
        # check, while the full RGB behavior distance must still detect them.
        value = self.target.copy()
        value[::2] += .125
        value[1::2] -= .125
        self.assertLess(M.illumination_distance(value, self.target, np=np), 1e-16)
        self.assertEqual(M.behavior_distance(value, self.target, np=np), .125)

    def test_record_keeps_actual_PNG_and_RAW_separate(self):
        row = {'role': 'clear'}
        png = np.full((256, 256, 3), 64, np.uint8)
        base_png = png.copy()
        png[0, 0] = 255
        rec = M.record_metrics(self.target, png, self.target, row, np=np,
                               baseline_raw=self.target, baseline_png=base_png)
        self.assertEqual(rec['metrics']['raw']['mae'], 0.)
        self.assertGreater(rec['metrics']['native_png']['mae'], 0.)
        self.assertEqual(rec['U96_behavior_distance']['raw'], 0.)
        self.assertGreater(rec['U96_behavior_distance']['native_png'], 0.)

    def test_legacy_dark_uses_teacher_without_invented_clean_target(self):
        row = {'role': 'dark', 'behavior_teacher': {'pin': 'bound-by-worker'}}
        png = np.full((256, 256, 3), 64, np.uint8)
        rec = M.record_metrics(self.target, png, None, row, np=np,
                               teacher_raw=self.target, teacher_png=png)
        self.assertEqual(rec['behavior_distance'], {'raw': 0., 'native_png': 0.})
        self.assertIsNone(rec['regional_metrics']['raw'])
        self.assertEqual(rec['metrics']['native_png'], dict.fromkeys(M.FIELDS))
        self.assertEqual(rec['U96_illumination_distance'], {'raw': 0., 'native_png': 0.})
        with self.assertRaises(ValueError):
            M.record_metrics(self.target, png, None, {'role': 'clear'}, np=np)
        with self.assertRaises(ValueError):
            M.record_metrics(self.target, png, None, row, np=np)

    def test_invalid_arrays_and_half_pairs_refused(self):
        png = np.full((256, 256, 3), 64, np.uint8)
        for bad in (self.target.astype(np.uint8), self.target[:128], self.target+np.nan,
                    self.target-.5, self.target+1):
            with self.assertRaises(ValueError):
                M.pixel_metrics(bad, self.target, np=np)
        with self.assertRaises(ValueError):
            M.record_metrics(self.target, png, self.target, {'role': 'clear'}, np=np,
                             baseline_raw=self.target)


class BankGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.protocol = json.loads(PARENT.read_text(encoding='utf-8'))
        cls.selection = C.select_parent_cases(PARENT.read_bytes())
        cls.expected = cls.selection['monitors'] + cls.selection['guards']

    def fixture(self):
        bank = {}
        for row in self.expected:
            rec = {key: row[key] for key in ('id', 'reference_id', 'role', 'split')}
            rec.update({'metrics': {}, 'regional_metrics': {},
                        'U96_behavior_distance': dict.fromkeys(M.VIEWS, 0.),
                        'U96_illumination_distance': dict.fromkeys(M.VIEWS, 0.)})
            for view in M.VIEWS:
                if row.get('target') is None:
                    rec['regional_metrics'][view] = None
                    rec['metrics'][view] = dict.fromkeys(M.FIELDS)
                else:
                    values = ({field: row['metrics'][view][field] for field in M.FIELDS}
                              if row.get('cohort') != 'monitor' else {'mae': .125, 'adjacent_rgb_error': .0625})
                    rec['metrics'][view] = values.copy()
                    rec['regional_metrics'][view] = {region: values.copy() for region in M.REGIONS}
            if row.get('behavior_teacher') is not None:
                rec['behavior_distance'] = copy.deepcopy(row['behavior_distance'])
            bank[row['id']] = rec
        return bank

    def altered(self, baseline):
        bank = copy.deepcopy(baseline)
        for row in self.selection['monitors']:
            if row['role'] not in M.ROLES[:2]:
                continue
            for view in M.VIEWS:
                for region in M.REGIONS:
                    bank[row['id']]['regional_metrics'][view][region]['adjacent_rgb_error'] *= .999
                bank[row['id']]['metrics'][view] = bank[row['id']]['regional_metrics'][view]['whole'].copy()
        return bank

    def original_gates(self, bank, baseline):
        """Extract only retained pure scalar gates as independent historical oracle."""
        old = ast.parse((ROOT/'research/codeformer_u96_decoder_readout_v1/worker.py').read_text(encoding='utf-8'))
        functions = [copy.deepcopy(node) for node in ast.walk(old) if isinstance(node, ast.FunctionDef)
                     and node.name in ('legacy_gate', 'monitor_gate')]
        namespace = {'p': self.protocol, 'baseline': baseline, 'math': math,
                     'LEARNING': self.protocol['learning'], 'need': M.need,
                     'ordered_float_sum': M.ordered_float_sum}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<retained scalar gate oracle>', 'exec'), namespace)
        guard_ids = {row['id'] for row in self.selection['guards']}
        monitor_ids = {row['id'] for row in self.selection['monitors']}
        return (namespace['legacy_gate']({key: bank[key] for key in guard_ids}),
                namespace['monitor_gate']({key: bank[key] for key in monitor_ids}, 0))

    def test_initializer_preservation_passes_and_strictgain_is_not_required(self):
        bank = self.fixture()
        result = M.evaluate_bank(bank, None, self.protocol, state='initializer')
        self.assertTrue(result['pass'])
        self.assertFalse(result['strict_feasibility']['evaluated'])
        self.assertFalse(result['full_pilot_endpoints']['pass'])
        self.assertFalse(result['full_pilot_endpoints']['binding_for_this_diagnostic'])

    def test_exact_legacy_and_ordinary_monitor_match_retained_gate_source(self):
        baseline = self.fixture()
        bank = self.altered(baseline)
        result = M.evaluate_bank(bank, baseline, self.protocol, state='proposal_A')
        old_legacy, old_monitor = self.original_gates(bank, baseline)
        self.assertEqual(result['legacy'], old_legacy)
        self.assertEqual(result['monitor']['groups'], old_monitor['groups'])
        self.assertEqual(result['monitor']['pass'], old_monitor['pass'])

    def test_tiny_feasible_gain_does_not_relax_one_percent_qualification(self):
        baseline = self.fixture()
        result = M.evaluate_bank(self.altered(baseline), baseline, self.protocol)
        self.assertTrue(result['pass'])
        self.assertFalse(result['full_pilot_endpoints']['pass'])
        self.assertTrue(result['strict_feasibility']['pass'])

    def test_PNG_only_dark_MAE_regression_stops_despite_edgegain(self):
        baseline = self.fixture()
        bank = self.altered(baseline)
        row = next(row for row in self.selection['monitors'] if row['role'] == 'lowlight_linear_0p35')
        bank[row['id']]['regional_metrics']['native_png']['whole']['mae'] += .000001
        bank[row['id']]['metrics']['native_png']['mae'] += .000001
        result = M.evaluate_bank(bank, baseline, self.protocol)
        self.assertFalse(result['pass'])
        group = result['strict_feasibility']['groups']['lowlight_linear_0p35']
        self.assertTrue(group['views']['raw']['dark_mae']['pass'])
        self.assertFalse(group['views']['native_png']['dark_mae']['pass'])

    def test_legacy_thresholds_are_never_rebased_to_new_U96(self):
        baseline = self.fixture()
        bank = self.altered(baseline)
        key = next(row['id'] for row in self.selection['guards'] if row['role'] == 'clear')
        baseline[key]['behavior_distance'] = dict.fromkeys(M.VIEWS, .5)
        bank[key]['behavior_distance'] = dict.fromkeys(M.VIEWS, .4)
        result = M.evaluate_bank(bank, baseline, self.protocol)
        self.assertFalse(result['legacy']['pass'])
        self.assertFalse(result['legacy']['thresholds_reset_to_U96'])

    def test_monitor_clear_behavior_and_dark_illumination_are_both_binding(self):
        baseline = self.fixture()
        for role, field in (('clear', 'U96_behavior_distance'),
                            ('lowlight_linear_0p35', 'U96_illumination_distance')):
            bank = self.altered(baseline)
            key = next(row['id'] for row in self.selection['monitors'] if row['role'] == role)
            bank[key][field]['native_png'] = .03125
            result = M.evaluate_bank(bank, baseline, self.protocol)
            self.assertFalse(result['monitor']['groups'][role]['pass'])

    def test_missing_extra_duplicate_changed_role_or_null_target_record_refused(self):
        bank = self.fixture()
        first = next(iter(bank))
        variants = [dict(list(bank.items())[1:]), {**bank, 'extra': {'id': 'extra'}},
                    list(bank.values())+[bank[first]]]
        bad = copy.deepcopy(bank)
        bad[first]['role'] = 'dark'
        variants.append(bad)
        dark = next(row['id'] for row in self.selection['guards'] if row['role'] == 'dark')
        bad = copy.deepcopy(bank)
        bad[dark]['metrics']['raw']['mae'] = .1
        variants.append(bad)
        for variant in variants:
            with self.assertRaises(ValueError):
                M.evaluate_bank(variant, None, self.protocol, state='initializer')

    def test_changed_quality_policy_and_nonfinite_scalars_refused(self):
        bank = self.fixture()
        policy = copy.deepcopy(self.protocol)
        policy['quality_policy']['monitor_metric_delta_individual_maximum'] = .031
        with self.assertRaises(ValueError):
            M.evaluate_bank(bank, None, policy, state='initializer')
        first = next(iter(bank))
        bank[first]['regional_metrics']['raw']['whole']['mae'] = float('nan')
        with self.assertRaises(ValueError):
            M.evaluate_bank(bank, None, self.protocol, state='initializer')

    def test_exact_case_and_mean_boundaries_use_no_epsilon(self):
        baseline = self.fixture()
        # Zero metric baseline makes the exact floating boundary explicit.
        ids = sorted(row['id'] for row in self.selection['monitors'] if row['role'] == 'clear')
        for key in ids:
            for view in M.VIEWS:
                baseline[key]['metrics'][view]['mae'] = 0.
                baseline[key]['regional_metrics'][view]['whole']['mae'] = 0.
        bank = self.altered(baseline)
        for key in ids:
            bank[key]['metrics']['raw']['mae'] = .01
            bank[key]['regional_metrics']['raw']['whole']['mae'] = .01
        passed = M.evaluate_bank(bank, baseline, self.protocol)['monitor']['groups']['clear']['views']['raw']['whole/mae']
        self.assertTrue(passed['pass'])
        for key in ids:
            bank[key]['metrics']['raw']['mae'] = .0100000001
            bank[key]['regional_metrics']['raw']['whole']['mae'] = .0100000001
        self.assertFalse(M.evaluate_bank(bank, baseline, self.protocol)['monitor']['groups']['clear']['views']['raw']['whole/mae']['pass'])
        for key in ids:
            bank[key]['metrics']['raw']['mae'] = 0.
            bank[key]['regional_metrics']['raw']['whole']['mae'] = 0.
        bank[ids[0]]['metrics']['raw']['mae'] = .03
        bank[ids[0]]['regional_metrics']['raw']['whole']['mae'] = .03
        self.assertTrue(M.evaluate_bank(bank, baseline, self.protocol)['monitor']['groups']['clear']['views']['raw']['whole/mae']['pass'])
        bank[ids[0]]['metrics']['raw']['mae'] = math.nextafter(.03, math.inf)
        bank[ids[0]]['regional_metrics']['raw']['whole']['mae'] = math.nextafter(.03, math.inf)
        self.assertFalse(M.evaluate_bank(bank, baseline, self.protocol)['monitor']['groups']['clear']['views']['raw']['whole/mae']['pass'])

    def test_complete72_wrapper_keeps_gradient_and_quality_banks_separate(self):
        bank = self.fixture()
        for row in self.selection['optimization']:
            bank[row['id']] = {'id': row['id'], 'role': row['role']}
        self.assertTrue(M.bank_gate(bank, None, self.protocol, self.selection)['pass'])
        bank.pop(self.selection['optimization'][0]['id'])
        with self.assertRaises(ValueError):
            M.bank_gate(bank, None, self.protocol, self.selection)

    def test_import_has_only_standard_library_math(self):
        tree = ast.parse(Path(M.__file__).read_text(encoding='utf-8'))
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertEqual(imports, ['math'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
