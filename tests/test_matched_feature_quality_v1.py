"""Scalar quality fixtures only; no saved pixels, model, gradient or worker.

Real inherited controls come from the exact retained 2.7MB pixel audit. Any
synthetic improvement below changes scalar test fixtures only and makes no
claim that an image, model output or new candidate has actually improved.
"""
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from research.codeformer_matched_feature_learning_v1 import quality as Q


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json'
AUDIT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/saved_pixel_audit_v1.json'


def monitor_selection():
    original = json.loads((ROOT / Q.CANDIDATE_SOURCE).read_bytes())
    rows = []
    for row in original['cases']:
        if row['cohort'] == 'adaptive_monitor':
            row = copy.deepcopy(row)
            row['cohort'] = 'monitor'
            row['split'] = row['historical_split']
            rows.append(row)
    return {'new_monitors': rows}


def new_bank(selection, value=1.0):
    return {row['id']: {
        **{field: row[field] for field in ('id', 'reference_id', 'role', 'split')},
        'metrics': {view: dict.fromkeys(Q.FIELDS, value) for view in Q.VIEWS}}
        for row in selection['new_monitors']}


class NewMatchedBankQuality(unittest.TestCase):
    def setUp(self):
        self.selection = monitor_selection()
        self.initial = new_bank(self.selection)

    def evaluate(self, current=None, *, update=48, initial=None, selection=None):
        return Q.evaluate_new_bank(
            copy.deepcopy(self.initial) if current is None else current,
            self.initial if initial is None else initial,
            self.selection if selection is None else selection, update=update)

    def set_role(self, bank, role, view, field, value):
        for row in bank.values():
            if row['role'] == role:
                row['metrics'][view][field] = value

    def test_fixed40_membership_matches_exact_retained_candidate_and_original_parent(self):
        raw = (ROOT / Q.CANDIDATE_SOURCE).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                         (Q.CANDIDATE_BYTES, Q.CANDIDATE_SHA256))
        original = json.loads(raw)
        monitors = [row for row in original['cases'] if row['cohort'] == 'adaptive_monitor']
        self.assertEqual(len(monitors), 40)
        self.assertEqual({row['reference_id'] for row in monitors}, set(Q.MONITOR_REFERENCES))
        self.assertEqual({row['id'] for row in monitors}, {
            reference + '__' + role for reference in Q.MONITOR_REFERENCES for role in Q.ROLES})
        parent = json.loads(PARENT.read_bytes())
        self.assertEqual(set(parent['learning']['monitor_reference_order']), set(Q.MONITOR_REFERENCES))
        self.assertEqual({row['role'] for row in monitors}, set(Q.ROLES))
        self.assertTrue(all(row['historical_split'] == 'TRAIN' for row in monitors))

    def test_initializer_byte_equal_metrics_and_scope_flags(self):
        result = self.evaluate(update=0)
        self.assertTrue(result['pass'])
        self.assertTrue(result['initializer_canonical_metric_payloads_equal'])
        self.assertEqual(result['complete_new_monitor_cases'], 40)
        self.assertEqual(result['monitor_photographs'], 8)
        self.assertTrue(result['inherited_complete_bank_gates_required'])
        self.assertEqual(result['inherited_quality_cases_required'], 56)
        self.assertFalse(result['full_quality_pass_established'])
        self.assertFalse(result['pixels_or_model_state_verified_here'])
        self.assertFalse(result['historical_broader_failures_cleared'])
        self.assertFalse(result['model_qualified'])
        self.assertTrue(result['TRAIN_only_not_DEV_FINAL_or_native_qualification'])

    def test_initializer_rejects_any_metric_change_and_equal_numeric_different_bytes(self):
        key = next(iter(self.initial))
        for value in (math.nextafter(1.0, math.inf), 1):
            current = copy.deepcopy(self.initial)
            current[key]['metrics']['native_png']['coarse_rgb_mae'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'byte-equal canonical'):
                self.evaluate(current, update=0)
        original, current = new_bank(self.selection, 0.0), new_bank(self.selection, 0.0)
        current[key]['metrics']['raw']['mae'] = -0.0
        with self.assertRaisesRegex(ValueError, 'byte-equal canonical'):
            self.evaluate(current, update=0, initial=original)

    def test_checkpoint_binding_counts_are0_10_30_30_and_equality_is_allowed(self):
        for update, count in ((0, 0), (8, 10), (24, 30), (48, 30)):
            with self.subTest(update=update):
                result = self.evaluate(update=update)
                self.assertTrue(result['pass'])
                self.assertEqual(len(result['role_mean_nonincrease_checks']), count)
                self.assertTrue(result['zero_added_tolerance'])
                self.assertFalse(result['quality_tolerance_added'])
                self.assertEqual(result['original_new_bank_reference'], 'initial_u64')

    def test_all30_terminal_endpoints_reject_single_representable_increment(self):
        for update in (24, 48):
            for view in Q.VIEWS:
                for role in Q.ROLES:
                    for field in Q.FIELDS:
                        current = copy.deepcopy(self.initial)
                        self.set_role(current, role, view, field, math.nextafter(1.0, math.inf))
                        with self.subTest(update=update, view=view, role=role, field=field):
                            result = self.evaluate(current, update=update)
                            self.assertFalse(result['pass'])
                            failed = [row for row in result['role_mean_nonincrease_checks'] if not row['pass']]
                            self.assertEqual(len(failed), 1)
                            self.assertEqual((failed[0]['view'], failed[0]['role'], failed[0]['field']),
                                             (view, role, field))

    def test_zero_reference_rejects_positive_subnormal_without_epsilon(self):
        original = new_bank(self.selection, 0.0)
        current = copy.deepcopy(original)
        self.set_role(current, Q.ROLES[-1], 'native_png', 'mae', math.nextafter(0.0, math.inf))
        result = self.evaluate(current, initial=original, update=8)
        self.assertFalse(result['pass'])
        self.assertEqual(result['cohort_means']['current']['native_png'][Q.ROLES[-1]]['mae'],
                         math.nextafter(0.0, math.inf))

    def test_u8_binds_all_five_MAE_roles_in_both_views(self):
        for view in Q.VIEWS:
            for role in Q.ROLES:
                current = copy.deepcopy(self.initial)
                self.set_role(current, role, view, 'mae', math.nextafter(1.0, math.inf))
                with self.subTest(view=view, role=role):
                    self.assertFalse(self.evaluate(current, update=8)['pass'])

    def test_u8_validates_but_does_not_bind_adjacent_or_coarse_fields(self):
        current = new_bank(self.selection, 2.0)
        for view in Q.VIEWS:
            for role in Q.ROLES:
                self.set_role(current, role, view, 'mae', 1.0)
        self.assertTrue(self.evaluate(current, update=8)['pass'])
        for update in (24, 48):
            self.assertFalse(self.evaluate(current, update=update)['pass'])
        current[next(iter(current))]['metrics']['raw']['coarse_rgb_mae'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Finite nonnegative'):
            self.evaluate(current, update=8)

    def test_ordered_eight_means_are_insertion_independent_and_whole_all_roles(self):
        bank = copy.deepcopy(self.initial)
        for role in Q.ROLES:
            ids = sorted(key for key, row in bank.items() if row['role'] == role)
            for index, key in enumerate(ids):
                for view in Q.VIEWS:
                    for field in Q.FIELDS:
                        bank[key]['metrics'][view][field] = 1e16 if index == 0 else 1.0
        reversed_bank = dict(reversed(list(bank.items())))
        forward = self.evaluate(bank, initial=bank, update=0)['cohort_means']['current']
        reverse = self.evaluate(reversed_bank, initial=bank, update=0)['cohort_means']['current']
        self.assertEqual(forward, reverse)
        for role in Q.ROLES:
            ids = sorted(key for key, row in bank.items() if row['role'] == role)
            for view in Q.VIEWS:
                for field in Q.FIELDS:
                    total = 0.0
                    for key in ids:
                        total = total + bank[key]['metrics'][view][field]
                    self.assertEqual(forward[view][role][field], total / 8)

    def test_bank_missing_extra_unknown_or_key_mismatch_is_rejected(self):
        for kind in ('missing', 'extra', 'unknown', 'key_mismatch'):
            current = copy.deepcopy(self.initial)
            key = next(iter(current))
            if kind == 'missing':
                del current[key]
            elif kind == 'extra':
                row = copy.deepcopy(current[key]); row['id'] = 'unadmitted'
                current['unadmitted'] = row
            elif kind == 'unknown':
                row = current.pop(key); row['id'] = 'unadmitted'; current['unadmitted'] = row
            else:
                current[key]['id'] = 'unadmitted'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.evaluate(current)

    def test_duplicate_metric_ids_rejected_for_current_and_initial_lists(self):
        records = list(copy.deepcopy(self.initial).values())
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate new monitor metric ID'):
            self.evaluate(records)
        with self.assertRaisesRegex(ValueError, 'Duplicate new monitor metric ID'):
            self.evaluate(initial=records)
        self.assertTrue(self.evaluate(list(copy.deepcopy(self.initial).values()))['pass'])

    def test_exact_selected_membership_metadata_and_duplicate_ids(self):
        for field, value in (('id', 'unadmitted'), ('reference_id', 'tr_ffhq_10113'),
                             ('role', Q.ROLES[1]), ('split', 'DEV'), ('cohort', 'optimize'),
                             ('optimization_permitted', True)):
            selected = copy.deepcopy(self.selection)
            selected['new_monitors'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.evaluate(selection=selected)
        selected = copy.deepcopy(self.selection)
        selected['new_monitors'][-1] = copy.deepcopy(selected['new_monitors'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate selected monitor ID'):
            self.evaluate(selection=selected)
        for selection in ({}, {'new_monitors': ()}, {'new_monitors': self.selection['new_monitors'][:-1]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.evaluate(selection=selection)

    def test_metric_reference_role_and_split_cannot_be_corrupted(self):
        key = next(iter(self.initial))
        for field, value in (('reference_id', 'tr_ffhq_10113'), ('role', Q.ROLES[1]), ('split', 'FINAL')):
            for argument in ('current', 'initial'):
                bank = copy.deepcopy(self.initial); bank[key][field] = value
                with self.subTest(field=field, argument=argument), self.assertRaises(ValueError):
                    self.evaluate(**{argument: bank})

    def test_views_and_three_field_schema_are_exact(self):
        key = next(iter(self.initial))
        for kind in ('missing_view', 'extra_view', 'missing_field', 'extra_field', 'not_dict'):
            bank = copy.deepcopy(self.initial)
            metrics = bank[key]['metrics']
            if kind == 'missing_view': del metrics['native_png']
            elif kind == 'extra_view': metrics['other'] = copy.deepcopy(metrics['raw'])
            elif kind == 'missing_field': del metrics['raw']['coarse_rgb_mae']
            elif kind == 'extra_field': metrics['raw']['unadmitted'] = 0.0
            else: bank[key]['metrics'] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.evaluate(bank)

    def test_all_metric_values_finite_nonnegative_real_and_both_banks_validated(self):
        key = sorted(self.initial)[-1]
        for bad in (None, True, '1.0', -.1, float('nan'), float('inf'), float('-inf'), 10**1000):
            for field in Q.FIELDS:
                for argument in ('current', 'initial'):
                    bank = copy.deepcopy(self.initial)
                    bank[key]['metrics']['native_png'][field] = bad
                    with self.subTest(bad=bad, field=field, argument=argument), self.assertRaisesRegex(
                            ValueError, 'Finite nonnegative'):
                        self.evaluate(**{argument: bank})

    def test_finite_rows_cannot_produce_infinite_cohort_means(self):
        with self.assertRaisesRegex(ValueError, 'Finite nonnegative'):
            self.evaluate(new_bank(self.selection, 1e308))

    def test_checkpoint_type_and_value_are_explicit(self):
        for update in (True, False, 0.0, 8.0, '24', None, -1, 4, 16, 49):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'Explicit matched checkpoint'):
                self.evaluate(update=update)

    def test_inputs_are_unchanged_and_result_cannot_mutate_them(self):
        current = copy.deepcopy(self.initial)
        before = copy.deepcopy((current, self.initial, self.selection))
        result = self.evaluate(current)
        self.assertEqual((current, self.initial, self.selection), before)
        result['cohort_means']['initial_u64']['raw'][Q.ROLES[0]]['mae'] = 0.0
        result['roles'].append('unadmitted')
        self.assertEqual((current, self.initial, self.selection), before)

    def test_new_bank_runtime_does_not_read_local_preparation_data(self):
        with mock.patch.object(Q, '_pinned_bytes', side_effect=AssertionError('Unexpected data read')):
            self.assertTrue(self.evaluate()['pass'])


class ExactInheritedQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_bytes, cls.audit_bytes = PARENT.read_bytes(), AUDIT.read_bytes()
        cls.protocol = json.loads(cls.parent_bytes)
        cls.module = Q.inherited_metrics()
        cls.controls = cls.module.historical_controls(cls.parent_bytes, cls.audit_bytes)
        cls.monitors = [row for row in cls.protocol['cases'] if row['cohort'] == 'monitor']

    def setUp(self):
        self.initial = copy.deepcopy(self.controls['saved_u64'])
        self.reference = copy.deepcopy(self.controls['frozen_U96'])

    def improving_scalar_fixture(self):
        current = copy.deepcopy(self.initial)
        for row in self.monitors:
            record = current[row['id']]
            for view in self.module.VIEWS:
                for region in self.module.REGIONS:
                    for field in self.module.FIELDS:
                        record['regional_metrics'][view][region][field] *= .999
                record['metrics'][view] = record['regional_metrics'][view]['whole'].copy()
        return current

    def evaluate(self, current=None, *, update=48, **changes):
        arguments = {'current': self.improving_scalar_fixture() if current is None else current,
            'initial_u64': self.initial, 'frozen_U96': self.reference,
            'parent_bytes': self.parent_bytes, 'saved_audit_bytes': self.audit_bytes, 'update': update}
        arguments.update(changes)
        return Q.evaluate_inherited(**arguments)

    def test_old_metrics_byte_pin_and_exact_original_controls_are_used(self):
        raw = (ROOT / Q.INHERITED_SOURCE).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                         (Q.INHERITED_BYTES, Q.INHERITED_SHA256))
        self.assertEqual(len(self.parent_bytes), 197079)
        self.assertEqual(hashlib.sha256(self.parent_bytes).hexdigest(),
                         'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e')
        self.assertEqual(len(self.audit_bytes), 2677742)
        self.assertEqual(len(self.initial), 56)
        expected = self.module.evaluate_bank(self.initial, self.initial, self.reference,
            self.parent_bytes, phase='initializer', saved_audit_bytes=self.audit_bytes)
        actual = self.evaluate(copy.deepcopy(self.initial), update=0)
        self.assertEqual(actual, expected)
        self.assertTrue(actual['pass'])
        self.assertFalse(actual['model_qualified'])

    def test_phase_mapping_is_only_a_unit_mock_of_bound_original_call(self):
        for update, phase in ((0, 'initializer'), (8, 'u8'), (24, 'u16'), (48, 'u16')):
            sentinel = object()
            with self.subTest(update=update), mock.patch.object(
                    self.module, 'evaluate_bank', return_value=sentinel) as evaluate:
                result = self.evaluate(copy.deepcopy(self.initial), update=update)
                self.assertIs(result, sentinel)
                evaluate.assert_called_once_with(self.initial, self.initial, self.reference,
                    self.parent_bytes, phase=phase, saved_audit_bytes=self.audit_bytes)

    def test_real_inherited_terminal_and_u64_retention_bind_at24_and48(self):
        for update in (24, 48):
            result = self.evaluate(update=update)
            self.assertTrue(result['pass'])
            self.assertTrue(result['original_terminal']['binding'])
            self.assertTrue(result['original_terminal']['pass'])
            self.assertTrue(result['additional_retention']['pass'])
            unchanged = self.evaluate(copy.deepcopy(self.initial), update=update)
            self.assertFalse(unchanged['pass'])
            self.assertTrue(unchanged['original_terminal']['pass'])
            self.assertFalse(unchanged['additional_retention']['pass'])

    def test_u8_still_requires_original_clear_improvement(self):
        unchanged = self.evaluate(copy.deepcopy(self.initial), update=8)
        self.assertFalse(unchanged['pass'])
        self.assertFalse(unchanged['original_terminal']['binding'])
        self.assertTrue(self.evaluate(update=8)['pass'])

    def test_terminal_u64_structure_retention_cannot_be_hidden_by_original1percent(self):
        for update in (24, 48):
            current = self.improving_scalar_fixture()
            for row in self.monitors:
                if row['role'] == self.module.ROLES[0]:
                    key = row['id']
                    current[key]['regional_metrics']['native_png']['central128']['adjacent_rgb_error'] = (
                        self.initial[key]['regional_metrics']['native_png']['central128']['adjacent_rgb_error'] * 1.00001)
            result = self.evaluate(current, update=update)
            self.assertTrue(result['original_terminal']['pass'])
            self.assertFalse(result['additional_retention']['pass'])
            self.assertFalse(result['pass'])

    def test_original_terminal_dark_absolute_gate_is_unchanged_at24_and48(self):
        for update in (24, 48):
            current = self.improving_scalar_fixture()
            for row in self.monitors:
                if row['role'] == self.module.ROLES[-1]:
                    key = row['id']
                    value = self.reference[key]['metrics']['native_png']['adjacent_rgb_error'] + 1e-6
                    current[key]['metrics']['native_png']['adjacent_rgb_error'] = value
                    current[key]['regional_metrics']['native_png']['whole']['adjacent_rgb_error'] = value
            result = self.evaluate(current, update=update)
            self.assertTrue(result['additional_retention']['pass'])
            self.assertFalse(result['original_terminal']['pass'])
            self.assertFalse(result['pass'])

    def test_original_one_percent_failure_is_binding_even_if_other_checks_pass(self):
        current = self.improving_scalar_fixture()
        role = self.module.ROLES[0]
        for row in self.monitors:
            if row['role'] == role:
                key = row['id']
                current[key]['regional_metrics']['raw']['central128']['adjacent_rgb_error'] = (
                    self.reference[key]['regional_metrics']['raw']['central128']['adjacent_rgb_error'])
        for update in (24, 48):
            result = self.evaluate(current, update=update)
            self.assertFalse(result['original_terminal']['pass'])
            self.assertFalse(result['pass'])

    def test_legacy_teacher_and_monitor_movement_guards_remain_binding(self):
        current = self.improving_scalar_fixture()
        key = next(row['id'] for row in self.protocol['guards'] if row['role'] == 'clear')
        current[key]['behavior_distance']['raw'] = .5
        result = self.evaluate(current)
        self.assertFalse(result['legacy']['pass'])
        self.assertFalse(result['pass'])
        self.assertFalse(result['legacy']['thresholds_reset_to_U96'])
        current = self.improving_scalar_fixture()
        key = next(row['id'] for row in self.monitors if row['role'] == 'clear')
        current[key]['U96_behavior_distance']['native_png'] = 1.0
        self.assertFalse(self.evaluate(current, update=8)['pass'])

    def test_missing_legacy_or_monitor_rows_and_duplicate_ids_cannot_pass(self):
        for key in (self.protocol['guards'][0]['id'], self.monitors[0]['id']):
            current = self.improving_scalar_fixture(); del current[key]
            with self.assertRaisesRegex(ValueError, 'Complete exact56'):
                self.evaluate(current)
        current = list(self.improving_scalar_fixture().values())
        current.append(copy.deepcopy(current[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate record ID'):
            self.evaluate(current)

    def test_initial_and_frozen_controls_cannot_be_rebased(self):
        for argument, bank in (('initial_u64', self.initial), ('frozen_U96', self.reference)):
            changed = copy.deepcopy(bank)
            changed[self.monitors[0]['id']]['regional_metrics']['raw']['outside']['mae'] += 1e-9
            with self.subTest(argument=argument), self.assertRaisesRegex(ValueError, 'exact saved historical'):
                self.evaluate(**{argument: changed})

    def test_parent_audit_pin_and_import_origin_changes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'retained standardized parent'):
            self.evaluate(parent_bytes=self.parent_bytes + b' ')
        with self.assertRaisesRegex(ValueError, 'independently audited u64 pixel receipt'):
            self.evaluate(saved_audit_bytes=self.audit_bytes + b' ')
        with mock.patch.object(Q, 'INHERITED_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'unchanged inherited quality source bytes'):
                self.evaluate()
        with mock.patch.object(self.module, '__file__', str(ROOT / 'wrong_metrics.py')):
            with self.assertRaisesRegex(ValueError, 'import origin differs'):
                self.evaluate()

    def test_invalid_checkpoint_cannot_select_weaker_inherited_phase(self):
        for update in (True, False, 0.0, 4, 16, '24', None):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'Explicit matched checkpoint'):
                self.evaluate(update=update)


class InertQualitySource(unittest.TestCase):
    def test_every_dependency_size_or_digest_failure_precedes_any_import(self):
        # Change declared pins only; retained source and gate code stay untouched.
        for name in Q.INHERITED_DEPENDENCIES:
            for field in ('bytes', 'sha256'):
                pins = copy.deepcopy(Q.INHERITED_DEPENDENCIES)
                pins[name][field] = (pins[name][field] + 1 if field == 'bytes' else '0' * 64)
                with self.subTest(dependency=name, field=field), mock.patch.object(
                        Q, 'INHERITED_DEPENDENCIES', pins), mock.patch.object(
                        Q.importlib, 'import_module') as importer:
                    with self.assertRaisesRegex(ValueError, 'unchanged inherited quality dependency source bytes'):
                        Q.inherited_metrics()
                    importer.assert_not_called()

    def test_all_dependency_bytes_are_authenticated_before_first_import(self):
        events = []
        original_pin = Q._pinned_bytes
        original_import = Q.importlib.import_module

        def pin(relative, size, digest, label):
            result = original_pin(relative, size, digest, label)
            events.append(('pinned', relative))
            return result

        def imported(name):
            events.append(('import', name))
            return original_import(name)

        with mock.patch.object(Q, '_pinned_bytes', side_effect=pin), mock.patch.object(
                Q.importlib, 'import_module', side_effect=imported):
            Q.inherited_metrics()
        expected_sources = [Q.INHERITED_SOURCE] + [
            value['path'] for value in Q.INHERITED_DEPENDENCIES.values()]
        self.assertEqual(events[:len(expected_sources)], [('pinned', path) for path in expected_sources])
        self.assertEqual(events[len(expected_sources)],
                         ('import', 'research.codeformer_u64_cleananchors_continuation_v1.metrics'))
        self.assertEqual(events[len(expected_sources) + 1:],
                         [('import', name) for name in Q.INHERITED_DEPENDENCIES])

    def test_each_dependency_wrong_loaded_origin_stops_before_gate_call(self):
        evaluator = Q.inherited_metrics()
        # Mock origin metadata and observe the gate entry; no old source is edited.
        for name in Q.INHERITED_DEPENDENCIES:
            dependency = Q.importlib.import_module(name)
            with self.subTest(dependency=name), mock.patch.object(
                    dependency, '__file__', str(ROOT / 'wrong_dependency.py')), mock.patch.object(
                    evaluator, 'evaluate_bank') as gate:
                with self.assertRaisesRegex(ValueError, 'dependency import origin differs'):
                    Q.evaluate_inherited({}, {}, {}, b'', b'', update=24)
                gate.assert_not_called()

    def test_dependency_checks_preserve_all_old_gate_source_bytes(self):
        files = [Q.INHERITED_SOURCE] + [pin['path'] for pin in Q.INHERITED_DEPENDENCIES.values()]
        before = {path: (ROOT / path).read_bytes() for path in files}
        evaluator = Q.inherited_metrics()
        self.assertTrue(callable(evaluator.evaluate_bank))
        for name, pin in Q.INHERITED_DEPENDENCIES.items():
            with self.subTest(dependency=name):
                raw = before[pin['path']]
                self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                                 (pin['bytes'], pin['sha256']))
                self.assertEqual((ROOT / pin['path']).read_bytes(), raw)
        self.assertEqual((ROOT / Q.INHERITED_SOURCE).read_bytes(), before[Q.INHERITED_SOURCE])

    def test_quality_source_parses_without_neural_or_worker_entrypoints(self):
        tree = ast.parse(inspect.getsource(Q), feature_version=(3, 10))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or '')
        self.assertFalse(any(name.split('.')[0] in {'torch', 'numpy', 'tensorflow', 'jax', 'PIL'}
                             or name.endswith('.worker') for name in imported))

    def test_runtime_import_blocker_proves_inert_original_helper_import(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'torch','numpy','tensorflow','jax','PIL'} or fullname.endswith('.worker'):
   raise RuntimeError('Forbidden import:'+fullname)
  return None
sys.meta_path.insert(0,Block())
from research.codeformer_matched_feature_learning_v1 import quality
assert 'research.codeformer_u64_cleananchors_continuation_v1.metrics' not in sys.modules
quality.inherited_metrics()
assert not any(name.split('.')[0] in {'torch','numpy','tensorflow','jax','PIL'} or name.endswith('.worker') for name in sys.modules)
print('INERT_PINNED_SCALAR_QUALITY_ONLY')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'INERT_PINNED_SCALAR_QUALITY_ONLY')


if __name__ == '__main__':
    unittest.main()
