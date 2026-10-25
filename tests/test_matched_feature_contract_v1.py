"""Pure contract/sampler tests; no framework, image, worker or VM execution."""
import ast
import copy
from collections import Counter
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


class FrameworkBlocker(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} or fullname.endswith('.worker'):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden model/framework/worker import: ' + fullname)
        return None


BLOCKER = FrameworkBlocker()
sys.meta_path.insert(0, BLOCKER)
from research.codeformer_matched_feature_learning_v1 import contract as C

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/evidence/u64_parent_protocol.json'


class MatchedContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = ((PREP / 'data_admission_v1.json').read_bytes(),
                   (PREP / 'candidate_data_v1.json').read_bytes(), PARENT.read_bytes())
        cls.selection = C.select_cases(*cls.raw)
        cls.operations = C.operation_plan(cls.selection)

    def test_exact_source_bytes_are_required_before_JSON_decode(self):
        for position in range(3):
            for value in (self.raw[position] + b' ', b'{', bytearray(self.raw[position])):
                bad = list(self.raw); bad[position] = value
                with self.subTest(position=position, kind=type(value).__name__), mock.patch.object(
                        C.json, 'loads', side_effect=AssertionError('Decode before authentication')):
                    # Earlier arguments may decode first; call authenticator of changed argument directly.
                    with self.assertRaisesRegex(ValueError, 'Exact retained'):
                        C._pinned_json(value, ('admission', 'candidate', 'parent')[position])

    def test_same_length_modified_candidate_bytes_are_refused(self):
        changed = bytearray(self.raw[1]); changed[110] ^= 1
        with self.assertRaisesRegex(ValueError, 'Exact retained candidate'):
            C.select_cases(self.raw[0], bytes(changed), self.raw[2])

    def test_selection_is120_optimization40_monitor56_legacy_quality(self):
        actual = self.selection
        self.assertEqual([len(actual[key]) for key in ('new_optimization', 'new_monitors', 'old_monitors', 'old_guards', 'old_quality')],
                         [120, 40, 32, 24, 56])
        self.assertEqual(len({r['reference_id'] for r in actual['new_optimization']}), 24)
        self.assertEqual(len({r['reference_id'] for r in actual['new_monitors']}), 8)
        self.assertFalse(actual['independent_holdout'])
        self.assertFalse(actual['DEV_native_wild_FINAL_optimized'])

    def test_normalization_retains_preparation_pins_and_separate_later_admission(self):
        original = json.loads(self.raw[1])['cases']
        normalized = self.selection['new_optimization'] + self.selection['new_monitors']
        for source, row in zip(original, normalized):
            with self.subTest(case=row['id']):
                self.assertFalse(source['optimization_permitted'])
                self.assertEqual(source['input_only_decision'], 'PENDING')
                self.assertEqual(row['split'], source['historical_split'])
                for key in ('input', 'target'):
                    self.assertEqual(row['historical_candidate_workspace_' + key], source[key])
                    self.assertEqual(row[key]['sha256'], source[key]['sha256'])
                    self.assertEqual(row[key]['bytes'], source[key]['bytes'])
                    self.assertTrue(row[key]['path'].startswith('data/' + ('inputs/' if key == 'input' else 'targets/')))
                self.assertIs(row['optimization_permitted'], source['cohort'] == 'optimize')
                self.assertFalse(row['synthetic_corruption_mask_is_app_preparation'])
        self.assertEqual(C.select_cases(*self.raw), self.selection)

    def test_assets_refuse_namespace_escape_wrong_schema_bool_size_and_bad_hash(self):
        source = self.selection['new_optimization'][0]['historical_candidate_workspace_input']
        for field, value in [('path', '../escape.png'), ('bytes', True), ('bytes', 0),
                             ('sha256', 'g' * 64), ('sha256', 'A' * 64)]:
            changed = copy.deepcopy(source); changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                C._asset(changed, source['path'], 'data/inputs/x.png')
        changed = dict(source, extra=True)
        with self.assertRaises(ValueError):
            C._asset(changed, source['path'], 'data/inputs/x.png')

    def test_old_metadata_is_exact_and_dark_nulls_stay_null(self):
        original = json.loads(self.raw[2])
        by = {row['id']: row for row in original['cases'] + original['guards']}
        for row in self.selection['old_quality']:
            self.assertEqual(row, by[row['id']])
        dark = [row for row in self.selection['old_guards'] if row['role'] == 'dark']
        self.assertEqual(len(dark), 8)
        self.assertTrue(all(row['target'] is None for row in dark))

    def test_optimization_monitor_and_guard_photograph_groups_stay_disjoint(self):
        sets = [{row['reference_id'] for row in self.selection[key]}
                for key in ('new_optimization', 'new_monitors', 'old_guards')]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(sets[a] & sets[b])
        excluded = {'tr_ffhq_00084', 'tr_ffhq_00178', 'tr_ffhq_00323', 'tr_ffhq_00616',
                    'tr_ffhq_01926', 'tr_ffhq_03917'}
        self.assertFalse(set.union(*sets) & excluded)

    def test_returned_selection_does_not_share_mutable_state_between_calls(self):
        changed = C.select_cases(*self.raw)
        changed['new_optimization'][0]['input']['sha256'] = '0' * 64
        changed['old_quality'][0]['id'] = 'mutated'
        self.assertEqual(C.select_cases(*self.raw), self.selection)

    def test_changed_scope_admission_flag_order_and_monitor_permission_are_refused(self):
        changes = [lambda s: s['source_pins']['candidate'].update(sha256='0' * 64),
                   lambda s: s['new_monitors'][0].update(optimization_permitted=True),
                   lambda s: s['new_optimization'][0].update(split='DEV'),
                   lambda s: s['new_optimization'].reverse(),
                   lambda s: s['old_quality'].pop(),
                   lambda s: s['old_quality'][0].update(optimization_permitted=True)]
        for index, change in enumerate(changes):
            altered = copy.deepcopy(self.selection); change(altered)
            with self.subTest(change=index), self.assertRaises(ValueError):
                C.operation_plan(altered)

    def test_caller_scope_labels_do_not_establish_payload_integrity(self):
        # Deliberately document the boundary: operation_plan validates scope,
        # not payload hashes. Worker must rederive selection and verify assets.
        altered = copy.deepcopy(self.selection)
        altered['new_optimization'][0]['input']['sha256'] = '0' * 64
        self.assertEqual(C.operation_plan(altered), self.operations)
        self.assertNotEqual(altered, C.select_cases(*self.raw))

    def test_maximum_schedule_has_exact1736_forwards480_backwards96_commits(self):
        plan = self.operations
        self.assertEqual([row['slot'] for row in plan], list(range(1, 1737)))
        self.assertEqual(len(plan), 1736)
        self.assertEqual(sum(row['backward'] for row in plan), 480)
        self.assertEqual(sum(row['optimizer_application_after_this_forward'] for row in plan), 96)
        self.assertEqual(Counter(row['arm'] for row in plan), {'A': 868, 'B': 868})
        phases = Counter(row['phase'] for row in plan)
        self.assertEqual(phases, {'initial_quality': 192, 'initial_optimization_readout': 240,
                                 'same_state_deterministic_repeat': 8, 'photographic_optimization': 480,
                                 'quality_snapshot': 576, 'terminal_optimization_readout': 240})

    def test_both_initializers_and_repeats_finish_before_any_training(self):
        first_training = next(row['slot'] for row in self.operations if row['backward'])
        self.assertEqual(first_training, 441)
        self.assertTrue(all(row['new_update'] == 0 and not row['backward'] for row in self.operations[:440]))
        for arm in C.ARMS:
            initial = [r for r in self.operations[:440] if r['arm'] == arm]
            self.assertEqual(len(initial), 220)
            self.assertEqual(sum(r['phase'] == 'same_state_deterministic_repeat' for r in initial), 4)

    def test_each_commit_contains_same_photograph_all_five_roles_once(self):
        by_id = {row['id']: row for row in self.selection['new_optimization']}
        for arm in C.ARMS:
            for update in range(1, 49):
                rows = [r for r in self.operations if r['arm'] == arm and r['backward'] and r['new_update'] == update]
                self.assertEqual(len(rows), 5)
                actual = [by_id[row['id']] for row in rows]
                self.assertEqual(tuple(r['role'] for r in actual), C.ROLES)
                self.assertEqual(len({r['reference_id'] for r in actual}), 1)
                self.assertEqual([r['optimizer_application_after_this_forward'] for r in rows], [False] * 4 + [True])

    def test_two_complete_passes_same_exposure_order_for_both_arms(self):
        a, b = [[(r['id'], r['new_update'], r['epoch']) for r in self.operations if r['arm'] == arm and r['backward']]
                for arm in C.ARMS]
        self.assertEqual(a, b)
        self.assertEqual(Counter(r[0] for r in a), {r['id']: 2 for r in self.selection['new_optimization']})
        for epoch in (1, 2):
            by_epoch = [r for r in a if r[2] == epoch]
            self.assertEqual(len(by_epoch), 120)
            self.assertEqual(len({r[0].split('__')[0] for r in by_epoch}), 24)

    def test_monitor_or_legacy_guard_never_backpropagates(self):
        prohibited = {r['id'] for key in ('new_monitors', 'old_quality') for r in self.selection[key]}
        self.assertFalse(prohibited & {r['id'] for r in self.operations if r['backward']})
        self.assertTrue(all(not r['optimizer_application_after_this_forward'] for r in self.operations if not r['backward']))

    def test_quality_checkpoints_bind96_evaluation_cases_per_arm(self):
        expected = {r['id'] for r in self.selection['old_quality'] + self.selection['new_monitors']}
        for arm in C.ARMS:
            for update in (8, 24, 48):
                rows = [r for r in self.operations if r['arm'] == arm and r['phase'] == 'quality_snapshot' and r['new_update'] == update]
                self.assertEqual(len(rows), 96)
                self.assertEqual({r['id'] for r in rows}, expected)
                self.assertTrue(all(not r['backward'] for r in rows))

    def test_old_new_clear_ID_collisions_stay_separate_scope_qualified_calls(self):
        for arm in C.ARMS:
            rows = [r for r in self.operations if r['arm'] == arm and r['phase'] == 'initial_quality']
            self.assertEqual(len(rows), 96)
            self.assertEqual(len({r['id'] for r in rows}), 88)
            self.assertEqual(len({r['case_key'] for r in rows}), 96)
            self.assertEqual(Counter(r['bank'] for r in rows), {'old_quality': 56, 'new_monitors': 40})
            for ref in C.MONITOR_REFERENCES:
                clear = [r for r in rows if r['id'] == ref + '__clear']
                self.assertEqual(len(clear), 2)
                old, new = [C.resolve_case(self.selection, r) for r in clear]
                self.assertTrue(old['input']['path'].startswith('assets/'))
                self.assertTrue(new['input']['path'].startswith('data/'))
                self.assertNotEqual(old['target'], new['target'])

    def test_resolver_requires_scope_and_refuses_monitor_backward(self):
        for operation in self.operations:
            row = C.resolve_case(self.selection, operation)
            self.assertEqual(row['id'], operation['id'])
        good = copy.deepcopy(self.operations[0])
        for field, value in [('bank', 'UNKNOWN'), ('case_key', good['id']), ('id', 'UNKNOWN'), ('backward', True)]:
            bad = copy.deepcopy(good); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.resolve_case(self.selection, bad)

    def test_terminal_optimization_readout_follows_checkpoint48_without_selection(self):
        for arm in C.ARMS:
            rows = [r for r in self.operations if r['arm'] == arm]
            self.assertTrue(all(r['phase'] == 'terminal_optimization_readout' and r['new_update'] == 48 for r in rows[-120:]))
            self.assertTrue(all(r['phase'] == 'quality_snapshot' and r['new_update'] == 48 for r in rows[-216:-120]))

    def test_sampler_is_frozen_SHA_rank_and_rejects_extra_or_boolean_epochs(self):
        for epoch in (1, 2):
            expected = sorted(C.OPTIMIZATION_REFERENCES, key=lambda r: (
                hashlib.sha256((C.ORDER_SEED + ':' + str(epoch) + ':' + r).encode('ascii')).hexdigest(), r))
            self.assertEqual(C.photograph_order(epoch), tuple(expected))
        self.assertNotEqual(C.photograph_order(1), C.photograph_order(2))
        for epoch in (0, 3, 1.0, True, '1', None):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                C.photograph_order(epoch)

    def test_learning_rate_four_update_warmup_and_constant_maximum(self):
        actual = [C.learning_rate(update) for update in range(1, 49)]
        self.assertEqual(actual[:4], [2.5e-6, 5e-6, 7.500000000000001e-6, 1e-5])
        self.assertEqual(actual[4:], [1e-5] * 44)
        for value in (0, 49, True, False, 1.0, '1', None):
            with self.subTest(update=value), self.assertRaises(ValueError):
                C.learning_rate(value)

    def test_actual_displacement_bounds_bind_values_without_projection_or_epsilon(self):
        result = C.displacement_checks([0.0], [1e-4], [0.0])
        self.assertTrue(result['pass'])
        self.assertFalse(C.displacement_checks([0.0], [math.nextafter(1e-4, math.inf)], [0.0])['pass'])
        self.assertTrue(C.displacement_checks([.00095], [.001], [0.0])['pass'])
        self.assertFalse(C.displacement_checks([.001], [math.nextafter(.001, math.inf)], [0.0])['pass'])
        self.assertFalse(C.displacement_checks([0.0], [0.0], [0.0])['pass'])

    def test_actual_displacement_invalid_banks_are_refused(self):
        for before, after, initial in (([], [], []), ([0.0], [], [0.0]), ([True], [0.0], [0.0]),
                                       ([0.0], [math.inf], [0.0]), ([0.0], [0.0], [math.nan]),
                                       ((0.0,), [0.0], [0.0])):
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                C.displacement_checks(before, after, initial)

    def test_caps_leave_output_archive_reserve_and_keep_finite_time(self):
        limits = C.LIMITS
        self.assertGreaterEqual(limits['initial_free_disk_bytes'], limits['maximum_output_bytes'] +
                                limits['maximum_archive_bytes'] + limits['continuing_free_disk_bytes'])
        self.assertEqual(limits['full_forward_attempts'], sum(limits[k] for k in (
            'training_forwards', 'initializer_forwards', 'deterministic_repeat_forwards',
            'later_quality_forwards', 'terminal_optimization_readout_forwards')))
        self.assertEqual(limits['worker_seconds'], 2 * limits['per_arm_seconds'])
        self.assertGreater(limits['external_seconds'], limits['worker_seconds'] + limits['termination_grace_seconds'])
        self.assertGreater(limits['export_external_seconds'], limits['export_seconds'])
        self.assertEqual(limits['automatic_retry_resume'], 0)
        self.assertEqual(limits['gradient_queries'], 0)

    def test_contract_is_explicitly_unrun_not_model_qualification(self):
        description = C.prospective_contract()
        self.assertEqual(description['arm_A']['trainable_coordinates'], 168960)
        self.assertEqual(description['arm_B']['trainable_coordinates'], 220288)
        self.assertEqual(description['external_microbatch_divisor_exactly_once'], 5)
        self.assertFalse(description['worker_transfer_and_launch_guide_ready'])
        self.assertFalse(description['model_qualified'])
        self.assertFalse(description['two_arm_result_is_quality_qualification'])
        self.assertEqual(description['new_model_or_gradient_optimizer_training_VM_calls'], 0)
        self.assertTrue(description['broader_DEV_and_independent_FINAL_still_required'])
        description['optimizer']['maximum_learning_rate'] = 999
        description['limits']['full_forward_attempts'] = 0
        self.assertEqual(C.prospective_contract()['optimizer']['maximum_learning_rate'], 1e-5)
        self.assertEqual(C.prospective_contract()['limits']['full_forward_attempts'], 1736)

    def test_inherited_contract_and_objective_pins_are_authenticated_before_import(self):
        for field in ('PARENT_CONTRACT', 'PARENT_OBJECTIVE'):
            bad = dict(getattr(C, field)); bad['sha256'] = '0' * 64
            with self.subTest(field=field), mock.patch.object(C, field, bad), mock.patch.object(
                    C.importlib, 'import_module', side_effect=AssertionError('Import before authentication')):
                with self.assertRaisesRegex(ValueError, 'Exact inherited pure selection'):
                    C.select_cases(*self.raw)

    def test_source_contains_no_neural_execution_and_blocker_saw_no_attempt(self):
        tree = ast.parse(Path(C.__file__).read_text(), feature_version=(3, 10))
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse({'backward', 'step', 'cuda', 'autograd', 'train', 'forward'} & calls)
        self.assertEqual(BLOCKER.attempted, [])
        self.assertFalse({'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} & {name.split('.')[0] for name in sys.modules})


if __name__ == '__main__':
    unittest.main()
