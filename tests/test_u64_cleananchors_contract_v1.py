import copy
from pathlib import Path
import unittest
from research.codeformer_u64_cleananchors_continuation_v1 import contract as c


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json'


class ContinuationContract(unittest.TestCase):
    def means(self, edge=0.98, mae=0.1):
        return {view: {role: {'mae': mae, 'adjacent_rgb_error': edge}
                       for role in c.ROLES} for view in c.VIEWS}

    def test_parent_selection_preserves_null_guards_and_no_monitor_training(self):
        selection = c.select_parent_cases(PARENT.read_bytes())
        self.assertEqual(len(selection['optimization']), 64)
        self.assertTrue(all(row['optimization_permitted'] is False for row in selection['monitors']))
        self.assertTrue(all(row.get('target') is None for row in selection['guards'] if row['role'] == 'dark'))

    def test_modified_parent_rejected_before_selection(self):
        raw = PARENT.read_bytes()
        with self.assertRaises(ValueError):
            c.select_parent_cases(raw + b' ')

    def test_plan_covers_each_optimization_row_once_and_nothing_else(self):
        selection = c.select_parent_cases(PARENT.read_bytes())
        plan = c.operation_plan(selection)
        actual = [row for row in plan if row['backward']]
        self.assertEqual(len(actual), 64)
        self.assertEqual({row['id'] for row in actual}, {row['id'] for row in selection['optimization']})
        self.assertEqual(len({row['id'] for row in actual}), 64)
        for update in range(1, 17):
            group = [row for row in actual if row['new_update'] == update]
            self.assertEqual(len(group), 4)
            self.assertEqual(sum(row['optimizer_application_after_this_forward'] for row in group), 1)

    def test_plan_bank_and_repeats_are_monitor_only(self):
        selection = c.select_parent_cases(PARENT.read_bytes())
        plan = c.operation_plan(selection)
        repeats = [row['id'] for row in plan if row['phase'] == 'deterministic_same_u64_repeat']
        self.assertEqual(repeats, [row['id'] for row in selection['monitors'][:4]])
        bank_ids = {r['id'] for r in selection['monitors']+selection['guards']}
        for update in (0, 4, 8, 16):
            rows = [r for r in plan if r['new_update'] == update and r['phase'] in ('initial_u64_quality_bank', 'new_quality_snapshot')]
            self.assertEqual({r['id'] for r in rows}, bank_ids)
            self.assertEqual(len(rows), 56)

    def test_bad_scope_duplicates_rejected(self):
        selection = c.select_parent_cases(PARENT.read_bytes())
        selection['guards'][0] = copy.deepcopy(selection['optimization'][0])
        with self.assertRaises(ValueError):
            c.operation_plan(selection)

    def test_beating_one_percent_but_losing_u64_is_rejected(self):
        initial, reference = self.means(0.98), self.means(1.0)
        new = self.means(0.985, 0.09)
        result = c.retention_checks(initial, new, reference, phase='u16')
        self.assertFalse(result['pass'])
        self.assertTrue(result['checks']['native_png/global_gaussian_1p2/retained_original_one_percent'])
        self.assertFalse(result['checks']['native_png/global_gaussian_1p2/retain_u64_structure'])

    def test_terminal_requires_clear_and_dark_improvement_in_both_views(self):
        initial, reference, new = self.means(), self.means(1.0), self.means(0.979, 0.09)
        self.assertTrue(c.retention_checks(initial, new, reference, phase='u16')['pass'])
        new['native_png']['clear']['mae'] = 0.1
        self.assertFalse(c.retention_checks(initial, new, reference, phase='u16')['pass'])

    def test_terminal_rejects_harmed_blur_rgb_despite_better_edges(self):
        initial, reference, new = self.means(), self.means(1.0), self.means(0.97, 0.09)
        new['raw']['local_gaussian_1p2']['mae'] = 0.101
        self.assertFalse(c.retention_checks(initial, new, reference, phase='u16')['pass'])

    def test_early_eight_does_not_invent_monotone_edge_requirement(self):
        initial, reference, new = self.means(), self.means(1.0), self.means(0.99, 0.09)
        new['raw']['lowlight_linear_0p35']['mae'] = 0.1
        new['native_png']['lowlight_linear_0p35']['mae'] = 0.1
        self.assertTrue(c.retention_checks(initial, new, reference, phase='u8')['pass'])
        self.assertFalse(c.retention_checks(initial, new, reference, phase='u16')['pass'])

    def test_incomplete_views_nonfinite_bool_and_unknown_phase_refused(self):
        initial, reference = self.means(), self.means(1.0)
        for mutation in ('missing', 'nan', 'bool'):
            new = self.means()
            if mutation == 'missing':
                del new['native_png']
            else:
                new['raw']['clear']['mae'] = float('nan') if mutation == 'nan' else True
            with self.assertRaises(ValueError):
                c.retention_checks(initial, new, reference, phase='u16')
        with self.assertRaises(ValueError):
            c.retention_checks(initial, initial, reference, phase='u32')

    def test_prospective_contract_copy_is_not_mutable_global_state(self):
        value = c.prospective_contract()
        value['limits']['optimizer_updates'] = 999
        self.assertEqual(c.prospective_contract()['limits']['optimizer_updates'], 16)
        self.assertFalse(c.prospective_contract()['sealed_packet_or_launch_guide_ready'])


if __name__ == '__main__':
    unittest.main()
