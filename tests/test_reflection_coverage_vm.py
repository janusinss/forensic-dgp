import importlib.util
from unittest.mock import patch
import unittest


class VmContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.train_reflection_coverage_vm'),
                             'VM-only matched reflection runner missing')

    def test_local_cpu_refused_before_data_or_optimizer_operations(self):
        from scripts.train_reflection_coverage_vm import main
        with patch('scripts.train_reflection_coverage_vm.require_vm_gpu',side_effect=RuntimeError('VM required')):
            with patch('scripts.train_reflection_coverage_vm.Path.cwd',side_effect=AssertionError('Read data too early')):
                with self.assertRaisesRegex(RuntimeError,'VM required'):main(None)

    def test_lifetime_counters_keep_saved_source_moments_and_bounded_budget(self):
        from scripts.train_reflection_coverage_vm import counters
        row=counters(12)
        self.assertEqual(row['experiment_updates'],252)
        self.assertEqual(row['optimizer_updates'],882)
        self.assertEqual(row['fresh_optimizer_updates'],672)
        self.assertEqual(row['epoch'],42)
        for invalid in (0,13,True):
            with self.assertRaises(ValueError):counters(invalid)

    def test_control_positive_slot_uses_same_source_and_camera_condition(self):
        from scripts.train_reflection_coverage_vm import supplemental_ids
        self.assertEqual(supplemental_ids('control',[279,11]),[271,11])
        self.assertEqual(supplemental_ids('reflective',[279,11]),[279,11])
        with self.assertRaises(ValueError):supplemental_ids('unknown',[279,11])
        with self.assertRaises(ValueError):supplemental_ids('control',[279,15])


if __name__=='__main__':unittest.main()
