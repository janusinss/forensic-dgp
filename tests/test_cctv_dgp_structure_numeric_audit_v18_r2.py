import copy
import importlib.util
import math
from pathlib import Path
import unittest

from cctv_dgp_structure_audit_numeric_v18_r2 import (
    PSNR_LOG_ULPS, verify_psnr_summaries, verify_preservation_report)
import cctv_dgp_structure_v18 as q


class Log10AuditTests(unittest.TestCase):
    def setUp(self):
        self.group = {'cases': 5, 'MSE': .002, 'SSIM': .9, 'MAE': .01,
                      'ArcFace_observed_fixed': .85, 'PSNR': -10 * math.log10(.002)}
        self.local = {'degraded': self.group.copy()}

    def test_only_small_log_roundoff_passes(self):
        recorded = copy.deepcopy(self.local)
        recorded['degraded']['PSNR'] += 2 * math.ulp(self.group['PSNR'])
        self.assertTrue(verify_psnr_summaries(self.local, recorded)['exact_non_PSNR_means_and_counts'])
        recorded['degraded']['PSNR'] += 10 * PSNR_LOG_ULPS * math.ulp(self.group['PSNR'])
        with self.assertRaisesRegex(ValueError, 'rounding bound'):
            verify_psnr_summaries(self.local, recorded)

    def test_any_non_PSNR_change_rejected(self):
        for key in ['MSE', 'SSIM', 'MAE', 'ArcFace_observed_fixed', 'cases']:
            recorded = copy.deepcopy(self.local)
            recorded['degraded'][key] += 1e-10 if key != 'cases' else 1
            with self.assertRaisesRegex(ValueError, 'Exact non-PSNR'):
                verify_psnr_summaries(self.local, recorded)

    def test_schema_nonfinite_and_false_PSNR_rejected(self):
        for value in [math.nan, math.inf, None, True, self.group['PSNR'] + .001]:
            recorded = copy.deepcopy(self.local)
            recorded['degraded']['PSNR'] = value
            with self.assertRaises(ValueError):
                verify_psnr_summaries(self.local, recorded)
        with self.assertRaisesRegex(ValueError, 'schema'):
            verify_psnr_summaries(self.local, {})

    def test_unchanged_failed_decisions_required(self):
        baseline = copy.deepcopy(self.local)
        baseline['degraded']['MSE'] *= 2
        baseline['degraded']['PSNR'] = -10 * math.log10(baseline['degraded']['MSE'])
        report = q.strict_preservation(self.local, baseline)
        report['failed_groups_metrics'] = ['degraded:SSIM']
        report['strict_preservation_passed'] = False
        report['qualified_for_separate_generalization_protocol'] = False
        rounded = copy.deepcopy(report)
        rounded['degraded_PSNR_gain_dB'] += math.ulp(self.group['PSNR'])
        self.assertTrue(verify_preservation_report(report, rounded, self.local, baseline)['exact_decisions_failures_MSE_gain'])
        for key, value in [('failed_groups_metrics', []), ('strict_preservation_passed', True),
                           ('qualified_for_separate_generalization_protocol', True),
                           ('degraded_MSE_improvement_fraction', report['degraded_MSE_improvement_fraction'] + 1e-12)]:
            changed = copy.deepcopy(rounded)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'Exact preservation'):
                verify_preservation_report(report, changed, self.local, baseline)

    def test_source_changes_only_declared_blocks(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('external_audit_log_correction', root / 'scripts/recover_cctv_dgp_structure_v18_audit_r2.py')
        recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recovery)
        original = (root / 'outputs/cctv_dgp_structure_vm_v18/scripts/audit_cctv_dgp_structure_v18.py').read_text()
        corrected = recovery.corrected_source(original)
        reverse = corrected
        for old, new in reversed(recovery.REPLACEMENTS):
            reverse = reverse.replace(new, old)
        self.assertEqual(reverse, original)
        for unchanged in ['delta <= 5e-5', 'delta <= 2e-6',
                "objective == metrics['mean_fit_stop_objective'] == snapshot['mean_fit_stop_objective']",
                "stop['full_cohort_loss_ratio'] <= .99", "changed[update] > 0"]:
            self.assertIn(unchanged, corrected)
        with self.assertRaises(ValueError):
            recovery.corrected_source(corrected)


if __name__ == '__main__':
    unittest.main()
