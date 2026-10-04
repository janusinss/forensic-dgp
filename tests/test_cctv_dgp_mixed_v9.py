"""Protocol boundary tests using stdlib; no local model forward/backward/training."""
import importlib.util
import math
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

from cctv_dgp_mixed_v9 import fixed_loss_weights, safe_path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class MixedProtocolBoundaries(unittest.TestCase):
    def test_equal_training_errors_do_not_change_relative_weights(self):
        self.assertEqual(fixed_loss_weights({str(i): 0.2 for i in range(10)}), {str(i): 1.0 for i in range(10)})

    def test_weights_stay_bounded_and_normalized(self):
        means = {str(i): v for i, v in enumerate([0, 1e-15, 0.0001, 0.0002, 0.001, 0.02, 0.06, 0.1, 0.2, 1])}
        weights = fixed_loss_weights(means)
        self.assertTrue(math.isclose(sum(weights.values()) / 10, 1))
        self.assertLessEqual(max(weights.values()) / min(weights.values()), 16)
        self.assertEqual(weights['0'], weights['1'])
        self.assertGreater(weights['0'], weights['9'])

    def test_nonfinite_or_negative_training_information_stops_calibration(self):
        for bad in (float('nan'), float('inf'), -0.01):
            with self.subTest(bad=bad):
                means = {str(i): 0.1 for i in range(10)}; means['0'] = bad
                with self.assertRaises(AssertionError):
                    fixed_loss_weights(means)

    def test_incomplete_group_coverage_stops_calibration(self):
        with self.assertRaises(AssertionError):
            fixed_loss_weights({str(i): 0.1 for i in range(9)})

    def test_zero_error_cannot_create_a_training_recipe(self):
        with self.assertRaises(AssertionError):
            fixed_loss_weights({str(i): 0 for i in range(10)})

    def test_changed_asset_paths_cannot_escape_workspace(self):
        for name in ('../x', '/x', 'C:/x', 'x\\y'):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                safe_path(ROOT / 'scratch', name)

    def test_return_archive_rejects_external_paths(self):
        auditor = load('mixed_audit_paths', ROOT / 'scripts/audit_cctv_dgp_mixed_v9.py')
        for name in ('../x', '/x', 'C:/x', 'x\\y'):
            member = tarfile.TarInfo(name); member.size = 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                auditor.safe_members([member])

    def test_return_archive_rejects_symlinks_and_duplicates(self):
        auditor = load('mixed_audit_links', ROOT / 'scripts/audit_cctv_dgp_mixed_v9.py')
        member = tarfile.TarInfo('outputs/shortcut'); member.type = tarfile.SYMTYPE; member.linkname = '/etc/passwd'
        with self.assertRaises(AssertionError):
            auditor.safe_members([member])
        member = tarfile.TarInfo('outputs/a'); member.size = 1
        with self.assertRaises(AssertionError):
            auditor.safe_members([member, member])

    def test_local_training_guard_precedes_output_creation(self):
        trainer = load('mixed_train_guard', ROOT / 'scripts/train_cctv_dgp_mixed_v9.py')
        import cctv_dgp_targets_v6
        with patch.object(Path, 'mkdir', side_effect=AssertionError('Output creation before VM guard')) as mkdir:
            with patch.object(cctv_dgp_targets_v6, 'require_vm', side_effect=RuntimeError('VM-only boundary')):
                with self.assertRaisesRegex(RuntimeError, 'VM-only boundary'):
                    trainer.train(ROOT / 'scratch' / 'unused_guard_probe', 'unused')
            mkdir.assert_not_called()
        self.assertIsNone(trainer.CONTEXT['out'])

    def test_identity_regression_rejects_better_pixel_score(self):
        from cctv_dgp_pilot import qualifies
        baseline = {'degraded': {'cases': 4, 'identity_pairs': 4, 'MSE': 0.1, 'SSIM': 0.6, 'ArcFace_observed_fixed': 0.5}}
        candidate = {'degraded': {'cases': 4, 'identity_pairs': 4, 'MSE': 0.08, 'SSIM': 0.7, 'ArcFace_observed_fixed': 0.49}}
        self.assertFalse(qualifies(candidate, baseline, baseline))
        candidate['degraded']['ArcFace_observed_fixed'] = 0.5
        self.assertTrue(qualifies(candidate, baseline, baseline))


if __name__ == '__main__':
    unittest.main()
