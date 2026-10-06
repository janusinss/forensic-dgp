"""Finite/split/quality/launch boundary regressions; no model fitting or backward."""
import ast
import collections
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cctv_dgp_structure_v18 as q

spec = importlib.util.spec_from_file_location('v18_launch_test', ROOT / 'scripts/launch_cctv_dgp_structure_v18.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class CapacityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = json.loads((ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2/broader_codes_protocol_v16_r2.json').read_text())
        refs, cases = q.selected_cohort(p)
        cls.p = {'references': refs, 'training_cases': cases}
        cls.batches = q.schedule(refs, cases, ['dataset/asian_faces', 'dataset/thumbnails128x128'],
            ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'])

    def test_fixed_training_exposures_cover_every_case_without_validation(self):
        self.assertEqual(len(self.batches), 600)
        counts = collections.Counter(cid for batch in self.batches for cid in batch)
        self.assertEqual(len(counts), 50)
        self.assertEqual(set(counts.values()), {120})
        self.assertEqual(sum(counts.values()), 6000)
        cases = {case['id']: case for case in self.p['training_cases']}
        self.assertTrue(all(ref['role'] == 'train' for ref in self.p['references']))
        for batch in self.batches:
            self.assertEqual(len(set(batch)), 10)
            self.assertEqual(len({cases[cid]['source'] + '/' + cases[cid]['profile'] for cid in batch}), 10)

    def test_loss_calibration_rejects_nonfinite_zero_and_clips_norm_ratio(self):
        self.assertEqual(q.calibrated_identity_weight(2, 4), .5)
        self.assertEqual(q.calibrated_identity_weight(100, .01), 10)
        self.assertEqual(q.calibrated_identity_weight(.000001, 100), 1e-4)
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): q.calibrated_identity_weight(value, 1)
        weights = q.normalized_group_weights({str(index): index + 1. for index in range(10)})
        self.assertAlmostEqual(sum(weights.values()), 10)
        with self.assertRaises(ValueError): q.normalized_group_weights({'missing': 1.})

    def test_pixel_gain_cannot_override_one_preservation_regression(self):
        baseline = {name: {'cases': 5, 'MSE': .01, 'PSNR': 20., 'SSIM': .7, 'ArcFace_observed_fixed': .5}
            for name in ['degraded', 'clear', 'dataset/asian_faces/blur_lr24']}
        candidate = copy.deepcopy(baseline)
        for row in candidate.values(): row.update(MSE=.008, PSNR=20.969100130080562)
        self.assertTrue(q.strict_preservation(candidate, baseline)['qualified_for_separate_generalization_protocol'])
        candidate['dataset/asian_faces/blur_lr24']['ArcFace_observed_fixed'] = .49
        self.assertFalse(q.strict_preservation(candidate, baseline)['strict_preservation_passed'])
        candidate['dataset/asian_faces/blur_lr24']['ArcFace_observed_fixed'] = float('nan')
        with self.assertRaises(ValueError): q.strict_preservation(candidate, baseline)
        candidate.pop('clear')
        with self.assertRaises(ValueError): q.strict_preservation(candidate, baseline)

    def test_partial_trace_is_not_completed_and_role_change_rejected(self):
        records = [{'update': index + 1, 'case_ids': ids, 'loss': 1., 'reconstruction': .5,
                    'identity': .1, 'gradient_norm': 1., 'seconds': .01}
                   for index, ids in enumerate(self.batches[:3])]
        self.assertEqual(q.verify_trace(self.p, records, False)['exposures'], 30)
        with self.assertRaises(ValueError): q.verify_trace(self.p, records, True)
        records[0]['case_ids'] = ['validation_case'] + records[0]['case_ids'][1:]
        with self.assertRaises(ValueError): q.verify_trace(self.p, records, False)

    def test_launcher_allows_idle_outer_tmux_but_blocks_live_tasks(self):
        self.assertEqual(launch.live_tmux_tasks('dgp_training\tbash\t0\nold\tpython\t1\nviewer\ttail\t0\n'), [])
        self.assertEqual(launch.live_tmux_tasks('other\tpython3\t0\n'), [{'session': 'other', 'command': 'python3'}])
        with self.assertRaises(ValueError): launch.live_tmux_tasks('unparseable')

    def test_preflight_uses_Path_typed_arguments_and_parses_on_python310(self):
        paths = [Path('/home/janusdominic0/forensic-dgp') / name for name in ['new', 'parent', 'r2', 'mixed', 'baseline']]
        code = launch.preflight_code(*paths, '0' * 64)
        tree = ast.parse(code, feature_version=(3, 10))
        call = next(node.value for node in tree.body if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'verify')
        self.assertEqual(len(call.args), 6)
        self.assertTrue(all(isinstance(arg, ast.Call) and arg.func.id == 'Path' for arg in call.args[:5]))

    def test_coarse_reference_excludes_padding_and_matches_constant_error(self):
        import numpy as np
        spec = importlib.util.spec_from_file_location('v18_audit_test', ROOT / 'scripts/audit_cctv_dgp_structure_v18.py')
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        raw = np.full((256, 256, 3), .5, np.float32)
        target = np.zeros((256, 256, 3), np.uint8)
        mask = np.zeros((256, 256), bool)
        mask[47:211, 55:197] = True
        self.assertAlmostEqual(auditor.raw_reconstruction(raw, target, mask), .375, places=6)
        raw[~mask] = 1
        target[~mask] = 255
        self.assertAlmostEqual(auditor.raw_reconstruction(raw, target, mask), .375, places=6)


    @unittest.skipIf(sys.platform == 'linux', 'Windows local-training boundary; no execution on VM')
    def test_actual_trainer_entry_rejects_windows_before_outputs_or_models(self):
        sentinel = ROOT / 'outputs/_v18_training_guard_should_not_exist'
        self.assertFalse(sentinel.exists())
        args = [sys.executable, '-B', '-X', 'utf8', str(ROOT / 'scripts/train_cctv_dgp_structure_v18.py')]
        for key in ['root', 'parent', 'r2', 'mixed', 'baseline']: args += ['--' + key, str(sentinel)]
        args += ['--expected-sha', '0' * 64]
        # Parent is nonexistent, so make the verified guard available without
        # constructing any neural model or mocking the VM authorization.
        args[args.index('--parent') + 1] = str(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2')
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Actual training/backward is permitted only', result.stderr)
        self.assertFalse(sentinel.exists())


if __name__ == '__main__': unittest.main()
