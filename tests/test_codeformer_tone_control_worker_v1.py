"""Meaningful CPU guards/metric/failure-state fixtures; no framework/model execution."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from research.codeformer_tone_control_training_v1 import worker as w, contract as c


def banks():
    original = {}
    profiles = [('clear', 61), ('blur', 53), ('lowlight', 53), ('compound', 53), ('other', 186)]
    for profile, count in profiles:
        for n in range(count):
            original[f'{profile}_{n}'] = {'profile': profile, 'source': 'FFHQ validation',
                'evidence': 'paired_synthetic', 'metrics': {view: {'mae': .1, 'ssim': .5,
                    'luminance_bias': .02, 'coarse_rgb_mae': .04} for view in ('raw', 'native_png')}}
    return original, deepcopy(original)


class WorkerTests(unittest.TestCase):
    def test_supervisor_timeout_worker_ancestry_is_not_competing_training(self):
        root = Path('/home/test/forensic-dgp') / c.VM_NAME
        pin = 'a' * 64
        import shlex
        script = str(root / 'scripts/supervise_codeformer_tone_control_training_v1_vm.py')
        supervisor = shlex.join([sys.executable, '-B', '-u', script, '--root', str(root), '--protocol-sha', pin])
        command = shlex.join([sys.executable, '-B', '-u', '-m', c.MODULE, '--root', str(root), '--protocol-sha', pin, '--run'])
        timeout = 'timeout --signal=TERM --kill-after=60 3000 ' + command
        rows = [(1, 0, '/bin/bash'), (20, 1, supervisor), (21, 20, timeout), (22, 21, command)]
        conflicts, accepted = w.idle_conflicts(rows, 22, root, pin)
        self.assertEqual(conflicts, []); self.assertEqual(accepted, [20, 21, 22])
        rows.append((40, 1, 'python train.py'))
        self.assertEqual(w.idle_conflicts(rows, 22, root, pin)[0][0]['pid'], 40)
        rows[2] = (21, 20, timeout.replace(pin, 'b' * 64))
        with self.assertRaises(ValueError): w.idle_conflicts(rows, 22, root, pin)

    def test_import_does_not_load_neural_framework(self):
        self.assertFalse(any(key in sys.modules for key in ('torch', 'tensorflow', 'jax')))

    def test_local_cli_fails_before_framework_or_output(self):
        root = Path(__file__).resolve().parents[1]
        done = subprocess.run([sys.executable, '-B', '-m', c.MODULE, '--root', str(root), '--protocol-sha', '0' * 64,
                               '--run'], cwd=root, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('Manual NVIDIA L4 VM only', done.stderr)
        self.assertNotIn('ModuleNotFoundError', done.stderr)

    def test_actual_counts_not_inferred_from_archive(self):
        count = dict.fromkeys(c.COUNTS, 0)
        w.finite_counts(count, 0)
        with self.assertRaises(ValueError): w.finite_counts(count, 50)
        w.finite_counts(dict(c.COUNTS), 50, terminal=True)
        count = dict(c.COUNTS); count['controller_TRAIN'] -= 1
        with self.assertRaises(ValueError): w.finite_counts(count, 50, terminal=True)

    def test_cap_attempt_inflight_and_partial_states(self):
        count = dict.fromkeys(c.COUNTS, 0); count['head_optimizer_step_attempts'] = 2
        with self.assertRaises(ValueError): w.finite_counts(count, 0)
        count['head_optimizer_step_attempts'] = 0; count['head_backward_calls'] = 1
        with self.assertRaises(ValueError): w.finite_counts(count, 0)
        self.assertTrue(w.checkpoint_safe(0, 4, False, True))
        self.assertFalse(w.checkpoint_safe(49, 4, True, True))
        self.assertFalse(w.checkpoint_safe(50, 0, False, False))
        self.assertFalse(w.checkpoint_safe(51, 0, False, True))

    def test_floor_png_and_protected_support(self):
        rgb = np.full((256, 256, 3), .5, np.float32)
        pixels = np.full(rgb.shape, 234, np.uint8)
        mask = np.ones((256, 256), bool); mask[:7] = False
        output = w.delivered_png(rgb, pixels, mask)
        self.assertTrue(np.all(output[mask] == 127))
        self.assertTrue(np.array_equal(output[~mask], pixels[~mask]))
        rgb[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): w.delivered_png(rgb, pixels, mask)

    def test_paired_metric_domain_and_native_nulls(self):
        target = np.full((256, 256, 3), .5, np.float32)
        raw = np.full_like(target, .6); mask = np.ones((256, 256), bool)
        png = w.delivered_png(raw, np.full(raw.shape, 153, np.uint8), mask)
        value = w.pixel_metrics(raw, png, target, mask, evidence='paired_synthetic')
        self.assertAlmostEqual(value['raw']['mae'], .1, places=6)
        self.assertAlmostEqual(value['raw']['luminance_bias'], .1, places=6)
        self.assertAlmostEqual(value['raw']['coarse_rgb_mae'], .1, places=6)
        native = w.pixel_metrics(raw, png, None, mask, evidence='native_unpaired')
        self.assertTrue(all(v is None for row in native.values() for v in row.values()))
        with self.assertRaises(ValueError): w.pixel_metrics(raw, png, target, mask, evidence='native_unpaired')

    def test_guards_compute_both_failure_families(self):
        original, candidate = banks()
        self.assertTrue(w.preservation_guards(candidate, original)['pass'])
        candidate['clear_0']['metrics']['native_png']['mae'] += .031
        candidate['blur_0']['metrics']['raw']['coarse_rgb_mae'] += .031
        failed = w.preservation_guards(candidate, original)
        self.assertFalse(failed['clear']['pass']); self.assertFalse(failed['normal_blur']['pass'])
        self.assertEqual(failed['clear']['views']['native_png']['mae']['cases_over_0_03'], 1)
        self.assertEqual(failed['normal_blur']['views']['raw']['coarse_rgb_mae']['cases_over_0_03'], 1)

    def test_mean_limit_is_distinct_from_worst_limit(self):
        original, candidate = banks()
        for row in candidate.values():
            if row['profile'] == 'clear': row['metrics']['raw']['mae'] += .011
        failed = w.preservation_guards(candidate, original)['clear']['views']['raw']['mae']
        self.assertLess(failed['worst_delta'], .03); self.assertGreater(failed['mean_delta'], .01)
        self.assertFalse(failed['pass'])

    def test_missing_control_cannot_shrink_gate(self):
        original, candidate = banks(); del candidate['blur_52']
        with self.assertRaises(ValueError): w.preservation_guards(candidate, original)

    def test_dark_nondegradation_both_metrics_and_profiles(self):
        base, candidate = banks()
        self.assertTrue(w.dark_retention(candidate, base)['pass'])
        candidate['lowlight_0']['metrics']['native_png']['mae'] += .001
        self.assertFalse(w.dark_retention(candidate, base)['profiles']['lowlight']['pass'])
        candidate['compound_0']['metrics']['native_png']['ssim'] -= .001
        self.assertFalse(w.dark_retention(candidate, base)['profiles']['compound']['pass'])

    def test_source_guard_checkpoint_and_neural_call_order(self):
        source = Path(w.__file__).read_text(encoding='utf-8')
        ast.parse(source, feature_version=(3, 10))
        self.assertLess(source.index('root = c.manual_root(root)'), source.index('torch = model.activate_runtime'))
        self.assertLess(source.index("checkpoint('composite_head0050')"), source.index("all_banks['head0050'] = bank"))
        self.assertLess(source.index("write_json('preservation_guards.json'"), source.index("c.require(gates['pass']"))
        self.assertLess(source.index('inflight = True'), source.index('optimizer.step()'))
        self.assertLess(source.index('optimizer.step()'), source.index('completed = update'))
        self.assertIn('(loss / 4).backward()', source)
        self.assertNotIn('torch.load(', source)  # Only guarded/hash-first model helper loads parents.
        self.assertIn("checkpoint('stopped_composite'", source)
        self.assertIn('os.urandom(16).hex()', source)


if __name__ == '__main__':
    unittest.main()
