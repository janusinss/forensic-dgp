import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest

import cctv_dgp_input_selection_v19 as q


class InferenceContractTests(unittest.TestCase):
    def test_training_promotion_threshold_refit_and_native_use_rejected(self):
        result = {'complete': True, 'protocol_sha256': 'pin', 'seconds': 1,
            'training': False, 'validation_used': True, 'optimizer_updates': 0, 'backward_calls': 0,
            'teacher_used': False, 'native_used': False, 'native_reserved_used': False,
            'best_checkpoint_selected': False, 'thresholds_refitted': False, 'production_promoted': False}
        q.validate_result_scope(result, 'pin')
        for key in ['training', 'teacher_used', 'native_used', 'native_reserved_used',
                    'best_checkpoint_selected', 'thresholds_refitted', 'production_promoted']:
            with self.assertRaises(ValueError):
                q.validate_result_scope({**result, key: True}, 'pin')
        for key in ['backward_calls', 'optimizer_updates']:
            with self.assertRaises(ValueError):
                q.validate_result_scope({**result, key: 1}, 'pin')
        for seconds in [0, 1201, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                q.validate_result_scope({**result, 'seconds': seconds}, 'pin')

    def test_automatic_cannot_change_pixels_paths_or_scores(self):
        item = {'raw': 'raw/a.npy', 'prediction': 'image/a.png', 'embedding': 'vec/a.npy', 'SSIM': .8}
        row = {'decision': {'branch': 'retained_dgp_v2'}, 'arms': {arm: copy.deepcopy(item) for arm in q.ARMS}}
        q.validate_automatic_alias(row)
        for key, value in [('raw', 'raw/improved.npy'), ('prediction', 'image/sharp.png'),
                           ('embedding', 'vec/target.npy'), ('SSIM', .9)]:
            forged = copy.deepcopy(row)
            forged['arms']['automatic_v19'][key] = value
            with self.assertRaisesRegex(ValueError, 'alias'):
                q.validate_automatic_alias(forged)
        with self.assertRaisesRegex(ValueError, 'Undeclared'):
            q.validate_automatic_alias({**row, 'decision': {'branch': 'pretrained_codeformer_none'}})

    def test_canonical_means_retain_all_source_groups_and_perfect_resize(self):
        rows = []
        for source in ['dataset/asian_faces', 'dataset/thumbnails128x128']:
            for profile in ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']:
                rows.append({'source': source, 'profile': profile, 'MSE': 0 if profile == 'clear' else .01,
                             'SSIM': .8, 'MAE': .1, 'ArcFace_observed_fixed': .7})
        summaries = q.aggregate(rows)
        self.assertEqual(len(summaries), 14)
        self.assertEqual(summaries['clear']['cases'], 2)
        self.assertIsNone(summaries['clear']['PSNR'])
        self.assertEqual(summaries['degraded']['cases'], 8)
        self.assertEqual(summaries['degraded']['PSNR'], 20)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            q.aggregate([r for r in rows if r['source'] != 'dataset/asian_faces'])

    def test_unsafe_member_paths_rejected(self):
        root = Path(__file__).resolve().parents[1] / 'outputs'
        for name in ['../outside', '/outside', 'C:/outside', 'a\\b', 'a//b', 'a/./b', '']:
            with self.assertRaises(ValueError): q.safe(root, name)

    def test_runner_uses_inference_only_and_checks_all_frozen_modules(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'scripts/run_cctv_dgp_input_selection_v19.py').read_text()
        tree = ast.parse(source, feature_version=(3, 10))
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        forbidden = {'backward', 'enable_vm_training', 'step', 'AdamW', 'Adam', 'SGD', 'autograd_grad'}
        self.assertFalse(any(isinstance(f, ast.Attribute) and f.attr in forbidden or
                             isinstance(f, ast.Name) and f.id in forbidden for f in calls))
        self.assertIn('with torch.inference_mode():', source)
        self.assertIn("before == after and counts == q.expected_counts()", source)
        self.assertIn('t.grad is not None', source)
        self.assertLess(source.index('require_vm(root)'), source.index('DGPStructureResidualHead().cuda()'))

    def test_bootstrap_diagnostics_use_path_objects_and_allow_idle_tmux(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('v19_bootstrap', root / 'scripts/launch_cctv_dgp_input_selection_v19.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        paths = [Path('/home/u/forensic-dgp/' + x) for x in ['v19', 'parent', 'r2', 'mixed', 'baseline']]
        preflight = module.preflight_code(*paths, 'pin')
        ast.parse(preflight)
        self.assertIn('require_vm(Path(', preflight)
        self.assertEqual(module.live_tmux_tasks('shell\tbash\t0\nlogs\ttail\t0\n'), [])
        self.assertEqual(module.live_tmux_tasks('training\tpython\t0\n'), [{'session': 'training', 'command': 'python'}])
        with self.assertRaises(ValueError): module.live_tmux_tasks('unrecognized')


if __name__ == '__main__': unittest.main()
