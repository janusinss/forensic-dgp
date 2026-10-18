"""CPU-only finite counters, guards, captures and source-order fixtures."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_role_preserving_training_v1_r1 import contract as c, worker as w


class WorkerCPUFixtures(unittest.TestCase):
    def test_source_python310_no_framework_import(self):
        tree = ast.parse(Path(w.__file__).read_text(), feature_version=(3, 10))
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) and
            (getattr(n, 'module', '') or '').startswith('torch') for n in ast.walk(tree)))

    def test_exact_budget_terminal(self):
        self.assertEqual((c.NEURAL_FORWARD_CAP, c.RESERVATION_CAP), (1399, 1854))
        w.finite_counts(dict(c.COUNTS), 50, terminal=True)
        bad = dict(c.COUNTS); bad['initializer_parity_perceptual_forwards'] = 0
        with self.assertRaises(ValueError): w.finite_counts(bad, 50, terminal=True)
        bad = dict(c.COUNTS); bad['backward_calls'] = 201
        with self.assertRaises(ValueError): w.finite_counts(bad, 50)

    def test_attempts_are_not_commits(self):
        counts = {k: 0 for k in c.COUNTS}; counts['optimizer_step_attempts'] = 1
        w.finite_counts(counts, 0)
        with self.assertRaises(ValueError): w.finite_counts(counts, 2)
        self.assertFalse(w.checkpoint_safe(0, 4, True, True))
        self.assertFalse(w.checkpoint_safe(50, 0, False, False))

    def test_checkpoint_partial_slot_blocks_second_updated_copy(self):
        scratch = ROOT / 'scratch'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            out = Path(directory); (out/'snapshot_0050').mkdir()
            partial = out/'snapshot_0050'/('checkpoint.pt.tmp-' + 'a'*32)
            partial.write_bytes(b'partial-evidence')
            with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                w.checkpoint_slot_admission(out, 'stopped_checkpoint.pt')
            self.assertEqual(partial.read_bytes(), b'partial-evidence')
            w.checkpoint_slot_admission(out, 'snapshot_0000/checkpoint.pt')

    def test_new_warmup_is_exact_finite(self):
        self.assertEqual(w.learning_rate(1, 1e-5), 1e-5/50)
        self.assertEqual(w.learning_rate(50, 2e-5), 2e-5)
        for update in (0, 51):
            with self.assertRaises(ValueError): w.learning_rate(update, 1e-5)

    def test_png_floor_and_protected_input(self):
        raw = np.full((256,256,3), .5, np.float32)
        pixels = np.full((256,256,3), 17, np.uint8)
        mask = np.ones((256,256), bool); mask[0,0] = False
        value = w.delivered_png(raw, pixels, mask)
        self.assertTrue(np.all(value[mask] == 127))
        self.assertTrue(np.array_equal(value[0,0], pixels[0,0]))

    def test_native_has_no_invented_target_metrics(self):
        value = w.pixel_metrics(None, None, None, None, evidence='native_unpaired')
        self.assertTrue(all(v is None for view in value.values() for v in view.values()))
        with self.assertRaises(ValueError): w.pixel_metrics(None, None, object(), None, evidence='native_unpaired')

    def test_clear_and_blur_guards_unchanged(self):
        metrics = {k: 0. for k in ('mae','mse','psnr','ssim','luminance_bias','coarse_rgb_mae')}
        baseline = {}
        for i in range(406):
            profile = 'clear' if i < 61 else 'blur' if i < 114 else 'light'
            baseline[str(i)] = {'evidence':'paired_synthetic','source':'FFHQ validation','profile':profile,
                'metrics':{v:dict(metrics) for v in ('raw','native_png')}}
        candidate = json.loads(json.dumps(baseline))
        self.assertTrue(w.preservation_guards(candidate, baseline)['pass'])
        candidate['0']['metrics']['native_png']['mae'] = .030001
        report = w.preservation_guards(candidate, baseline)
        self.assertFalse(report['pass'])
        self.assertEqual(report['clear']['views']['native_png']['mae']['cases_over_0_03'], 1)

    def test_capture_is_actual_single_forward_and_restores_on_error(self):
        class Token:
            def detach(self): return self
            def clone(self): return 'actual-tokens'
        class Network:
            def __init__(self):
                self.quantize = SimpleNamespace(get_codebook_feat=lambda indices, shape: 'lookup')
                self.hooks = []
            def register_forward_hook(self, fn):
                self.hooks.append(fn)
                return SimpleNamespace(remove=lambda: self.hooks.remove(fn))
        net = Network(); original = net.quantize.get_codebook_feat
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with w.capture_forward(net) as capture:
                self.assertEqual(net.quantize.get_codebook_feat(Token(), [1]), 'lookup')
                net.hooks[0](net, (), ('native','actual-logits','actual-feature'))
                self.assertEqual(capture, {'tokens':'actual-tokens','logits':'actual-logits','latent':'actual-feature'})
                raise RuntimeError('fixture')
        self.assertIs(net.quantize.get_codebook_feat, original)
        self.assertEqual(net.hooks, [])

    def test_windows_guard_precedes_model_import(self):
        with patch.object(c.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'Manual NVIDIA L4'):
                w.run(Path('missing'), 'a'*64)
        self.assertNotIn('torch', sys.modules)

    def test_compact_cli_does_not_dump_large_protocol(self):
        from research.codeformer_role_preserving_training_v1_r1 import transport
        value = {'complete':True, 'protocol':{'dependencies':{str(i):'x'*200 for i in range(4000)}}, 'data_bundle':{'x':'z'*2300000}}
        text = StringIO()
        with patch.object(c, 'manual_root', side_effect=lambda p:p), patch.object(transport, 'verify_transfer', return_value=value), redirect_stdout(text):
            w.main(['--root','unused','--protocol-sha','a'*64,'--verify-transfer'])
        self.assertLess(len(text.getvalue().encode()), 4096)
        self.assertEqual(json.loads(text.getvalue())['dependencies'], 4000)

    def test_checkpoint50_after_eval_and_before_quality_stop(self):
        source = Path(w.__file__).read_text()
        final = source.index('bank50 = snapshot(50)')
        checkpoint = source.index("checkpoint('snapshot_0050/checkpoint.pt')", final)
        quality = source.index("c.require(guards['pass'] and dark['pass']", checkpoint)
        self.assertLess(final, checkpoint); self.assertLess(checkpoint, quality)
        self.assertIn("k not in ('protocol', 'data_bundle')", source)
        self.assertIn("os.urandom(16).hex()", source)
        self.assertIn("'committed_state_digest_equal': True", source)
        self.assertIn('rng_equal(committed_rng, actual_rng)', source)
        self.assertIn('rng_equal(committed_rng, rng())', source)
        self.assertIn("json.dumps(value, separators=(',', ':')", source)


if __name__ == '__main__': unittest.main(verbosity=2)
