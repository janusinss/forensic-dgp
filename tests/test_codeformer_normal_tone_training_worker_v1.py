"""CPU-only source/metadata fixtures; no Torch/model/real optimizer execution."""
import ast
import builtins
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from research.codeformer_normal_tone_training_v1 import worker as w
from research.codeformer_normal_tone_training_v1 import contract as c
from research.codeformer_adain_gradient_isolation_v1 import worker as old


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'research/codeformer_normal_tone_training_v1/model.py'


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class NewTrainingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(w.__file__).read_text()
        cls.tree = ast.parse(cls.source)
        cls.model_source = MODEL.read_text()
        cls.model_tree = ast.parse(cls.model_source)

    def test_import_has_no_framework_or_model(self):
        names = []
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                names.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module.split('.')[0])
        self.assertTrue(set(names).isdisjoint({'torch', 'numpy', 'PIL', 'cv2', 'research'}))

    def test_manual_guard_before_contract_framework_and_checkpoint(self):
        run = function(self.tree, 'run')
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        guard = min(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'require_vm_path')
        imports = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Import) and
                   any(alias.name == 'torch' for alias in node.names)]
        construction = [node.lineno for node in calls if isinstance(node.func, ast.Attribute) and
                        node.func.attr in ('load_trainable', 'load_teacher', 'load_behavior_teacher')]
        self.assertLess(guard, min(imports))
        self.assertLess(min(imports), min(construction))
        self.assertIn("contract.resolve_parent(root)", self.source)
        self.assertIn("parity_manifest(source_root, protocol, dependency_hashes)", self.source)

    def test_windows_guard_refuses_without_torch_or_io(self):
        imported = []
        real_import = builtins.__import__
        def check(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'PIL', 'cv2'):
                imported.append(name)
                raise AssertionError('Framework import before guard')
            return real_import(name, *args, **kwargs)
        with patch.object(w.platform, 'system', return_value='Windows'), patch('builtins.__import__', side_effect=check):
            with self.assertRaisesRegex(ValueError, 'Linux L4 VM only'):
                w.run(Path('does-not-exist'), {}, '0' * 64)
        self.assertEqual(imported, [])

    def test_first_fifty_schedule_and_new_call_census(self):
        self.assertEqual(w.expected_call_caps(), c.COUNTS)
        self.assertEqual(w.expected_call_caps()['gradient_queries'], 11)
        self.assertEqual(w.expected_call_caps()['perceptual_training_forwards'], 148)
        self.assertEqual(sum(w.expected_call_caps().values()), 2075)
        neural_keys = ('route_witness_forwards', 'route_witness_perceptual_forwards',
            'perceptual_prerequisite_forwards', 'candidate_training_forwards', 'perceptual_training_forwards',
            'behavior_teacher_forwards', 'teacher_forwards', 'snapshot_forwards', 'baseline_forwards',
            'initializer_parity_candidate_forwards')
        self.assertEqual(sum(w.expected_call_caps()[key] for key in neural_keys), 1614)
        for update in range(1, 51):
            for base in (1e-5, 2e-5):
                self.assertEqual(w.learning_rate(update, base), old.learning_rate(update, base))
        with self.assertRaises(ValueError):
            w.learning_rate(51, 1e-5)

    def test_clear_blur_and_eligibility_guards_ast_unchanged(self):
        old_tree = ast.parse(Path(old.__file__).read_text())
        self.assertEqual(c.EVALUATION, old.DEFAULT_PROTOCOL['evaluation'])
        for name in ('clear_quality_gate', 'blur_quality_gate', 'validate_training_examples'):
            self.assertEqual(ast.dump(function(self.tree, name)), ast.dump(function(old_tree, name)))
        baseline = {'x': {'raw': .1, 'native_png': .1}}
        self.assertFalse(w.clear_quality_gate({'x': {'raw': .14, 'native_png': .1}}, baseline)['pass'])

    def test_only_training_call_supplies_normal_flag(self):
        call_sites = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call) and
                      isinstance(node.func, ast.Attribute) and node.func.attr == 'forward_training_terms']
        self.assertEqual(len(call_sites), 2)  # Witness and actual training.
        normal_flags = [keyword for node in call_sites for keyword in node.keywords if keyword.arg == 'normal_observation']
        self.assertEqual(len(normal_flags), 1)
        self.assertEqual(ast.unparse(normal_flags[0].value), 'slot in (1, 3)')
        run_text = ast.unparse(function(self.tree, 'run'))
        self.assertIn("(total / protocol['training']['accumulate']).backward()", run_text)
        self.assertNotIn('/ 16', run_text)
        self.assertNotIn('cached_basis', self.source)

    def test_actual_model_shared_helper_and_fresh_losses(self):
        centered = ast.unparse(function(self.model_tree, 'centered_error_l1'))
        coarse = ast.unparse(function(self.model_tree, 'centered_coarse_l1'))
        wrapper = ast.unparse(function(self.model_tree, 'forward_training_terms'))
        self.assertIn('sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)', centered)
        self.assertNotIn('.detach(', centered)
        self.assertIn('(error - mean_luminance).abs().mean()', centered)
        self.assertIn('functional.avg_pool2d(rgb, 32, 32)', coarse)
        self.assertIn('return centered_error_l1(error)', coarse)
        self.assertEqual(wrapper.count('_base.forward_training_terms('), 1)
        self.assertIn("terms['coarse_rgb_l1'] = centered_coarse_l1(output, clean512)", wrapper)
        self.assertIn('sum((_base.LOSS_WEIGHTS[name] * value for name, value in terms.items()))', wrapper)
        self.assertNotIn('total -', wrapper)

    def test_every_exposure_records_both_coarse_losses_and_luminance_mean(self):
        metadata = ast.unparse(function(self.model_tree, 'coarse_supervision_metadata'))
        for key in ('ordinary_coarse_l1', 'centered_coarse_l1', 'mean_coarse_luminance_error'):
            self.assertIn(key, metadata)
        self.assertIn('output.detach().float()', metadata)
        self.assertIn('torch.no_grad()', metadata)
        self.assertIn("'coarse_supervision_metadata': coarse_metadata", self.source)
        self.assertIn("'selected_term_matches_reporting_exactly'", self.source)
        witness = ast.unparse(function(self.tree, 'route_witness'))
        self.assertIn("'normal_observation_flag': False", witness)
        self.assertIn('not new normal-loss full-gradient proof', witness)

    def test_synthetic_queries_before_adam_use_actual_loss_helper(self):
        prerequisite = ast.unparse(function(self.tree, 'normal_formula_prerequisites'))
        self.assertIn('model.centered_error_l1(error)', prerequisite)
        self.assertIn('model.centered_error_l1(error + 0.125)', prerequisite)
        self.assertIn('torch.autograd.grad(loss, error', prerequisite)
        self.assertIn('signs / 12.0 - weights64 * signs.mean() / 4.0', prerequisite)
        self.assertIn('2e-07 + 2e-05 * expected_max', prerequisite)
        self.assertIn("('constant_chroma'", prerequisite)
        self.assertIn('_restore_rng(before_rng, torch, np)', prerequisite)
        self.assertLess(self.source.index('normal_formula = normal_formula_prerequisites('),
                        self.source.index('optimizer = torch.optim.Adam('))

    def test_model_wrapper_control_and_intervention_with_scalar_fakes(self):
        node = function(self.model_tree, 'forward_training_terms')
        terms = {'token_ce': 1., 'latent_mse': 2., 'coarse_rgb_l1': 3., 'pixel_l1_rgb': 4.}
        weights = {'token_ce': .5, 'latent_mse': 1., 'coarse_rgb_l1': 4., 'pixel_l1_rgb': 1.}
        output = SimpleNamespace(device=SimpleNamespace(type='cuda'))
        calls = []
        def fresh(*args, **kwargs):
            calls.append(kwargs)
            return 18.5, terms.copy(), output
        def require(value, message):
            if not value:
                raise ValueError(message)
        base = SimpleNamespace(require=require, forward_training_terms=fresh, LOSS_WEIGHTS=weights,
            torch=SimpleNamespace(autocast=lambda **kwargs: nullcontext(),
                                  isfinite=lambda value: SimpleNamespace(item=lambda: np.isfinite(value))))
        centered_calls = []
        def centered(value, clean):
            centered_calls.append((value, clean)); return .25
        namespace = {'_base': base, 'centered_coarse_l1': centered}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODEL), 'exec'), namespace)
        wrapped = namespace['forward_training_terms']
        total, dark_terms, returned = wrapped(None, None, None, None, 'clean', normal_observation=False)
        self.assertEqual(total, 18.5)
        self.assertEqual(dark_terms, terms)
        self.assertIs(returned, output)
        self.assertFalse(centered_calls)
        total, normal_terms, returned = wrapped(None, None, None, None, 'clean', normal_observation=True)
        self.assertEqual(normal_terms, dict(terms, coarse_rgb_l1=.25))
        self.assertEqual(total, 7.5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(centered_calls), 1)
        self.assertEqual(terms['coarse_rgb_l1'], 3.)

    def test_compact_preflight_summary_and_checkpoint_allowlist(self):
        summary = w.compact_transfer({'complete': True, 'protocol_sha256': '0' * 64,
            'static_files': {str(i): '1' * 64 for i in range(30)}, 'dependencies_verified': 2011,
            'protocol': {'giant': 'x' * 3000000}})
        self.assertLess(len(json.dumps(summary)), 4096)
        self.assertNotIn('protocol', summary)
        self.assertEqual(w.checkpoint_byte_cap(False), 384 * 1024**2)
        self.assertEqual(w.checkpoint_byte_cap(True), 832 * 1024**2)
        self.assertEqual(c.CHECKPOINT_PATHS, [c.OUTPUTS + '/snapshot_0000/checkpoint.pt',
            c.OUTPUTS + '/snapshot_0050/checkpoint.pt', c.OUTPUTS + '/stopped_checkpoint.pt'])
        self.assertIn("'resume_safe': safe", self.source)
        self.assertIn("'no_automatic_resume_supported': True", self.source)


class ArtifactControlTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'scratch'; scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='tone-training-cpu-', dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_new_root_tmux_and_unchanged_call_refusal(self):
        home = self.root; base = home / 'forensic-dgp'; base.mkdir()
        packet = base / w.PACKAGE_DIR; packet.mkdir()
        with patch.object(w.platform, 'system', return_value='Linux'), patch.object(w.Path, 'home', return_value=home):
            with patch.dict(w.os.environ, {}, clear=True):
                with self.assertRaises(ValueError): w.require_vm_path(packet)
            with patch.dict(w.os.environ, {'TMUX': 'human'}):
                self.assertEqual(w.require_vm_path(packet), packet.resolve())
        counts = {'optimizer_step_attempts': 49}
        w.reserve_call(counts, 'optimizer_step_attempts', 50)
        with self.assertRaises(ValueError): w.reserve_call(counts, 'optimizer_step_attempts', 50)
        self.assertEqual(counts['optimizer_step_attempts'], 50)

    def test_capped_write_retains_previous_and_partial_evidence(self):
        path = self.root / 'checkpoint-synthetic-bytes.bin'
        w._atomic_binary(path, lambda stream: stream.write(b'1234'), byte_cap=4)
        with self.assertRaises(ValueError): w._atomic_binary(path, lambda stream: stream.write(b'x'), byte_cap=4)
        self.assertEqual(path.read_bytes(), b'1234')
        with self.assertRaises(w.PilotStop):
            w._atomic_binary(self.root / 'partial.bin', lambda stream: stream.write(b'12345'), byte_cap=4)
        self.assertEqual(len(list(self.root.glob('partial.bin.tmp-*'))), 1)

    def test_pair_parity_checks_clean_target_and_input(self):
        rgb = np.full((256, 256, 3), 113, np.uint8)
        clean8 = np.full((512, 512, 3), 129, np.uint8)
        value, clean = rgb.astype(np.float32) / np.float32(255), clean8.astype(np.float32) / np.float32(255)
        path = self.root / 'input.png'; Image.fromarray(rgb).save(path)
        expected = {'input_path': 'input.png', 'input_sha256': w.sha256(path),
            'input256_rgb8_sha256': w.array_sha(rgb, np), 'input256_float32_sha256': w.array_sha(value, np),
            'clean512_rgb8_sha256': w.array_sha(clean8, np), 'clean512_float32_sha256': w.array_sha(clean, np),
            'update': 1, 'slot': 1, 'reference_id': 'tr_ffhq_00001', 'source_sha256': '1' * 64,
            'profile': 'severe_normal', 'image_eligible': False}
        pair = SimpleNamespace(record=dict(expected, role='train'), input256=value, clean512=clean,
                               parameters={'image_eligible': False})
        self.assertTrue(w.verify_pair_parity(self.root, expected, pair, np, Image)['complete'])
        pair.clean512 = clean.copy(); pair.clean512[0, 0, 0] += .001
        with self.assertRaises(ValueError): w.verify_pair_parity(self.root, expected, pair, np, Image)


if __name__ == '__main__':
    unittest.main()
