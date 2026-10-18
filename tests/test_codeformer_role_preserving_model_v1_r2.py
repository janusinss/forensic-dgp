"""Standard-library role/signature/AST fixtures; never import a neural framework."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_role_preserving_training_v1_r2/model.py'
NAME = 'role_preserving_model_CPU_fixture'
SPEC = importlib.util.spec_from_file_location(NAME, SOURCE)
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = MODEL
SPEC.loader.exec_module(MODEL)
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'), feature_version=(3, 10))


class ModelCPUFixtures(unittest.TestCase):
    def test_import_is_framework_free(self):
        self.assertFalse(any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')))

    def test_normal_has_only_two_distillation_terms(self):
        for eligible in (True, False):
            self.assertEqual(MODEL.role_term_names('normal', eligible), ('behavior_rgb_l1', 'behavior_token_kl'))

    def test_dark_preserves_all_seven_and_eligibility(self):
        self.assertEqual(MODEL.role_term_names('dark', True), ('token_ce', 'latent_mse', 'coarse_rgb_l1',
            'pixel_l1_rgb', 'lpips_vgg', 'behavior_rgb_l1', 'behavior_token_kl'))
        self.assertEqual(MODEL.role_term_names('dark', False), ('token_ce', 'latent_mse', 'coarse_rgb_l1',
            'behavior_rgb_l1', 'behavior_token_kl'))

    def test_invalid_dispatch_refuses(self):
        for role, eligibility in [('clear', True), ('dark', 1), ('normal', None), (None, False)]:
            with self.assertRaises(ValueError):
                MODEL.role_term_names(role, eligibility)

    def test_exact_prospective_weights(self):
        self.assertEqual(MODEL.LOSS_WEIGHTS, {'token_ce': .5, 'latent_mse': 1., 'coarse_rgb_l1': 4.,
            'pixel_l1_rgb': 1., 'lpips_vgg': 1., 'behavior_rgb_l1': 2., 'behavior_token_kl': .1})

    def test_loss_signature_explicit_keyword_roles(self):
        sig = inspect.signature(MODEL.forward_role_terms)
        for name in ('role', 'image_eligible', 'behavior'):
            self.assertEqual(sig.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(sig.parameters[name].default, inspect.Parameter.empty)
        self.assertEqual(tuple(sig.parameters)[:5], ('net', 'clean_teacher', 'perceptual', 'input256', 'clean512'))

    def test_teacher_kind_is_explicit_detached_target_metadata(self):
        value = MODEL.BehaviorTargets('delivered', 'logits', 'original')
        self.assertEqual(value.teacher_kind, 'original')
        with self.assertRaises(Exception):
            value.teacher_kind = 'full_coarse_parent'

    def test_model_helpers_refuse_before_activation(self):
        for function, args in [(MODEL.state_digest, (None,)), (MODEL.cpu_state, (None,)),
            (MODEL.parameter_census, (None,)), (MODEL.optimizer_groups, (None,)),
            (MODEL.prepare_input, (None,)), (MODEL.delivered_rgb, (None,)),
            (MODEL.behavior_targets, (None, None)), (MODEL.hq_targets, (None, None)),
            (MODEL.load_role_models, ('x', 'y', 'z')), (MODEL.historical_modules, ())]:
            with self.assertRaisesRegex(ValueError, 'Activate guarded'):
                function(*args)

    def test_windows_activation_refuses_before_torch(self):
        with patch.object(MODEL.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'manual Linux L4'):
                MODEL.activate_runtime(Path('missing'), 'a'*64)
        self.assertNotIn('torch', sys.modules)

    def test_only_lazy_torch_import_and_no_gradient_optimizer_execution(self):
        top_imports = [node for node in TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any((getattr(n, 'module', '') or '').startswith('torch')
            or any(a.name == 'torch' for a in n.names) for n in top_imports))
        imports = [n for n in ast.walk(TREE) if isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names)]
        self.assertEqual(len(imports), 1)
        activation = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'activate_runtime')
        self.assertIn(imports[0], list(ast.walk(activation)))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('backward', 'autograd', 'optim', 'step') for n in ast.walk(TREE)))

    def test_one_candidate_forward_and_no_role_argument_to_network(self):
        function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_role_terms')
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'net']
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg for k in calls[0].keywords}, {'w', 'detach_16', 'adain'})
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'role' for n in ast.walk(calls[0])))

    def test_normal_rejects_clean_before_candidate_forward(self):
        source = inspect.getsource(MODEL.forward_role_terms)
        self.assertLess(source.index('clean512 is None and codes is None'), source.index('native, logits, feature = net('))
        self.assertIn("if role == 'dark':", source)
        self.assertIn("if image_eligible:", source)

    def test_factory_loads_weights_only_without_restore(self):
        source = inspect.getsource(MODEL.load_role_models)
        self.assertIn("weights_only=True, mmap=True", source)
        self.assertIn("state = saved['model']", source)
        self.assertNotIn("saved['optimizer']", source)
        self.assertNotIn('set_rng_state', source)
        self.assertIn('torch.random.fork_rng', source)
        self.assertIn('candidate_trainable_tensors', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
