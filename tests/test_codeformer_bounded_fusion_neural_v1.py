"""Prospective feasibility CPU metadata/algebra tests; no neural construction."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from research.codeformer_bounded_fusion_feasibility_v1 import neural as n


def protocol_fixture():
    cases = [{'id': 'train_u%04d_slot%d' % (u, s), 'update': u, 'slot': s,
              'reference_id': 'tr_fixture_%04d' % (u * 2 + s // 3),
              'input_path': 'data/input.png', 'input_sha256': '1' * 64,
              'target512_path': 'data/clean512.png', 'target512_sha256': '2' * 64,
              'target256_path': 'data/clean256.png', 'target256_sha256': '3' * 64}
             for u in range(1, 9) for s in range(1, 5)]
    return {'seed': 20261015, 'gate': deepcopy(n.GATE), 'call_caps': deepcopy(n.CAPS),
            'cases': cases, 'calibration_inputs': [{}] * 200,
            'canary_ids': [cases[0]['id'], cases[-1]['id']],
            'assets': {'original': {'sha256': n.ORIGINAL_SHA, 'bytes': 376637898},
                       'stopped': {'sha256': n.STOPPED_SHA, 'bytes': 855229944}}}


class BoundedFusionNeuralTests(unittest.TestCase):
    def test_exact_complete_call_plan_including_canaries(self):
        p = protocol_fixture()
        plan = n.validate_protocol(p)
        self.assertEqual(len(plan), 170)
        self.assertEqual(sum(n.fusion_calls_for(mode, section) for _, mode, section in plan), 960)
        self.assertEqual([sum(section == key for _, _, section in plan)
                          for key in ('paths', 'repeats', 'zero_delta', 'identity')], [160, 4, 4, 2])
        self.assertEqual(n.CAPS['optimizer_updates'], 0)
        self.assertEqual(n.CAPS['gradient_queries'], 0)

    def test_rejects_silent_extra_case_or_canary_or_larger_caps(self):
        for mutate in (lambda p: p['cases'].append(deepcopy(p['cases'][0])),
                       lambda p: p['canary_ids'].append(p['cases'][1]['id']),
                       lambda p: p['call_caps'].__setitem__('codeformer_forwards', 171)):
            p = protocol_fixture(); mutate(p)
            with self.assertRaises(ValueError): n.validate_protocol(p)

    def test_rejects_route_or_weight_recipe_changes(self):
        for key, value in [('gate_floor', 0.), ('beta', .2), ('central_support', [0, 256])]:
            p = protocol_fixture(); p['gate'][key] = value
            with self.assertRaises(ValueError): n.validate_protocol(p)
        p = protocol_fixture(); p['assets']['stopped']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): n.validate_protocol(p)

    def test_gate_bounded_monotonic_and_center_only(self):
        observed = [n.observed_gate(np.full((256, 256, 3), level, np.uint8))['gate']
                    for level in [0, 64, 128, 192, 255]]
        self.assertEqual(observed[0], 1.)
        self.assertEqual(observed[-1], .25)
        self.assertEqual(observed, sorted(observed, reverse=True))
        gray = np.full((256, 256, 3), 128, np.uint8)
        bordered = gray.copy(); bordered[:32] = 255; bordered[224:] = 0
        self.assertEqual(n.observed_gate(gray), n.observed_gate(bordered))

    def test_gate_takes_rgb_only_and_rejects_float_or_crop(self):
        for invalid in [np.zeros((256, 256, 3), np.float32),
                        np.zeros((192, 192, 3), np.uint8),
                        {'input': np.zeros((256, 256, 3), np.uint8), 'clean': 1}]:
            with self.assertRaises(ValueError): n.observed_gate(invalid)

    def test_deterministic_blas_environment_precedes_cuda_and_rejects_drift(self):
        env = {}
        n.deterministic_environment(env)
        self.assertEqual(env, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        n.deterministic_environment(env)
        with self.assertRaises(ValueError):
            n.deterministic_environment({'CUBLAS_WORKSPACE_CONFIG': ':16:8'})
        source = Path(n.__file__).read_text()
        self.assertLess(source.index('deterministic_environment(os.environ)'), source.index('    import torch'))

    def test_numpy_canary_replay_zero_and_nonzero_delta(self):
        base = np.full((1, *n.SITES['32']), .5, np.float32)
        output, used, d = n.bounded_numpy_reference(base, base.copy(), .25, '32')
        self.assertTrue(np.array_equal(base, output))
        self.assertTrue(np.array_equal(used, np.zeros_like(base)))
        self.assertEqual(d.shape, (1, 256, 1, 1))
        other = base + np.float32(10)
        output, used, d = n.bounded_numpy_reference(base, other, .25, '32')
        self.assertTrue(np.all(np.max(np.abs(used), axis=(-2, -1), keepdims=True) <= d * np.float32(.25)))
        self.assertTrue(np.all(output > base))
        full, _, _ = n.bounded_numpy_reference(base, other, 1., '32')
        self.assertTrue(np.all(full >= output))

    def test_numpy_canary_rejects_wrong_shape_dtype_or_nonfinite(self):
        base = np.zeros((1, *n.SITES['32']), np.float32)
        for other in [base.astype(np.float64), base[:, :, :-1], np.full_like(base, np.inf)]:
            with self.assertRaises(ValueError): n.bounded_numpy_reference(base, other, 1., '32')
        with self.assertRaises(ValueError): n.bounded_numpy_reference(base, base, 2., '32')

    def test_class_method_and_hook_namespace_restore_on_exception(self):
        class Owner:
            def forward(self): return 'original'
        item = Owner()
        self.assertNotIn('forward', vars(item))
        with self.assertRaises(RuntimeError):
            with n.temporary_attribute(item, 'forward', lambda: 'temporary'):
                self.assertEqual(item.forward(), 'temporary')
                raise RuntimeError('canary failure')
        self.assertNotIn('forward', vars(item))
        self.assertEqual(item.forward(), 'original')
        item.forward = lambda: 'existing override'
        retained = item.forward
        with n.temporary_attribute(item, 'forward', lambda: 'new'):
            self.assertEqual(item.forward(), 'new')
        self.assertIs(item.forward, retained)

    def test_windows_and_non_tmux_rejected_before_any_heavy_dependency(self):
        with self.assertRaisesRegex(ValueError, 'Linux'):
            n.manual_root(Path.cwd(), system='Windows', environment={'TMUX': 'yes'})
        with self.assertRaisesRegex(ValueError, 'tmux'):
            n.manual_root(Path.cwd(), system='Linux', environment={})

    def test_caller_guard_precedes_torch_import_or_model_construction(self):
        class StopBeforeNeural(RuntimeError): pass
        cb = {name: (lambda *a: None) for name in n.CALLBACK_NAMES}
        def guard(*a): raise StopBeforeNeural('source rejection')
        cb['guard'] = guard
        with patch.object(n, 'manual_root', return_value=Path.cwd()):
            with self.assertRaisesRegex(StopBeforeNeural, 'source rejection'):
                n.execute(Path.cwd(), {}, 'unused-original', 'unused-stopped', cb)

    def test_ast_has_no_optimizer_backward_gradient_or_weight_export_call(self):
        tree = ast.parse(Path(n.__file__).read_text())
        forbidden = {'backward', 'grad', 'step', 'zero_grad', 'save', 'enable_grad', 'Adam', 'AdamW'}
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                calls.append(node.func.attr if isinstance(node.func, ast.Attribute)
                             else node.func.id if isinstance(node.func, ast.Name) else '')
        self.assertFalse(set(calls) & forbidden)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(name and name.split('.')[0] in ('torch', 'numpy', 'PIL', 'third_party') for name in names))

    def test_source_explicitly_captures_actual_indices_and_full_stopped_content(self):
        source = Path(n.__file__).read_text()
        for required in ['get_codebook_feat', 'pre_generator_mean', 'pre_generator_std',
                         'encoder_skips', "content == baselines[case['id'], 'content']",
                         'state_before_after_equal', 'parameter_grad_fields_all_none',
                         'content_preservation_verified', 'repeat_zero_identity_canaries_pass']:
            self.assertIn(required, source)
        tree = ast.parse(source)
        gate_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'observed_gate')
        self.assertEqual([a.arg for a in gate_fn.args.args], ['rgb8', 'gate'])


if __name__ == '__main__':
    unittest.main()
