"""Pure standard-library contract/AST/guard tests; no model or Torch execution."""
import ast
import builtins
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'research/codeformer_u96_decoder_readout_v1/model.py'


def load_pure():
    spec = importlib.util.spec_from_file_location('pure_decoder_readout_prototype', SOURCE)
    module = importlib.util.module_from_spec(spec)
    original = builtins.__import__

    def no_framework(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'cv2', 'PIL', 'third_party'):
            raise AssertionError('Forbidden model/framework import in pure tests: ' + name)
        return original(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=no_framework), \
            patch('subprocess.run', side_effect=AssertionError('No subprocess during pure import')):
        spec.loader.exec_module(module)
    return module


class DecoderReadoutPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_pure()
        cls.tree = ast.parse(SOURCE.read_text(encoding='utf-8'))

    def context(self, **changes):
        values = dict(system='Linux', root='/home/test/forensic-dgp/' + self.module.VM_NAME,
                      home='/home/test', device='cuda:0', tmux='/tmp/tmux-1/default,1,0',
                      gpu_names=('NVIDIA L4',), acknowledgement=self.module.MANUAL_ACKNOWLEDGEMENT)
        values.update(changes)
        return self.module.validate_manual_context(**values)

    def test_import_is_inert_and_reproducible(self):
        self.assertEqual(load_pure().STATUS, 'UNRUN_PROTOTYPE_NOT_PACKET_TRAINED')

    def test_existing_rgb_readout_names_shapes_and_scalar_count(self):
        schema = self.module.expected_readout_shapes()
        self.assertEqual(schema, {'base.generator.blocks.24.weight': (3, 64, 3, 3),
                                  'base.generator.blocks.24.bias': (3,)})
        self.assertEqual(sum(math.prod(shape) for shape in schema.values()), 1731)
        schema.clear()
        self.assertEqual(len(self.module.expected_readout_shapes()), 2)

    def test_all_four_retained_U96_schemas_complete(self):
        schema = self.module.expected_adapter_shapes()
        self.assertEqual(len(schema), 24)
        self.assertEqual(sum(math.prod(shape) for shape in schema.values()), 51328)
        for site, channels in self.module.CHANNELS.items():
            self.assertEqual(schema['adapters.' + site + '.depthwise.weight'], (32, 1, 3, 3))
            self.assertEqual(schema['adapters.' + site + '.up.weight'], (channels, 32, 1, 1))

    def test_contract_census_and_unproven_claims(self):
        contract = self.module.architecture_contract()
        self.assertEqual(contract['total_tensors'], contract['base_tensors'] + contract['adapter_tensors'])
        self.assertEqual(contract['total_tensors'], contract['frozen_tensors'] + contract['trainable_tensors'])
        self.assertEqual((contract['total_tensors'], contract['frozen_tensors'],
                          contract['trainable_tensors'], contract['trainable_parameters']), (539, 537, 2, 1731))
        self.assertFalse(contract['new_head'])
        self.assertFalse(contract['gradient_or_optimizer_or_quality_verified'])
        self.assertFalse(contract['quality_gates_changed'])
        self.assertEqual(contract['initial_RAW_PNG_logits_LQ_parity'], 'UNMEASURED_REQUIRES_MANUAL_VM')
        self.assertEqual(contract['status'], 'UNRUN_PROTOTYPE_NOT_PACKET_TRAINED')

    def test_initializer_pins_match_retained_exact_method(self):
        contract = self.module.architecture_contract()['initializer']
        self.assertEqual(contract, {
            'R2_file_sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
            'R2_canonical515_sha256': 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f',
            'U96_file_sha256': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0',
            'U96_canonical24_sha256': 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'})

    def test_pure_manual_environment_accepts_only_exact_scope(self):
        self.assertTrue(self.context())
        invalid = [dict(system='Windows'), dict(system='Darwin'), dict(device='cpu'), dict(device='cuda:1'),
                   dict(root='/home/test/forensic-dgp/old-packet'), dict(root='/home/test/forensic-dgp/' + self.module.VM_NAME + '/'),
                   dict(root='/home/test/forensic-dgp/../forensic-dgp/' + self.module.VM_NAME),
                   dict(tmux=''), dict(tmux=None), dict(gpu_names=()), dict(gpu_names=('NVIDIA A100',)),
                   dict(gpu_names=('NVIDIA L4', 'NVIDIA L4')), dict(gpu_names=['NVIDIA L4']),
                   dict(acknowledgement=None)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.context(**changes)

    def test_windows_factory_stops_before_framework_probe_or_artifact_access(self):
        with patch.object(self.module.platform, 'system', return_value='Windows'), \
                patch.object(self.module.subprocess, 'run', side_effect=AssertionError('No GPU probe permitted')), \
                patch.object(self.module, '_file_digest', side_effect=AssertionError('No artifact read permitted')):
            with self.assertRaisesRegex(ValueError, 'Manual Linux'):
                self.module.build_restorer(object(), {}, r2_artifact='unopened', u96_artifact='unopened',
                    manual_vm_root='unused', prototype_source_sha256='0' * 64,
                    acknowledgement=self.module.MANUAL_ACKNOWLEDGEMENT)

    def test_windows_delivery_stops_before_accessing_supplied_framework(self):
        class ForbiddenFramework:
            def __getattribute__(self, name):
                raise AssertionError('No framework operation permitted: ' + name)
        with patch.object(self.module.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'Manual Linux'):
                self.module.connected_rgb256(object(), torch=ForbiddenFramework(), manual_vm_root='unused')

    def test_only_framework_import_is_inside_factory_after_guard(self):
        imports = [node for node in ast.walk(self.tree) if isinstance(node, ast.Import)
                   and any(alias.name == 'torch' for alias in node.names)]
        self.assertEqual(len(imports), 1)
        factory = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_restorer')
        first_call = next(node for node in factory.body if isinstance(node, ast.Assign))
        self.assertEqual(first_call.value.func.id, '_guard_manual_vm')
        self.assertLess(first_call.lineno, imports[0].lineno)
        self.assertIn(imports[0], factory.body)
        top = [node for node in self.tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(node) or 'third_party' in ast.unparse(node) for node in top))

    def test_delivery_graph_has_real_clamp_resize_without_surrogate(self):
        delivery = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'connected_rgb256')
        calls = [node for node in ast.walk(delivery) if isinstance(node, ast.Call)]
        attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
        self.assertIn('clamp', attributes)
        self.assertIn('interpolate', attributes)
        self.assertFalse(set(attributes) & {'detach', 'floor', 'round', 'no_grad', 'inference_mode', 'numpy', 'cpu', 'backward'})
        resize = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'interpolate')
        self.assertEqual(ast.literal_eval(resize.args[1]), (256, 256))
        self.assertEqual({item.arg: ast.literal_eval(item.value) for item in resize.keywords},
                         {'mode': 'bilinear', 'align_corners': False})
        self.assertFalse(any(isinstance(node, ast.With) for node in ast.walk(delivery)))

    def test_no_trainer_optimizer_loader_or_automatic_execution(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ('optim', 'backward', 'step', 'load', 'save', 'GradScaler'))
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, ('Adam', 'AdamW', 'SGD', 'DataLoader'))
        self.assertFalse(any(isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)
                             for node in self.tree.body))

    def test_hook_and_lock_cleanup_is_exception_safe_in_source(self):
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef) and node.name == 'U96DecoderReadout')
        forward = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        guarded = [node for node in forward.body if isinstance(node, ast.Try) and node.finalbody]
        self.assertEqual(len(guarded), 1)
        cleanup = ast.unparse(guarded[0].finalbody[0])
        self.assertIn('handle.remove()', cleanup)
        self.assertIn('self._forward_lock.release()', cleanup)
        self.assertIn('finally:', cleanup)
        self.assertIn('blocking=False', ast.unparse(forward))

    def test_existing_trainable_layer_and_eval_policy_are_explicit(self):
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef) and node.name == 'U96DecoderReadout')
        init = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        text = ast.unparse(init)
        self.assertIn('self.requires_grad_(False)', text)
        self.assertIn('self.base.generator.blocks[24].requires_grad_(True)', text)
        self.assertIn('self.eval()', text)
        train = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
        self.assertIn('super().train(False)', ast.unparse(train))
        self.assertNotIn('Conv2d', text)

    def test_stock_forward_and_cpu_rng_scope_are_explicit(self):
        text = ast.unparse(self.tree)
        self.assertIn('torch.random.fork_rng(devices=[])', text)
        self.assertIn("{'device': 'cpu', 'dtype': torch.float32}", text)
        self.assertIn('self.base(sample, w=1.0, detach_16=True, code_only=False, adain=True)', text)
        self.assertIn('_OWNERS_LOCK', text)
        self.assertIn('weakref.ref(candidate)', text)
        self.assertIn('initial_frozen_digest', text)
        self.assertIn('if name not in base_readout_names', text)


if __name__ == '__main__':
    unittest.main()
