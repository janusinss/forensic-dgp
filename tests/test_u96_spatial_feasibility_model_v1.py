"""Pure standard-library contract/AST/guard tests; no model or Torch execution."""
import ast
import builtins
import importlib.util
import math
from pathlib import Path
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'research/codeformer_u96_spatial_feasibility_v1/model.py'


def load_pure():
    spec = importlib.util.spec_from_file_location('pure_spatial_feasibility_prototype', SOURCE)
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


class SpatialFeasibilityPureTests(unittest.TestCase):
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

    def test_complete_existing_spatial_schema_matches_independent_review(self):
        expected = {
            'base.generator.blocks.22.norm1.weight': (64,),
            'base.generator.blocks.22.norm1.bias': (64,),
            'base.generator.blocks.22.conv1.weight': (64, 64, 3, 3),
            'base.generator.blocks.22.conv1.bias': (64,),
            'base.generator.blocks.22.norm2.weight': (64,),
            'base.generator.blocks.22.norm2.bias': (64,),
            'base.generator.blocks.22.conv2.weight': (64, 64, 3, 3),
            'base.generator.blocks.22.conv2.bias': (64,),
            'base.generator.blocks.24.weight': (3, 64, 3, 3),
            'base.generator.blocks.24.bias': (3,),
        }
        actual = self.module.expected_spatial_shapes()
        self.assertEqual(actual, expected)
        self.assertEqual(sum(math.prod(shape) for shape in actual.values()), 75843)
        spatial = {name: shape for name, shape in actual.items() if '.22.' in name}
        self.assertEqual((len(spatial), sum(math.prod(shape) for shape in spatial.values())), (8, 74112))
        names = sorted(actual)
        self.assertTrue(all('.22.' in name for name in names[:8]))
        self.assertEqual(sum(math.prod(actual[name]) for name in names[:8]), 74112)
        self.assertEqual(sum(math.prod(actual[name]) for name in names[8:]), 1731)
        actual.clear()
        self.assertEqual(len(self.module.expected_spatial_shapes()), 10)

    def test_new_root_and_schema_refuse_historical_readout_scope(self):
        self.assertEqual(self.module.VM_NAME, 'cctv_codeformer_u96_spatial_feasibility_v1_vm')
        with self.assertRaises(ValueError):
            self.context(root='/home/test/forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm')
        contract = self.module.architecture_contract()
        self.assertEqual((contract['existing_residual_block'], contract['existing_outer_normalization_frozen'],
                          contract['existing_RGB_readout']), (22, 23, 24))
        self.assertFalse(contract['new_architecture_modules'])
        self.assertEqual(contract['RGB_readout_control_shapes'],
                         {name: list(shape) for name, shape in self.module.expected_readout_shapes().items()})

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
                          contract['trainable_tensors'], contract['trainable_parameters']), (539, 529, 10, 75843))
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
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef) and node.name == 'U96SpatialFeasibility')
        forward = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        guarded = [node for node in forward.body if isinstance(node, ast.Try) and node.finalbody]
        self.assertEqual(len(guarded), 1)
        cleanup = ast.unparse(guarded[0].finalbody[0])
        self.assertIn('handle.remove()', cleanup)
        self.assertIn('self._forward_lock.release()', cleanup)
        self.assertIn('finally:', cleanup)
        self.assertIn('blocking=False', ast.unparse(forward))

    def test_existing_trainable_layer_and_eval_policy_are_explicit(self):
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef) and node.name == 'U96SpatialFeasibility')
        init = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        text = ast.unparse(init)
        self.assertIn('self.requires_grad_(False)', text)
        self.assertIn('self.base.generator.blocks[24].requires_grad_(True)', text)
        self.assertIn('self.base.generator.blocks[22].requires_grad_(True)', text)
        self.assertNotIn('self.base.generator.blocks[21].requires_grad_(True)', text)
        self.assertNotIn('self.base.generator.blocks[23].requires_grad_(True)', text)
        self.assertIn('self.eval()', text)
        train = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
        self.assertIn('super().train(False)', ast.unparse(train))
        self.assertNotIn('Conv2d', text)

    def test_runtime_residual_and_frozen_normalization_schema_bound(self):
        factory = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_restorer')
        text = ast.unparse(factory)
        for binding in ('spatial = r2_base.generator.blocks[22]',
                        'final_normalization = r2_base.generator.blocks[23]',
                        "type(spatial).__name__ == 'ResBlock'",
                        'spatial.in_channels == spatial.out_channels == 64',
                        "not hasattr(spatial, 'conv_out')",
                        'type(norm) is torch.nn.GroupNorm', 'norm.num_groups == 32',
                        'norm.num_channels == 64', 'norm.eps == 1e-06',
                        'for conv in (spatial.conv1, spatial.conv2):'):
            self.assertIn(binding, text)
        self.assertIn('set(spatial._modules)', text)

    def test_forward_binds_block22_norm23_readout24_connected_identity(self):
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)
                       and node.name == 'U96SpatialFeasibility')
        forward = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        text = ast.unparse(forward)
        for expression in ('spatial.register_forward_pre_hook(spatial_guard)',
                           'spatial.register_forward_hook(capture_spatial)',
                           'final_normalization.register_forward_pre_hook(normalization_guard)',
                           'final_normalization.register_forward_hook(capture_normalized)',
                           'args[0] is last_spatial', 'args[0] is last_normalized',
                           'args[0].requires_grad', 'output.requires_grad',
                           'base_calls == spatial_calls == normalization_calls == readout_calls == 1',
                           'self.last_spatial_calls = spatial_calls',
                           'self.last_normalization_calls = normalization_calls'):
            self.assertIn(expression, text)
        guards = {node.name: node for node in forward.body if isinstance(node, ast.FunctionDef)}
        self.assertIn('not args[0].requires_grad', ast.unparse(guards['spatial_guard']))
        self.assertNotIn('not args[0].requires_grad', ast.unparse(guards['readout_guard']))
        self.assertIn('output.grad_fn is not None', ast.unparse(guards['capture_spatial']))

    def test_complete_state_capture_is_explicit_locked_and_neural_free(self):
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)
                       and node.name == 'U96SpatialFeasibility')
        capture = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'capture_complete_state_for_audit')
        text = ast.unparse(capture)
        self.assertIn('_bound_root(root)', text)
        self.assertIn('self._forward_lock.acquire(blocking=False)', text)
        self.assertIn('self.validate_invariants(check_frozen_digests=True)', text)
        self.assertIn('value.detach().cpu().contiguous().clone()', text)
        self.assertIn('len(captured) == TOTAL_TENSORS', text)
        self.assertIn('self._forward_lock.release()', text)
        calls = [node.func for node in ast.walk(capture) if isinstance(node, ast.Call)]
        forbidden = {'forward', 'backward', 'save', 'load', 'step', 'zero_grad', 'autograd', 'optim'}
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in calls))
        targets = [node.targets[0] for node in ast.walk(capture) if isinstance(node, ast.Assign)]
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr.startswith('last_') for node in targets))
        self.assertTrue(any(isinstance(node, ast.Try) and node.finalbody for node in capture.body))

    def detached_capture_fixture(self, *, fail_invariants=False, finite=True):
        """Execute only the isolated capture method on plain non-neural stubs."""
        wrapper = next(node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)
                       and node.name == 'U96SpatialFeasibility')
        capture = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'capture_complete_state_for_audit')
        method_module = ast.Module(body=[capture], type_ignores=[])
        dtype = object()

        class PlainArray:
            def __init__(self, values):
                self.values = list(values)
                self.dtype = dtype
                self.device = SimpleNamespace(type='cpu')
                self.requires_grad = False
                self.grad_fn = None
            def detach(self):
                return self
            def cpu(self):
                return self
            def contiguous(self):
                return self
            def clone(self):
                return PlainArray(self.values)

        class PlainStateHolder:
            def __init__(self):
                self._forward_lock = threading.Lock()
                self.state = {'tensor.' + str(i): PlainArray([i]) for i in range(539)}
                self.invariant_calls = []
                self.last_base_calls = 12
                self.last_spatial_calls = 12
            def validate_invariants(self, *, check_frozen_digests):
                self.invariant_calls.append(check_frozen_digests)
                if fail_invariants:
                    raise ValueError('fixture invariant failure')
            def state_dict(self):
                return self.state
            def named_parameters(self):
                return self.state.items()

        namespace = {'_bound_root': lambda root: root, 'root': 'not-a-real-VM',
                     'require': self.module.require, 'TOTAL_TENSORS': 539,
                     'torch': SimpleNamespace(float32=dtype, isfinite=lambda value:
                         SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: finite)))}
        exec(compile(method_module, '<isolated source-only capture fixture>', 'exec'), namespace)
        holder = PlainStateHolder()
        return holder, lambda: namespace['capture_complete_state_for_audit'](holder)

    def test_complete_capture_functional_fixture_clones_all539_without_counter_changes(self):
        holder, capture = self.detached_capture_fixture()
        result = capture()
        self.assertEqual(len(result), 539)
        self.assertEqual(set(result), set(holder.state))
        for name in result:
            self.assertIsNot(result[name], holder.state[name])
            self.assertEqual(result[name].values, holder.state[name].values)
        result['tensor.0'].values[0] = -1
        self.assertEqual(holder.state['tensor.0'].values, [0])
        self.assertEqual(holder.invariant_calls, [True])
        self.assertEqual((holder.last_base_calls, holder.last_spatial_calls), (12, 12))
        self.assertTrue(holder._forward_lock.acquire(blocking=False))
        holder._forward_lock.release()

    def test_complete_capture_fixture_failure_releases_lock(self):
        for options in ({'fail_invariants': True}, {'finite': False}):
            holder, capture = self.detached_capture_fixture(**options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                capture()
            self.assertTrue(holder._forward_lock.acquire(blocking=False))
            holder._forward_lock.release()

    def test_complete_capture_fixture_refuses_concurrent_holder(self):
        holder, capture = self.detached_capture_fixture()
        self.assertTrue(holder._forward_lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(ValueError, 'Concurrent'):
                capture()
            self.assertEqual(holder.invariant_calls, [])
            self.assertFalse(holder._forward_lock.acquire(blocking=False))
        finally:
            holder._forward_lock.release()

    def test_stock_forward_and_cpu_rng_scope_are_explicit(self):
        text = ast.unparse(self.tree)
        self.assertIn('torch.random.fork_rng(devices=[])', text)
        self.assertIn("{'device': 'cpu', 'dtype': torch.float32}", text)
        self.assertIn('self.base(sample, w=1.0, detach_16=True, code_only=False, adain=True)', text)
        self.assertIn('_OWNERS_LOCK', text)
        self.assertIn('weakref.ref(candidate)', text)
        self.assertIn('initial_frozen_digest', text)
        self.assertIn('if name not in base_trainable_names', text)


if __name__ == '__main__':
    unittest.main()
