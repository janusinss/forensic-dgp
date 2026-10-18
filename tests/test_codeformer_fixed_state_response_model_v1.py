"""NumPy/AST/manual-guard fixtures only; never import a neural framework."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

from research.codeformer_fixed_state_response_v1 import model as m

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(m.__file__).read_text(encoding='utf-8')
TREE = ast.parse(SOURCE, feature_version=(3,10))


def response(base, input_rgb, observed, logit, target):
    delta = np.float32(.25 * math.tanh(logit))
    raw = np.clip(base + delta, np.float32(0), np.float32(1))
    raw[~observed] = input_rgb[~observed]
    weights = np.asarray([.2126,.7152,.0722],np.float32)
    luminance = (raw * weights).sum(axis=2,dtype=np.float32)
    mean = np.float32(luminance[observed].sum(dtype=np.float32) / np.float32(observed.sum()))
    loss = np.float32(np.float32(mean - np.float32(target)) ** np.float32(2))
    return raw, float(delta), float(mean), float(loss)


class FrozenResponseMath(unittest.TestCase):
    def test_zero_head_preserves_observed_base_and_protected_input(self):
        base = np.full((2,2,3),.4,np.float32)
        inp = np.full_like(base,.9); mask=np.asarray([[1,0],[1,1]],bool)
        raw,delta,mean,loss=response(base,inp,mask,0,.45)
        self.assertEqual(delta,0)
        self.assertTrue(np.array_equal(raw[mask],base[mask]))
        self.assertTrue(np.array_equal(raw[~mask],inp[~mask]))
        self.assertAlmostEqual(mean,.4,places=6)
        self.assertAlmostEqual(loss,.0025,places=7)

    def test_actual_clipping_mean_differs_from_unclipped_offset_proxy(self):
        base=np.broadcast_to(np.asarray([.95,.3,.1],np.float32),(2,2,3)).copy()
        inp=np.full_like(base,.8); mask=np.asarray([[1,1],[0,1]],bool)
        raw,delta,mean,loss=response(base,inp,mask,math.atanh(.4),.6)
        self.assertAlmostEqual(delta,.1,places=7)
        self.assertTrue(np.array_equal(raw[mask][:,0],np.ones(3,np.float32)))
        unclipped=float((base[mask] @ np.asarray([.2126,.7152,.0722],np.float32)).mean())+delta
        self.assertLess(mean,unclipped)
        self.assertAlmostEqual(loss,(mean-np.float32(.6))**2,places=7)

    def test_protected_pixels_do_not_enter_target_or_response_mean(self):
        base=np.full((2,2,3),.3,np.float32); mask=np.asarray([[1,0],[1,1]],bool)
        a=response(base,np.zeros_like(base),mask,-.2,.4)
        b=response(base,np.ones_like(base),mask,-.2,.4)
        self.assertEqual(a[1:],b[1:])
        self.assertFalse(np.array_equal(a[0][~mask],b[0][~mask]))

    def test_signed_correction_is_bounded_and_not_a_role_switch(self):
        self.assertEqual(np.float32(.25*math.tanh(1000)),np.float32(.25))
        self.assertEqual(np.float32(.25*math.tanh(-1000)),np.float32(-.25))
        base=np.full((2,2,3),.5,np.float32); mask=np.ones((2,2),bool)
        normal=response(base,base,mask,-.2,.45)
        dark=response(base,base,mask,-.2,.5)
        self.assertTrue(np.array_equal(normal[0],dark[0]))
        self.assertEqual(normal[1:3],dark[1:3])
        self.assertNotEqual(normal[3],dark[3])

    def test_target_scalar_loss_uses_FP32_difference(self):
        base=np.full((2,2,3),.4,np.float32); mask=np.ones((2,2),bool)
        _,_,mean,loss=response(base,base,mask,0,.39999999)
        error=np.float32(np.float32(mean)-np.float32(.39999999))
        self.assertEqual(loss,float(np.float32(error*error)))


class ManualSourceBoundaries(unittest.TestCase):
    def test_module_import_never_imports_neural_framework(self):
        self.assertFalse(any(n in sys.modules for n in ('torch','tensorflow','jax')))
        imports=[]
        for function in [n for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
            for node in ast.walk(function):
                if isinstance(node,ast.Import) and any(a.name=='torch' for a in node.names):imports.append(function.name)
        self.assertEqual(imports,['activate_runtime'])
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in TREE.body))

    def test_windows_guard_refuses_before_path_contract_or_Torch(self):
        with mock.patch.object(m.platform,'system',return_value='Windows'), \
             mock.patch.object(m.Path,'home',side_effect=AssertionError('filesystem reached')):
            with self.assertRaisesRegex(ValueError,'local neural calls forbidden'):
                m.activate_runtime(Path('.'),'0'*64)
        self.assertNotIn('torch',sys.modules)

    def test_all_Torch_helpers_refuse_without_activation(self):
        calls=[lambda:m.tensor_state_digest({}),lambda:m.tensor_digest({}),lambda:m.state_digest(None),
               lambda:m.cpu_state(None),lambda:m._head(None),lambda:m.infer_rgb(None,None),
               lambda:m.load_frozen_parents_and_heads(None,None,None),lambda:m.mean_target(None,None),
               lambda:m.head_response(None,None,None,None,None,0),lambda:m.frozen_state_receipt(None,None,{})]
        for call in calls:
            with self.assertRaisesRegex(ValueError,'guarded manual L4'):
                call()

    def test_feature_wrapper_delegates_to_single_data_helper(self):
        name='research.codeformer_fixed_state_response_v1.data'
        fake=types.ModuleType(name);fake.features=mock.Mock(return_value='sentinel')
        with mock.patch.dict(sys.modules,{name:fake}):
            self.assertEqual(m.features_numpy('input','mask'),'sentinel')
        fake.features.assert_called_once_with('input','mask')

    def test_no_optimizer_autograd_backward_or_old_activation_calls(self):
        forbidden=[]
        for node in ast.walk(TREE):
            if isinstance(node,ast.Call):
                name=ast.unparse(node.func)
                if any(x in name for x in ('autograd','backward','optim.','optimizer.','set_rng_state','manual_seed')):
                    forbidden.append(name)
        self.assertEqual(forbidden,[])
        self.assertNotIn('codeformer_tone_control_training_v1.model',SOURCE)
        self.assertNotIn('_BASE.activate_runtime',SOURCE)

    def test_model_loads_follow_hash_and_origin_checks(self):
        self.assertLess(SOURCE.index("paths[name] = _check_pin"),SOURCE.index('saved = torch.load'))
        self.assertLess(SOURCE.index('_BASE = _historical_imports'),SOURCE.index('_BASE.load_trainable'))
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.load']
        self.assertEqual(len(calls),1) # the two composite states use a fixed two-iteration loop
        self.assertEqual({k.arg:ast.literal_eval(k.value) for k in calls[0].keywords},
                         {'map_location':'cpu','weights_only':True,'mmap':True})
        self.assertIn('Path(spec.origin).resolve(strict=True)',SOURCE)

    def test_frozen_factory_exact_shapes_digests_and_construction_RNG(self):
        self.assertEqual(sum(math.prod(s) for s in m.HEAD_SHAPES.values()),8257)
        self.assertEqual(len(m.HISTORICAL_MODULES),9)
        self.assertIn('torch.random.fork_rng(devices=[0])',SOURCE)
        self.assertIn('all(not m.training for m in module.modules())',SOURCE)
        self.assertIn('p.grad is None',SOURCE)
        self.assertIn('p.requires_grad',SOURCE)
        self.assertIn('base.load_state_dict',SOURCE)
        self.assertIn('head.load_state_dict',SOURCE)
        self.assertIn('strict=True',SOURCE)

    def test_checkpoint_asset_pins_match_closed_state_receipt_without_loading(self):
        doc=json.loads((ROOT/'outputs/codeformer_tone_control_training_v1_return_review/state_v1/state_audit_v1.json').read_text())
        cp=doc['sections']['checkpoints']['checkpoints']
        for asset,slot,key in [('initial_composite','composite_head0000','head0000'),
                               ('stopped_composite','stopped_composite','head0050')]:
            self.assertEqual(m.ASSETS[asset]['sha256'],cp[slot]['sha256'])
            self.assertEqual(m.ASSETS[asset]['bytes'],cp[slot]['bytes'])
            self.assertEqual(m.HEAD_DIGESTS[key],cp[slot]['head_tensor_sha256'])
            self.assertEqual(m.BASE_DIGEST,cp[slot]['base_model_tensor_sha256'])

    def test_historical_import_paths_are_all_bound_before_import(self):
        p=json.loads((ROOT/'outputs/codeformer_adain_gradient_isolation_v1_return/protocol.json').read_text())
        self.assertTrue(set(m.HISTORICAL_MODULES.values())<=set(p['files_sha256']))
        for relative in m.HISTORICAL_MODULES.values():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),p['files_sha256'][relative])

    def test_each_response_has_one_head_call_and_each_infer_one_CodeFormer_call(self):
        fn={n.name:n for n in TREE.body if isinstance(n,ast.FunctionDef)}
        head_calls=[n for n in ast.walk(fn['head_response']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='head']
        cf_calls=[n for n in ast.walk(fn['infer_rgb']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='net']
        self.assertEqual(len(head_calls),1);self.assertEqual(len(cf_calls),1)
        self.assertIn('torch.inference_mode()',ast.unparse(fn['head_response']))
        self.assertIn('torch.inference_mode()',ast.unparse(fn['infer_rgb']))

    def test_source_preserves_clipped_support_FP32_loss_and_ordinary_numpy_results(self):
        self.assertIn('torch.where(mt, (bt + delta.view(1,1,1,1)).clamp(0,1), it)',SOURCE)
        self.assertIn("mask = mt.to(dtype=torch.float32)",SOURCE)
        self.assertIn('loss = (mean - target_tensor).square().mean()',SOURCE)
        self.assertIn("result['raw'] = cpu.numpy().copy()",SOURCE)
        self.assertIn('return cpu.numpy().copy()',SOURCE)
        self.assertIn('tensor_state_digest(value if isinstance(value, dict)',SOURCE)


if __name__=='__main__':
    unittest.main()
