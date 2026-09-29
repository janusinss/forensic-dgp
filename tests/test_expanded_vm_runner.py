import importlib.util
import unittest
from unittest.mock import patch
import torch


class ExpandedRunnerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.train_expanded_feature_vm'))
        from scripts import train_expanded_feature_vm
        return train_expanded_feature_vm

    def test_cpu_refused_before_reading_or_writing_workspace(self):
        api=self.api()
        with patch('torch.cuda.is_available',return_value=False),patch('pathlib.Path.resolve',side_effect=AssertionError('workspace accessed')):
            with self.assertRaisesRegex(RuntimeError,'VM GPU'):api.main('.',preflight=True)

    def test_heads_identical_initialization_and_detached_encoder_features(self):
        api=self.api()
        from scripts.compare_pixel_heads_vm import PixelHead
        from scripts.compare_presence_heads_vm import PresenceHead
        torch.set_num_threads(2);torch.manual_seed(42)
        pixel=PixelHead(1).state_dict();presence=PresenceHead(4).state_dict()
        p,g=api.make_heads(pixel,presence,'cpu');q,h=api.make_heads(pixel,presence,'cpu')
        self.assertEqual(api.head_digest(p,g),api.head_digest(q,h))
        f=torch.randn(2,256,8,8,requires_grad=True)
        z=p(f,(16,16));t=torch.zeros_like(z);t[0,:,4:10,4:10]=1
        loss=api.training_loss(z,g(f),t);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIsNone(f.grad)
        self.assertTrue(all(v.grad is not None for v in list(p.parameters())+list(g.parameters())))
        self.assertEqual(api.head_digest(p,g),api.head_digest(q,h))

    def test_known_gate_counts(self):
        api=self.api()
        raw=torch.tensor([[[[True,False],[False,True]]],[[[True,False],[False,False]]]])
        target=torch.tensor([[[[True,True],[False,False]]],[[[False,False],[False,False]]]])
        counts=api.pixel_counts(raw&torch.tensor([False,True])[:,None,None,None],target)
        self.assertEqual(counts,{'tp':0,'fp':1,'fn':2,'visible':6,'empty':1,'negative_fp':1,'positive':1,'negative':1})

    def test_fixed_protocol_has_no_validation_selection(self):
        api=self.api();p=api.fixed_protocol()
        self.assertEqual(p['epochs']*p['steps_per_epoch'],1600)
        self.assertEqual(p['arms'],['fixed','anatomical'])
        self.assertEqual(p['thresholds'],{'pixel':.5,'presence':.5})
        self.assertEqual(p['selection'],'final only; external unchanged safeguards')
