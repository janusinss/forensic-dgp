"""Contract checks on tiny modules; no model optimizer or training."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from face_occlusion_adapter import (FaceOcclusionAdapter, initialize_pair, strip_source_prefix,
                                    load_adapter, FORMAT, parameter_groups)


class TinySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.segmentation_head = nn.Sequential(nn.Conv2d(4, 1, 3, padding=1))

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


class FaceOcclusionAdapterTests(unittest.TestCase):
    def source(self):
        with torch.random.fork_rng():
            torch.manual_seed(7)
            network = TinySegmentation()
            with torch.no_grad():
                network.encoder[0].weight.fill_(.2)
                network.segmentation_head[0].weight.fill_(.8)
                network.segmentation_head[0].bias.fill_(3.)
            return {'module.'+k: v.clone() for k,v in network.state_dict().items()}

    def test_pair_has_identical_new_heads_distinct_backbones_and_immutable_source(self):
        source = self.source()
        preserved = copy.deepcopy(source)
        before_rng = torch.get_rng_state().clone()
        pair = initialize_pair(source, factory=TinySegmentation, seed=42)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        for k, v in source.items():
            self.assertTrue(torch.equal(v, preserved[k]))
        pretrained, random = pair['pretrained'], pair['random']
        for k, v in pretrained.network.segmentation_head.state_dict().items():
            self.assertTrue(torch.equal(v, random.network.segmentation_head.state_dict()[k]))
        self.assertFalse(torch.equal(pretrained.network.encoder[0].weight, random.network.encoder[0].weight))
        self.assertTrue(torch.equal(pretrained.network.encoder[0].weight, source['module.encoder.0.weight']))
        self.assertFalse(torch.equal(pretrained.network.segmentation_head[0].weight, source['module.segmentation_head.0.weight']))
        self.assertTrue(torch.equal(pretrained.reference_visible_head[0].weight, source['module.segmentation_head.0.weight']))
        self.assertTrue(all(not p.requires_grad for p in pretrained.reference_visible_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in pretrained.network.parameters()))
        self.assertEqual(pretrained.target, 'covered_region_is_one')

    def test_detect_is_direct_head_logits_and_running_statistics_stay_fixed(self):
        pair = initialize_pair(self.source(), factory=TinySegmentation)
        model = pair['pretrained']
        model.train()
        bn = model.network.encoder[1]
        before = {k: v.clone() for k,v in bn.state_dict().items()}
        self.assertFalse(bn.training)
        self.assertTrue(model.network.encoder[0].training)
        self.assertFalse(model.reference_visible_head.training)
        with torch.no_grad():
            model.network.segmentation_head[0].weight.zero_()
            model.network.segmentation_head[0].bias.fill_(2.)
        with torch.inference_mode():
            output = model.detect(torch.rand(2, 3, 8, 8))
        self.assertEqual(output.shape, (2, 1, 8, 8))
        self.assertTrue(torch.equal(output, torch.full_like(output, 2.)))
        for k,v in before.items():
            self.assertTrue(torch.equal(v, bn.state_dict()[k]))

    def test_strict_source_loading_and_prefix_validation(self):
        source = self.source()
        bad = dict(source); bad.pop(next(iter(bad)))
        with self.assertRaises(RuntimeError):
            initialize_pair(bad, factory=TinySegmentation)
        for bad in ({}, {'encoder.weight': torch.zeros(1)}, {'module.x': 'invalid'}):
            with self.assertRaises(ValueError):
                strip_source_prefix(bad)
        bad = dict(source); bad[next(iter(bad))] = torch.tensor(float('nan'))
        with self.assertRaises(ValueError):
            strip_source_prefix(bad)

    def test_invalid_input_and_format_are_rejected(self):
        model = initialize_pair(self.source(), factory=TinySegmentation)['random']
        for x in (torch.zeros(1, 1, 8, 8), torch.full((1,3,8,8), float('nan')), torch.full((1,3,8,8), 2.)):
            with self.assertRaises(ValueError):
                model.detect(x)
        with self.assertRaises(ValueError):
            FaceOcclusionAdapter(nn.Identity(), nn.Identity())

    def test_versioned_state_round_trip_rejects_initial_and_wrong_semantics(self):
        model = initialize_pair(self.source(), factory=TinySegmentation)['pretrained']
        payload = dict(format=FORMAT, target=model.target, architecture='resnet18-unet',
                       smp_version='0.5.0', model=model.state_dict(),
                       initialization='pretrained', optimizer_updates=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.pth'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'completed optimizer'):
                load_adapter(path, factory=TinySegmentation)
            loaded, _ = load_adapter(path, factory=TinySegmentation, allow_initial=True)
            image = torch.rand(1,3,8,8)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(image), loaded(image)))
            for key,value in [('target','visible_face'), ('format','unknown'), ('smp_version','0.3.0')]:
                bad = {**payload,key:value};torch.save(bad,path)
                with self.assertRaises(ValueError):
                    load_adapter(path,factory=TinySegmentation,allow_initial=True)

    def test_parameter_groups_cover_only_trainable_network_with_fixed_rates(self):
        model = initialize_pair(self.source(), factory=TinySegmentation)['pretrained']
        groups = parameter_groups(model)
        self.assertEqual([g['lr'] for g in groups], [1e-5,1e-4])
        optimized = [p for g in groups for p in g['params']]
        self.assertEqual(len(optimized), len({id(p) for p in optimized}))
        self.assertEqual({id(p) for p in optimized}, {id(p) for p in model.network.parameters()})
        self.assertFalse({id(p) for p in optimized} & {id(p) for p in model.reference_visible_head.parameters()})


if __name__ == '__main__':
    unittest.main()
