"""Forward-only V3 loss, checkpoint lineage and VM execution boundary checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import cctv_dgp_perceptual_training_v3 as recipe
import run_cctv_dgp_perceptual_vm_v3 as runner
from test_cctv_dgp_perceptual_v3 import fixture


class PerceptualTrainingContracts(unittest.TestCase):
    def tearDown(self):
        recipe.configure_paths(ROOT)

    def test_starting_state_is_loaded_from_the_audited_return(self):
        parent = ROOT / 'scratch/test-parent-return'
        recipe.configure_paths(ROOT / 'scratch/test-v3-bundle', parent)
        weights = {'probe': torch.tensor([1.])}
        with patch.object(recipe, 'sha', return_value=recipe.START_SHA), \
                patch('torch.load', return_value=weights) as load, \
                patch.object(recipe, 'state_hash', return_value=recipe.START_STATE, create=True):
            result = recipe.load_starting_state(ROOT / 'scratch/separate-original-data-root')
        self.assertIs(result, weights)
        load.assert_called_once_with(parent / 'camera_identity/best.pth',
                                     map_location='cpu', weights_only=True)

    def test_changed_starting_tensor_state_is_rejected(self):
        with patch.object(recipe, 'sha', return_value=recipe.START_SHA), \
                patch('torch.load', return_value={}), \
                patch.object(recipe, 'state_hash', return_value='wrong-state', create=True):
            with self.assertRaisesRegex(ValueError, 'starting tensor state'):
                recipe.load_starting_state(ROOT)

    def test_boolean_or_nonfinite_scales_are_rejected(self):
        for scales in ([True, 1., 1., 1.], [float('nan'), 1., 1., 1.],
                       [0., 1., 1., 1.], [1., 1., 1.]):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                recipe.validate_scales(scales)

    def test_calibration_applies_only_to_signed_feature_policy(self):
        def initialize(instance, path, device):
            replacement = fixture('postactivation')
            nn.Module.__init__(instance)
            instance.features = replacement.features
            instance.register_buffer('mean', replacement.mean)
            instance.register_buffer('std', replacement.std)
            instance.eval().requires_grad_(False)

        with patch.object(recipe, 'read', return_value={'preactivation_scales': [.5, .4, .3, .2]}), \
                patch('cctv_dgp_pilot.PilotPerceptual.__init__', initialize):
            model = recipe.CalibratedPerceptualV3('unused', 'cpu')
        image = torch.tensor([-1., .25, .5]).reshape(1, 3, 1, 1)
        target = torch.zeros_like(image)
        with torch.no_grad():
            expected_post = sum(w * torch.nn.functional.l1_loss(a, b)
                                for w, a, b in zip((.1, .2, 1., 1.), model.taps(image), model.taps(target)))
            torch.testing.assert_close(model(image, target), expected_post, rtol=0, atol=0)
            model.select_policy('preactivation')
            expected_pre = sum(w * scale * torch.nn.functional.l1_loss(a, b)
                               for w, scale, a, b in zip((.1, .2, 1., 1.), (.5, .4, .3, .2),
                                                        model.taps(image), model.taps(target)))
            torch.testing.assert_close(model(image, target), expected_pre, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_local_run_is_rejected_before_loading_or_output_creation(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.load') as load, patch.object(runner, 'DGPSynthesizer') as model, \
                patch.object(runner, 'verify_bundle') as verify, \
                patch.object(Path, 'mkdir') as mkdir, patch('torch.optim.Adam') as optimizer:
            with self.assertRaisesRegex(RuntimeError, 'CUDA required'):
                runner.run(ROOT, ROOT / 'scratch/forbidden-local-training')
        for operation in (load, model, verify, mkdir, optimizer):
            operation.assert_not_called()

    def test_cuda_on_another_host_is_rejected(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner.sys, 'platform', 'linux'), \
                patch.object(runner.platform, 'node', return_value='different-vm'):
            with self.assertRaisesRegex(RuntimeError, 'Linux VM'):
                runner.require_vm(ROOT)


if __name__ == '__main__':
    unittest.main()
