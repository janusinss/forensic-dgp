"""Forward-only regression for PyTorch 2.9 InstanceNorm buffer writeback."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_frozen_norm import install_frozen_instance_norm


def legacy_instance_norm(x, running_mean=None, running_var=None, weight=None,
                         bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    """Translate the published 2.9 native repeat/batch_norm/mean writeback path."""
    size, channels = x.shape[:2]
    repeat = lambda value: None if value is None else value.repeat(size)
    mean, var = repeat(running_mean), repeat(running_var)
    result = F.batch_norm(x.reshape(1, size * channels, *x.shape[2:]),
                          mean, var, repeat(weight), repeat(bias),
                          use_input_stats, momentum, eps)
    if running_mean is not None:
        running_mean.copy_(mean.reshape(size, channels).mean(0))
    if running_var is not None:
        running_var.copy_(var.reshape(size, channels).mean(0))
    return result.reshape_as(x)


def norm():
    layer = nn.InstanceNorm2d(3, affine=True, track_running_stats=True).eval()
    with torch.no_grad():
        layer.running_mean.copy_(torch.tensor([0.1, 1.5175138, 5.446949]))
        layer.running_var.copy_(torch.tensor([1.1, 31.353886, 818.8146]))
    return layer


def bytes_of(model):
    return {key: value.detach().cpu().numpy().tobytes()
            for key, value in model.state_dict().items()}


class FrozenInstanceNormRegression(unittest.TestCase):
    def test_legacy_tail_batch_preserves_registered_statistics(self):
        layer = norm()
        before = bytes_of(layer)
        install_frozen_instance_norm(layer)
        with patch("torch.nn.functional.instance_norm", legacy_instance_norm), torch.no_grad():
            layer(torch.zeros(6, 3, 4, 4))
        self.assertEqual(bytes_of(layer), before)

    def test_output_matches_legacy_eval_without_replacing_input_statistics(self):
        layer, reference = norm(), norm()
        reference.load_state_dict(layer.state_dict(), strict=True)
        install_frozen_instance_norm(layer)
        values = torch.linspace(-1, 1, 6 * 3 * 4 * 4).reshape(6, 3, 4, 4)
        with patch("torch.nn.functional.instance_norm", legacy_instance_norm), torch.no_grad():
            expected = reference(values)
            actual = layer(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_and_partial_batches_preserve_schema_and_trainable_parameters(self):
        layer = norm()
        before = bytes_of(layer)
        parameters = {key: (id(value), value.requires_grad)
                      for key, value in layer.named_parameters()}
        self.assertEqual(install_frozen_instance_norm(layer), 1)
        self.assertEqual(install_frozen_instance_norm(layer), 1)
        with torch.no_grad():
            for size in (8, 6, 1, 6):
                output = layer(torch.zeros(size, 3, 4, 4))
                self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(bytes_of(layer), before)
        self.assertEqual(parameters, {key: (id(value), value.requires_grad)
                                     for key, value in layer.named_parameters()})

    def test_accidental_training_mode_is_rejected(self):
        layer = norm()
        install_frozen_instance_norm(layer)
        layer.train()
        with self.assertRaisesRegex(RuntimeError, "frozen evaluation"):
            layer(torch.zeros(6, 3, 4, 4))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
