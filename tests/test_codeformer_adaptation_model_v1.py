"""CPU metadata/value tests only: no backward, optimizer or learned updates."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from research.codeformer_adaptation_v1.common import tensor_state_digest
from research.codeformer_adaptation_v1.model import (
    TRAINABLE_GROUPS, FROZEN_GROUPS, delivered_rgb, parameter_group, prepare_input,
)


class GeometryAndStateTests(unittest.TestCase):
    def test_input_exactly_matches_selected_benchmark_policy(self):
        image = torch.linspace(0, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
        with torch.no_grad():
            expected = F.interpolate(image, size=(512, 512), mode="bilinear",
                                     align_corners=False) * 2 - 1
            self.assertTrue(torch.equal(prepare_input(image), expected))

    def test_output_clipping_precedes_return_resize(self):
        output = torch.linspace(-1.2, 1.2, 3 * 512 * 512).reshape(1, 3, 512, 512)
        with torch.no_grad():
            expected = F.interpolate(((output + 1) / 2).clamp(0, 1), (256, 256),
                                     mode="bilinear", align_corners=False)
            self.assertTrue(torch.equal(delivered_rgb(output), expected))

    def test_rejects_wrong_geometry_range_and_nan(self):
        for image in (torch.zeros(1, 3, 128, 128), torch.full((1, 3, 256, 256), 2.),
                      torch.full((1, 3, 256, 256), float("nan"))):
            with self.assertRaises(ValueError):
                prepare_input(image)

    def test_no_silent_new_parameter_group(self):
        for prefix in TRAINABLE_GROUPS + FROZEN_GROUPS:
            self.assertEqual(parameter_group(prefix + ".weight"), prefix)
        with self.assertRaises(ValueError):
            parameter_group("new_undeclared_head.weight")

    def test_hash_covers_scalar_dtype_names_shape_and_values(self):
        initial = {"a": torch.tensor(1.), "b": torch.tensor([2.])}
        digest = tensor_state_digest(initial)
        self.assertEqual(digest, tensor_state_digest(dict(reversed(list(initial.items())))))
        for mutation in ({"a": torch.tensor(1.), "b": torch.tensor([3.])},
                         {"c": torch.tensor(1.), "b": torch.tensor([2.])},
                         {"a": torch.tensor([1.]), "b": torch.tensor([2.])},
                         {"a": torch.tensor(1., dtype=torch.float64), "b": torch.tensor([2.])}):
            self.assertNotEqual(digest, tensor_state_digest(mutation))


if __name__ == "__main__":
    unittest.main()
