"""Meaningful interface checks; no backward calls or local training."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from dgp_face_code_conditioner_v11 import DGPFaceCodePrior, check_rgb


class Contracts(unittest.TestCase):
    def test_training_enable_rejects_local_machine_before_parameters_change(self):
        model = DGPFaceCodePrior(nn.Conv2d(3, 3, 1), nn.Linear(2, 2))
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, 'only on'):
            model.enable_vm_training('.')
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.conditioner.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))

    def test_training_graph_rejects_before_encoder_access(self):
        model = DGPFaceCodePrior(nn.Identity(), nn.Identity())
        with patch.object(model, 'encode', side_effect=AssertionError('must not enter encoder')):
            with self.assertRaisesRegex(RuntimeError, 'authorization'):
                model.training_codes(torch.zeros(1, 3, 256, 256))
        with self.assertRaisesRegex(RuntimeError, 'local training'):
            model.train(True)

    def test_rgb_contract_rejects_wrong_scale_shape_dtype_and_nonfinite(self):
        check_rgb(torch.zeros(1, 3, 256, 256, dtype=torch.float32))
        bad = [torch.zeros(0, 3, 256, 256), torch.zeros(1, 3, 128, 128),
               torch.zeros(1, 3, 256, 256, dtype=torch.float64),
               torch.full((1, 3, 256, 256), 1.01), torch.full((1, 3, 256, 256), float('nan'))]
        for tensor in bad:
            with self.subTest(shape=tensor.shape), self.assertRaisesRegex(ValueError, 'finite nonempty'):
                check_rgb(tensor)

    def test_unknown_codes_or_fidelity_reject_before_codebook_access(self):
        model = DGPFaceCodePrior(nn.Identity(), nn.Identity())
        cases = [torch.zeros(1, 256, dtype=torch.float32), torch.zeros(0, 256, dtype=torch.int64),
                 torch.full((1, 256), -1), torch.full((1, 256), 1024), torch.zeros(1, 255, dtype=torch.int64)]
        for codes in cases:
            with self.subTest(shape=codes.shape), self.assertRaisesRegex(ValueError, 'integer face codes'):
                model.decode(codes)
        with self.assertRaisesRegex(ValueError, 'Fidelity'):
            model(torch.zeros(1, 3, 256, 256), fidelity=1.5)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
