import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from scripts.train_projection_vm import main, assign_gradient, flat_parameters


class ProjectionVMTests(unittest.TestCase):
    def test_guard_precedes_files_and_model(self):
        with patch('scripts.train_projection_vm.require_vm_gpu',side_effect=RuntimeError('VM required')), patch('scripts.train_projection_vm.load_completion') as load:
            with self.assertRaises(RuntimeError):main(SimpleNamespace())
            load.assert_not_called()

    def test_flat_assignment_preserves_parameters(self):
        params=[torch.nn.Parameter(torch.tensor([1.,2.])),torch.nn.Parameter(torch.tensor([[3.]]))]
        before=flat_parameters(params).clone()
        assign_gradient(params,torch.tensor([4.,5.,6.]))
        self.assertTrue(torch.equal(params[0].grad,torch.tensor([4.,5.])))
        self.assertTrue(torch.equal(params[1].grad,torch.tensor([[6.]])))
        self.assertTrue(torch.equal(before,flat_parameters(params)))
        with self.assertRaises(ValueError):assign_gradient(params,torch.ones(2))
        with self.assertRaises(ValueError):assign_gradient(params,torch.full((3,),float('nan')))
