import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from scripts.train_refinement_vm import verify_input,main


class RefinementVMTests(unittest.TestCase):
    def test_input_binding(self):
        rgb=np.zeros((2,2,3),dtype=np.uint8);target=np.zeros((1,2,2),dtype=np.uint8)
        record={'input_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
        verify_input(rgb,target,record,target.copy())
        with self.assertRaisesRegex(ValueError,'RGB'):verify_input(rgb+1,target,record,target)
        with self.assertRaisesRegex(ValueError,'Target'):verify_input(rgb,target,record,target+1)

    def test_gpu_guard(self):
        with patch('torch.cuda.is_available',return_value=False),patch('pathlib.Path.resolve',side_effect=AssertionError('access')):
            with self.assertRaisesRegex(RuntimeError,'VM GPU'):main('.')
