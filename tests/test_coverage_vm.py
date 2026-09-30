import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from scripts.train_coverage_vm import freeze_pixels, real_gate, require_vm_gpu, main


class CoverageVMTests(unittest.TestCase):
    def test_non_vm_refused_before_model_or_files(self):
        with patch('scripts.train_coverage_vm.sys.platform','win32'), patch('scripts.train_coverage_vm.load_completion') as load:
            with self.assertRaisesRegex(RuntimeError,'Linux VM CUDA'):
                main(SimpleNamespace())
            load.assert_not_called()

    def test_linux_without_cuda_refused(self):
        with patch('scripts.train_coverage_vm.sys.platform','linux'), patch('scripts.train_coverage_vm.torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):
                require_vm_gpu()

    def test_replay_bytes_are_lossless(self):
        q = torch.arange(256,dtype=torch.uint8).reshape(1,16,16).expand(3,-1,-1)
        mask = torch.zeros(1,16,16)
        mask[:,:,8:] = 1
        x,m = freeze_pixels(q.float()/255,mask)
        self.assertTrue(torch.equal(x,q))
        self.assertTrue(torch.equal(m.float(),mask))
        with self.assertRaises(ValueError):
            freeze_pixels(torch.full_like(q.float(),.1234567),mask)
        with self.assertRaises(ValueError):
            freeze_pixels(q.float()/255,mask+.1)

    def test_real_selection_rejects_each_regression(self):
        baseline = dict(iou=.2,visible_false_positive=.01,empty_mask_cases=2,negative_false_positive_cases=0)
        better = dict(baseline,iou=.3)
        self.assertTrue(real_gate(better,baseline,.2))
        self.assertFalse(real_gate(baseline,baseline,.2))
        for key in ('visible_false_positive','empty_mask_cases','negative_false_positive_cases'):
            worse = dict(better)
            worse[key] += 1
            self.assertFalse(real_gate(worse,baseline,.2))


if __name__=='__main__':
    unittest.main()
