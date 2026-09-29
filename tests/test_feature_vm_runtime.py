import unittest
from unittest.mock import patch
from feature_vm_runtime import require_cuda, balanced_batches

class VMRuntimeTests(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=False)
    def test_runner_refuses_cpu_before_reading_bundle(self, _):
        import train_feature_mixed_vm
        with patch('pathlib.Path.read_text', side_effect=AssertionError('read before GPU guard')):
            with self.assertRaisesRegex(RuntimeError, 'VM GPU'):
                train_feature_mixed_vm.main(preflight=True)

    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_refused(self, _):
        with self.assertRaisesRegex(RuntimeError, 'VM GPU'):require_cuda()

    @patch('torch.cuda.mem_get_info', return_value=(1024**3, 24*1024**3))
    @patch('torch.cuda.is_available', return_value=True)
    def test_low_free_vram_refused(self, *_):
        with self.assertRaisesRegex(RuntimeError, 'free'):require_cuda()

    def test_balanced_complete_reproducible_epoch(self):
        groups=[];start=0
        for n in [43,25,160,40,160,40]:
            groups.append(list(range(start,start+n)));start+=n
        batches=list(balanced_batches(groups,42))
        self.assertEqual(batches,list(balanced_batches(groups,42)))
        self.assertEqual(len(batches),80)
        self.assertEqual(set(sum(batches,[])),set(range(468)))
        for batch in batches:
            self.assertEqual(len(batch),12)
            for group in groups:self.assertEqual(sum(i in group for i in batch),2)
