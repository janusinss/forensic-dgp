"""VM boundary/cache counterexamples; no CUDA execution or model training."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


class VmBoundaryContracts(unittest.TestCase):
    def test_runner_rejects_local_execution_before_paths_or_model_creation(self):
        from scripts.train_native_expert_vm import main
        with patch('native_expert.sys.platform', 'win32'), \
                patch('native_expert.torch.cuda.is_available', return_value=True):
            # Empty arguments would fail on access if the guard were later.
            with self.assertRaisesRegex(RuntimeError, 'Linux VM'):
                main(argparse.Namespace())

    def test_fixture_support_and_byte_hashes_are_enforced(self):
        from scripts.train_native_expert_vm import verify_fixture
        case = {'input': torch.zeros(3, 256, 256, dtype=torch.uint8),
                'mask': torch.zeros(1, 256, 256, dtype=torch.uint8),
                'geometry': torch.zeros(1, 256, 256, dtype=torch.uint8),
                'valid': torch.ones(1, 256, 256, dtype=torch.uint8)}
        case['mask'][0, 0, 0] = case['geometry'][0, 0, 0] = 1
        row = {'pixel_sha256': {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                               for k, v in case.items()}, 'hole_pixels': 1,
               'geometry_pixels': 1, 'valid_pixels': 65536}
        verify_fixture(case, row)
        changed = {k: v.clone() for k, v in case.items()}; changed['valid'][0, 0, 0] = 0
        updated = {**row, 'valid_pixels': 65535, 'pixel_sha256': {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                                                             for k, v in changed.items()}}
        with self.assertRaises(ValueError): verify_fixture(changed, updated)
        changed = {k: v.clone() for k, v in case.items()}; changed['input'][0, 1, 1] = 1
        with self.assertRaises(ValueError): verify_fixture(changed, row)

    def test_inventory_names_cannot_escape_or_hide_windows_backslashes(self):
        from scripts.train_native_expert_vm import safe_path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(safe_path(root, 'inputs/one.png'), root / 'inputs/one.png')
            for name in ('../outside', '/absolute', 'C:/outside', 'inputs\\one.png', './one.png'):
                with self.subTest(name=name), self.assertRaises(ValueError): safe_path(root, name)


if __name__ == '__main__': unittest.main()
