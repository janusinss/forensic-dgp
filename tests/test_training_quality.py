import unittest
import numpy as np
import torch
from unittest.mock import patch
from dataset import DegradedFacesDataset
import dataloader
import evaluation
import importlib.util
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


class TrainingQualityTests(unittest.TestCase):
    def test_checkpoint_restores_optimizer_and_rejects_missing_file(self):
        self.assertIsNotNone(importlib.util.find_spec('training_state'), 'checkpoint helper is missing')
        from training_state import save_training_state, load_training_state
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pth'
            save_training_state(path, model, optimizer, scheduler, 27, 25.0, {'seed': 42})
            other = torch.nn.Linear(2, 1)
            opt = torch.optim.Adam(other.parameters())
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5)
            state = load_training_state(path, other, opt, sch)
            self.assertEqual(state['epoch'], 27)
            torch.testing.assert_close(other.weight, model.weight)
            self.assertEqual(opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
            self.assertEqual(sch.last_epoch, scheduler.last_epoch)
            with self.assertRaises(FileNotFoundError):
                load_training_state(Path(tmp) / 'missing.pth', other)

    def test_validation_uses_all_samples_and_eval_mode(self):
        self.assertTrue(hasattr(evaluation, 'validate_model'), 'validation loop is missing')
        class Model(torch.nn.Module):
            def forward(self, x):
                assert not self.training
                return x
        class Metric:
            def compute_metrics(self, rec, target):
                return {'PSNR': rec.mean().item(), 'SSIM': 0.5, 'ArcFace_Sim': None}
        x = torch.tensor([1., 2., 6.]).reshape(3, 1, 1, 1)
        loader = DataLoader(TensorDataset(x, x, x), batch_size=2)
        result = evaluation.validate_model(Model(), loader, Metric(), 'cpu')
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['PSNR'], 3.)
        self.assertIsNone(result['ArcFace_Sim'])

    def test_split_is_repeatable_and_disjoint(self):
        self.assertTrue(hasattr(dataloader, 'split_paths'), 'held-out split is missing')
        paths = [str(i) for i in range(100)]
        train, val = dataloader.split_paths(paths, 0.1, 42)
        self.assertEqual((train, val), dataloader.split_paths(paths[::-1], 0.1, 42))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(val), 10)

    def test_heavy_blur_has_sufficient_kernel_support(self):
        self.assertTrue(hasattr(DegradedFacesDataset, 'apply_primary_blur'), 'controlled blur is missing')
        ds = object.__new__(DegradedFacesDataset)
        ds.heavy_blur_probability = 1.0
        image = np.zeros((256, 256, 3), np.uint8)
        with patch('numpy.random.rand', side_effect=[0.0, 0.9]), patch('numpy.random.uniform', return_value=10.0), patch('cv2.GaussianBlur', return_value=image) as blur:
            ds.apply_primary_blur(image)
        self.assertGreaterEqual(blur.call_args.args[1][0], 61)

    def test_validation_repeats_without_changing_random_state(self):
        self.assertTrue(hasattr(DegradedFacesDataset, 'degrade_seeded'), 'repeatable degradation is missing')
        ds = object.__new__(DegradedFacesDataset)
        ds.seed, ds.epoch = 42, 0
        ds.apply_compound_degradation = lambda img: np.random.rand(4)
        np.random.seed(7)
        expected = np.random.rand()
        np.random.seed(7)
        a = ds.degrade_seeded(None, 2)
        self.assertEqual(np.random.rand(), expected)
        np.testing.assert_array_equal(a, ds.degrade_seeded(None, 2))
        ds.epoch = 1
        self.assertFalse(np.array_equal(a, ds.degrade_seeded(None, 2)))


if __name__ == '__main__':
    unittest.main()
