import importlib.util
import unittest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from dataset import DegradedFacesDataset


class Phase5Tests(unittest.TestCase):
    def test_fixed_validation_reports_source_groups_and_every_sample(self):
        import phase5_utils
        self.assertTrue(hasattr(phase5_utils, 'evaluate_identity'))
        from models.identity_loss import ArcFaceIdentityLoss, ARCFACE_TEMPLATE
        from torch.utils.data import DataLoader, TensorDataset
        loss_fn = ArcFaceIdentityLoss(encoder=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten()))
        images = torch.rand(2,3,112,112)
        lm = torch.zeros(2,68,2)
        for section, point in [(slice(36,42),0),(slice(42,48),1),(30,2),(48,3),(54,4)]:
            lm[:,section] = ARCFACE_TEMPLATE[point]
        loader = DataLoader(TensorDataset(images,images,lm,torch.arange(2)),batch_size=2)
        metrics = phase5_utils.evaluate_identity(torch.nn.Identity(), loader, loss_fn, 'cpu',
                  ['dataset/ffhq/a.png','dataset/asian_faces/b.png'])
        self.assertEqual(metrics['samples'],2)
        self.assertEqual(metrics['identity_pairs'],2)
        self.assertAlmostEqual(metrics['ArcFace_fixed'],1.,places=5)
        self.assertEqual(metrics['groups']['asian_faces']['samples'],1)

    def test_landmark_cache_reuses_results_and_invalidates_changed_pixels(self):
        self.assertTrue(hasattr(DegradedFacesDataset, 'cached_landmarks'))
        with tempfile.TemporaryDirectory(dir='outputs') as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root/'face.png'), np.zeros((32,32,3),np.uint8))
            ds = DegradedFacesDataset(str(root), landmark_cache_dir=str(root/'cache'))
            pixels = np.zeros((256,256,3),np.uint8)
            with patch.object(ds, '_get_landmarks', return_value=np.ones((68,2),np.float32)) as detector:
                ds.cached_landmarks(pixels)
                ds.cached_landmarks(pixels)
                self.assertEqual(detector.call_count, 1)
                ds.cached_landmarks(pixels+1)
                self.assertEqual(detector.call_count, 2)

    def test_missing_dataset_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            DegradedFacesDataset('dataset/__missing_phase5_test__')

    def test_identity_loss_backpropagates_but_freezes_recognizer(self):
        self.assertIsNotNone(importlib.util.find_spec('models.identity_loss'))
        from models.identity_loss import ArcFaceIdentityLoss, ARCFACE_TEMPLATE
        encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        loss_fn = ArcFaceIdentityLoss(encoder=encoder)
        landmarks = torch.zeros(1, 68, 2)
        landmarks[:,36:42] = ARCFACE_TEMPLATE[0]
        landmarks[:,42:48] = ARCFACE_TEMPLATE[1]
        landmarks[:,30] = ARCFACE_TEMPLATE[2]
        landmarks[:,48] = ARCFACE_TEMPLATE[3]
        landmarks[:,54] = ARCFACE_TEMPLATE[4]
        image = torch.rand(1,3,112,112,requires_grad=True)
        target = torch.rand_like(image)
        loss, valid = loss_fn(image, target, landmarks)
        loss.backward()
        self.assertEqual(valid, 1)
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
        grid, mask = loss_fn.alignment_grid(landmarks, 112,112)
        aligned = torch.nn.functional.grid_sample(image, grid, align_corners=False)
        torch.testing.assert_close(aligned, image, atol=1e-4, rtol=1e-4)
        zero, count = loss_fn(image, target, torch.zeros_like(landmarks))
        self.assertEqual(count, 0)
        self.assertEqual(zero.item(), 0)

    def test_selection_rejects_identity_regression_despite_psnr_gain(self):
        self.assertIsNotNone(importlib.util.find_spec('phase5_utils'))
        from phase5_utils import qualifies
        baseline = {'PSNR':20., 'SSIM':.66, 'ArcFace_fixed':.35, 'identity_pairs':100}
        candidate = dict(baseline, PSNR=21., ArcFace_fixed=.34)
        self.assertFalse(qualifies(candidate, baseline, baseline))
        candidate['ArcFace_fixed'] = .36
        self.assertTrue(qualifies(candidate, baseline, baseline))
        candidate['identity_pairs'] = 99
        self.assertFalse(qualifies(candidate, baseline, baseline))
        candidate['ArcFace_fixed'] = None
        self.assertFalse(qualifies(candidate, baseline, baseline))

    def test_ema_does_not_double_update_shared_weights(self):
        self.assertIsNotNone(importlib.util.find_spec('phase5_utils'))
        from phase5_utils import ModelEMA
        model = torch.nn.Module()
        model.a = torch.nn.Linear(1,1,bias=False)
        model.b = model.a
        model.a.weight.data.zero_()
        ema = ModelEMA(model, decay=.5)
        model.a.weight.data.fill_(2)
        ema.update(model)
        self.assertEqual(ema.model.a.weight.item(), 1.)
        self.assertFalse(ema.model.a.weight.requires_grad)


if __name__ == '__main__':
    unittest.main()
