"""Torch & CPU inference unit tests for Conditioned SFT V2 parity, gating, gradients, and NPZ serialization."""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from research import codeformer_conditioned_sft_inference_v2 as inf_v2
from research.codeformer_conditioned_sft_learning_v2 import (
    architecture_v2 as arch_v2,
    conditioning_v2 as cond_v2,
    objective_v2 as obj_v2,
)

ROOT = Path(__file__).resolve().parents[1]
TESTING_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
WEIGHTS_DIR = TESTING_ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights'
V1_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm'
B8_CKPT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/checkpoints/B_u0008/trainable.npz'


class TestConditionedSFTInferenceV2(unittest.TestCase):
    def test_01_trainable_bundle_npz_roundtrip_and_gradient_reachability(self):
        bundle = arch_v2.build_v2_trainable_modules(torch=torch)
        npz_raw = inf_v2.save_v2_trainable_npz_bytes(bundle, np=np)
        loaded = inf_v2.inspect_v2_trainable_npz(npz_raw, torch=torch, np=np)
        self.assertEqual(len(loaded), sum(1 for _ in bundle.parameters()))

        # Synthetic non-training gradient reachability test across pyramid, dark_refiner, and head_v2
        e_u96 = {
            s: torch.randn((2, arch_v2.CHANNELS[s], int(s), int(s)), dtype=torch.float32)
            for s in arch_v2.SITES
        }
        clean = {
            s: torch.randn((2, arch_v2.CHANNELS[s], int(s), int(s)), dtype=torch.float32)
            for s in arch_v2.SITES
        }
        site_maps = {
            s: torch.full((2, 1, int(s), int(s)), 0.8, dtype=torch.float32)
            for s in arch_v2.SITES
        }
        g_dark = torch.tensor([[[[0.0]]], [[[1.0]]]], dtype=torch.float32)
        g_sn = torch.tensor([[[[0.0]]], [[[0.8]]]], dtype=torch.float32)
        cond_info = {'site_maps': site_maps, 'g_sensor_noise': g_sn}

        # Perturb zero-init output weights slightly so upstream layers also receive non-zero backprop in 1 step
        with torch.no_grad():
            for name, p in bundle.named_parameters():
                if name.endswith(('out_proj.weight', 'freq_gate.weight')):
                    p.fill_(0.01)

        used = bundle.refine_sft_features(e_u96, cond_info)
        loss_sft, _ = obj_v2.sft_pyramid_distillation_loss_torch(used, clean, e_u96, g_dark, torch=torch)
        coarse32 = used['32'].detach()
        post_loss = 0.0
        for s in arch_v2.SITES:
            dec = torch.randn_like(used[s])
            fus = torch.randn_like(used[s])
            base_post = fus + 0.1 * dec
            out_s = bundle.refine_post_sft(s, base_post, used[s], dec, fus, coarse32, cond_info)
            post_loss = post_loss + (out_s - clean[s]).abs().mean()
        total = loss_sft + post_loss
        total.backward()

        for name, p in bundle.named_parameters():
            self.assertIsNotNone(p.grad, f'Missing gradient on {name}')
            self.assertTrue(torch.isfinite(p.grad).all().item(), f'Nonfinite gradient on {name}')
            self.assertGreater(float(p.grad.abs().max().item()), 0.0, f'Zero gradient on {name}')

    def test_02_torch_and_numpy_conditioning_parity(self):
        sample_png = V1_STAGE / 'data/inputs/tr_ffhq_08698__compound_g1p2_lr64_jpeg70.png'
        rgb = np.asarray(Image.open(sample_png), dtype=np.float32) / np.float32(255.0)
        c_np = cond_v2.compute_v2_conditioning_numpy(rgb, np=np)
        t = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]
        c_th = cond_v2.compute_v2_conditioning_torch(t, torch=torch)
        self.assertAlmostEqual(c_np['g_dark'], c_th['g_dark'], places=5)
        self.assertAlmostEqual(c_np['g_blur'], c_th['g_blur'], places=5)
        self.assertAlmostEqual(c_np['g_sensor_noise'], c_th['g_sensor_noise'], places=5)
        self.assertGreater(c_np['g_blur'], 0.5)

    def test_03_step0_real_u64_bitwise_parity(self):
        restorer = inf_v2.load_conditioned_sft_v2_restorer(
            WEIGHTS_DIR / 'codeformer_r2.pth',
            WEIGHTS_DIR / 'codeformer_u96_adapter.pth',
            V1_STAGE / 'weights/terminal_u0064_heads.npz',
            v1_b8_trainable_path=B8_CKPT,
        )
        self.assertEqual(restorer.v2_gamma, 0.0)
        sample_png = V1_STAGE / 'data/inputs/tr_ffhq_08698__global_gaussian2p0.png'
        rgb_u8 = np.asarray(Image.open(sample_png), dtype=np.uint8)
        comp = restorer.restore_compare_all(rgb_u8)
        np.testing.assert_array_equal(comp['v2_rgb8'], comp['u64_rgb8'])
        self.assertIsNotNone(comp['v1_b8_rgb8'])
        self.assertGreater(comp['sharpness']['u64'], comp['sharpness']['input'])


if __name__ == '__main__':
    unittest.main()
