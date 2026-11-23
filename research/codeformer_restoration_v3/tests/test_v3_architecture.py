"""Unit tests for CodeFormer Restoration V3 architecture and 12-regime CCTV degradations."""
from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from research.codeformer_restoration_v3.architecture_v3 import (
    _WMAP_FLOORS_V33,
    BASE_TENSOR_COUNT,
    SITES,
    CodeFormerV3Restorer,
    _hp5,
    delivered_views_v3,
    detect_v3_npz_variant,
    export_v3_npz,
    inspect_v3_npz,
    load_v3_modules_npz,
    load_v3_npz,
    replicate_pad_observed,
    v3_shapes,
    v3_state_sha,
)
from research.codeformer_restoration_v3.degradations_v3 import (
    REGIMES,
    REGIME_WEIGHTS,
    generate_cctv_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CF_WEIGHTS = REPO_ROOT / "outputs" / "codeformer_restoration_pretrained_v1" / "codeformer.pth"


def _synthetic_face_512(seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32) / 511.0
    r = 0.55 + 0.25 * np.sin(6.0 * xx) * np.cos(4.0 * yy)
    g = 0.45 + 0.20 * np.cos(5.0 * xx + yy)
    b = 0.40 + 0.20 * np.sin(3.0 * yy - 2.0 * xx)
    img = np.stack([r, g, b], axis=-1)
    img += rng.normal(0.0, 0.02, img.shape).astype(np.float32)
    return np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)


class TestV3Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        cls.model = CodeFormerV3Restorer()

    def test_01_param_budget_and_frozen_base(self):
        base_sd = self.model.base.state_dict()
        self.assertEqual(len(base_sd), BASE_TENSOR_COUNT)
        base_trainable = sum(p.numel() for p in self.model.base.parameters() if p.requires_grad)
        self.assertEqual(base_trainable, 0)

        trainable = self.model.trainable_parameter_count()
        breakdown = self.model.module_param_breakdown()
        print(f"\n[V3 Params] base={self.model.base_parameter_count():,} (frozen 515 tensors), trainable={trainable:,}")
        print(f"[V3 Breakdown] {breakdown}")
        self.assertGreaterEqual(trainable, 2_500_000)
        self.assertLessEqual(trainable, 3_800_000)

        # Ensure .train(True) keeps self.base strictly in eval() and frozen
        self.model.train(True)
        self.assertFalse(self.model.base.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.base.parameters()))
        self.model.eval()

    def test_02_replicate_pad_observed(self):
        x = torch.full((1, 3, 256, 256), 128.0 / 255.0)
        obs = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        obs[:, :, 32:224, 48:208] = 1.0
        # Distinct border values inside observed region
        x[:, :, 32:224, 48:208] = 0.25
        x[:, :, 32, 48:208] = 0.90  # top border
        x[:, :, 223, 48:208] = 0.10  # bottom border

        padded = replicate_pad_observed(x, obs)
        # Observed pixels unchanged
        self.assertTrue(torch.allclose(padded[:, :, 32:224, 48:208], x[:, :, 32:224, 48:208]))
        # Top bar replicated from row 32 (0.90)
        self.assertTrue(torch.allclose(padded[:, :, 0:32, 48:208], torch.full((1, 3, 32, 160), 0.90)))
        # Bottom bar replicated from row 223 (0.10)
        self.assertTrue(torch.allclose(padded[:, :, 224:256, 48:208], torch.full((1, 3, 32, 160), 0.10)))

    def test_03_step0_identity_properties(self):
        self.model.eval()
        torch.manual_seed(7)
        x = torch.rand((1, 3, 256, 256))
        # Include exact 0.0 and 1.0 boundary pixels
        x[0, :, 0, 0] = 0.0
        x[0, :, 0, 1] = 1.0
        obs = torch.ones((1, 1, 256, 256))

        with torch.no_grad():
            deg_out = self.model.v3["deg_encoder"](x, obs)
            x_photo, _ = self.model.v3["photo_net"](x, deg_out["d_maps"]["32"], deg_out["z_deg"])
            max_photo_diff = float((x_photo - x).abs().max().item())
            self.assertLess(max_photo_diff, 1e-5, f"PhotoCorrectNet not identity at step 0: {max_photo_diff}")

            # DegGatedTransformerLoRA exact parity with untouched frozen ft_layers at step 0
            q_in = torch.randn((256, 1, 512))
            pos = self.model.base.position_emb.unsqueeze(1)
            ref = q_in
            for layer in self.model.base.ft_layers:
                ref = layer(ref, query_pos=pos)
            v3_q = self.model.v3["transformer_lora"](
                q_in, pos, self.model.base.ft_layers,
                deg_out["z_deg"], deg_out["d_maps"]["16"], deg_out["lora_gate"],
            )
            max_lora_diff = float((v3_q - ref).abs().max().item())
            self.assertLess(max_lora_diff, 1e-4, f"LoRA step-0 delta not zero: {max_lora_diff}")

            # V3PyramidSFT zero-init residual
            enc_dummy = {s: torch.randn((1, ch, int(s), int(s))) for s, ch in [("32", 256), ("64", 256), ("128", 128), ("256", 128)]}
            enc_used = self.model.v3["pyramid_sft"](enc_dummy, deg_out["d_maps"])
            for s in SITES:
                self.assertEqual(float((enc_used[s] - enc_dummy[s]).abs().max().item()), 0.0)

            # V3FreqHead zero-init residual
            for s in SITES:
                dec_s = torch.randn_like(enc_dummy[s])
                fus_s = torch.randn_like(enc_dummy[s])
                out_s = self.model.v3["freq_head"].forward_site(
                    s, enc_dummy[s], dec_s, fus_s, enc_dummy["32"], deg_out["d_maps"][s]
                )
                self.assertEqual(float((out_s - fus_s).abs().max().item()), 0.0)

    def test_04_finite_grads_fp32_and_amp(self):
        for use_amp in (False, True):
            model = CodeFormerV3Restorer()
            model.train(True)
            x = torch.rand((1, 3, 256, 256))
            obs = torch.ones((1, 1, 256, 256))
            obs[:, :, :16, :] = 0.0

            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
                out = model(x, obs)
                loss = (
                    out["y_256"].mean()
                    + 0.2 * out["y_512"].mean()
                    + 0.1 * out["x_photo_256"].mean()
                    + 0.1 * out["code_logits"].mean()
                    + 0.1 * out["sufficiency"].mean()
                    + 0.1 * sum(v.mean() for v in out["deg_preds"].values())
                    + 0.1 * out["alpha_bypass"].mean()
                    + 0.1 * out["lora_gate"].mean()
                    + 0.1 * out["alpha_adain"].mean()
                    + 0.1 * out["g_photo"].mean()
                    + 0.1 * sum(v.mean() for v in out["enc_used"].values())
                )
            loss.backward()

            # Zero gradients on frozen base
            for name, p in model.base.named_parameters():
                self.assertIsNone(p.grad, f"Frozen base param {name} unexpectedly received grad")

            # Every trainable submodule in model.v3 must have finite, non-zero grad norm
            for mod_name, mod in model.v3.items():
                total_norm = 0.0
                for pname, p in mod.named_parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all().item(), f"Non-finite grad in {mod_name}.{pname} (amp={use_amp})")
                        total_norm += float(p.grad.float().abs().sum().item())
                self.assertGreater(total_norm, 0.0, f"Zero total grad in module {mod_name} (amp={use_amp})")

            # Stage-1 only fast path + mask-aware encode_gt_teacher check
            model.zero_grad(set_to_none=True)
            gt_512 = torch.rand((1, 3, 512, 512))
            teacher_out = model.encode_gt_teacher(gt_512, observed_mask=obs)
            self.assertEqual(tuple(teacher_out["code_indices_gt"].shape), (1, 256))
            out_s1 = model(x, obs, stage1_only=True)
            self.assertIsNone(out_s1["y_256"])
            s1_loss = out_s1["x_photo_256"].mean() + out_s1["code_logits"].mean() + sum(v.mean() for v in out_s1["enc_used"].values())
            s1_loss.backward()
            self.assertTrue(torch.isfinite(s1_loss).item())

        # Gradient checkpointing mode backward sanity check
        ckpt_model = CodeFormerV3Restorer(grad_checkpoint=True)
        ckpt_model.train(True)
        out_c = ckpt_model(torch.rand((1, 3, 256, 256)), torch.ones((1, 1, 256, 256)))
        loss_c = out_c["y_256"].mean()
        loss_c.backward()
        self.assertTrue(torch.isfinite(loss_c).item())

    def test_05_npz_export_load_bit_exact_parity(self):
        m1 = CodeFormerV3Restorer()
        # Perturb weights slightly so zero-init tensors are non-trivial
        with torch.no_grad():
            for p in m1.v3.parameters():
                p.add_(torch.randn_like(p) * 1e-3)
        m1.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "v3_modules.npz"
            sha1 = export_v3_npz(m1, npz_path)
            raw = npz_path.read_bytes()
            inspected = inspect_v3_npz(raw, v3_shapes(m1))
            self.assertEqual(v3_state_sha(inspected), sha1)

            m2 = CodeFormerV3Restorer()
            m2.base.load_state_dict(m1.base.state_dict(), strict=True)
            load_v3_npz(npz_path, m2)
            m2.eval()
            self.assertEqual(v3_state_sha(m2.v3.state_dict()), sha1)

            # Bit-exact forward parity
            canvas = _synthetic_face_512(99)[:256, :256]
            obs = np.ones((256, 256), dtype=bool)
            out1 = m1.restore_canvas(canvas, obs)
            out2 = m2.restore_canvas(canvas, obs)
            self.assertTrue(np.array_equal(out1, out2), "NPZ reload did not produce bit-exact forward parity")

    def test_06_unobserved_bit_exact_preservation(self):
        canvas = np.full((256, 256, 3), 128, dtype=np.uint8)
        obs = np.zeros((256, 256), dtype=bool)
        obs[20:236, 40:216] = True
        canvas[obs] = _synthetic_face_512(42)[:256, :256][obs]

        raw_f32, delivered_u8 = self.model.restore_canvas(canvas, obs, return_delivered=True)
        self.assertEqual(raw_f32.shape, (256, 256, 3))
        self.assertEqual(raw_f32.dtype, np.float32)
        self.assertEqual(delivered_u8.shape, (256, 256, 3))
        self.assertEqual(delivered_u8.dtype, np.uint8)
        self.assertTrue(np.array_equal(delivered_u8[~obs], canvas[~obs]))
        expected_unobs_f32 = canvas[~obs].astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(raw_f32[~obs], expected_unobs_f32))

        # Also test rint rounding mode
        _, delivered_rint = delivered_views_v3(raw_f32, canvas, obs, rounding="rint")
        self.assertTrue(np.array_equal(delivered_rint[~obs], canvas[~obs]))

    def test_07_all_12_degradation_regimes_and_geometry(self):
        self.assertEqual(len(REGIMES), 12)
        self.assertAlmostEqual(sum(REGIME_WEIGHTS.values()), 1.0, places=6)
        gt = _synthetic_face_512(2026)

        for idx, regime in enumerate(REGIMES):
            inp1, gt256_1, gt512_1, obs1, t1 = generate_cctv_pair(
                gt, regime=regime, seed=1000 + idx, apply_geometry=(idx % 2 == 0)
            )
            inp2, gt256_2, gt512_2, obs2, t2 = generate_cctv_pair(
                gt, regime=regime, seed=1000 + idx, apply_geometry=(idx % 2 == 0)
            )
            # Determinism
            self.assertTrue(np.array_equal(inp1, inp2), f"Non-deterministic inp for {regime}")
            self.assertTrue(np.array_equal(gt256_1, gt256_2))
            self.assertTrue(np.array_equal(obs1, obs2))
            self.assertEqual(t1, t2)

            # Shapes & dtypes
            self.assertEqual(inp1.shape, (256, 256, 3))
            self.assertEqual(gt256_1.shape, (256, 256, 3))
            self.assertEqual(gt512_1.shape, (512, 512, 3))
            self.assertEqual(obs1.shape, (256, 256))
            self.assertEqual(inp1.dtype, np.uint8)
            self.assertEqual(obs1.dtype, bool)
            self.assertEqual(t1["regime"], regime)
            for k in ("exposure", "blur", "noise", "codec_lr", "severity", "sufficiency"):
                self.assertGreaterEqual(t1[k], 0.0)
                self.assertLessEqual(t1[k], 1.0)

        # Explicit geometry test: ensure non-square padding produces ~observed bars with (128,128,128)
        found_bars = False
        for s in range(10):
            inp_g, gt256_g, _, obs_g, tg = generate_cctv_pair(gt, regime="clear", seed=500 + s, apply_geometry=True)
            self.assertTrue(tg["geometry_applied"])
            if not obs_g.all():
                found_bars = True
                # Corners outside observed mask must be exact neutral (128, 128, 128)
                self.assertTrue((inp_g[0, 0] == 128).all() or (inp_g[0, 255] == 128).all() or (inp_g[255, 0] == 128).all())
                break
        self.assertTrue(found_bars, "Expected non-square prepare_crop geometry to produce ~observed bars")

        # Regression guard: small sigma_min/sigma_max & mild_transition across 100 seeds (no high < low inversion)
        from research.codeformer_restoration_v3.degradations_v3 import _gaussian_or_disk_blur

        rng_blur = np.random.default_rng(777)
        patch_f = np.full((64, 64, 3), 0.5, dtype=np.float32)
        for _ in range(500):
            _gaussian_or_disk_blur(patch_f, rng_blur, 0.20, 0.45)
            _gaussian_or_disk_blur(patch_f, rng_blur, 0.35, 0.85)
        for s in range(50):
            generate_cctv_pair(gt, regime="mild_transition", seed=9000 + s, apply_geometry=(s % 2 == 0))

    @unittest.skipUnless(OFFICIAL_CF_WEIGHTS.is_file(), "Official codeformer.pth not present")
    def test_08_real_codeformer_cpu_latency(self):
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            model = CodeFormerV3Restorer(OFFICIAL_CF_WEIGHTS)
            model.eval()
            canvas = _synthetic_face_512(77)[:256, :256]
            obs = np.ones((256, 256), dtype=bool)
            inp_512 = torch.nn.functional.interpolate(
                torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float()[None] / 255.0,
                size=(512, 512), mode="bilinear", align_corners=False,
            ) * 2.0 - 1.0

            # Warmup & time untouched official CodeFormer base
            with torch.no_grad():
                _ = model.base(inp_512, w=1.0, detach_16=True, adain=True)
                t0_cf = time.perf_counter()
                _ = model.base(inp_512, w=1.0, detach_16=True, adain=True)
                cf_elapsed = time.perf_counter() - t0_cf

            # Warmup & time V3 single pass
            _ = model.restore_canvas(canvas, obs)
            t0 = time.perf_counter()
            raw_f32 = model.restore_canvas(canvas, obs)
            v3_elapsed = time.perf_counter() - t0
            ratio = v3_elapsed / max(cf_elapsed, 1e-6)
            print(
                f"\n[V3 CPU Timing] base_cf={cf_elapsed:.3f}s, v3={v3_elapsed:.3f}s, "
                f"overhead_ratio={ratio:.3f}x (4 threads FP32)"
            )
            self.assertEqual(raw_f32.shape, (256, 256, 3))
            self.assertTrue(np.isfinite(raw_f32).all())
            self.assertLessEqual(v3_elapsed, 10.0, f"Single-pass CPU latency too slow: {v3_elapsed:.3f}s")
        finally:
            torch.set_num_threads(prev_threads)

    def test_09_v33_clean_architecture_and_npz_autodetect(self):
        torch.manual_seed(33)
        m33 = CodeFormerV3Restorer(arch_variant="v33")
        m33.eval()

        # 1. Zero-LoRA param count & frozen 515-tensor base
        self.assertNotIn("transformer_lora", m33.v3)
        self.assertEqual(len(m33.base.state_dict()), BASE_TENSOR_COUNT)
        self.assertEqual(sum(p.numel() for p in m33.base.parameters() if p.requires_grad), 0)
        trainable = m33.trainable_parameter_count()
        self.assertLessEqual(trainable, 1_600_000)
        self.assertGreaterEqual(trainable, 1_000_000)
        self.assertEqual(trainable, 1_209_411)
        self.assertEqual(len(m33.v3.state_dict()), 323)

        # 2. V3FreqHead hp5_only: zero spatial residual params & strict non-negative VQ detail transfer
        for site in SITES:
            fsite = m33.v3["freq_head"].sites[site]
            self.assertFalse(hasattr(fsite, "out_proj"))
            self.assertFalse(hasattr(fsite, "enc_p"))
            ch = {"32": 256, "64": 256, "128": 128, "256": 128}[site]
            dec_s = torch.randn((1, ch, int(site), int(site)))
            fus_s = torch.randn((1, ch, int(site), int(site)))
            enc_s = torch.randn((1, ch, int(site), int(site)))
            d_s = torch.randn((1, {"32": 96, "64": 64, "128": 48, "256": 32}[site], int(site), int(site)))
            with torch.no_grad():
                out_s, gate_s = m33.v3["freq_head"].forward_site(site, enc_s, dec_s, fus_s, enc_s, d_s, return_gate=True)
            self.assertTrue((gate_s > 0.0).all().item())
            self.assertTrue((gate_s < 0.75).all().item())
            expected_out = fus_s + gate_s * _hp5(dec_s - fus_s)
            self.assertTrue(torch.allclose(out_s, expected_out, atol=1e-6))

        # 3. V3PyramidSFT wmap_gated: gate_sft == 0.0 EXACTLY when w_maps <= 0.55 + soft norm cap <= 0.35 * rms(enc)
        enc_dummy = {s: torch.randn((2, ch, int(s), int(s))) for s, ch in [("32", 256), ("64", 256), ("128", 128), ("256", 128)]}
        d_dummy = {s: torch.randn((2, ch, int(s), int(s))) for s, ch in [("32", 96), ("64", 64), ("128", 48), ("256", 32)]}
        with torch.no_grad():
            for s in SITES:
                m33.v3["pyramid_sft"].stages[s].out_proj.weight.fill_(2.5)
                m33.v3["pyramid_sft"].stages[s].out_proj.bias.fill_(1.0)
            w_low = {s: torch.full((2, 1, int(s), int(s)), _WMAP_FLOORS_V33[s]) for s in SITES}
            w_low["32"].fill_(0.55)  # exact cutoff boundary
            enc_low = m33.v3["pyramid_sft"](enc_dummy, d_dummy, w_maps=w_low)
            for s in SITES:
                self.assertEqual(float((enc_low[s] - enc_dummy[s]).abs().max().item()), 0.0)

            w_hi = {s: torch.full((2, 1, int(s), int(s)), 0.95) for s in SITES}
            enc_hi = m33.v3["pyramid_sft"](enc_dummy, d_dummy, w_maps=w_hi)
            for s in SITES:
                delta_rms = (enc_hi[s] - enc_dummy[s]).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
                enc_rms = enc_dummy[s].float().pow(2).mean(dim=(1, 2, 3)).sqrt()
                self.assertTrue((delta_rms <= 0.35 * enc_rms + 1e-5).all().item())

        # 4. Calibrated w_maps floors & PhotoCorrectNet bright low-sufficiency damping
        with torch.no_grad():
            x_bright = torch.full((1, 3, 256, 256), 0.75)
            x_dark = torch.full((1, 3, 256, 256), 0.15)
            obs = torch.ones((1, 1, 256, 256))
            d32 = torch.zeros((1, 96, 32, 32))
            z = torch.zeros((1, 128))
            suff_low = torch.tensor([[0.20]])
            x_p0, g_bright = m33.v3["photo_net"](x_bright, d32, z, observed_mask=obs, sufficiency=suff_low)
            _, g_dark = m33.v3["photo_net"](x_dark, d32, z, observed_mask=obs, sufficiency=suff_low)
            self.assertLess(float((x_p0 - x_bright).abs().max().item()), 1e-5)
            self.assertLess(float(g_bright.item()), float(g_dark.item()) * 0.5)

            for s in SITES:
                m33.v3["deg_encoder"].w_heads[s][-1].bias.fill_(-50.0)
            deg_out = m33.v3["deg_encoder"](x_dark, obs)
            for s in SITES:
                self.assertGreaterEqual(float(deg_out["w_maps"][s].min().item()), _WMAP_FLOORS_V33[s] - 1e-6)

        # 5. NPZ auto-detect round-trip for BOTH v30 and v33 (zero-pickle allow_pickle=False)
        m30 = CodeFormerV3Restorer(arch_variant="v30")
        m30.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            p30 = Path(tmpdir) / "v30.npz"
            p33 = Path(tmpdir) / "v33.npz"
            sha30 = export_v3_npz(m30, p30)
            sha33 = export_v3_npz(m33, p33)
            self.assertEqual(detect_v3_npz_variant(p30), "v30")
            self.assertEqual(detect_v3_npz_variant(p33), "v33")
            self.assertEqual(v3_state_sha(inspect_v3_npz(p30.read_bytes())), sha30)
            self.assertEqual(v3_state_sha(inspect_v3_npz(p33.read_bytes())), sha33)

            m33_reload = CodeFormerV3Restorer(arch_variant="v33")
            m33_reload.base.load_state_dict(m33.base.state_dict(), strict=True)
            load_v3_modules_npz(p33, m33_reload)
            m33_reload.eval()
            self.assertEqual(v3_state_sha(m33_reload.v3.state_dict()), sha33)
            canvas = _synthetic_face_512(333)[:256, :256]
            obs_np = np.ones((256, 256), dtype=bool)
            self.assertTrue(np.array_equal(m33.restore_canvas(canvas, obs_np), m33_reload.restore_canvas(canvas, obs_np)))

    def test_10_v34_preconditioned_architecture_and_npz_autodetect(self):
        torch.manual_seed(34)
        m34 = CodeFormerV3Restorer(arch_variant="v34")
        m34.eval()

        # 1. Exact 3 front-end modules ONLY (ZERO transformer_lora, pyramid_sft, freq_head, output_composer)
        self.assertEqual(set(m34.v3.keys()), {"deg_encoder", "photo_net", "pre_net"})
        for forbidden in ("transformer_lora", "pyramid_sft", "freq_head", "output_composer"):
            self.assertNotIn(forbidden, m34.v3)
        self.assertEqual(len(m34.base.state_dict()), BASE_TENSOR_COUNT)
        self.assertEqual(sum(p.numel() for p in m34.base.parameters() if p.requires_grad), 0)

        trainable = m34.trainable_parameter_count()
        breakdown = m34.module_param_breakdown()
        print(f"\n[V3.4 Params] trainable={trainable:,}, tensors={len(m34.v3.state_dict())}, breakdown={breakdown}")
        self.assertEqual(trainable, 2_241_395)
        self.assertEqual(len(m34.v3.state_dict()), 344)

        # .train(True) keeps frozen base in eval()
        m34.train(True)
        self.assertFalse(m34.base.training)
        self.assertTrue(all(not p.requires_grad for p in m34.base.parameters()))
        m34.eval()

        # 2. Exact step-0 identity properties + highlight pinning f(1) == 1 in FP32
        x = torch.rand((2, 3, 256, 256))
        x[:, :, 0, 0] = 0.0
        x[:, :, 0, 1] = 1.0
        obs = torch.ones((2, 1, 256, 256))
        with torch.no_grad():
            deg_out = m34.v3["deg_encoder"](x, obs)
            self.assertAlmostEqual(float(deg_out["w_pred"].mean().item()), 0.82, places=4)
            self.assertTrue(((deg_out["w_pred"] >= 0.60) & (deg_out["w_pred"] <= 0.98)).all().item())
            self.assertAlmostEqual(float(deg_out["g_photo"].mean().item()), float(torch.sigmoid(torch.tensor(-2.0)).item()), places=5)
            self.assertAlmostEqual(float(deg_out["g_pre"].mean().item()), float(torch.sigmoid(torch.tensor(-2.0)).item()), places=5)

            x_photo, params_4x4 = m34.v3["photo_net"](x, deg_out["f4"], deg_out["z_deg"], deg_out["g_photo"])
            self.assertEqual(tuple(params_4x4.shape), (2, 5, 4, 4))
            self.assertLess(float((x_photo - x).abs().max().item()), 1e-4, "V34PhotoCorrectNet not identity at step 0")

            x_pre = m34.v3["pre_net"](x_photo, obs, deg_out["z_deg"], deg_out["g_pre"])
            self.assertEqual(float((x_pre - x_photo).abs().max().item()), 0.0, "V34NAFPreConditionNet not identity at step 0")

        # Negative-bias b_haze gradient test (verifies zero Dead-ReLU trap) + bf16 FP32 enforcement
        from research.codeformer_restoration_v3.architecture_v3 import _compute_v34_hf_channels

        m_haze = CodeFormerV3Restorer(arch_variant="v34")
        m_haze.train(True)
        with torch.no_grad():
            m_haze.v3["photo_net"].param_head.bias[:3].fill_(-0.05)
        m_haze.zero_grad(set_to_none=True)
        x_hazy = torch.full((1, 3, 256, 256), 0.70)
        f4_h = torch.randn((1, 112, 4, 4))
        z_h = torch.randn((1, 128))
        g_h = torch.ones((1, 1))
        x_dh, _ = m_haze.v3["photo_net"](x_hazy, f4_h, z_h, g_h)
        (x_dh - 0.30).abs().mean().backward()
        b_bias_grad = m_haze.v3["photo_net"].param_head.bias.grad[:3]
        self.assertTrue(torch.isfinite(b_bias_grad).all().item())
        self.assertGreater(float(b_bias_grad.abs().sum().item()), 0.0, "b_haze died under negative bias=-0.05")

        # FP32 enforcement under bf16 autocast for _compute_v34_hf_channels & V34DegradationEncoder scalar heads
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            l_hf, c1_hf, cc_hf = _compute_v34_hf_channels(x[:1].to(torch.bfloat16))
            self.assertEqual(l_hf.dtype, torch.float32)
            self.assertEqual(c1_hf.dtype, torch.float32)
            self.assertEqual(cc_hf.dtype, torch.float32)
            d_bf16 = m34.v3["deg_encoder"](x[:1].to(torch.bfloat16), obs[:1].to(torch.bfloat16))
            self.assertEqual(d_bf16["w_pred"].dtype, torch.float32)
            self.assertEqual(d_bf16["p_clean"].dtype, torch.float32)
            self.assertEqual(d_bf16["alpha_clean"].dtype, torch.float32)
            self.assertEqual(d_bf16["g_photo"].dtype, torch.float32)
            self.assertEqual(d_bf16["g_pre"].dtype, torch.float32)

        # Highlight pinning f(1) == 1.0 in FP32 even with strong non-zero dehaze/tone/gamma weights
        m_pin = CodeFormerV3Restorer(arch_variant="v34")
        m_pin.eval()
        with torch.no_grad():
            m_pin.v3["photo_net"].param_head.weight.fill_(0.8)
            m_pin.v3["photo_net"].param_head.bias.copy_(torch.tensor([0.5, 0.7, 0.9, 2.5, -1.2]))
            m_pin.v3["photo_net"].wb_head.weight.zero_()
            m_pin.v3["photo_net"].wb_head.bias.zero_()
            x_white = torch.ones((1, 3, 256, 256))
            f4_rand = torch.randn((1, 112, 4, 4))
            z_rand = torch.randn((1, 128))
            g_one = torch.ones((1, 1))
            x_out_w, _ = m_pin.v3["photo_net"](x_white, f4_rand, z_rand, g_one)
            self.assertLess(float((x_out_w - 1.0).abs().max().item()), 1e-5, "V34PhotoCorrectNet violated f(1)==1 highlight pin")
            # Also verify dehaze never brightens midtones when tone=1, gamma=1
            m_pin.v3["photo_net"].param_head.weight.zero_()
            m_pin.v3["photo_net"].param_head.bias.copy_(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
            x_mid = torch.full((1, 3, 256, 256), 0.6)
            x_dehazed, _ = m_pin.v3["photo_net"](x_mid, f4_rand, z_rand, g_one)
            self.assertLessEqual(float(x_dehazed.max().item()), 0.6 + 1e-6)

        # 3. Hard-deadzone alpha_clean == 0.0 strictly in eval() when p_clean <= 0.72
        with torch.no_grad():
            m34.eval()
            # Force p_exp=p_blur=p_noise=p_grid ~= 0 so p_clean ~= sigmoid(clean_head)
            m34.v3["deg_encoder"].deg_head[-1].weight.zero_()
            m34.v3["deg_encoder"].deg_head[-1].bias.fill_(-20.0)
            m34.v3["deg_encoder"].clean_head[-1].weight.zero_()
            # sigmoid(0.90) ~= 0.7109 <= 0.72 -> alpha_clean MUST be strictly 0.0000
            m34.v3["deg_encoder"].clean_head[-1].bias.fill_(0.90)
            d_low = m34.v3["deg_encoder"](x[:1], obs[:1])
            self.assertLessEqual(float(d_low["p_clean"].item()), 0.72)
            self.assertEqual(float(d_low["alpha_clean"].item()), 0.0)
            # sigmoid(4.0) ~= 0.982 >= 0.92 -> alpha_clean MUST be strictly 1.0
            m34.v3["deg_encoder"].clean_head[-1].bias.fill_(4.0)
            d_hi = m34.v3["deg_encoder"](x[:1], obs[:1])
            self.assertGreaterEqual(float(d_hi["p_clean"].item()), 0.92)
            self.assertEqual(float(d_hi["alpha_clean"].item()), 1.0)

        # 4. z_q_ste bit-exact forward parity + non-zero finite gradients into pre_net & photo_net
        m_ste = CodeFormerV3Restorer(arch_variant="v34")
        x1 = torch.rand((1, 3, 256, 256))
        obs1 = torch.ones((1, 1, 256, 256))
        m_ste.eval()
        with torch.no_grad():
            out_eval = m_ste(x1, obs1, skip_decoder=False)
        m_ste.train(True)
        out_train = m_ste(x1, obs1, skip_decoder=False)
        # dec_256 (before train vs eval alpha_clean curve) must be bit-exact between STE train and argmax eval!
        self.assertTrue(
            torch.equal(out_train["dec_256"].detach(), out_eval["dec_256"]),
            "V3.4 z_q_ste forward dec_256 is not bit-exact with eval argmax",
        )
        loss_ste = out_train["y_256"].mean() + 0.1 * out_train["p_clean"].mean()
        loss_ste.backward()
        for name, p in m_ste.base.named_parameters():
            self.assertIsNone(p.grad, f"Frozen base param {name} unexpectedly received grad in V3.4")
        for mod_name in ("deg_encoder", "photo_net", "pre_net"):
            gnorm = 0.0
            for pname, p in m_ste.v3[mod_name].named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all().item(), f"Non-finite grad in V3.4 {mod_name}.{pname}")
                    gnorm += float(p.grad.float().abs().sum().item())
            self.assertGreater(gnorm, 0.0, f"Zero total grad in V3.4 {mod_name}")

        # 5. skip_decoder=True fast path
        m_ste.zero_grad(set_to_none=True)
        out_skip = m_ste(x1, obs1, skip_decoder=True)
        self.assertIsNone(out_skip["y_256"])
        self.assertEqual(tuple(out_skip["code_logits"].shape), (1, 256, 1024))
        self.assertEqual(tuple(out_skip["x_pre_256"].shape), (1, 3, 256, 256))

        # 6. Flat zero-pickle .npz auto-detect round-trip for v30, v33, AND v34
        m30 = CodeFormerV3Restorer(arch_variant="v30").eval()
        m33 = CodeFormerV3Restorer(arch_variant="v33").eval()
        m34_fresh = CodeFormerV3Restorer(arch_variant="v34").eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            p30 = Path(tmpdir) / "v30.npz"
            p33 = Path(tmpdir) / "v33.npz"
            p34 = Path(tmpdir) / "v34.npz"
            sha30 = export_v3_npz(m30, p30)
            sha33 = export_v3_npz(m33, p33)
            sha34 = export_v3_npz(m34_fresh, p34)
            self.assertEqual(detect_v3_npz_variant(p30), "v30")
            self.assertEqual(detect_v3_npz_variant(p33), "v33")
            self.assertEqual(detect_v3_npz_variant(p34), "v34")
            self.assertEqual(v3_state_sha(inspect_v3_npz(p30.read_bytes())), sha30)
            self.assertEqual(v3_state_sha(inspect_v3_npz(p33.read_bytes())), sha33)
            self.assertEqual(v3_state_sha(inspect_v3_npz(p34.read_bytes())), sha34)

            m34_reload = CodeFormerV3Restorer(arch_variant="v34")
            m34_reload.base.load_state_dict(m34_fresh.base.state_dict(), strict=True)
            load_v3_modules_npz(p34, m34_reload)
            m34_reload.eval()
            self.assertEqual(v3_state_sha(m34_reload.v3.state_dict()), sha34)

            canvas = _synthetic_face_512(343)[:256, :256]
            obs_np = np.ones((256, 256), dtype=bool)
            obs_np[:, :16] = False
            raw1, u8_1 = m34_fresh.restore_canvas(canvas, obs_np, return_delivered=True)
            raw2, u8_2 = m34_reload.restore_canvas(canvas, obs_np, return_delivered=True)
            self.assertTrue(np.array_equal(raw1, raw2))
            self.assertTrue(np.array_equal(u8_1[~obs_np], canvas[~obs_np]))

    @unittest.skipUnless(OFFICIAL_CF_WEIGHTS.is_file(), "Official codeformer.pth not present")
    def test_11_v34_real_codeformer_cpu_latency(self):
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            model = CodeFormerV3Restorer(OFFICIAL_CF_WEIGHTS, arch_variant="v34")
            model.eval()
            canvas = _synthetic_face_512(77)[:256, :256]
            obs = np.ones((256, 256), dtype=bool)
            _ = model.restore_canvas(canvas, obs)
            t0 = time.perf_counter()
            raw_f32 = model.restore_canvas(canvas, obs)
            v34_elapsed = time.perf_counter() - t0
            print(f"\n[V3.4 CPU Timing] v34={v34_elapsed:.3f}s (4 threads FP32)")
            self.assertEqual(raw_f32.shape, (256, 256, 3))
            self.assertTrue(np.isfinite(raw_f32).all())
            self.assertLess(v34_elapsed, 8.0, f"V3.4 single-pass CPU latency exceeded 8.0s: {v34_elapsed:.3f}s")
        finally:
            torch.set_num_threads(prev_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
