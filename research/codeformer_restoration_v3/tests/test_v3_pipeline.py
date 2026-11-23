"""CPU smoke & contract test suite for CodeFormer Restoration V3 train/eval pipeline.

Verifies:
1. Zero-Leakage Split Guard against real workspace manifests (split.json, protocol.json,
   v2_data_manifest.json, evaluation_bundle_v1.json, QMUL-SurvFace & ChokePoint subsets).
2. StratifiedRoleBatchSampler balanced clear/normal-degraded/lowlight-joint quotas
   + MultiBackendImageStore (.tar shard & directory) + degradations_v3 integration.
3. Deterministic frozen DEV-Syn generator (PNG + manifest SHA-256 reproducibility).
4. Differentiable FP32 ArcFace (w600k_r50.onnx) & FrozenLPIPSVGG16 non-zero input gradients,
   plus Stage-3 GAN clear-sample masking.
5. Stage 1, Stage 2, and Stage 3 (with lazy R1) one-step backward smoke tests on CPU,
   verifying 515 frozen base tensors never receive gradients and EMA + atomic .npz roundtrip.
6. Comprehensive eval_v3 metrics (raw_fp32, png_floor, png_rint), contact sheet, markdown
   summary, and --stage soup execution.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from research.codeformer_restoration_v3.architecture_v3 import (
    CodeFormerV3Restorer,
    export_v3_npz,
    load_v3_npz,
)
from research.codeformer_restoration_v3.datasets_v3 import (
    MultiBackendImageStore,
    REGIME_GROUPS,
    StratifiedRoleBatchSampler,
    V31_SHARPNESS_CRITICAL_NORMAL_REGIMES,
    V34_PRIORITY_NORMAL_REGIMES,
    V3RestorationDataset,
    ZeroLeakageSplitGuard,
    canonicalize_face_id,
    generate_dev_syn,
    make_stratified_dataloader,
)
from research.codeformer_restoration_v3.degradations_v3 import (
    FFHQ_CANONICAL_5PT_NORM,
    REGIMES,
    generate_cctv_pair,
)
from research.codeformer_restoration_v3.eval_v3 import (
    compute_paired_pixel_metrics,
    evaluate_runner_on_cases,
    format_markdown_summary,
    load_dev_native_cctv,
    load_dev_syn,
    main as eval_main,
    make_v2_runner,
    make_v3_runner,
    save_contact_sheet,
)
from research.codeformer_restoration_v3.losses_v3 import (
    DifferentiableArcFaceLoss,
    MultiScaleHFEnergyLoss,
    PatchGANDiscriminator,
    Stage1Loss,
    Stage2Loss,
    Stage3GANHelper,
    V34Stage1Loss,
    V34Stage2Loss,
    compute_lpips_and_gram,
    compute_wmap_loss,
)
from research.codeformer_restoration_v3.train_v3 import (
    SOUP_ALPHAS,
    V33_PARTIAL_SKIP_PREFIXES,
    V3EMA,
    _load_v3_weights_from_path,
    _pareto_eval_score_v31,
    atomic_export_v3_npz,
    build_optimizer_and_scheduler,
    configure_stage_trainability,
    evaluate_stage1_exit_gates,
    main as train_main,
    reset_freq_head,
    run_model_soup,
    train_step_stage1,
    train_step_stage1_v34,
    train_step_stage2,
    train_step_stage2_v34,
    train_step_stage3,
)

ROOT = Path(__file__).resolve().parents[3]


def _make_dummy_face_u8(seed: int, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    r = np.clip(0.55 + 0.25 * np.sin(xx * 6.0 + seed) + 0.05 * rng.standard_normal((size, size)), 0, 1)
    g = np.clip(0.45 + 0.25 * np.cos(yy * 6.0 + seed) + 0.05 * rng.standard_normal((size, size)), 0, 1)
    b = np.clip(0.40 + 0.20 * np.sin((xx + yy) * 4.0) + 0.05 * rng.standard_normal((size, size)), 0, 1)
    return np.clip(np.rint(np.stack([r, g, b], axis=-1) * 255.0), 0, 255).astype(np.uint8)


def _make_synthetic_batch(batch_size: int = 2, has_pad: bool = True) -> dict:
    gt_512 = torch.rand(batch_size, 3, 512, 512)
    gt_256 = torch.nn.functional.interpolate(gt_512, size=(256, 256), mode="bilinear", align_corners=False)
    x_in_256 = (gt_256 * 0.5 + 0.1).clamp(0.0, 1.0)
    obs = torch.ones(batch_size, 1, 256, 256)
    if has_pad:
        obs[:, :, :, :16] = 0.0
        obs[:, :, :, -16:] = 0.0
        x_in_256 = x_in_256 * obs + 0.5 * (1.0 - obs)
    is_clear = torch.tensor([True] + [False] * (batch_size - 1), dtype=torch.bool)
    deg_targets = torch.tensor([[0.0, 0.0, 0.0, 0.0]] + [[0.6, 0.4, 0.3, 0.5]] * (batch_size - 1))
    return {
        "x_in_256": x_in_256,
        "gt_256": gt_256,
        "gt_512": gt_512,
        "observed_mask": obs,
        "is_clear": is_clear,
        "deg_targets": deg_targets,
    }


class TestV3Pipeline(unittest.TestCase):
    def test_zero_leakage_split_guard(self):
        guard = ZeroLeakageSplitGuard(workspace_root=ROOT, strict_files=True)
        rep = guard.report()
        self.assertEqual(rep.monitor_v2_count, 8)
        self.assertEqual(rep.val_pool_count, 4000)
        self.assertEqual(rep.reserved_final_cctv_count, 45)
        self.assertGreater(rep.train_pool_count, 70000)

        # Zero overlap across allowed_train_ids and all held-out sets
        guard.assert_zero_leakage(
            train_ids=list(guard.allowed_train_ids)[:100],
            dev_syn_ids=list(guard.val_ids)[:20],
        )

        # Must reject monitor_v2 leakage into TRAIN
        leaked_mon = next(iter(guard.monitor_v2_ids))
        with self.assertRaises(AssertionError):
            guard.assert_zero_leakage(train_ids=[leaked_mon])

        # Must reject split['validation'] leakage into TRAIN
        leaked_val = next(iter(guard.val_ids))
        with self.assertRaises(AssertionError):
            guard.assert_zero_leakage(train_ids=[leaked_val])

        # Must reject FINAL CCTV leakage into TRAIN
        leaked_final = next(iter(guard.reserved_final_cctv_ids))
        with self.assertRaises(AssertionError):
            guard.assert_zero_leakage(train_ids=[leaked_final])

        # Path-based CCTV rejection
        self.assertFalse(guard.is_train_allowed("choke_0001", "ChokePoint/P1E_S1_C1/0001.jpg"))
        self.assertFalse(guard.is_train_allowed("surv_3504", "QMUL-SurvFace/3504.jpg"))

        # Regression: _lr24 and va_asian_ canonicalization (audit #4)
        self.assertEqual(canonicalize_face_id("comparison_va_ffhq_28544_blur_lr24"), "ffhq_28544")
        self.assertEqual(canonicalize_face_id("comparison_va_asian_09306_blur_lr24"), "asian_09306")
        self.assertEqual(canonicalize_face_id("va_asian_09306"), "asian_09306")

        # Strict missing manifests must fail fast (audit #1)
        with tempfile.TemporaryDirectory() as empty_td:
            with self.assertRaises(FileNotFoundError):
                ZeroLeakageSplitGuard(workspace_root=empty_td, manifests_dir=empty_td, strict_files=True)

    def test_fail_fast_cli_guards_and_fork_pid_and_landmarks(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Fail-fast missing codeformer.pth in train_v3 (audit #2)
            with self.assertRaises(FileNotFoundError):
                train_main([
                    "--stage", "1",
                    "--out-dir", str(td_path / "run"),
                    "--data-dir", str(td_path),
                    "--codeformer-path", str(td_path / "missing_cf.pth"),
                ])

            # Fail-fast missing LPIPS in Stage 2 train_v3 (audit #3)
            with self.assertRaises(FileNotFoundError):
                train_main([
                    "--stage", "2",
                    "--out-dir", str(td_path / "run2"),
                    "--data-dir", str(td_path),
                    "--allow-random-base",
                    "--codeformer-path", str(td_path / "missing_cf.pth"),
                    "--perceptual-dir", str(td_path / "missing_perc"),
                    "--no-arcface",
                ])

            # Fail-fast missing ArcFace in eval_v3 (audit #3)
            with self.assertRaises(FileNotFoundError):
                eval_main([
                    "--out-dir", str(td_path / "eval_out"),
                    "--allow-random-base",
                    "--codeformer-path", str(td_path / "missing_cf.pth"),
                    "--no-lpips",
                    "--arcface-onnx", str(td_path / "missing_arc.onnx"),
                    "--models", "v3",
                    "--cohorts", "dev_native_cctv",
                ])

            # Fail-fast 0-case cohort in eval_v3 (audit #9)
            with self.assertRaises(RuntimeError):
                eval_main([
                    "--out-dir", str(td_path / "eval_out2"),
                    "--allow-random-base",
                    "--codeformer-path", str(td_path / "missing_cf.pth"),
                    "--no-lpips",
                    "--no-arcface",
                    "--qmul-json", str(td_path / "no_qmul.json"),
                    "--chokepoint-json", str(td_path / "no_choke.json"),
                    "--models", "v3",
                    "--cohorts", "dev_native_cctv",
                ])

            # Linux fork()-safety PID reset on MultiBackendImageStore (audit #5)
            tar_path = td_path / "fork_test.tar"
            with tarfile.open(tar_path, "w") as tf:
                img = _make_dummy_face_u8(99, size=512)
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                raw = buf.getvalue()
                info = tarfile.TarInfo(name="00001.png")
                info.size = len(raw)
                tf.addfile(info, io.BytesIO(raw))
            store = MultiBackendImageStore(tar_path)
            _ = store.read_rgb(store.entries[0][2])
            self.assertTrue(len(store._handles) > 0)
            old_handle = next(iter(store._handles.values()))
            # Simulate forked child PID: handles dict must reset without closing old_handle
            store._handles_pid = -99999
            _ = store.read_rgb(store.entries[0][2])
            self.assertFalse(old_handle.closed)
            old_handle.close()
            store.close()

            # Warped 5-pt landmarks propagation (audit #13)
            img512 = _make_dummy_face_u8(77, size=512)
            _, _, _, _, meta_no_geom = generate_cctv_pair(img512, regime="clear", seed=1, apply_geometry=False)
            _, _, _, _, meta_geom = generate_cctv_pair(img512, regime="clear", seed=2, apply_geometry=True)
            lm0 = np.asarray(meta_no_geom["landmarks_256"], dtype=np.float32)
            lm1 = np.asarray(meta_geom["landmarks_256"], dtype=np.float32)
            self.assertEqual(lm0.shape, (5, 2))
            self.assertEqual(lm1.shape, (5, 2))
            self.assertTrue(np.isfinite(lm1).all())
            np.testing.assert_allclose(lm0, FFHQ_CANONICAL_5PT_NORM * 256.0, atol=1e-4)
            self.assertGreater(float(np.abs(lm1 - lm0).mean()), 1.0)

    def test_stratified_sampler_and_tar_backend(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tar_path = td_path / "shard_000.tar"
            with tarfile.open(tar_path, "w") as tf:
                for i in range(12):
                    img = _make_dummy_face_u8(100 + i, size=512)
                    buf = io.BytesIO()
                    Image.fromarray(img).save(buf, format="PNG")
                    raw = buf.getvalue()
                    info = tarfile.TarInfo(name=f"{i:05d}.png")
                    info.size = len(raw)
                    tf.addfile(info, io.BytesIO(raw))

            ds = V3RestorationDataset(tar_path, split_guard=None, split="train", prepare_crop_prob=0.5)
            self.assertEqual(len(ds), 12)
            loader, sampler = make_stratified_dataloader(
                ds, batch_size=6, num_workers=0, pin_memory=False, persistent_workers=False
            )
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            for batch in batches:
                self.assertEqual(tuple(batch["x_in_256"].shape), (6, 3, 256, 256))
                self.assertEqual(tuple(batch["gt_512"].shape), (6, 3, 512, 512))
                regimes = batch["regime"]
                self.assertTrue(any(r in REGIME_GROUPS["clear"] for r in regimes))
                self.assertTrue(any(r in REGIME_GROUPS["normal_degraded"] for r in regimes))
                self.assertTrue(any(r in REGIME_GROUPS["lowlight_joint"] for r in regimes))
            ds.store.close()

    def test_dev_syn_generator_determinism(self):
        guard = ZeroLeakageSplitGuard(workspace_root=ROOT, strict_files=True)
        val_ffhq = sorted([cid for cid in guard.val_ids if cid.startswith("ffhq_")])[:2]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "src"
            src_dir.mkdir()
            for idx, cid in enumerate(val_ffhq):
                num = cid.split("_")[1]
                Image.fromarray(_make_dummy_face_u8(200 + idx, size=512)).save(src_dir / f"{num}.png")

            m1_path = generate_dev_syn(src_dir, td_path / "dev1", guard, num_ids=2, seed=12345)
            m2_path = generate_dev_syn(src_dir, td_path / "dev2", guard, num_ids=2, seed=12345)

            m1 = json.loads(m1_path.read_text(encoding="utf-8"))
            m2 = json.loads(m2_path.read_text(encoding="utf-8"))
            self.assertEqual(m1["num_cases"], 24)
            self.assertEqual(m1["cases"], m2["cases"])

            # Verify target uses exact same resampler as generate_cctv_pair (audit #6)
            first_cid = m1["selected_ids"][0]
            num = first_cid.split("_")[1]
            src_arr = np.asarray(Image.open(src_dir / f"{num}.png").convert("RGB"), dtype=np.uint8)
            _, expected_gt256, _, _, _ = generate_cctv_pair(src_arr, regime="clear", seed=0, apply_geometry=False)
            saved_gt256 = np.asarray(Image.open(td_path / "dev1" / "targets" / f"{first_cid}.png").convert("RGB"), dtype=np.uint8)
            np.testing.assert_array_equal(saved_gt256, expected_gt256)
            pm = compute_paired_pixel_metrics(saved_gt256 / 255.0, expected_gt256 / 255.0)
            self.assertEqual(pm["mse"], 0.0)
            self.assertTrue(pm["psnr_infinite"])

    def test_arcface_and_lpips_gradients_and_gan_clear_mask(self):
        pred = torch.rand(2, 3, 256, 256, requires_grad=True)
        target = torch.rand(2, 3, 256, 256)
        mask = torch.ones(2, 1, 256, 256)

        # 1. FrozenLPIPSVGG16 gradient check (loads local .pth if present, else direct FrozenLPIPSVGG16)
        from research.codeformer_adaptation_v1.perceptual import FrozenLPIPSVGG16
        from research.codeformer_restoration_v3.losses_v3 import load_frozen_lpips

        perc_dir = ROOT / "outputs/codeformer_perceptual_assets_v1"
        if (perc_dir / "vgg16_features.pth").is_file() and (perc_dir / "lpips_vgg_v01.pth").is_file():
            lpips = load_frozen_lpips(perc_dir, device="cpu")
        else:
            lpips = FrozenLPIPSVGG16().eval().requires_grad_(False)
            with torch.no_grad():
                for w in lpips.calibration:
                    w.fill_(0.05)
        loss_lp = lpips(pred * 2.0 - 1.0, target * 2.0 - 1.0)
        self.assertTrue(torch.isfinite(loss_lp).all())
        grad_lp = torch.autograd.grad(loss_lp, pred, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(grad_lp).all())
        self.assertGreater(float(grad_lp.abs().sum().item()), 0.0)

        # 2. Real DifferentiableArcFaceLoss (w600k_r50.onnx) FP32 non-zero gradient check
        arc_path = Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()
        if arc_path.is_file():
            arc = DifferentiableArcFaceLoss(onnx_path=arc_path, device="cpu")
            loss_id = arc(pred, target, mask=mask)
            self.assertTrue(torch.isfinite(loss_id).all())
            grad_id = torch.autograd.grad(loss_id, pred, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad_id).all())
            self.assertGreater(float(grad_id.abs().sum().item()), 0.0)

        # 3. Stage-3 GAN strictly masks out clear samples
        disc = PatchGANDiscriminator(base_ch=16)
        gan = Stage3GANHelper(disc, lambda_gan=0.01, r1_gamma=10.0, r1_every=16)
        all_clear = torch.tensor([True, True], dtype=torch.bool)
        g_clear = gan.generator_gan_loss(pred, target, mask, all_clear)
        self.assertEqual(float(g_clear.item()), 0.0)
        d_clear, _ = gan.discriminator_step_loss(pred.detach(), target, mask, all_clear, step=16)
        self.assertIsNone(d_clear)

    def test_stage1_stage2_stage3_one_step_backward_and_eval_soup(self):
        torch.manual_seed(7)
        model = CodeFormerV3Restorer(codeformer_weights_path=None, grad_checkpoint=True)
        batch = _make_synthetic_batch(batch_size=2, has_pad=True)
        batch["landmarks"] = torch.from_numpy((FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32)).unsqueeze(0).expand(2, -1, -1).clone()
        device = torch.device("cpu")

        # --- Stage 1 one-step backward ---
        configure_stage_trainability(model, "1")
        model.zero_grad(set_to_none=True)
        s1_metrics = train_step_stage1(model, Stage1Loss(), batch, device=device)
        self.assertTrue(np.isfinite(s1_metrics["loss_total"]))
        # Base backbone must remain 100% frozen with zero gradients
        for p in model.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        # Stage-1 trainable modules must receive gradients; freq_head/output_composer must NOT
        for mod_name in ("deg_encoder", "photo_net", "transformer_lora", "pyramid_sft"):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in model.v3[mod_name].parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Stage 1 expected non-zero grad for {mod_name}")
        for mod_name in ("freq_head", "output_composer"):
            for p in model.v3[mod_name].parameters():
                self.assertIsNone(p.grad)

        # --- Stage 2 one-step backward ---
        configure_stage_trainability(model, "2")
        model.zero_grad(set_to_none=True)
        s2_loss = Stage2Loss(lpips_model=None, arcface_model=None, device=device, w_arcface=0.35)
        s2_metrics = train_step_stage2(model, s2_loss, batch, device=device)
        self.assertTrue(np.isfinite(s2_metrics["loss_total"]))
        for mod_name in ("freq_head", "output_composer", "pyramid_sft"):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in model.v3[mod_name].parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Stage 2 expected non-zero grad for {mod_name}")
        for p in model.base.parameters():
            self.assertIsNone(p.grad)

        # EMA update & atomic .npz export/load roundtrip
        ema = V3EMA(model, decay=0.999)
        ema.update(model)

        # --- Stage 3 one-step backward (with lazy R1 at step=16) ---
        configure_stage_trainability(model, "3")
        model.zero_grad(set_to_none=True)
        disc = PatchGANDiscriminator(base_ch=16)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)
        gan_helper = Stage3GANHelper(disc, lambda_gan=0.01, r1_gamma=10.0, r1_every=16)
        s3_loss = Stage2Loss(lpips_model=None, arcface_model=None, device=device, w_arcface=0.45)
        s3_metrics = train_step_stage3(
            model, s3_loss, gan_helper, opt_d, batch, device=device, step=16, do_d_step=True
        )
        self.assertTrue(np.isfinite(s3_metrics["loss_total"]))
        self.assertGreater(s3_metrics["d_r1"], 0.0)
        for mod_name in ("freq_head", "output_composer"):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in model.v3[mod_name].parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Stage 3 expected non-zero grad for {mod_name}")

        # --- Eval metrics, contact sheet & Model Soup ---
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            s2_npz = td_path / "s2.npz"
            s3_npz = td_path / "s3.npz"
            atomic_export_v3_npz(model, s2_npz)
            atomic_export_v3_npz(model, s3_npz)
            load_v3_npz(s2_npz, model=model)

            # Build a tiny 2-case DEV manifest
            for sub in ("inputs", "targets", "observed"):
                (td_path / sub).mkdir()
            img_gt = _make_dummy_face_u8(50, size=256)
            img_in = np.clip(img_gt.astype(np.int32) - 20, 0, 255).astype(np.uint8)
            obs_u8 = np.full((256, 256), 255, dtype=np.uint8)
            obs_u8[:, :16] = 0
            Image.fromarray(img_in).save(td_path / "inputs/c1.png")
            Image.fromarray(img_gt).save(td_path / "targets/c1.png")
            Image.fromarray(obs_u8).save(td_path / "observed/c1.png")
            manifest = td_path / "dev_syn_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "c1",
                                "reference_id": "ref1",
                                "regime": "lowlight_sensor_noise",
                                "input": "inputs/c1.png",
                                "target": "targets/c1.png",
                                "observed": "observed/c1.png",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cases = load_dev_syn(manifest)
            summary, imgs = evaluate_runner_on_cases(make_v3_runner(model), cases, save_outputs_for_sheet=True)
            for view in ("raw_fp32", "png_floor", "png_rint"):
                ov = summary[view]["overall"]
                self.assertEqual(ov["count"], 1)
                self.assertTrue(ov["unobserved_exact_all"])
                self.assertIsNotNone(ov["mae"])
                self.assertIsNotNone(ov["ssim7"])
                self.assertIsNotNone(ov["central_laplacian_ratio"])

            sheet_p = save_contact_sheet(cases, {"v3": imgs}, td_path / "sheet.png")
            self.assertTrue(sheet_p.is_file())
            md = format_markdown_summary({"dev_syn": {"v3": summary}})
            self.assertIn("png_floor", md)
            self.assertIn("png_rint", md)

            soup_rep = run_model_soup(
                model=model,
                s2_path=s2_npz,
                s3_path=s3_npz,
                dev_manifest=manifest,
                out_dir=td_path / "soup_out",
                alphas=(0.0, 0.5, 1.0),
            )
            self.assertTrue((td_path / "soup_out/v3_modules_ema.npz").is_file())
            self.assertEqual(len(soup_rep["candidates"]), 3)

            # Stage-1 Exit Gate DEV eval check
            s1_ev = evaluate_stage1_exit_gates(model, manifest, device=device)
            for k in (
                "degraded_code_top1_acc",
                "cf_base_code_top1_acc",
                "delta_top1_acc",
                "lowlight_photo_mae",
                "clear_photo_drift",
            ):
                self.assertIn(k, s1_ev)
                self.assertTrue(np.isfinite(s1_ev[k]))

    def test_warmup_scheduler_and_v2_cpu_and_vm_native_layout(self):
        # 1. Warmup + cosine scheduler ramp & resume state_dict check
        model = CodeFormerV3Restorer(codeformer_weights_path=None)
        configure_stage_trainability(model, "1")
        opt, sched = build_optimizer_and_scheduler(model, "1", max_steps=20, lr_override=3e-4, warmup_steps=5)
        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        # Step 0 starts at 0.1 * 3e-4 = 3e-5, ramps up to 3e-4 by step 5, then decays
        self.assertAlmostEqual(lrs[0], 3e-5, places=7)
        self.assertAlmostEqual(lrs[5], 3e-4, places=7)
        self.assertLess(lrs[9], lrs[5])

        # Resume state_dict roundtrip
        sd = sched.state_dict()
        opt2, sched2 = build_optimizer_and_scheduler(model, "1", max_steps=20, lr_override=3e-4, warmup_steps=5)
        sched2.load_state_dict(sd)
        self.assertEqual(sched2.last_epoch, sched.last_epoch)

        # 2. VM <crops_dir>/<case_id>/{input.png,observed.png} resolution
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            qjson = td_path / "qmul.json"
            qjson.write_text(
                json.dumps({"cases": [{"id": "dev_le15_01", "role": "development", "global_person_id": 1, "size_bin": "le15", "source_file": "x.jpg"}]}),
                encoding="utf-8",
            )
            cdir = td_path / "qmul_crops" / "dev_le15_01"
            cdir.mkdir(parents=True)
            Image.fromarray(_make_dummy_face_u8(1, size=256)).save(cdir / "input.png")
            Image.fromarray(np.full((256, 256), 255, dtype=np.uint8)).save(cdir / "observed.png")
            loaded = load_dev_native_cctv(qmul_json=qjson, chokepoint_json=td_path / "none.json", qmul_crops_dir=td_path / "qmul_crops")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["input_path"], cdir / "input.png")
            self.assertEqual(loaded[0]["observed_path"], cdir / "observed.png")

        # 3. make_v2_runner(device="cpu") check if local V2 weights are installed
        v2_dir = ROOT / "integrated_app/models/implementation"
        if (v2_dir / "v2_p2_u0224_heads.npz").is_file() and (v2_dir / "codeformer_r2.pth").is_file():
            v2_runner = make_v2_runner(v2_weights_dir=v2_dir, device="cpu")
            self.assertTrue(callable(v2_runner))

    def test_v31_degradation_v30_invariance_and_2400_fuzz_and_sampler_weighting(self):
        img512 = _make_dummy_face_u8(31415, size=512)

        # 1. Bit-exact v30 invariance (default calibration == explicit calibration="v30")
        for idx, reg in enumerate(REGIMES):
            for geom in (False, True):
                a = generate_cctv_pair(img512, regime=reg, seed=5000 + idx * 2 + int(geom), apply_geometry=geom)
                b = generate_cctv_pair(
                    img512, regime=reg, seed=5000 + idx * 2 + int(geom), apply_geometry=geom, calibration="v30"
                )
                for k in range(4):
                    np.testing.assert_array_equal(a[k], b[k])
                self.assertEqual(a[4], b[4])

        # 2. 2,400-call v31 fuzz across all 12 regimes x mixed geometry (200 calls/regime -> 0 errors)
        low_noise_joint = 0
        clean_lowlight = 0
        exp_jitter_survface = 0
        total_calls = 0
        for r_idx, reg in enumerate(REGIMES):
            for i in range(200):
                geom = (i % 2) == 1
                inp, gt256, gt512, obs, meta = generate_cctv_pair(
                    img512,
                    regime=reg,
                    seed=100000 + r_idx * 1000 + i,
                    apply_geometry=geom,
                    calibration="v31",
                )
                total_calls += 1
                self.assertEqual(inp.shape, (256, 256, 3))
                self.assertEqual(inp.dtype, np.uint8)
                self.assertEqual(gt256.shape, (256, 256, 3))
                self.assertEqual(gt512.shape, (512, 512, 3))
                self.assertEqual(obs.shape, (256, 256))
                for k in ("exposure", "blur", "noise", "codec_lr", "severity", "sufficiency"):
                    self.assertTrue(np.isfinite(meta[k]))
                    self.assertGreaterEqual(meta[k], 0.0)
                    self.assertLessEqual(meta[k], 1.0)
                if reg == "joint_lowlight_blur_codec" and meta["noise"] <= 0.10:
                    low_noise_joint += 1
                if reg == "lowlight_sensor_noise" and meta["noise"] <= 0.12:
                    clean_lowlight += 1
                if reg == "severe_lr_survface" and meta["exposure"] > 0.0:
                    exp_jitter_survface += 1

        self.assertEqual(total_calls, 2400)
        self.assertGreater(low_noise_joint, 40)
        self.assertGreater(clean_lowlight, 35)
        self.assertGreater(exp_jitter_survface, 10)

        # 3. StratifiedRoleBatchSampler v31 2x weighting on 4 sharpness-critical normal_degraded regimes
        sampler = StratifiedRoleBatchSampler(num_samples=4000, batch_size=16, seed=7, calibration="v31")
        counts = {r: 0 for r in REGIME_GROUPS["normal_degraded"]}
        for batch_spec in sampler:
            for _, r in batch_spec:
                if r in counts:
                    counts[r] += 1
        crit_avg = np.mean([counts[r] for r in V31_SHARPNESS_CRITICAL_NORMAL_REGIMES])
        other_avg = np.mean([counts[r] for r in counts if r not in V31_SHARPNESS_CRITICAL_NORMAL_REGIMES])
        self.assertGreater(crit_avg / max(other_avg, 1.0), 1.6)

    def test_v31_hf_energy_and_single_pass_lpips_gram_nan_safety(self):
        from research.codeformer_adaptation_v1.perceptual import FrozenLPIPSVGG16

        hf_loss = MultiScaleHFEnergyLoss()
        lpips = FrozenLPIPSVGG16().eval().requires_grad_(False)
        with torch.no_grad():
            for w in lpips.calibration:
                w.fill_(0.05)

        mask = torch.ones(2, 1, 256, 256)
        mask[:, :, :, :24] = 0.0
        mask[:, :, :, -24:] = 0.0

        cases = [
            ("pure_black", torch.zeros(2, 3, 256, 256), torch.zeros(2, 3, 256, 256)),
            ("pure_white", torch.ones(2, 3, 256, 256), torch.ones(2, 3, 256, 256)),
            ("black_vs_rand", torch.zeros(2, 3, 256, 256), torch.rand(2, 3, 256, 256)),
            ("rand_masked", torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256)),
        ]
        for name, p_init, tgt in cases:
            pred = p_init.clone().detach().requires_grad_(True)
            l_hf = hf_loss(pred, tgt, mask)
            self.assertTrue(torch.isfinite(l_hf).all(), f"HF energy non-finite on {name}")
            g_hf = torch.autograd.grad(l_hf, pred, retain_graph=False)[0]
            self.assertTrue(torch.isfinite(g_hf).all(), f"HF energy grad non-finite on {name}")

            pred2 = p_init.clone().detach().requires_grad_(True)
            y_masked = pred2 * mask + tgt * (1.0 - mask)
            l_lp, l_gr = compute_lpips_and_gram(lpips, y_masked * 2.0 - 1.0, tgt * 2.0 - 1.0, compute_gram=True)
            self.assertTrue(torch.isfinite(l_lp).all(), f"LPIPS non-finite on {name}")
            self.assertTrue(torch.isfinite(l_gr).all(), f"Gram non-finite on {name}")
            g_comb = torch.autograd.grad(l_lp + l_gr, pred2, retain_graph=False)[0]
            self.assertTrue(torch.isfinite(g_comb).all(), f"LPIPS+Gram grad non-finite on {name}")

        # Verify exact numerical equivalence for l_lpips when w_vgg_gram == 0
        pred_eq = torch.rand(2, 3, 256, 256)
        tgt_eq = torch.rand(2, 3, 256, 256)
        s2_v30 = Stage2Loss(lpips_model=lpips, arcface_model=None, w_vgg_gram=0.0)
        _, m_v30 = s2_v30({"y_256": pred_eq}, tgt_eq, observed_mask=mask)
        y_m = pred_eq * mask + tgt_eq * (1.0 - mask)
        ref_lpips = float(lpips(y_m * 2.0 - 1.0, tgt_eq * 2.0 - 1.0).item())
        self.assertEqual(m_v30["l_lpips"], ref_lpips)
        self.assertEqual(m_v30["l_vgg_gram"], 0.0)

    def test_v31_wmap_loss_grad_flow_and_stage2b_stage3b_trainability_and_pareto(self):
        torch.manual_seed(11)
        model = CodeFormerV3Restorer(codeformer_weights_path=None, grad_checkpoint=True)
        batch = _make_synthetic_batch(batch_size=2, has_pad=True)
        batch["landmarks"] = (
            torch.from_numpy((FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32))
            .unsqueeze(0)
            .expand(2, -1, -1)
            .clone()
        )
        device = torch.device("cpu")

        # 1. compute_wmap_loss directly drives gradients into deg_encoder.w_heads
        model.zero_grad(set_to_none=True)
        deg_out = model.v3["deg_encoder"](batch["x_in_256"], batch["observed_mask"])
        l_wm = compute_wmap_loss(
            deg_out["w_maps"],
            batch["deg_targets"],
            observed_mask=batch["observed_mask"],
            is_clear=batch["is_clear"],
            sufficiency=deg_out["sufficiency"],
        )
        self.assertTrue(torch.isfinite(l_wm).all())
        self.assertGreater(float(l_wm.item()), 0.0)
        l_wm.backward()
        whead_grad = sum(
            float(p.grad.abs().sum().item())
            for p in model.v3["deg_encoder"].w_heads.parameters()
            if p.grad is not None
        )
        self.assertGreater(whead_grad, 0.0)

        # 2. Stage-2B one-step backward + exact LR schedule (1.2e-4 -> 1.5e-5, 4.0e-5 -> 5.0e-6)
        configure_stage_trainability(model, "2b")
        opt_2b, sched_2b = build_optimizer_and_scheduler(model, "2b", max_steps=100, warmup_steps=10)
        self.assertEqual(len(opt_2b.param_groups), 2)
        self.assertAlmostEqual(opt_2b.param_groups[0]["initial_lr"], 1.2e-4, places=10)
        self.assertAlmostEqual(opt_2b.param_groups[1]["initial_lr"], 4.0e-5, places=10)
        for _ in range(100):
            opt_2b.step()
            sched_2b.step()
        self.assertAlmostEqual(opt_2b.param_groups[0]["lr"], 1.5e-5, places=8)
        self.assertAlmostEqual(opt_2b.param_groups[1]["lr"], 5.0e-6, places=8)

        model.zero_grad(set_to_none=True)
        s2b_loss = Stage2Loss(
            lpips_model=None,
            arcface_model=None,
            device=device,
            w_charb=0.55,
            w_ssim=0.20,
            w_chroma=0.15,
            w_lpips=0.65,
            w_vgg_gram=0.20,
            w_hf_energy=0.18,
            w_arcface=0.45,
            w_lap=0.10,
            w_fft=0.12,
            w_aux512=0.12,
            w_wmap=0.30,
            w_ret_code=0.12,
            w_ret_sft=0.03,
            w_ret_photo=0.10,
            w_ret_deg=0.15,
            w_clear_l1=2.0,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
        m2b = train_step_stage2(model, s2b_loss, batch, device=device)
        for key in ("loss_total", "l_hf_energy", "l_wmap", "l_ret_deg", "l_vgg_gram"):
            self.assertIn(key, m2b)
            self.assertTrue(np.isfinite(m2b[key]))
        for p in model.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for mod_name in ("deg_encoder", "photo_net", "transformer_lora", "pyramid_sft", "freq_head", "output_composer"):
            gnorm = sum(
                float(p.grad.abs().sum().item())
                for p in model.v3[mod_name].parameters()
                if p.grad is not None
            )
            self.assertGreater(gnorm, 0.0, f"Stage-2B expected non-zero grad for {mod_name}")

        # 3. Stage-3B one-step backward + exact frozen/trainable boundary + lambda_gan ramp
        configure_stage_trainability(model, "3b")
        opt_3b, sched_3b = build_optimizer_and_scheduler(model, "3b", max_steps=100, warmup_steps=10)
        self.assertEqual(len(opt_3b.param_groups), 2)
        self.assertAlmostEqual(opt_3b.param_groups[0]["initial_lr"], 5.0e-5, places=10)
        self.assertAlmostEqual(opt_3b.param_groups[1]["initial_lr"], 3.5e-5, places=10)
        for _ in range(100):
            opt_3b.step()
            sched_3b.step()
        self.assertAlmostEqual(opt_3b.param_groups[0]["lr"], 5.0e-6, places=8)
        self.assertAlmostEqual(opt_3b.param_groups[1]["lr"], 3.5e-6, places=8)

        model.zero_grad(set_to_none=True)
        disc = PatchGANDiscriminator(base_ch=16)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=0.0)
        gan_helper = Stage3GANHelper(disc, lambda_gan=0.06, r1_gamma=10.0, r1_every=16)
        s3b_loss = Stage2Loss(
            lpips_model=None,
            arcface_model=None,
            device=device,
            w_charb=0.40,
            w_ssim=0.15,
            w_chroma=0.15,
            w_lpips=0.75,
            w_vgg_gram=0.25,
            w_hf_energy=0.20,
            w_arcface=0.55,
            w_lap=0.05,
            w_fft=0.12,
            w_aux512=0.10,
            w_wmap=0.25,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.0,
            w_ret_deg=0.0,
            w_clear_l1=2.5,
            v31_clear_anchor=True,
            v31_aux512=True,
        )

        # Verify lambda_gan ramp (scale=0.0 vs scale=0.5 vs scale=1.0; eval() freezes spectral_norm u-vec)
        disc.eval()
        y_probe = torch.rand(2, 3, 256, 256, requires_grad=True)
        g0 = gan_helper.generator_gan_loss(y_probe, batch["gt_256"], batch["observed_mask"], batch["is_clear"], lambda_scale=0.0)
        g5 = gan_helper.generator_gan_loss(y_probe, batch["gt_256"], batch["observed_mask"], batch["is_clear"], lambda_scale=0.5)
        g1 = gan_helper.generator_gan_loss(y_probe, batch["gt_256"], batch["observed_mask"], batch["is_clear"], lambda_scale=1.0)
        disc.train()
        self.assertEqual(float(g0.item()), 0.0)
        self.assertAlmostEqual(float(g5.item()), 0.5 * float(g1.item()), places=6)

        m3b = train_step_stage3(
            model, s3b_loss, gan_helper, opt_d, batch, device=device, step=16, do_d_step=True, lambda_scale=0.5
        )
        self.assertTrue(np.isfinite(m3b["loss_total"]))
        self.assertAlmostEqual(m3b["gan_lambda_eff"], 0.03, places=6)
        self.assertGreater(m3b["d_r1"], 0.0)

        # Trainable in Stage-3B: pyramid_sft, freq_head, output_composer, deg_encoder.w_heads
        for mod_name in ("pyramid_sft", "freq_head", "output_composer"):
            gnorm = sum(
                float(p.grad.abs().sum().item())
                for p in model.v3[mod_name].parameters()
                if p.grad is not None
            )
            self.assertGreater(gnorm, 0.0, f"Stage-3B expected non-zero grad for {mod_name}")
        whead_3b_grad = sum(
            float(p.grad.abs().sum().item())
            for p in model.v3["deg_encoder"].w_heads.parameters()
            if p.grad is not None
        )
        self.assertGreater(whead_3b_grad, 0.0, "Stage-3B expected non-zero grad for deg_encoder.w_heads")

        # Frozen in Stage-3B: base, photo_net, transformer_lora, deg_encoder.stem & non-w_heads
        for p in model.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for mod_name in ("photo_net", "transformer_lora"):
            for p in model.v3[mod_name].parameters():
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
        whead_ids = {id(p) for p in model.v3["deg_encoder"].w_heads.parameters()}
        for p in model.v3["deg_encoder"].stem.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for p in model.v3["deg_encoder"].parameters():
            if id(p) not in whead_ids:
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)

        # 4. SOUP_ALPHAS and _pareto_eval_score_v31 ranking crisp > blurry > oversharpened
        self.assertEqual(SOUP_ALPHAS, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        crisp = {
            "overall": {"psnr_db": 29.0, "ssim7": 0.88, "lpips": 0.112, "arcface_cos": 0.892, "central_laplacian_ratio": 0.95, "catastrophic_id_flips": 0},
            "by_role": {"global_defocus_blur": {"central_laplacian_ratio": 0.88}, "clear": {"psnr_db": 47.0}, "lowlight_sensor_noise": {"psnr_db": 26.0}, "haze_colorcast_glare": {"psnr_db": 24.0}},
        }
        blurry = {
            "overall": {"psnr_db": 29.5, "ssim7": 0.89, "lpips": 0.155, "arcface_cos": 0.885, "central_laplacian_ratio": 0.68, "catastrophic_id_flips": 0},
            "by_role": {"global_defocus_blur": {"central_laplacian_ratio": 0.55}, "clear": {"psnr_db": 47.0}, "lowlight_sensor_noise": {"psnr_db": 26.0}, "haze_colorcast_glare": {"psnr_db": 24.0}},
        }
        halos = {
            "overall": {"psnr_db": 27.5, "ssim7": 0.85, "lpips": 0.120, "arcface_cos": 0.880, "central_laplacian_ratio": 1.38, "catastrophic_id_flips": 0},
            "by_role": {"global_defocus_blur": {"central_laplacian_ratio": 1.30}, "clear": {"psnr_db": 47.0}, "lowlight_sensor_noise": {"psnr_db": 26.0}, "haze_colorcast_glare": {"psnr_db": 24.0}},
        }
        self.assertGreater(_pareto_eval_score_v31(crisp), _pareto_eval_score_v31(blurry))
        self.assertGreater(_pareto_eval_score_v31(crisp), _pareto_eval_score_v31(halos))

    def test_v32_lanczos_gt_reset_freq_head_and_stage2c_3c(self):
        import cv2

        # 1. Exact param count & zero NPZ schema breakage + freq_gates output contract
        torch.manual_seed(32)
        model = CodeFormerV3Restorer(codeformer_weights_path=None, grad_checkpoint=True)
        self.assertEqual(model.trainable_parameter_count(), 2_696_707)

        batch = _make_synthetic_batch(batch_size=2, has_pad=True)
        batch["landmarks"] = (
            torch.from_numpy((FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32))
            .unsqueeze(0)
            .expand(2, -1, -1)
            .clone()
        )
        device = torch.device("cpu")

        with torch.no_grad():
            out_s1 = model(batch["x_in_256"], batch["observed_mask"], stage1_only=True)
            self.assertIn("freq_gates", out_s1)
            self.assertIsNone(out_s1["freq_gates"])
            out_full = model(batch["x_in_256"], batch["observed_mask"], stage1_only=False)
            self.assertIsInstance(out_full["freq_gates"], dict)
            self.assertEqual(set(out_full["freq_gates"].keys()), {"32", "64", "128", "256"})

        # 2. v30 and v31 bit-exact invariance preserved
        img512 = _make_dummy_face_u8(20260320, size=512)
        for idx, reg in enumerate(REGIMES):
            for geom in (False, True):
                a = generate_cctv_pair(img512, regime=reg, seed=7000 + idx * 2 + int(geom), apply_geometry=geom)
                b = generate_cctv_pair(
                    img512, regime=reg, seed=7000 + idx * 2 + int(geom), apply_geometry=geom, calibration="v30"
                )
                for k in range(4):
                    np.testing.assert_array_equal(a[k], b[k])
                self.assertEqual(a[4], b[4])
                # v31 GT must still match v30 Bilinear GT bit-for-bit
                c = generate_cctv_pair(
                    img512, regime=reg, seed=7000 + idx * 2 + int(geom), apply_geometry=geom, calibration="v31"
                )
                np.testing.assert_array_equal(a[1], c[1])
                np.testing.assert_array_equal(a[2], c[2])

        # 3. v32 Lanczos GT >= 1.35x Laplacian variance vs v30 Bilinear GT on high-frequency synthetic face
        yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
        hf_pat = (
            0.50
            + 0.22 * np.sin(xx * 0.95) * np.cos(yy * 0.85)
            + 0.18 * np.sin((xx + yy) * 1.35)
            + 0.10 * np.cos((xx - yy) * 1.65)
        )
        hf_u8 = np.clip(np.rint(np.stack([hf_pat, hf_pat * 0.95, hf_pat * 0.90], axis=-1) * 255.0), 0, 255).astype(
            np.uint8
        )
        _, gt_v30, _, _, _ = generate_cctv_pair(hf_u8, regime="clear", seed=1, apply_geometry=False, calibration="v30")
        _, gt_v32, _, _, _ = generate_cctv_pair(hf_u8, regime="clear", seed=1, apply_geometry=False, calibration="v32")
        lap_v30 = float(cv2.Laplacian(cv2.cvtColor(gt_v30, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var())
        lap_v32 = float(cv2.Laplacian(cv2.cvtColor(gt_v32, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var())
        self.assertGreaterEqual(lap_v32 / max(lap_v30, 1e-6), 1.35)

        # 4. v32 clear MSE == 0.0 (both use_geom=False and use_geom=True without JPEG) + bar preservation
        found_no_geom_exact = False
        found_geom_exact = False
        found_geom_jpeg = False
        for s in range(40):
            inp0, gt0, _, _, _ = generate_cctv_pair(hf_u8, regime="clear", seed=100 + s, apply_geometry=False, calibration="v32")
            if np.array_equal(inp0, gt0):
                found_no_geom_exact = True
            inp1, gt1, _, obs1, _ = generate_cctv_pair(hf_u8, regime="clear", seed=200 + s, apply_geometry=True, calibration="v32")
            if not obs1.all():
                np.testing.assert_array_equal(inp1[~obs1], gt1[~obs1])
                self.assertTrue((inp1[~obs1] == 128).all())
            if np.array_equal(inp1, gt1):
                found_geom_exact = True
            else:
                found_geom_jpeg = True
        self.assertTrue(found_no_geom_exact, "Expected exact MSE==0.0 for v32 clear (use_geom=False)")
        self.assertTrue(found_geom_exact, "Expected exact MSE==0.0 for v32 clear (use_geom=True, no JPEG)")
        self.assertTrue(found_geom_jpeg, "Expected some mild-JPEG clear samples for v32 (use_geom=True)")

        # 5. 600-call v32 fuzz across all 12 regimes (50 calls/regime x mixed geometry -> 0 errors)
        total_v32 = 0
        for r_idx, reg in enumerate(REGIMES):
            for i in range(50):
                inp, gt256, gt512, obs, meta = generate_cctv_pair(
                    img512,
                    regime=reg,
                    seed=320000 + r_idx * 100 + i,
                    apply_geometry=(i % 2 == 1),
                    calibration="v32",
                )
                total_v32 += 1
                self.assertEqual(inp.shape, (256, 256, 3))
                self.assertEqual(gt256.shape, (256, 256, 3))
                self.assertEqual(gt512.shape, (512, 512, 3))
                self.assertEqual(obs.shape, (256, 256))
                for k in ("exposure", "blur", "noise", "codec_lr", "severity", "sufficiency"):
                    self.assertTrue(np.isfinite(meta[k]))
                    self.assertGreaterEqual(meta[k], 0.0)
                    self.assertLessEqual(meta[k], 1.0)
        self.assertEqual(total_v32, 600)

        # 6. Sampler v32 2x weighting & generate_dev_syn(..., calibration="v32")
        sampler_v32 = StratifiedRoleBatchSampler(num_samples=2000, batch_size=16, seed=9, calibration="v32")
        counts_v32 = {r: 0 for r in REGIME_GROUPS["normal_degraded"]}
        for bspec in sampler_v32:
            for _, r in bspec:
                if r in counts_v32:
                    counts_v32[r] += 1
        crit_v32 = np.mean([counts_v32[r] for r in V31_SHARPNESS_CRITICAL_NORMAL_REGIMES])
        other_v32 = np.mean([counts_v32[r] for r in counts_v32 if r not in V31_SHARPNESS_CRITICAL_NORMAL_REGIMES])
        self.assertGreater(crit_v32 / max(other_v32, 1.0), 1.6)

        guard = ZeroLeakageSplitGuard(workspace_root=ROOT, strict_files=True)
        val_ffhq = sorted([cid for cid in guard.val_ids if cid.startswith("ffhq_")])[:1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "src"
            src_dir.mkdir()
            num = val_ffhq[0].split("_")[1]
            Image.fromarray(hf_u8).save(src_dir / f"{num}.png")
            m_v32_path = generate_dev_syn(src_dir, td_path / "dev_v32", guard, num_ids=1, seed=20260320, calibration="v32")
            m_v32 = json.loads(m_v32_path.read_text(encoding="utf-8"))
            self.assertEqual(m_v32["calibration"], "v32")
            saved_gt = np.asarray(Image.open(td_path / "dev_v32" / "targets" / f"{val_ffhq[0]}.png").convert("RGB"), dtype=np.uint8)
            np.testing.assert_array_equal(saved_gt, gt_v32)

        # 7. --reset-freq-head zeros freq_head outputs without altering photo_net/pyramid_sft,
        #    and is NOT triggered on --resume
        with torch.no_grad():
            for p in model.v3["freq_head"].parameters():
                p.fill_(-0.75)
            for p in model.v3["photo_net"].parameters():
                p.add_(0.03)
            for p in model.v3["pyramid_sft"].parameters():
                p.add_(0.02)
        photo_before = {k: v.detach().clone() for k, v in model.v3["photo_net"].state_dict().items()}
        pyr_before = {k: v.detach().clone() for k, v in model.v3["pyramid_sft"].state_dict().items()}
        reset_freq_head(model)
        for k, v in model.v3["photo_net"].state_dict().items():
            self.assertTrue(torch.equal(v, photo_before[k]))
        for k, v in model.v3["pyramid_sft"].state_dict().items():
            self.assertTrue(torch.equal(v, pyr_before[k]))
        with torch.no_grad():
            enc_d = {s: torch.randn((1, ch, int(s), int(s))) for s, ch in [("32", 256), ("64", 256), ("128", 128), ("256", 128)]}
            d_d = {s: torch.randn((1, ch, int(s), int(s))) for s, ch in [("32", 96), ("64", 64), ("128", 48), ("256", 32)]}
            for s in ("32", "64", "128", "256"):
                dec_s = torch.randn_like(enc_d[s])
                fus_s = torch.randn_like(enc_d[s])
                out_s, gate_s = model.v3["freq_head"].forward_site(s, enc_d[s], dec_s, fus_s, enc_d["32"], d_d[s], return_gate=True)
                self.assertEqual(float((out_s - fus_s).abs().max().item()), 0.0)
                self.assertEqual(float(gate_s.abs().max().item()), 0.0)

        # CLI check: --resume must NOT reset freq_head even when --reset-freq-head is passed
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_dir = td_path / "data"
            data_dir.mkdir()
            for i in range(2):
                Image.fromarray(_make_dummy_face_u8(500 + i, size=512)).save(data_dir / f"{i:05d}.png")
            out_dir = td_path / "run_2c"
            out_dir.mkdir()
            # Prepare a resume checkpoint with distinct non-zero freq_head weights at step=1
            m_res = CodeFormerV3Restorer(codeformer_weights_path=None)
            with torch.no_grad():
                for s in ("32", "64", "128", "256"):
                    m_res.v3["freq_head"].sites[s].freq_gate[-1].bias.fill_(0.42)
            configure_stage_trainability(m_res, "2c")
            opt_r, sched_r = build_optimizer_and_scheduler(m_res, "2c", max_steps=1, warmup_steps=0)
            ema_r = V3EMA(m_res, decay=0.9985)
            resume_pt = out_dir / "latest.pt"
            torch.save(
                {
                    "stage": "2c",
                    "step": 1,
                    "epoch": 1,
                    "best_metric": -1e9,
                    "v3_state_dict": m_res.v3.state_dict(),
                    "opt_g": opt_r.state_dict(),
                    "sched_g": sched_r.state_dict(),
                    "ema": ema_r.state_dict(),
                },
                resume_pt,
            )
            train_main([
                "--stage", "2c",
                "--reset-freq-head",
                "--resume", str(resume_pt),
                "--max-steps", "1",
                "--batch-size", "2",
                "--grad-accum", "1",
                "--num-workers", "0",
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
                "--allow-random-base",
                "--codeformer-path", str(td_path / "missing.pth"),
                "--no-lpips",
                "--no-arcface",
                "--no-split-guard",
                "--device", "cpu",
            ])
            journal_lines = [json.loads(line) for line in (out_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertFalse(journal_lines[0]["reset_freq_head"])

        # 8. w_pos_gate & w_adain_floor finite non-zero gradients into freq_gate & alpha_adain_head
        configure_stage_trainability(model, "2c")
        with torch.no_grad():
            for s in ("32", "64", "128", "256"):
                model.v3["freq_head"].sites[s].freq_gate[-1].bias.fill_(-0.6)
            model.v3["deg_encoder"].alpha_adain_head[-1].bias.fill_(0.0)  # sigmoid(0)=0.5 < 0.85
        model.zero_grad(set_to_none=True)
        out_reg = model(batch["x_in_256"], batch["observed_mask"], stage1_only=False)
        reg_loss_fn = Stage2Loss(
            lpips_model=None,
            arcface_model=None,
            device=device,
            w_charb=0.0,
            w_ssim=0.0,
            w_chroma=0.0,
            w_lpips=0.0,
            w_arcface=0.0,
            w_lap=0.0,
            w_fft=0.0,
            w_aux512=0.0,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.0,
            w_clear_l1=0.0,
            w_adain_floor=0.10,
            w_pos_gate=0.15,
        )
        l_reg, m_reg = reg_loss_fn(out_reg, batch["gt_256"], observed_mask=batch["observed_mask"])
        self.assertGreater(m_reg["l_adain_floor"], 0.0)
        self.assertGreater(m_reg["l_pos_gate"], 0.0)
        l_reg.backward()
        adain_grad = sum(
            float(p.grad.abs().sum().item())
            for p in model.v3["deg_encoder"].alpha_adain_head.parameters()
            if p.grad is not None
        )
        self.assertGreater(adain_grad, 0.0)
        for s in ("32", "64", "128", "256"):
            fg_grad = sum(
                float(p.grad.abs().sum().item())
                for p in model.v3["freq_head"].sites[s].freq_gate.parameters()
                if p.grad is not None
            )
            self.assertGreater(fg_grad, 0.0, f"Expected non-zero w_pos_gate grad at site {s}")

        # 9. Stage-2C LR schedule & one-step backward
        configure_stage_trainability(model, "2c")
        opt_2c, sched_2c = build_optimizer_and_scheduler(model, "2c", max_steps=100, warmup_steps=10)
        self.assertEqual(len(opt_2c.param_groups), 2)
        self.assertAlmostEqual(opt_2c.param_groups[0]["initial_lr"], 1.5e-4, places=10)
        self.assertAlmostEqual(opt_2c.param_groups[1]["initial_lr"], 3.0e-5, places=10)
        for _ in range(100):
            opt_2c.step()
            sched_2c.step()
        self.assertAlmostEqual(opt_2c.param_groups[0]["lr"], 1.5e-5, places=8)
        self.assertAlmostEqual(opt_2c.param_groups[1]["lr"], 4.0e-6, places=8)

        model.zero_grad(set_to_none=True)
        s2c_loss = Stage2Loss(
            lpips_model=None,
            arcface_model=None,
            device=device,
            w_charb=0.35,
            w_ssim=0.12,
            w_chroma=0.15,
            w_lpips=0.85,
            w_vgg_gram=0.30,
            w_hf_energy=0.25,
            w_arcface=0.60,
            w_lap=0.05,
            w_fft=0.15,
            w_aux512=0.10,
            w_wmap=0.30,
            w_ret_code=0.10,
            w_ret_sft=0.0,
            w_ret_photo=0.10,
            w_ret_deg=0.15,
            w_adain_floor=0.10,
            w_pos_gate=0.15,
            w_clear_l1=2.5,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
        m2c = train_step_stage2(model, s2c_loss, batch, device=device)
        for key in ("loss_total", "l_adain_floor", "l_pos_gate", "l_hf_energy", "l_wmap"):
            self.assertIn(key, m2c)
            self.assertTrue(np.isfinite(m2c[key]))
        for p in model.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for mod_name in ("deg_encoder", "photo_net", "transformer_lora", "pyramid_sft", "freq_head", "output_composer"):
            gnorm = sum(
                float(p.grad.abs().sum().item())
                for p in model.v3[mod_name].parameters()
                if p.grad is not None
            )
            self.assertGreater(gnorm, 0.0, f"Stage-2C expected non-zero grad for {mod_name}")

        # 10. Stage-3C LR schedule, exact trainable/frozen boundary & lambda_gan=0.08 ramp
        configure_stage_trainability(model, "3c")
        opt_3c, sched_3c = build_optimizer_and_scheduler(model, "3c", max_steps=100, warmup_steps=10)
        self.assertEqual(len(opt_3c.param_groups), 2)
        self.assertAlmostEqual(opt_3c.param_groups[0]["initial_lr"], 5.0e-5, places=10)
        self.assertAlmostEqual(opt_3c.param_groups[1]["initial_lr"], 3.0e-5, places=10)
        for _ in range(100):
            opt_3c.step()
            sched_3c.step()
        self.assertAlmostEqual(opt_3c.param_groups[0]["lr"], 5.0e-6, places=8)
        self.assertAlmostEqual(opt_3c.param_groups[1]["lr"], 3.0e-6, places=8)

        model.zero_grad(set_to_none=True)
        disc = PatchGANDiscriminator(base_ch=16)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=0.0)
        gan_helper = Stage3GANHelper(disc, lambda_gan=0.08, r1_gamma=10.0, r1_every=16)
        s3c_loss = Stage2Loss(
            lpips_model=None,
            arcface_model=None,
            device=device,
            w_charb=0.25,
            w_ssim=0.08,
            w_chroma=0.15,
            w_lpips=0.90,
            w_vgg_gram=0.35,
            w_hf_energy=0.25,
            w_arcface=0.65,
            w_lap=0.03,
            w_fft=0.15,
            w_aux512=0.08,
            w_wmap=0.25,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.0,
            w_ret_deg=0.0,
            w_adain_floor=0.10,
            w_pos_gate=0.15,
            w_clear_l1=3.0,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
        m3c = train_step_stage3(
            model, s3c_loss, gan_helper, opt_d, batch, device=device, step=16, do_d_step=True, lambda_scale=0.5
        )
        self.assertTrue(np.isfinite(m3c["loss_total"]))
        self.assertAlmostEqual(m3c["gan_lambda_eff"], 0.04, places=6)
        self.assertGreater(m3c["d_r1"], 0.0)

        for mod_name in ("pyramid_sft", "freq_head", "output_composer"):
            gnorm = sum(
                float(p.grad.abs().sum().item())
                for p in model.v3[mod_name].parameters()
                if p.grad is not None
            )
            self.assertGreater(gnorm, 0.0, f"Stage-3C expected non-zero grad for {mod_name}")
        for sub_name, sub_mod in (
            ("w_heads", model.v3["deg_encoder"].w_heads),
            ("alpha_adain_head", model.v3["deg_encoder"].alpha_adain_head),
        ):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in sub_mod.parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Stage-3C expected non-zero grad for deg_encoder.{sub_name}")

        for p in model.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for mod_name in ("photo_net", "transformer_lora"):
            for p in model.v3[mod_name].parameters():
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
        trainable_deg_ids = (
            {id(p) for p in model.v3["deg_encoder"].w_heads.parameters()}
            | {id(p) for p in model.v3["deg_encoder"].alpha_adain_head.parameters()}
        )
        for p in model.v3["deg_encoder"].parameters():
            if id(p) not in trainable_deg_ids:
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)

    def test_v33_degradation_warmstart_stage2d_3d_soup_and_app_test_eval(self):
        torch.manual_seed(33)
        img512 = _make_dummy_face_u8(20260330, size=512)

        # 1. calibration="v33": center-face local_partial_blur + m_blur + clear MSE == 0.0 + confetti + heavy blur
        found_center_blur = False
        found_clear_exact_geom = False
        found_confetti = False
        for s in range(40):
            inp, gt256, gt512, obs, meta = generate_cctv_pair(
                img512, regime="local_partial_blur", seed=4000 + s, apply_geometry=(s % 2 == 1), calibration="v33"
            )
            self.assertIn("blur_mask", meta)
            bm = meta["blur_mask"]
            self.assertEqual(bm.shape, (256, 256))
            self.assertEqual(bm.dtype, np.float32)
            self.assertGreaterEqual(float(bm.min()), 0.0)
            self.assertLessEqual(float(bm.max()), 1.0)
            if (~obs).any():
                self.assertEqual(float(bm[~obs].max()), 0.0)
            if obs.all():
                center_m = float(bm[96:160, 96:160].mean())
                corner_m = float(np.mean([bm[:24, :24].mean(), bm[:24, -24:].mean(), bm[-24:, :24].mean(), bm[-24:, -24:].mean()]))
                if center_m > 0.70 and corner_m < 0.15:
                    found_center_blur = True

            inp_c, gt_c, _, obs_c, _ = generate_cctv_pair(
                img512, regime="clear", seed=4200 + s, apply_geometry=True, calibration="v33"
            )
            if not obs_c.all():
                np.testing.assert_array_equal(inp_c[~obs_c], gt_c[~obs_c])
                self.assertTrue((inp_c[~obs_c] == 128).all())
            if np.array_equal(inp_c, gt_c):
                found_clear_exact_geom = True

            _, _, _, _, meta_d = generate_cctv_pair(
                img512, regime="daylight_sensor_noise", seed=4400 + s, apply_geometry=False, calibration="v33"
            )
            if meta_d["noise"] >= 0.70:
                found_confetti = True

        self.assertTrue(found_center_blur, "Expected v33 center-face-only local_partial_blur mode")
        self.assertTrue(found_clear_exact_geom, "Expected v33 clear MSE==0.0 with use_geom=True")
        self.assertTrue(found_confetti, "Expected v33 crisp RGB pixel-confetti daylight noise submode")

        # 2. Partial warm-start from V3.2 (v30) .npz into V3.3 (v33) model
        m32 = CodeFormerV3Restorer(codeformer_weights_path=None, arch_variant="v30")
        with torch.no_grad():
            for p in m32.v3.parameters():
                p.add_(0.017)
        m33 = CodeFormerV3Restorer(codeformer_weights_path=None, arch_variant="v33", grad_checkpoint=True)
        pyr_before = {k: v.detach().clone() for k, v in m33.v3["pyramid_sft"].state_dict().items()}
        freq_before = {k: v.detach().clone() for k, v in m33.v3["freq_head"].state_dict().items()}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            v32_npz = td_path / "v32_src.npz"
            export_v3_npz(m32, v32_npz)
            rep = _load_v3_weights_from_path(
                m33, v32_npz, torch.device("cpu"), partial=True, skip_prefixes=V33_PARTIAL_SKIP_PREFIXES
            )
            self.assertGreater(rep["loaded"], 100)
            self.assertGreater(rep["skipped"], 50)
            for mod_name in ("deg_encoder", "photo_net", "output_composer"):
                for k, v in m33.v3[mod_name].state_dict().items():
                    self.assertTrue(torch.equal(v, m32.v3[mod_name].state_dict()[k]), f"Warm-start mismatch in {mod_name}.{k}")
            for k, v in m33.v3["pyramid_sft"].state_dict().items():
                self.assertTrue(torch.equal(v, pyr_before[k]), f"pyramid_sft.{k} should remain fresh on V3.3 partial init")
            for k, v in m33.v3["freq_head"].state_dict().items():
                self.assertTrue(torch.equal(v, freq_before[k]), f"freq_head.{k} should remain fresh on V3.3 partial init")

            # 3. Stage-2D 1-step forward/backward with finite non-zero grads
            batch = _make_synthetic_batch(batch_size=2, has_pad=True)
            bm_t = torch.zeros(2, 1, 256, 256)
            bm_t[1, :, 64:192, 64:192] = 1.0  # center-face blur on degraded sample
            batch["blur_mask"] = bm_t * batch["observed_mask"]
            batch["landmarks"] = (
                torch.from_numpy((FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32))
                .unsqueeze(0)
                .expand(2, -1, -1)
                .clone()
            )
            device = torch.device("cpu")

            configure_stage_trainability(m33, "2d")
            m33.zero_grad(set_to_none=True)
            s2d_loss = Stage2Loss(
                lpips_model=None,
                arcface_model=None,
                device=device,
                w_charb=0.35,
                w_ssim=0.12,
                w_chroma=0.15,
                w_lpips=0.85,
                w_vgg_gram=0.25,
                w_hf_energy=0.04,
                w_arcface=0.60,
                w_lap=0.05,
                w_fft=0.10,
                w_aux512=0.08,
                w_wmap=0.35,
                w_ret_code=0.0,
                w_ret_sft=0.0,
                w_ret_photo=0.10,
                w_ret_deg=0.15,
                w_adain_floor=0.10,
                w_pos_gate=0.15,  # must stay 0.0 cleanly for positive hp5_only gates
                w_clear_l1=1.5,
                v33_mode=True,
            )
            m2d = train_step_stage2(m33, s2d_loss, batch, device=device)
            self.assertTrue(np.isfinite(m2d["loss_total"]))
            self.assertEqual(m2d["l_pos_gate"], 0.0)
            self.assertEqual(m2d["l_ret_sft"], 0.0)
            for p in m33.base.parameters():
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
            for mod_name in ("deg_encoder", "photo_net", "pyramid_sft", "freq_head", "output_composer"):
                gnorm = sum(float(p.grad.abs().sum().item()) for p in m33.v3[mod_name].parameters() if p.grad is not None)
                self.assertGreater(gnorm, 0.0, f"Stage-2D expected non-zero grad for {mod_name}")

            # 4. Stage-3D 1-step forward/backward with lazy R1 & exact frozen boundary
            configure_stage_trainability(m33, "3d")
            m33.zero_grad(set_to_none=True)
            disc = PatchGANDiscriminator(base_ch=16)
            opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=0.0)
            gan_helper = Stage3GANHelper(disc, lambda_gan=0.03, r1_gamma=10.0, r1_every=16)
            s3d_loss = Stage2Loss(
                lpips_model=None,
                arcface_model=None,
                device=device,
                w_charb=0.25,
                w_ssim=0.08,
                w_chroma=0.15,
                w_lpips=0.85,
                w_vgg_gram=0.25,
                w_hf_energy=0.04,
                w_arcface=0.60,
                w_lap=0.03,
                w_fft=0.10,
                w_aux512=0.06,
                w_wmap=0.30,
                w_ret_code=0.0,
                w_ret_sft=0.0,
                w_ret_photo=0.0,
                w_ret_deg=0.0,
                w_adain_floor=0.10,
                w_pos_gate=0.0,
                w_clear_l1=1.5,
                v33_mode=True,
            )
            m3d = train_step_stage3(
                m33, s3d_loss, gan_helper, opt_d, batch, device=device, step=16, do_d_step=True, lambda_scale=0.5
            )
            self.assertTrue(np.isfinite(m3d["loss_total"]))
            self.assertAlmostEqual(m3d["gan_lambda_eff"], 0.015, places=6)
            self.assertGreater(m3d["d_r1"], 0.0)
            for p in m33.v3["photo_net"].parameters():
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
            for mod_name in ("pyramid_sft", "freq_head", "output_composer"):
                gnorm = sum(float(p.grad.abs().sum().item()) for p in m33.v3[mod_name].parameters() if p.grad is not None)
                self.assertGreater(gnorm, 0.0, f"Stage-3D expected non-zero grad for {mod_name}")

            # 5. Soup Pareto selector prioritizes native CCTV same_id_arc >= 0.650
            base_pf = {
                "overall": {"psnr_db": 29.0, "ssim7": 0.88, "lpips": 0.115, "arcface_cos": 0.890, "central_laplacian_ratio": 0.95, "catastrophic_id_flips": 0},
                "by_role": {},
            }
            nat_good = {"same_id_arcface_cos_mean": 0.662, "png_floor": {"overall": {"central_laplacian": 0.084}}}
            nat_bad = {"same_id_arcface_cos_mean": 0.590, "png_floor": {"overall": {"central_laplacian": 0.084}}}
            self.assertGreater(
                _pareto_eval_score_v31(base_pf, native_summary=nat_good),
                _pareto_eval_score_v31(base_pf, native_summary=nat_bad),
            )

            # 6. --app-test-dir eval smoke check with V3.3 .npz auto-detection + 2x eye zoom contact sheet
            v33_npz = td_path / "v33_ema.npz"
            export_v3_npz(m33, v33_npz)
            app_dir = td_path / "app_test"
            (app_dir / "inputs").mkdir(parents=True)
            (app_dir / "gt").mkdir(parents=True)
            (app_dir / "observed").mkdir(parents=True)
            gt_u8 = _make_dummy_face_u8(101, size=256)
            in_u8 = np.clip(gt_u8.astype(np.int32) - 15, 0, 255).astype(np.uint8)
            obs_u8 = np.full((256, 256), 255, dtype=np.uint8)
            Image.fromarray(in_u8).save(app_dir / "inputs/face1.png")
            Image.fromarray(gt_u8).save(app_dir / "gt/face1.png")
            Image.fromarray(obs_u8).save(app_dir / "observed/face1.png")
            Image.fromarray(in_u8).save(app_dir / "inputs/wild3.png")
            Image.fromarray(obs_u8).save(app_dir / "observed/wild3.png")
            (app_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "face_1__direct",
                                "role": "with_gt_direct",
                                "route": "direct",
                                "has_gt": True,
                                "input": "inputs/face1.png",
                                "gt": "gt/face1.png",
                                "observed": "observed/face1.png",
                            },
                            {
                                "case_id": "wild3__step2",
                                "role": "without_gt_step2",
                                "route": "picture_region_gaussian",
                                "has_gt": False,
                                "input": "inputs/wild3.png",
                                "gt": None,
                                "observed": "observed/wild3.png",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            eval_out = td_path / "eval_app"
            eval_main([
                "--out-dir", str(eval_out),
                "--allow-random-base",
                "--codeformer-path", str(td_path / "missing.pth"),
                "--v3-npz", str(v33_npz),
                "--app-test-dir", str(app_dir),
                "--cohorts", "app_test_images",
                "--models", "v3",
                "--no-lpips",
                "--no-arcface",
                "--device", "cpu",
            ])
            summ = json.loads((eval_out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("app_test_images", summ)
            ov_app = summ["app_test_images"]["v3"]["png_floor"]["overall"]
            self.assertEqual(ov_app["count"], 2)
            self.assertIsNotNone(ov_app["mae"])
            self.assertIsNotNone(ov_app["central_laplacian"])
            self.assertIsNotNone(ov_app["noise_sigma"])
            sheet_path = eval_out / "contact_sheets" / "app_test_images.png"
            self.assertTrue(sheet_path.is_file())
            with Image.open(sheet_path) as sim:
                # 2 cases x (256 full + 256 2x-eye-zoom) + 28px header = 1052px height; 3 cols (Input, v3, GT) = 768px width
                self.assertEqual(sim.size, (768, 1052))

    def test_v34_degradation_sampler_stage1v4_stage2v4_and_app_test_eval(self):
        torch.manual_seed(34)
        img512 = _make_dummy_face_u8(20260340, size=512)

        # 1. calibration="v34": NN pixelation + coarse confetti + pure blur x_struct_clean == x_in + clear MSE==0 + 600-call fuzz
        nn_pix_hits = {"severe_lr_survface": 0, "h264_macroblock_interlace": 0, "compound_cctv_codec": 0}
        coarse_confetti_hits = {"daylight_sensor_noise": 0, "lowlight_sensor_noise": 0}
        found_clear_exact = False
        total_v34 = 0
        for r_idx, reg in enumerate(REGIMES):
            for i in range(50):
                geom = (i % 2) == 1
                inp, gt256, gt512, obs, meta = generate_cctv_pair(
                    img512,
                    regime=reg,
                    seed=340000 + r_idx * 100 + i,
                    apply_geometry=geom,
                    calibration="v34",
                )
                total_v34 += 1
                self.assertEqual(inp.shape, (256, 256, 3))
                self.assertEqual(gt256.shape, (256, 256, 3))
                self.assertEqual(gt512.shape, (512, 512, 3))
                self.assertEqual(obs.shape, (256, 256))
                self.assertIn("x_struct_clean_256", meta)
                self.assertIn("x_photo_tgt_256", meta)
                sc256 = meta["x_struct_clean_256"]
                ph256 = meta["x_photo_tgt_256"]
                self.assertEqual(sc256.shape, (256, 256, 3))
                self.assertEqual(ph256.shape, (256, 256, 3))
                if not obs.all():
                    self.assertTrue((gt256[~obs] == 128).all())
                    self.assertTrue((sc256[~obs] == 128).all())
                    self.assertTrue((ph256[~obs] == 128).all())
                for k in ("exposure", "blur", "noise", "codec_lr", "grid", "severity", "sufficiency"):
                    self.assertTrue(np.isfinite(meta[k]))
                    self.assertGreaterEqual(meta[k], 0.0)
                    self.assertLessEqual(meta[k], 1.0)
                if reg in ("global_defocus_blur", "directional_motion_blur", "local_partial_blur"):
                    np.testing.assert_array_equal(sc256[obs], inp[obs])
                    np.testing.assert_array_equal(ph256[obs], inp[obs])
                    self.assertEqual(meta["exposure"], 0.0)
                    self.assertEqual(meta["noise"], 0.0)
                    self.assertEqual(meta["grid"], 0.0)
                if reg == "clear" and np.array_equal(inp, gt256):
                    found_clear_exact = True
                if reg in nn_pix_hits and meta.get("is_nn_pixelated"):
                    nn_pix_hits[reg] += 1
                if reg in coarse_confetti_hits and meta.get("is_coarse_confetti"):
                    coarse_confetti_hits[reg] += 1

        self.assertEqual(total_v34, 600)
        self.assertTrue(found_clear_exact, "Expected clear MSE == 0.0 in calibration='v34'")
        for reg, cnt in nn_pix_hits.items():
            self.assertGreater(cnt, 0, f"Expected NN pixelation hits in v34 {reg}")
        for reg, cnt in coarse_confetti_hits.items():
            self.assertGreater(cnt, 0, f"Expected 2-4px coarse confetti hits in v34 {reg}")

        # 2. StratifiedRoleBatchSampler calibration="v34": 12.5% clear, 25% lowlight_joint, 62.5% normal_degraded (2x priority)
        sampler_v34 = StratifiedRoleBatchSampler(num_samples=4000, batch_size=16, seed=34, calibration="v34")
        self.assertEqual(sampler_v34.n_clear, 2)
        self.assertEqual(sampler_v34.n_lowlight, 4)
        self.assertEqual(sampler_v34.n_normal, 10)
        counts_v34 = {r: 0 for r in REGIME_GROUPS["normal_degraded"]}
        for bspec in sampler_v34:
            for _, r in bspec:
                if r in counts_v34:
                    counts_v34[r] += 1
        prio_avg = np.mean([counts_v34[r] for r in V34_PRIORITY_NORMAL_REGIMES])
        other_avg = np.mean([counts_v34[r] for r in counts_v34 if r not in V34_PRIORITY_NORMAL_REGIMES])
        self.assertGreater(prio_avg / max(other_avg, 1.0), 1.6)

        # 3. Stage 1v4 & Stage 2v4 1-step forward/backward with bf16 + grad_checkpoint=True
        m34 = CodeFormerV3Restorer(codeformer_weights_path=None, grad_checkpoint=True, arch_variant="v34")
        device = torch.device("cpu")
        batch = _make_synthetic_batch(batch_size=2, has_pad=True)
        batch["x_struct_clean_256"] = batch["gt_256"].clone()
        batch["x_photo_tgt_256"] = batch["gt_256"].clone()
        batch["regime"] = ["clear", "severe_lr_survface"]
        batch["is_nn_pixelated"] = torch.tensor([False, True], dtype=torch.bool)
        batch["landmarks"] = (
            torch.from_numpy((FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32))
            .unsqueeze(0)
            .expand(2, -1, -1)
            .clone()
        )

        # Stage 1v4 LR schedule (3e-4 -> 3e-5) + 1-step forward/backward under CPU bf16 autocast
        configure_stage_trainability(m34, "1v4")
        opt_1v4, sched_1v4 = build_optimizer_and_scheduler(m34, "1v4", max_steps=100, warmup_steps=10)
        self.assertEqual(len(opt_1v4.param_groups), 1)
        self.assertAlmostEqual(opt_1v4.param_groups[0]["initial_lr"], 3.0e-4, places=10)
        for _ in range(100):
            opt_1v4.step()
            sched_1v4.step()
        self.assertAlmostEqual(opt_1v4.param_groups[0]["lr"], 3.0e-5, places=8)

        m34.train(True)
        m34.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            m1v4 = train_step_stage1_v34(m34, V34Stage1Loss(), batch, device=device, bf16=False)
        for key in ("loss_total", "l_struct_clean", "l_photo", "l_id_lock", "l_deg_aux", "l_clean_bce", "l_code"):
            self.assertIn(key, m1v4)
            self.assertTrue(np.isfinite(m1v4[key]), f"Non-finite 1v4 metric {key}")
        for p in m34.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for mod_name in ("deg_encoder", "photo_net", "pre_net"):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in m34.v3[mod_name].parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Stage-1v4 expected non-zero grad for {mod_name}")

        # Stage 2v4 LR schedule (1.2e-4 -> 1.2e-5, 2.0e-4 -> 2.0e-5) + 1-step forward/backward under bf16
        configure_stage_trainability(m34, "2v4")
        opt_2v4, sched_2v4 = build_optimizer_and_scheduler(m34, "2v4", max_steps=100, warmup_steps=10)
        self.assertEqual(len(opt_2v4.param_groups), 2)
        self.assertAlmostEqual(opt_2v4.param_groups[0]["initial_lr"], 1.2e-4, places=10)
        self.assertAlmostEqual(opt_2v4.param_groups[1]["initial_lr"], 2.0e-4, places=10)
        for _ in range(100):
            opt_2v4.step()
            sched_2v4.step()
        self.assertAlmostEqual(opt_2v4.param_groups[0]["lr"], 1.2e-5, places=8)
        self.assertAlmostEqual(opt_2v4.param_groups[1]["lr"], 2.0e-5, places=8)

        m34.zero_grad(set_to_none=True)
        s2v4_loss = V34Stage2Loss(lpips_model=None, arcface_model=None, device=device)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            m2v4 = train_step_stage2_v34(m34, s2v4_loss, batch, device=device, bf16=False)
        for key in ("loss_total", "l_charb", "l_chroma", "l_struct_clean", "l_code", "l_id_lock", "l_clean_bce", "l_clear_l1", "l_w_reg"):
            self.assertIn(key, m2v4)
            self.assertTrue(np.isfinite(m2v4[key]), f"Non-finite 2v4 metric {key}")
        for p in m34.base.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for mod_name in ("deg_encoder", "photo_net", "pre_net"):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in m34.v3[mod_name].parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Stage-2v4 expected non-zero grad for {mod_name}")

        # 4. --app-test-dir eval smoke check with V3.4 .npz auto-detection
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            v34_npz = td_path / "v34_ema.npz"
            export_v3_npz(m34, v34_npz)
            app_dir = td_path / "app_test"
            (app_dir / "inputs").mkdir(parents=True)
            (app_dir / "gt").mkdir(parents=True)
            (app_dir / "observed").mkdir(parents=True)
            gt_u8 = _make_dummy_face_u8(104, size=256)
            in_u8 = np.clip(gt_u8.astype(np.int32) - 12, 0, 255).astype(np.uint8)
            obs_u8 = np.full((256, 256), 255, dtype=np.uint8)
            Image.fromarray(in_u8).save(app_dir / "inputs/face4.png")
            Image.fromarray(gt_u8).save(app_dir / "gt/face4.png")
            Image.fromarray(obs_u8).save(app_dir / "observed/face4.png")
            (app_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "face_4__direct",
                                "role": "with_gt_direct",
                                "route": "direct",
                                "has_gt": True,
                                "input": "inputs/face4.png",
                                "gt": "gt/face4.png",
                                "observed": "observed/face4.png",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            eval_out = td_path / "eval_v34_app"
            eval_main([
                "--out-dir", str(eval_out),
                "--allow-random-base",
                "--codeformer-path", str(td_path / "missing.pth"),
                "--v3-npz", str(v34_npz),
                "--app-test-dir", str(app_dir),
                "--cohorts", "app_test_images",
                "--models", "v3",
                "--no-lpips",
                "--no-arcface",
                "--device", "cpu",
            ])
            summ = json.loads((eval_out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("app_test_images", summ)
            ov_app = summ["app_test_images"]["v3"]["png_floor"]["overall"]
            self.assertEqual(ov_app["count"], 1)
            self.assertIsNotNone(ov_app["mae"])
            self.assertTrue((eval_out / "contact_sheets" / "app_test_images.png").is_file())


if __name__ == "__main__":
    unittest.main()
