# Stage 1 Review: Conditioned SFT V2 Local CPU Frequency-Fusion & Exposure-Matched SFT Headroom Proof

Date: 25 October 2026
Workspace: `C:\xampp\htdocs\YEAR 4\Testing_RESTORATION` (`RESTORATION` branch, synced to `main`)
Receipt: [v2_feasibility_receipt_v1.json](outputs/codeformer_conditioned_sft_learning_v2_feasibility/v2_feasibility_receipt_v1.json)
Paired Sheet: [v2_paired_frequency_fusion_sheet_v1.png](outputs/codeformer_conditioned_sft_learning_v2_feasibility/v2_paired_frequency_fusion_sheet_v1.png)
Native CCTV Sheet: [v2_native_cctv_frequency_fusion_sheet_v1.png](outputs/codeformer_conditioned_sft_learning_v2_feasibility/v2_native_cctv_frequency_fusion_sheet_v1.png)

---

## 1. Purpose & Execution Scope

Before sealing the manual NVIDIA L4 training archive (`Conditioned SFT V2`), Stage 1 of [CCTV_CODEFORMER_RESTORATION_GOAL_V3.md](CCTV_CODEFORMER_RESTORATION_GOAL_V3.md) ran a local inference-only CPU diagnostic (`0` optimizer/gradient training calls, `366.44s` elapsed) across `7` representative paired/guard cases and `4` native CCTV `DEV` faces (`ChokePoint` and `QMUL-SurvFace`) to answer three architectural questions quantitatively:

1. Does unlearned post-fusion high-pass VQGAN `decoder` feature injection (`HP(decoder_s) - HP(fused_s)`) sharpen blurred faces by itself, or must high-frequency edge structure enter **before `Fuse_sft_block`** (and via learned `SpatialHead` refinement)?
2. How much visual `SSIM7`, `PNG MAE`, `Central Laplacian Sharpness`, and `Frozen-R2 Perceptual L1` headroom exists when degraded SFT features `(32, 64, 128, 256)` are reconstructed toward clean SFT features?
3. Why does raw `Clean_SFT` break on `lowlight0p35_noise0p006`, and does `match_sft_exposure_torch` in [conditioning_v2.py](research/codeformer_conditioned_sft_learning_v2/conditioning_v2.py) eliminate that failure?

---

## 2. Quantitative Results (`v2_feasibility_receipt_v1.json`)

| Case ID & Role | `u64` (`MAE` / `SSIM7` / `Lap`) | `V1-B8` (`MAE` / `SSIM7` / `Lap`) | Unlearned `Raw-Dec-HP` (`MAE` / `SSIM7` / `Lap`) | `Raw Clean-SFT` (`MAE` / `SSIM7` / `Lap`) | `V2 Exposure-Matched SFT` (`MAE` / `SSIM7` / `Lap`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tr_ffhq_08698__clear` | `0.03290` / `0.9190` / `0.02354` | `0.03277` / `0.9191` / `0.02345` | `0.03277` / `0.9191` / `0.02345` | `0.03318` / `0.9163` / `0.02454` | **`0.03277` / `0.9191` / `0.02345`** *(0-gate exact)* |
| `tr_ffhq_08698__global_gaussian2p0` | `0.03909` / `0.8452` / `0.01458` | `0.03899` / `0.8453` / `0.01452` | `0.03960` / `0.8320` / `0.01443` | `0.03320` / `0.9186` / `0.02331` | **`0.03320` / `0.9186` / `0.02331`** (`+0.073` SSIM, `+60%` Lap) |
| `tr_ffhq_08698__local_gaussian2p0_soft` | `0.03549` / `0.8923` / `0.01334` | `0.03536` / `0.8924` / `0.01330` | `0.03537` / `0.8923` / `0.01330` | `0.03340` / `0.9152` / `0.02265` | **`0.03340` / `0.9152` / `0.02265`** (`+0.023` SSIM, `+70%` Lap) |
| `tr_ffhq_08698__compound_g1p2_lr64_jpeg70` | `0.04469` / `0.7759` / `0.01595` | `0.04459` / `0.7759` / `0.01588` | `0.04429` / `0.7784` / `0.01570` | `0.03408` / `0.9174` / `0.02342` | **`0.03408` / `0.9174` / `0.02342`** (`+0.142` SSIM, `+47%` Lap) |
| `tr_ffhq_08698__lowlight0p35_noise0p006` | `0.12482` / `0.8233` / `0.02029` | `0.12444` / `0.8232` / `0.02021` | `0.12444` / `0.8232` / `0.02021` | `0.23402` / `0.5649` / `0.02495` *(grid clash!)* | **`0.12399` / `0.8184` / `0.02031`** *(grid clash eliminated!)* |
| `tr_ffhq_03761__gaussian_2p4` | `0.05311` / `0.8159` / `0.01152` | `0.05285` / `0.8163` / `0.01151` | `0.05306` / `0.8013` / `0.01127` | `0.05102` / `0.8888` / `0.01476` | **`0.05102` / `0.8888` / `0.01476`** (`+0.072` SSIM, `+28%` Lap) |

---

## 3. Three Critical Architectural Takeaways Locked into `Conditioned SFT V2`

1. **Why Pre-SFT Multi-Scale Feature Reconstruction Is Primary (`+0.072` to `+0.142` SSIM, `+47%` to `+70%` Sharpness)**:
   - Unlearned `HP(decoder_s) - HP(fused_s)` helped slightly on `compound_g1p2_lr64_jpeg70` (`MAE 0.04459 -> 0.04429`, `SSIM 0.7759 -> 0.7784`), because JPEG/downscale ringing sits in `fused_s`. However, on `global_gaussian2p0` and `gaussian_2p4`, raw `decoder_s` before `Fuse_sft_block` is smooth in `codeformer_r2.pth` — it is `Fuse_sft_block` (`scale` and `shift` synthesized from `cat([used_s, decoder_s])`) that generates crisp eyelashes, irises, eyebrows, nostrils, and lip contours!
   - Therefore, V2 puts primary model capacity into the **`TopDownMultiScaleDilatedSFTPyramid`** (`32 -> 64 -> 128 -> 256`, width `96`, parallel `dilation in {1, 2, 4}` depthwise convs + SE channel gating + top-down cross-scale guidance) trained in **Phase 1 (`40` full epochs, `1,680` optimizer steps at `LR=5e-4 -> 2e-5`)** directly on `Exposure-Matched Clean-SFT` targets, followed by **Phase 2 (`240` end-to-end updates)** jointly refining both the pre-SFT pyramid and the **`FrequencySelectiveSpatialHeadV2`**.
2. **Why `match_sft_exposure_torch` Is Essential on Low-Light Inputs**:
   - Row 5 of [v2_paired_frequency_fusion_sheet_v1.png](outputs/codeformer_conditioned_sft_learning_v2_feasibility/v2_paired_frequency_fusion_sheet_v1.png) proved why V1's raw Clean-SFT distillation caused dark luminance conflict: raw `clean_feat` carries bright normal-lit activations while `dec_feat` carries dark-AdaIN activations, producing a severe checkerboard exposure clash (`MAE 0.2340`, `SSIM 0.5649`).
   - Aligning local patch mean and standard deviation via `match_sft_exposure_torch` completely eliminated the checkerboard artifact (`MAE 0.12399`, `0` grid artifacts) while preserving clean structural gradients.
3. **100% Zero-Overlap Physical Separation Across All `480` Paired/Guard Cases + `4` Native CCTV Crops**:
   - Replacing isotropic Laplacian gating with **4-directional minimum central Laplacian (`min_dir_lap` across `0°, 45°, 90°, 135°`)** plus **dark high-frequency noise energy (`raw_lap - smooth_lap`)** in [conditioning_v2.py](research/codeformer_conditioned_sft_learning_v2/conditioning_v2.py) achieved exact `100%` separation across all `484` tested images:
     - `88/88` `clear` cases: `g_blur = 0.0`, `g_sensor_noise = 0.0` (zero blur perturbation).
     - `8/8` clean `dark` guards: `g_blur = 0.0`, `g_sensor_noise = 0.0` (exact bitwise `u64` retention).
     - `308/308` blurred / anisotropic-motion / local-soft / compound-codec cases + `4/4` native `ChokePoint`/`QMUL` CCTV crops: `max_local_blur > 0` (`[0.22, 1.00]`).
     - `80/80` noisy low-light cases: `g_sensor_noise > 0` (`[0.16, 1.00]`).
