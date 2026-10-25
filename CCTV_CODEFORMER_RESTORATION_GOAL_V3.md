# Active Goal V3: Multi-Scale Dilated SFT Distillation & Frequency-Selective VQGAN Decoder Fusion (`Conditioned SFT V2`)

Created: 25 October 2026
Workspace: `C:\xampp\htdocs\YEAR 4\Testing_RESTORATION` (Branch: `RESTORATION`, auto-synced to `main` in `C:\xampp\htdocs\YEAR 4\Testing`)
Predecessor Goals: [CCTV_CODEFORMER_RESTORATION_GOAL_V2.md](CCTV_CODEFORMER_RESTORATION_GOAL_V2.md) and [CCTV_CODEFORMER_U96_ACTIVE_GOAL_V1.md](CCTV_CODEFORMER_U96_ACTIVE_GOAL_V1.md)

---

## 1. Ready-to-Run `/goal` Command Prompt

Trigger autonomous execution in chat by pasting:

```text
/goal Execute CCTV_CODEFORMER_RESTORATION_GOAL_V3.md end-to-end in C:\xampp\htdocs\YEAR 4\Testing_RESTORATION (auto-syncing commits and outputs to main in C:\xampp\htdocs\YEAR 4\Testing) to achieve unmistakable, forensically faithful visual sharpness improvements (eyes, eyebrows, nose, lips, jaw) on blurred, compressed, and low-light 256x256 CCTV faces over U96, u64, and Conditioned SFT V1 B8 while strictly preserving clear/dark photometry and facial identity: (1) run a local CPU frequency-selective VQGAN decoder + SFT diagnostic proof quantifying visible edge/SSIM headroom before training, (2) build the 56-identity multi-degradation CCTV TRAIN dataset (defocus, anisotropic motion blur, LR48-64 + JPEG40-70, lowlight+sensor noise) and unit-tested Conditioned SFT V2 package (515 R2 frozen, exact step-0 u64 parity, zero-init multi-scale dilated 1/2/4 pre-SFT blocks at 32/64/128/256, degradation-gated high-pass VQGAN decoder edge fusion, exposure-decoupled gradient routing, and Two-Phase Clean-SFT + Frozen-R2 Perceptual/SSIM/Laplacian training), (3) seal and independently audit the manual NVIDIA L4 VM packet and 5-step gcloud/tmux guide, (4) upon L4 return, audit all checkpoints, paired monitor/DEV metrics (MAE, SSIM, Laplacian edge sharpness), and unpaired native CCTV (QMUL-SurvFace, ChokePoint) comparison sheets, and (5) ship an interactive browser upload & side-by-side multi-model comparison UI verified via inline Playwright at 375px, 768px, and 1280px.
```

---

## 2. Root-Cause Diagnosis: Why V1 (`B8`) Solved Clear Drift Yet Stayed Visually Soft

In [CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_RETURN_REVIEW.md](CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_RETURN_REVIEW.md), `Conditioned SFT V1` (`Arm B u0008`) eliminated the `Matched V1` clear-PNG regression (`-0.000273` MAE) and passed `96/96` gates with coarse dark-drift alignment, yet visual inspection of [returned_l4_multi_role_sheet_v1.png](outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/visuals/returned_l4_multi_role_sheet_v1.png) shows blurred/compound faces remain noticeably soft (`-0.00039` PNG MAE, `~0.1` RGB8 gray levels) compared to the Clean-SFT Oracle's `-0.0195` to `-0.0317` PNG MAE jump (`SSIM 0.802 -> 0.916` and `0.686 -> 0.919` in [CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md](CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md)):

1. **Receptive-Field Bottleneck (`3x3` vs `15–25px` Blur)**: `U96`'s `ResidualSkipAdapter` uses a single `3x3` depthwise convolution (`51,328` parameters across all 4 sites), which cannot deconvolve `gaussian_2p0`/`2p4` or CCTV motion/downscale blur spanning `15–25px` at `512x512`.
2. **Fixed `w=1.0` SFT Fidelity Suppresses Crisp VQGAN Decoder Edges**: Inside `Fuse_sft_block`, `w=1.0` overwrites crisp VQGAN codebook `decoder` features with blurry `enc_feat`, while `SpatialHead` (`fused + tanh(logits) * (fused - decoder)`) stayed pinned near `0`.
3. **Micro-Step Displacement Cap (`LR=1e-5`, `L_inf <= 6.5e-5`) & Pure Pixel-L1 Smoothing**: Eight Adam updates at `LR=1e-5` moved weights by at most `6.5e-5`, and pixel-space `RGB_L1` penalizes crisp facial edges under sub-pixel uncertainty, favoring smooth averages.
4. **Cross-Exposure Gradient Conflict**: Projecting `lowlight0p35_noise0p006` gradients onto normal-lit `clear` gradients inverted coarse dark luminance and halted training at `u0008`.
5. **Narrow Training Pool (`24` Identities / `5` Profiles)**: Only `24` of the `64` root-admitted clean `TRAIN` identities were used for optimization, without anisotropic motion blur or harsher `LR48` + `JPEG40` CCTV compression.

---

## 3. V3 Model & Training Architecture (`codeformer_conditioned_sft_learning_v2`)

### A. Frozen Base Ownership & Exact Step-0 Parity
1. **Immutable Predecessors**: Retain `R2`, `U96`, `u64`, `Matched V1`, and `Conditioned SFT V1` (`A_u0008`, `B_u0008`) untouched.
2. **Frozen 515-Tensor `R2` Base + Step-0 `u64` Parity**: Keep all `515` `R2` base tensors frozen (`a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f`) and zero-initialize all new residual branches so step `0` reproduces `u64` bitwise.

### B. Multi-Scale Dilated Pre-SFT Pyramid + Frequency-Selective VQGAN Decoder Fusion
1. **Multi-Scale Dilated Pre-SFT Residual Pyramid (`32, 64, 128, 256`)**:
   - Upgrade pre-SFT adaptation at each site with parallel dilated depthwise-separable branches (`dilation in {1, 2, 4}`, channel width `64`, Squeeze-and-Excitation channel gating, zero-initialized output projection) gated by `g_deg(x)`, expanding effective receptive field to `~21x21` pixels per block to invert wide defocus, motion, and downsampling blur before `Fuse_sft_block`.
2. **Degradation-Gated Frequency-Selective VQGAN Decoder Edge Injection**:
   - Decompose post-fusion `fused` and VQGAN codebook `decoder` features into low-frequency structure/tone `LP(.)` (`5x5` Gaussian/box filter) and high-frequency detail `HP(z) = z - LP(z)`.
   - Inject spatial-attention-weighted high-frequency codebook structure strictly gated by input degradation `g_deg(x)`:
     $$\text{corrected}_s = \text{u64\_head}_s + g_{\text{deg}}(x) \cdot \left( \sigma(G_s) \odot \left(\text{HP}(\text{decoder}_s) - \text{HP}(\text{fused}_s)\right) + R_s(\text{used}_s, \text{decoder}_s, \text{fused}_s, \text{coarse}_{32}) \right)$$
   - Guarantees: low-frequency facial proportions, gaze, expression, and skin tone stay locked to the observed input (`0` identity hallucination), sharp `clear` inputs (`g_deg(x) -> 0`) suffer zero degradation, and blurred/compressed faces recover crisp eye, eyebrow, eyelash, nostril, lip, and jaw edges.
3. **Exposure-Decoupled Conditioning & Coarse Drift Protection**:
   - Partition normal-lit (`g_dark(x) == 0`) and low-light (`g_dark(x) > 0`) coarse `16x16` tone paths inside the forward graph and gradient router so low-light brightening and normal-lit tone preservation never oppose one another.

### C. Two-Phase Unconstrained-Quality L4 GPU Training & Perceptual/Structural Objective
1. **Expanded `56`-Identity Realistic CCTV `TRAIN` Suite (`8` Held-Out Monitors + `56` Inherited Quality Cases Untouched)**:
   - Use all `56` non-monitor root-admitted clean `TRAIN` FFHQ photographs (`root_clean_target_review_v1.json`) across **6 realistic CCTV profiles**:
     - `clear` (sharp photometric & identity anchor)
     - `global_defocus_blur` (`sigma in [1.8, 2.6]`)
     - `directional_motion_blur` (linear/anisotropic CCTV motion kernel `7–13px`)
     - `local_soft_surroundings` (mixed sharp/soft facial regions)
     - `compound_cctv_codec` (downsample `48x48–64x64` + blur + `JPEG 40–70`)
     - `lowlight_sensor_noise` (`0.30–0.40x` linear exposure + shot/read noise)
2. **Phase 1 — Multi-Epoch Clean-SFT Feature Distillation (`LR = 3e-4` cosine, fast encoder-only updates)**:
   - Pre-extract or compute detached frozen-`R2` clean target encoder features `(32, 64, 128, 256)` for all `56` clean `TRAIN` targets and train the multi-scale dilated pre-SFT pyramid for `15–20` epochs with `L1 + Cosine + Spatial-Gradient` feature alignment to capture the Clean-SFT Oracle's sharpness headroom rapidly without decoder backprop overhead.
3. **Phase 2 — End-to-End Perceptual + Structural Joint Optimization (`LR = 1e-4 -> 1e-5` cosine)**:
   - Jointly optimize the dilated pre-SFT pyramid + frequency-selective `SpatialHead` under:
     $$L_{\text{total}} = L_{\text{spatial-RGB}} + \lambda_{\text{perc}} L_{\text{R2-enc-perc}} + \lambda_{\text{SSIM}} (1 - \text{SSIM}_7) + \lambda_{\text{Lap}} L_{\text{multi-scale-Laplacian}} + \lambda_{\text{SFT}} L_{\text{SFT}} + \lambda_{\text{coarse}}(g_{\text{dark}}) L_{\text{block16}}$$
   - Where $L_{\text{R2-enc-perc}}$ measures multi-scale feature distance through the **frozen `R2` encoder** on the restored output vs. clean target (domain-matched facial perceptual loss requiring zero external downloads), paired with exposure-split gradient conflict projection (`normal` roles projected against `clear`; `lowlight` projected only on zero-mean spatial structure, never coarse exposure).
4. **Acceptance & Gate Policy**:
   - Strict non-regression on `clear` and `dark` cohorts (`RAW` and `native_png`), while lifting artificial micro-displacement caps on degraded inputs so blurred/compound faces achieve **substantially higher SSIM, lower MAE, higher Laplacian edge energy, and unmistakable visual facial clarity** on side-by-side `256x256` sheets.

---

## 4. Five-Stage Execution Plan

1. **Stage 1 — Local CPU Frequency-Fusion & Oracle Headroom Proof**:
   - Test degradation-gated high-pass VQGAN `decoder` fusion on retained `u64`/`B8` cases locally first, generating side-by-side visual sheets and MAE/SSIM/edge metrics to verify visible facial sharpening before writing the VM training packet.
2. **Stage 2 — `56`-Identity CCTV Dataset Preparation, `v2` Package & Pure/CPU Unit Tests**:
   - Build deterministic `56`-identity × `6`-profile `TRAIN` assets, implement `research/codeformer_conditioned_sft_learning_v2/` and `research/codeformer_conditioned_sft_inference_v2.py`, and pass `100%` of unit tests.
3. **Stage 3 — Sealed Manual NVIDIA L4 VM Packet & 5-Step Launch Guide**:
   - Seal the two-phase L4 training archive, independent packet audit, and 5-step Windows Cloud SDK + `tmux` launch guide (`CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_VM.md`).
4. **Stage 4 — L4 Return Audit, Broader `DEV` & Native CCTV (`SurvFace`, `ChokePoint`) Visual Verification**:
   - Audit returned checkpoints, compute `MAE` / `SSIM` / `Laplacian sharpness` across held-out monitors + broader `DEV`, and render full-resolution side-by-side sheets (`Input | U96 | u64 | V1-B8 | V2 | Clean Target`) plus unpaired native CCTV sheets.
5. **Stage 5 — Interactive Browser Live Upload & Multi-Model Comparison UI + Playwright Audit**:
   - Deliver a local FastAPI review & live-upload restoration comparison app (`Original` vs `U96` vs `u64` vs `V1-B8` vs `V2`, with PNG/JSON evidence export) and verify layout/flows via inline Playwright at `375px`, `768px`, and `1280px`.
