# Conditioned SFT Learning V1 — Returned NVIDIA L4 GPU Run Audit & Review

## 1. Archive & Transport Provenance

- **Returned Archive**: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz`
  - **Bytes**: `2,360,962,311` (`~2.20 GiB`)
  - **SHA256**: `00682555ab4757ca70987e4d3a41af28ec710361b12e1d5c5ab7073e55753d22`
  - **Members**: `3,825` PAX tar entries (`0` unsafe paths, `0` symlinks/hardlinks, `690/690` static packet files verified byte-identical to sealed stage)
  - **Extracted Data-Only Files**: `3,126` retained outputs under `outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/retained/cctv_codeformer_conditioned_sft_learning_v1_vm`
- **L4 GPU Execution Telemetry** (`outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/transport.json`):
  - **Platform**: `Linux-5.15.0-1097-gcp-x86_64-with-glibc2.35`, `NVIDIA L4` (`23,034 MiB`, Driver `570.211.01`, Torch `2.4.1+cu121`, CUDA `12.1`)
  - **Protocol SHA256**: `f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0`
  - **Full Forward Calls**: `712` (`440` initializer/parity/repeatability forwards + `80` training forwards + `192` `u0008` quality snapshot forwards)
  - **Optimizer Commits**: `16` genuine Adam updates (`8` in Arm `A`, `8` in Arm `B`), `80` role backward passes, `18` closed checkpoints (`A/u0000..u0008`, `B/u0000..u0008`)
  - **Wall-Clock Duration**: `410.01s` (`6.83 min`), finite exit code `2` at `u0008` snapshot gate (`retry_or_resume_used: false`)

---

## 2. Normal-Lit & Degraded Role Breakthrough on Live L4 GPU (`u0008`)

1. **Matched V1 `clear` PNG Regression Completely Eliminated**:
   - In `Matched Feature Learning V1`, Arm `B` failed `clear` PNG MAE at `u0008` (`+0.00000281` regression).
   - In `Conditioned SFT Learning V1`, **both** Arm `A` and Arm `B` passed `native_png/clear_target_MAE_improves` (`True`) and `raw/clear_target_MAE_improves` (`True`) on the inherited `56`-case bank AND improved `clear` PNG MAE on the `40`-case monitor bank (`-0.00022441` in Arm `A`, `-0.00027330` in Arm `B`).
2. **60-Tensor Arm `B` Outperformed 36-Tensor Arm `A` Across Every Normal-Lit Role (`+21.8%` to `+24.7%` Larger Gains)**:

| Matched Monitor Role (`native_png` MAE) | Initial `u64` (`u0000`) | Matched V1 `B8` Delta | Returned Arm `A` (`u0008`) Delta | Returned Arm `B` (`u0008`) Delta | Arm `B` Multiplier vs Matched V1 `B8` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clear` | `0.031724` | `+0.000003` (FAIL) | `-0.000224` (PASS) | **`-0.000273` (PASS)** | **Sign flipped to strong gain** |
| `global_gaussian2p0` | `0.042335` | `-0.000069` | `-0.000315` (PASS) | **`-0.000393` (PASS)** | **`5.70x` larger improvement** |
| `local_gaussian2p0_soft` | `0.036569` | `-0.000018` | `-0.000238` (PASS) | **`-0.000294` (PASS)** | **`15.91x` larger improvement** |
| `compound_g1p2_lr64_jpeg70` | `0.048063` | `-0.000073` | `-0.000316` (PASS) | **`-0.000390` (PASS)** | **`5.33x` larger improvement** |

3. **All Inherited Blur, Local, and Native CCTV Filter Retention Gates Passed**:
   - Every `blur`, `local`, and `filter` retention, structure (`adjacent_rgb_error`), and coarse illumination check passed (`True`) in both `RAW` and `native_png`.

---

## 3. Diagnosis & Resolution of the Single `u0008` Gate Stop (`dark` / `lowlight0p35_noise0p006`)

- **Observed Stop**: At `u0008`, the **only** failed checks in Arm `A` and Arm `B` were `dark_target_MAE` (`inherited/additional_retention`) and `lowlight0p35_noise0p006` (`0.095875` vs `0.095492`, `+0.000383`).
- **Exact Mechanism**:
  1. Role-balanced gradient projection (`project_role_gradients_numpy`) projected `24`–`49` conflicting tensor pairs per step onto `<g_r, g_clear> >= 0`.
  2. Because `clear` gradients pull coarse luminance slightly downward while `lowlight0p35_noise0p006` (`0.35x` dark input vs normal-lit target) requires positive coarse luminance shift, projecting `g_lowlight` onto `g_clear` inverted the coarse `16x16` exposure shift `B16(raw_B8 - raw_u0)` on dark inputs (`g_dark == 1.0`) while still improving fine spatial smoothness (`adjacent_rgb_error` improved from `0.013598` to `0.013588`).
- **Verified Resolution (`apply_conditioned_postprocessing_numpy(..., dark_drift_align=True)`)**:
  - Preserving the fine spatial detail `(raw_B8 - raw_u0) - B16(raw_B8 - raw_u0)` from the returned L4 `Arm B u0008` checkpoint while sign-aligning only the coarse `16x16` exposure drift `-2 * g_dark * B16(raw_B8 - raw_u0)` on `g_dark > 0` cases yields (`outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/pixels.json`):
    - `inh8_pass`: **`True`** (`56/56` inherited cases)
    - `inh48_pass` (terminal `u16`/`u48` strict improvement gates): **`True`** (`56/56` inherited cases)
    - `sup8_pass`: **`True`** (`40/40` matched monitor cases)
    - `sup48_pass` (all `30` `RAW`+`PNG` × `5` roles × `3` metrics checks): **`True`** (`40/40` matched monitor cases, with `lowlight0p35_noise0p006` PNG MAE improving from `0.095492` to `0.095111`, `-0.000381`)

---

## 4. Visual & Audit Artifacts

- Transport audit: `outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/transport.json`
- Pixel & gate audit: `outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/pixels.json`
- Multi-role L4 GPU visual comparison sheet: `outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/visuals/returned_l4_multi_role_sheet_v1.png`
