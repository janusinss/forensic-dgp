# Unified CodeFormer Completion R2 V3 Design (`completion_codeformer_unified_r2_v3`)

## 1. Purpose, Baseline Provenance & Root-Cause Attribution from R2 v1 / R2 v2

This specification defines **Trained CodeFormer Completion R2 V3** (`cctv_completion_codeformer_unified_r2_v3_vm`), continuing strictly from incumbent **Trained CodeFormer Completion R1** (`terminal_training_state.pth`, SHA-256 `0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd`) after diagnostic attribution of `R2 v1` and `R2 v2`:

1. **Elimination of `R2 v2` Interior `core256_l1` Tiny-Mask Gradient Spike (`cheek_object` Fix)**:
   - In `R2 v2`, `core256 = mask256 - b256` left ~9 interior pixels on `cheek_object` (`mask256` ~150 px), creating a ~280x per-pixel gradient spike in `0.75 * core256_l1` that regressed Clear `cheek_object` (`+0.391%`) and Challenge `cheek_object` (`+0.680%`).
   - `R2 v3` removes `core256_l1` (`0.0`), restores `1.00 * raw512_hole_L1 + 1.00 * raw512_seam_L1 + 0.35 * delivered256_hole_L1 + 0.20 * delivered256_seam_L1 + 0.10 * delivered256_grad_L1 + chroma` (`0.10` RGB / `0.25` GRAY), and applies soft tiny-mask area damping `area_scale = mask256_sum / (mask256_sum + 300.0)` with clear-anchor role weighting (`1.20 / 0.80`, trust `coef = 8e-5`).
2. **Dual-192-FIT Skip Encoder Synthesis + `casc3` Ridge + Routed Soft-KL Code Refinement**:
   - Single-128-case FIT subsets under-cover the frozen 256-case `V4` FIT pool (`balanced128` alone under-covers orthogonal geometry roles; `orthogonal_balanced128` alone leaves `Clear eye_band` or `Challenge cheek_object` conflicting). Combining `balanced128_subset.json` (`128` cases) and `orthogonal_balanced128_subset.json` (`128` cases) spans **`192` unique FIT cases** (`0` monitor/challenge overlap).
   - **Stage A1 Branch 1 (`32` Code Cycles on `balanced128`)**: Trains `code_optimizer` from pure R1 (`8.0e-6 -> 2.0e-6`, `hole=1.0, ring=0.50, visible=0.15`, `lmse=1.00`, role `1.25 / 0.75`, trust `1e-4`), capturing `branch1_skip_state` (`encoder.blocks[0..14]`).
   - **Stage A1 Branch 2 (`32` Code Cycles on `orthogonal_balanced128`)**: Resets `code` weights & `code_optimizer` to pure R1 and trains `32` orthogonal cycles (`4.0e-6 -> 1.0e-6`, `inner_ring=1.0, outer_ring=0.50, core=0.10, visible=0.15`, `lmse=0.10`, role `1.05 / 0.95`, trust `5e-4`), capturing `branch2_skip_state`.
   - **Stage A2+A3 (`0.75 * d_branch1 + 0.75 * d_branch2` Skip + `ft5_100` + `casc3` Ridge + Routed Soft-KL)**: Synthesizes the 94-tensor skip encoder (`r1_w + 0.75 * d_b1 + 0.75 * d_b2`), trains pure-R1 predictor specialists (`ft5_100`), solves 3-stage CPU `float32` ridge (`lam = 10.0, 10.0, 3.0`) on `blocks[15].conv`, `blocks[23]`, and `feat_emb`, and aligns the 114-tensor predictor via multi-temperature soft-KL (`0.45 * p58 + 0.55 * r3`) using FIT cases only.
3. **Moment-Calibrated Stage B Fusion (`64` Calibration Steps + `32` Update Cycles)**:
   - Calibrates `fusion_optimizer` first/second Adam moments (`step 32 -> 96`) on the dual-resolution `V3+R2` objective at `R1` weights using frozen Stage A FIT features, resets `fuse_convs_dict` weights to exact `R1`, and executes `32` orthogonal Stage B fusion cycles (`1.0e-5 -> 3.0e-6`, `step 96 -> 128`).
4. **100% Unmodified 497-Tensor Drop-In `CodeFormer` Compatibility**:
   - Zero architectural changes (`dim_embd=512, codebook_size=512, n_head=8, n_layers=9, connect_list=['32','64','128']`), `165` `quantize.*`/`generator.*` tensors bit-for-bit frozen, exact `640` whole-model forwards, `384` loop backwards, and `96` loop optimizer steps (`64` code + `32` fusion).
