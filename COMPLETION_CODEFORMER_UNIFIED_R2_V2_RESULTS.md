# Unified CodeFormer Completion R2 v2: Actual Return & Diagnostic Audit

1 November 2026. **EXECUTED; VALID 332-TENSOR UNIFIED TRAINING STATE, PASSED STRICT-GRAY CHROMA (`-11.92%` SPREAD ACROSS ALL 8 GRAY CASES) & CHALLENGE MACRO HOLE GAIN (`+2.8709%`), RETAINED R1 FOR CLEAR MONITOR HOLE TOLERANCES (`clear/eye_band`, `clear/image_left_lens`).**

[Inspect every saved comparison](outputs/completion_codeformer_unified_r2_v2_return/review.html) | [Independent return audit receipt](outputs/completion_codeformer_unified_r2_v2_return/independent_return_audit.json) | [Visual sheet index](outputs/completion_codeformer_unified_r2_v2_return/visual_sheet_index.json)

---

## 1. Execution & Archive Integrity Summary

1. **L4 VM Execution (`g2-standard-4`, NVIDIA L4)**:
   - Completed all `32` paired updates on the de-aliased orthogonal 128-case schedule (`512` forward slots: `8` parity, `128` fit_code, `128` fit_RGB, `128` clear monitor, `64` challenge, `56` public development; `256` backward calls; `32` code optimizer steps; `32` fusion optimizer steps; `168` exact empty-mask bypasses).
2. **Return Archive & Terminal State Pins**:
   - Return archive `cctv-completion-codeformer-unified-r2-v2-results.tar.gz`: `1,614,165,734` bytes, SHA-256 `62028b6cdc7ec165a2dac48df1b00d132f958ebb905abf28d2b3c8e1da48736e`.
   - Terminal training state `outputs/terminal_training_state.pth`: SHA-256 `bf84a862124939253c2fc49350fe6742b80b204081e5cb8cc2972d61f593b9fa`.
   - Reconstructed 497-tensor logical state SHA-256: `b98c25c351e440d52485ef68210678112889c8aa83f584c14bc943d2cab055fe`.
   - Frozen 165-tensor `quantize`/`generator` prior SHA-256: `3a549c6fbe9ccf77f17c233fd4740b253a5c7282505c6dc2fe166437625bfebd` (bit-identical to `codeformer_inpainting.pth` and `R1`).
   - Outside-mask pixel preservation: `44,001,540 / 44,001,540` exact RGB bytes (`0` diff across all `256` evaluated PNGs).

---

## 2. Delivered Image & Gate Performance vs Incumbent R1 (`u0032` vs `u0000`)

| Gate / Cohort | Metric | R1 (`u0000`) | Unified R2 v2 (`u0032`) | Relative Change | Gate Status |
|---|---|---:|---:|---:|---|
| **Strict-Gray Chroma** (8 cases) | `GRAY_PNG_mean_channel_spread` | `4.6712` | `4.1142` | **`-11.92%` (8/8 gray cases improved)** | **PASS (`all 4 geometries non-worse`)** |
| **Challenge Monitor Hole** (16 cases) | `PNG_hole_L1` | `0.050733` | `0.049277` | **`+2.8709%` gain (`-2.87%` error)** | **Macro PASS (`>= +1.0%`), `cheek_object` `+0.97%` at `u32` (`-2.53%` at `u8`, `-1.54%` at `u16`)** |
| **Challenge Monitor Boundary** (16 cases) | `PNG_boundary_L1` | `0.029308` | `0.028301` | **`-3.4370%` (all 4 geometries improved)** | **PASS (`4/4 geometries improved`)** |
| **Clear Monitor Boundary** (32 cases) | `PNG_boundary_L1` | `0.026929` | `0.026590` | **`-1.2606%` (`3/4` geometries improved)** | **`image_left_lens` `+0.035%`** |
| **Clear Monitor Hole** (32 cases) | `PNG_hole_L1` | `0.048641` | `0.048987` | `-0.7111%` gain (`+0.71%` error) | **MISS (`eye_band` `+1.39%`, `image_left_lens` `+1.68%`)** |

---

## 3. Structural Root Cause Discovered via 4-Group Attribution (`scratch/ablate_r2v2_3groups.py`)

Decomposing `CodeFormer`'s 332 trainable tensors into:
- `skip_encoder` (`encoder.blocks[0..14]`, 118 tensors producing `enc_feat_dict['32','64','128']`),
- `bot_encoder` (`encoder.blocks[15..18]`, 46 tensors producing $16\times 16$ `lq_feat`),
- `predictor` (`position_emb`, `feat_emb`, `ft_layers`, `idx_pred_layer`, 114 tensors), and
- `fusion` (`fuse_convs_dict['32','64','128']`, 54 tensors)

proved two critical structural facts:
1. **`enc_feat_dict[f_size].detach()` in `CodeFormer.forward` blocked RGB pixel supervision from `skip_encoder`**:
   - In both `R2 v1` and `R2 v2`, `encoder.blocks[0..14]` received zero gradients during the RGB fusion step because of `.detach()` on `enc_feat_dict[f_size]`, and only received $16\times 16$ code loss gradients on 50% synthetically degraded inputs.
   - Meanwhile, updating `lq_feat` (`encoder.blocks[0..18]`) on degraded inputs caused discrete `argmax` code flips on Clear `eye_band` (`tr_ffhq_06809__eye_band` jumped `+8.18%`), driving the entire Clear `eye_band` regression (`+1.4647%` when `lq16=R2v2` vs **`-0.3041%`** when `lq16=R1`).
2. **`delivered256_core_L1` (`0.75`) over-weighted small eroded supports (`image_left_lens`, `cheek_object`)**:
   - Dividing by `core_mask.sum()` after $7\times 7$ erosion amplified gradient variance on small geometries (`image_left_lens` shifted from `-0.0477%` in `R2 v1` fusion to `+1.4464%` in `R2 v2` fusion).
   - Combining improved `skip_encoder` features (`enc_feat_dict['32','64','128']`) with `V3` hole+seam fusion while keeping `lq16` anchored to R1 (`Test E`) simultaneously achieved **`nonworse: True` across all 20 Clear Monitor hole and boundary subgroups**, **`+1.0938%` Challenge macro hole gain**, **`-5.5126%` Challenge boundary reduction**, and **`True` Strict-Gray Chroma pass (`4.1039`)**.
