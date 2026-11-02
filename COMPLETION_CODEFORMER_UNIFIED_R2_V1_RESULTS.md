# Unified CodeFormer Completion R2 v1: Actual Return & Diagnostic Audit

29 October 2026. **EXECUTED; VALID 332-TENSOR UNIFIED TRAINING STATE, PASSED STRICT-GRAY CHROMA & ALL BOUNDARY GATES, RETAINED R1 FOR TWO GEOMETRY HOLE TOLERANCES (`clear/eye_band`, `challenge/cheek_object`).**

[Inspect every saved comparison](outputs/completion_codeformer_unified_r2_v1_return/review.html) | [Independent return audit receipt](outputs/completion_codeformer_unified_r2_v1_return/independent_return_audit.json) | [Visual sheet index](outputs/completion_codeformer_unified_r2_v1_return/visual_sheet_index.json)

---

## 1. Execution & Archive Integrity Summary

1. **L4 VM Execution (`g2-standard-4`, NVIDIA L4)**:
   - Completed all `32` paired updates (`512` forward slots: `8` parity, `128` fit_code, `128` fit_RGB, `128` clear monitor, `64` challenge, `56` public development; `256` backward calls; `32` code optimizer steps; `32` fusion optimizer steps; `168` exact empty-mask bypasses).
   - Worker completed in `218.468s` (`supervisor.json`: `227.967s`, `timeout: false`, `worker_exit: 2` solely from the final prospective numeric gate screen).
2. **Return Archive & Terminal State Pins**:
   - Return archive `cctv-completion-codeformer-unified-r2-v1-results.tar.gz`: `1,614,097,048` bytes, SHA-256 `9783f814e2711acb90e3f3bd2fdf54bab3af8896f04a61458e20069d8006fc7b`.
   - Terminal training state `outputs/terminal_training_state.pth`: `701,932,773` bytes, SHA-256 `880d1aa89f9965870683bbe013dc14e91faf9368b45f9b3602e451712844d57e`.
   - Reconstructed 497-tensor logical state SHA-256: `c5112db7f7a10979bc72bf7208fe3c5d2fc15cbbe3916d31b87afccea17f86e2`.
   - Frozen 165-tensor `quantize`/`generator` prior SHA-256: `3a549c6fbe9ccf77f17c233fd4740b253a5c7282505c6dc2fe166437625bfebd` (bit-identical to `codeformer_inpainting.pth` and `R1`).
   - Outside-mask pixel preservation: `44,001,540 / 44,001,540` exact RGB bytes (`0` diff across all `256` evaluated PNGs).

---

## 2. Delivered Image & Gate Performance vs Incumbent R1 (`u0032` vs `u0000`)

| Gate / Cohort | Metric | R1 (`u0000`) | Unified R2 v1 (`u0032`) | Relative Change | Gate Status |
|---|---|---:|---:|---:|---|
| **Strict-Gray Chroma** (8 cases) | `GRAY_PNG_mean_channel_spread` | `4.6712` | `4.1662` | **`-10.81%` (false color reduced)** | **PASS (`all 4 geometries non-worse`)** |
| **Clear Monitor Boundary** (32 cases) | `PNG_boundary_L1` | `0.036324` | `0.035839` | **`-1.3351%` (all 4 geometries improved)** | **PASS (`4/4 geometries improved`)** |
| **Challenge Monitor Boundary** (16 cases) | `PNG_boundary_L1` | `0.052881` | `0.051166` | **`-3.2435%` (all 4 geometries improved)** | **PASS (`4/4 geometries improved`)** |
| **Challenge Monitor Hole** (16 cases) | `PNG_hole_L1` | `0.051271` | `0.049804` | **`+2.8612%` gain (`-2.86%` error)** | **Macro PASS (`>= +1.0%`), `cheek_object` `+1.42%`** |
| **Clear Monitor Hole** (32 cases) | `PNG_hole_L1` | `0.039815` | `0.039941` | `-0.3162%` gain (`+0.32%` error) | **MISS (`eye_band` `+2.10%`)** |

### Geometry Breakdown (`u0032` vs `u0000`)

- **Clear Monitor (32 cases)**:
  - `lower_face`: `PNG_hole_L1` **`-0.8055%`**, `PNG_boundary_L1` **`-0.4487%`**
  - `cheek_object`: `PNG_hole_L1` **`-0.7305%`**, `PNG_boundary_L1` **`-2.0349%`**
  - `image_left_lens`: `PNG_hole_L1` `+0.0286%`, `PNG_boundary_L1` **`-1.9605%`**
  - `eye_band`: `PNG_hole_L1` `+2.1001%` (driven by `tr_ffhq_06809__eye_band` and `tr_ffhq_06995__eye_band`), `PNG_boundary_L1` **`-0.7248%`**
- **Challenge Monitor (16 cases)**:
  - `image_left_lens`: `PNG_hole_L1` **`-8.0678%`**, `PNG_boundary_L1` **`-3.8772%`**
  - `eye_band`: `PNG_hole_L1` **`-1.9409%`**, `PNG_boundary_L1` **`-3.3076%`**
  - `lower_face`: `PNG_hole_L1` **`-1.7775%`**, `PNG_boundary_L1` **`-3.2366%`**
  - `cheek_object`: `PNG_hole_L1` `+1.4242%`, `PNG_boundary_L1` **`-2.3270%`**
  - Condition `strict_gray_mild_degraded`: `PNG_hole_L1` **`-4.2698%`**, `PNG_boundary_L1` **`-4.1476%`**
  - Condition `RGB_irregular_mild_degraded`: `PNG_hole_L1` **`-1.2809%`**, `PNG_boundary_L1` **`-2.2036%`**

---

## 3. CPU Subsystem Attribution (`scripts/ablate_unified_r2_v1_subsystems.py`)

Evaluating isolated parameter groups of Unified R2 v1 against R1 on CPU revealed the structural cause of the two geometry misses:

1. **`fusion_only` (`54` `fuse_convs_dict` tensors from R2 v1 + R1 `encoder` & `predictor`)**:
   - Improves **every single Clear geometry on `PNG_hole_L1`** (`lower_face -0.6077%`, `eye_band -0.0369%`, `image_left_lens -0.0477%`, `cheek_object -0.2773%`) AND passes **`GRAY chroma` (`4.0527`, `-13.2%` spread reduction)**.
2. **`encoder_only` (`164` `encoder` tensors from R2 v1 + R1 `predictor` & `fuse_convs_dict`)**:
   - Passes **every single Challenge gate** (`+3.3004%` Challenge PNG hole gain, `nonworse: True` across all 4 Challenge geometries and both conditions!), **but** causes the Clear `eye_band` hole regression (`+1.8105%`) because `CodeFormer.forward` detaches `enc_feat_dict['32','64','128']` at line 277 (`enc_feat_dict[f_size].detach()`) and only trained `encoder.blocks[0..14]` through the 16x16 code route (`code_only=True`) on 50% synthetically degraded inputs without pixel/fusion supervision.
