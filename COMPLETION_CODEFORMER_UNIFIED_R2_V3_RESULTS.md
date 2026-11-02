# Unified CodeFormer Completion R2 v3 — L4 Execution, Audit & Ratification Results

Date: 2 November 2026 (Asia/Manila)
Stage: **Stage 2 — L4 Execution, Independent Return Audit & R2 Candidate Ratification**
Parent Incumbent: **Trained CodeFormer Completion R1** (`terminal_training_state.pth`, SHA-256 `0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd`; inference checkpoint `codeformer_completion_finetune_r1.pth`, SHA-256 `7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0`)
Promoted Incumbent: **Trained CodeFormer Completion R2** (`outputs/completion_codeformer_unified_r2_v3_candidate/codeformer_completion_r2.pth`, SHA-256 `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`, logical 497-tensor SHA-256 `300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0`; recovery parent `outputs/completion_codeformer_unified_r2_v3_return/cctv_completion_codeformer_unified_r2_v3_results/outputs/terminal_training_state.pth`, SHA-256 `23d81ba89e53accb9af09f1bd20a871f3071eae24cd41b620c08fc73c68ffd7b`)

---

## 1. Executive Verdict

**PROMOTED TO ACTIVE INCUMBENT (`Trained CodeFormer Completion R2`)**

Unified CodeFormer Completion R2 v3 (`Option A` — Joint 332-Tensor Code+Fusion Co-Adaptation with Support-Weighted $16\times 16$ Code/Latent Supervision, Dual-Resolution $256+512$ Hole/Boundary/Gradient L1, Differentiable Inter-Channel Chroma Consistency Loss, and Clear-Anchor Trust-Region Weighting toward R1) passed **all 60 numeric ratchet gates** (`all_gates_pass: true`), **100% outside-mask byte preservation** (`44,001,540 / 44,001,540` exact RGB bytes across all 256 evaluated PNGs), and **all 20 visual comparison sheets** (`original_monitor_01..08.png`, `challenge_01..04.png`, `public_unpaired_01..08.png`).

---

## 2. L4 VM Execution & Independent Return Audit Summary

- **Execution Packet**: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-completion-codeformer-unified-r2-v3-execution.tar.gz` (`74,369,101` bytes, SHA-256 `80c4b4f992e08fb1e5135ddc7e34c51c90714cd1c0de5625140f29dc57c228af`, protocol SHA-256 `463c1977e9cd74f0a4fd0240df7a784f48b613043ef667714f1ef94adb4cd60a`)
- **Returned Archive**: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-completion-codeformer-unified-r2-v3-results.tar.gz` (`1,456,894,853` bytes, SHA-256 `f2eba0ddbc881c0f0abce7d207c03e22a1633391eaab4255d135aff8e608a199`, `848` payload files, `1,670,439,805` uncompressed bytes)
- **Hardware & Runtime**: Google Cloud `g2-standard-4` (`NVIDIA L4`, `PyTorch 2.6.0+cu124`), `352.01s` (`5m 52s`), `worker_exit = 0`
- **Accounting Counters**:
  - `forward_slots = 640` (`parity: 8, fit_code: 256, fit_RGB: 128, monitor: 128, challenge: 64, public_development: 56`)
  - `backward_calls = 384` (`code: 256, fusion: 128`)
  - `code_optimizer_steps = 64`, `fusion_optimizer_steps = 32`, `total_combined_updates = 32`
  - `empty_bypasses = 168` (`0` neural calls on empty-support controls)
- **Independent Audit Receipts**:
  - Admission & extraction: [admit_completion_codeformer_unified_r2_v3_return.py](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/scripts/admit_completion_codeformer_unified_r2_v3_return.py) -> [independent_return_audit.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/independent_return_audit.json)
  - Visual sheet & RAW/PNG replay audit: [build_completion_codeformer_unified_r2_v3_review.py](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/scripts/build_completion_codeformer_unified_r2_v3_review.py) -> [visual_sheet_index.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/visual_sheet_index.json), [review.html](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/review.html)
  - Frozen 165-tensor `quantize`/`generator` prior SHA-256: `3a549c6fbe9ccf77f17c233fd4740b253a5c7282505c6dc2fe166437625bfebd` (bit-exact across all cycles)

---

## 3. Complete 60-Gate Numeric Ratchet Comparison (`u0032` vs `u0000` R1 Baseline)

### 3.1 Clear Monitor Cohort (`32` Paired Clear Cases — 20/20 Gates PASS)

| Scope / Subgroup | Metric | R1 Baseline (`u0000`) | Unified R2 v3 (`u0032`) | Relative Change | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Macro (`32` cases)** | `PNG_hole_L1` | `0.048641` | `0.048129` | **`-1.0527%`** (`+1.0527%` gain) | **PASS** ($\ge 1.0\%$ gain) |
| **Macro (`32` cases)** | `PNG_boundary_L1` | `0.026929` | `0.026252` | **`-2.5129%`** | **PASS** (non-worsening) |
| **Macro (`32` cases)** | `raw_hole_L1` | `0.048655` | `0.048142` | **`-1.0551%`** | **PASS** (non-worsening) |
| **Macro (`32` cases)** | `raw_boundary_L1` | `0.026951` | `0.026279` | **`-2.4964%`** | **PASS** (non-worsening) |
| `lower_face` (`8` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.046847` / `0.029758` | `0.046042` / `0.028908` | **`-1.7182%`** / **`-2.8553%`** | **PASS** |
| `eye_band` (`8` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.048824` / `0.029356` | `0.048636` / `0.028786` | **`-0.3843%`** / **`-1.9405%`** | **PASS** |
| `image_left_lens` (`8` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.056771` / `0.025792` | `0.056493` / `0.025285` | **`-0.4894%`** / **`-1.9653%`** | **PASS** |
| `cheek_object` (`8` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.042124` / `0.022809` | `0.041270` / `0.021960` | **`-2.0272%`** / **`-3.7229%`** | **PASS** |

### 3.2 Challenge Monitor Cohort (`16` Degraded/Gray/Irregular Cases — 28/28 Gates PASS)

| Scope / Subgroup | Metric | R1 Baseline (`u0000`) | Unified R2 v3 (`u0032`) | Relative Change | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Macro (`16` cases)** | `PNG_hole_L1` | `0.050733` | `0.049370` | **`-2.6873%`** (`+2.6873%` gain) | **PASS** ($\ge 1.0\%$ gain) |
| **Macro (`16` cases)** | `PNG_boundary_L1` | `0.029308` | `0.027930` | **`-4.7020%`** | **PASS** (non-worsening) |
| **Macro (`16` cases)** | `raw_hole_L1` | `0.050747` | `0.049383` | **`-2.6879%`** | **PASS** (non-worsening) |
| **Macro (`16` cases)** | `raw_boundary_L1` | `0.029332` | `0.027954` | **`-4.6966%`** | **PASS** (non-worsening) |
| `lower_face` (`4` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.050763` / `0.032277` | `0.050444` / `0.030982` | **`-0.6286%`** / **`-4.0110%`** | **PASS** |
| `eye_band` (`4` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.049234` / `0.029584` | `0.047938` / `0.028620` | **`-2.6320%`** / **`-3.2611%`** | **PASS** |
| `image_left_lens` (`4` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.060977` / `0.030816` | `0.056649` / `0.028858` | **`-7.0968%`** / **`-6.3520%`** | **PASS** |
| `cheek_object` (`4` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.041956` / `0.024554` | `0.041953` / `0.023484` | **`-0.0074%`** / **`-4.3592%`** | **PASS** |
| `RGB_irregular_mild_degraded` (`8` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.047888` / `0.029500` | `0.047820` / `0.028495` | **`-0.1408%`** / **`-3.4080%`** | **PASS** |
| `strict_gray_mild_degraded` (`8` cases) | `PNG_hole_L1` / `PNG_boundary_L1` | `0.053577` / `0.029115` | `0.050921` / `0.027359` | **`-4.9574%`** / **`-6.0304%`** | **PASS** |

### 3.3 Strict-Gray Chroma Guard (`8` Strict-Gray Cases — 10/10 Gates PASS)

| Scope / Subgroup | Metric | R1 Baseline (`u0000`) | Unified R2 v3 (`u0032`) | Relative Change | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Macro (`8` cases)** | `GRAY_PNG_mean_channel_spread` | `4.6712` | `4.1891` | **`-10.3204%`** | **PASS** |
| **Macro (`8` cases)** | `GRAY_raw_mean_channel_spread` | `0.018323` | `0.016424` | **`-10.3621%`** | **PASS** |
| `lower_face` (`2` cases) | `GRAY_PNG_mean_channel_spread` | `5.6910` | `5.3014` | **`-6.8464%`** | **PASS** |
| `eye_band` (`2` cases) | `GRAY_PNG_mean_channel_spread` | `4.5059` | `3.8820` | **`-13.8461%`** | **PASS** |
| `image_left_lens` (`2` cases) | `GRAY_PNG_mean_channel_spread` | `3.5624` | `3.2742` | **`-8.0908%`** | **PASS** |
| `cheek_object` (`2` cases) | `GRAY_PNG_mean_channel_spread` | `4.9254` | `4.2987` | **`-12.7240%`** | **PASS** |

---

## 4. All-20-Sheet Visual Inspection Summary

1. **Clear Monitor (`original_monitor_01..08.png`, 32 cases)**:
   - Cleaner mouth/lip texture and smoother lower-cheek seam transitions across `lower_face` cases (`tr_ffhq_05179`, `tr_ffhq_05302`, `tr_ffhq_06809`, `tr_ffhq_07087`, `tr_ffhq_07202`, `tr_ffhq_07386`).
   - Fainter wireframe eyewear ghosting on `tr_ffhq_07087__eye_band` and cleaner temple/bridge alignment on `tr_ffhq_07087__image_left_lens`.
   - Smoother skin shading inside `cheek_object` masks across all 8 subjects.
2. **Challenge Monitor (`challenge_01..04.png`, 16 cases)**:
   - Visibly reduced cyan/blue sclera/iris tinting on strict-gray eye reconstructions (`v2m_05302_g1_gd`, `v2m_06781_g2_gd`) and warm/orange lip tinting on strict-gray lower-face reconstructions (`v2m_05179_g0_gd`, `v2m_07386_g0_gd`).
   - Cleaner $7\times 7$ inner boundary transitions on `RGB_irregular_mild_degraded`.
3. **Public Unpaired Covering Families (`public_unpaired_01..08.png`, 32 cases)**:
   - **`04_sunglasses_degraded` ([public_unpaired_02.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/visual_sheets/public_unpaired_02.png))**: Eliminates invented wireframe eyeglasses that R1 `u0` hallucinated after removing dark sunglasses, reconstructing natural eyes and brow ridge.
   - **`00_cloth_mask_native/degraded` & `01_pink_mask_native/degraded` ([public_unpaired_01.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/visual_sheets/public_unpaired_01.png))**: Reduces dark cheek-shadow seams at the mask border.
   - **`val_25_hand_mouth` & `val_362_hair_eye` ([public_unpaired_06.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/visual_sheets/public_unpaired_06.png))**: Suppresses false warm lip tint and blue eye-shadow tint on monochrome photographs.
   - **`08_uncovered_native/degraded` & `09_clear_glasses_native/degraded` ([public_unpaired_04.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/visual_sheets/public_unpaired_04.png), [public_unpaired_05.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_return/visual_sheets/public_unpaired_05.png))**: Bit-exact identity preservation via empty-mask bypass (`0` neural calls).
