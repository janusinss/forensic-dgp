# FACE_MASK_COMPLETION Gradual Model Improvement Goal

Created: 29 October 2026 (Asia/Manila)
Scope: `FACE_MASK_COMPLETION` research checkout (`C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION`) and selective drop-in updates to `IMPLEMENTATION` (`C:\Users\janus\.codex\worktrees\implementation-one-shot\Testing`).
Focus: **Face Mask & Covering Completion (`CodeFormer` Inpainting 497-tensor architecture) + Covering Proposal Detector (`GatedUNet` 50-tensor segmenter)**. Ignore unrelated restoration model experiments (`Conditioned SFT V2` remains fixed).

---

## 1. Core Objective & Ratcheting Baseline Rule

Gradually and measurably improve the Face Mask & Covering Completion system across sequential, independently verified stages—starting from incumbent **Trained CodeFormer Completion R1** (`codeformer_completion_r1.pth`, SHA-256 `7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0`, full recovery state `terminal_training_state.pth`, SHA-256 `0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd`).

### Automatic Baseline Ratcheting ("Gradual Improvement Loop")
1. Every candidate stage initializes from the **current verified best recovery state** (starting with R1 `terminal_training_state.pth`).
2. A candidate replaces the active incumbent **only** if it passes all hard invariants and quality gates:
   - **Exact Outside-Support Preservation**: 100% bit-exact RGB pixels outside the reviewed removal mask in both float RAW and rounded uint8 PNG, plus zero neural calls on empty-mask controls.
   - **Clear Cohort Gate (32 paired cases)**: $\ge 1\%$ relative reduction in delivered-PNG masked-region L1 error versus the active incumbent, with non-worsening macro and per-geometry (`lower_face`, `eye_band`, `image_left_lens`, `cheek_object`) hole and $7\times 7$ inner-boundary L1.
   - **Challenge Cohort Gate (16 degraded/gray/irregular cases)**: $\ge 1\%$ relative reduction in delivered-PNG masked-region L1 error versus the active incumbent, with non-worsening subgroup hole and boundary L1.
   - **Strict-Gray Chroma Guard (8 strict-gray cases)**: Non-worsening (or strictly lower) mean channel spread $\text{mean}(\max(R,G,B) - \min(R,G,B))$ inside the generated support in both RAW and PNG.
   - **All-Sheet Visual Review**: No new invented eyewear frames, severe eye asymmetry, or plastic/orange scarf/hand blotches across the 48 paired monitor/challenge cases and 32 public covering-family cases.
3. When a candidate passes all gates and visual review, it is **promoted to the new active incumbent** (`R2`, `R3`, ...), its full Adam/RNG terminal state becomes the parent for the next stage, and its 497-tensor `params_ema` inference checkpoint is exported for `IMPLEMENTATION`.

---

## 2. Staged Improvement Roadmap

### Stage 1: Unified Pre-Training Overhaul (R2 Candidate Packet) — **COMPLETED**
Upgrade the training objective, spatial supervision, and regularization before the next manual L4 VM run while keeping the 497-tensor `CodeFormer` inference architecture 100% drop-in compatible:
1. **Support-Weighted Spatial Code & Latent Supervision (Code Route — 278 Tensors)**:
   - Downsample the $256\times 256$ binary mask to $16\times 16$ (`m16 = max_pool2d(m256, 16, 16)`) with a 1-cell context ring (`r16 = clamp(max_pool2d(m16, 3, 1, 1) - m16, 0, 1)`).
   - Weight $16\times 16$ Cross-Entropy and `lq_feat` Latent MSE by $w_i = 1.0\,m_{16,i} + 0.5\,r_{16,i} + 0.15\,(1 - m_{16,i} - r_{16,i})$ normalized over the 256 positions so mask-touched and boundary cells dominate encoder/predictor gradients.
2. **Joint 332-Tensor Co-Adaptation**:
   - Train both the 278 code tensors (164 encoder + 114 predictor) and 54 `fuse_convs_dict` (`32`, `64`, `128`) tensors in paired cycles from R1's `terminal_training_state.pth`, keeping the 165 `quantize`/`generator` prior tensors frozen.
3. **Dual-Resolution (Delivered-256 + 512) Hole, Seam-Boundary & Structure Gradient Loss**:
   - Optimize exact `delivered_float` at $256\times 256$ alongside $512\times 512$ hole L1, $7\times 7$ inner-boundary L1 at both $256$ and $512$, and spatial gradient L1 ($|\nabla_x \hat{I} - \nabla_x I^*|_1 + |\nabla_y \hat{I} - \nabla_y I^*|_1$) inside the mask.
4. **Differentiable Inter-Channel Chroma Consistency Loss**:
   - Penalize pairwise RGB channel-difference errors $|(\hat{R}-\hat{G}) - (R^*-G^*)| + |(\hat{G}-\hat{B}) - (G^*-B^*)| + |(\hat{B}-\hat{R}) - (B^*-R^*)|$ inside the mask (`0.35` RGB, `0.70` GRAY) to eliminate blue/orange false color on grayscale inputs.
5. **Clear-Anchor Trust-Region & O(1) Budgeted Serialization**:
   - Apply `1.25` anchor vs `0.75` augmented microbatch weighting, cosine LR decay ($8\times 10^{-6} \to 2\times 10^{-6}$), L2 parameter anchoring toward $\theta_{\text{R1}}$ (`coef=1e-4`), and O(1) `OutputBudget` disk accounting.
   - **Sealed R2 Execution Packet**: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-completion-codeformer-unified-r2-v1-execution.tar.gz` (`73,376,283` bytes, SHA-256 `83ee54a8a29b9a92bdb26ee1533957638a9c9a2dfa4065895354f690ca59f672`, protocol SHA-256 `60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347`) and [COMPLETION_CODEFORMER_UNIFIED_R2_V1_VM.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/COMPLETION_CODEFORMER_UNIFIED_R2_V1_VM.md).

### Stage 2: L4 Execution, Independent Return Audit & R2 Candidate Packaging — **COMPLETED**
1. Verify transfer CLI with scientific imports blocked and publish the 5-step manual L4 VM guide (`upload -> install -> tmux -> verify/run -> download`).
2. Upon return of the 3 result files in `C:\xampp\htdocs\YEAR 4\Testing\outputs`, run independent transport, 497-tensor state/Adam/RNG, RAW-to-PNG pixel replay, outside-mask byte preservation, and 20-sheet visual review audits.
3. **Ratified New Active Incumbent (`Trained CodeFormer Completion R2`)**:
   - **Inference Checkpoint**: [codeformer_completion_r2.pth](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/codeformer_completion_r2.pth) (`370,773,572` bytes, SHA-256 `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`, logical 497-tensor SHA-256 `300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0`).
   - **Recovery Parent State (`terminal_training_state.pth`)**: `outputs/completion_codeformer_unified_r2_v3_return/cctv_completion_codeformer_unified_r2_v3_results/outputs/terminal_training_state.pth` (`701,961,297` bytes, SHA-256 `23d81ba89e53accb9af09f1bd20a871f3071eae24cd41b620c08fc73c68ffd7b`).
   - **Audit Report**: [COMPLETION_CODEFORMER_UNIFIED_R2_V3_RESULTS.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/COMPLETION_CODEFORMER_UNIFIED_R2_V3_RESULTS.md) (`60/60` numeric gates PASS: Clear `PNG_hole_L1` `-1.0527%`, Challenge `PNG_hole_L1` `-2.6873%`, Strict-Gray `PNG_mean_channel_spread` `-10.3204%`, `44,001,540 / 44,001,540` exact outside RGB bytes).

### Stage 3: Covering Proposal Segmenter Audit & Diagnosis (`covering_detector_segmenter_v1.pth` vs Historical VM Candidates) — **DIAGNOSED**
1. **Root Cause of `covering_detector_segmenter_v1.pth` Empty Masks**:
   - Extracted without fine-tuning from `completion_pilot/epoch_2.pth` (SHA-256 `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`), where `GatedUNet(3, 1, 24)` (`50` FP32 tensors) was trained exclusively on procedural flat-color + Gaussian-noise geometric rectangles/ellipses (`completion.py:synthetic_covering`).
2. **Historical VM-Returned `GatedUNet(3, 1, 24)` Audit**:
   - Evaluated all 4 historical VM-returned `GatedUNet(3, 1, 24)` checkpoints (`balanced_penalty_ep9`, `balanced_ctrl_ep10`, `replay_ep10`, `consistency_ep10`) on the 9 `Conditioned SFT V2` restored canvases ([nine_case_restored_r1_vs_r2_audit.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/nine_case_restored_r1_vs_r2_audit.json)): all 4 produce false-positive masks on `08_uncovered_degraded` (`1,251` to `12,525` px) and return `0` pixels on `05_white_glare_degraded` and `val_362_hair_eye_degraded`.
   - Under the VM-only training rule ([FACE_MASK_COMPLETION_SPEC_V1.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/FACE_MASK_COMPLETION_SPEC_V1.md#L19)), `covering_detector_segmenter_v1.pth` is retained until a dedicated multi-family `detector_supported_review_v3` L4 VM run is executed. See [COVERING_DETECTOR_AND_R2_REPLACEABILITY_REPORT.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/COVERING_DETECTOR_AND_R2_REPLACEABILITY_REPORT.md).

### Stage 4: Selective Drop-In Integration into `IMPLEMENTATION` — **COMPLETED**
1. **Replaceability Verified & Staged (`R1 -> R2`)**:
   - Staged [codeformer_completion_r2.pth](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/models/completion/codeformer_completion_r2.pth) (`370,773,572` bytes, SHA-256 `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`, logical 497-tensor SHA-256 `300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0`, `nlink=1`) alongside retained `codeformer_completion_r1.pth` and `covering_detector_segmenter_v1.pth`.
   - Updated [completion_assets.json](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/implementation/completion_assets.json), [completion_assets.py](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/implementation/completion_assets.py), [completion_runtime.py](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/implementation/completion_runtime.py), [test_completion_runtime.py](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/tests/test_completion_runtime.py), and [COMPLETION_MODEL_CARD.md](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/COMPLETION_MODEL_CARD.md) with zero changes (`0` diff bytes) across all 18 protected `Conditioned SFT V2` restoration files.
2. **Regression, 9-Case Restored-Canvas & Inline Playwright Verification**:
   - Passed `77/77` core completion/restoration unit tests (`1` Windows symlink privilege skip), `45/45` additional Python unit tests, and `3/3` Node `.cjs` test suites.
   - Verified `1,521,399 / 1,521,399` exact outside-mask RGB bytes across all 9 `Conditioned SFT V2` restored degraded integration cases ([nine_case_restored_r1_vs_r2_comparison.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/nine_case_restored_r1_vs_r2_comparison.png), [nine_case_restored_r1_vs_r2_audit.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/nine_case_restored_r1_vs_r2_audit.json)).
   - Passed inline Playwright responsive verification across `375x667`, `768x1024`, and `1280x800` (`0` overflow, `0` console/page errors) and full live browser E2E R2 completion on `val_362_hair_eye_degraded.png` (`13.42s`, `1` neural call, `exact_outside: true`, bundle generated; [implementation-r2-playwright-flow.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/scratch/implementation-r2-playwright-flow.json)).
