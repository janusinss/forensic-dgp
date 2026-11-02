# Replaceability Verdict: Trained CodeFormer Completion R2 & Covering Proposal Segmenter Audit

Created: 2 November 2026 (Asia/Manila)
Scope: `FACE_MASK_COMPLETION` (`c:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION`) and `IMPLEMENTATION` (`C:\Users\janus\.codex\worktrees\implementation-one-shot\Testing`).

---

## 1. Executive Verdict: Is the Current Model Replaceable with the Updated Model?

1. **Face Mask & Covering Completion Generator (`CodeFormer` 497-tensor architecture): YES — IMMEDIATELY REPLACEABLE (`R1 -> R2`)**
   - **Current Incumbent (`R1`)**: `codeformer_completion_r1.pth` (`370,764,381` bytes, SHA-256 `7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0`, logical SHA-256 `55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6`).
   - **Ratified Replacement (`R2`)**: [codeformer_completion_r2.pth](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/codeformer_completion_r2.pth) (`370,773,572` bytes, SHA-256 `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`, logical SHA-256 `300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0`).
   - **Why R2 Replaces R1**:
     - Passed **60/60 numeric ratchet gates** on the L4 VM return ([COMPLETION_CODEFORMER_UNIFIED_R2_V3_RESULTS.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/COMPLETION_CODEFORMER_UNIFIED_R2_V3_RESULTS.md)):
       - Clear Cohort (`32` paired cases): `macro.PNG_hole_L1` `0.048641 -> 0.048129` (**`-1.0527%`** error / `+1.0527%` gain), `macro.PNG_boundary_L1` **`-2.5129%`**; all 20 subgroup metrics improved.
       - Challenge Cohort (`16` paired cases): `macro.PNG_hole_L1` `0.050733 -> 0.049370` (**`-2.6873%`** error / `+2.6873%` gain), `macro.PNG_boundary_L1` **`-4.7020%`**; all 28 subgroup metrics improved.
       - Strict-Gray Chroma Guard (`8` cases): `macro.GRAY_PNG_mean_channel_spread` `4.6712 -> 4.1891` (**`-10.3204%`**); all 10 metrics improved.
       - Outside-Support Preservation: `44,001,540 / 44,001,540` exact RGB bytes across all 256 evaluated PNGs (`168` empty-mask bypasses).
     - Verified on the **9 `Conditioned SFT V2` restored degraded integration canvases** ([nine_case_restored_r1_vs_r2_audit.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/nine_case_restored_r1_vs_r2_audit.json), [nine_case_restored_r1_vs_r2_comparison.png](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/outputs/completion_codeformer_unified_r2_v3_candidate/nine_case_restored_r1_vs_r2_comparison.png)):
       - Preserves **`1,521,399 / 1,521,399`** exact outside-mask RGB bytes from `restoration-result.png` (`input.png`) with `0` neural calls on the 2 clear/uncovered controls.
       - Reduces grayscale false-color chroma spread on `val_362_hair_eye_degraded` from **`13.2211 -> 12.5458` (`-5.11%`)** and reduces inside-mask chroma spread on `03_dark_sunglasses_degraded`, `05_white_glare_degraded`, `val_18_hand_eyes_degraded`, and `val_6_flower_mouth_degraded`.
       - Shares the exact 497-tensor `CodeFormer` (`dim_embd=512, codebook_size=512, n_head=8, n_layers=9, connect_list=['32','64','128']`) architecture and `{"params_ema": ...}` schema, requiring zero changes to any of the 18 protected `Conditioned SFT V2` restoration files.

2. **Covering Proposal Segmenter (`GatedUNet(3, 1, 24)` 50-tensor architecture): NOT YET REPLACEABLE BY HISTORICAL CHECKPOINTS — RETAIN `covering_detector_segmenter_v1.pth` UNTIL L4 VM SEGMENTER TRAINING**
   - **Current Incumbent**: `covering_detector_segmenter_v1.pth` (`2,681,549` bytes, SHA-256 `ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf`, logical SHA-256 `888250f7a54aa95dac5789943bb09a8c3763ca9e986940421ed9aa400986dd0f`).
   - **Root Cause of `v1` Empty Proposals on Degraded Photos**: Extracted from `completion_pilot/epoch_2.pth`, which trained `GatedUNet(3, 1, 24)` exclusively on flat-color + Gaussian-noise geometric rectangles/ellipses (`completion.py:synthetic_covering`).
   - **Why Historical VM-Returned `GatedUNet(3, 1, 24)` Checkpoints Cannot Replace `v1`**:
     - Audited all 4 historical VM-returned `GatedUNet(3, 1, 24)` checkpoints (`balanced_penalty_ep9`, `balanced_ctrl_ep10`, `replay_ep10`, `consistency_ep10`) on the 9 `Conditioned SFT V2` restored canvases:
       - All 4 produce **severe false-positive masks on `08_uncovered_degraded`** (`8,380` px, `12,525` px, `1,251` px, and `1,744` px respectively, vs `0` px for `v1_epoch2`).
       - All 4 still return `0` pixels on `05_white_glare_degraded` and `val_362_hair_eye_degraded`, and near-zero IoU (`0.0031–0.0218`) on `val_6_flower_mouth_degraded` because they were trained on the old 68-case mask-only `detector_expanded_review_v2` manifest before multi-family `detector_supported_review_v3` (`133` train cases) existed.
   - **Standing Policy Constraint**: Per [FACE_MASK_COMPLETION_SPEC_V1.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/FACE_MASK_COMPLETION_SPEC_V1.md#L19) and [PRACTICAL_DIRECT_DETECTOR_RESULTS.md](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/PRACTICAL_DIRECT_DETECTOR_RESULTS.md#L94), all neural optimizer training is manual L4 VM work (`Local inference and saved-evidence audits are permitted`). Furthermore, `IMPLEMENTATION`'s guided editor already uses manual/imported mask review by default (`COMPLETION_MODEL_CARD.md` line 55: *"The guided editor no longer requests automatic proposals"*).

---

## 2. Nine-Case `Conditioned SFT V2` Restored-Canvas Comparison (`R1` vs `R2`)

| Case ID | Covering Family | Reviewed Mask Pixels | Outside RGB Bytes Exact | R1 Inside-Mask Chroma Spread | R2 Inside-Mask Chroma Spread | Mean $|\text{R2} - \text{R1}|$ Inside Mask (uint8) |
|---|---|---:|---:|---:|---:|---:|
| `00_cloth_mask_degraded` | `face_mask` | 18,531 | 141,015 | 64.1882 | 64.7260 | 1.9765 |
| `03_dark_sunglasses_degraded` | `sunglasses` | 17,140 | 145,188 | 37.2845 | **36.8634** | 2.0617 |
| `05_white_glare_degraded` | `strong_lens_glare` | 2,851 | 188,055 | 43.8692 | **43.7531** | 2.7444 |
| `val_18_hand_eyes_degraded` | `hand` | 14,683 | 152,559 | 60.7464 | **59.0729** | 2.8145 |
| `val_362_hair_eye_degraded` | `obstructing_hair` (gray) | 5,802 | 179,202 | 13.2211 | **12.5458 (-5.11%)** | 3.2843 |
| `val_244_knit_scarf_degraded` | `scarf` | 14,259 | 153,831 | 30.4885 | 31.5898 | 3.9883 |
| `val_6_flower_mouth_degraded` | `other_object` | 9,425 | 168,333 | 20.5406 | **19.5415** | 2.5232 |
| `08_uncovered_degraded` | `uncovered_control` | 0 | 196,608 | 0.0000 | 0.0000 | 0.0000 |
| `09_clear_glasses_degraded` | `clear_glasses_control` | 0 | 196,608 | 0.0000 | 0.0000 | 0.0000 |
| **Total / Macro** | **9 Cases** | **82,691** | **1,521,399 / 1,521,399** | — | — | — |

---

## 3. Stage 4 Selective Drop-In Integration into `IMPLEMENTATION` — Verified

1. **Staged Model & Updated Pins**:
   - Staged [codeformer_completion_r2.pth](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/models/completion/codeformer_completion_r2.pth) (`370,773,572` bytes, SHA-256 `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`, logical SHA-256 `300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0`, `nlink=1`) while preserving `codeformer_completion_r1.pth` and `covering_detector_segmenter_v1.pth`.
   - Updated [completion_assets.json](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/implementation/completion_assets.json), [completion_assets.py](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/implementation/completion_assets.py), [completion_runtime.py](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/implementation/completion_runtime.py), [test_completion_runtime.py](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/tests/test_completion_runtime.py), and [COMPLETION_MODEL_CARD.md](file:///C:/Users/janus/.codex/worktrees/implementation-one-shot/Testing/COMPLETION_MODEL_CARD.md).
2. **Protected Restoration Invariant (`18/18` Files Unchanged)**:
   - Verified `0` modified bytes across all 18 protected `Conditioned SFT V2` restoration source, dependency, and vendor files (`git diff --name-only` touches only the 5 completion files above).
3. **Unit Tests & Live Inline Playwright Verification**:
   - Passed `77/77` core completion/restoration unit tests (`1` Windows symlink privilege skip), `45/45` additional Python unit tests, and `3/3` Node `.cjs` test suites.
   - Passed inline Playwright multi-viewport audit at `375x667`, `768x1024`, and `1280x800` (`0` horizontal overflow, `0` console/page errors) and full live browser E2E completion on `val_362_hair_eye_degraded.png` (`13.42s`, `1` neural call, `weights_sha256: ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`, `exact_outside: true`, `outside_ref: restoration-result.png`; [implementation-r2-playwright-flow.json](file:///c:/xampp/htdocs/YEAR%204/Testing_FACE_MASK_COMPLETION/scratch/implementation-r2-playwright-flow.json)).
