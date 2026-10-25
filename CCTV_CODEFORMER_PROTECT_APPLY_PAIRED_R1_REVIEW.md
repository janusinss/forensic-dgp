# Four-photo Protect/Apply paired development review

All 48 planned frozen inference outputs were saved and independently audited. This precommitted study does not support automatic or always-on surroundings blur. Keep preparation optional, retain the original bypass, and do not select a new strength from these outputs.

## What was compared

Four development photographs (10401, 11612, 11659 and 12600) each supplied a clear control, global Gaussian blur and localized Gaussian blur: 12 correlated cases, not 12 independent people. Each input was restored with frozen U96 and postfusion LR u64, both without preparation and with the same fixed Protect/Apply preparation. That produced 48 outputs, with 48 attempts, returns and saves and no unknown outcome. The supervisor exited successfully in 459.279 seconds. No optimizer, training or VM execution occurred.

Input review and masks were locked before generated outputs or clean targets were inspected. The known synthetic blur used sigma 1.2; the fixed manual preparation radius was 1.0, with guard 8 and feather 16. These different operators approximate blur, rather than matching it exactly. The observed clear control was direct 1024-to-256 BILINEAR; the separately frozen target was direct 1024-to-256 LANCZOS. All four photographs are synthetic paired development evidence, not native CCTV or FINAL evaluation.

## Results and adverse cases

Positive error deltas below mean preparation was worse. MAE uses RGB values on the 0–1 scale; no significance or usefulness threshold is inferred from these small differences.

| Frozen model | Mean whole PNG MAE delta | Whole PNG MAE worse | PNG SSIM worse | Central PNG MAE worse |
| --- | ---: | ---: | ---: | ---: |
| U96 | +0.00032875 | 11/12 | 11/12 | 8/12 |
| postfusion LR u64 | +0.00033521 | 11/12 | 11/12 | 9/12 |

Both models worsened whole PNG MAE for all four clear controls and all four globally blurred controls. In the intended localized-blur subset, whole and central PNG MAE worsened in three of four photographs per model. No preparation comparison improved both whole PNG MAE and SSIM. Photograph 12600 localized blur reduced MAE but reduced SSIM; photograph 11659 localized blur increased SSIM slightly but increased MAE. All unfavorable cases remain included. RAW whole MAE also worsened in 11/12 cases per model.

The separate model comparison favors u64 on some measurements: compared with U96, its central PNG MAE improved in all 12 cases in each preparation condition, while whole PNG MAE improved in eight and worsened in four. The adverse whole-image cases were 11612 clear/global/local and 11659 clear. These results retain u64 as the preferred experimental challenger; they do not establish a qualified successor or justify main-app promotion.

Root personally inspected every generated cell in all three exact-256-pixel sheets before reading the numerical summary. U96/u64 and original/prepared results looked similar, with small tone and edge differences and no consistent conspicuous new facial structure. Tinted glasses, frames, reflections and makeup remained visible. The peer independently verified all 84 rendered panels against saved pixels, without replacing root's visual judgment.

Protect plus guard stayed exact in the prepared inputs. Generated protected-region pixels changed in all 24 original/prepared comparisons. That is not by itself a damage verdict, but it demonstrates that preprocessing protection does not guarantee identical generated appearance. Apply covered this study's whole frame, so its empty outside-Apply region supplies no new outside-Apply preservation evidence.

## Decision and remaining work

The earlier wild5 visual benefit remains a case-specific, unpaired observation. This four-photo study provides no general validation for an automatic blur matcher. Keep manual Protect/Apply with explicit review, immutable-original previews and bypass; do not always blur an image, copy arbitrary noise/mosaic, or rerun strength sweeps on these cases. Globally soft, mixed or uncertain images can bypass. Request a clearer crop when usable facial structure is insufficient.

The broader U96/u64 preservation comparison remains separate and incomplete. Its resource-aware tail uses only the 655 unfilled slots and preserves the original 61 outputs; no candidate is promoted from runtime or archive completion. Full saved-output gate auditing, all 90 final paired sheets, independent FINAL review and production app integration remain required. Separate completion/covering development remains deferred behind restoration acceptance.

Per the user's decision, [FINAL review materials](outputs/codeformer_restoration_FINAL_review_materials_v1/REVIEW_GUIDE.md) are prepared as blank forms first. Reviewers and adjudicator remain unassigned, the protocol remains UNFROZEN and reserved FINAL pixels remain unopened.

## Evidence

- [Independent saved-output audit](outputs/codeformer_protect_apply_paired_study_v1_independent_audit/independent_all48_saved_audit_v1.json): all 240 arrays, 48 PNGs, source bindings, delivery transformations, global/regional RAW/PNG metrics, 209 journals and 99 resource observations verified; 87,593 checks passed.
- [All numerical comparisons](outputs/codeformer_protect_apply_paired_study_v1_independent_audit/all48_independent_numeric_comparisons_v1.json) and [all 24 preparation deltas](outputs/codeformer_protect_apply_paired_study_v1_independent_audit/all24_prepared_minus_original_v1.csv), retaining every adverse case.
- [Root visual review before numerical summary](outputs/codeformer_protect_apply_paired_study_v1_render_v1/root_visual_review_before_numeric_v1.json) and [independent exact panel binding](outputs/codeformer_protect_apply_paired_study_v1_independent_audit/independent_visual_panel_binding_v1.json).
- Exact sheets: [clear](outputs/codeformer_protect_apply_paired_study_v1_render_v1/visuals/01_clear.png), [global blur](outputs/codeformer_protect_apply_paired_study_v1_render_v1/visuals/02_global_gaussian_1p2.png), [localized blur](outputs/codeformer_protect_apply_paired_study_v1_render_v1/visuals/03_local_gaussian_1p2.png).
- [Finite supervisor](outputs/codeformer_protect_apply_paired_study_v1_runner_preparation/supervisor_v1.json) and [sealed plan](outputs/codeformer_protect_apply_paired_study_v1_runner_preparation/finite_paired_plan_v1.json).
