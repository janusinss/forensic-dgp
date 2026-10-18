# CodeFormer adaptation V1: returned results, 15 October 2026

**Real training completed 200 of 1,200 planned updates, then the declared clear-face preservation stop fired. The returned model has visible improvements and measurable regressions. It is retained for research and is not an accepted replacement. Do not relaunch or resume this recipe unchanged.**

[All 52 comparison pages](outputs/codeformer_adaptation_v1_return_review/visual_sheets/INDEX.md) show every one of the 406 cases at updates 0 and 200. Columns are input, paired clean target or unpaired note, untouched CodeFormer, and our stopped adaptation. [Severe-blur examples](outputs/codeformer_adaptation_v1_return_review/visual_sheets/comparison_01.png), [dark compound examples](outputs/codeformer_adaptation_v1_return_review/visual_sheets/comparison_15.png), [native ChokePoint](outputs/codeformer_adaptation_v1_return_review/visual_sheets/comparison_41.png), and [the clear control that stopped training](outputs/codeformer_adaptation_v1_return_review/stop_case_comparison.png) make the tradeoffs visible. User acceptance is still pending.

## What the returned evidence proves

The 2,594,787,706-byte results archive has SHA-256 `98b101f8d543c34de033ec0583114681499927bdc8ed872b8f31d4d8d5edac92`. Its checksum/export companions agree. All 2,338 regular members, 2,837,257,704 payload bytes and 720 retained packet inputs were independently checked. The imported namespace and import journal match. Export completion means the stopped evidence was saved; `run_results_present` is false and `failure_present` is true.

Snapshot 0 matches every tensor of the untouched CodeFormer initializer. All 350 intended trainable tensors changed by update 200; all 165 frozen restoration codebook/generator tensors remained exact. Adam states, learning-rate schedule, saved sampler/RNG and stopped/update-200 model states were checked independently on CPU without constructing or running a neural network. Final teacher/perceptual states were not re-exported, so their unchanged-state receipts are not a second independent comparison of those final tensors.

The run used 800 TRAIN exposures over all 391 approved references: 373 references twice and 18 three times, or **2.046 reference-pass equivalents**. This is not 200 epochs. Optimization took 341.24 seconds; the worker took 577.79 seconds. Peak reserved CUDA memory was about 7.95 GiB. CUDA, gradients, intended parameter changes, initial inference parity and timing prerequisites passed. No update-600 or update-1,200 result exists.

## Why it stopped

Across 61 clear controls, the mean regression passed, but one individual control failed in both RAW and downloaded PNG views:

| View | Mean MAE increase, limit 0.01 | Worst individual increase, limit 0.03 | Result |
| --- | ---: | ---: | --- |
| RAW 256 | 0.00142718 | **0.04562926** | Fail |
| Delivered PNG | 0.00205938 | **0.04431654** | Fail |

The sole case above the individual limit in each view is `comparison_va_asian_09306_clear`. The increase is about 11.64/255 RAW and 11.30/255 PNG, exceeding the allowed 7.65/255. Its PNG SSIM also falls, 0.891817 to 0.867407. Visual inspection shows brighter appearance and facial-feature changes. This is a quality stop, not a broken optimizer, transfer error or PNG-rounding defect. The validation failure must not be moved into TRAIN to make the requirement pass.

## Actual improvements and tradeoffs

The following paired synthetic photographic results use the same 53 HQ validation references per profile and identical saved inputs for both checkpoints. They are development evidence, not native CCTV or identity-recovery evidence.

| Profile | Original PNG PSNR | Update-200 PNG PSNR | Change | PNG SSIM change |
| --- | ---: | ---: | ---: | ---: |
| Clear | 31.416 | 32.331 | +0.915 dB | +0.02897 |
| Light degradation | 28.233 | 27.171 | -1.062 dB | +0.01664 |
| Normally exposed blur | 20.485 | 15.539 | **-4.946 dB** | **+0.08744** |
| JPEG | 30.805 | 31.400 | +0.595 dB | +0.02336 |
| Lowlight | 15.229 | 21.419 | **+6.190 dB** | **+0.12879** |
| Compound dark/blur | 12.602 | 20.173 | **+7.572 dB** | **+0.18138** |

All 53 normally exposed blur cases worsen MAE/MSE, yet 51 improve SSIM. The root reviewer inspected all 52 pages: many severe-blur and compound outputs have fewer false textures, false glasses and inconsistent facial features than the initializer. Several remain overly smooth, change expression or appearance, or miss target details. Thus neither "no improvement" nor "all blur quality became worse" describes this run accurately. An aggregate score would hide these profile-specific tradeoffs.

The clearest measured problem is inappropriate brightening. Normally exposed blur output mean luminance rises from 0.41458 to 0.55385 against target 0.41691: an average **35.51/255 increase from original CodeFormer** (the resulting signed bias relative to the target is about 34.92/255), with all 53 cases shifting upward. Lowlight receives almost the same shift, where it usually helps. The saved-pixel error decomposition attributes essentially all of blur's extra luminance MSE to global bias; this analysis does not provide a deployable correction or establish exact facial identity.

The actual loss routing was unbalanced: only **20/162** blur exposures received pixel/LPIPS losses, compared with **188/188** lowlight exposures. The other blur examples still received code and latent supervision. A rule intended to avoid imposing unsupported fine detail on sub-32-pixel observations also removed their direct photometric constraint. This imbalance and the brightness drift are measured facts. Which encoder/AdaIN/fusion route caused each change remains a hypothesis until a controlled comparison tests it.

## Native CCTV and visual-review limits

ChokePoint and QMUL-SurvFace remain separate, unpaired development sources. No clean-reference PSNR/SSIM was computed for them. ChokePoint's update-200 images often soften detail relative to original CodeFormer; this review does not establish a consistent native improvement. Many severely degraded, turned or obstructed QMUL examples still show implausible structures, glare-like areas or insufficient facial information in both checkpoints. A clearer, more frontal or less-covered crop is needed for such cases; synthetic gains do not qualify these outputs.

Every comparison page was opened by the root AI reviewer at its original sheet resolution. Each face tile is a declared 192-pixel review thumbnail from the retained native 256-pixel PNG; there is no sharpening or exposure correction. The full 256 RAW/PNG files remain retained. This descriptive AI review is not blinded human/user acceptance or independent reserved-final evaluation. FFHQ pretrained exposure, unknown subject-level overlap and repeated development use remain limitations. No ethnicity, Zamboanga, hidden-identity or completion-family result is claimed.

## Next design direction

Keep the CodeFormer architecture and begin a changed recipe from its untouched initializer. Test matched normal/dark TRAIN degradations, coarse photometric supervision at every severity, and original-behavior preservation on TRAIN clear/light controls. Use low-resolution variants of appropriate TRAIN references rather than fitting the failed validation face. Preserve the current inference policy and all old individual RAW/PNG stops. A prospective normal-blur photometric guard should prevent the broad exposure drift from hiding behind favorable lowlight averages.

This is a justified design direction, not permission to resume the failed checkpoint or add epochs unchanged. A new finite recipe, independent review, verified transfer and human manual VM launch are still required. App/default changes and completion/detector development remain deferred until accepted restoration.

## Audit records

- [Independent archive/import verification](outputs/codeformer_adaptation_v1_return_review/archive_review/independent_archive_import_v1.json) and [training/state review](outputs/codeformer_adaptation_v1_return_review/training_review/INDEPENDENT_TRAINING_REVIEW_V1.md).
- [All RAW/PNG pixel checks](outputs/codeformer_adaptation_v1_return_review/metrics_review/RETURN_METRICS_REVIEW_V1.md) and [supplementary paired SSIM](outputs/codeformer_adaptation_v1_return_review/metrics_review/secondary_metrics_v1/SSIM_REVIEW_V1.md).
- [Measured photometry and next-design review](outputs/codeformer_adaptation_v1_return_review/post_v1_review/POST_V1_DESIGN_REVIEW.md).
- [Independent scientific cross-review](outputs/codeformer_adaptation_v1_return_review/publication_review/SCIENTIFIC_PUBLICATION_REVIEW_V1.md).
- [Complete comparison index](outputs/codeformer_adaptation_v1_return_review/visual_sheets/INDEX.md) and `visual_review_v1.json` beside it bind the reviewed pages to their hashes.
