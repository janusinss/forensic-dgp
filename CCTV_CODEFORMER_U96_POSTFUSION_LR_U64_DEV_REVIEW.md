# U96 versus post-fusion LR u64: fixed development review

22 October 2026. **Experimental candidate; U96 remains the incumbent.** The new u64 shows modest numerical progress and received positive user feedback, but this review does not establish consistent conspicuous native facial-detail improvement or broad model qualification.

## What was completed

The fixed 68-case cohort contains 36 paired synthetic cases (four photographs, nine profiles), eight native QMUL-SurvFace cases and 24 native ChokePoint cases. Terms, input-only labels, original order, appearance flags and 40 excluded zero-call controls remain unchanged. The original 47 saved outputs were retained and the distinct recovery generated only the remaining 89: 136 known saved returns, 137 reservations including the earlier failed pre-call reservation. All 454 original files remain unchanged. The original Windows status-write failure and historical unknown reservation are retained; the recovery exited zero without timeout or unknown tail outcome. No optimization or new training occurred locally.

Independent metadata checks passed. The saved-output audit passed all 3,630 integrity checks, verifying every native512-to-unclipped/clamped/bilinear RAW output and exact floor-quantized PNG. Padding outside observed support remains exact for every output. Root personally reviewed all 17 planned sheets, 68 rows and 136 generated cells at exact256. These are integrity and review facts, not a new quality gate or a preservation qualification for all possible inputs.

## Paired synthetic findings

The table uses u64 minus U96 for MAE/PSNR/SSIM; negative MAE and positive PSNR/SSIM favor u64. Adjacent-error reduction is relative to U96 against the aligned target. Whole-image and central-region values are separate; the central local-blur improvement is much smaller than the whole-image value. Each row averages only four DEV photographs. These metrics are separate from the small adaptive TRAIN feature-error gates.

| Profile | PNG MAE difference | Whole adjacent-error reduction | Central reduction | PSNR difference dB | SSIM difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| clear | -0.001577 | +2.1114% | +2.2362% | +0.4733 | +0.002058 |
| gaussian_2p4 | -0.001279 | +1.0564% | +1.0340% | +0.2272 | +0.002358 |
| gaussian_6p0 | +0.000813 | +0.2384% | +0.1798% | -0.0502 | -0.000781 |
| anisotropic_h5_v0p7 | -0.000260 | +0.7501% | +0.6374% | +0.0596 | +0.000776 |
| anisotropic_h0p7_v5 | -0.000393 | +0.5837% | +0.3391% | +0.0718 | +0.001391 |
| repeated_rows_4_blend0p75 | -0.001462 | +1.1389% | +0.8241% | +0.1835 | +0.002131 |
| global_mosaic_8 | +0.000260 | +0.1443% | -0.1212% | -0.0073 | -0.002219 |
| local_gaussian_3p0 | -0.001501 | +1.8541% | +0.2425% | +0.3019 | +0.000925 |
| local_mosaic_8 | -0.001085 | +1.0597% | -0.0803% | +0.1543 | -0.001869 |

MAE was lower in seven of nine profile means, whole adjacent error in nine, central adjacent error in seven, PSNR higher in seven and SSIM higher in six. Strong Gaussian blur and global mosaic worsened MAE, PSNR and SSIM. Global/local mosaic worsened central adjacent error; local mosaic also worsened SSIM. No thresholds were changed or new aggregate pass invented. Clear-image paired error improved on average, but inherited fine-feature and makeup deviations remain visible in both models. Localized blur and mosaic remain evident; u64 does not solve strong anonymizing degradation.

## Native unpaired findings

QMUL-SurvFace and ChokePoint remain separate sources without aligned clean targets. Candidate-versus-U96 PNG observed-region MAE averaged 0.00450209 for QMUL and 0.00389327 for ChokePoint. These measure output change only. Native PSNR/SSIM, recovered identity, ethnicity and Zamboanga performance are not claimed.

Root found small tonal/edge changes and broadly similar generated structure across arms, with existing softness, glare, lighting, eye and turned-face limitations. No consistent conspicuous native detail improvement was established. Existing input-only labels were not changed after seeing results. Independent FINAL and broader preservation remain outstanding.

## Human feedback and local test

The user reported: “The new u64 looks clearer and preserves appearance.” This is positive unblinded development feedback. The exact uploaded case was not identified, so no measured case binding or independent FINAL acceptance is inferred.

[Experimental comparison page](http://127.0.0.1:59807/restoration-postfusion-test) and [test guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LR_U64_TEST.md>) compare identical prepared inputs with U96 and the complete R2+U96+terminal-u64 candidate. The source-reviewed frozen CPU model has a separate four-case L4 correspondence witness. The experimental browser flow passed 375/768/1280 layouts, actual PNG/ZIP download readback, input-review refusal and stale-result clearing. Its saved outputs matched the corresponding fixed-DEV outputs. A concurrent409 timing assertion was not established and its earlier test stop remains retained. This is separate-page functionality, not the complete application or model qualification.

## Decision and next work

Retain u64 as the preferred experimental challenger for further review, with U96 as the unchanged incumbent/control. Do not switch the main app default, append epochs or auto-launch another pilot based on this result. Retain all original models, failed states, optimizer/RNG/sampler evidence, source terms and splits. The small TRAIN screen passed; fixed DEV demonstrates mixed modest progress, not a failed export and not a fully qualified successor.

Next finish broader clear/degraded preservation and source-separated development review, then freeze an independent FINAL evaluation protocol before held-out outputs are examined. The protected-surroundings workflow remains optional and separate: immutable original upload, every protected input pixel unchanged, explicit separate Protect/Apply painting with erase, preview/correction/confirmation/bypass, blur only where permitted with an outside transition. Separate Apply painting is still an implementation gap; always-on blur and arbitrary noise/mosaic copying are not justified. Completion/removal-mask model development remains deferred until restoration is satisfactory. The active goal remains incomplete.

## Evidence

[Saved-pixel audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_aggregate_audit_v1/saved_pixel_audit_v1.json>), [independent terminal metadata](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_metadata_audit/independent_terminal_metadata_v1.json>), [source-separated numerical summary](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_aggregate_audit_v1/numeric_summary_v1.json>), [all17 exact-cell sheets](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_aggregate_audit_v1/visual_manifest_v1.json>), and [root visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_aggregate_audit_v1/root_visual_review_v1.json>). Earlier pending flags remain immutable; the companion root review closes only the documented DEV-sheet review.
