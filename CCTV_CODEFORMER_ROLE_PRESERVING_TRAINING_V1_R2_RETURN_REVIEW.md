# R2 return: trained model created; preservation remains unqualified

**18 October 2026: 50 new committed Adam updates; genuine inference checkpoint
created; restoration quality stopped.** The model is available for explicitly
experimental local comparison. It has not replaced the main application's
default restoration path. User acceptance and independent FINAL review remain
pending. This is our fine-tuned adaptation of CodeFormer.

## Actual model and training evidence

[codeformer_role_preserving_r2.pth](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth>) is 376,609,465 bytes. SHA-256:
`d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896`. Canonical 515-tensor digest: `a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f`.
The inference file contains only `params_ema`: this is the existing loader's
compatibility key, with exact final trained weights and **no new EMA averaging**.
Its full training source checkpoint, Adam/RNG/sampler state, archive, failures
and untouched CodeFormer/original DGP checkpoints remain preserved.

The VM committed 50 new Adam updates across 200 microbatches: 100 normal and 100
dark observations from 100 selected TRAIN references. These are optimizer
updates, **not 50 epochs**. The candidate started from the historical full-coarse
50-update research model with fresh Adam; historical optimizer/RNG/sampler state
was not continued. Independent inspection confirmed all 515 initial tensors match
that parent, all 350 trainable tensors changed, all 165 frozen tensors stayed
exact, and all 350 saved Adam states reached step 50. The final checkpoint and
stopped-state alias have the same committed model/Adam/RNG/sampler/schedule/counters.

The 1,854 ordered call reservations, 200 TRAIN records and 50 attempt/commit records
match the finite source schedule. Reservations are not independent live profiler
measurements. Recorded worker duration was 612.674 seconds and actual optimization
185.172 seconds. Recorded phase/RSS/CUDA/disk use remained within the declared
limits; memory and disk figures are sampled evidence.

R2 passed its revised initialization check and the before-Adam scoped gradient
prerequisites. Only the cross-object initializer logit numerical criterion changed
to the declared PyTorch 2.9 FP32 testing defaults plus exact robust token choices.
The restoration objective, exact scoped/unscoped training witness and all quality
requirements stayed unchanged. The exact earlier low-level rounding cause is
unknown. Training stopped after final evaluation with:
`Preservation or dark-retention quality gate failed; actual trained50state retained`.
The independently reproduced outcome is **preservation FAIL; dark-retention PASS**.

## Paired development results

All 812 new RAW/PNG pairs and 812 retained untouched/parent pairs were independently
checked. The initial R2 snapshot reproduces the parent in all 406 cases. The 406
development cases comprise 358 paired synthetic cases and 48 unpaired native CCTV
cases. RAW floats are separate from downloaded floor-quantized PNGs; no sharpening
or display brightening was added. Lower MAE and higher SSIM are better. These are
paired development readouts, not identity or native CCTV accuracy.

FFHQ validation, reported separately:

| Profile | Cases | Untouched MAE / SSIM | Parent MAE / SSIM | Trained R2 MAE / SSIM |
|---|---:|---:|---:|---:|
| clear | 53 | 0.018031 / 0.917899 | 0.024786 / 0.920579 | 0.030824 / 0.922882 |
| blur | 53 | 0.065760 / 0.471176 | 0.069366 / 0.554005 | 0.082335 / 0.535750 |
| lowlight | 53 | 0.153549 / 0.527545 | 0.101779 / 0.614690 | 0.080872 / 0.627295 |
| compound | 53 | 0.203687 / 0.389666 | 0.138071 / 0.520845 | 0.106888 / 0.538211 |

The lowlight and compound cohorts improve over the retained parent on both mean
PNG MAE and SSIM. Clear and normally blurred inputs still change appearance or
tone. Preservation compares 61 clear controls and 53 normal-blur controls against
untouched CodeFormer, with unchanged mean/worst error-increase limits 0.01/0.03:

| Control | View | Metric | Mean error increase | Worst increase | Cases above 0.03 |
|---|---|---|---:|---:|---:|
| clear | raw | mae | 0.014727 | 0.081838 | 7 |
| clear | native_png | mae | 0.014131 | 0.080445 | 7 |
| normal_blur | raw | luminance_bias | 0.056622 | 0.115966 | 46 |
| normal_blur | raw | coarse_rgb_mae | 0.037619 | 0.094972 | 31 |
| normal_blur | native_png | luminance_bias | 0.053521 | 0.112063 | 43 |
| normal_blur | native_png | coarse_rgb_mae | 0.035993 | 0.093069 | 29 |

The historical paired `dataset/asian_faces` folder is a source label, not a claim
about a person's ethnicity or capture location. Its four-case cohorts remain
separate from FFHQ and from the thumbnail source:

| Profile | Cases | Untouched MAE / SSIM | Parent MAE / SSIM | Trained R2 MAE / SSIM |
|---|---:|---:|---:|---:|
| clear | 4 | 0.015265 / 0.900890 | 0.039684 / 0.908359 | 0.052799 / 0.907077 |
| blur_lr24 | 4 | 0.041622 / 0.558804 | 0.062960 / 0.641276 | 0.078086 / 0.633864 |
| lowlight_lr32 | 4 | 0.173051 / 0.393203 | 0.124557 / 0.629311 | 0.100540 / 0.631911 |
| compound_lr24 | 4 | 0.235406 / 0.389055 | 0.175826 / 0.557560 | 0.140487 / 0.593366 |

Historical paired `dataset/thumbnails128x128`, also separate:

| Profile | Cases | Untouched MAE / SSIM | Parent MAE / SSIM | Trained R2 MAE / SSIM |
|---|---:|---:|---:|---:|
| clear | 4 | 0.018487 / 0.915988 | 0.022272 / 0.925862 | 0.026945 / 0.922359 |
| blur_lr24 | 4 | 0.075360 / 0.435996 | 0.070374 / 0.532831 | 0.080858 / 0.510614 |
| lowlight_lr32 | 4 | 0.180457 / 0.392385 | 0.126728 / 0.527998 | 0.103242 / 0.528560 |
| compound_lr24 | 4 | 0.247288 / 0.310918 | 0.177311 / 0.422012 | 0.145633 / 0.445646 |

The complete per-source/profile RAW/PNG comparisons, including light and motion
profiles and paired PSNR, remain in [source_profile_comparisons.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/pixels_v1/source_profile_comparisons.json>).
No source group is treated as evidence of Zamboanga performance.

## Native CCTV and root visual review

ChokePoint 24 and QMUL-SurvFace 24 native development crops have no aligned clean
reference. Their MAE/PSNR/SSIM remain null; no paired score is assigned. Source,
terms, provenance, resolution and overlap metadata remain with the retained input
bindings. FINAL pixels were not used, and photograph/source IDs do not establish
independent human identities.

Root personally viewed all 34 planned sheets covering 406 development cases.
Saved cells preserve exact 256-pixel evidence, but most tall sheets were resized
by the viewer from 1280×3415 to 768×2048. Root noted fewer severe artifacts on many
dark/compound outputs, while clear/normal faces could become brighter, smoother
or change visible detail. Several very degraded QMUL crops produced speculative
features or unusable structure. Request a clearer crop in those cases.
[root_visual_review_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/root_visual_review_v1.json>) records the per-sheet notes and limits.
This is development review, not independent FINAL evaluation or user acceptance.

## Local experimental testing

[Open the local testing guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_R2_LOCAL_TEST_GUIDE.md>). The measured research URL
was http://127.0.0.1:52104/restoration-test at final browser verification; runtime ports can change after restart.
The existing application's normal model selection/design remains separate from
this opt-in restoration-only route. It offers input/resize, untouched CodeFormer
and the trained R2, original/R2 selection overrides, PNGs and a comparison bundle.
Clear-glasses/visible appearance must be preserved. Coverings remain in this
restoration-only test; detector/mask-review/completion work remains deferred until
restoration acceptance.

Root's bundled inline Playwright checked actual model inference and download/refusal
flow, not a substituted mock. Its functional evidence is bounded to the tested
inputs and layouts. Missing input review and unusable structure produce HTTP 422;
operator input review is **not a calibrated automatic quality threshold**.
No automatic generation from block-like or heavily covered inputs is qualified.
The final browser/download evidence is recorded separately:

[scratch/codeformer_r2_browser_flow_v3.json](<C:/xampp/htdocs/YEAR 4/Testing/scratch/codeformer_r2_browser_flow_v3.json>)

[scratch/codeformer_r2_final_layout_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/scratch/codeformer_r2_final_layout_v1.json>)

[outputs/codeformer_role_preserving_training_v1_r2_return_review/local_test_v1/download_audit_v2.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/local_test_v1/download_audit_v2.json>)

[outputs/codeformer_role_preserving_training_v1_r2_return_review/research_app_final_source_review_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/research_app_final_source_review_v1.json>)

[outputs/codeformer_role_preserving_training_v1_r2_return_review/research_benchmark_hardlink_guard_review_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/research_benchmark_hardlink_guard_review_v1.json>)
Functional browser success does not override the failed preservation checks,
establish CPU/CUDA byte parity, prove identity recovery or qualify all real CCTV.

## Milestone boundary and next work

A genuine trained restoration model now exists and can be tested locally.
The immediate next action is user comparison feedback on visible eyes, brows,
nose, mouth, jaw and appearance against the input and untouched benchmark.
No next pilot, extra epochs, automatic resume or main-model promotion is prepared
by this milestone. A later model change needs the retained failure evidence and
a justified finite manual L4 plan. The full goal still requires useful preservation,
user acceptance, independent FINAL review and accepted app Auto/override/PNG flow.
Separate covering detection and seven-family plausible completion follow restoration
acceptance; exact hidden identity must not be claimed. Earlier storage offload is
a completed separate task; no new deletion or migration occurs here.

Evidence: [transport_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/transport_v1.json>),
[state_audit_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/state_v1/state_audit_v1.json>),
[audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_return_review/pixels_v1/audit.json>),
[model_export_v1.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_model/model_export_v1.json>). Archive SHA-256:`d47c88c8889b9bda66eeba03f498c7ab81173e4a75ffa2572a75049e73084689`;
protocol:`82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a`. Overall goal remains incomplete.
