# Filtered-face stress test — 18 October 2026

**R2 works on some blurred inputs but is not ready for all of these degradations.**
The user supplied five images to test readiness for real-life restoration and
confirmed that unblurred originals are unavailable. They are user-supplied,
unknown-origin, unpaired stress examples with unknown reuse terms. They are not verified CCTV captures,
clean training pairs, independent FINAL identities or Zamboanga evidence.

[View the comparison](outputs/codeformer_user_wild_stress_v1/comparison_v1.png).
Each image panel preserves the app's exact 256-pixel output. No sharpening,
display brightening, completion or mask removal was applied.

## Actual local results

| Input | Observed issue | Actual result and limit |
| --- | --- | --- |
| wild1 | Broad smooth blur | Both models create a sharper, plausible face. R2 is brighter and smoother. Fine facial details are unverified; this remains a borderline stress comparison. |
| wild2 | Repeated horizontal streaks | Both outputs retain artifacts around the eyes and nose. R2 brightening does not remove this failure. The exact filter or capture process is unknown. |
| wild3 | Strong block artifacts | Untouched CodeFormer smooths more of the facial blocks. R2 retains conspicuous blocks around eyes, nose, cheeks and mouth, despite detailed hair. This is a restoration regression worth targeting. |
| wild4–5 | Coarse face mosaic / strong localized face blur | Input-only review judged central structure insufficient. The app returned the clearer-image request; no restoration or download was generated. |

Root inspected all five prepared inputs before generation and all six downloaded
result PNGs afterward. A separate reviewer inspected the five original files and
six outputs. For wild1, root and one reviewer allowed a cautious comparison from
coarse visible geometry; another preferred requesting a clearer crop. That
disagreement is retained. A plausible generated face does not resolve it or prove
the lost details are correct.

The three comparisons used the actual existing local app, untouched CodeFormer
and the genuine R2 update50 checkpoint, fidelity1, AdaIN enabled, unchanged
256-to-512 inference input and 256 output. They took approximately 12 seconds
each on the measured local CPU. All six RAW arrays, six PNG downloads and three
ZIP downloads passed independent saved-artifact checks: source hashes, native
geometry, observed support, model lineage, finite FP32 arrays, floor conversion,
protected padding and ZIP/response agreement. Nine CPU audit fixtures passed.

Both insufficient inputs returned HTTP422 with the clearer-crop message and no
result/download payload. **This was operator-directed input review, not a new
automatic threshold or an automatic image-quality detector.** Source inspection
places this check before model loading/inference, but no live per-request neural
counter was measured. Playwright checked the empty page and an actual result at
375, 768 and 1280 pixels, with no overflow or page exceptions. This establishes
bounded functional behavior, not universal restoration usefulness.

## What this changes in the improvement plan

The user's accepted useful brightening remains a benefit to preserve. These
examples add explicit artifact-removal requirements: judge visible eyes, brows,
nose, mouth and jaw, rather than hair detail, brightness or a realistic-looking
output alone. Brightening and removal of block/streak artifacts are separate
observations. This is face-restoration work; blur alone does not call for the
separate covering/completion model.

The current R2 loader reuses the exact 200 retained matched TRAIN observations.
Their V2 camera generator includes Gaussian blur, downsampling, JPEG, noise and
normal/dark exposure variants. Its motion-kernel field is zero and is not applied
by the renderer. It has no explicit repeating-line corruption, nearest-neighbor
face mosaic or spatially localized face-blur operator. Historical V1 compound
generation does include a horizontal motion convolution; that is different from
repeating lines, and R2's new selected observations are loaded from saved V2
pairs. This bounded source finding says nothing about unknown original CodeFormer
pretraining coverage or the exact process that made these five pictures.

Before a new training pilot, prepare and audit known synthetic variants from
approved clean TRAIN sources that cover the demonstrated artifact families.
Keep the five user images as a separate visual challenge set; do not use their
restored outputs as clean targets. Confirm any new data's source/terms and
overlap limits. Match useful blur/block/streak severity separately from
information-insufficient mosaic/blur cases. Calibrate a future input-only
abstention rule with both readable degraded faces and unusable faces; a file's
large dimensions or crisp background are insufficient evidence.

Use fixed unmodified RAW/PNG comparisons against untouched CodeFormer and the
retained parent/R2. Useful dark-image gains, normal/clear preservation and these
artifact failures must all remain visible. No existing quality gate is relaxed;
no new loss, automatic threshold or training recipe is selected in this test.
New actual training remains a finite, verified manual L4/tmux run. Extra epochs
on the unchanged failed recipe are not justified by this five-image test.

## Readiness and retained evidence

Without aligned unblurred references, no PSNR/SSIM, identity accuracy or exact
recovery claim is assigned to these examples. They reveal current limitations
and support user review. They do not establish readiness for all real-life CCTV.
The restoration-first goal remains active, with independent FINAL review and
qualified automatic selection/input handling still pending. Covering/completion
development remains deferred until restoration acceptance.

- [Input inventory and source hashes](outputs/codeformer_user_wild_stress_v1/input_inventory_v1.json).
- [Independent visual review](outputs/codeformer_user_wild_stress_v1/visual_review_v1.json).
- [Actual browser requests and nine downloads](scratch/codeformer_r2_wild_v1/request_manifest.json).
- [Independent RAW/PNG/download audit](outputs/codeformer_role_preserving_training_v1_r2_return_review/wild_examples_v1/download_audit_v1.json)
  and [nine CPU fixtures](outputs/codeformer_role_preserving_training_v1_r2_return_review/wild_examples_v1/CPU_source_review_v1.json).
- [Earlier R2 improvement review](CCTV_CODEFORMER_R2_IMPROVEMENT_REVIEW_V1.md)
  and [local testing guide](CCTV_CODEFORMER_R2_LOCAL_TEST_GUIDE.md).
