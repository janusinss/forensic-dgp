# Supplemental source recovery — 14 October 2026

The original completion archive's **13 regular payloads** all match the retained
outputs by size and SHA256. The archive carries weights, full state, metrics,
split/config and previews; it contains no source or VM commit receipt.

Ten historical source files were recovered read-only from Git commits
`795767ab52728b44200110230888987affe6fa7c` and
`8b8efcc57a28b7c55a62d0e36f1fbf36336b938a`. Git blob IDs and SHA256 agree.
The earlier version defaults to radius-3 external feathering; the later version
changes that default to zero and records the mask-only composition policy.
The data-renderer bytes are identical across both versions and today's file.
This strengthens the target-code ancestry evidence. It does **not** prove the
exact VM fit revision: the old config and return archive have no source pin.
Historical neural outputs and their rejected selection remain unchanged.

[Recovered source and exact archive bindings](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_target_provenance_review_v1/source_history_review.json>).
Source files were parsed, not imported or executed. No old training was resumed,
no app/model/mask was changed and no threshold was selected. A future exact
resume still requires the original fit source and environment, not a source
candidate chosen from matching defaults. The original review follows intact.

# Detector target provenance review — 14 October 2026

**The retained auxiliary detector/completion checkpoint completed two epochs,
but those epochs contain 19,000 optimizer updates. It is still unqualified.**
Its good procedural-validation overlap did not translate into reliable masks on
the exposed natural covering photographs. Extending the same recipe is not a
demonstrated solution. This count belongs to the separate mask/completion model;
it does not describe the primary DGP's training history.

[Read-only provenance and metric recount](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_target_provenance_review_v1/review.json>).
[Current confidence maps and directly reviewed masks](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETECTOR_CONFIDENCE_V1_R1_RESULTS.md>).

## What the retained training state proves

The checkpoint is `outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth`,
SHA256 `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`.
It is the same segmenter fingerprint used by the app confidence diagnostic.
Read-only loading of `last_state.pth` confirms epoch 2 and **100 optimizer state
tensors, each with step 19,000**. Every exported epoch-2 model tensor equals the
saved EMA tensor. No model was constructed, executed, displaced or fitted in
this review; reading optimizer state does not perform an optimizer update.

The preserved split has 76,000 TRAIN source paths and 4,000 validation source
paths. TRAIN has 9,500 paths in `asian_faces` and 66,500 in
`thumbnails128x128`; validation has 500 and 3,500 respectively. Membership has
no repeated path or TRAIN/validation path intersection. These folder names do
not establish ethnicity, capture region or identity disjointness. Reencoded
duplicates and shared identities are not ruled out by path checks. The full
original dataset-content signature is recorded, not recomputed in this review.

The config records batch size 8, 256×256, width 24, seed 42, learning rate
1e-4 and a full two-epoch run. Its original fit used restorer fingerprint
`b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c`;
that is not the current permanent DGP fingerprint. Keep this component ancestry
explicit when comparing or resuming. The original app DGP remains unchanged at
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

`best_selection.json` records epoch 0: **“Restoration-only baseline; no completion
weights selected.”** An exported non-smoke epoch checkpoint and a loader that
accepts it are not selection or quality evidence. The retained epoch-2 segmenter
is an unqualified proposal source; its unused generator weights were loaded by
the recent detector container but no generator forward ran.

## Recount of the original procedural validation

All three 40,000-row records—baseline, epoch 1 and epoch 2—were independently
reduced from their saved case measurements. Each contains the same 4,000
validation paths, each with ten kind/condition views. These are not 40,000
independent faces or CCTV captures. Every global mean and every visible-error
group mean matches its saved summary exactly; this is saved-measurement replay,
not fresh inference or a verification of every historical image pixel.

| Epoch-2 measurement | Recounted value | Meaning |
|---|---:|---|
| Mean procedural mask IoU | 0.9748324102 | Overlap on the synthetic target task |
| Global visible MAE | 0.0244323174 | Above baseline 0.0244178421 |
| Global visible MAE increase | 0.0000144753 | Exceeds unchanged 0.000001 tolerance |
| Clear original false-mask cases | 62 / 4,000 | Empty reference, nonempty prediction |
| Clear degraded false-mask cases | 94 / 4,000 | Empty reference, nonempty prediction |

Six visible-error groups exceed their baseline tolerance: clear lower/eyes/
object/irregular views and both uncovered conditions. Therefore better hole
error and mask overlap do not make the original completion checkpoint pass its
visible-preservation requirement. This review does not relax that requirement.
The historical 400-case synthetic retention benchmark uses a different cohort
and pixel aggregation; its 0.9746899906 IoU must not be substituted for this
40,000-view mean, or for natural-photo or native-CCTV performance.

## Target and degradation interpretation

The inspected current data code declares `none`, `lower`, `eyes`, `object` and
`irregular`. Its coverings are geometric polygons, rectangles or polylines
filled with randomized color/noise, not rendered examples of all seven natural
covering families. Current degradation uses OpenCV blur, 64–256-pixel
downsampling at size 256, noise and JPEG, with an enlarged target support.
Those inspected target semantics differ from natural hands, hair, patterned
cloth, lens reflections and objects whose appearance is already in a photograph.

This is a **target/appearance-gap hypothesis**, not a uniquely proved failure
cause. Current source bytes are not a cryptographic pin of the original fit
implementation. For example, today's trainer inserts `compositing_policy`,
whereas the old saved configuration lacks that field. The saved validation
records prove the five target-kind groups and their measurements; they do not
prove every historical renderer instruction. Preserve the original fit code
and environment when available; do not claim an exact resume from current code
and weights alone.

The fixed practical gallery uses 128-pixel downsampling, blur radius 1.2 and
noise sigma 3. Its downsampling and noise magnitudes are inside the inspected
training ranges. Pillow/OpenCV kernels and JPEG policies differ, and renderer
equivalence was not established. Thus merely declaring these review images
“too severely downsampled for training” is unsupported. The detector also misses
several undegraded photographs. Adding harsher degradation alone has not been
established as the cure.

The newly recorded confidence maps show empty proposals on 4/14 covered original
views and 11/14 matched degraded views, plus an uncovered-face false proposal.
Approximate operator footprints are assistance, not hidden-face or segmentation
truth. This evidence motivates reviewing natural covering semantics and learned
features together; a tiny mask margin or a scalar threshold change does not
resolve the demonstrated missing seeds and visible-region extra selections.

## Earlier experiments remain binding

Four original result-file hashes match the retained history reconciliation,
share the same epoch-2 parent and V2 label-manifest fingerprint, and retain their
reused-control hash links. They were read, not rerun. Historical local fitting
predates the VM-only rule and provides no authorization for new local fitting.

1. Three covered TRAIN examples and one clear control could be fitted in the
   historical 80-step diagnostic. This rejects a universal inability to learn
   any whole mask; it does not prove broad retention or generalization.
2. Raising learning rate improved some real training fit but regressed the
   synthetic benchmark. An unchanged higher-rate recipe is not qualified.
3. Removing replay or the visible penalty did not solve the tested preservation
   tradeoff. Reusing their names or settings is not a new justified experiment.
4. Corrected-label evaluation was independently recounted for four models,
   two label versions and three strata: 24 aggregates. Initial real-validation
   IoU changed only from 0.0603962 to 0.0607782. The known mannequin stays a
   separate reported stratum; the correction does not explain all missing masks.
5. The historical 99-point TRAIN threshold diagnostic already found the initial
   detector's best sampled IoU only 0.1030018, at threshold .01 with visible
   false positives. No new threshold sweep or app threshold selection occurred.

These are separate historical cohorts and protocols, not a matched causal
ranking against the 36 current photographs. The existence of improved tiny-
cohort fit does not qualify an architecture for the full covering scope.

## Consequence for the full objective

A later detector design must resolve target semantics, original-code provenance,
natural covering/degradation transfer and clear-face specificity together, while
retaining earlier failures and unchanged safeguards. It needs a distinct,
finite, prospectively frozen manual-L4 test if the review justifies actual
training. More epochs, a new encoder or another loss sweep is not selected by
this saved-state review. Completion-generator quality and automatic versus
assisted estimates remain separate requirements.

The primary restoration design question remains pending after the joint,
RAW-only and nearby-gradient recipes failed. No fourth restorer recipe or
epoch continuation is prepared while it awaits discussion.
[Required restorer discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md>).
Training, derivatives and parameter changes remain manual on the existing L4;
direct VM access remains maintenance only. This review used zero neural forwards,
gradient queries, optimizer updates or VM calls. Original checkpoints, data,
labels, splits, caches, failures and the app remain intact. The five milestones,
all seven covering families, independent final review and honest requests for
clearer or less-covered inputs remain required. Goal incomplete.
