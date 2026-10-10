# Current DGP training review and improvement direction — 9 October 2026

Start improvement from the DGP checkpoint currently used by the main local face
workflow. Longer training is a hypothesis to test alongside the demonstrated
supervision limitations. This review establishes checkpoint lineage and records
the user's decisions; it does not prepare or authorize automatic training.

## What “20–31 epochs” means here

The current file is
`outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth`, SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
All 622 tensors exactly match the retained identity-v2 best checkpoint, which
is byte-identical to its epoch-2 checkpoint. That selected branch completed
113 updates per epoch, 226 updates and 1,804 training-case exposures in total.
The 452-update pilot total includes a separate rejected no-identity branch.

This is original Phase3 training **plus two selected fine-tuning epochs**.
It is not a model trained for only two epochs over its entire lifetime. The
current weights contain no epoch, optimizer or scheduler metadata. This review
does not establish the complete ancestral epoch count. File names alone cannot
establish that count or a resumable optimizer state.

The separate Phase4 return records epoch labels 27–31. Every one has a lower
fixed embedding-similarity score than its own starting baseline. Those rejected
weights are not the main app's current DGP. The separate V9 run completed twenty
epochs from V2 but also failed appearance safeguards; its selected best retained
the V2 tensors. Later failed decoder pilots and zero-update diagnostics do not
accumulate training epochs in the current app checkpoint.

## Evidence for a controlled longer study

These values use the same historical paired photographic development cohort:
440 degraded and 110 clear cases. They are exported PNG measurements, not native
CCTV metrics or proof of recovered identity.

| Identity-v2 stage | Degraded PSNR dB | Degraded SSIM | Fixed embedding similarity |
| --- | ---: | ---: | ---: |
| Original Phase3 starting state | 13.2317 | 0.52754 | 0.27663 |
| Fine-tune epoch 1 | 15.5650 | 0.64935 | 0.32391 |
| Fine-tune epoch 2, currently retained | 16.2167 | 0.65590 | 0.32686 |

Weighted mean training loss decreased from 0.259160 to 0.247125. However, blur
and motion pixel errors increased from epoch 1 to epoch 2 in both source groups,
while low-light and compound averages improved. These two checkpoints do not
prove undertraining or predict that more epochs will improve every facial feature.

V9 supplies a concrete counterexample: twenty pixel-supervised epochs improved
paired distortion scores but reduced embedding similarity in every degraded
source/profile group. V6 already tested genuine HQ targets with identity
supervision for two epochs and also failed preservation. Neither more epochs,
HQ targets nor an identity term alone is an established solution. The previous
finite applied-step failures remain preserved and are not relabelled as passing.

An epoch is a pass through the declared training schedule. Its value depends on
the examples, degradation coverage, optimizer and learning rate. Compare both
updates and exposure counts, with validation at checkpoints.
[PyTorch training tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html)

## Selected preparation direction

1. **Freeze the current starting state.** Use an isolated copy and establish
   exact initial inference parity with the existing app normalization and input
   geometry. Preserve the original Phase3 comparator, current app checkpoint,
   historical splits and all failed gates.
2. **Reuse audited data first.** There are 391 accepted HQ training references
   and 53 original validation-role references, plus 390 accepted lower-resolution
   Asian-source replay references. Keep the existing roles. Native CCTV
   development crops guide input-only and visual review; they cannot be treated
   as paired clean references. Full historical subject overlap remains
   unestablished. Do not open or optimize on reserved final identities.
3. **Resolve supervision before freezing executable code.** Review direct
   visible reconstruction in our DGP spatial/feature path against the failed
   V33 controls and V40/V41 combined objective. Retain identity supervision,
   clear-input anchors and all appearance acceptance requirements. Determine
   whether the current decoder can learn useful local detail before committing
   to a longer whole-model run. A copied failed recipe is not a new experiment.
4. **Compare training lengths within one declared recipe.** Proposed first
   continuation checkpoints are additional epochs 1, 2 and 5, with a five-epoch
   maximum. This is a design target, not a ready training command. Numerical
   learning rates, trainable tensors, loss weights, update/exposure counts,
   early structure stops, wall-time, VRAM, storage and export caps must be frozen
   before transfer. Extend to ten or more epochs only after independently
   audited preservation and useful native development gains justify a distinct
   finite continuation. Do not bypass an existing failure or weaken its gate.
5. **Qualify outputs before changing the app.** Compare resize, retained DGP
   and declared pretrained baselines on identical prepared 256×256 inputs.
   Review raw floats and delivered PNGs separately, all visible facial features,
   clear glasses and ordinary hair. Select useful preserved outputs rather than
   the last epoch. Insufficient structure still requires a clearer crop.

The architecture and objective for step 3 are unresolved. No new executable
protocol, training packet, hyperparameter claim or proven improvement is issued
by this document. The latest decision supports evidence-based changes starting
from the current DGP; it does not establish that an unchanged extension will work.

## Credits, migration and deadline

The user reports USD 34 remaining and will notify us near USD 5 to collect files
before changing VMs. They chose no additional monetary cap per experiment.
Finite epochs/updates and timing/stop rules still apply. This is a user-reported
balance, not a live billing query or a guarantee of affordable runtime. A running
idle VM continues to incur compute charges.
[Google Cloud billing model](https://cloud.google.com/products/compute/pricing)

The existing L4/g2-standard-4 remains the training target until the user identifies
a replacement. All training is manually launched inside tmux from verified
transfers and exact commands. No connection, VM maintenance or actual training
occurred during this review.

A future successful continuation must export model and optimizer/scheduler states,
random-generator state, next epoch/update and schedule position, architecture
code, environment versions, data/split/provenance hashes and failure records.
Verify the Windows copy before any migration cleanup. A weights-only export
supports a newly declared fine-tune; it cannot claim an exact training resume.
Failures retain their stop receipts and are not automatically resumed.

The thesis submission or defense is around or after 20 November 2026; no exact day
was supplied. Aim to reserve the period before submission for independent final
review, app regressions, bundled inline Playwright verification and evidence-based
thesis updates. A longer training run does not replace these milestones.

All seven covering families remain separately required: masks, sunglasses,
strong lens glare, hands, obstructing hair, scarves and objects. Show the
automatic removal area for optional correction before generation; preserve
visible appearance, clear glasses and non-obstructing hair. Return one plausible
estimate with original/mask/result and PNG/bundle downloads; request a less-covered
crop when needed. Report automatic and assisted results separately. Neither
restoration training nor a completion estimate establishes exact hidden identity,
ethnicity or Zamboanga performance. The complete goal remains active/incomplete.

## Review artifacts

[Reproducible metadata/weight review script](scripts/review_cctv_dgp_current_training_history_v1.py)
reads existing files with weights-only CPU loading. It performs zero model
forwards, gradient queries, backward calls and optimizer updates.
[Machine-readable review](outputs/cctv_dgp_current_training_review_v1/training_history.json)
binds the source files and records the user's decisions.

The historical results and app code remain unchanged:
[V2 results](CCTV_DGP_PILOT_RESULTS.md),
[V6 HQ comparison](CCTV_DGP_TARGETS_RESULTS_V6.md),
[V9 twenty-epoch result](CCTV_DGP_MIXED_V9_RESULTS.md),
[current app integration](CCTV_DGP_APP_V3_INTEGRATION.md),
[preceding architecture review](CCTV_DGP_POST_ACTUAL_STEP_ARCHITECTURE_REVIEW.md).
