# CodeFormer V2 return audited: quality-stopped at 50 updates — 15 October 2026

**CURRENT: 50/600 updates; real weight changes; declared quality checks failed.**
The downloaded archive, training state and every returned RAW/PNG output were
independently checked. All 52 sheets / 406 cases and the three failed clear
controls at full 256 pixels were reviewed. V2 has useful-looking improvements
and higher paired SSIM on 50/53 HQ blur cases, but is not an accepted app model.

[V2 results, causes and limitations](CCTV_CODEFORMER_ADAPTATION_V2_RETURN_REVIEW.md).
[All comparison pages](outputs/codeformer_adaptation_v2_return_review/visual_sheets/INDEX.md).
[Latest brightness preference](CCTV_CODEFORMER_ADAPTATION_V2_USER_REVIEW_20261015.md).

The user accepts moderate brightening when it makes a dark face easier to see.
Evaluate that benefit separately from washed-out detail and altered visible
features; brightness change alone is not a verdict on usefulness. All three
failed clear faces also lose SSIM. Preserve
the historical V2 clear and normal-blur photometric failures unchanged. Any
future criteria must be declared prospectively, not fitted to those failures.

All 350 trainable tensors changed; 165 frozen tensors stayed exact. V2 reached
200 TRAIN exposures, not 600 updates or epochs. V1 u200 versus V2 u50 brightness
differences are descriptive, with unequal budgets and jointly changed recipes.
Both stopped states, original checkpoints, splits and gate failures remain
protected. Do not relaunch V1/V2 unchanged or append epochs after these stops.

Next is a bounded zero-optimizer VM diagnostic of training/evaluation precision,
photometry and loss paths before choosing further training. Its design review
is saved under outputs/codeformer_adaptation_v2_return_review/next_diagnostic_review_v1.md;
it is not an execution-ready packet. Actual neural/gradient diagnostic work and
all training remain human-launched on L4 in tmux. No new VM run was launched.

Restoration comes first. Detector/completion development remains deferred until
user-accepted restoration. Native CCTV remains unpaired and unqualified; final
identities, app adoption/flow and independent final review remain outstanding.
The full thesis goal stays active. Earlier PREPARED/UNRUN commands below are
retained historical evidence, superseded by this current return status.

---

# CodeFormer V2 preparation verified — 15 October 2026

**PREPARED / UNRUN.** The next manual restoration fine-tune targets excessive
brightening while preserving useful facial-detail gains. No V2 training, neural
forward, gradient or optimizer update was run during local preparation.

The user's personal review favored most V1 outputs over untouched CodeFormer,
while explicitly identifying increased brightness. This is positive feedback,
not acceptance of its failed preservation case or every source/profile.
[Exact user feedback](CCTV_CODEFORMER_ADAPTATION_V1_USER_REVIEW_20261015.md).
[Audited V1 results and all comparisons](CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md).

## What changes

The separate candidate starts from untouched CodeFormer. Matched normal/dark
TRAIN pairs balance exposure at each selected severity. A coarse RGB loss applies
even at 24/28-pixel observations, while fine losses retain their 32-pixel eligibility.
A separate frozen original CodeFormer teacher anchors normal clear/light TRAIN
behavior. This is a joint data/loss/schedule experiment; it cannot identify the
causal contribution of each change. Architecture and inference policy stay fixed:
prepared256 → internal512, fidelity1, AdaINTrue → delivered256.

Adam remains explicit: encoder/token LR1e-5, fusion2e-5, 50-update warmup and
cosine to2e-6 at update600. There are at most 600 updates / 2400 image exposures,
with snapshots0/50/200/600. Both old clear RAW/PNG guards remain, with additional
prospective normal-blur brightness/coarse-error regression guards. They are abort
checks, not proof of useful structure or a new interpretation of historical gates.

The same391 audited HQ TRAIN references and53 separate validation sources are
retained. All318 validation input PNGs and pair records are exact V1 VM bytes;
the88 fixed comparisons include48 native/unpaired and40 synthetic/paired views.
Reserved final identities remain excluded. Every generated TRAIN input and
behavior-teacher anchor is retained for return auditing. Source terms, FFHQ
pretraining/subject-overlap uncertainty and repeated development exposure remain
explicit. Native CCTV receives no aligned-clean PSNR/SSIM or Zamboanga claim.

## Verified transfer and limits

- **59 combined local tests passed**, plus eight independent CPU formula fixtures.
- **1,410 archive members / 1,333,451,654 payload bytes** were checked against
  retained originals and exact namespaces. The archive was independently streamed.
- Twelve original comparison checkpoints and V1's update0/update200/stopped states
  and failure were independently checked unchanged.
- Require20GiB after install, retain4GiB reserve and22GiB reserved-VRAM ceiling.
  Initialization600s, first20 benchmark600s, optimization3600s, each snapshot600s,
  worker7200s and export600s; external termination grace is separately declared.
- Conservative output planning includes all training images, teacher anchors,
  RAW/PNG snapshots, complete-or-incomplete checkpoint evidence, metadata and
  archive overhead. Runtime measured byte caps remain authoritative.

Archive: `outputs/cctv-codeformer-adaptation-v2-execution.tar.gz`.
Compressed bytes: **1,282,458,459**.
SHA256: `61acb9cd98b74ad1412d1ecbe4e916415e99aa80377d0c94332409fabf2c682b`.
Protocol SHA256: `029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183`.

[Five upload/install/tmux/run/download steps](CCTV_CODEFORMER_ADAPTATION_V2_VM.md).
[Frozen prospective design](CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md) retains its
historical design-only wording and exact approved bytes; this verified preparation
supersedes that packaging status without editing the frozen design or packet.

## Evidence and acceptance

[Independent worker/model review](outputs/codeformer_adaptation_v2_design_review/worker_review_v1/INDEPENDENT_WORKER_MODEL_REVIEW.md),
[independent packet readback](outputs/codeformer_training_readiness_v2/root_review_v1/independent_packet_review_v1.json),
[combined source tests](outputs/codeformer_training_readiness_v2/root_review_v1/source_readiness.json),
and [independent archive audit](outputs/codeformer_training_readiness_v2/root_review_v1/independent_archive_review.json)
establish preparation integrity. L4 execution, resulting image quality and the
new recipe's usefulness are unmeasured. Do not auto-launch, repeat V1 or change
the app default. Actual training remains the user's manual tmux action.

After download, independently audit archive bytes, weights/Adam/RNG, exact inputs,
teacher anchors, actual exposures, RAW/PNG conversions and profile metrics, then
review all intended images against untouched CodeFormer and retained V1.
User acceptance and independent reserved-final review remain required before
integration and bundled inline Playwright app verification. Restoration remains
first; detector/completion improvement stays deferred. The full goal is active.
