# V42 return: structure and preservation failure

The independently audited V42 run stops at **50 of 3,905 permitted updates**.
Delivered structure improvement is **0.00692364% against the unchanged 1%
requirement**. Compound-degradation appearance similarity also regresses in
both photographic sources. Raw outputs fail the same requirements. The
stopped model is not qualified for development review or app promotion.

This was a successful export of a failed training run. The downloaded archive
is 1,156,863,656 bytes and matches SHA256
`308ad100ea6760bf0b6b42621a908d083f8691c3b8e53ab17b132dca0c88d9c2`.
The checksum, export receipt and 35,268 regular archive members agree. The
frozen protocol remains
`c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d`.
Returned code was not executed.

## What trained

Only our added spatial decoder trained: 57 tensors and 17,952 parameters. The
original DGP, stored normalization, fixed initial decoder and recognizer remain
unchanged. All 57 learned tensors change; relative aggregate weight change is
1.10770% by L2 norm. The stopped decoder exactly matches snapshot 50.

Fifty five-profile batches expose 250 cases from 50 photographic TRAIN references.
One full epoch requires 781 updates. The run therefore finishes **zero complete
epochs**, or approximately 6.402% of its first epoch. It neither completes the
five-epoch comparison nor adds epochs to the frozen original DGP. The original
app checkpoint is retained at SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

| Audited output stage | Structure improvement | Required | Preservation failures |
| --- | ---: | ---: | --- |
| Delivered PNG | 0.00692364% | 1% | Compound ArcFace group score in both sources |
| Raw float | 0.00709086% | 1% | Compound ArcFace group score in both sources |

Delivered compound score declines are 0.000102419 for `dataset/asian_faces` and
0.0000779603 for `dataset/thumbnails128x128`, above the unchanged 0.000001
allowance. The raw declines are 0.0000666213 and 0.0000711431. These are fixed
embedding-score comparisons, not proof of an exact identity change. Source
names are not ethnicity labels or native CCTV capture sources.

Both degraded source groups have positive but inadequate structure gains. The
delivered brightness-only share of pixel-MSE improvement is 0.422997%, below
the existing 20% limit. Passing that check or improving aggregate MSE/SSIM
does not override the structure and appearance failures.

## Independent audit and visual review

The unchanged prospective return auditor passes in 863.41 seconds. It rechecks
all 7,810 delivered PNG pixel-metric sets, saved embedding dot products, every
group decision and 100 stored raw previews across snapshots 0 and 50. All 100
CPU/model replays pass: maximum raw error is 2.32458115e-6, maximum PNG byte
difference is one, and maximum embedding error is 3.44589353e-7. Serialized
optimizer, scheduler, RNG and schedule-position keys are verified. The failed
gate still prohibits an automatic resume.

Other raw float arrays were hashed on the VM and were not retained. Their
pixel values are not independently recomputed. The audit's success verifies
the recorded failure; it does not establish full raw quality, native usefulness
or independent final evaluation.

All 50 preview cases are visually inspected at their original 256-pixel size on
ten sheets. No convincing additional eye, nose, mouth or outline definition is
established over the retained DGP. Existing blur in glasses, lips and fine
boundaries remains. These previews are photographic TRAIN cases with synthetic
degradation. No new native CCTV, development or reserved final pixels are opened.

Saved-preview mean absolute raw change is 0.000555067, with a maximum change of
0.010635. On average, 14.1115% of preview PNG channel entries change. The output
is therefore neither unchanged nor a demonstrated useful restoration improvement.

An additional frozen CPU inference comparison uses the same 50-case cohort for
both states. Its weighted correction loss falls from 2.25000003 to 2.24850720
(0.0663482%). For the 40 degraded cases it falls by 0.249583%; the ten clear
controls acquire a correction loss of 0.0206139 from an initial zero. This
comparison covers only the three reconstruction terms, not the combined
identity/hinge objective. It uses no gradients or optimizer. Losses recorded
on different reference batches are not treated as a learning curve.

## Timing and next action

The VM receipt reports 1,368.48 seconds for the worker and 63.20 seconds for
export. The two complete snapshots take 654.27 and 656.64 seconds; the 50
recorded optimization steps together take 7.75 seconds. The declared timing
and storage projections passed. The stop was scientific, not a storage stop.
These measurements do not predict that a longer failed recipe would qualify.

Retain V42, its complete training state, stop receipt, splits and original
checkpoint. Do not bypass the assertion, resume its failed trajectory or repeat
it unchanged. V40–V42 have now missed the structure requirement with this
added decoder. The assumption that connectivity and direct correction targets
would make this frozen-base recipe sufficiently useful has failed.

The diagnostic question required by AGENTS.md has been presented. The proposed
design review is in CCTV_DGP_POST_V42_ARCHITECTURE_REVIEW.md. No new learning
recipe is implemented or launched during this closure. The current app is
unchanged. The five-milestone restoration goal and all seven automatic/assisted
completion families remain active and incomplete.

Evidence: `outputs/cctv_dgp_residual_epochs_v42_independent_audit.json`,
`outputs/cctv_dgp_residual_epochs_v42_return_import.json`,
`outputs/cctv_dgp_residual_epochs_vm_v42_return/`, and
`outputs/cctv_dgp_v42_return_review_v1/`.
