# Broader TRAIN filter preparation — 18 October 2026

**512 fixed paired inputs are generated, independently pixel-verified and
personally reviewed. No new model has been trained.** This advances the
fusion-focused R2 adaptation justified by the completed
[path-isolation test](CCTV_CODEFORMER_FILTER_PATH_ISOLATION_V1_REVIEW.md).
The user's desired useful brightening remains part of the restoration goal.

## Sources and fixed coverage

A fresh data-only source audit checked all **391 historical TRAIN and 53 DEV
photographs**, their 888 current source/target files, direct1024 LANCZOS256
targets and direct512 pixel fingerprints. Source-file, decoded-source-pixel
and recorded photo-URL intersections between TRAIN and DEV were zero. These
checks do not prove different people or absence of CodeFormer pretraining
overlap. The source refresh used separate hash/decode reads and leaf file guards;
its [limits](outputs/codeformer_filter_train_v2_preparation/source_audit_limitations_v1.json)
are retained. The producer and independent verifier subsequently decoded the
selected sources from hash-verified byte buffers and rechecked them at closure.

Before generating or viewing the new corruptions, root independently verified a
deterministic 64-TRAIN selection over nine recorded license/official-category
strata. Each stratum received one place; the remaining places used proportional
largest remainders and fixed hash ordering. FFHQ's official source category
does not override the thesis's historical TRAIN/DEV role. The selection includes
22 photographs from R2's previous 100-source set and 42 outside that set; these
are not 42 proven unseen people. All 391/53 historical memberships remain intact.

Exact per-photo authors, URLs, licenses and change/source notices accompany the
data. Attribution, noncommercial, dataset and upstream CodeFormer conditions
remain recorded; this is not blanket public-distribution or portrait-use
clearance. No new public image acquisition is claimed. No user wild1–wild5,
native CCTV, historical DEV or reserved FINAL photograph entered this TRAIN set.

Eight variants per photograph cover clear inputs, Gaussian2.4, two anisotropic
orientations, four-row repetition/blending, global8-cell mosaic, localGaussian3
and local8-cell mosaic. Local128×128 rectangles span **47 deterministic
positions** across the 64 photographs. Their size is fixed; no detector or
anatomical/removal mask is used. Parameters, arithmetic and review criteria were
fixed before generation. These are declared synthetic proxies, not verified
reproductions of unknown filters or native camera processes.

## Actual generation and pixel verification

One root-observed data-only run exited0 in **46.53 seconds**, producing
**1,256 files /105,926,107 bytes**. It made zero neural, gradient or optimizer
calls. Writes were exclusive under a new namespace, bounded by 1,300 files,
512MiB and a600-second checkpoint-based elapsed-time guard. This guard is not
an external interrupting watchdog. Partial-failure preservation remains in the
producer; no actual generation failure occurred.

The independent verifier imported neither the producer nor SciPy nor a neural
runtime. It rebuilt all512 corruptions with separately implemented filter math:
**zero saved-pixel channel mismatches**. It also verified all128 direct clean
reference images, source/target/attribution joins, footprint pixels, exact
outside-footprint preservation, all512 review-sheet cells and closed file/directory
namespace hashes. Eighteen producer fixtures and fifteen independent verifier
fixtures passed. These are data-integrity checks, not restoration performance.

Data manifest SHA256:
`16c804f3f6492332c02ed7c33ab1e2448d6ae84e4e80b0eede16cdd25068e683`.
Independent saved-pixel receipt SHA256:
`cc606ff9660c882bd06ec9d380c0df9a5410d7a83f93c9e7efbd49750bce0e3e`.

## Input-only review and conservative holds

Root personally viewed all **32 sheets /512 exact256 input panels**, with no
clean-target or restored-output panels. All28 nonclear pages preceded the four
clear-control pages; per-page decisions were recorded sequentially. Prior
historical/source and other-corruption exposure prevents a blinded or independent
FINAL claim. The metadata audit checks the recorded sequence and joins; it does
not independently establish the visual judgments.

The retained root labels are **390 readable,93 borderline,1 insufficient and28
out of scope**. Five clear controls contain recognizable additional partial or
full background faces: `tr_ffhq_04454`, `tr_ffhq_20546`, `tr_ffhq_45314`,
`tr_ffhq_49284` and `tr_ffhq_61989`. All40 variants of these sources are excluded
from prospective training without rewriting earlier input labels or replacing
the sources with easier photographs. Some secondary faces were less apparent
in the stronger corruptions; source scope is therefore checked separately from
per-input readability.

A second assistant observer reviewed all four global-mosaic sheets before
reading root's per-case labels. Aggregate root counts had been disclosed, so
this was not fully blinded. The observers agreed on46/64 labels and differed
on18, including six root-readable/observer-borderline cases and two pose-scope
concerns. Thirteen were readable for both; one of those belongs to an already
excluded multi-face source. These disagreements are preserved, not converted
into a calibrated automatic threshold or independent human FINAL evidence.

The separate [hold ledger](outputs/codeformer_filter_train_v2_preparation/candidate_holds_v1.json)
retains all original decisions. It requires both observers readable for the
global-mosaic candidates and holds both pose-disputed sources,
`tr_ffhq_28856` and `tr_ffhq_62653`, across all variants. This removes16 former
root candidates from the prospective list. Nothing is deleted or relabeled.

The resulting **370 candidates from57 photographs** comprise57 clear controls
and313 nonclear inputs. All nine recorded source/terms strata remain represented.
The nonclear candidate census is:

| Corruption group | Inputs |
| --- | ---: |
| Gaussian and two anisotropic orientations | 167 |
| Repeated rows | 57 |
| Global mosaic | 12 |
| Local mosaic | 25 |
| Local Gaussian | 52 |

**Every optimization flag remains false.** These are candidates for an explicit
finite packet, not automatic training admission. Readable repeated-row cases
still require the new protocol's explicit admission; no old strong-control
exclusion was silently changed. Borderline, insufficient, disputed and multi-face
examples remain in the complete evidence. The earlier all72/core63/flagged9
comparison and24 refused controls stay intact. No model output selected these
photographs or their parameters.

One observation was corrected explicitly: coarse blocks initially suggested
glasses in `tr_ffhq_32052`; later observed-input variants show uncovered eyes
with dark surrounding appearance and no glasses frame. The earlier uncertain
usability label and its correction remain recorded. No covering-detector label
or hidden anatomy is inferred from that impression.

## Next finite model experiment

The [design draft](research/codeformer_filter_fusion_training_v1/DESIGN_DRAFT.md)
targets only R2's72 fusion tensors /10,659,584 parameters, initially freezing
the other443 tensors. It keeps fidelity1, AdaIN on and the existing delivered256
RAW/PNG path. A frozen decoder must still carry gradients through its operations
to the fusion blocks; its candidate forward cannot be wrapped in no-grad.
Teachers alone use no-grad. Disconnected token/latent objectives are not
advertised as fusion-training losses.

The current candidate count supports **at most157 new Adam updates /628
microbatches**, with314 filter slots (313 candidates plus one declared padding
duplicate),157 clear and157 dark slots. This is not157 epochs or a complete pass
over all391 photographs. The earlier164-update draft is preserved separately
before the second-observer holds. Exact sampler/dark joins, executable losses,
gradient prerequisites, resource/call/disk census and transfer/return files must
still close before a manual L4/tmux packet can be called launch-ready.

Paired spatial supervision should address filter artifacts; untouched-original
clear behavior and frozen-R2 dark behavior serve different roles. R2's existing
preservation failures remain failures. Final original-based RAW/PNG preservation
limits, dark checks, the complete historical406 DEV/source-separated native
review and the separate72 mixed TRAIN/DEV filter comparisons remain required.
Native CCTV has no aligned clean reference. No independent FINAL identity,
Zamboanga performance or exact restoration claim follows from this preparation.

No VM training, checkpoint export, app/default change or new completion work
occurred. R2 remains the existing research candidate. Actual training stays
under the user's verified manual finite L4/tmux workflow. Acceptance still needs
useful visible outputs, preservation, user review, independent FINAL and the
accepted app/Auto/override/download flow. Covering/completion improvement remains
deferred until restoration acceptance. The overall goal stays active.

Historical Drive offload is already complete and was not repeated: the separate
15October batch removed12,394,485,961 verified cold-copy bytes. **Zero bytes were
deleted or reclaimed by this preparation.** Original models, caches, splits,
provenance, failures and migration backups remain protected.

## Evidence

- [Source refresh and frozen64 selection](outputs/codeformer_filter_train_v2_preparation/cohort_selection_v1.json), [root contract approval](outputs/codeformer_filter_train_v2_preparation/root_contract_approval_v1.json).
- [Generated512 data](outputs/codeformer_filter_train_v2_data/manifest.json), [independent exact-pixel audit](outputs/codeformer_filter_train_v2_independent_audit/saved_pixel_audit_v1.json).
- [Root512 input-only review](outputs/codeformer_filter_train_v2_preparation/root_input_only_review_v1.json), [independent metadata audit](outputs/codeformer_filter_train_v2_preparation/admission_metadata_audit_v1.json).
- [Second-observer global-mosaic review](outputs/codeformer_filter_train_v2_preparation/second_observer_global_mosaic_review_v1.json), [conservative candidate holds](outputs/codeformer_filter_train_v2_preparation/candidate_holds_v1.json).
- [Draft finite training design](outputs/codeformer_filter_fusion_training_v1_preparation/proposed_training_design_v1.json), [storage offload record](GOOGLE_DRIVE_OFFLOAD_COMPLETE_20261015.md).
