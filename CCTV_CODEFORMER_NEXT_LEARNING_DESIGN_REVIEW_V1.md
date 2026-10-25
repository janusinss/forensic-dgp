# Next learning experiment: coverage and reconstruction-feature adaptation

## Material Passport

- Origin Skill: Academic Research Suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-24
- Verification Status: UNVERIFIED — a recommended design, not implemented or executed
- Version Label: next_learning_design_review_v1

## Recommendation and research question

Prepare one **matched two-arm learning experiment**, after new TRAIN inputs and
the common supervision are admitted. Both arms start from the complete original
trained u64, keep its current inference architecture, and use the same data,
loss, optimizer settings and exposure order. Arm A trains the existing
post-fusion heads. Arm B additionally trains the existing U96 feature adapters.
This tests whether learning earlier feature correction adds value beyond
repairing data coverage and supervision with the current head path.

The recommendation answers the user's standing instruction to choose the best
approach. It is not a claim that the user selected either arm or approved this
specific protocol. The prior diagnostic question remains unanswered. Under the
three-failure rule, no model/training code is changed, no new factory is
implemented, and no packet or launch command is ready. The proposed experiment
is concrete enough for design discussion before that work.

Hypothesis: a matched broader task will expose clear/tone and structure
tradeoffs; allowing the existing pre-SFT adapters to learn alongside the heads
can outperform head-only learning without sacrificing inherited preservation.
The result can contradict this hypothesis. It cannot establish that more
parameters, one particular layer, or this degradation model uniquely caused
the historical failures.

## What the evidence rules out

The latest tiny step missed one localized PNG retention endpoint by only
1.25644616249e-7. That is distinct from the much larger inherited DEV problems:
observed u64 worst excesses against retained original CodeFormer reach
0.060644016108 for clear PNG MAE and 0.115910026084 for normal-blur PNG luminance
error, against the unchanged 0.03 worst limit. Preserving u64 alone can preserve
those faults. The 571-output comparison is incomplete; these are observed
counterexamples, not a full 716-output mean-gate verdict. Unknown slot 572 and
the expired deadline remain binding; this design does not resume that run.

The 24-photograph training-input review found no target/transform corruption.
Its 16 optimization photographs and eight adaptive monitors nevertheless cover
one sigma-1.2 blur and exposure-0.35 darkening. The weighted RAW objective and
separate RAW/PNG/legacy gates measure different things. Their verified scalar
agreement does not demonstrate adequate training coverage or compatible goals.

Earlier HQ-code substitution gave only modest, visually similar gains on strong
blur. V1/V2/FP32 experiments already trained code prediction with CE/latent
supervision. Repeating generic code-predictor training is not the next design.
The clean-SFT-input oracle showed substantial conditional visible headroom,
but used unavailable clean features. U96 already learned feature adapters;
neither repeating that oracle nor restarting the old adapter recipe is new.
The proposed B arm instead updates both existing adapter blocks from the
complete trained-u64 function under the same new learning task as A.

Sources: [broader counterexamples](outputs/codeformer_u96_postfusion_lr_broader_v3_interruption_audit_v1/PARTIAL_COUNTEREXAMPLES_V1.md),
[three-route diagnosis](CCTV_CODEFORMER_U64_POST_CHECKED_STEP_DIAGNOSTIC_MATRIX_V1.md),
[current data/loss review](CCTV_CODEFORMER_U64_EXISTING_TRAIN_EVIDENCE_REVIEW_V1.md),
[HQ-code history](CCTV_CODEFORMER_HQ_CODE_ORACLE_V1_REVIEW.md), and
[clean-feature intervention](CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md).

## Controlled comparison

| Item | Arm A: task/supervision control | Arm B: expanded existing trainable blocks |
| --- | --- | --- |
| Initial function and complete state | Exact original trained u64: 575 tensors | Exactly the same 575 tensors |
| Trainable block | 36 post-fusion head tensors / 168,960 parameters | Same heads plus 24 U96 adapter tensors / 220,288 parameters |
| Frozen block | 515 R2 plus 24 U96 tensors | All 515 R2 tensors |
| Intended contrast | Does the common new task suffice in the current path? | Does permitting upstream feature correction improve that result? |
| Unchanged deployment inputs | Observed crop only; no target, role label or oracle features | Identical |

Use canonical complete initializer
`47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a`.
Keep original R2, U96, original u64 and every failed successor as separate
immutable artifacts. The B candidate may update its private U96-adapter copy;
the retained U96 checkpoint is never overwritten.

The architecture already contains both blocks. The current source deliberately
asserts that the 24 adapters are frozen and that their used features carry no
gradient. Merely toggling `requires_grad` in place is invalid. Any later B
implementation needs a separately reviewed factory with explicit new ownership,
trainable/frozen census, graph connectivity and zero-update RAW/PNG parity.
Historical factories, hooks and source hashes remain untouched.

Both arms retain the frozen codebook, code predictor, generator, fidelity and
AdaIN policy. Their selected codes cannot improve through training these
blocks. Incorrect code predictions remain a limitation, not something this
experiment claims to repair. B changes both parameter count and optimization
conditioning; a favorable A/B result supports the block package under this
task, not a pure capacity or layer-causality claim.

## Proposed common training task

Use 24 optimization photographs: the 16 retained optimization photographs plus
eight from the existing unused TRAIN reserve. Exclude the deferred photographs,
preflight-only witnesses, old adaptive monitor photographs, legacy guards,
DEV/native/wild and FINAL. Freeze the eight additions by a deterministic rank
before any new output is seen. Stored clean-target admission and terms must
rehash; photograph separation does not prove person/pretraining independence.
No new acquisition is required to decide whether these existing sources suffice.

Each optimization photograph receives five proposed singleton role exposures:

1. Exact clear photographic target.
2. Recoverable whole-image Gaussian blur.
3. Localized Gaussian blur with an explicitly recorded region and transition.
4. Recoverable blur plus downsampling and JPEG compression.
5. Low light with a small, explicitly simulated sensor-noise component.

This is a **new synthetic-data proposal**, not an admitted dataset or a camera
calibration. Operator parameters, seeds, boundaries and quantizers must be
specified and sealed during data preparation. Preserve all original 96 files;
new variants need new paths. Input-only review must classify the proposed
observations before model outputs, distinguishing readable learning cases from
uncertain or insufficient stress controls. An insufficient case is retained
with no invented target-readability verdict or silent easier replacement.
If 24 photographs/five roles cannot be admitted, the design is not launch-ready;
do not silently shrink the study or change operators after output review.

Both arms receive one identical photographic objective. A reasonable candidate
for discussion is whole RGB L1 plus separately logged adjacent-residual and
coarse RGB/luminance error, with all five role contributions visible. Clear
photographs remain genuine targets. Low-light supervision must retain useful
brightening without using a generated bright face as photographic truth.
The added coarse term targets the demonstrated tone problem; it is not proof
of a solution. Exact coefficients, reductions, regional rules, gradient
accumulation and whether an additional term is redundant must be fixed from
source/array analysis **before** creating a training worker. No weights are
selected by fitting these DEV counterexamples. Equal role sampling does not
make gradient norms equal.

The old eight adaptive TRAIN monitor photographs and exact legacy quality bank
remain unchanged. Additional corruption checks can be separately declared;
they cannot replace the original gate denominators or become new independent
holdouts. Train on original observations, not favorable surroundings-prepared
outputs. Preparation remains a separate original/prepared comparison.

## Finite execution proposal and admission requirements

The proposed learning ceiling is **48 updates per arm**, two fixed passes over
24 photographs; each update accumulates the same five singleton exposures.
That is at most 240 training exposures per arm / 480 across both. This is a
new finite pilot, not extending a failed state or claiming full-corpus epochs.
The exact optimizer/LR is not copied from the paper or increased because it
sounds faster: both arms need the same predeclared Adam policy, justified by
the admitted loss scales and state-preservation bounds before sealing. Record
per-block actual displacement and role contribution evidence.

Proposed quality checkpoints are zero, 8, 24 and 48 updates. Existing RAW/PNG,
structure, clear/normal-tone, dark and individual preservation requirements
remain exact. A failed required checkpoint stops that arm; it is not rescued
by an earlier state, later epoch, smaller post-hoc step or the other arm.
The initial inherited DEV failure is disclosed as a problem to repair, not
relabeled a passing reference. Unchanged original-u64 small-bank retention
does not substitute for the broader original-CodeFormer gates.

Before manual execution, enumerate every initialization/parity, training,
monitor and terminal inference call, backward, optimizer commit and saved state
in an exact finite schedule. The 480 training exposures do not include those
evaluation calls. Set enforced per-phase, per-arm, whole-worker and external
timeouts, memory/disk/output/export ceilings, and stop on a timing projection
that cannot finish inside the declared limits. Exact resource numbers and
runtime estimates are **not yet measured or sealed**; no command here is ready
to run. Preserve optimizer moments, RNG/sampler position, checkpoints, journals,
resource samples and unknown/partial operations. There is no automatic retry.

All actual model training, new gradient measurements and candidate parameter
applications occur only in the user's manual NVIDIA L4 tmux workflow. A later
ready packet must supply exact gcloud uploads, checksum/install commands and
single-source downloads. No VM connection or actual training was used to write
this proposal.

## Decision rule and stopping the diagnostic loop

The immediate result is a controlled learning comparison, not automatic
qualification. Independently audit saved states, RAW arrays, delivered PNGs and
all planned visual sheets. A must show whether the new common task works with
the current path. B earns preference only if it adds useful visible structure
and preserves appearance at the same finite exposure budget. Lower training
loss, sharper invented detail, brightness or an accepted archive is insufficient.

Any survivor still faces the unchanged broader DEV gates, source-separated
native review, user acceptance, independent FINAL and full app verification.
Keep the unknown broader slot and historical teacher-RAW absence disclosed.
Do not reopen missing old slots or pretend partial cohort means are final.
If neither arm qualifies its finite screen or shows useful visible gain,
retain both failures and discuss a different mechanism. Do not launch another
near-identical weighting, step-size or extra-epoch sequence automatically.

## Primary-source grounding and limits

The [CodeFormer paper](https://papers.neurips.cc/paper_files/paper/2022/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf)
separates discrete-prior/code prediction from controllable reconstruction and
describes an ill-posed restoration problem. It supports distinguishing those
paths; it does not validate our adapters or this proposed pilot.

The authors' [training guide](https://github.com/sczhou/CodeFormer/blob/master/docs/train.md)
separates VQ prior, code-sequence learning and controllable-module training.
Its [stage-II configuration](https://github.com/sczhou/CodeFormer/blob/master/options/CodeFormer_stage2.yml)
and [stage-III configuration](https://github.com/sczhou/CodeFormer/blob/master/options/CodeFormer_stage3.yml)
use broader synthetic corruptions and code/feature supervision than this
project's recent narrow diagnostic cohort. Their iteration counts, LR and
degradation ranges are reference recipes, not appropriate defaults or promises
for our L4 adaptation. The [joint trainer](https://github.com/sczhou/CodeFormer/blob/master/basicsr/models/codeformer_joint_model.py)
also distinguishes large-degradation code learning from image-level losses.
These primary sources were consulted 24 October 2026; online master branches
can change. Any adopted code must later be revision-pinned and reviewed.

This review used source text, existing reports and primary literature only.
No neural inference, autograd, optimizer, parameter application, dataset
mutation, browser/app change or VM operation occurred. The full goal remains
active and incomplete, including optional protected-surroundings preparation
and the later deferred covering-family scope. No exact hidden identity,
ethnicity or Zamboanga-performance claim is made.
