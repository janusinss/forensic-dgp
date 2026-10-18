# Finite FP32 CodeFormer comparison V1 — prospective design

**16 October 2026: design only, not an execution packet.** No new training,
upload, VM connection or application change is authorized by a saved result
receipt. Actual training stays human-launched on L4 in tmux after source and
transfer verification. This experiment tests a demonstrated numerical mismatch;
it does not assume FP32 will fix all brightness or preservation failures.

## Question and controlled change

Does matching the candidate training path to deployed FP32 inference improve
the first-50-update restoration/preservation tradeoff relative to retained V2?
The diagnostic shows original FP32 training/inference and saved behavior images
agree, while BF16 changes selected codes before any update. This warrants a
short actual optimizer comparison instead of another zero-update diagnostic.

Start from the untouched official CodeFormer initializer SHA-256
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.
Retain the failed V1/V2 states and their historical gates. Do not resume them.
The only intended optimization treatment is **candidate FP32 without autocast**,
including a consistent teacher/behavior path and FP32 losses. Keep TF32 off,
seed 20261015 and the same software/runtime policy, model, Adam, batch order,
trainable groups, frozen decoder/codebook and loss definitions as V2.

Stop at **50 optimizer updates even when guards pass**. The original schedule
horizon stays **600** separately from the execution cap of 50. This keeps the
same first-50 warmup rates: encoder/token 2e−7 to 1e−5, fusion 4e−7 to 2e−5.
Changing a max-update constant alone is insufficient: V2 validates its 600
horizon, data design fingerprint, snapshots and terminal counters together.
A new worker must expose these two horizons explicitly and audit the exact
first-50 learning-rate sequence against retained V2.

## Inputs, targets and attribution

Use the same exact first 200 saved TRAIN observations, normal/dark pair order,
100 distinct TRAIN reference IDs, genuine clean targets and teacher assets
as V2. Verify regenerated or loaded candidate inputs against all 200 saved V2
PNG and float32 input hashes before their training forward. Distinct reference
IDs do not establish 100 different people; existing identity/overlap limits stay
in force. Do not accept only
an eight-case sample as proof of complete training exposure parity. Keep the
unchanged V2 data/model modules and their source-design bindings separate from
the new experiment design; do not rewrite their historical design provenance.

Retain all 391 already computed HQ-cache index/feature tensors with clean-input,
source and model-state bindings (about 103.4 MB; no extra teacher forwards).
Independently validate the new cache's frozen codebook lookup and compare all
50 saved FP32 behavior-anchor pairs against retained V2 targets. V2 did not
export its HQ cache, so exact old/new HQ-cache tensor equality is not established
by its return. Bind the same assets, code and runtime rather than inventing that
evidence. Include the newly retained cache in the final storage/export ledger.

Use the retained 53 photographic validation references × six profiles and 88
comparison cases: 406 outputs at update0 and again at update50. Preserve exact
input and target geometry, RAW, floor PNG and profile/source records. Compare
against untouched CodeFormer and retained V2 u50 on identical inputs. Native
CCTV remains unpaired; paired synthetic scores stay separate. These development
observations are repeatedly inspected and cannot become an independent final
test. Reserved FINAL identities remain untouched.

At update1 slot3, reuse the already budgeted candidate and untouched teacher
passes to test initial image-anchor L1 ≤1e−6 and absolute token KL ≤1e−6.
This is a prospective numerical path check, not a relaxed quality criterion.
Save actual dtypes, selected codes and parity evidence. Refuse mixed-precision
candidate execution. A path-check failure stops before the first Adam update.

## Finite calls and timing

| Successful endpoint counter | Bound |
|---|---:|
|Optimizer attempts/committed updates|50 / 50|
|Candidate training forwards / backward calls / input PNGs|200 / 200 / 200|
|Behavior-teacher forwards|50|
|HQ teacher cache forwards|391|
|Snapshot candidate forwards|812: two × 406|
|Benchmark adapter / initializer parity forwards|2 / 2|
|LPIPS training / prerequisite forwards|148 / 1|
|Prerequisite gradient queries|1|

Count reservations before calls and independently derive expected successful
counts from the declared first-50 exposure sequence. Successful counts are not
requirements to keep running after a quality/resource stop. Record exactly
which call/update was interrupted and preserve partial evidence. No new GAN,
EMA, gradient projection, fusion freezing, conditioning module, masking or
completion work is included.

| Resource | Prospective hard bound |
|---|---:|
|Initialization|600 seconds|
|First20-update optimization benchmark|600 seconds|
|Projected50 optimization / actual optimization|900 / 900 seconds|
|Each406-output snapshot|600 seconds|
|Whole worker / external termination grace|3,000 / 180 seconds|
|Export / termination grace|600 / 30 seconds|
|CUDA current and peak reserved memory|22 GiB|
|Free space after installation / retained reserve|16 / 4 GiB|
|Result payload / compressed archive|3.5 / 3.6 GiB|
|Temporary checkpoint|1 GiB|

These are design limits, not measured FP32 Adam memory or runtime. The prior
zero-update diagnostic peak cannot establish optimizer headroom. Refuse occupied
GPU/workloads; monitor memory during every first20 step and all later steps.
Use exclusive output/lock files and external process-group timeouts. Preserve
safe committed state on failure; no automatic retry, resume or appended epochs.
Runtime estimate and final caps must be rechecked against an exact storage and
call-accounting implementation before packet publication.

## Quality and review

Keep V2's exact RAW and delivered-PNG clear controls: mean MAE regression ≤0.01,
each-case regression ≤0.03. Keep its normal-blur luminance/coarse-RGB guards
and target construction unchanged. Any failed guard retains the stop and
outputs. The diagnostic is not grounds to waive the old stops. The old DGP
one-percent feature gate is not the CodeFormer adaptation's quality definition.

Moderate brightening is acceptable when it makes features easier to see. Report
dark-image visibility alongside paired fidelity, normal/clear appearance,
contrast, clipping, false texture and feature changes. Do not select a result
from brightness or a loss reduction alone. FP32 is a testable treatment, not a
promise of improvement. Opposing normal/dark fusion objectives remain a further
design issue if this comparison fails; blanket rollback would discard measured
dark-image benefits.

After return, independently audit archive, actual input/learning-rate parity,
all RAW/PNG outputs, checkpoint and Adam changes, preserved frozen tensors,
timing/calls/stops and every planned visual sheet. A 50-update completion or
passing development guards alone is not app adoption, useful native CCTV,
independent final evaluation or user acceptance. A successful result may justify
a separately designed longer finite run; it never auto-launches one.

## Evidence supporting this design

[Audited diagnostic](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_RETURN_REVIEW.md),
[independent pixel/training review](outputs/codeformer_path_diagnostic_v1_return_review/next_training_review_pixels_v1.md),
[independent state/implementation review](outputs/codeformer_path_diagnostic_v1_return_review/next_training_review_state_v1.md).

The prospective packet must preserve original sources, assets, splits and gate
records; stage verified dependencies from the existing V2 sibling without
reuploading redundant large weights. New commands will name a new destination,
validate hashes and refuse existing run evidence. Do not run the old diagnostic
or V2 guide again. Restoration remains first; the overall thesis goal remains
active and detector/completion model development stays deferred.
