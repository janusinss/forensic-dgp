# Filter-focused fusion adaptation from R2 — design draft, 18 October 2026

**DRAFT / NOT LAUNCH READY.** This is a proposed finite experiment, not a
training packet, authorized automatic launch, new trained model or qualification.
TRAIN admission, exact sampler, executable objective, operation/disk census,
gradient prerequisites and independent review must close before sealing it.
No neural, gradient, optimizer, VM or browser call was made for this design.

## Why this experiment differs

The completed `CCTV_CODEFORMER_FILTER_PATH_ISOLATION_V1_REVIEW.md` supplies a
bounded intervention: disabling the four fidelity branches suppresses selected
rows, grids and local haze but changes glasses, expression and accessories.
AdaIN-off alone leaves important defects. Keep production fidelity weight1 and
AdaIN on; train the demonstrated artifact-passing route rather than globally
bypassing it. Five purposive probes from three photographs do not prove a
population-wide cause or identify a historical loss as the unique cause.

R2 has useful user-preferred brightening and genuine50-update training, but
already fails original-based clear/normal preservation. Starting from it does
not make those failures acceptable. Its frozen behavior is a retention target
for dark observations only, not a replacement qualification baseline. Original
CodeFormer remains the clear benchmark; native CCTV remains unpaired.

This is not the old bounded-fusion feasibility V1 recipe. That was a zero-update
input-luminance-gated residual-blend measurement on an earlier FP32 state. This
proposal optimizes fusion weights from R2 with known paired filter targets,
without an inference brightness gate, residual-blend controller or learned mask.
It also differs from R2's350-tensor training: the upstream token/style path is
frozen and normal/filtered/dark observations receive distinct spatial targets.

## Initializer and exact trainability

Load only the515 model tensors from:

`outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth`

File SHA-256: `d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896`.
Its515-tensor canonical digest is
`a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f`.
Do not install old Adam, RNG, sampler or completed-update counters. Freeze a
separate R2 teacher; freeze untouched original CodeFormer for clear supervision.

Set `requires_grad=True` only for names beginning `fuse_convs_dict.`. The exact
saved schema has **72 tensors /10,659,584 scalar parameters**, not72 modules.
Each of sites32/64/128/256 has18 tensors: encode_enc ResBlock conv1/conv2/conv_out
weights/biases and norm1/norm2 affine weights/biases, plus scale/shift convolution
weights/biases at indices0 and2. The other443 tensors stay frozen:278
encoder/predictor and165 generator/codebook tensors. Preserve exact before/after
fingerprints for all443; optimizer membership must equal the exact72-name list.
FP32 fusion values occupy42,638,336 bytes; two Adam FP32 moment arrays occupy
85,276,672 bytes, excluding scalar state/serialization overhead.

Keep the existing512 internal geometry, prepared256 RGB input, w1, detach16,
AdaIN on, native clamp-before-bilinear256 delivery and floored PNG policy. No
alignment, post-brightening, sharpening, per-image setting choice or completion.
Use eval module modes with autograd enabled only for fusion. Eval does not
disable autograd; these fusion blocks use GroupNorm rather than running BatchNorm
statistics, and the frozen transformer has dropout0. No encoder-style monkeypatch
is needed: its features already have no gradient. The historical style-detach
witness requiring `style_requires_grad_before=True` would be invalid here.

**Frozen-generator gradient trap:** freezing decoder weights means
`requires_grad=False`; it does not authorize wrapping candidate generator blocks
in `no_grad` or `inference_mode`. After the first fusion site, activations require
gradients through later frozen decoder convolutions back to earlier fusion sites.
Keep ordinary autograd enabled across the candidate forward, including its frozen
decoder. Only frozen teacher forwards use no-grad. Do not detach decoder
activations to save memory. Require finite nonzero aggregate gradients at every
fusion site in the prospective witness.

## Admission before optimization

Root is preparing64 selected TRAIN photographs ×8 declared variants,512
observations. Twenty-two are in the prior R2's100-source training set;42 were
not in that set. This is not64 new-to-R2 or new-to-pretraining identities;
pretraining/person overlap remains unknown. This is a bounded admission set, not the entire
391-reference research pool or a new population sample. Audit all391 retained
source joins/terms/splits and bind a deterministic source selection before model
outputs. No easiest-looking model result may choose the photographs or severity.
Keep historical391 TRAIN/53 DEV, all72 filter observations, core63/flagged9,
24 refused controls, full406 DEV/native groups and reserved FINAL intact.

Review every proposed input with the frozen input-only sufficiency criteria.
An insufficient TRAIN variant remains a documented exclusion/refusal control;
do not optimize a fabricated face target for it or weaken the input threshold.
Do not silently replace a rejected source with a visually easier one. Training
eligibility is not established by membership in the generated512. Freeze the
eligible list before sampler/worker sealing. Report source groups separately.

Cover repeated rows, recoverable global blocks, local blocks and local blur.
Declare parameters before generation. Vary local footprint positions by
deterministic source/profile mapping while keeping the declared128×128 size;
do not learn only the earlier central128×128 rectangle. Footprint size diversity
is not supplied by this preparation and remains a limitation. The footprint is synthetic corruption support, not an
anatomical/removal/completion mask and not a network input. Keep strong variants
as refusal controls when visible structure is insufficient. Unknown-origin
wild1–wild5 have no clean references or audited TRAIN rights and remain unpaired
stress tests only. No DEV or FINAL input is admitted for optimization.

Reuse audited clean targets and source-specific lowlight/compound TRAIN joins for
dark retention. Pure synthetic-filter coverage is not evidence of real camera
deinterlacing, deliberate anonymization reversal, native identity or CCTV accuracy.
If an intended family has no eligible representation after review, the packet
must retain that evidence gap rather than claiming the family is covered.

## Proposed coherent loss roles

The following is one prospective objective to review, not a rate/weight search.
Do not inherit disconnected token objectives into the optimizing scalar. With
the encoder/predictor frozen, token CE/KL and latent losses cannot update fusion;
their values/readouts are diagnostics only. No HQ target network, LPIPS, GAN,
adversarial discriminator or EMA is needed for this first spatial-route pilot.
The existing shared prior remains frozen; a pixel/gradient loss can still cause
oversmoothing or artifacts and must be judged on saved images.

For optimization only, define `U(x)` as bilinear256 of the **unclipped** RGB
`(native512.float()+1)/2`, align_cornersFalse. Define original/R2 teacher targets
with the exact same U map under no-grad, detached as ordinary tensors. This
keeps initially identical R2 dark-anchor error zero while retaining gradients
for out-of-range predictions. It is distinct from the unchanged delivered
clamp-before-resize RAW/PNG path used for all acceptance metrics. Save both
domains and clipping fractions; never present unclipped loss values as delivered
quality. No per-image offset removal, centered output, gamma or normalization
that erases the desired brightening is applied.

Use four microbatches per new update, with exactly these roles:

| Slot | Target / prospective scalar |
| --- | --- |
| Filter A | Paired clean256; `Lspatial +0.25*Ladjacent` |
| Filter B | A different declared eligible corruption family/reference; same scalar |
| Clear/readable normal | Frozen untouched original teacher; `2*mean(abs(Ucandidate-Uoriginal))` |
| Dark | Frozen R2 teacher on the same audited dark input; `2*mean(abs(Ucandidate-UR2))` |

For a global corruption, Lspatial is whole256 RGB mean L1. For a local
corruption, Lspatial is0.5 times RGB mean L1 inside its exact footprint plus0.5
times RGB mean L1 outside, each divided by its own pixel/channel count. This
prevents the large clear context from hiding central defects. Ladjacent is the
mean of horizontal/vertical whole-image absolute differences of the RGB
prediction-minus-clean residual, one independent mean per direction. It
includes cross-footprint edges; it is not a learned confidence or identity score.
The full target, footprint and target hashes must be retained. No regional SSIM
or arbitrary anatomical subregion is used.

Sum each complete slot scalar divided by4, backward once per slot, step once
after all four. Clear supervision asks fusion to compensate R2's inherited clear
drift toward original behavior; dark supervision retains the user's preferred
R2 response. Filter supervision asks for known visible spatial structure. These
roles can still compete through shared fusion weights; balanced counts are not
a proof of compatible gradients. Original is an imperfect behavior teacher, not
clean truth, and R2 is not truth on native dark faces. Do not add both R2 and
original targets to the same clear slot or average away that conflict silently.

Use same-reference role pairings where available so input degradation, rather
than a hidden source distribution change, carries the role distinction. Role,
target and footprint labels are training metadata only; inference receives just
the image. A fitted network may still confuse lowlight with filters. All existing
dark/clear DEV controls and native visual review remain necessary.

## Finite optimization and prerequisites

Propose **maximum224 new Adam updates /896 microbatches**: at most448 filter,
224 clear,224 dark. The generated set has64 clear and448 nonclear observations
(seven nonclear profiles, grouped into corruption families); generation does not
establish that every nonclear observation is eligible. Seal the exact finite
update count as `ceil(admitted_nonclear_count/2)`, capped224, after input-only
review. Visit every admitted nonclear variant once before any duplicate; an odd
count permits one explicit, predeclared final padding duplicate. The two filter
slots need not be different families if that would prevent complete admitted
coverage; declare the balanced sequence and every repetition first. Clear/dark
anchors cycle deterministically across their audited joins. Do not call this
224 epochs, a complete pass over all391 sources or an authorization for later
continuation. Exact passes/counts are reported only after sampler sealing.

Fresh empty Adam: betas(0.9,0.99), eps1e-8, weight_decay0; one fusion group.
Prospective base LR1e-5, ten-update linear warmup `1e-5*u/10`, then constant1e-5
through the sealed final update, at most224. No rate grid, clipping, scaler, mixed precision, GAN, EMA, automatic
extension/resume or selected best-looking intermediate checkpoint. This rate is
a conservative design choice, not measured optimality or a claim that LR alone
fixes previous failures. FP32 throughout. Preserve actual optimizer attempts,
committed updates, partial microbatches and in-flight state.

Before constructing Adam, the same executable objective must demonstrate:

1. Same-context eval candidate/frozen-R2 initialization parity, exact delivered
   arrays and stable hard tokens; retain full differences before any decision.
2. Finite nonzero aggregate fusion gradient on representative admitted filter
   rows/local blocks/local blur, including every site's aggregate. Query exact72
   trainable tensors, not stale350-tensor helper partitions. Do not demand a
   nonzero dark-anchor gradient when identical candidate/teacher start at zero.
3. Clear-original gradient route is connected and finite where models differ;
   teachers have no grad, all443 frozen tensors have `grad=None`, and saved
   source readouts show no change in upstream logits/LQ/tokens for fixed inputs.
4. A whole-slot vs reconstructed component-sum gradient check on one filter
   witness, fresh Adam empty-state assertion, no cached gradient reuse, and
   exact restoration of modes/RNG/callables/grad fields before training.

Log per-slot scalars, unweighted terms, current LR, gradient/step norms,
nonfinite/zero counts and each site's changes. Report gradient norms as
source-bound readouts unless full arrays are independently replayed. Hold all
fixed teacher/model sources read-only. Freeze exact seed/sampler metadata in
the future protocol; no seed sweep is authorized by this draft.

## Evaluation and stop policy

Save initial, update50 and the sealed final update (at most224) model/Adam/RNG/
sampler/counter states and full delivered RAW/PNG. If the final count is<=50,
deduplicate that snapshot rather than add calls. Use all406 existing DEV cases at each planned
snapshot, plus the72 existing admitted filter observations with known targets.
Retain all24 refused controls without new model calls. Report core63/flagged9
and source groups; don't replace full406 by the eight-reference filter test.
No clean scores are calculated for the48 native CCTV cases. At initial and
update20, also retain24 predeclared TRAIN guard probes (eight clear, eight dark,
eight filtered),48 forwards total. These are repeatedly measured TRAIN probes,
not independent holdout or substitutes for full DEV gates. If final count is<20,
seal a corresponding smaller guard schedule before launch. Their resource/route/
new-regression checks can stop the run before50; no extra state is fabricated.

R2 snapshot0 already fails original-based gates: record these failures honestly.
It need not abort before the proposed intervention merely because those known
failures exist. Stop optimization on nonfinite values/state/gradients, broken
frozen-state identity, absent intended route, resource/census violation, or a
newly introduced/worsened per-case preservation limit against the recorded
R2 initial state. Preserve the actual failed/stopped state and images; do not
restart, disable the gate, append epochs or choose a prettier earlier export.
A draft interim comparator for an original-relative metric regression r is
`r_new <= max(existing_limit, r_initial)`: an inherited failed case may not worsen,
and a previously passing case may not cross the original limit. Apply the matching
mean limit0.01 and per-case limit0.03, not one blended summary. Its exact
floating-point comparison convention must be frozen prospectively in the
executable contract; no tolerance is selected from returned outcomes. This is an
additional safety stop, never a replacement qualification gate.

**Final qualification still requires every original rule.** On RAW and PNG
separately, clear61-case RGB-MAE and normal-blur53-case absolute luminance-bias/
coarse-RGB-MAE regression against untouched original retain mean<=0.01 and
worst<=0.03. R2 is not substituted as that baseline. Lowlight53 and compound53
retain the existing parent-relative PNG mean-MAE-no-increase / mean-SSIM-no-drop
rules; additionally record R2-relative dark retention to protect the accepted
direction. No inherited failure is converted to pass. The historical restoration
improvement gates remain distinct; reducing filter error alone is insufficient.

New filter family RGB/SSIM/adjacent/local-region measurements are descriptive
until a prospective usefulness criterion is sealed. Do not adopt a post-hoc1%
target from these five probes or select an optimal w/AdaIN per example. User
visible usefulness, unchanged preservation, all planned visual sheets, returned
independent audit, native source review and reserved FINAL are still required.
No app/default/Auto change follows automatically from a completed pilot.

## Proposed resources and exact ledger still needed

Prospective bounds: manual existing NVIDIA L4 only, FP32; worker4800s,
training1800s, each snapshot600s, initialization600s, preflight300s, witness180s,
export600s, external supervisor30s grace; RSS8GiB, CUDA reserved22GiB,
post-install free20GiB and disk reserve4GiB. Bounds must be enforced before
operations, not inferred from logs. Historical R2's612.67s total/185.17s optimizer
work motivates bounded feasibility; it does not measure this proposed workload.

Let D be newly evaluated filter DEV count and F=406+D. Here D72/F478 is the
proposed maximum. Three candidate snapshots need3F=1434 calls. Candidate TRAIN896,
frozen original-clear/R2-dark teacher targets448, frozen R2 baselineF478,
initial/update20 TRAIN guards48, and new original/parent filter baselines2D144
add up to3448 top-level forwards before
prerequisites. Reserve at most18 further forward calls only after their exact
purpose/order is enumerated, giving a provisional3466-forward ceiling. No HQ
or LPIPS calls are planned. Eight hundred ninety-six backwards and224 optimizer attempts
plus fresh finite gradient-query reservations must be enumerated. Bind every
operation source and do not treat unenumerated calls as free contingency.

CPU filter outputs are not claimed bit-exact to GPU. Current-runtime frozen
controls must supply all numerical gate comparisons or establish the retained
runtime/parity chain prospectively. A new VM/runtime must not silently reuse
CPU values as exact CUDA witnesses. No baseline is silently overwritten.

Exact output/checkpoint/archive/in-flight disk accounting remains open. Proposed
payload/archive ceiling6GiB/6.25GiB and regular-file ceiling12000 must close
against all RAW/PNG pairs, three planned physical checkpoints, the updated failure slot,
teachers' retained target arrays and metadata before sealing. With515FP32 model
values and only72 Adam states, use the measured saved schema plus serialization
upper bounds; do not reuse old832MiB-per-state guesses blindly. Keep at least
the actual initial and every completed planned snapshot; preserve honest partial
state on failure. No alias may hide an uncertain optimizer or partial write.
Lower caps if an exact source-derived ledger permits it; otherwise the packet
must request enough space before launch rather than deleting research evidence.

## Remaining prerequisite gaps

The512 input-only admission, source/terms/overlap audit, exact clear/dark joins,
sampler, loss implementation, gradient prerequisites, operation ledger,
checkpoint/disk accounting and independent source/data/transport reviews remain
unfinished. No exact runtime estimate, executable protocol hash, transfer archive,
tmux command or new checkpoint is supplied by this draft. Actual training stays
manual. Completion development stays deferred until restoration acceptance.

Read evidence: `CCTV_CODEFORMER_FILTER_PATH_ISOLATION_V1_REVIEW.md`,
`CCTV_CODEFORMER_FILTER_COMPARISON_V1_REVIEW.md`,
`CCTV_CODEFORMER_R2_IMPROVEMENT_REVIEW_V1.md`,
`research/codeformer_role_preserving_training_v1_r2/DESIGN.md`,
`research/codeformer_role_preserving_training_v1_r2/model.py`,
`research/codeformer_role_preserving_training_v1_r2/contract.py`,
`research/codeformer_adain_gradient_isolation_v1/gradient_route.py`,
`third_party/codeformer/codeformer_arch.py`, `third_party/codeformer/vqgan_arch.py`,
`outputs/codeformer_filter_comparison_v1_preparation/model_binding_v1.json`,
and `CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_DESIGN.md`.
