# CodeFormer loss response V1 returned: measurement succeeded, no model promotion

**17 October 2026. The finite diagnostic completed and its returned evidence
passed independent checks. It produced no new model checkpoint.** The previous
AdaIN50 preservation failure remains binding; this TRAIN-only measurement does
not repeat or pass the61-clear/53-blur DEV guards.

[All16 exact256 image sheets](outputs/codeformer_loss_response_v1_return_review/pixels_v1/visual_sheets/INDEX.md).
[Root personal visual review](outputs/codeformer_loss_response_v1_return_review/root_visual_review_v1.json).
[Independent saved-pixel measurements](outputs/codeformer_loss_response_v1_return_review/pixels_v1/SCIENTIFIC_READBACK.md).
[State and gradient review](outputs/codeformer_loss_response_v1_return_review/state_v1/STATE_AND_GRADIENT_REVIEW.md).
[Independent next-design review](outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.md).

## 1. What ran and what is verified

| Evidence | Result |
|---|---|
| Returned archive |2,567,636,721 bytes; SHA-256 `b0e2b6d0aa245fcebe334c21c1b50cfbfde3ca65d494bb1d1fd8a9d17730de86` |
| Protocol | `23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab` |
| Imported payload |4,609 regular files /2,881,853,749 bytes; exact manifest, hashes, portable namespace and single gzip CRC checked |
| Actual VM work |425 neural forwards,193 gradient queries,18 real disposable Adam steps; zero committed training updates/backward calls |
| Design |Two saved states ×16 fixed TRAIN observations ×10 conditions; eight photographic references,12 image-eligible observations and four normal behavior anchors |
| Worker time |585.444 seconds: initialization41.517, source159.511, probe384.416; supervised worker598.694 seconds |
| Other stages |Preflight16.966 seconds; internal export203.374 seconds; overlapping worker/supervisor times must not be added |

The return contains32 source outputs,320 probe outputs,2,016 non-None component
gradient arrays,20 branch summaries and20 restoration receipts. The636 ordered
reservations agree with the425/193/18 census. Original0 counter0→1 and stopped50
counter50→51 are disposable transitions; every branch restores the same saved
state and none becomes an appended training update. No failure, watchdog timeout
or log-cap stop was recorded.

The [import receipt](outputs/codeformer_loss_response_v1_return_review/return_import_v1.json)
and [independent transport/runtime audit](outputs/codeformer_loss_response_v1_return_review/independent_transport_runtime_v1.md)
bind all4,609 files. All17 static files match frozen stage r1. Parent dependencies
and the current instruction documents' preserved historical suffixes match their
roles. No returned Python was executed during local auditing.

## 2. Saved pixels and personal visual review

All352 saved native512→RAW256 conversions reconstruct **bit exactly**. All352
PNG files equal the unchanged floor conversion of saved RAW. All32 source
native/RAW/PNG/token sets equal the same-state baseline sets, so their visual
aliases are exact. Eight direct source1024→clean512/256 targets and16 canonical
TRAIN inputs match their original target/input fingerprints. Independent metric
readback differs from the worker by at most2.78e-17 scalar arithmetic.

Native512 remains unclipped normalized CodeFormer output; unclipped/clipped
native RGB descriptors are separate from clamp-before-bilinear RAW256 and PNG.
There are1,280 probe-domain measurement records and128 source-domain records,
with all case, coarse-colour, residual/spatial and token descriptors retained.
These descriptors do not constitute an independently qualified facial-structure
or recovered-identity score.

Root personally viewed **all16 planned sheets**, covering every320 probe PNG and
32 source aliases. The independent sheet audit verifies all416 exact256 cells.
Differences among deletions are mostly small. Severe inputs retain noticeable
expression/eye/mouth/texture discrepancies against their clean targets. The
original-state severe dark case u0006_s2 invents glasses already in its baseline;
the stopped-state examples are often lighter and smoother. Some stopped-state
full and momentum-control outputs look nearly identical. There is no consistent
across-feature visual winner that solves preservation. This is root review of
TRAIN evidence, not user satisfaction or reserved final review.

## 3. What the finite output responses show

PNG mean paired-MAE changes below compare the full proposal with its same-state
baseline; negative is lower error. Each row retains four observations.

| State | A-normal | A-dark | B-normal | B-dark |
|---|---:|---:|---:|---:|
| original0 |-0.00103511 |-0.00723388 |+0.00037699 |-0.00798001 |
| stopped50 |+0.00003855 |-0.00117759 |-0.00007462 |-0.00107942 |

Original0 B-normal has two of four PNG errors worsen, while RAW has three of
four worsen. Stopped50 has two of four normal cases worsen in each A/B group.
All eight dark cases improve under the full proposal at each state. A favourable
mean therefore does not establish uniform preservation.

1. **Removing coarse RGB trades normal error for dark benefit.** Normal PNG MAE
   improves relative to full in6/8 original0 and5/8 stopped50 cases. Dark mean
   MAE becomes worse than full by0.00168097 and0.00026232 respectively, although
   these local dark proposals retain some benefit over their baselines. This
   deletion removed normal and dark coarse contributions together; it does not
   isolate a global-tone cause.
2. **Behavior RGB anchoring is protective at the stopped state.** Its deletion
   worsens8/8 normal PNG cases, mean+0.00010135 versus full, while improving8/8
   dark cases, mean-0.00013332. At original0 its72 physical zero gradient arrays
   make the deletion exactly equal to full. Zero at the initializer does not
   mean the anchor is useless after training.
3. **LPIPS deletion harms normal cases here.** All8/8 normal PNG cases worsen
   at both states. CE deletion improves normal paired errors8/8 at original0 but
   only4/8 at stopped50. Neither gradient size nor one-state success supports
   dropping a loss universally.
4. **Retained momentum matters.** Original0 zero-gradient Adam changes no model
   rows. Stopped50 zero-gradient Adam changes all350 trainable rows and89/4,096
   selected tokens; full changes91 tokens from baseline. Full differs from
   momentum in10 tokens across16 cases. Mean PNG output change is0.0006512
   full-versus-momentum, compared with0.0028255 full-versus-baseline. These
   overlapping, nonlinear responses are not additive causal percentages.
5. **The result is state dependent and TRAIN only.** Same16 observations are
   gradient sources and probes. A local component deletion is not the cause of
   the preceding50-update trajectory, held-out generalization, native CCTV
   performance or a justification to add epochs.

## 4. Gradient and old-state evidence, with explicit limits

The independent audit recomposes every assigned saved gradient in the exact
seven-component float32 order: **3,150+3,150=6,300 exact digests**. `None`, real
zero arrays and explicit momentum-control zero gradients stay distinct. Routes
agree: CE/KL reach encoder/predictor, HQ reaches encoder, and the four image
objectives reach fusion under isolated AdaIN. Thus image-objective changes alone
cannot directly constrain the encoder's CE/HQ/KL path.

All34 source-recorded direct-full/component checks meet their frozen numerical
bound. Direct full vectors were not exported; this is not independent autograd
replay. The identity supplement confirms the exact two initializer-parity case
IDs/order and16 repeat IDs/order at each state. Repeated arrays were not returned
separately, so those repeat equality flags remain source-bound runtime captures.

Root separately loaded only the two pre-existing, hash-pinned checkpoints with
`weights_only=True`, CPU mapping and mmap. The base model, fixed-rate Adam and
RNG digests match the diagnostic. A NumPy literal Adam-equation readback verifies
all assigned-gradient hashes and every old parameter base digest, without a
model, autograd, optimizer object/step or CUDA operation.

Of9,270 per-tensor after-hash comparisons,6,607 match:3,286/4,635 original0 and
3,321/4,635 stopped50. All2,970 frozen comparisons match; trainable after hashes
match3,637/6,300, leaving2,663 nonexact. They remain nonexact. The literal
CPU operation order is not a specification of CUDA implementation order, but
rounding alone is not established as the sole cause. Close scalar norms cannot
provide a per-element error bound without the missing full CUDA delta arrays.
No tolerance pass or exact full CUDA replay is claimed.

[CPU readback](outputs/codeformer_loss_response_v1_return_review/CPU_Adam_v1/receipt.json)
and [independent review](outputs/codeformer_loss_response_v1_return_review/independent_CPU_Adam_review_v1.md)
retain these limitations. Model/Adam/freeze/mode/hook restoration after each
branch remains bound to the sealed worker's runtime captures, complemented by
the independently checked old base states and saved gradient/pixel evidence.

## 5. Runtime, checker and export boundaries

NVIDIA L4, Python3.10.12, Torch2.9.1+cu129 and CUDA12.9 are recorded. FP32 is used;
autocast and TF32 are disabled, and cuDNN benchmark is off. Global deterministic
algorithms/cuDNN deterministic mode were false, so universal determinism is not
established. The compatible VM's exact16-input regeneration gate ran before
model construction according to successful bound control flow; per-case generated
camera receipts and actual NumPy/Pillow/OpenCV versions were not exported. The
earlier five local camera-runtime mismatches stay preserved, not relabeled.

Admission projected RAM4,896,053,648/CUDA15,155,644,968 bytes, with available
RAM15,196,766,208/CUDA23,455,727,616 bytes before loading. Successful guards and
initial availability are not measured peak RSS/CUDA or minimum free disk; those
values were not exported. Future finite workers must explicitly record these
measurements. Caps were not exceeded according to bound successful guards.

The checksum and home export receipt are present and verified. Final active
`export.log` and post-export `finished.json` bytes were excluded by the declared
immutable boundary and are unsupplied. Their absence does not negate the actual
completion/immutable export evidence, but they cannot be claimed reviewed.

Two audit-checker assumptions were corrected with exact failed sources retained:
dictionary equality for independently reduced scalar norms and `Counter` applied
to record values instead of keys. One peer substring check was also corrected.
None changes a returned byte, recipe, quality requirement or input label.
Thirteen CPU state/pixel fixtures pass; four corrupted/truncated/concatenated
gzip fixtures are refused and a valid stream roundtrip passes. No new neural,
gradient, optimizer or VM calls occurred locally. Reading old tensors and NumPy
equation calculations are audits, not training or model writes.

## 6. Decision and next bounded question

Retain the previous checkpoints and all failed gates. Keep LPIPS and behavior
anchors in the design review; this return gives evidence against removing them
to fix normal preservation. Do not resume the stopped50 recipe, add epochs,
remove CE universally, or change the application model from this diagnostic.

The next prospective question is whether **normal-only global luminance DC in
the coarse objective** can be isolated from spatial/chroma supervision while
leaving dark coarse supervision unchanged. Compare a normal-coarse omission
control as well as an explicitly defined luminance-centered replacement, with
the same restored states, full objective, baseline and momentum controls.
Per-channel RGB centering would also remove chroma DC; it is not an exposure-only
intervention. Coarse L1 is not an additive DC-plus-residual decomposition.
This is a falsifiable objective hypothesis, distinct from the already rejected
inference brightness gate, and remains unmeasured. Full data/call/resource/stop
controls and independent review are required before another manual packet.

The input-only usable-face threshold remains separately unimplemented and
uncalibrated; no DEV case or usability label is changed using generated images.
The full restoration-first CodeFormer goal stays active: unchanged initializer,
original DGP/failure preservation, source-separated native/unpaired evidence,
untouched FINAL, user acceptance, independent final review and app Auto/override/
PNG/Playwright verification. Seven-family detection/completion improvement stays
deferred until restoration acceptance. No cleanup, Drive transfer, reclaimed
storage, new VM connection or overall-goal completion is claimed here.
