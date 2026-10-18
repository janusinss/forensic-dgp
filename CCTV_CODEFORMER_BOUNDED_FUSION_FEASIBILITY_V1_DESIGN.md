# Bounded fusion feasibility V1 — prospective zero-update TRAIN measurement

**16 October 2026: DESIGN ONLY. No worker, protocol, execution archive or upload/launch command is implemented or sealed.** The next safe action is a finite measurement of a nonzero, stopped FP32 fusion delta before spending on two-arm training. It does not resume training, replace the original benchmark, or qualify the failed model. All actual neural work, if subsequently prepared and independently reviewed, remains the user's manual L4/tmux workflow.

Proposed new VM directory: `~/forensic-dgp/cctv_codeformer_bounded_fusion_feasibility_v1_vm`; proposed export prefix: `cctv-codeformer-bounded-fusion-feasibility-v1`. These names reserve a new namespace, not an existing prepared packet. A future protocol must bind the final source, immutable parent assets, all input/target/calibration hashes, call ledger and exact hard limits. No protocol/archive SHA exists yet.

## Question and evidence boundary

Can a simple observed-input gate on a bounded fusion difference improve the stopped FP32 model's normal/dark tradeoff beyond the same bounded difference with a constant gate? This is a **conditional inference feasibility question on a finite TRAIN cohort**, not a claim that a newly trained gate or fusion copy will learn that result.

The actual FP32 first50 run removed the initial behavior-image mismatch but failed the same three clear cases and normal-blur guards as BF16. Previous original-fusion interventions used the older V2 state and eight TRAIN observations. This measurement instead uses the returned **FP32 stopped state**, a new photometry-stratified32-observation TRAIN cohort, an explicit feature bound and a fixed input gate. It does not rerun the old precision/gradient sweep. No gradient is needed to answer this conditional question.

The current model is a credited CodeFormer adaptation, initialized from untouched official CodeFormer. Do not relabel it as a newly invented DGP architecture. Restoration remains first; detector/completion development and application adoption stay deferred.

## Immutable states and actual architecture

Use untouched official CodeFormer weights SHA-256 `1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7` and the returned FP32 stopped checkpoint SHA-256 `133c5a1e15e069bdd2abda4ebf818d3f4ff59b966677762753e894ff42e2583d` (855,229,944 bytes). The saved stopped model state digest is `51605ee7908c40a457e41ac03c614211e8028cc0386b85d13aa59a3d19a182eb`. Read checkpoints as CPU `weights_only`; load only the model state into inference modules. Never instantiate Adam, restore/advance RNG or sampler, call backward/autograd, or save a `.pt/.pth` result. Keep all original/stopped sources, tensors and `.grad` fields unchanged.

The immutable architecture source is `third_party/codeformer/codeformer_arch.py`, SHA-256 `ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb`. Its fusion block returns `dec + w*(dec*scale + shift)`, with learned scale/shift from concatenated encoder/decoder features. Four sites, in decoder order:

| Site | Tensor shape per example | float32 bytes per tensor |
|---|---|---:|
|32|`(256,32,32)`|1,048,576|
|64|`(256,64,64)`|4,194,304|
|128|`(128,128,128)`|8,388,608|
|256|`(128,256,256)`|33,554,432|

Encoder skip tensors are detached at fusion; AdaIN still carries the stopped encoder's moments. Keep FP32/no-autocast, TF32 off, eval/inference mode, w=1, detach_16=True, AdaIN=True, original hard code selection, unchanged generator/codebook, input256-to-bilinear512 and delivery policy. Gates broadcast as `(N,1,1,1)`; bounds as `(N,C,1,1)`. Do not alter codebook indices, encoder features/moments or geometry when changing the fusion mode.

At each intervention site compute frozen original `F0(e,x)` and frozen stopped `FS(e,x)` on **that mode's same current tensors** `e,x`:

`D = float32(.10) * sqrt(mean_xy(F0(e,x)^2) + float32(1e-6))`

`delta = D * tanh((FS(e,x)-F0(e,x))/D)`

`out = F0(e,x) + float32(g(input256)) * delta`.

The scale is per-example/per-channel; reduction, square, square-root, difference and tanh must remain float32 with a source-frozen operation order. The proposed .10 bound is a predeclared feasibility choice, not a tuned optimum or accepted training hyperparameter. A feature cap is not a bound on delivered pixel error. Earlier fusion changes modify later decoder tensors, so do not substitute cached per-site outputs from another branch.

## Fixed observed-input gate and measured ambiguity

Compute a gate without new learned parameters: on prepared RGB256 sRGB float64 `[0,1]`, `Y=.2126R+.7152G+.0722B`, and `z=mean(Y[32:224,32:224])`. The central192 support computes a scalar only; the model still receives the full unchanged256 image. Calibrate linear-interpolation q10/q90 across all200 retained TRAIN input RGB images, with no clean pixels, profile/reference labels, output metrics or DEV/FINAL inputs used in calibration. Freeze:

`q10 = 0.21007671428419458`, `q90 = 0.5546973290483728`.

`t=clip((q90-z)/(q90-q10),0,1)`; `g=.25+.75*t*t*(3-2*t)`.

The same scalar, range `[.25,1]`, gates every site. Fail on nonfinite or degenerate calibration; do not invent a fallback. Freeze coefficients, central support, quantile method, floor, ceiling and beta before any new forward. No gate tuning/grid is permitted after seeing outputs. Gate computation must use observed input only during **both training and inference**; targets may be used only for declared losses or separate paired measurement. Simulator exposure values, normal/dark tags, source/reference IDs, DEV/FINAL labels and user annotations never enter routing.

This is **not an exposure detector**. A CPU-only audit matched all200 input hashes and found100/100 matched simulated dark members receive at least as much gate strength as their normal partner. However normal/dark z ranges overlap over `[0.2320324,0.5404941]`:1,103/10,000 cross-photograph comparisons rank the normal input darker. That is a descriptive pairwise-overlap count, not a population classification error. Skin/visible appearance, hair, background and framing affect the statistic; the data do not isolate those causes or support ethnicity/bias estimates. A low-luminance normal input may receive nearly the full delta, while a brighter underexposed input receives little. This risk is the reason for a finite feasibility measurement before training.

Calibration evidence: `outputs/codeformer_fp32_comparison_v1_return_review/fixed_gate_TRAIN_input_readiness_v1.json`, SHA-256 `5d8dbbf3b660a911dd76eac187524ff0097b4fbba24d56b8a71a1a25ff04b064`.

## Deterministic32-case photometry-overlap selection

Selection uses the already retained first200 TRAIN observations and observed-input z only, before any new model output. Normal/dark metadata is used solely to balance diagnostic sampling; it is never a gate input. Treat a matched pair as `(update,A)` for slots1/2 or `(update,B)` for slots3/4.

For the100 normal observations, take eight equally spaced quantile levels0,1/7,...,1 with linear interpolation. At each level choose the nearest z from a pair not already selected; tie-break ascending update then slot. Repeat the same eight quantiles on the100 dark observations, excluding pairs already selected. Include both members of each selected pair. Sort the final cohort by update/slot. This fixes16 pairs/32 observations/16 photographic reference IDs, including the darkest normal and brightest dark stress cases. It is not an identity-generalization sample and makes no claim of16 distinct people.

The selected pairs are `u0002-B,u0007-B,u0010-B,u0011-A,u0015-A,u0016-A,u0017-B,u0019-A,u0027-A,u0030-A,u0032-A,u0033-B,u0037-A,u0045-A,u0047-B,u0048-A`. The actual selected manifest retains exact paths, bytes/hashes, calibration, selection order and gate values:

`outputs/codeformer_fp32_comparison_v1_return_review/bounded_fusion_feasibility_TRAIN_selection_v1.json`, SHA-256 `44a999a9c88baf77cc393310a1c6452b74d3f329211b2d8fe1f18bb7af42a86e`.

Use two deterministic canaries: lowest-z normal `u0011/slot1` (z=.23203241387527238, g=.9912552878173654), and highest-z dark `u0032/slot2` (z=.5404941074559164, g=.2537168415025497). These challenge the gate's ambiguity; do not replace them with aesthetically successful outputs. This algorithm, unlike the earlier readiness addendum's simple fixed-update suggestion, is the prospective cohort for this new design only. Prior notes/designs remain unchanged.

## Five paths and successful call ledger

For every selected input run these complete paths, in source-frozen order:

| Mode | Encoder/token/AdaIN | Fusion | Full CF forwards | Fusion-block evaluations |
|---|---|---|---:|---:|
|Original|untouched original|original|32|128|
|Stopped|returned FP32 stopped|stopped|32|128|
|Rollback reference|same stopped|original, g=0|32|128|
|Bounded constant|same stopped|original plus bounded delta, g=1|32|256|
|Bounded fixed input gate|same stopped|same bound, fixed g(input)|32|256|

For modes Stopped/Rollback/Bounded constant/Bounded fixed, capture and compare exact selected tokens, logits hashes, latent feature hashes, every detached encoder skip hash and the actual pre-generator AdaIN tensor/moments. The same source/state/input precision must produce identical stopped content across those modes. Do not recompute code selection from printed logits as a substitute for captured selected indices. Each mode's downstream decoder context differs after fusion; that is the intended intervention. Hash original/stopped model states before/after, require all `.grad is None`, and restore any temporary method/hook in `finally` before a repeat or export.

Additional canaries:

- Repeat native Original and Stopped for each of the two canaries:4 full CF forwards/16 fusion evaluations. Repeats must match their earlier RAW/tokens exactly under the frozen deterministic runtime.
- Zero-delta original/original wrapper with constant and conditioned gates for both canaries:4 forwards/32 fusion evaluations. Both blocks use original weights on the same context; the difference must be zero and outputs match the untouched original. This verifies the proposed initialization without relying on tiny newly learned gates.
- Native identity wrapper for each stopped canary:2 forwards/16 fusion evaluations. It evaluates both fusion blocks but returns `FS` directly, without subtract/add, and must exactly match native Stopped. This tests source/context/hook restoration without assuming subtract/add is bit-exact. The native-identity canary supersedes the older readiness addendum's optional unbounded arithmetic canary for this design; no existing file is rewritten.

**Successful totals:170 full CodeFormer forwards,960 fusion-block evaluations,0 HQ/behavior-teacher forwards,0 LPIPS,0 gradients/backward,0 optimizer attempts/updates.** Count every full forward and every original/stopped fusion-block evaluation before it occurs. Do not count a dual-block mode as one evaluation or fabricate successful totals after an early stop. No precision sweep, generation/completion or extra optimization is in scope. Codebook lookup/encoder/transformer/generator passes occur only inside the170 declared full forwards. A source-derived terminal ledger must account for all canaries before packet sealing.

## Evidence payload and tensor-memory bounds

Save every native512 RGB float32 prediction, declared delivered256 RAW array and exact floor-to8bit PNG, plus actual selected token indices, input/source/state/mode bindings, gate, relevant content hashes and timings. Keep native RAW predictions unbounded; deliver by the unchanged clamp-before-bilinear256 policy. Use clean512 and direct clean256 targets only for paired metrics, never to generate/correct outputs or choose a route. Bind their genuine source and derivation hashes. No clean target is concatenated to model inputs or gate features.

For every dual-block call save per-channel D, original/stopped difference maxima/RMS, used delta maxima/RMS, nonfinite counts, tanh saturation fractions and actual gate. These are captured sufficient statistics, not proof of replaying every internal feature. For both canaries in both bounded modes, additionally retain actual `F0`, `FS` and used delta arrays at32/64 sites. The independent CPU auditor can recompute D/tanh/gate on those returned arrays without any NN/grad calls. Stop and preserve evidence if state/content restoration, finite arithmetic or local bound checks fail; do not replace a failed mode with a different gate/beta.

Source-derived array sizes before PNG/NPY headers and metadata:

| Saved artifact | Bytes |
|---|---:|
|170 native512 RGB float32 arrays|534,773,760|
|170 delivered256 RGB float32 arrays|133,693,440|
|2 canaries ×2 bounded modes ×3 arrays at32/64|62,914,560|
|170 selected-token int64 arrays|348,160|
|Maximum170 uncompressed RGB256 PNG pixel payload before encoder overhead|33,423,360|

A PNG file must have a strict implementation byte cap rather than treating raw pixel payload as a file-size guarantee. Whole-output file counts, NPY/PNG/JSON bounds and archive overhead require a complete sealing ledger. Candidate preliminary ceilings below include a substantial margin; they do not justify unlimited records.

At batch1, the four encoder skip tensors total47,185,920 bytes (45MiB), already cloned by the original architecture. Largest site output is33,554,432 bytes (32MiB). A straightforward bounded block can have up to seven such explicit arithmetic tensors,234,881,024 bytes (224MiB), plus the original block's concatenation/ResBlock intermediates and CUDA workspace. This is a named-tensor accounting bound, not a complete GPU peak guarantee. Stream side statistics and CPU canary writes at each site; release diagnostic tensor references before the next site/case. Never retain all32 cases' full GPU features. Do not clone whole models per mode or accumulate full native activation histories; use one original and one stopped module with controlled fusion access. Inference-only original activations must stay detached; no autograd tape is needed. Checkpoint CPU load may temporarily include optimizer tensors from the saved file, but no optimizer object may be constructed; discard unused loaded entries promptly and record actual host/GPU peaks.

## Finite resource stops and prospective readiness

| Resource | Proposed hard ceiling, to be verified against implementation |
|---|---:|
|Model/source/input initialization|300 seconds|
|Neural measurement including all canaries|600 seconds|
|Whole worker / external termination grace|900 /60 seconds|
|Export / termination grace|300 /30 seconds|
|CUDA current and peak reserved|22GiB|
|Initial free space after installation / retained reserve|8 /4GiB|
|Result payload / compressed archive|1.25 /1.35GiB|
|Regular output files / total metadata|2,500 /32MiB|
|Noncheckpoint temporary write reserve|128MiB|

These are design ceilings, not measurements of a built worker. Prove exact buffers, file caps, call order and monotonic timers from the actual implementation before sealing. Use exclusive new result/lock files, fresh GPU/workload checks, process-group supervisor timeouts and a terminal-state check before export. Fail finitely on time/projection, disk, memory, source/state/content or namespace violations. Export must preserve failure evidence without saved adapted weights. No automatic retries/resume, old diagnostic execution, cleanup/deletion or unchanged pilot launch is authorized. Reusing static verified dependencies and existing weights avoids new large uploads; runtime must never silently choose an unverified parent path.

## Interpretation and decision after return

Report all five paths on identical32 TRAIN inputs, with paired clean metrics separated from rendered-image observations. Preserve native512 versus delivered RAW/PNG; summarize normal/dark and their actual severity variants, and show every case. Check tone, contrast, coarse RGB, gradients and visible facial features separately. No native CCTV, reserved DEV/FINAL or held-out quality claim follows from this TRAIN experiment. The gate only seeing input does not make the evaluation independent of its calibration distribution.

Existing61-clear/53-normal-blur DEV guards remain authoritative and unchanged for any future trained candidate; this diagnostic does not evaluate all of them, cannot claim to pass them, and cannot waive either previous failure. The existing failed FP32 model stays failed even if an intervention looks better. Any clear TRAIN controls can be shown against the same original threshold values as a descriptive subset only; do not rename that subset as the full guard.

A useful positive result needs actual movement beyond the bounded constant control toward normal appearance preservation while retaining helpful dark visibility, without new central defects, false detail or arbitrary suppression. A gate merely following brightness or shrinking all effects is not demonstrated separation. Show individual regressions, not only a pooled mean. Root and independent review must inspect every planned sheet; the user's visible-quality preference remains required. Define the final reporting/advancement rule prospectively in the protocol, without optimizing thresholds on DEV or selecting beta/quantiles from clean output scores. No automatic advance to training occurs.

If the fixed gate worsens overlap stress cases or has no useful tradeoff, reject this simple photometry gate rather than running a parameter grid or increasing epochs. A positive finite measurement only justifies a separately reviewed training design; frozen stopped-state intervention cannot predict learning dynamics. Remaining encoder/AdaIN drift, delivery supervision and preservation coverage stay explicit alternative factors. All original checkpoints, splits, targets and gate failures remain protected. The thesis goal remains incomplete.

## Evidence links

[Actual FP32 return](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md), [post-FP32 direction](CCTV_CODEFORMER_POST_FP32_DESIGN_REVIEW_V1.md), [independent state/loss evidence](outputs/codeformer_fp32_comparison_v1_return_review/next_design_state_review_v1.json).

This new document specifies a readiness measurement; it does not mutate the earlier approved design or frozen worker. Its calibrated scalar/selection records were computed from saved input PNGs with CPU arithmetic only. No new model construction, neural inference, gradient, optimizer, VM action or application change occurred while writing it.
