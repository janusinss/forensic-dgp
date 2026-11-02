# Predictor-only completion learning from R1

26 October 2026. **Prospective finite design; runtime preparation is in progress.**
No transfer archive is admitted, no new training has run and R1 stays selected.
The [completed same-codebook probe](COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md)
supports this distinct test. Its oracle gain is not a promised training gain.

## Question and fixed model paths

Can ordinary covered-input code prediction improve completion while R1's context
features, learned fusion and original inpainting prior stay fixed?

Initialize from the exact complete R1 parent and original CodeFormer inpainting
checkpoint. Train only 114 existing tensors / 19,450,880 scalars:
`position_emb`, `feat_emb`, `ft_layers` and `idx_pred_layer`. Freeze all other
383 tensors: encoder164, fusion54 and original quantizer/generator165. Keep the
network in evaluation mode; only the intended predictor parameters require
gradients during FIT. Inference remains ordinary hard top-1, fusion w=1,
no AdaIN and the unchanged app input/delivery path. No oracle indices or clean
RGB are supplied during evaluation or deployment.

The actual probe showed useful code-choice opportunity at fixed fusion. Actual
no-fusion outputs worsened all 48 holes and boundaries. V2 changed both encoder
and fusion; V3/V4 trained fusion with the predictor fixed. This experiment
isolates a different learning path rather than repeating those stopped routes.

## Objective and optimizer

Use mean cross entropy over the 256 code positions against the retained clean
reference's nearest indices in the **same original inpainting codebook**. Divide
each one-case loss by four exactly once before accumulation; four distinct FIT
references form one optimizer update. No pixel, boundary, adversarial or latent
MSE training term, image-gradient surrogate, straight-through estimator, new
teacher or substituted HQ512 code IDs.

The frozen encoder's `lq_feat` has no learning path to the post-encoder predictor,
so an MSE on it cannot train these 114 tensors. The official architecture exposes
`code_only` logits for CE; its optional feature target is a different path.
This supports the interface, not our exact freeze policy or expected quality.
[Official index-training source](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/models/codeformer_idx_model.py).

Construct one Adam optimizer for the exact 114-name allowlist. Map R1's per-name
moments and step32 into the new 114-parameter topology; do not reset them or load
278 positional IDs blindly. Retain LR1e-5, betas(0.9,0.99), epsilon1e-8, zero
weight decay and global L2 clip1. Verify every copied slot before the first
update. Restore the exact parent RNG boundary after initial parity/readouts;
the new schedule is already materialized and consumes no global RNG.

All intended gradients must exist and be finite; frozen parameters must have
no gradients. Record every intended parameter's gradient norm and per-path
norm, clipping and actual update deltas. A numerically zero entry must be
reported honestly; no invented gradient, optimizer step or image gain follows.

## Fixed data and exposure

Use the immutable balanced128 subset and 32-cycle schedule derived from the
retained 256-case FIT pool. Each of 32 FIT photograph reference IDs contributes
four existing observations: one per mask geometry, two clear anchors, one gray
augmentation and one RGB augmentation. Totals are64anchors/64augmented,
96RGB/32GRAY,16per augmentation condition and all64 FIT RGB/GRAY label sets.
Each cycle contains four distinct references and the same 2+1+1 role balance.

These are **128 selected observations, not an epoch over all256 cases**. The
full pool stays intact. Selection used reference order/roles, not image quality
scores. The eight adaptive monitor reference IDs and all16 of their label sets
never optimize. Preserve their exposure history; do not call them independent
validation. Photograph-ID separation does not establish distinct real people
or absence of pretraining overlap. Reserved final evaluation remains separate.

Reuse all80 existing original-inpainting teacher NPZs:64FIT targets,16monitor
diagnostics. Their old unquantized latent arrays stay retained but unused.
Original label, clean512, provenance, terms and codebook hashes must join exactly.
No new teacher forward or target generation is permitted by this design.

## Finite execution and saved evidence

Maximum32 new Adam updates /128 backwards. Read out all48 paired cases at
updates0,8,16,32, and all32 fixed public development photographs at0and32.
Public photographs are unpaired; do not compute hidden-region target scores
for them. All prepared images and reviewed masks remain unchanged.

The complete run plans:

- 256 saved evaluation outputs, including8 public empty-mask bypasses.
- 248 decoded evaluation calls plus8 calls for four fresh ordinary app parity
  pairs:256 full generation calls.
- 128 code-only FIT calls:384 model invocations overall. The common forward hook
  sees code-only calls too; do not call all384 full generation calls.
- 160 additional empty-mask control bypasses, for168 total including saved
  public controls. Zero teacher calls, fusion updates or oracle overrides.

Enforce each work-type ceiling before dispatch, plus an overall420-call hard
ceiling. All four fresh parity pairs and all48 initial R1 RAW/PNG historical
matches must pass before any backward/update. Save every128 FIT logits array,
target-code array and case/cycle/microbatch metadata so CE can be independently
replayed. Save all256 evaluation arrays/PNGs, predicted codes, four parity
bundles, all per-case metrics and failure receipts. No extra FIT pass is hidden
in diagnostics.

Intermediate8/16 readouts are safety checks only. Stop on nonfinite state/loss/
gradients, changed frozen state, wrong ownership/counts, resource limits or the
retained10% catastrophic RAW/PNG hole/boundary regression rule on either paired
cohort. They are not selectable winners. The final32 endpoint alone can satisfy
the prospective numeric screen. A failed final endpoint cannot be replaced by
an earlier image snapshot or an automatic appended epoch.

## Quality requirements

Keep the existing separate clear and challenge >=1% delivered PNG hole-L1
reduction versus R1, with all declared RAW/PNG macro, geometry and challenge
condition hole/boundary subgroups nonworse within1e-12. Exact outside-mask float
and PNG source pixels, unchanged empty bypasses and fresh ordinary parity remain
mandatory. Report cumulative original-baseline results separately.

Add a prospective gray-appearance guard because the probe exposed false color
despite passing aggregate image errors. On the exact eight strict-gray challenge
cases, compute generated-support mean(maxRGB−minRGB) separately for float RAW
delivery in0–1 and PNG in0–255. Final32 must be nonworse than R1 for the macro
and each of four geometry groups, with1e-12 tolerances in the respective units.
Retain every individual value and inspect all eight. Two cases per geometry
are a narrow regression guard, not a color-validation study. No recoloring or
postprocessing is introduced to pass it. Intermediate gray results are reported;
they do not select a checkpoint or replace the old catastrophic rule.

The same-codebook gray targets themselves worsened oracle chroma. This pilot
may fail the stronger guard, which must remain visible. Lower CE, higher code
agreement, completed training and successful recovery-state export are not
quality acceptance. Inspect every planned saved comparison, including public
covering families; unknown hidden expressions/appearance remain estimates.

## Resources and failure recovery

Prospective limits for this new unsealed design:30-minute worker hard limit,
28-minute cooperative neural-work deadline and120-second save reserve;
20GiB CUDA allocation cap;8GiB free plus installation payload at start;
2GiB free floor. Require owned subprocess supervision and finite export900s.
These are finite caps, not measured training-time promises.

Reserve1GiB for one terminal checkpoint/failed partial,2.5GiB for all output
payloads,24MiB for retained teacher arrays,8MiB metadata including4MiB stdout,
and256KiB per PNG. Raw streamed tar and gzip each have a2.75GiB limit. Payload
and compressed archive can coexist within5.25GiB, below the6GiB start-to-floor
span. The raw tar is streamed and does not add a third disk-sized copy.

Save actual332 non-prior tensor state, merged historical278 code-Adam slots
with only114 updated, unchanged54 fusion-Adam slots, RNG, full new sampler,
progress/counts and any accumulated114 gradients. Preserve original mapping and
step32 moments for frozen164 encoder slots. Do not duplicate the active114
moments again in the checkpoint. Declare mixed per-name steps explicitly;
this is not a secretly active278-parameter optimizer.

If an unexpected original-prior mutation occurs, retain its actual165 tensors
as failure evidence within the1GiB terminal reservation. Always hash all497
actual tensor bytes and record finite flags before rejecting them. Save once
through the bounded writer, fsync/hash and exclusively commit; failed partials
remain. Hard kill, disk or serialization failure can still prevent recovery and
must be reported, never reconstructed as a successful checkpoint. No automatic
resume, overwrite, original-checkpoint mutation or historical pilot launch.

## Before any transfer or adoption

Runtime/source and helper tests, exact gradient/optimizer ownership witnesses,
serialization/namespace/resource fixtures, independently audited protocol/data
and the actual import-blocked transfer CLI must pass before sealing a packet.
The current source draft is not ready to run. Only then publish a verified
manual Windows upload / VM install / tmux / single-source download guide.

Returned weights, Adam/RNG/sampler, all saved RAW/PNGs and every visual sheet
must be audited independently. A useful accepted model must then pass broader
development, user comparison, independent final human review and Windows CPU
app/install checks. Automatic proposals and selective IMPLEMENTATION integration
remain separate unfinished requirements. The existing R1 app and restoration
sources/checkpoints remain unchanged throughout this preparation.
