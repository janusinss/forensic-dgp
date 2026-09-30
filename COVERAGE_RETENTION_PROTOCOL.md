# Retention-constrained feasibility pilot — 30 September 2026

Status: helper and VM runner implemented; nine local tests pass. CUDA execution
is pending. Use `RETENTION_VM.md` for the packaged pilot commands.
This is a bounded feasibility experiment, not a demonstrated quality improvement.

## Evidence and hypothesis

`REPLAY_LOSS_RESULTS.md` shows aggregate loss improvement can outweigh replay
regression. Gradient projection also failed the original selection gates. Test
whether checking actual candidate parameters against fixed training-only replay
loss ceilings permits useful real-mask adaptation without that observed tradeoff.
Do not infer validation retention from satisfying a training-loss constraint.

## Fixed recipe

- Start at the original parent, SHA256
  `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`.
  Freeze the generator. Use extended73 real data and the existing638 cached replay
  cases, with the existing protocol and inventory hash checks.
- Attempt only the first21 batches of the frozen extended schedule (epoch1),
  batch8. Keep ordinary mixed supervised loss plus teacher KL, AdamW nominal
  LR1e-5, weight decay1e-4, gradient clipping1, visible penalty0.25. No projected
  gradient, new data, threshold search or additional epochs in this pilot.
- At initialization, measure supervised replay loss separately on covered and
  clear cases using all638 cached cases, weighted by their complete frozen
  schedule exposures (420 slots per class). These two GPU-measured parent values
  are immutable ceilings. Do not substitute rounded CPU report values. Teacher
  KL remains in the update objective but is not a zero-valued acceptance ceiling.
- For each scheduled batch, compute its gradient once. Trial LR factors are
  exactly1,0.5,0.25,0.125 relative to nominal. Evaluate all638 replay cases after
  each trial. Accept only finite covered AND clear losses no greater than their
  parent ceilings plus absolute1e-6 numerical tolerance. Limits never accumulate
  from the previous step. All trials start with the same pre-batch parameters,
  gradients and optimizer state; only an accepted trial advances AdamW history.
- Stop after21 scheduled batches or3 consecutive batches with all trials
  rejected. Report stop reason and attempted/accepted counts. Maximum84 trials,
  hence53,592 candidate replay image evaluations, plus638 parent evaluations,
  ordinary batch work and validation/export. This cap is a compute bound, not a
  wall-clock estimate. No automatic continuation, looser ceiling or retries beyond
  these bounds if progress stalls.

## Transaction and execution requirements

`coverage_retention.guarded_step` snapshots module parameters/buffers, optimizer
moments/step counters, gradients, module modes and Torch RNG state. Failed trials
and exceptions restore them. Accepted trials retain updated optimizer state but
restore nominal LR for the next batch. Evaluator must be deterministic, use eval
mode, and avoid external writes. Current detector uses GroupNorm; do not silently
introduce train-time BatchNorm/dropout or augmentations into this transaction.

The VM runner must evaluate replay under inference mode, preserve exposure weights,
and verify group counts. Log each trial's LR factor and both loss values, batch
indices, acceptance, ceilings, all artifact hashes and elapsed time. Check frozen
generator equality. Preflight is CUDA forward/evaluation only, zero updates.
Run actual training only on the Linux CUDA VM. Do not execute a model optimizer
locally; the helper tests use scalar fixtures only.

## Interpretation and outputs

Evaluate unchanged425 development cases at parent and final retained state,
including separate human/mannequin/glare reports, and export masks/checkpoint even
if the pilot stalls. Preserve original real/synthetic selection gates. Save a
selected candidate only if both gates pass; application promotion additionally
requires reviewed end-to-end completion improvement and the10-row visual grid.
No acceptance decision uses validation or test gradients/losses.

The existing ordinary epoch1 is a historical comparator at21 attempted scheduled
batches. It is not matched on accepted updates or compute if trials reject, and
an early-stopped pilot is not a matched21-batch comparison. State these differences
instead of attributing changes solely to the constraint. A stalled pilot is useful
negative feasibility evidence, not permission to weaken the original safeguards.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/coverage_vm_bundle/`.
Planned VM output: `outputs/retention_training_vm/` under that root.
Planned return archive: `retention-results.tar.gz` under that root, downloaded to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\retention-results.tar.gz`.

Next: execute CUDA preflight and the bounded pilot using `RETENTION_VM.md`, then
return the results archive for independent recount and checkpoint verification.
