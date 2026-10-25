# Existing U96 decoder RGB readout: one finite manual L4 pilot

Status: source preparation. No trained successor or quality claim. Complete U96 is
the exact retained R2 checkpoint plus its four frozen residual adapters. Optimize
only the stock final Conv2d weight and bias, 1,731 scalars; freeze the other537
tensors. No new head or code-predictor update. Missing anatomy remains uncertain.

Use the exact already admitted TRAIN photographs:16optimization and8adaptive
monitor photographs, each with local Gaussian1.2, global Gaussian1.2, clear and
linear0.35lowlight roles. Retain the24legacy guards on8other photographs. Monitors
are reused adaptive TRAIN evidence, not unseen identities or FINAL evaluation.
Photographic clean256 is the optimization target for all four roles. RGB L1 plus
0.25adjacent residual L1; equal inside/outside weights on the central128local case.
There is no generated behavior-copy optimization loss. Behavior checks remain.

Fresh Adam, maximum LR0.00005, four-update linear warmup, betas0.9/0.99,
eps1e-8, zero weight decay, norm clip1. No AMP, EMA, fused optimizer or historical
optimizer import. At most64updates, four distinct-photograph role microbatches
per update:256exposures, exactly four passes over64correlated cases, not64epochs.

Snapshots0/1/4/16/32/64 retain56monitor/legacy outputs each. Independent complete
U96 references120, snapshots336 and training256 total at most712full forwards.
One separately counted stub exception cleanup probe performs no CodeFormer
forward. No unbudgeted witnesses/retries. Every candidate call uses ordinary
grad mode; the frozen independent reference uses no_grad. Initial full native,
logits, LQ, RAW256 and PNG parity is mandatory before optimizer steps.

All historical RAW/PNG preservation checks remain. At16stop unless global OR
central-local monitor adjacent-error mean improves in BOTH RAW and PNG. At64
require all four global/local RAW/PNG1%structure endpoints and dark MAE/adjacent
nonregression. Retain broader inherited DEV failures; a small TRAIN pass cannot
erase them. Brightness alone cannot qualify a successor. Broader fixed DEV,
all planned visual sheets, user acceptance and independent FINAL remain required.

Worker1800s, external1860s, per-phase180s; external termination grace20s.
Export300s/internal and330s/external. Initial8GiBfree after initializer copy,
continuing4GiBdisk reserve, peak10GiBRSS/18GiBCUDAreserved,2GiBoutput and return
archive caps. Expected10–20minutes is an unmeasured planning estimate. Stop on
timing/resource/nonfinite/gradient/state/parity/preservation failures. No retries.

Save initial and every committed two-tensor state, postclip gradients, fresh Adam,
CPU/CUDA/Python/NumPy RNG, fixed sampler, durable attempted/returned counters,
pending operations and failures. Export failure does not imply training failure;
export completion does not imply training success. Interrupted outcomes remain
unknown until independent evidence resolves them. No automatic resume.

Use a fresh exact VM stage and the sealed guide only after full source review.
Actual model/gradient/optimizer/training calls run manually inside tmux on L4.
Local source/data/transfer audits import no Torch and do not construct a model.
