# Projection step magnitudes — 30 September 2026

Recomputed descriptive statistics from all420 saved optimizer-step records;
no new model inference, gradients or fitting. Source log hashes and per-epoch
breakdowns are saved in `outputs/projection_validation/step_magnitudes.json`.

| Quantity | Ordinary | Projected |
| --- | ---: | ---: |
| Replay-ascent steps | 95/210 | 71/210 |
| Sum of positive first-order replay changes | 0.010489 | 0.005280 |
| Sum of negative-change magnitudes | 0.328433 | 0.326119 |
| Positive share of absolute directional magnitude | 3.09% | 1.59% |
| Median parameter-step norm | 0.002691 | 0.002674 |
| Largest positive first-order change | 0.000828 | 0.000950 |

On113 projected steps, the removed component has median norm14.86% of the
real-gradient norm (maximum56.63%). Positive first-order changes remaining on
projected steps sum to0.002962. The maximum individual positive change increased,
even though the total positive magnitude decreased. Nothing was clipped out of
these statistics by a post-hoc significance threshold.

These sums combine different batches and changing model states. They are **not**
an accumulated loss trajectory or a measured increase/decrease of held-out loss.
The reference gradient also combines replay supervision with teacher consistency;
it is not an IoU gradient. The logged vectors are unavailable, so the scalar
statistics are a verified-script report, not independent vector reconstruction.

## Decision

Do not infer from95 versus71 ascent steps that actual-step projection will solve
forgetting. The positive share is small in this aggregate, and reducing it by
about half did not restore validation quality. Projection also removed real
learning components while real training fit was already weak. This supports
ending this raw-projection recipe, not immediately sweeping its strength or
changing optimizers based solely on gradient signs.

Next diagnostic: compare parent/ordinary/projected fit on the638 archived replay
training cases against their400-case synthetic validation performance, grouped
by covering/degradation. Use inference only. This distinguishes failing replay
training fit from failure to retain performance on held-out synthetic variations.
It can inform whether the next intervention concerns representation, replay
coverage or optimization. Preserve the original selection safeguards and avoid
choosing a new training recipe before that comparison. No model promoted.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM experiment: `~/forensic-dgp/coverage_vm_bundle/outputs/projection_training_vm/`.
