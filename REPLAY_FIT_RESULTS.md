# Archived replay fit — 30 September 2026

CPU inference on all638 distinct replay cases actually sampled by the matched
coverage/projection schedule. Cache, checkpoint and protocol hashes checked.
No optimizer updates; prediction threshold remains0.5.

| Metric on seen replay cases | Parent | Ordinary | Projected |
| --- | ---: | ---: | ---: |
| IoU | 0.96904 | 0.94926 | 0.95078 |
| Missed fraction | 0.01582 | 0.03384 | 0.03370 |
| Visible false-positive fraction | 0.001599 | 0.001823 | 0.001671 |
| Clear cases with false masks | 2/262 | 11/262 | 9/262 |

All eight covered kind/degradation groups lose IoU relative to parent. Largest
projected loss is degraded irregular:0.96109 ->0.91824. Degraded object remains
the lowest projected training-group IoU:0.90596. Clean lower/eyes/object remain
above0.979 but still regress. Empty/empty cases use the existing aggregate IoU0
convention; inspect clear-case false positives rather than interpreting that0
as an accuracy score.

Final synthetic validation IoU is0.95082 ordinary /0.95268 projected, compared
with parent baseline0.97469. Training aggregate composition differs:376 positive
and262 clear unique cases versus320/80 validation. Cases sampled more than once
are counted once here. Do not interpret a direct train-validation difference as
a matched generalization gap; grouped values are available in the machine report.

## Finding and next decision

Forgetting occurs on the exact replay inputs seen during fitting, not only on
unseen synthetic variations. This rejects an explanation based solely on missing
validation coverage. Real-image fit also remains weak, so neither increasing
replay diversity nor another unchanged projection run is justified by these data.

Before a new training proposal, quantify loss tradeoffs on these same stored
inputs: teacher consistency and supervised BCE/Dice/visible penalty at parent,
ordinary and projected states. The gradient audit concerns their combination,
whereas selection concerns binary segmentation. A decomposition can establish
whether the optimized surrogate improves while the selection metrics worsen.
Use inference only and do not tune thresholds or change gates. No model promotion.

Evidence: `outputs/replay_fit_comparison/results.json` (per-case counts and
kind/degradation groups), `scripts/compare_replay_fit.py`. Count sanity checks
passed for perfect positive masks and false positives on empty targets. No visual
inspection of this replay batch was performed in this turn.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM experiment: `~/forensic-dgp/coverage_vm_bundle/outputs/projection_training_vm/`.
