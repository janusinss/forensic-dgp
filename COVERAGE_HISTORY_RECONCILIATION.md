# Coverage diagnosis versus earlier experiments — 30 September 2026

Read the original machine-readable overfit, learning-rate, real-only and penalty
results. Verified common parent checkpoint hashes and both reused-control result
hash links. Evidence: `outputs/coverage_history_reconciliation/results.json`.
No fitting or model change occurred.

| Earlier intervention | Observed outcome | What it rules out at tested settings |
| --- | --- | --- |
| Four-example fitting,80 updates | Current LR fit IoU0.8913; higher LR also fitted | A universal inability to learn any complete mask |
| Full mixed LR1e-5 versus1e-4,84 updates | Training IoU0.2424 versus0.3874; synthetic0.9724 versus0.8587 | Raising LR alone as a demonstrated retention-preserving fix |
| Remove replay at LR1e-4 | Training IoU0.3301 versus mixed0.3874; synthetic0.6038 versus0.8587 | Removing replay as a demonstrated solution to broad poor fit |
| Remove visible penalty at LR1e-5 | Training IoU0.2921; real FP worsens, all gates fail | Dropping the penalty as an established output-quality improvement |

Historical labels are V2; new coverage labels are V3 plus five additions.
Historical replay was178 FFHQ/22 Asian versus current100/100; teacher weight was0
versus current1; budgets and real exposure differ. Therefore these are prior
negative tests of simple hypotheses, not a matched causal ranking against the
latest run. Reports of historical local training predate the user's VM-only rule;
no new local training is authorized or performed.

The earlier loss audit used six real-only batches. It measured penalty-versus-
BCE/Dice parameter gradients, **not** real-versus-synthetic replay gradients or
teacher-consistency interaction. The real-only removal experiment changed batch
weighting as well as removing replay and cannot isolate this interaction.

## Next bounded diagnostic, before further fitting

Prepare a VM-only zero-update gradient audit at the initial and current final
checkpoints, using a fixed prefix of the already frozen mixed training schedules.
Decompose real supervised, replay supervised and teacher-consistency contributions
with their actual batch weights. Verify the sum reproduces the full gradient,
then report norms and pairwise cosines, including original versus added covered
cases where sampled. Archive exact case IDs and checkpoint hashes. No validation
or test gradients, no optimizer updates, no threshold tuning.

This can determine whether competing directions are present at those sampled
states; it cannot prove that gradient conflict causes poor validation or identify
the optimal update. It should guide one specific intervention only if evidence
supports it, rather than trigger another broad learning-rate/loss sweep.
Actual VM execution is pending; no new training command yet.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM data/checkpoints: `~/forensic-dgp/coverage_vm_bundle/outputs/coverage_training_vm/`.
