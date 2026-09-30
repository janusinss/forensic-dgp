# Detector decision review — 30 September 2026

Decision: stop extending the current frozen-feature head series without a new
evidence-backed hypothesis. No recent candidate satisfies the existing safeguards.
No application/generator change, gate relaxation, threshold search or extra VM run.

Recomputed from completed saved case counts; no new inference or fitting.
Machine-readable evidence: `outputs/detector_decision_review/results.json`.

| Candidate | Synthetic gated IoU | Label-informed presence IoU |
|---|---:|---:|
| Expanded fixed | 0.94662 | 0.95218 |
| Expanded anatomical | 0.93227 | 0.93899 |
| Extra-update control | 0.95096 | 0.95672 |
| Border weight 2 | 0.94910 | 0.95490 |
| Semantic residual | 0.95683 | 0.96284 |
| RGB residual | 0.95501 | 0.96111 |

Original synthetic IoU requirement is 0.97469, alongside the unchanged missed/FP/
empty/clear safeguards. The label-informed calculation keeps each raw mask on
known positive cases and outputs empty on known clear cases. It is diagnostic,
uses labels, and is never deployable. It is not a universal mathematical bound on
all conceivable gates; it establishes that correcting presence labels alone does
not make these raw masks satisfy the requirements.

The matched comparisons do not support anatomical placement, border weighting or
RGB input as improvements over their respective controls. Recall gains repeatedly
cost visible-region specificity. Current frozen-gate errors and glare transfer
remain unresolved. Later residual runs inherit adapted parents, so do not attribute
their cross-experiment gains entirely to architecture. Repeated development-set
inspection is not independent final-test evidence.

## Alternatives review before more code

Teacher-output preservation is a recognized approach: [Hinton et al., knowledge
distillation](https://arxiv.org/abs/1503.02531) and [Li and Hoiem, Learning without
Forgetting](https://arxiv.org/abs/1606.09282) motivate retaining a reference model's
behavior during transfer. Those papers do not prove benefit for this detector.
Crucially, this repository already implements Bernoulli-KL synthetic replay in
`detector_replay.py`, and contains earlier returned consistency experiments.
Therefore distillation is not a new experiment to launch blindly.

Next: reconcile those earlier original-segmenter/consistency results, checkpoint
ancestry and evaluation versions with this six-candidate table. Identify whether
failure is lack of real-mask learning, retention tradeoff, or label-version mismatch
before proposing a distinct teacher/representation experiment. Do not repeat a
previously failed recipe under a new name. Glare-data coverage must also be handled
explicitly; small ambiguous highlight proposals remain disabled pending scope.

Goal still requires both original real/synthetic safeguards and reviewed end-to-end
completion improvement. This report is a decision checkpoint, not goal completion.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM experiments: `~/forensic-dgp/expanded_feature_bundle/outputs/`.
