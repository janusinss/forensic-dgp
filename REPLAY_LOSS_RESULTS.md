# Frozen-checkpoint loss audit — 30 September 2026

Decision: do not repeat ordinary or projected training unchanged. Neither final
checkpoint qualifies for promotion. The mixed objective improves substantially
while supervised replay loss and binary mask quality regress.

## Verified measurement

Inference only, zero optimizer updates. All three checkpoints completed evaluation
on 73 real training inputs and the 638 unique cached replay inputs. Each result is
weighted by its actual exposure in the frozen 210-batch extended schedule: 840
real and 840 replay slots, each evenly divided between covered and clear cases.
Checkpoint/cache/protocol hashes were checked during inference; the independent
summary verifies the real manifest hash, all 711 indices, exposure counts and
reported aggregates. Each inference batch checks supervised decomposition against
the existing training loss. The completed JSON contains all three arms; the
inference process has exited.

| Exposure-weighted loss | Parent | Ordinary final | Projected final |
| --- | ---: | ---: | ---: |
| Real supervised | 2.407467 | 0.850346 | 0.859522 |
| Replay supervised | 0.037972 | 0.077584 | 0.068594 |
| Replay teacher KL | 0 | 0.008552 | 0.008164 |
| Mixed objective | 1.222720 | 0.472517 | 0.472222 |

Mixed objective = 0.5 real supervised + 0.5 replay supervised + replay teacher
KL. Supervised = BCE + soft Dice + 0.25 hard-visible penalty. These are losses
at three frozen states, not the losses accumulated along an optimization path.
Ordinary training differentiates this objective; projected training modifies its
real gradient, so it does not simply follow the objective's unmodified gradient.

| Supervised loss by input class | Parent | Ordinary final | Projected final |
| --- | ---: | ---: | ---: |
| Real covered | 4.167891 | 1.251747 | 1.273555 |
| Real clear | 0.647043 | 0.448946 | 0.445488 |
| Replay covered | 0.060234 | 0.079165 | 0.075751 |
| Replay clear | 0.015711 | 0.076002 | 0.061437 |

The replay regression appears in the supervised objective itself, not solely in
a mismatch between soft loss and thresholded IoU. Both covered and clear replay
groups worsen. Clear replay soft Dice loss rises from 0.011220 to 0.068919 ordinary
and 0.055494 projected; this is consistent with the previously measured clear
false-mask counts of 2/262, 11/262 and 9/262. Soft Dice on empty targets is sensitive
to total predicted foreground probability; this observation alone does not justify
removing Dice or tuning a threshold.

The large real-loss reduction outweighs replay deterioration in the aggregate.
Projection slightly improves retention compared with ordinary training but remains
below the parent on both replay loss and replay IoU. Lower total loss cannot be
used as evidence that either checkpoint is ready for the application.

## Next experiment specification to prepare

Prepare one VM-only retention-constrained pilot, rather than another loss-weight
sweep. The hypothesis is that checking actual post-update replay behavior can
prevent the tradeoff accepted by the aggregate objective. This is a proposed
experiment, not an established improvement or a ready-to-run training package.

1. Fix training-only covered and clear replay reference losses at the parent;
   retain separate limits so one subgroup cannot compensate for the other.
2. Specify a bounded update acceptance/backtracking rule before execution. Check
   actual candidate parameters, restore optimizer state as well as parameters on
   rejection, and stop/report if no useful update is feasible. Do not constrain
   teacher KL to its zero parent value, which would demand unchanged predictions.
3. Keep initialization, real data, replay membership and sampling schedule fixed;
   use the verified ordinary arm as control only if all relevant settings match.
4. Preserve the original real/synthetic validation gates and report every attempted
   and accepted update. Training-loss constraints do not guarantee validation
   retention. Completion preview review remains necessary before promotion.

Before packaging, define the exact replay check scope, numerical tolerance,
maximum retries and compute budget, then test acceptance/rollback on tiny tensors
locally. No local model fitting. If the constraint prevents meaningful real-mask
learning, report that result instead of silently relaxing the limits.

## Reproduction and locations

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM experiment root: `~/forensic-dgp/coverage_vm_bundle/`.
Checkpoints: `outputs/projection_training_vm/{ordinary,projected}/epoch_10.pth`
on VM; locally under `outputs/downloaded_projection/` with the same suffix.

- `scripts/audit_replay_losses.py`: inference audit, refuses an existing output
  directory; do not rerun over the completed evidence.
- `scripts/summarize_replay_losses.py`: repeatable, model-free verification.
- `outputs/replay_loss_audit/results.json`: full evidence, SHA256
  `4cd248812e68f3a3dfe00f513bc040c4fbd2f1547f0cd26298ec5b990c675934`.
- `outputs/replay_loss_audit/summary.json`: verified covered/clear summaries.

Verification command from the local root:

```powershell
venv/Scripts/python.exe scripts/summarize_replay_losses.py
```

No baseline, application, generator, labels, thresholds or selection gates changed.
No new completion output was generated in this audit. The overall goal is still
open: it requires a qualifying detector and reviewed end-to-end completion gains.
