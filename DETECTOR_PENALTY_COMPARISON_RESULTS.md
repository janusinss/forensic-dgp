# Hard-visible penalty ablation — 29 September 2026

Completed a new penalty0 arm against the verified penalty0.25 control from the
four-epoch lr1e-5 replay experiment. Same checkpoint, V2 data, original split,
200 replay sources, batch8, seed42, 21updates/epoch, four epochs and optimizer.
Every real training image appeared each epoch. No teacher consistency term.
Complete starting metrics and comparison metadata matched exactly. No test tuning.

| Epoch4 metric | Penalty0.25 control | Penalty0 |
|---|---:|---:|
| Real training IoU | 0.24240 | 0.29212 |
| Real validation IoU | 0.20196 | 0.28058 |
| Real validation missed covered pixels | 77.83% | 65.69% |
| Real visible-pixel false-positive rate | 1.377% | 3.146% |
| Synthetic validation IoU | 0.97244 | 0.96869 |
| Synthetic uncovered cases with false masks | 1/80 | 4/80 |

Initial real visible false-positive rate was1.808%, and synthetic IoU was0.97469.
All four new epochs failed both selection gates. Epoch1/2 synthetic IoU exceeded
baseline slightly, but false-positive criteria failed; an IoU-only decision would
have hidden that regression. No candidate was promoted.

Visual review confirms incomplete mask bodies plus increased hair, cap, skin and
background predictions. There is no demonstrated end-to-end completion gain.
The experiment supports the coverage/preservation tradeoff identified by the
gradient audit; it does not establish an optimal intermediate penalty or prove
that adjusting this scalar alone can solve the task.

## Integrity and artifacts

Four per-epoch diagnostic inference exports were saved, reloaded and checked for
unchanged generator tensors. They are marked `diagnostic_only` in metadata and
remain under outputs; the general inference loader does not enforce that custom
marker, so never configure them as application weights without explicit review.
Control weights from the earlier run were not saved; its verified metrics and
preview provide the matched comparison. No production code changed.

Local root `c:\xampp\htdocs\YEAR 4\Testing\`; VM counterpart `~/forensic-dgp/`.
Results currently local: `outputs/detector_penalty_comparison/results.json`,
`PROTOCOL.md`, `validation_masks.jpg`, and `epoch_1.pth` through `epoch_4.pth`.
Reproduction script: `outputs/run_detector_penalty_comparison.py`.

## Next bounded work

Do not launch another unchanged GPU run or choose an intermediate weight by
repeatedly inspecting this validation set. Next inspect training-only probability
separation/precision-recall for the saved baseline and diagnostic checkpoint:
determine whether covered and visible pixels are poorly separated, or whether a
threshold/calibration issue is contributing. No validation threshold sweep or
post-hoc relaxed selection gates. Use that evidence to choose between calibration
work and a materially different segmentation/data approach. Goal remains incomplete:
no detector meets both validation safeguards and no completion improvement is shown.
