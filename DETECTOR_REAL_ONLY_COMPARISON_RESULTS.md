# Real-only versus replay diagnostic — 29 September 2026

Completed the matched local real-only arm at lr 1e-4: four epochs, 21 updates per
epoch, original completion epoch-2 initialization, seed42, native256px, existing
BCE/Dice + .25 hard-visible penalty. Real image IDs and order are the exact real
subsequence of the earlier mixed batches. All68 real training images occur every
epoch. Generator tensors remain unchanged. No weights were saved or deployed.

Control is the already-completed lr1e-4 replay arm, not a rerun. Original checkpoint,
corrected-label manifest and complete starting metrics match exactly. The real-only
batch has four images versus eight mixed images. Since the objective averages over
the batch, removing replay also changes relative real-loss weighting. This tests
the practical removal of replay, not isolated gradient conflict or an equal-compute
comparison. Neither arm uses teacher consistency.

| Epoch4 metric | Mixed replay | Real-only |
|---|---:|---:|
| Real training IoU | 0.38737 | 0.33008 |
| Real validation IoU | 0.26627 | 0.23944 |
| Synthetic validation IoU | 0.85874 | 0.60377 |
| Real validation missed covered pixels | 72.36% | 75.69% |
| Synthetic completely missed masks | 7/320 | 22/320 |

Peak real-validation IoU was0.35362 at replay epoch2 and0.32253 at real-only epoch3.
Neither peak qualifies: all candidates fail unchanged synthetic retention gates.
The original synthetic baseline was0.97469. No test-set data entered training or
selection. The known mannequin is included in aggregate selection and separately
reported in results.json; excluding it does not rescue the conclusion.

Visual review: real-only still predicts mask edges/patches instead of full bodies,
misses dark patterned regions and the respirator example, and produces some cap/
background false positives. The viewed uncovered controls remain unmarked. This
does not demonstrate improved completion outputs.

## Decision

Keep replay in future experiments. At this tested rate/budget, removing replay
does not solve real training fit and causes substantially more forgetting. These
results do not establish an optimal learning rate or exclude effects of longer
training, but provide no justification for another unchanged VM run.

Next diagnostic should inspect per-image and per-region loss/gradient contributions
on training data, especially whether the hard-visible penalty and easy uncovered
examples dominate covered-mask learning. Establish that evidence before proposing
another loss change. Keep validation gates fixed and do not promote any diagnostic
model. This proposed loss audit has not yet been run.

Artifacts are local at `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_real_only_comparison\`:
`results.json` (both arms, hashes, all curves), `validation_masks.jpg`.
Reproduction: `outputs/run_detector_real_only_comparison.py`. The corresponding VM
root is `~/forensic-dgp/`; artifacts appear there only after explicit transfer.
