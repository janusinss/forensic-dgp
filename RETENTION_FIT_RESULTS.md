# Retention pilot training fit — 30 September 2026

Inference completed on all73 real training images under parent and retained
checkpoints, plus25 real validation images under the retained checkpoint.
Zero optimizer updates. Manifest and checkpoint hashes checked; image/mask hashes
validated by the existing manifest loader. Fixed threshold0.5, no tuning.

| Training metric, all73 images | Parent | Retained |
| --- | ---: | ---: |
| Aggregate mask IoU | 0.071212 | 0.066046 |
| Missed covered fraction | 0.920358 | 0.927646 |
| Visible false-positive fraction | 0.017457 | 0.014085 |
| Empty covered cases | 14/47 | 16/47 |
| Clear cases with false masks | 9/26 | 9/26 |

Unweighted per-image supervised loss decreases2.907424 ->2.777603. Covered-image
loss decreases4.146071 ->3.986258, but covered-only IoU decreases0.073359
->0.067707. Reduced loss does not establish improved mask coverage.

The62 distinct images appearing in accepted batches also regress in IoU:
0.061321 ->0.057914. These comprise36 covered and all26 clear training images.
The other11 covered images have no accepted-batch exposure in this short pilot;
their IoU also falls0.102599 ->0.091646. Group membership is retrospective exposure
analysis, not a new selection rule or an independent evaluation split. Counts
do not imply that every accepted gradient helped every image in its batch.

All25 final-checkpoint real validation predictions match returned VM PNGs
pixel-for-pixel under CPU inference. This resolves the real checkpoint-to-mask
reproduction limitation. It does not reproduce synthetic masks, historical trial
losses or optimizer states. Generator invariance was verified in the earlier audit.

## Decision and next step

The short constrained pilot does not merely fail to transfer a good training fit:
binary coverage is poor on the real training images themselves and regresses even
among accepted-batch images. Lower visible false positives accompany worse recall.
Do not promote, extend this run automatically, or relax validation gates.

Before concluding the constraint caused this behavior, compare the already saved
ordinary **epoch1** checkpoint on these identical73 training images. The current
pilot has21 scheduled batches but18 accepted updates; the earlier ordinary final
has210 updates and is not a matched-budget comparator. Use the archived epoch1
checkpoint and inference only; no new VM run is needed. Report this accepted-dose
difference explicitly. If the ordinary epoch1 also lacks useful coverage, this
pilot cannot establish a causal constraint failure from the final masks alone.

Evidence: `outputs/retention_fit/results.json` contains per-image metrics,
accepted-batch exposure and all25 reproduction differences (zero).
Script: `scripts/diagnose_retention_fit.py`; refuses an existing output directory.
Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM artifact source: `~/forensic-dgp/coverage_vm_bundle/outputs/retention_training_vm/`.
No application, generator, thresholds, labels or gates changed.
