# Corrected-label comparison — 28 September 2026

Created `dataset/detector_expanded_review_v2/` alongside the unchanged original.
All 100 images, split membership, training masks and test masks remain unchanged.
Only validation `new_covered_03.png` has a refined mask. The enlarged overlay
corrects our earlier description: the old label already excluded most of the cheek
opening; the defects were approximate boundaries, including an upper-edge overshoot.
The refinement removes 1,313 previously labeled pixels and adds 1,412 mask pixels.
It is an assistant polygon annotation, not independently adjudicated ground truth.

Manifest SHA-256:
`e0a28146f1f271b94e26351cc093446ca3ea6e6a843a4c53155aa81382e38180`.
The manifest records its parent hash and previous mask hash. The known mannequin,
`new_covered_40.png`, is flagged and reported separately, not deleted.
Other images are labeled `not_flagged_as_mannequin`, not certified as real people.

## Same predictions, original and corrected validation labels

CPU inference on the same 25 validation images; IoU aggregates pixels. No training,
test-set tuning, checkpoint selection change or synthetic benchmark modification.

| Checkpoint | Original labels, 25 | V2 labels, 25 | V2 excluding known mannequin, 24 |
|---|---:|---:|---:|
| Initial completion epoch 2 detector | 0.06040 | 0.06078 | 0.06444 |
| Real-only selected epoch 9 | 0.33699 | 0.33786 | 0.34771 |
| Replay epoch 10 | 0.32002 | 0.32044 | 0.33835 |
| Consistency epoch 10 | 0.31524 | 0.31568 | 0.33475 |

Excluding the known mannequin, missed covered pixels remain 61.64% for real-only,
62.86% for replay and 63.25% for consistency. Correcting this label does not reverse
the ranking or explain the widespread incomplete predictions. Real-only still has
the previously measured synthetic regression; it is not promoted by this table.
The separate mannequin stratum is a single example, not a generalization estimate.

## Artifacts and next action

Windows root: `c:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
Relative artifact paths map to those roots, but local results are not automatically
present on the VM:

- `outputs/detector_label_v2_review/evaluation.json`: per-image counts, both label
  versions, separate strata and checkpoint hashes.
- `outputs/detector_label_v2_review/correction.png` and `correction.json`: visual
  before/after, exact polygon coordinates and label change counts.
- `outputs/prepare_detector_labels_v2.py` and `evaluate_detector_labels_v2.py`:
  local audit/reproduction scripts. Preparation refuses an existing destination.
- `outputs/detector-reviewed-v2.tar.gz`: complete versioned dataset package.

Do not repeat GPU training yet. Next run a small, training-only overfit diagnostic
on representative dark, pleated and patterned mask examples. This will test whether
the current model/objective can learn complete masks on seen examples before
spending time on further generalization experiments. Keep validation/test data
out of that diagnostic, and do not treat overfit results as deployable weights.
