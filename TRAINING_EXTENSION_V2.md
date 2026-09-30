# Mixed-covering training extension V2 — 30 September 2026

Local dataset: `C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_training_extension_v2\`.
Eventual VM destination: `~/forensic-dgp/dataset/detector_training_extension_v2/`.
This package is local only. No actual training occurred.

The original 100 V3 records remain exactly unchanged. Five training-only additions
now cover two opaque-eyewear examples, one transparent-glasses control, one
side-profile respirator, and one hand over a surgical mask.

| Split | Covered | Clear | Total |
| --- | ---: | ---: | ---: |
| Train | 47 | 26 | 73 |
| Validation | 15 | 10 | 25 |
| Test | 4 | 3 | 7 |

Manifest SHA256: `1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c`.
The actual detector manifest loader accepted all 105 records; copied file hashes
and original record preservation passed. `scripts/build_mixed_training_extension.py`
binds the prior extension, proposal and overlap audit hashes. It refuses overwrite.

## Mask annotation decisions

`outputs/real_expansion_proposals_v3/preview.png` contains the reviewed overlays.
The profile-mask lower contour was corrected. It and the hand/mask union were
accepted as approximate assistant pilot polygons. The patterned mask remains
excluded because its strap/fabric boundary is uncertain. No expert annotation
quality is claimed; source resolution and boundary precision remain limitations.

Native and cropped variants were screened against 4,200 frozen reference paths:
no exact/DCT<=6 flags. Nearest-reference distances were 16/14/16; nearest images
were visually inspected and show distinct scenes. This does not certify identity
separation or eliminate every alternate-crop risk. Source-pool screening also
excluded flagged within-pool duplicates before this batch was selected.

## Experiment decision still pending

Five extra training images are a small coverage intervention, not proof that the
retention failure is solved. No generator/application baseline was changed.
This extension adds no paired uncovered target and cannot establish accuracy of
the generated hidden face. Existing repeated development-validation limitations
still apply; the seven test images are not an untouched test set.

Next: define a fixed-budget original-versus-extended-data comparison, keeping
initial state, synthetic sources, loss, real/synthetic batch ratio and quality
gates fixed. Check that replay cannot sample these new source images as synthetic
unoccluded negatives. Validate that exclusion before packaging any VM recipe.
Do not tune gates or choose a checkpoint from training fit alone. Training-readiness
metadata in the manifest is documentary and not enforced by old runners.
