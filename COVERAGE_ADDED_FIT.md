# Added-example fit diagnostic — 30 September 2026

Inference only; zero optimizer updates. Five added **training** examples were
evaluated using pinned parent, control-final and extended-final checkpoints.
Manifest/checkpoint hashes verified. Exposure counts come from the actual frozen
extended-arm schedule. These are not held-out quality estimates.

| Example | Exposures | Parent IoU | Control IoU | Extended IoU |
| --- | ---: | ---: | ---: | ---: |
| Opaque eyewear13 | 9 | 0 | 0 | 0.000630 |
| Opaque eyewear49 | 7 | 0 | 0 | 0.000174 |
| Profile respirator | 10 | 0.09143 | 0.13937 | 0.18969 |
| Hand over mask | 10 | 0.00114 | 0.10236 | 0.14512 |

Transparent-glasses control46 received16 exposures; every model predicts zero
occluded pixels. The shared recount helper reports IoU0 when both masks are empty;
that convention must **not** be read as a control failure.

Extended target-region mean probabilities for the two opaque examples are only
0.00818 and0.01432 at threshold0.5, versus approximately0.0000084/0.0000012 in the
control. The model changed, but it did not learn complete lens coverage. Lowering
thresholds on these images is not a justified evaluation-policy change.

The five-row input/target/parent/control/extended preview was inspected. The
second eyewear prediction fires mostly above the lenses; profile and hand masks
remain fragmented. This establishes poor fit of the newly added examples under
this210-update mixed recipe. It does not isolate optimizer budget, feature
limitations or replay conflict, and does not prove that more labels cannot help.

Evidence: `outputs/coverage_added_fit/results.json`, `preview.png`, binary masks.
Reproducer: `scripts/diagnose_coverage_added_fit.py` (CPU inference only).
No application or generator checkpoint changed; no new VM run launched.

Next: quantify fit/exposure for all73 real training images, separating old mask,
new mask, opaque eyewear, glare and clear-control cases. Use this training-only
diagnostic to distinguish broad underfitting from a category-specific failure
before defining another VM intervention. Preserve the original selection gates;
do not repeat the failed recipe unchanged or claim completion gains.

Local: `C:\xampp\htdocs\YEAR 4\Testing\outputs\coverage_added_fit\`.
Parent VM experiment: `~/forensic-dgp/coverage_vm_bundle/outputs/coverage_training_vm/`.
