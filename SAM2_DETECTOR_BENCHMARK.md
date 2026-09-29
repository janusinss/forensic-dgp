# SAM2 boundary-refinement benchmark — 29 September 2026

Experimental adapter `detector_refinement.py`; not connected to the application.
Its only inference inputs are RGB pixels and detector probabilities. Source masks
are scoring inputs only. No ground-truth proposal selection, prompts or landmarks.

## Frozen protocol

Use consistency epoch10 detector at threshold0.5. Find its largest connected
component, use the deepest interior pixel as a positive point and a component
bounding box expanded10% per dimension. Empty detections abstain. Ask SAM2 for
multiple masks and choose its highest predicted-quality score. Fall back to raw
detector output if the proposal is empty, exceeds85% image coverage or excludes
the positive seed. Invalid outputs fail explicitly. This generic-object model does
not know whether the prompt identifies a face covering: hair/background prompts
can produce confidently wrong masks. The largest-component rule can miss separate
occluders; the bounding box can constrain fragmented masks to only part of a body.

Official source [facebookresearch/sam2](https://github.com/facebookresearch/sam2),
revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`; official SAM2.1 tiny weights from
`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`.
Checkpoint SHA256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
CPU FP32, four torch threads, `apply_postprocessing=False`, no compiled CUDA
extension or hole/sprinkle filtering. SAM2 and its configuration dependencies live
under ignored outputs directories; project requirements and installed PyTorch were
not changed. Windows sandbox required escalated reads of pip-created dependency
files. Runtime is approximately2–3seconds per prompted crop on this host.

Five new adapter tests passed;20 focused adapter/detector/replay tests passed.
Four training examples checked mechanics before the unchanged real-validation run.
These are not independent quality evidence. Training probe results were mixed:
IoU worsened for the patterned example, improved for the respirator and pleated
mask; the negative case remained empty. No subsequent prompt tuning was performed.

## Corrected real validation (25 cases)

| Metric | Raw consistency detector | SAM2 refinement |
|---|---:|---:|
| IoU | 0.31568 | 0.43854 |
| Missed covered pixels | 65.31% | 52.60% |
| Visible-pixel false positives | 1.398% | 1.141% |
| Empty detections on covered images | 2/14 | 2/14 |
| Uncovered cases with false masks | 1/11 | 1/11 |

Excluding the one known mannequin, IoU is0.33475 raw versus0.48733 refined; the
mannequin remains in aggregate selection. Visual inspection found improved mask
bodies on several surgical/respirator cases, but wrong-object segmentation on
hair, cap, hand and background prompts, and unchanged missed detections.
These results pass the real side of the baseline comparison but do not establish
synthetic retention, deployment readiness or end-to-end completion improvement.

## Completed synthetic benchmark — rejected for promotion

The process exited successfully. Verified all400 unique expected IDs, matching
strata, saved masks and valid confusion counts. No execution failures. Raw metrics
reproduce the prior consistency detector result (IoU0.9570553556862881).

| Synthetic metric | Original baseline | Raw consistency | SAM2 refinement |
|---|---:|---:|---:|
| IoU | 0.97469 | 0.95706 | 0.74929 |
| Missed covered pixels | 1.648% | 3.327% | 17.243% |
| Visible-pixel false positives | 0.1333% | 0.1487% | 1.5362% |
| Empty covered cases | 1/320 | 0/320 | 0/320 |
| Uncovered cases with false masks | 0/80 | 1/80 | 1/80 |

Four of five synthetic safeguards fail. No deployment or generator training is
justified by this candidate. Real validation improvement remains a useful result,
but does not satisfy the combined selection criteria. Completion-output gain has
not been demonstrated; the candidate was rejected before that expensive stage.
Full metrics/groups are in `outputs/sam2_synthetic_validation/summary.json`.

Degraded irregular masks are the largest false-positive group (10.450% of visible
pixels). Inspected the four highest-FP cases in `largest_false_positives.jpg`,
explicitly selected for failure diagnosis:000399,000299,000379,000089. SAM2 selects
much of the face/person instead of just the covering. This is a separate failure
from the missing blur margin; expanding masks cannot solve both. No post-hoc
threshold, dilation, source-aware switch or oracle proposal selection was applied.

Next distinct investigation: evaluate feasibility of a learned occlusion-specific
head over frozen pretrained image features, using training data only to establish
fit before another frozen validation. Generic object segmentation is insufficient
to identify the required completion region. This is a hypothesis, not an approved
replacement or proven improvement. Do not repeat SAM prompt tuning on these same
validation examples or weaken retention gates.

### Interim visual diagnosis (not a selection result)

Inspected the fixed first source's ten variants in
`outputs/sam2_synthetic_validation/first_source_preview.jpg`: clean object masks
are close to the target, while degraded examples miss a band around the object.
`completion_data.py` explicitly dilates degraded geometry by radius8 at256px to
cover camera contamination. SAM2's object boundary therefore differs from the
required completion region, even when object segmentation looks reasonable.
This explains part of the observed drop; it does not explain away wrong-object
prompts or authorize changing targets/gates after seeing validation results.
No dilation, threshold, prompting or fallback policy was changed. The preview
script uses fixed case IDs000000–000009, not selected successes.

Local root `c:\xampp\htdocs\YEAR 4\Testing\`; VM equivalent `~/forensic-dgp/`.
Ignored local artifacts: `outputs/sam2_training_probe/`,
`outputs/sam2_real_validation/` (results,summary,preview),
`outputs/sam2_synthetic_validation/` (completed cases, masks, summary and previews),
`outputs/vendor_sam2/`, `outputs/sam2_dependencies/`,
`outputs/sam2.1_hiera_tiny.pt`. Research scripts are `outputs/run_sam2_*py`.
VM copies exist only after explicit transfer. No GPU job has been launched.
