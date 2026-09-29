# Training-only probability audit — 29 September 2026

Compared three saved detectors on all68 real training images at 256px. No model
updates, validation/test threshold sweep or application configuration changes.
Swept99 thresholds from0.01 to0.99. This grid is a diagnostic, not an exhaustive
precision-recall curve or proof that every possible threshold fails.

| Detector | Best training IoU in grid | Threshold | Visible-pixel FP at that threshold |
|---|---:|---:|---:|
| Initial completion epoch2 | 0.10300 | 0.01 | 2.963% |
| No-penalty epoch4 | 0.33929 | 0.04 | 10.883% |
| Consistency epoch10 | 0.43505 | 0.09 | 5.901% |

Under the initial detector's threshold0.5 training constraints for visible FP,
uncovered cases with false masks and completely missed covered cases, no grid
threshold qualifies for the no-penalty model. Consistency's best constrained
training IoU is0.35682 at0.51, with covered recall0.39893. These are training
diagnostics, not replacements for real/synthetic validation gates. No threshold
was chosen for deployment.

Covered-pixel median probabilities are approximately1.51e-7 initially,0.139 in
no-penalty epoch4, and0.253 in consistency epoch10. Poor separation and low coverage
remain even on training images. This is evidence against scalar threshold tuning
as a sufficient fix; it is not a formal calibration test or a proof that the
architecture cannot learn a better representation.

Local evidence: `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_probability_audit\results.json`.
Script: `outputs/run_detector_probability_audit.py`. VM root equivalent is
`~/forensic-dgp/`, requiring explicit transfer for these ignored artifacts.

## Next distinct approach: pretrained boundary refinement benchmark

The [official SAM2 repository](https://github.com/facebookresearch/sam2) supplies
pretrained checkpoints and image prediction; its
[image predictor](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)
supports point and box prompts. This suggests testing whether generic pretrained
object boundaries repair fragmented detector masks without more scalar tuning.
It does not provide a face-occlusion classifier or guarantee correct mask material.

Predeclare a frozen protocol on training examples first: derive prompts only from
input pixels and the existing detector, never reference masks; preserve an empty
mask when the detector abstains; compare raw masks versus refined masks on the
same fixed real and synthetic validation with existing safeguards. Explicitly
count missed prompt generation and refinement failures. Do not select the best
SAM proposal using ground-truth IoU. Report manual prompts as a separate assisted
mode, never as automatic results. The target remains single-image automatic
restoration/completion with retained manual correction as a fallback.

SAM2 is not installed in the local project runtime and no weights were downloaded
in this audit. Use an isolated environment and pinned upstream revision/checkpoint
hash before benchmarking. The official documentation recommends WSL for Windows;
verify CPU feasibility before preparing a GPU-only recipe.

FaceOcc is another research lead, but its published task is visible-face
extraction: subtracting its output from an image would also mark background as
missing face. It is not a drop-in hole mask. See the
[authors' repository](https://github.com/face3d0725/FaceExtraction) and
[paper](https://arxiv.org/abs/2201.08425). Do not silently substitute that task for
occluder segmentation. These sources justify candidate investigation, not claimed
performance on our data.
