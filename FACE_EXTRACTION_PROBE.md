# FaceExtraction inference compatibility — 30 September 2026

Strict checkpoint loading and CPU forward inference succeeded with
segmentation-models-pytorch0.5.0. All eight outputs have shape1x1x256x256 and are
finite. Zero optimizer updates. Original checkpoint SHA256 remains
`01d3c3939c28e47a45acb9a5ea8f8ee460e5ecf046c0caa8404e05915e10901b`.
Only the known `module.` prefix was removed; no missing keys were ignored.
Encoder initialization used no automatic weight download because the complete
author checkpoint supplies the weights. RGB values are in[0,1], without added
ImageNet normalization, following the author's evaluation preprocessing.

The sample is fixed by manifest order: first six covered training records and
first two uncovered training records. No validation/test images or prediction-
based sample selection. Manifest and image/mask hashes were checked.

Visual inspection of `outputs/face_extraction_probe/preview.png` shows visible
upper-face regions separated from black, patterned and surgical coverings in
these examples. Clear faces retain most facial regions. Face contours, background,
hair and glasses regions are also excluded; therefore the complement remains
unsuitable as a direct occlusion mask. We have no corresponding visible-face
ground truth for this sample, so no visible-face accuracy score is claimed.
Eight selected-by-order images cannot establish demographic/generalization gains.

## Dependencies and reproducibility

Pinned packages installed only under `outputs/face_extraction_dependencies/`:
segmentation-models-pytorch0.5.0, timm1.0.15, huggingface-hub0.29.3,
safetensors0.5.3 and PyYAML6.0.2. Existing Torch2.13.0+cpu/torchvision are reused;
the project's venv package set was not upgraded. Runtime file access needed
approved execution outside the sandbox after its access restrictions prevented
reading installed files. No downloaded repository training script was executed.

`scripts/probe_face_extraction.py` is inference-only and refuses existing output.
`outputs/face_extraction_probe/results.json` records versions, source images and
checkpoint hash. Probability arrays and visible-face masks are saved separately.
These files are diagnostic artifacts; no production inference path uses them.

## Next implementation

Prepare a separate direct-occlusion segmentation adapter using this strictly
loadable encoder/decoder initialization and a new binary occlusion head. The
visible-face head must remain separately available for provenance/diagnostics;
do not relabel its output or invert it as an application fix. Validate output
shape and explicit target semantics with synthetic tensors before a VM pilot.

Specify a matched initialization control and bounded real+replay training recipe
before executing actual training on the VM. Existing73 real labels and frozen
replay are the initial controlled data; introducing external occluder data would
be a separate intervention requiring overlap and alpha-label review. Keep all
original real/synthetic gates and end-to-end completion review. This checkpoint's
FFHQ pretraining overlap remains unresolved, so report that limitation explicitly.
No benefit from direct-occlusion adaptation has yet been demonstrated.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing VM root: `~/forensic-dgp/coverage_vm_bundle/`.
Assets have not been uploaded to VM; no training command/package ready yet.
