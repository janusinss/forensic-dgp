# Segmentation initialization review — 30 September 2026

Decision: investigate face-specific occlusion pretraining and realistic occluder
data before another optimizer experiment. No candidate is yet verified as a
drop-in detector or as better on this project's outputs. No new training launched.

## What has already failed here

Original segmenter: mixed replay, teacher consistency, gradient projection and
actual-update retention trials have not met all real/synthetic safeguards.
Frozen SAM2 features: fixed/anatomical heads, border emphasis and residual heads
also failed retention. See `DETECTOR_ARCHITECTURE_DECISION.md`,
`COVERAGE_HISTORY_RECONCILIATION.md`, `RETENTION_EPOCH1_COMPARISON.md`.
These results do not establish that all pretrained encoders or data strategies
fail. They rule out treating another identical head/optimizer run as a new idea.

## Primary-source candidates and target compatibility

| Candidate | Verified source evidence | Project implication |
| --- | --- | --- |
| FaceOcc / FaceExtraction | Authors publish dataset and pretrained model links; evaluator instantiates a ResNet18 U-Net | Face-specific initialization is worth checking, but visible-face segmentation is not our occlusion target |
| NatOcc / RandOcc | Authors release naturalistic/random occluder synthesis and RealOcc evaluation data | A materially different data source from simple synthetic coverings; audit labels and overlap before use |
| S3POT | Authors publish contrast-driven reference generation, feature enhancement and prompt selection code | Distinct from our frozen SAM2 head; operational reference/adapter dependencies need verification |
| SegFormer | Official repository offers general semantic-segmentation code and pretrained models | General initialization option, not a ready-made face-occlusion detector |

FaceOcc authors provide a model/dataset license statement and download links in
their [official repository](https://github.com/face3d0725/FaceExtraction).
Their [evaluation code](https://raw.githubusercontent.com/face3d0725/FaceExtraction/main/evaluation_cofw.py)
uses RGB ToTensor, a single-logit ResNet18 U-Net and logit threshold0. Training
uses an ImageNet-initialized encoder in the
[published trainer](https://raw.githubusercontent.com/face3d0725/FaceExtraction/main/train.py).
The paper/repository task is extracting visible face, so inverse output includes
non-face background; it cannot silently replace the current missing-region mask.
Their [dataset loader](https://raw.githubusercontent.com/face3d0725/FaceExtraction/main/Dataset/dataset.py)
references CelebA-HQ, FFHQ and internet occluders. External pretraining overlap
with our FFHQ benchmark is therefore unresolved, not certified absent.

[NatOcc/RandOcc repository](https://github.com/kennyvoo/face-occlusion-generation)
provides synthesis code, occluder assets/splits and RealOcc links. The
[CVPRW paper](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Voo_Delving_Into_High-Quality_Synthetic_Face_Occlusion_Segmentation_Datasets_CVPRW_2022_paper.pdf)
includes transparent glasses in its occlusion definition, unlike our clear-glass
preservation policy. Imported masks must not become ground truth without a label
mapping audit, particularly strong reflections versus transparent lenses.

[S3POT paper](https://arxiv.org/abs/2602.00635) proposes generated-reference contrast
to select spatial prompts. The
[author repository](https://github.com/Bh-Johnny/S3SPOT) requires reference faces,
parsing masks and a trained adapter. Its README offers general and per-image
training; a usable pretrained adapter download was not established in this review.
No validation/test-image fitting is permitted in our fixed evaluation. Generated
references remain hypotheses, not evidence of a person's true hidden appearance.

[Official SegFormer](https://github.com/NVlabs/SegFormer) offers general-purpose
weights. Assigning a random binary output head would require training and would
not constitute a pretrained occlusion benchmark. Its existence alone does not
justify another architecture sweep.

## Concrete next work

1. Inspect FaceExtraction's pinned source, checkpoint availability/state layout,
   dependencies and exact label construction. Record revision and asset hashes.
2. Establish whether occluder annotations/assets can supply a direct covered-region
   target without inferring background as occlusion. Audit source overlap and
   clear-glass/glare semantics against our current data; keep splits unchanged.
3. If pretrained face-extraction inference is executable, inspect a fixed small
   training-only sample as a compatibility diagnostic, not a replacement score.
   Never use known validation masks to define its face support at inference.
4. Only after target compatibility is resolved, specify one direct-occlusion
   adaptation experiment with fixed controls and unchanged final safeguards.
   Actual fitting remains VM-only. Do not auto-download all external datasets
   or run a broad architecture/weight sweep.

This sequence prioritizes usable supervision and target semantics before spending
VM time. It does not abandon automatic single-image detection or the required
end-to-end restoration/completion gains. Existing generator/application unchanged.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing VM root: `~/forensic-dgp/coverage_vm_bundle/`.
This document is a research decision; no new VM training package exists yet.
