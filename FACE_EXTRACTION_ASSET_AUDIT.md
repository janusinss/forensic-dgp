# FaceExtraction asset audit — 30 September 2026

Author source cloned read-only for inspection to `outputs/face_extraction_source/`.
Pinned commit: `e75d4a83a696bd7379128319244ef6e5e7885fc8`.
Repository: https://github.com/face3d0725/FaceExtraction

Published Google Drive checkpoint downloaded successfully through the author's
README link. File: `outputs/face_extraction_epoch16.ckpt`,57,430,037 bytes.
SHA256: `01d3c3939c28e47a45acb9a5ea8f8ee460e5ecf046c0caa8404e05915e10901b`.
Tensor-only `torch.load(..., weights_only=True)` succeeds; all values are tensors
and finite. Keys use `module.encoder`, `module.decoder`, `module.segmentation_head`.
First encoder convolution is64x3x7x7; head is1x16x3x3. These match the published
ResNet18 U-Net configuration but do not establish executable architecture
compatibility until a strict state load and forward check succeed.
Machine evidence: `outputs/face_extraction_checkpoint_audit.json`.

## Exact label semantics found in code

`Dataset/dataset.py` constructs synthetic visible-face targets as
`face_support * (1 - occluder_alpha)`. Its real branch likewise removes the RGBA
occluder alpha from a parsed face mask. `Dataset/utils.py` selects facial classes
1 through10, excluding class4 eyeglasses by default; the manually annotated real
branch includes class4 in support before applying alpha. Hair/background are
outside the visible-face target. Thus neither raw output nor its complement is
the project's desired covered-region mask.

The RGBA assets offer a possible direct occluder label source, but actual asset
content, alpha boundaries, licensing/provenance and clear-glass versus glare
semantics have not been inspected. No dataset imported or labels generated.
The source loader explicitly mixes FFHQ/CelebA-derived occluders; disjointness
from our FFHQ data cannot be claimed. Full training identities/splits are not
established merely by downloading the checkpoint.

## Integration constraints and next action

The local venv lacks `segmentation_models_pytorch` and `gdown`; no packages were
installed. The published evaluator uses RGB ToTensor and raw-logit threshold0,
with a ResNet18 U-Net. Do not add ImageNet normalization solely because its
encoder was ImageNet-initialized; follow the actual evaluation contract.

Next: prepare an isolated inference adapter with a compatible, pinned dependency
or audited architecture; strip only the known `module.` prefix, require strict
state loading, and run a fixed small training-only visible-face diagnostic.
Keep output labeled visible-face probability. This diagnostic tests compatibility
and useful features, not direct occlusion accuracy or readiness for application
use. A direct-occlusion head requires separate supervised adaptation on the VM;
do not treat an inverted visible-face mask as that adaptation.

No external repository code was executed. No optimizer/model training, application
change or new VM recipe. Source code license remains in the cloned repository.
Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing VM root: `~/forensic-dgp/coverage_vm_bundle/` (assets not uploaded there).
