# Eyewear annotation progress — 30 September 2026

## Corrected proposals and overlap result

`outputs/eyewear_annotation_proposals_v2/` now contains corrected lens outlines.
Both opaque examples and the transparent control were inspected against native
sources and overlays and accepted as approximate assistant-reviewed pilot labels.
The first draft is preserved. Pixel areas are now 11,086 / 15,672 / 0.
All three remain training-disabled pending integration; the ambiguous tinted
example remains excluded. No user or independent expert approval is claimed.

`scripts/audit_eyewear_annotation_overlap.py` compared both native and resized
images with 4,200 frozen reference paths using exact bytes, decoded pixels and
DCT distance <=6. No flags. Closest distances: 12 / 14 / 16. The three nearest
reference images were visually inspected and are distinct scenes. This screening
does not prove identity-disjointness or rule out every alternate crop.
Reference manifest hashes and nearest matches are saved in `overlap_audit.json`;
the review manifest binds that audit hash. Binary shapes and source/image/mask
hashes were checked. No held-out membership changed.

Next: prepare a versioned extension manifest binding these accepted labels to
original training sources, then finish the medical-mask/mixed-covering proposals
before designing a data-coverage experiment. Keep old V3 manifests immutable.

## Initial native-source review

Four native Asian-face sources from the previously screened training-only queue
were inspected. This extends real annotation coverage beyond medical masks;
no inference scores or validation errors were used to select these four sources.

| Queue index | Native source | Result |
| --- | --- | --- |
| 13 | asian_face_07002.jpg | Opaque lenses; proposal needs close rim review |
| 49 | asian_face_08830.jpg | Opaque lenses; proposal has outer-edge undercoverage |
| 46 | asian_face_06721.jpg | Transparent lenses, visible eyes; empty pilot label reviewed |
| 84 | asian_face_04931.jpg | Tinted lenses with some eye detail; held without pixel label |

Sources are under `dataset/asian_faces/`. All remain disabled for training,
including the reviewed control. Reviewer is the assistant; these are approximate
pilot annotations, not independent expert ground truth. No tiny-glare assumption
was added. The transparent control preserves glasses frames and visible eyes.

`scripts/prepare_eyewear_annotation_proposals.py` verifies source hashes and
original training membership, then exports native-coordinate polygons, binary
masks and a three-row overlay. Full source images are resized to 256x256, with
anisotropic resize explicitly recorded; this is not face alignment. Crop/group
compatibility must be checked before merging into the reviewed training dataset.

Evidence: `outputs/eyewear_annotation_proposals_v1/manifest.json` and `preview.png`.
All three binary masks, source hashes and positive-pixel counts were verified;
overlays were visually inspected. Positive areas are 10,583 / 14,930 / 0 pixels.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/`. The new artifacts are local only. No optimizer was run.

Next: resolve opaque-lens contour errors, finalize compatible face crops, and
assemble a reviewed training-only extension with transparent controls and mixed
covering types. Preserve V3 held-out membership and all original quality gates.
Do not launch a new VM experiment on these incomplete proposals.
