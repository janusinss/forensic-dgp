# Native CCTV source extension and matched comparison — 5 October 2026

**The new ChokePoint development comparison is acquired, independently audited
and visually reviewed. The retained trained DGP still softens visible features
without a convincing structural clarity gain over resizing. CodeFormer gives
clearer plausible estimates, but remains a declared pretrained comparison
baseline. No checkpoint, Auto threshold or application change is adopted.**

These are unpaired native CCTV observations, with no aligned clean reference,
PSNR/SSIM, identity accuracy, ethnicity inference or Zamboanga performance claim.
Source label separation is established within the frozen release cohort;
cross-source and historical/pretrained person overlap remain unknown.

## Asian-source priority and the QMUL extension

The existing QMUL-SurvFace acquisition and original 24 development/32 reserved
cohort remain intact. The [primary paper's Table 4](https://arxiv.org/html/1804.09691v6)
lists source datasets from China and Japan among other countries; the acquired
release does not map each image to a source country. Numerical person IDs and
appearance cannot supply those missing labels.

A separate input-only extension selects larger native crops from the release's
training folder, excluding all 56 original selected person labels and selected
byte hashes. Before viewing pixels, the protocol fixes two native minimum-side
bins (64–95 and at least 96), up to eight cases per bin, deterministic hashed
ordering, a 16,384-candidate scan and a 120-second worker cap. The scan reads
16,132 headers in 7.68 seconds and finds only three eligible cases: two in the
first bin and one in the second. The six/seven-case shortfalls are retained;
this bounded scan does not establish that no larger faces exist elsewhere.

All three input-only cases are out of scope: strong downward framing, strong
side view, and insufficient recognizable face geometry. No model runs or
training admission follow that review. An independent 8.19-second auditor
replays the header order, exclusions, selection, native bytes, preparation and
sheet cells. The original 32 reserved crops remain unviewed.

Evidence: `outputs/cctv_native_structure_extension_v1/`;
`scripts/prepare_cctv_native_structure_extension_v1.py` and
`scripts/verify_cctv_native_structure_extension_v1.py`.

The [author's dataset links](https://rpwang.net/links.htm) were also checked:
the linked COX page `https://vipl.ict.ac.cn/view_database.php?id=3` returned 404
during this session. This checks one link, not the availability of all Asian
capture sources. No institutional request, agreement or message was sent.

## ChokePoint provenance, terms and original resolution

The [official ChokePoint release page](https://arma.sourceforge.net/chokepoint/)
links to the [Zenodo record](https://zenodo.org/records/815657). This is a separate
public native surveillance source, not an Asian-population or Philippine proxy.
NICTA's Australian publication/funding context does not establish the acquired
sequence's capture country or any person's ethnicity. Capture country/site are
not established by the acquired metadata and remain unspecified here.

Original camera frames are 800×600. The publisher's separately distributed
face crops are normalized to 96×96; **that normalized crop archive was not used**.
The acquired original P1E_S1 camera archives contain 6,876 JPEG frames, 2,292
per camera. The [release README](https://zenodo.org/records/815657/files/README.txt)
identifies AXIS P1343 capture, an indoor P1 sequence and camera C1 as most frontal.
Manual eye coordinates and recognition person labels are metadata, not clean
restoration targets. Temporal views have different pose/expression and are not
aligned clean/degraded pairs.

Use is limited to noncommercial research under the publisher's original notice.
Retain the notice and acknowledge NICTA and Wong et al., *Patch-based Probabilistic
Image Quality Assessment for Face Selection and Improved Video-based Face Recognition*, CVPR Workshops 2011,
[DOI 10.1109/CVPRW.2011.5981881](https://doi.org/10.1109/CVPRW.2011.5981881).
`LICENSE_SOURCE.html` is retained with acquisition, crops and model outputs;
derivative notices explicitly identify modified research crops/estimates.

The three publisher-linked files are downloaded serially and verified against
the publisher's MD5 and size. Local SHA256 fingerprints are also saved.
Acquisition totals 385,773,166 bytes and takes 450.97 seconds against the
600-second worker cap. No archive code is executed.

| Acquired file | Bytes | Observed SHA256 |
| --- | ---: | --- |
| README.txt | 1,446 | `5b70e99c9f3ad84c67882d1d19482977e2fcf01a2c4e3a82ac7d123831ee54a3` |
| groundtruth.tar.xz | 456,396 | `2ba86bf1ebd3dbe14d170a8fede5d0911ab383feae1ab30ca2e779ae7671dc1d` |
| P1E_S1.tar.xz | 385,315,324 | `3e310249afa86309175a13a04eb3e5cbea60f38e4018f9e4b8bd3fd80e39fcba` |

Raw files: `dataset/cctv_chokepoint_raw_v1/`.
Acquisition receipt: `outputs/cctv_chokepoint_acquisition_v1/acquisition.json`.

## Preserved source-reader failure and independent correction

The first source audit rejects an unexpected nonimage camera-container member
before RGB decoding or inference. Its frozen runner, request, nested copies and
failure remain in `outputs/cctv_chokepoint_release_audit_v1/`.

A separate bounded inventory identifies only three extra `.directory` files:
48/50-byte Dolphin desktop metadata. The separately versioned R1 auditor allows
only those three exact member paths, lengths and hashes. It never executes them,
and does not relax image, annotation, identity or restoration criteria. Original
V1 source is unchanged. Full annotation namespaces and declared XML root names
are retained separately rather than silently rewriting publisher metadata.

R1 independently verifies the source files, three nested archives, all 6,876
native JPEG headers, 72 XML annotation sequences and the selected sequence's
25 person labels in 43.43 seconds. All selected annotated frame names exist.
RGB pixels are not decoded at this release-audit stage. Receipt:
`outputs/cctv_chokepoint_release_audit_v1_r1/verification.json`.

## Frozen source-specific development and reserved identities

The native gallery fixes C1, a hashed person-label order, 12 development identities
and 13 reserved identities before decoding crops or seeing model outputs. Two
frames per person are selected at the 0.33/0.67 positions of the ordered valid
two-eye trajectory: 24 development cases and 26 reserved metadata-only cases.
The two label sets have zero overlap. This custom restoration split does not
reproduce the publisher's G1/G2 recognition protocol, which shares identities.

Native ROI geometry uses the eye midpoint and eye distance: horizontal extent
±2 eye distances, top −1.5 and bottom +3.3, with floor/ceil and camera bounds.
It is axis aligned and unrotated. All 24 development ROIs are untruncated and
range from 93×112 to 189×227. Native crops are center padded with RGB 128, then
bilinearly resized to 256×256; nearest-neighbor support masks declare padding.
No denoising, contrast adjustment, artificial degradation or facial alignment
is applied. Generation takes 13.47 seconds within the 90-second worker cap.

All six input-only sheets are actually viewed at their original 520×612 pixels
before model inference. All 24 cases have usable rough frontal/mild-turn facial
geometry. Dim/coarse/motion-soft details remain uncertain; glasses and
non-obstructing hair remain part of visible appearance. No completion is requested.
The 13 reserved identities have no decoded/cropped/rendered face files.

A separate 14.58-second auditor rederives all 50 metadata selections and the
label split, reproduces the 24 native camera ROIs, prepared inputs/support and
every input sheet cell. Source person overlap with QMUL, old photographic
training or pretrained datasets is unknown; dataset-name differences cannot
prove person non-overlap. ChokePoint is reported separately from QMUL.

Evidence: `outputs/cctv_chokepoint_native_development_v1/selection_plan.json`,
`frozen_subset.json`, `input_review.json`, `independent_geometry_audit.json`.
Preparation/audit runners: `scripts/prepare_chokepoint_native_gallery_v1.py`,
`scripts/verify_chokepoint_native_gallery_v1.py`.

## Identical-input frozen restoration comparison

The comparison protocol is frozen after input-only review and before inference.
Every model receives the same prepared 256×256 RGB image and canonical NumPy
float32/255 tensor. It includes five arms:

| Arm | Role and processing |
| --- | --- |
| Resize | Prepared native crop, no model |
| Phase 3 DGP | Original checkpoint; retained evaluation statistics |
| Retained trained identity-v2 DGP | Current own DGP checkpoint; retained evaluation statistics |
| CodeFormer w1 | Declared pretrained restoration baseline; official weights, fidelity 1, ADAIN enabled, internal 512 then bilinear 256 |
| Current DGP Auto | Exact cached resize/DGP alias from the unchanged input-only 24/8 policy |

Each model produces 24 outputs: 72 CPU forwards in 177.31 seconds against a
300-second worker cap. No optimizer, backward pass or parameter/buffer update
occurs; all three reported model state hashes remain unchanged. Seventy-two
untouched float32 raw arrays are separate from 120 PNGs. PNG delivery floors
raw×255 and restores only declared padding; no display sharpening or colour
enhancement is used. Auto selects DGP in 11 cases and resize in 13; this is an
input quality suggestion, not validated automatic face-usability detection.

CodeFormer receives the common unrotated crops rather than its complete
recommended detection/FFHQ alignment pipeline. This comparison does not establish
performance of that full pipeline. The baseline's sharper results under this
declared common-input path remain visible and are not discounted by that limit.

Independent saved-output audit takes 3.05 seconds: 102 source/226 artifact
bindings, 72 raw/PNG compositions, 24 exact resize controls, 24 exact Auto aliases,
24 recomputed input-policy choices and all 120 original sheet cells. This audit
does not independently replay model states or neural inference; it checks the
reported state parity and reconstructs saved output arithmetic.

All six output sheets are actually reviewed at their original 1340×1192 pixels.
Phase 3 adds conspicuous blur. The retained own DGP is less blurred than Phase 3
and retains rough face shape, but smooths visible glasses, eyes, brows and mouths
and changes tone compared with resizing. No convincing structural clarity gain
over resizing is established in these 24 development cases. Auto sometimes
selects that softer output on inputs already containing usable detail.

CodeFormer generally produces clearer, more coherent plausible features and
visible glasses. Fine eye/gaze, skin and hair detail remain estimates; localized
lens highlights and downcast eyes need particular caution. No clean native
reference establishes exact appearance. Completion and automatic covering-family
acceptance are not tested by this uncovered/clear-glasses native comparison.

Evidence: `outputs/cctv_chokepoint_native_comparison_v1/plan.json`, `results.json`,
`saved_output_audit.json`, `visual_review.json`, raw arrays and six comparison
sheets. Runners: `scripts/run_chokepoint_native_comparison_v1.py` and
`scripts/verify_chokepoint_native_comparison_v1.py`.

## Milestone decision and remaining work

Keep all checkpoints, protocols, failures, splits and quality gates. Do not
refit Auto on these exposed outputs or promote a pretrained baseline as our
trained DGP. A new architecture/training recipe requires separate justification,
explicit learned contribution and a finite manual L4 package; no new pilot
is prepared or launched by this milestone.

The two separate processing hypotheses are closed as negative in
`CCTV_DGP_PROCESSING_DIAGNOSTICS_V3.md`: universal six-pixel completion context
suppression exposes visible patch boundaries, and per-image DGP statistics
fail two predeclared training-control SSIM guards. Neither is adopted.

The app remains unchanged from its earlier verified research integration;
34 regressions and bundled inline Playwright are historical checks, not reruns
claimed here. Useful own-DGP native output, automatic and assisted covering-family
acceptance and independent final review remain required. Both original QMUL32
and ChokePoint13 reserved identity sets remain unviewed. The full goal is active.
VM work follows the user's verified-transfer-files/pasteable-commands preference;
all actual training remains on the existing L4 VM.
