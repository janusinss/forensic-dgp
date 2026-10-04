# Practical covering examples: source research and acquisition status

Updated 3 October 2026. Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
Intended VM repository counterpart after transfer: `~/forensic-dgp/`.
This records additional-source research and the selected Mendeley/COFW
extensions. Eight Mendeley and 42 COFW reviewed detector sources are admitted
to versioned training registries; complete archives are not admitted wholesale.
The camera diagnostic is complete; no new COFW training has run.

## Current COFW preparation — verified 3 October 2026

The complete official 503,327,162-byte color archive now passes publisher MD5
and ZIP checks; the earlier partial remains excluded history. All 1,345
author-training images are independently verified, including 44 grayscale
sources. No test RGB was decoded/viewed. Current official metadata declares
CC BY 4.0 and public access.
[Official record](https://data.caltech.edu/records/bc0bf-nc666)

Native source review and versioned covering proposals admit 42 training-only
examples: 31 covered and 11 clear, including hands, obstructing hair,
cloth/scarves, objects, eyewear/masks and clear-glasses controls. Source/geometry
and registry audits pass. Twelve raster-ambiguous native pixels are explicitly
unsupervised; old held-out support is unchanged. These are approximate assistant
detector annotations, not paired true uncovered faces.

Supported V3 has 133 training, 25 validation and seven previously inspected test
records; all 123 V2 records are preserved. Exact RGB checks against 290 declared
prior source/crop files find no match, without claiming full 80,000-source,
identity or pretraining separation. Report:
`C:\xampp\htdocs\YEAR 4\Testing\COFW_DATA_PREPARATION.md` ↔
`~/forensic-dgp/COFW_DATA_PREPARATION.md` after transfer. New files are local only.
Paired camera inputs are now verified: 133 native/133 degraded training views,
all previous 91 pairs exact, unchanged targets/support/geometry and no held-out
inputs. Evidence is `outputs/cofw_camera_pairs_v2/independent_verification.json`
under both roots after transfer. The distinct finite comparison is now packaged
and independently audited: `VARIED_COVERING_VM.md`, local bundle
`outputs/varied-covering-vm-bundle-v2.tar.gz`, intended VM workspace
`~/forensic-dgp/varied_covering_vm_bundle/`. It includes all165 registered raw
sources and compares existing versus varied data with matched fresh optimizers,
clear controls and128 updates per branch. Actual CUDA execution is pending.
No checkpoint promotion or wholesale archive admission. Next: VM preflight/pilot.

## Existing local candidates

Source-only inspection of the previous review queues finds potential additional
examples, not an approved dataset extension:

| Source | Observation | Status |
| --- | --- | --- |
| `dataset/real_occlusion_review/masked (1889).jpg` | Respirator and scarf, small face; scarf is principally around the neck | Not sufficient evidence of scarf-over-face handling; original resolution hold remains |
| `dataset/real_occlusion_review/masked (1892).jpg` | Hand overlaps a respirator; small face and stock watermark | Combined-covering candidate; crop/mask review still required |
| `dataset/asian_faces/asian_face_04535.jpg` | Reflective eyewear and foreground food at the lower face boundary | Native overlap extent needs review; not an approved object mask |
| `dataset/thumbnails128x128/65601.png` | Red sunglasses; hand near chin | Hand is near the chin, not clearly over facial features; do not count as a standalone hand case |

These source paths are relative to both repository roots. Prior queue status and
training/split membership stay unchanged. They do not fill the missing hair,
scarf, standalone-hand and general-object review requirements by themselves.

## Caltech Occluded Faces in the Wild (COFW)

COFW is a relevant additional source because its creator paper describes real
occlusions involving hair, hands and object interactions. The paper specifies
1345 training faces (845 LFPW plus 500 COFW) and 507 test faces, with landmark
positions and occlusion flags. Those flags are not pixel removal masks or paired
uncovered facial targets. Any use for this project needs source/crop review and
separate removal-mask annotation. [Creator paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/12/BurgosArtizzuICCV13rcpr.pdf)

The official CaltechDATA record is DOI `10.22002/D1.20099`, version 1.0, credited
to Xavier Burgos-Artizzu, Pietro Perona and Piotr Dollar. The downloaded record
API metadata lists `cc-by-4.0` and public file access. Retain attribution and the
source citation with derived assets. The color archive is 503,327,162 bytes;
published MD5 `8b21d126c4e1fb307cb463578eef0511`.
[Official record](https://data.caltech.edu/records/bc0bf-nc666)

Verified source metadata snapshot:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cofw_source_research_v1\record.json`,
SHA256 `95a54786dd23eeeef0db5b33e1b5bb79b73c09accf12d0eb6a93a1318896b94e`.
Documentation archive: `documentation.zip`, 1,140,264 bytes; MD5
`cc3a8f6fc6b49f6186985c1c68c6fa52` matches the publisher.
If transferred, these live under `~/forensic-dgp/outputs/cofw_source_research_v1/`.

## Earlier acquisition failure — superseded, partial remains excluded

The official color transfer ended with curl exit 18 after 378.9 seconds and only
2,153,540 bytes (approximately 5.7 KB/s), versus 503,327,162 expected. It is now
named `COFW_color.zip.partial` in the research directory. Preserve it as a failed
transfer artifact; it is not a valid dataset archive and must not be extracted,
sampled or admitted to a gallery/training manifest. No image matrix was extracted
or opened. The first ZIP member's filename was inspected as metadata only.
No transfer process remains running.

Acquisition status evidence: `research_status.json`, SHA256
`c22ccc480221c292e87092d66d41d5ebf438815329e2b600760a33f4b0b6705c`.
An isolated `h5py==3.14.0` reader was installed under
`outputs/cofw_read_dependencies/`; the main application environment was not changed.
Elevated execution verified the reader version. It was unused at this earlier
milestone. The complete archive and training-only extension above supersede
that hold without altering the earlier evidence.

Keep the publisher's original train/test membership when acquiring usable images.
Use train sources for the next developmental gallery; leave test images separate.
Check overlap with existing project sources and record exposure. Do not claim
identity disjointness or lack of generator pretraining overlap based on file hashes.

## RealOcc: verified source review, no training admission

The creator paper describes 550 aligned RealOcc faces from Pexels/Unsplash and
270 RealOcc-Wild photographs. Its binary task segments visible face: occluders
and non-face pixels are background. Transparent/translucent glasses are also
background under its definition, which differs from our preserve-clear-glasses
scope. The public archive is linked by the authors. [Creator paper](https://arxiv.org/html/2205.06218v1#S3.SS1),
[Creator repository](https://github.com/kennyvoo/face-occlusion-generation)

Only `RealOcc.7z` was acquired. It is 226,205,702 bytes, observed SHA256
`de6a26ecc00457c0067991d95906b15638a2f217c5b5ea01a5db42435510577f`
(acquisition fingerprint, not publisher checksum). Safe archive-member validation
preceded extraction. Local root is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\realocc_source_v1\`; intended VM root is
`~/forensic-dgp/outputs/realocc_source_v1/` after transfer. The initial Google Drive
warning HTML is preserved as `.partial` and excluded from dataset use.

All 550 JPEG/PNG pairs load, agree in dimensions and match the complete author
`val.txt` membership. Source masks have values 0 and 1; visible-face semantics
agree with inspected overlays. Do not invert them into occlusion targets: the
complement includes background, clear glasses and non-obstructing hair. Separate
operator removal proposals must follow our scope. Author validation membership
remains intact; these inspected sources cannot become an unseen project holdout.

Integrity record `source_review/integrity.json` SHA256
`aaf1645debcbf3d43e794fdf92def776a4ba6f314dac3182a04f679f45b7114e` includes source
hashes, dimensions, binary counts and contact sheets. Initial acquisition metadata
has `image_integrity_pending`; this completed record supersedes that earlier
status without changing it. Source-only review finds candidate hands, scarves,
other objects and hair obstructing eyes. Native crop/footprint review and duplicate
checks are needed before a fixed development-gallery extension.

No explicit dataset training license was found in the archive/repository during
this acquisition. Attribution/source links are retained; training admission is
false and would require terms review. Pixel/file duplicate checks cannot prove
identity disjointness or absence from generator pretraining. RealOcc-Wild was
not downloaded. No source images or masks were committed to Git.

## Current execution decision

The verified RealOcc archive supplies a separate eight-source development-gallery
draft: two hands, one obstructing-hair case, two scarves, two objects and one
nearly hidden face for rejection. Native source/crop and green removal proposals
were inspected before freezing 16 native/degraded cases. No publisher masks are
used as covering targets; no training admission occurs. Exact file hashes were
checked against the 115-source supported review manifest and prior ten-source
practical cohort, with no match. This does not establish full-corpus, identity or
pretraining disjointness. Frozen protocol is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\broad_covering_gallery_v1\frozen_protocol.json`
(VM `~/forensic-dgp/outputs/broad_covering_gallery_v1/frozen_protocol.json` after transfer),
SHA256 `b7a1a447482d650bb0a18b13d649f0c548c3c59acd6db202111169599c805bc9`.
The integrated comparison is complete: 28 saved outputs, 32 automatic/assisted
rows, zero optimizer updates. Source and saved-pixel audits pass. Assistant review
finds useful assisted hand/hair/scarf/object estimates, one partial scarf and weak
automatic native-covering detection. Both reviewed nearly hidden inputs reject;
the automatic proposals miss both. Report is
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_BROAD_OUTPUT_RESULTS.md` (VM
`~/forensic-dgp/PRACTICAL_BROAD_OUTPUT_RESULTS.md` after transfer).
The 36-case pretrained XSeg and returned direct-detector comparisons are complete.
XSeg violates clear-glasses/background scope; direct coverage worsens under camera
degradation and hair remains undetected. Neither changes the app checkpoint.
Reports are `PRACTICAL_XSEG_RESULTS.md` and `PRACTICAL_DIRECT_DETECTOR_RESULTS.md`
under the repository roots above. The partial COFW archive stays excluded; the
newly verified complete archive is separate. RealOcc author-validation
membership and publisher labels remain unchanged.

## Uploaded Mendeley source and next VM decision

The user supplied the ZIP after three automated publisher HTTP 403 responses.
All 11,987 JPEGs pass CRC/decode checks. There are 6,101 distinct decoded images,
with 5,886 repeated instances and no supplied annotations or splits. Six source
sheets expose 363 distinct file IDs. Only eight related mask/sunglasses/hand/hair
captures receive reviewed detector labels, including one clear control. They
belong to one training-only cohort; no scarf/general-object training coverage or
paired uncovered targets is established. Publisher version 1 is CC BY 4.0;
retain attribution. [Publisher dataset](https://data.mendeley.com/datasets/s57wnx78vh/1)

Source/data report is
`C:\xampp\htdocs\YEAR 4\Testing\MENDELEY_OCCLUSION_DATA.md` (VM
`~/forensic-dgp/real_camera_vm_bundle/MENDELEY_OCCLUSION_DATA.md` after transfer).
Supported V2 has 91 training, 25 unchanged validation and seven unchanged
previously inspected test sources. Exact byte/native-pixel overlap checks against
115 prior raw sources find no match; full-corpus, identity and pretraining overlap
remain unverified. Paired camera RGB changes preserve labels/support/geometry.

The three-arm native83/camera83/camera91 VM diagnostic is complete and audited,
112 updates each. Camera degradation improves transfer and the small source
addition helps hands; external hair and nearly hidden cases remain missed.
Retention/clear checks fail, so no checkpoint is selected. Four generated estimates
still show missed covering remnants. Full report:
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_RESULTS.md` ↔
`~/forensic-dgp/REAL_CAMERA_RESULTS.md` after a future transfer.

The COFW preparation above now supplies varied training-only coverings and clear
controls. Keep the inspected RealOcc validation gallery excluded from training;
the entire Mendeley ZIP is not automatically usable labelled data. Paired V3
camera inputs are verified. The distinct finite V2 protocol/package is now
independently verified; see `VARIED_COVERING_VM.md`. Next: VM preflight/pilot. No
generator/restoration retraining is indicated by the current mask-remnant failures.
