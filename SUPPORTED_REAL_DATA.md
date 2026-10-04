# Supported real-mask dataset — status updated 3 October 2026

## Current V3 COFW extension — data verified; finite VM pilot packaged

`C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v3\manifest.json`
has 165 records: 133 training (89 covered/44 clear), 25 unchanged validation
(15 covered/10 clear), and seven unchanged previously inspected test records
(four covered/three clear). Intended Linux path after transfer:
`~/forensic-dgp/dataset/detector_supported_review_v3/manifest.json`.
All 123 V2 records and active image/mask/core/valid-support bytes are exact.

Forty-two author-training COFW images add 31 varied coverings and 11 clear
controls. Native source/geometry audits pass; ambiguous boundaries, padding and
12 declared raster-disagreement pixels are unsupervised while RGB remains intact.
These are assistant pilot detector annotations, not uncovered-face targets.
No COFW test images or full-archive training admission.

The unchanged three-tensor reader verifies all records. Dataset audit verifies
493 files, preserved held-out support/mannequin and old raw-source hashes.
Registry SHA256 `abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd`;
`outputs/cofw_supported_dataset_validation_v1/verification.json` SHA256
`a403f14457d7ba850ff2dbea4236b1db88276cd9e5d4f8192a65e0a0159261cc`.

Source/terms/refinement history: `COFW_DATA_PREPARATION.md` under both roots after
transfer. New artifacts are local only. Old raw-source paths are external
dependencies listed in the audit; include/verify them in the next package.
`training_recipe_ready=false`; zero model forwards/optimizer updates or promotion.
The new paired cache `outputs/cofw_camera_pairs_v2/manifest.json` contains
133 native/133 degraded training views. All degraded pixels independently
reconstruct; previous 91 pairs remain exact and no held-out input is included.
Its `independent_verification.json` SHA256 is
`c7fb19ff5d936c5e43f8a5c3a56b8ca59db40986d04f3be4c156660258652471`.
Both five-row preview sheets are inspected. These are relative paths under both
roots after transfer. The separate finite V2 VM protocol is now packaged and
independently verified; data-stage `training_recipe_ready=false` remains intact.
Runbook `VARIED_COVERING_VM.md` uses matched `existing91`/`varied133` branches,
128 updates each, fresh optimizers and more clear controls, with original gates
preserved. Bundle: `outputs/varied-covering-vm-bundle-v2.tar.gz`; intended VM root
`~/forensic-dgp/varied_covering_vm_bundle/`. All165 raw sources are included,
closing the earlier provenance-transfer hold. Actual CUDA execution is pending;
no local training/model qualification. Next: VM preflight/pilot and return audit.

## Previous V2 extension and executed VM diagnostic

The new registry is
`C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v2\manifest.json`.
Its isolated VM counterpart after bundle transfer is
`~/forensic-dgp/real_camera_vm_bundle/dataset/detector_supported_review_v2/manifest.json`.
It preserves all 115 V1 records exactly and adds eight reviewed Mendeley captures
as one training-only related cohort: seven covered sources and one clear control.
Full validation/test support, active metadata and bytes remain unchanged.

| Split | Covered | Clear | Total |
| --- | ---: | ---: | ---: |
| Training | 58 | 33 | 91 |
| Validation | 15 | 10 | 25 |
| Previously inspected test | 4 | 3 | 7 |

Independent audit verifies 291 data files plus the manifest. New labels preserve
source aspect ratio and exclude uncertain native polygon boundaries/crop edges
from supervision while retaining observed RGB. They are assistant pilot detector
labels, not uncovered-face targets. The full uploaded ZIP and unlabelled/non-face
images are excluded. Provenance and annotation history are in
`C:\xampp\htdocs\YEAR 4\Testing\MENDELEY_OCCLUSION_DATA.md` ↔
`~/forensic-dgp/real_camera_vm_bundle/MENDELEY_OCCLUSION_DATA.md` after transfer.

The separately audited 182-view native/degraded cache changes RGB only, with
identical targets/support/geometry. Five camera/scheduling contract tests pass
without model/optimizer work. Registry `training_recipe_ready=false` remains its
data-stage declaration; the frozen protocol in
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_VM.md` ↔
`~/forensic-dgp/real_camera_vm_bundle/REAL_CAMERA_VM.md` explicitly consumes it
with supported reductions. Package readiness means VM preflight and the fixed
three-arm detector diagnostic only. That preparation-time CUDA hold is
superseded by the completed return audit below.

The native83/camera83/camera91 L4 run is now complete and independently audited,
112 updates per branch. Camera augmentation improves degraded fit; added sources
help hands, but retention/clear checks fail and external hair stays undetected.
All eight preview sheets and four downstream estimates are reviewed. Report:
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_RESULTS.md` ↔
`~/forensic-dgp/REAL_CAMERA_RESULTS.md` after future transfer.

Next: varied reviewed covering/complete-footprint data with clear/retention
counterexamples before a distinct VM protocol. No `best.pth`, app swap or
qualification occurred. The older native-expert recipe remains preserved and
must not start automatically; registry/labels/splits remain unchanged.

## Previous V1 dataset verification — 2 October 2026

The separate 115-record registry and explicit three-tensor reader are verified.
All 105 previous records retain their original image/mask bytes, metadata and
split order. All 73 original training image/mask tensor pairs are exactly equal
to the legacy reader. The ten native additions have reviewed labels and explicit
supervision support. These data checks do not qualify a model or a GPU recipe.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; latest executed bundle:
`~/forensic-dgp/coverage_vm_bundle/`.
The new files are local only. Their intended repository counterpart is
`~/forensic-dgp/dataset/detector_supported_review_v1/`; no new transfer or VM
execution has occurred. Actual model training remains VM-only.

## Fixed membership and supervision

| Split | Original records | New records | Covered | Clear | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 73 | 10 | 51 | 32 | 83 |
| Validation | 25 | 0 | 15 | 10 | 25 |
| Test | 7 | 0 | 4 | 3 | 7 |

New covering sources: 171, 216, 348 and 374. New transparent controls:
207, 208, 217, 230, 240 and 241. The other fourteen qualified sources remain
unlabelled and absent from the registry; they are not empty-mask controls.
Native source, polygon, overlay and independent geometry/support review is
recorded in `C:\xampp\htdocs\YEAR 4\Testing\REAL_REFLECTION_PROPOSALS.md`.

Every original record uses full 256×256 supervision and source support. No
difficult validation or test pixel is ignored. The known mannequin remains
validation index8, parent index31, basename `new_covered_40.png`; human/mannequin
metrics must still be reported separately without removing it from the gate.
The existing test set was previously viewed and is not a pristine final holdout.

Each new sample preserves source aspect ratio using the reviewed uniform
pixel-center affine. Only padding is RGB96. Source171's unresolved lower
helmet/strap band retains its observed RGB, but is excluded from every target
reduction: 7,268 native unknown pixels; 39,400 supervised output pixels versus
51,200 source-supported output pixels. All known positive pixels remain supervised.
The other additions also exclude true padding. These are unpaired real detector
labels, not uncovered-face or high-resolution reconstruction targets.

## Explicit reader contract

Reader: `C:\xampp\htdocs\YEAR 4\Testing\supported_real_data.py`.
Registry: `C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v1\manifest.json`.
The new schema is `dgp-supported-real-masks-v1` with `supported_records` and no
legacy `records` key. The actual legacy `detector_training.load_manifest` rejects
this registry immediately. Existing pair-only training scripts must not consume
these partial labels.

```python
from supported_real_data import SupportedMasks

data = SupportedMasks("dataset/detector_supported_review_v1/manifest.json", split="train")
image, mask, valid = data[0]
```

The tensors are float32 RGB `[3,256,256]` in `[0,1]`, covered-region target
`[1,256,256]`, and supervised-pixel support `[1,256,256]`. `valid=1` means the
target is supervised. The reader does no resizing, augmentation or model work.
It requires reviewed/support declarations, canonical in-root PNG paths, binary
maps, exact hashes and disjoint cross-split source/group ownership; artifact
hashes are rechecked on access. Unknown observed RGB remains input context.

A future runner must explicitly use support for all supervised BCE, Dice and
hard-visible reductions. The existing `reflection_coverage.supported_segmentation_loss`
has a fixture contract confirming that changed ignored logits leave loss unchanged
and receive zero fixture gradient. That test creates no model or optimizer.
`training_recipe_ready=false` is an accurate readiness declaration, not a CUDA
execution lock provided by this data reader. The distinct schema blocks accidental
legacy consumption; a future runner still needs its own protocol and checks.

## Independent verification

Builder: `C:\xampp\htdocs\YEAR 4\Testing\scripts\build_supported_real_dataset.py`.
Independent auditor:
`C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_supported_real_dataset.py`.
Saved verification:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\supported_real_dataset_validation_v1\verification.json`.
These have no newly executed VM counterpart. Both tools preserve existing output
directories; do not rerun them over the completed dataset/evidence.

The auditor checked all 252 dataset files, active original metadata as well as
embedded lineage, raw source hashes, copied bytes, split/source/group ownership,
all ten reviewed native annotations and acceptance bindings. It independently
reconstructed oriented RGB, polygons, geometry and support, checked the excluded
pending sources, compared all original training tensor pairs, and confirmed that
the mannequin basename and full held-out support remain intact.

Seven targeted contracts pass: five reader/support tests and two independent
audit counterexamples. The latter reject changed active labels or renamed parent
basenames even when embedded lineage is correct, and reject ignored original
pixels. No model forwards, held-out forwards, optimizer updates or training occurred.

```powershell
cd 'C:\xampp\htdocs\YEAR 4\Testing'
venv/Scripts/python.exe -m unittest tests.test_supported_real_data tests.test_supported_real_dataset_audit
```

| Artifact | SHA256 |
| --- | --- |
| Supported registry | `860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace` |
| Independent verification | `308e5ca5b00bf8c52b36eba78605e41d188b754044307fd54d4f39edf16cbfe9` |
| Reader | `bbeabae1df2684fd862b1a88752533394a62cb562f098d355e52427480ac797c` |
| Executed builder | `0cf9ed0cc71a469707125824ec925a4bab61c74564761ee5a0318482d0cdaeec` |
| Independent auditor | `dd0691dd252f8fd6e5f2c23ae5069e96b373249b9f943641942604583b9f0aea` |

## Previous V1 decision

The previous teacher/replay, projection, post-update retention and feature-head
reports have now been reviewed. Projection did not restore retention and real
coverage remained incomplete; the constrained pilot failed useful real training fit despite preserving
training replay-loss ceilings. Earlier frozen SAM heads also failed retention,
including under a label-informed presence oracle. The latest pretrained detector
adaptation learns real coverings but fails the original synthetic safeguards.
These results do not establish that every distinct architecture is infeasible.

The subsequent frozen-head feasibility diagnostic is now complete and audited;
see `C:\xampp\htdocs\YEAR 4\Testing\FROZEN_COMPLEMENTARITY_RESULTS.md`.
Both heads miss8,205/21,853 reviewed lens pixels, and fixed union/intersection fail
the training checks. No selector restricted to those binary masks can recover the
common misses. These are training-only findings, not held-out model selection.

The bounded native-expert pilot was subsequently prepared and audited, but not
executed. Its inputs/recipe remain preserved. The later practical direct-detector
review demonstrated camera/covering gaps; the current V2 camera/source diagnostic
above is the next prescribed VM experiment. Retain the original app detector.
Useful training fit alone will not qualify a model. No new GPU training has started.

Original gates, accepted datasets, generator/application and Phase3 restoration
baseline remain unchanged. No commit/push, dependency reinstall or new VM upload
occurred. A qualifying detector and reviewed end-to-end completion improvement
are still required; hidden facial features remain plausible estimates.
