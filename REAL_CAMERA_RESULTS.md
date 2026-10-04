# Camera/source pilot results — 3 October 2026

The returned experiment and local follow-up are audited. Camera degradation is a
useful training ingredient, but **none of these checkpoints is selected for the
application**. Hair and nearly hidden faces remain detection failures. Four
downstream face estimates still contain missed cloth/finger remnants. The next
work is targeted covering-data/footprint preparation before another VM recipe.

| Artifact | Windows local | VM original or receiving counterpart |
| --- | --- | --- |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_RESULTS.md` | `~/forensic-dgp/REAL_CAMERA_RESULTS.md` after a future transfer |
| Download | `C:\xampp\htdocs\YEAR 4\Testing\outputs\real-camera-results.tar.gz` | `/home/janusdominic0/forensic-dgp/real_camera_vm_bundle/real-camera-results.tar.gz` |
| Extracted models | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_real_camera_v1\outputs\real_camera_vm\<arm>\last.pth` | `~/forensic-dgp/real_camera_vm_bundle/outputs/real_camera_vm/<arm>/last.pth` |
| Local follow-up | `C:\xampp\htdocs\YEAR 4\Testing\outputs\real_camera_inference_review_v2\` | `~/forensic-dgp/outputs/real_camera_inference_review_v2/` only after explicit transfer |
| Face estimates | `C:\xampp\htdocs\YEAR 4\Testing\outputs\real_camera_completion_review_v1\` | `~/forensic-dgp/outputs/real_camera_completion_review_v1/` only after explicit transfer |

## Verified execution and scope

The 161,238,229-byte archive matches its exact LF checksum:
`5dc73ce77291b8622530ae825676728776caa8fdbfd36250f1d0a4e23b1c3c24`.
All 1,866 regular archive members, inventory/protocol bindings, 1,848 saved
binary masks and 336 step records pass independent verification. All three
restricted-loaded checkpoints have 184 finite tensor states, including 92
unchanged reference-head/BN states. Encoder, decoder and covering-head weights
changed. Frozen source/code/data hashes remain intact.

CUDA preflight records one image and zero updates on NVIDIA L4, using
Torch 2.9.1+cu129. The three independent branches each log 112 updates, final
moment step 784 and cumulative model 994 updates. Recorded training/measurement
time is **53.16 seconds**, excluding startup and final archive export, with
4,536 forwarded images. This is a small detector diagnostic, not a full 80,000
image DGP epoch, generator training or identity-loss training.

Each branch starts from reflective epoch 42 with the same inherited moments.
`native83` uses 83 prior real sources; `camera83` mixes their native/degraded inputs;
`camera91` adds eight related reviewed sources. All use the same 280 reflection
fixtures. These measurements are **exposed training-cohort fit**, not an unseen
test. The original 425-case qualification gates were not evaluated or weakened.
Final optimizer files remain on the VM; their digest/step declarations are logged,
but their tensors were not returned for independent inspection.

## Training-cohort results

IoU below is pooled over supervised pixels. Unknown/padding predictions are
reported separately and never converted into supervised errors.

| Group | Source42 | native83 | camera83 | camera91 |
| --- | ---: | ---: | ---: | ---: |
| Old native83 | 0.9150 | 0.9051 | 0.9305 | 0.9188 |
| Old degraded83 | 0.7621 | 0.7861 | 0.9095 | 0.8861 |
| New native8 | 0.3474 | 0.3411 | 0.3730 | 0.6394 |
| New degraded8 | 0.2603 | 0.2681 | 0.3487 | 0.5185 |
| Reflection280 | 0.8094 | 0.9126 | 0.9154 | 0.9106 |

The predeclared five-check decision is reproduced exactly:

1. Camera83 improves old degraded IoU over native83: **pass**.
2. Camera83 retains old native/replay fit without increased errors: **fail**;
   visible false-positive rates increase despite higher IoU.
3. Camera91 improves new native/degraded fit over camera83: **pass**.
4. Camera91 retains old native/degraded/replay fit over camera83: **fail**.
5. Every final clear control stays empty: **fail**; camera83 marks 137 supervised
   pixels on training reflection clear fixture 171. Other branches have no clear
   errors in this diagnostic.

Camera91 changes four empty new covered-source native masks into nonempty masks.
That does not mean complete removal: the vertical hand remains a small fragment,
the mouth hand gains substantial coverage and the hair curtain remains nearly
empty. Its degraded hair case stays empty. All ten training-preview rows were
inspected. No `best.pth` was created; all three `last.pth` files remain unselected.

## Local practical transfer comparison

The frozen follow-up uses the previous 36 native/degraded development cases,
unchanged threshold 0.5 and 3px proposal margin. It runs 152 detector-image forwards
in 13.74 CPU seconds excluding loading: 44 fixed return-reproduction cases and 108
new practical predictions. No optimizer, generator or restorer runs in this step.
Model-state fingerprints remain unchanged.

Of 44 return masks, 43 reproduce exactly on CPU. Camera91 real169 differs at
one threshold-ambiguous pixel within the predeclared local probability distance
1e-5; thresholds/cases were not adjusted. Reproduction probabilities were checked
by the runner but not archived. All 108 archived practical probabilities,
216 raw/proposal masks and 144 metric records are independently verified.
Pillow 7px maximum filtering independently reproduces the OpenCV 3px expansion.
All six 36-case preview sheets were inspected.

Mean proposal recall on degraded cases:

| Common coverings | Source42 | native83 | camera83 | camera91 |
| --- | ---: | ---: | ---: | ---: |
| Face masks, 3 cases | 52.4% | 50.7% | 82.8% | 80.2% |
| Sunglasses, 2 cases | 45.8% | 71.8% | 74.8% | 78.7% |
| Strong white glare, 1 case | 41.8% | 44.2% | 78.9% | 83.0% |

| Broad coverings | Source42 | native83 | camera83 | camera91 |
| --- | ---: | ---: | ---: | ---: |
| Hands, 2 cases | 5.3% | 6.1% | 36.7% | 57.0% |
| Obstructing hair, 1 case | 0% | 0% | 0% | 0% |
| Scarves, 2 cases | 34.5% | 31.8% | 39.0% | 40.8% |
| Objects, 2 cases | 36.1% | 38.4% | 67.4% | 62.6% |

All four clear/uncovered controls remain empty for every model. No usable case
unexpectedly rejects. **Both nearly hidden hand inputs still fail rejection** for
every model because detection marks only parts of their covering. The hair pair
remains empty. Knit scarf has partial coverage; the degraded scarf/gloves case
remains empty or tiny fragments. Hands, mask bodies and lens interiors improve,
but missed fingers, straps, rims and lens margins still matter to removal.

These are approximate source-reviewed references, not exact hidden anatomy.
Cases were previously inspected; RealOcc author-validation sources remain excluded
from training. The three-quarter mask case remains a diagnostic outside the
initial frontal/mild-turn scope. These scores do not qualify population accuracy
or a production model.

## Actual face-output check

The existing pinned CodeFormer completion plus local Auto visible-restoration/
palette route generates four estimates: camera83/camera91 proposals on degraded
cloth-mask and hand-mouth inputs, with no manual edits. Four completion and four
restoration forwards take 38.03 CPU seconds excluding loading. No detector is loaded
and no model is trained or changed. Captured float stages independently reconstruct
all four final PNGs exactly; restoration preserves completed pixels before the
documented grayscale projection. The hand output remains grayscale.

Preview: `C:\xampp\htdocs\YEAR 4\Testing\outputs\real_camera_completion_review_v1\preview.png`.
Both cloth-mask estimates have plausible mouth/beard content but leave gray
cloth/strap edges. Both hand estimates contain a plausible lower face with
remaining fingers at cheeks/chin. Camera91 removes more of the hand, but neither
is complete removal. These are assistant visual observations, not expert/user
acceptance or same-person hidden-face recovery. Two diagnostic inputs do not
demonstrate whole-family readiness.

## Next work and preserved state

Prepare a distinct dataset/recipe addressing the demonstrated gaps:

1. Retain input camera degradation as a supported ingredient; it outperforms the
   matched native continuation on the degraded development cases.
2. Audit full removal footprints, including opaque mask edges/straps and fingers
   over the facial region. Do not overwrite historical labels or blindly expand
   every region into clear frames/normal hair.
3. Add varied reviewed hands, obstructing hair and scarves from training-eligible
   sources, with clear controls. Eight related captures are insufficient evidence
   of broader hair/general-covering learning; do not train on the inspected RealOcc
   validation gallery or pretend real covered photos have uncovered-face targets.
4. Keep clear-control/visible-retention counterexamples and a separate nearly-hidden
   detection/rejection requirement. Freeze any new protocol before another VM run;
   no unchanged repeat or gate waiver.
5. Recheck candidate output remnants and the original qualification gates before
   any application checkpoint change. All actual training remains on the L4 VM.

The main app retains its baseline detector and manual review/correction flow.
Phase 3 restoration weights, Phase 5 pending GPU status, CodeFormer weights,
historical results and the native-expert pilot are unchanged. No new app edit,
commit or push occurred. The active Goal is not complete.

## Evidence

Paths below are under Windows `C:\xampp\htdocs\YEAR 4\Testing\outputs\`; receiving
VM counterparts are `~/forensic-dgp/outputs/` only after an explicit future transfer.
Original returned evidence instead remains within the VM pilot directory above.

| Evidence | SHA256 |
| --- | --- |
| Return audit: `real_camera_results_validation_v1/verification.json` | `fbe47739fa7ec05f88b7ee22861d81b64b945c320109e4b7aee5a05c8af74245` |
| Practical follow-up audit: `real_camera_inference_validation_v1/verification.json` | `8050c978a555ec7a56a8c6c9653f4ee03f59fd683419222ce3ffd22d2a7d6bfe` |
| Face-output pixel audit: `real_camera_completion_review_v1/independent_verification.json` | `128bc2df8d47a0c4e187a2fda261d38fe9d0738c31038c65ea892fab9b394dc1` |
| Eight-sheet visual review: `real_camera_review_v1/visual_review.json` | `b1815f4d7fe662244933f6f80916cdedad0b0a6e71422a0b581938d31ad814a5` |

The first local practical attempt stopped before model construction because the
sandbox denied installed package reads. Its protocol/execution/interruption files
remain in `outputs\real_camera_inference_review_v1\`. V2 runs the same frozen code,
cases, models, threshold, margin and budget with dependency access in a fresh
directory. No package reinstall or hidden local training occurred.
