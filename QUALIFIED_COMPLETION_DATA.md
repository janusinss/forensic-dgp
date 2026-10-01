# Source-qualified completion data: 1 October 2026

This records the original four-source prototype milestone. The later expanded
28-source,280-case bounded detector pilot is documented in
[REFLECTION_COVERAGE.md](REFLECTION_COVERAGE.md), with launch instructions in
[REFLECTION_COVERAGE_VM.md](REFLECTION_COVERAGE_VM.md). That pilot is packaged;
its new CUDA run and model-output benefit are unverified. The original prototype
and evidence below are preserved.

An opt-in data prototype now separates uncovered references from intrinsically
obscured photos, preserves nonsquare geometry and provides partial lens-reflection
fixtures. It is not integrated into training or the application. No new optimizer
updates occurred, and no new detector/checkpoint is selected. The matched loss
experiment still fails the unchanged synthetic retention guard; see
[FACE_OCCLUSION_FOCUS_RESULTS.md](FACE_OCCLUSION_FOCUS_RESULTS.md).

## What is implemented

`completion_data_v2.py` requires exact source bytes and independently verified
training membership. Native review, two reviewed eye positions and an explicit
uncovered-reference decision are required. Pending, uncertain, excluded and
intrinsically obscured photos cannot become paired examples. A folder name or
procedural zero mask does not establish eligibility. This version leaves
`completion_data.py`, the old638-case replay and the400-case benchmark unchanged.

One affine scale fits native images inside a square with neutral padding.
Horizontal and vertical scales are equal. A conservative binary `valid` mask
excludes pixels influenced by padding; consumers must exclude `valid=0` from
all supervised losses/region reductions. This is **aspect preservation, not
facial alignment**. The earlier opt-in `SelectiveEyeAlignment` remains separate.
OpenCV documents the affine mapping and border/interpolation behavior used here.
[OpenCV geometric transformations](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).
FFHQ's own preparation uses landmark-based alignment/cropping; our padding does
not reproduce that pipeline or prove alignment of the Asian source photographs.
[NVIDIA FFHQ reference implementation](https://github.com/NVlabs/ffhq-dataset).

Reviewed native eye positions anchor two lens interiors. Four procedural styles
cover parts of those interiors: white patch, white streak, scene reflection and
blue glare. An uncovered frame image is the shared reference for all styles;
clear-lens controls have no hole or input/reference difference. This keeps frame
pixels visible rather than teaching completion to remove eyeglasses. Introduced
patches are opaque approximations with known support. The renderings remain
simplified; realistic transfer and improved output are unproven.

## Current source qualification and fixtures

The registry contains the180 already screened training-clear sources. It admits
only four prototypes after native source views: IDs0/4 from the Asian source
pool and101/102 from FFHQ. Their filenames are `asian_face_04576.jpg`,
`asian_face_09270.jpg`, `69436.png` and `05891.png`. The fixed source-only review
cohort was the first six training-clear source IDs in each pool, without model
error ranking or validation/test photos. Four suitable prototypes were chosen
from those twelve. Native eye centers were manually proposed and visually
checked; this is assistant screening, not independent expert adjudication.

| Usage | Sources | Training status |
| --- | ---: | --- |
| Paired uncovered prototypes | 4 | Fixture generation only |
| Likely intrinsic occlusion | 2 | Unpaired candidates; no accepted mask |
| Pending qualification | 174 | Excluded from paired examples |

Sources160/177 remain likely intrinsic occlusions. They are not relabeled or
enabled for detector training. Other pending sources remain excluded even if
contact-screened or native-viewed. The old masks/cache are retained. The four
native references are low resolution; resizing does not supply high-resolution
ground truth or establish the missing person's real facial features.

Twenty cases were prepared: four references × clear control/four reflection
styles. All123 prepared files independently hash/recount. Every reflection mask
is binary, partial and contained within supported lens interiors. All20 cases
have **zero visible pixels changed outside the reflection target**. All style
variants share their respective frame reference and unframed base. Affine/eye
metadata independently reconstructs. Native source, geometry and reflection
previews were inspected. There was no model inference in this prototype.

Sixteen new regression tests pass. Including the existing completion/alignment
contracts, **37 tests pass**. Tests were first observed failing for the missing
implementation. `git diff --check` passes.

## Files and reproducible checks

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository root: `~/forensic-dgp/`.
Previous experiment execution root: `~/forensic-dgp/coverage_vm_bundle/`.
The new prototype exists locally; it has not been uploaded or packaged for VM
training. Use the same relative artifact layout if transferring it later.

| Relative path | Windows location | Corresponding VM location |
| --- | --- | --- |
| `completion_data_v2.py` | `C:\xampp\htdocs\YEAR 4\Testing\completion_data_v2.py` | `~/forensic-dgp/completion_data_v2.py` |
| `QUALIFIED_COMPLETION_DATA.md` | `C:\xampp\htdocs\YEAR 4\Testing\QUALIFIED_COMPLETION_DATA.md` | `~/forensic-dgp/QUALIFIED_COMPLETION_DATA.md` |
| `outputs/qualified_reflection_v1/` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\qualified_reflection_v1\` | `~/forensic-dgp/coverage_vm_bundle/outputs/qualified_reflection_v1/` |
| `outputs/qualified_source_review_v1/` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\qualified_source_review_v1\` | `~/forensic-dgp/coverage_vm_bundle/outputs/qualified_source_review_v1/` |

`outputs/qualified_reflection_v1/registry.json` SHA256:
`9dd48c58bd17a1f95588fb745c4cfda06ed15e9932e0e08d17dcdf524cd19116`.
Independent `audit.json` SHA256:
`04549c2415f66a8dfea7b376a3866c75f1b3e63883da84ded987a4cb0289ca9f`.
The separate `visual_review.json` binds that review to both hashes.

Run the contracts in local PowerShell:

```powershell
Set-Location 'C:\xampp\htdocs\YEAR 4\Testing'
venv/Scripts/python.exe -m unittest tests.test_completion_data_v2 tests.test_qualified_reflection_review tests.test_qualified_reflection_audit tests.test_completion tests.test_completion_alignment
```

The two scripts are `scripts/prepare_qualified_reflection_review.py` and
`scripts/audit_qualified_reflection_review.py`. Both preserve existing output
evidence; **do not rerun their default output**. A new output version is required
for new fixtures or renderer changes. The registry binds the renderer/builder,
source-screen/cohort and previous fixed input hashes. Do not change those bound
files in place and continue using the old evidence.

## Next before VM training

1. Qualify a larger training-only uncovered base cohort and inspect native eye
   anchors. Keep uncertain eyewear unpaired until actual mask review.
2. Freeze the new recipe, source/mask hashes and augmentation previews. Preserve
   the old benchmark and retention safeguards; do not move evaluation examples
   into training or change its target policy to make a candidate pass.
3. Implement and verify a bounded matched VM pilot whose training reductions
   explicitly ignore padding. Hold initialization, budget and real supervision
   fixed when comparing reflection coverage against its matched control.
4. Audit returned masks and guard decisions. Only an eligible detector advances
   to fixed-cohort end-to-end completion review before any promotion.

The prototype is ready for **data review**, not GPU training. No new VM command
or training package is ready. Track1 retains Phase3
`checkpoints/dgp_zamboanga_final.pth`; full Phase5 identity training is pending.
Track2's generator and application remain unchanged. Identity separation,
external pretraining overlap and real-world generalization remain unresolved.
The full improvement goal remains active and unmet.
