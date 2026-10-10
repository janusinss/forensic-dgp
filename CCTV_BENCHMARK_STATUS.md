# CCTV restoration benchmark - current QMUL checkpoint comparison, 9 October 2026

All24 frozen QMUL DEV cases now compare resize, original Phase3, the current
identity-v2 checkpoint and CodeFormer fidelity1 on the same256-pixel inputs.
This replaces neither the historical checkpoint findings nor the separate
ChokePoint comparison. All saved outputs pass the artifact audit; all24 current
DGP outputs replay exactly. All96 delivered cells receive visual review.

The current DGP does not establish a convincing useful clarity gain over resize
on the six predeclared coarse frontal/mild cases. Seven insufficient cases retain
their input-only labels. This is unpaired assistant development evidence, not
native PSNR/SSIM, identity accuracy, independent final validation or qualification.
All reserved pixels remain unviewed. No app/model selection changes.

Current report: CCTV_DGP_CURRENT_QMUL_NATIVE_COMPARISON_V1_RESULTS.md.
V42 remains failed at0.00692364% versus1%, with appearance regressions. The
original-feature diagnostic remains prepared for manual VM execution; no return
is locally available in this review. Follow PROJECT_HANDOFF.md and
CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_VM.md for that next step.
The full restoration and seven-family completion goal remains incomplete.

The exact preceding dated benchmark report is preserved below and in
outputs/cctv_dgp_current_qmul_native_comparison_v1_closure/before_docs/.

---

# CCTV restoration benchmark — current status 5 October 2026

The source extension and matched ChokePoint development comparison are complete
and independently audited. All six comparison sheets are reviewed. The retained
own DGP remains unqualified for useful native restoration; the pretrained
CodeFormer comparison is clearer but has unverifiable fine detail. No new
training package, threshold fit or checkpoint adoption. Original QMUL24/32 and
new ChokePoint12/13 labeled identity roles remain separate; both reserved sets
remain unviewed. All current source/terms/resolution/split/overlap/review details:
[CCTV_NATIVE_SOURCE_EXTENSION_V1.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_NATIVE_SOURCE_EXTENSION_V1.md>).

The following 3 October report and its executable checklist are historical.
Its original bytes are preserved at
`outputs/dgp_processing_diagnostics_milestone_v3/before_docs/CCTV_BENCHMARK_STATUS.md`
and the separately pinned QMUL extension provenance snapshot. Do not rerun
its closed pilots. Follow the latest `PROJECT_HANDOFF.md` for current work;
VM execution remains verified-transfer-files/pasteable-commands only.

---

# CCTV restoration benchmark — 3 October 2026

The first public native-CCTV archive is acquired and its release structure is
independently audited. A native development subset and separate reserved subset
are now frozen; the first DGP/CodeFormer comparison and processing diagnostics are
complete, including the three-case aligned common-frame comparison. A separate
40-case paired camera-stress regression is also audited in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_PAIRED_RESTORATION_RESULTS.md`
↔ `~/forensic-dgp/CCTV_PAIRED_RESTORATION_RESULTS.md` after transfer.
It shows DGP smoothing/low-light/structure failures. The1,152-reference inherited
candidate pool is preserved; a deduplicated, geometrically qualified and input-
reviewed derived manifest now supports 902 source-balanced training references
and 110 validation references. The matched VM pilot archive is independently
verified; actual CUDA preflight/training and returned output are pending. Commands
are in `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_VM.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_VM.md` after extraction.
No checkpoint is selected. See
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_NATIVE_RESTORATION_RESULTS.md`
↔ `~/forensic-dgp/CCTV_NATIVE_RESTORATION_RESULTS.md` after transfer.
These are development findings, not independent final or Zamboanga validation.

| Artifact | Windows local | Linux counterpart after explicit transfer |
| --- | --- | --- |
| Original archive and LF checksum | `C:\xampp\htdocs\YEAR 4\Testing\dataset\cctv_survface_raw\QMUL-SurvFace-v1.zip` / `.zip.sha256` | `~/forensic-dgp/dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip` / `.zip.sha256` |
| Acquisition receipt and ZIP inventory | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_survface_acquisition_v1\` | `~/forensic-dgp/outputs/cctv_survface_acquisition_v1/` |
| Independent release audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_survface_structure_audit_v1\verification.json` | `~/forensic-dgp/outputs/cctv_survface_structure_audit_v1/verification.json` |
| Acquisition/audit runners | `C:\xampp\htdocs\YEAR 4\Testing\scripts\acquire_survface_benchmark.py` / `audit_survface_acquisition.py` | Same paths under `~/forensic-dgp/scripts/` |
| Current scope | `C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md` | `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` |

## Source choice and limits

[QMUL-SurvFace's official page](https://qmul-survface.github.io/) provides native
low-resolution surveillance faces and a public research download. It reserves
copyright to the original data owners. Its recognition benchmark is not itself
a paired clean/degraded restoration dataset. The release was acquired directly
through the author's linked Google Drive file, without a third-party mirror.

The [dataset paper, section 3 and Table 4](https://arxiv.org/html/1804.09691v6)
lists 17 source datasets across several countries, including China and Japan.
This makes it a useful first candidate under the user's Asian-source preference.
The paper describes resolution averaging 24×20 pixels. The downloaded metadata
does not supply per-image source-country/ethnicity labels, so do not infer such
labels from appearance, numerical IDs or aggregate collection locations.

[SCface](https://www.scface.org/) is another native surveillance/reference-photo
candidate. Its access requires an institutional request and a release agreement
signed by a full-time staff member. Its documented participants are Caucasian;
it would be a separately reported broader benchmark, not an Asian-population
proxy. No request, agreement or message was submitted, and SCface is not downloaded.

## Actual downloaded V1 inventory

The ZIP is 408,164,983 bytes. Acquisition plus central-directory inventory took
48.26 seconds. SHA256:
`2fbb0876bc4761217c6de5576905e2524b8ca50ad7905720a5b4b378a0ff8e13`.
This is an observed acquisition fingerprint, not a publisher-supplied checksum.

| Release role | Observed image count | Meaning |
| --- | ---: | --- |
| Training folder | 220,888 | 5,319 labeled global person IDs; no optimization performed |
| Identification gallery | 60,294 | 3,000 person IDs in actual MATLAB metadata |
| Mated probe | 60,423 | Matching gallery person IDs; another native capture, not a clean pixel target |
| Unmated probe | 121,736 | Includes 28,476 `distractors_*.jpg` files without global person labels |
| Verification images | 10,051 | Byte-identical copies from the canonical test folders, not extra independent cases |

There are 473,403 regular ZIP members and 473,392 JPEG entries. After excluding
the copied verification images there are 463,341 canonical image filenames.
The auditor reads/checks CRCs for release metadata and both copies of all 10,051
verification images. Gallery/probe metadata membership is exact, the recognition
labels agree with global person filenames, and all 5,320 positive/5,320 negative
verification pairs agree with the provided filename-based person labels.

The 5,319 labeled training IDs are disjoint from the 5,319 labeled test IDs.
The unlabeled distractors do not permit a complete identity-overlap claim; neither
does this check establish non-overlap with historical DGP/pretrained model data.
Use labeled identities when constructing an identity-disjoint restoration review.

Preserve release discrepancies:

1. The [published split table](https://qmul-survface.github.io/protocols.html)
   lists 220,890 training and 242,617 test images. The actual V1 archive has two
   fewer training images and 164 fewer canonical test images: 166 fewer in total.
   The exact reason has not been established.
2. The bundled README describes 5,319 gallery IDs, while the actual MATLAB labels
   contain 3,000. The official protocol page describes a 3,000-person watch list.
   Use actual versioned metadata; do not silently rewrite the downloaded README.
3. Not every distractor filename supplies a person ID. An initial audit assumption
   that every filename began with a numeric ID was rejected; the successful audit
   retains the unlabeled category instead of inventing identities.

The independent verification SHA256 is
`cf9124983a815776a787eeae577bbbbd173892be2f69dc33c354626641863222`.
The completed receipt and acquisition artifacts are preserved. Both runners refuse
to overwrite a completed/partial run; do not rerun them as a way to replace evidence.

During acquisition and release-structure auditing, no archive code was executed,
no original image was decoded or extracted, no model inference ran and no optimizer
update occurred. Later bounded development work is recorded in the results report.
The full archive's image
CRC/decode quality has not been audited. This is a source/structure check, not
reproduction of the original recognition challenge or a frozen restoration test.

## Historical next execution — 3 October 2026; closed commands

1. Upload the verified `cctv-dgp-vm-bundle.tar.gz` and LF checksum using the commands
   in `CCTV_DGP_VM.md`; preserve the immutable candidate/gate versions and split.
2. Execute the guarded CUDA preflight and matched 452-update pilot on the existing
   L4 VM. Setup compatibility is unverified until that VM check succeeds.
3. Independently audit the return and compare qualified checkpoints on the 24-case
   native development gallery. Native CCTV has no aligned clean pixel targets.
   Keep Phase 3 retained until candidate and visual output-review guards pass.
4. Integrate the validated primary DGP route with the existing app and supporting
   completion, including the insufficient-information request.
5. Reserve final evaluation for the frozen candidate and independent reviewers.
   The 32 reserved crops remain unviewed, with only header/hash checks performed.

No actual Zamboanga CCTV samples are available yet. Public source results remain
separate from future local validation. Asian capture location is not proof of
Philippine representativeness or an individual person's ethnicity.
