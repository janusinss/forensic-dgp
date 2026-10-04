# Genuine HQ target preparation: 4 October 2026

All 510 higher-resolution counterparts and their 256×256 targets are acquired,
independently audited and reviewed for source quality. The frozen review accepts
444 candidates (391 training-role, 53 validation-role), excludes 60 covered
references and excludes six unsuitable references. The initial 16 sample
decisions remain unchanged. This is assistant development review, not final
independent human assessment.
This is data preparation, not a trained model improvement or a release decision.
The full DGP-first Goal remains active.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing L4 VM root: `~/forensic-dgp/`; historical CCTV bundle:
`~/forensic-dgp/cctv_dgp_vm_bundle/`.
All new acquisition/auditing here runs locally with zero model forwards,
backward calls or optimizer updates. The latest 4 October user instruction restores
VM-only training after CLI start access was explained; it supersedes the brief
small-local-pilot permission. See `SYSTEM_WORKFLOW_AND_GOAL.md`. V6 completed 196
updates on the L4 in 259.18 seconds, with independent return audit/preview review
complete and no epoch selected. The separate V7 ten-pair training-fit test also
failed its declared criterion. Both returns are local and audited; the VM has
returned to its initial stopped state. The full Goal remains active.
Preparation/source quality does not establish model usefulness.
The intended VM copy of this report is `~/forensic-dgp/CCTV_DGP_HQ_TARGETS_STATUS.md`
after document transfer. The 444 canonical targets are now verified on the VM
inside `~/forensic-dgp/cctv_dgp_targets_vm_v6/`; the original 1024 sources remain
locally preserved.

## Why the target audit follows V5

The V5 audit and preview review are complete. No trained epoch qualified, and
both best files retain the V2 starting tensors. Its reference-header audit found
896/902 training originals smaller than 256 in both axes, including every FFHQ
image at 128×128. Reconstruction targets at 256 therefore contain enlarged
low-resolution detail. This is a measured limitation, not proof of failure cause.
See `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_CONFLICT_RESULTS.md`
↔ intended VM `~/forensic-dgp/CCTV_DGP_CONFLICT_RESULTS.md` after transfer.

The [official NVIDIA FFHQ release](https://github.com/NVlabs/ffhq-dataset) links
aligned 1024×1024 images and 128×128 thumbnails through per-image metadata.
The [official download specification](https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/download_ffhq.py)
pins its metadata file size and MD5. We verified those bytes and matched the
current thumbnail pixels to the official image IDs before fetching counterparts.
Per-image author, photo URL and license metadata are retained. The release's
noncommercial terms and restriction on facial-recognition development apply;
the project task remains restoration, with embedding similarity as a limited
research proxy. No ethnicity or Zamboanga performance is inferred from FFHQ.

## Completed bounded sample

The source audit selected 10 training-role and six validation-role references
at sorted image-ID quantiles before HQ inspection. It retained the original
historical roles, including where the official FFHQ category differs. The sample
is a data diagnostic, not a new training split or independent final test set.

Acquisition: 290,338,584 new bytes, including metadata, in 67.69 seconds.
Declared cap: 600 seconds and 320 MiB. All 16 sources passed official file-size,
file-MD5 and decoded-pixel checksums. Targets are genuine source images downsampled
to 256×256 with PIL RGB/LANCZOS; no learned sharpening produced these references.

The independent audit rebuilt all 16 targets and checked all 48 image cells in
the four preview pages against their actual files. It confirmed original roles,
thumbnail identity through pixel hashes, provenance/specifications and no exact
image or source-photo overlap between the sampled roles. This does not establish
full identity disjointness or remove earlier training exposure.

| Evidence | Windows local | VM copy if later transferred |
| --- | --- | --- |
| Sample sources, targets, result and audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_counterparts_v1\` | `~/forensic-dgp/outputs/cctv_dgp_hq_counterparts_v1/` |
| Official metadata cache | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_metadata_v1\ffhq-dataset-v2.json` | `~/forensic-dgp/outputs/cctv_dgp_hq_metadata_v1/ffhq-dataset-v2.json` only if needed |
| Full-cohort acquisition plan, rubric and sample review | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_cohort_plan_v2\` | `~/forensic-dgp/outputs/cctv_dgp_hq_cohort_plan_v2/` |
| Audited full source catalog | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_cohort_v2\` | `~/forensic-dgp/outputs/cctv_dgp_hq_cohort_v2/` |
| Frozen full source-quality review and candidate manifest | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_source_review_v3\` | `~/forensic-dgp/outputs/cctv_dgp_hq_source_review_v3/` if transferred |

Sample manifest SHA256:
`b4e5c357efd35c899b28c10ea18da229bed785732426208b6b0c18e29bac4375`.
Sample independent receipt SHA256:
`3d890f16554511b153fadadb65a0fe24918398815bdc18425fd0a1c4e23621b9`.
Sample source-review SHA256:
`681eaa46641bd5529a59577f05f6287aa0299051f4d6f9c785aed6084be25ff0`.

The four reviewed pages show finer edges and texture while preserving the
thumbnail's face geometry. This compares reference quality; it is not a DGP
output comparison. The old-target and new-target resize kernels also differ,
so this evidence alone does not isolate a training benefit from resolution.

Two training-role candidates are excluded as uncovered references in a future
version: `tr_ffhq_25564` has sunglasses hiding both eyes; `tr_ffhq_49766` has
windblown hair across facial features. Their source images and historical roles
are preserved. `tr_ffhq_00084` was inspected at 1024: its small lens reflections
leave the irises visible, so ordinary clear glasses are retained. Natural eye
state, expression, skin texture, facial hair and face-framing clothing are not
normalized to an invented appearance.

## Full-cohort acquisition and source-only review

Preparation bound all 510 existing FFHQ references to the official thumbnail
pixels: 451 original training-role and 59 validation-role images. All original
roles and historical evidence are retained. Sixteen sources are reused from the
audited sample; the remaining 494 have now been acquired. Official declared sizes
total 695,837,473 image bytes; the added acquisition is 673,292,731 bytes (642 MiB).
The verified metadata is reused rather than downloaded again.

The source-only acquisition used four workers, a 1 GiB new-data ceiling and
a 1,800-second cap. A network read timeout stopped its first execution after
507 complete source files; that failure receipt remains preserved. An independent
cache preflight verified all 507 files and identified only three missing files.
A bounded recovery reused the cache and fetched 4,296,660 bytes (4.1 MiB),
finishing the catalog in 42.18 seconds under a 300-second/12-MiB recovery cap.
All acquisition/recovery/audit processes are now terminal; no job remains live.

The whole-cohort independent audit completed in 61.84 seconds. It checked all
510 thumbnail bindings, official source file/pixel checksums, derived targets
and preview cells across 32 source-only pages. Original train/validation roles
are 451/59, with no exact source-image or source-photo URL overlap between roles.
Full identity overlap across the earlier 80,000 images remains unestablished.
Published original-photo alignment geometry has a minimum edge of 1,024.70
pixels and median 1,391.46 pixels; this describes metadata geometry rather than
measured camera detail. Source-quality review is now complete in a separate
versioned ledger; the acquisition/audit snapshots retain their original pending
review fields. `training_ready` remains false until a finite VM protocol exists.

Full results SHA256:
`7388a67b655d8259db8a30cefcdfa5b520597e80fd71bfd1d5952051fac98370`.
Whole-cohort independent receipt SHA256:
`d4989390d2c69bcd3c3abb8169aa1c02b4ad6a50aed3c5575b544aa93e96c0c0`.
Recovery completion receipt SHA256:
`70a77da100255650a19ce350f555f144e4c001875034fcc38dd17b09601fd7a8`.

Plan SHA256:
`ad65e35c70104b72a38dfe3cdac1f1027029445313f9d5530dbb18af7cde9501`.
Source-review rubric SHA256:
`1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e`.

The rubric is fixed before reviewing the remaining sources or new model outputs.
Review all candidates for visible facial features, coverings, severe original
blur/darkness, watermarks, framing and source detail. Clear glasses and
non-obstructing hair remain valid. Exclude or flag uncertainty based on the
source image, not a model's validation errors. Do not move held-out references
into training. Preserve exclusions as evidence and report sources separately.
The existing Asian-source images remain unchanged; no higher-resolution Asian
counterpart or population-representation claim has been established.

## Frozen whole-cohort source-quality review

All 32 catalog pages were inspected at their original resolution, with additional
1024×1024 source inspections for all 38 ambiguous cases. Decisions used the
fixed source rubric without model outputs or validation-error filtering.
Transparent lenses with visible eye structure, non-obstructing hair and natural
expressions remain valid. Covered features, strong feature-obscuring reflections,
severe source blur and an artificial doll reference are recorded as exclusions.
All excluded source files and historical roles remain preserved.

| Decision | Training role | Validation role | Total |
| --- | ---: | ---: | ---: |
| Accepted clean-restoration candidate | 391 | 53 | 444 |
| Covering exclusion | 56 | 4 | 60 |
| Reference-quality exclusion | 4 | 2 | 6 |

`C:\xampp\htdocs\YEAR 4\Testing\scripts\freeze_cctv_hq_source_review_v3.py`
↔ intended VM `~/forensic-dgp/scripts/freeze_cctv_hq_source_review_v3.py`
verified every page/ID/role, all 510 actual source/target SHA256 pairs, all 38
full-resolution follow-up decisions and agreement with the original 16 sample
decisions. It completed in 1.14 seconds with zero model operations. Its integrity
check cannot independently judge the assistant's visual decisions or establish
useful model output.

Frozen decisions SHA256:
`706cc7531066d1f9c859fc8abeaaea185877dd953a846578e09bf05f02a8cd86`.
Eligible-reference manifest SHA256:
`cadeedd2906fe169b7cc9741aac9fc29db8aedc1ba21d4438f97443490c1281e`.
Source-review integrity receipt SHA256:
`d3fb788eb775d2c178179e283b7a353653473ed088b2a19141ec02e6467078d4`.
These are local data-preparation artifacts; no new bundle has reached the VM.

## Commands and verification evidence

The assistant ran these local scripts; the user does not need to repeat them:

```powershell
cd "C:\xampp\htdocs\YEAR 4\Testing"
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_cctv_ffhq_counterparts_v1.py -v
.\venv\Scripts\python.exe -X utf8 -u scripts/acquire_cctv_ffhq_counterparts_v1.py --prepare-only
.\venv\Scripts\python.exe -X utf8 -u scripts/acquire_cctv_ffhq_counterparts_v1.py --ca-file scratch/gcloud-windows-roots-v1.pem
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_ffhq_counterparts_v1.py
.\venv\Scripts\python.exe -X utf8 -u scripts/prepare_cctv_ffhq_cohort_v2.py
.\venv\Scripts\python.exe -X utf8 -u scripts/acquire_cctv_ffhq_cohort_v2.py
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_ffhq_cohort_v2.py
.\venv\Scripts\python.exe -X utf8 -u scripts/freeze_cctv_hq_source_review_v3.py
```

Six integrity tests passed: corrupted bytes, wrong pixels/IDs, immutable role
selection, approved HTTPS hosts and ordinary public-download confirmation fields.
Six acquisition/audit/preparation sources parse as Python 3.10. The local trust
bundle enables verified TLS; no credentials or private keys were exported and
global certificate checking was not disabled.

These scripts, manifests and receipts are now execution evidence. Preserve them
when changing the recipe. A later version should be additive. Transfer commands
for a future training bundle must be prepared only after it exists and verifies;
`git pull` cannot install these uncommitted scripts or git-ignored image outputs.

**Next:** define a controlled finite VM pilot using the 444 approved genuine
targets. Freeze its target set, alignment/degradation recipe and safeguards
before evaluation; re-evaluate the starting model on the same new targets.
No new training protocol is ready or running. The retained V2 model, native
development failures, covering-family requirements, DGP-led integration and
independent final review remain part of the full Goal.
