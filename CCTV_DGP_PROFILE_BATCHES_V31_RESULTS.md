# V31: audited early structure stop

V31 ran 50 of its maximum 800 optimizer updates on the existing NVIDIA L4 and
stopped at the prescribed structure check. Its **0.694525%** improvement was
below the unchanged **1%** early requirement. The traceback is the deliberate
stop working as specified. Export completed successfully; export completion
does not mean that training passed. Preserve the stopped checkpoint and failure.
Do not resume V31, rerun it unchanged or promote it into the application.

The returned archive is 1,252,002,625 bytes, SHA256
`cbb89bb5fb85ca7e8b77203188164ab2472342f8d7f3b3ecd7a07f6b53c6f49b`.
It matches the original VM checksum and export receipt. All **27,623** regular
allowlisted returned files and **5,467** approved TRAIN assets were verified.
The independent local audit used trusted prospective code, retained the source,
gradients, failure and checkpoint hashes, and performed no local gradients,
backwards or optimizer updates.

The printed `feature_error` measures delivered-PNG high-frequency error in the
fixed landmark regions against paired TRAIN targets. It is not PSNR, a native
CCTV ground-truth score or proof of recovered identity.

| Evidence | Audited result |
| --- | ---: |
| Baseline degraded TRAIN feature error | 0.001876217941608562 |
| V31 stopped50 feature error | 0.001863187133907816 |
| V31 gain / early requirement | 0.694525% / 1% — failed |
| V30 gain on the identical baseline | 0.805717% — failed |
| Delivered preservation groups at V31 stopped50 | All 17 pass |

The preservation checks compare observed MSE, SSIM and fixed recognizer similarity
with the retained original DGP. Their passing group averages do not waive the
structure stop or certify every individual feature. These are paired synthetic
photographic TRAIN observations; they remain separate from unpaired native CCTV
and any reserved final evaluation. Source names do not establish ethnicity.

| Degraded source group | Cases | V31 feature-error reduction |
| --- | ---: | ---: |
| dataset/asian_faces | 1,560 | 0.810400% |
| dataset/thumbnails128x128 | 1,564 | 0.667178% |

The audit independently recomputed 7,810 snapshot rows, including delivered PNG
metrics, saved vectors and mean-only controls. CPU inference replay covered the
initial 50 proof cases and 50 fixed previews at each completed snapshot: 150
original-DGP forwards, 100 candidate forwards and 250 recognizer forwards.
Maximum raw preview difference was 0.0000022054 against the 0.00001 limit; PNG
disagreement was at most one byte; maximum preview vector difference was
0.0000002794 against the 0.00005 limit. It did not replay all 3,905 stopped neural
outputs. All 38,394,279 saved component-gradient values were checked without
local autograd. The checker completed in 988.116 seconds.

VM worker time was 977.422 seconds, about 16.3 minutes. Fifty steps represent
0.0640 of the 781-reference epoch; no full epoch or 800-update result exists.
Peak allocated VRAM was 8.30 GiB. The export took 71.531 seconds. The saved
terminal maintenance observation at 2026-10-07T09:30:56.542780+00:00 showed no V31
worker, an idle GPU and about 36.68 GiB free. This is a dated observation, not a
guarantee of current free space. No VM workload was started, stopped or changed
by the agent during this return audit.

All 50 fixed cases were viewed in ten original-detail sheets with five exact
256-pixel columns: input, retained DGP, V30 stopped50, V31 stopped50 and paired
TRAIN target. Eyes, nose, mouth, outline and visible appearance were inspected
together. Clear glasses, hairstyles, expressions and adjacent hands remain in
the review. Clear inputs become softer. Degraded inputs have smoother block
edges, but eyelids, nostrils, lips and hair often remain diffuse. Motion examples
retain more broad usable structure than the severe compound examples. V31 shows
little visible incremental clarity over V30; useful whole-face improvement has
not been established. The severe compound cases call for a clearer crop.
This agent review is not the required independent human final review.

| V31 TRAIN diagnostic bucket | Degraded cases | Feature-error reduction |
| --- | ---: | ---: |
| Entire TRAIN corpus | 3,124 | 0.694525% |
| Selected by the first 50 updates | 200 | 0.636665% |
| Not yet selected by update50 | 2,924 | 0.699337% |
| Fixed preview subset | 40 | 0.743707% |

The first 50 paired batches covered 50 references, 250 cases and 50 clear controls.
None of the fixed 50 preview cases had been optimized by that point; they still
belong to TRAIN, not held-out evaluation. V30's first 50 batches touched 218
references and 47 clear controls. The paired-batch intervention did not qualify
under the retained check. Different reference coverage at this stop prevents
claiming that pairing is generally inferior or that sampling uniquely caused
the failure.

The next architecture review concerns the original DGP spatial feature path.
The source and safe tensor-state review confirms that all FPN state entries
stayed identical while all 12 active decoder tensors changed. Five lateral
convolutions and three top-down convolutions contain **11 fusion tensors /
479,616 parameters**. This is an untested partition, not evidence of a broken
gradient path or a qualified model. A distinct finite, zero-update VM diagnostic
should measure its connectivity and improvement/preservation tradeoffs before
selecting a new training recipe. Preserve the MobileNet backbone, evaluation
normalization, original checkpoint and inactive-head finding. Do not weaken the
preservation losses or the existing 1%/10% gates. No new training packet has
been released by this milestone.

A separate completion review audited five official MAT source/license files.
No MAT checkpoint was downloaded or run; no completion model or app setting
changed. Checkpoint provenance, dependencies, local parity and useful outputs
remain unverified. A different completion prior cannot remove an object copied
outside the reviewed removal area. Automatic and assisted results still need
separate review across all seven covering families.

The application's own trained DGP remains primary. Original checkpoints, splits,
native/evaluation boundaries, gate failures and actual Windows research-cache
backup remain preserved. Useful native restoration, whole covering-family
quality and independent final review remain outstanding. Reserved final pixels
were not opened. No real Zamboanga CCTV samples or local-performance claims
exist. The full goal remains active and incomplete.

[Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_profile_batches_v31_independent_audit.json>)
 · [Exact comparison preparation audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_profile_batches_v31_failure_review_v1/independent_preparation_audit.json>)
 · [All 50 visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_profile_batches_v31_failure_review_v1/visual_review.json>)
 · [Feature-path state review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_v31_feature_path_review_v1/review.json>)
 · [Separate completion source review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_SOURCE_REVIEW_V1.md>)
