# COFW covering-data preparation — 3 October 2026

The new detector registry has 165 records: 133 training, 25 validation and seven
previously inspected test records. It adds 42 reviewed COFW training sources and
preserves every previous record and its active image/mask/support bytes. Source,
geometry and dataset audits pass. No model was trained, selected or changed in
the application. The next VM experiment is not packaged or ready to run yet.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Intended Linux repository root: `~/forensic-dgp/` after explicit transfer.
Paths below are relative to these roots. New COFW artifacts are local only;
`git pull` does not transfer ignored datasets or checkpoints.

## Why these examples were added

The completed camera diagnostic improved degraded hand detection but missed
obstructing hair, struggled with scarves and left covering remnants in generated
estimates. Retention/clear checks failed; no checkpoint qualified. See
`REAL_CAMERA_RESULTS.md`. These added detector labels address real covering
coverage. They do not supply the true hidden face or justify generator training.

## Verified source and terms

The official CaltechDATA record is DOI `10.22002/D1.20099`, credited to Xavier
Burgos-Artizzu, Pietro Perona and Piotr Dollar. Current metadata declares public
access and CC BY 4.0. Keep attribution, citation and modification history with
derived assets. [Official record](https://data.caltech.edu/records/bc0bf-nc666)

The creator paper describes 1,345 training faces (845 LFPW plus 500 COFW),
507 test faces, 29 landmark positions and occlusion flags. Sparse flags are not
pixel removal masks or paired uncovered-face targets.
[Creator paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/12/BurgosArtizzuICCV13rcpr.pdf)

The official color archive is verified: 503,327,162 bytes, publisher MD5
`8b21d126c4e1fb307cb463578eef0511`, observed SHA256
`bc6a79bda1bd88705082af9ebdd31188cf5841917b09534b25067e424a0130e5`.
Eight finite GET ranges completed in 27.55 seconds; ZIP CRC and publisher
checksum checks passed. Archive: `outputs/cofw_source_acquisition_v3/COFW_color.zip`.
Current terms snapshot: `outputs/cofw_source_acquisition_v3/current_record.json`,
SHA256 `3c0c1b317b7b14272452a0161347cc3fd5c3950d9b1aed27a917fa0a293c256c`.

The earlier 2,153,540-byte `outputs/cofw_source_research_v1/COFW_color.zip.partial`
and incomplete `outputs/cofw_training_sources_v1/` remain excluded historical
artifacts. The verified archive supersedes the earlier acquisition hold.
Only the two declared MATLAB/HDF5 members were extracted under
`outputs/cofw_source_review_v1/`. All 1,345 original training images were
independently decoded: 1,301 RGB and 44 genuinely grayscale images. Grayscale
channels were repeated without inventing color. Test shape metadata confirms
507 entries; no test RGB was decoded, inspected or admitted.

## Native review and annotations

All four source sheets, seven native coordinate sheets and seven final overlay
sheets were inspected. Forty-two author-training sources receive assistant
pilot detector labels: 31 coverings and 11 clear controls (eight uncovered faces
and three clear-glasses controls).

| New primary covering group | Sources | Detail |
| --- | ---: | --- |
| Hands | 9 | Facial overlaps only |
| Obstructing hair | 7 | Ordinary hairstyle remains |
| Cloth/scarf or uncertain cloth/object | 6 | Five cloth/scarf plus one mixed case |
| Other objects | 4 | Facial overlaps, excluding background objects |
| Eyewear/masks | 5 | Three sunglasses, respirator with clear goggles, costume mask |

Secondary tags describe overlapping coverings, not 42 proven independent people.
Additional roll in some training images does not extend the application's
claimed frontal/mild-turn operating scope.

Source polygons were drawn from native RGB and uniformly projected to 256px;
full removal proposals are recorded separately. Native one-pixel boundary bands,
crop edges, padding and ambiguous regions are unsupervised. Their observed RGB
remains in the input. Costume-mask eye openings and clear goggles are retained.
The existing inference removal margin is separate from these training targets.

Versioned refinements tightened a hand/cheek boundary and sunglasses rim,
extended observed hair strands and handled a crop-edge respirator. The first
independent audit found 12 native pixels with Pillow/OpenCV raster disagreement;
that failed report is preserved. V3 declares exactly those pixels unsupervised
before training. Source RGB/full proposals and old held-out support are unchanged.
The final independent audit finds zero supervised raster disagreements.
This verifies geometry and support, not independent expert pixel accuracy,
hidden identity or successful covering detection. Nearly hidden rejection remains
an unresolved model limitation.

## Dataset membership and evidence

Registry: `dataset/detector_supported_review_v3/manifest.json`, SHA256
`abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd`.
Parent V2 SHA256:
`3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e`.

| Split | Covered | Clear | Total |
| --- | ---: | ---: | ---: |
| Training | 89 | 44 | 133 |
| Validation | 15 | 10 | 25 |
| Previously inspected test | 4 | 3 | 7 |

All 123 previous records and bytes are preserved. New sources form one
conservative training-only author-source cohort. The existing validation
mannequin and its full support remain unchanged. Historical splits/gates and
exposed-gallery status are intact. The other 1,303 training images and the
507 test images are not admitted automatically.

Exact native/full-crop RGB comparisons find no overlap with 290 declared prior
source/crop files. This is not a full 80,000-source comparison, perceptual
duplicate search, identity-disjointness or pretraining-overlap proof.

| Evidence | SHA256 |
| --- | --- |
| `outputs/cofw_covering_proposals_v3/manifest.json` | `e180481c516a6025a162d5c3e135ff41212f9a120a628e7ccf8ec5b6029a8733` |
| `outputs/cofw_covering_data_validation_v3/verification.json` | `e16625fa2f66d51c33a8ee5481dec821648d531522a93d4f54236e187938faf1` |
| `outputs/cofw_supported_dataset_validation_v1/verification.json` | `a403f14457d7ba850ff2dbea4236b1db88276cd9e5d4f8192a65e0a0159261cc` |
| Preserved failed `outputs/cofw_covering_data_validation_v1/verification.json` | `127f186bb724c4418cd83d6a834dc729531b97d7bcf86068156c50df2c989476` |

Final audits verify all 1,345 native sources, 42 selected native pairs, 210
proposal assets and 493 registered data files. The unchanged three-tensor
reader loads the registry successfully. Zero model forwards/optimizer updates.
Previous raw-source provenance paths remain external to the V3 folder; their
dependency closure is listed in the dataset audit. A future bundle must include
or verify them. Copying V3 alone is not a complete provenance transfer.

## Verified paired camera inputs

`outputs/cofw_camera_pairs_v2/manifest.json` contains 266 views: 133 native and
133 degraded training inputs. Manifest SHA256:
`b6f4a800427e96a52468a0ec10eec478ec8f5875e1302b35ffd6c072dfec041c`.
Independent audit `outputs/cofw_camera_pairs_v2/independent_verification.json`
SHA256 `c7fb19ff5d936c5e43f8a5c3a56b8ca59db40986d04f3be4c156660258652471`
reconstructs every degraded pixel without calling the preparation helper.
The previous 91 input pairs and camera parameters remain exactly identical;
42 new COFW pairs preserve source geometry, target masks and supervision support.
True padding stays neutral and is excluded from filtering. No held-out input is
included and no model or optimizer runs.

The deterministic transform uses observed-ROI blur, reduced resolution, RGB
noise and JPEG, with the unchanged `real_camera_pairs.py` seed/parameter policy.
It is a fixed diagnostic perturbation, not a calibrated camera distribution.
Two five-row sheets (`preview_1.png`, `preview_2.png`) are inspected and recorded
in `visual_review.json`: hand/hair/cloth/object labels remain fixed, costume-mask
eye openings and clear goggles remain visible, and the clear-glasses control
has no positive label. This is input review, not learned model quality.

The incomplete `outputs/cofw_camera_pairs_v1/` is preserved with an interruption
record. Its first attempt stopped because legacy rows lack `source_id`; V2 uses
stable training indices for those identifiers and retains the original source
hash/group bindings. No incomplete cache is consumed.

## Next VM experiment

The paired native/camera inputs are verified. Freeze a finite matched comparison of existing-data
control versus varied-covering training, including clear/retention counterexamples.
Record family sampling and matching initialization/optimizer/update budgets.
All actual training runs on the user's L4 VM after package/preflight checks.
Keep the original 425-case qualification gates and held-out checks; the practical
gallery remains exposed development evidence. Inspect full removal footprints
and generated estimates before selection. No `best.pth`, app checkpoint swap or
generator/restoration retraining has occurred. Goal active.
