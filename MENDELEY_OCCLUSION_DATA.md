# Uploaded occlusion source and supported detector extension

Verified 2 October 2026. Eight reviewed detector sources are admitted to a separate
dataset version. The entire uploaded archive is not admitted to training. No
generator targets, local model training or new application checkpoint result from
this data preparation.

Windows repository: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM counterpart after explicit transfer: `~/forensic-dgp/`.
The original ZIP remains at the Windows repository root and is now Git-ignored.
It is not moved or deleted. The pilot bundle transfers only the selected data.

## Publisher provenance and actual archive

The publisher lists version 1, DOI `10.17632/s57wnx78vh.1`, published 10 June 2025,
credited to Laxmi Narayan Soni and Akhilesh A Waoo, under CC BY 4.0. It describes
12,000 face/non-face images with occlusions and low illumination. Those descriptions
are publisher claims, not verified pixel masks or our measured counts.
[Publisher dataset record](https://data.mendeley.com/datasets/s57wnx78vh/1)

After three automated HTTP 403 results, the user supplied
`C:\xampp\htdocs\YEAR 4\Testing\Occluded and Low-Light Human Face Detection Datase.zip`.
The filename really ends in `Datase.zip`. Earlier failed-access evidence is
preserved under local `outputs\mendeley_occlusion_candidate_v1\access_status.json`
(VM `~/forensic-dgp/outputs/mendeley_occlusion_candidate_v1/access_status.json` if
explicitly transferred). The upload resolves that acquisition block; no further
automated publisher-access attempts were made.

| Observed property | Verified value |
| --- | --- |
| Archive size / observed SHA256 | 26,504,627 bytes / `2208dfaa4973450073883618ceb8432f4740e9b82fede290c198aa42f51752ac` |
| Members and decoded images | 11,987 JPEGs; zero CRC/decode failures; IDs 1–11,987 |
| Native dimensions | 11,979 at 92×112; eight at 184×200 |
| Exact uniqueness | 6,102 byte-distinct files; 6,101 distinct dimension-bound decoded RGB images |
| Labels / splits in ZIP | No mask, box, class, README, licence or split files; only JPEGs |

The observed 11,987 differs from the publisher's claimed 12,000. Extraction did
not drop files: every member is verified. There is no publisher checksum to match
the upload independently. Keep attribution with selected/derived assets; this
fingerprint identifies the user-supplied archive, not a publisher-authenticated
release. Exact pixel deduplication removes 5,886 repeated instances, but different
frames of a related subject are still correlated samples.

Safe path/member checks preceded extraction. The independent audit re-read all
11,987 extracted files against ZIP bytes and CRC, decoded pixels, duplicate groups,
dimensions and source-sheet hashes. Against all 115 prior supported raw sources,
there is no exact byte or native RGB match. Full 80,000-image, resized/perceptual,
identity and model-pretraining overlap are not certified.

Source research root is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\mendeley_occlusion_candidate_v1\uploaded_source_review\`;
VM counterpart is `~/forensic-dgp/outputs/mendeley_occlusion_candidate_v1/uploaded_source_review/`
only after a separate transfer. It is not copied wholesale into the pilot.
Two initial source sheets and four targeted tail/rare-resolution sheets were
inspected: 363 distinct file IDs, not all 11,987 semantic image contents. The
review finds repeated low-resolution face crops, background/non-face crops,
illustrations and related mask/sunglasses/hand/hair captures. Verified scarf or
general-object training examples are not supplied by this selected subset.

## Selected annotations and immutable history

Selected IDs are 11339 (clear), 11340/11349/11353 (hands), 11347 (obstructing hair),
11343 (opaque sunglasses), 11344 (cloth mask), 11341 (sunglasses plus mask).
All are native 92×112. The eight related captures share one train-only group;
they do not represent eight independently sampled identities. Existing validation
and test membership is untouched. There is no same-person uncovered reference
or pristine high-resolution reconstruction target.

The removal target covers overlap with facial features, preserving outer hairstyle
and hands outside that area. Hair's inferred facial overlap is approximate. Native
polygon boundaries have a ±1-pixel unsupervised band; proposed coverings touching
the bottom crop row also exclude that observed row from supervision. Unknown
observed RGB stays in the input; only true padding becomes RGB96. Every supervised
loss must use the valid map. These are assistant pilot labels, without independent
expert adjudication or a pixel-accuracy guarantee.

V1 proposals are preserved. V2 tightens the upper-hand/combined-mask footprints and
hair's visible-skin edge after native inspection. V2's first geometry audit finds
three supervised OpenCV/Pillow disagreement pixels at a clipped bottom boundary.
V3 records crop-edge uncertainty without altering source RGB or polygons. All
supervised native pixels now agree between both rasterizers; 48 proposal assets
pass source/geometry/support verification. The eight-row V3 overlay was inspected
before acceptance. This numerical agreement does not prove semantic mask accuracy.

Proposal versions are local `outputs\mendeley_mask_proposals_v1\`, `v2\`, `v3\`
under the Windows root. Intended VM counterparts are
`~/forensic-dgp/outputs/mendeley_mask_proposals_v1/`, `v2/`, `v3/` after explicit transfer.
V3 `annotation_decisions.json` binds the exact reviewed files and acceptance scope.

## Portable supported dataset and paired inputs

Registry: `C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v2\manifest.json`.
Repository VM counterpart: `~/forensic-dgp/dataset/detector_supported_review_v2/`.
The isolated pilot instead uses
`~/forensic-dgp/real_camera_vm_bundle/dataset/detector_supported_review_v2/`.

| Split | Covered | Clear | Total | Change |
| --- | ---: | ---: | ---: | --- |
| Training | 58 | 33 | 91 | Seven covering sources and one clear source added |
| Validation | 15 | 10 | 25 | Active metadata, file bytes and support unchanged |
| Previously inspected test | 4 | 3 | 7 | Active metadata, file bytes and support unchanged |

All 115 preceding records remain byte/metadata-identical in the new registry.
The known mannequin remains validation index 8, `new_covered_40.png`, with full
support. The reader still returns explicit image/mask/valid triples; legacy pair
loaders cannot consume partial labels. The independent dataset audit verifies 291
data files plus the new manifest. The full archive and unlabelled sources are
not negative-mask controls. Dataset `training_recipe_ready=false` accurately
preserves the data-stage scope; a separate frozen VM protocol is required.

Input-only camera module `C:\xampp\htdocs\YEAR 4\Testing\real_camera_pairs.py`
(VM `~/forensic-dgp/real_camera_pairs.py` after transfer) prepares 91 native/degraded
pairs. The cache is local `outputs\real_camera_pairs_v1\`; isolated VM path is
`~/forensic-dgp/real_camera_vm_bundle/outputs/real_camera_pairs_v1/`. All 182 inputs
and their original label/support bindings pass independent verification; zero
held-out inputs are in this cache. Three data-transform and two schedule contracts
pass without a model, optimizer or actual training. Eight preview rows are inspected.

Camera transforms blur, resample, add noise and JPEG-compress only the observed
rectangular source ROI; labels, support and geometry stay exact. This is a bounded
detector diagnostic inspired by established degradation modelling, not a
reproduction of the full Real-ESRGAN generator method or demonstrated detector
improvement. [Real-ESRGAN creator paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html)

## Evidence and next step

| Artifact relative to either repository root | SHA256 |
| --- | --- |
| Source integrity: `outputs/mendeley_occlusion_candidate_v1/uploaded_source_review/integrity.json` | `a3c61ca1e62c284826e968326889a639dd6f37dca2330daaed312b0047a0e493` |
| Independent source audit: same folder, `independent_verification.json` | `24c6cf16eb54c397fb7967dab83f2ad9050816c0e8a381670091ef6a957c3067` |
| V3 proposal manifest: `outputs/mendeley_mask_proposals_v3/manifest.json` | `98c1b4f6b687b312f140969766fad8891dc8c1f57c9e794f61bc6cae624bf22b` |
| V3 proposal audit: same folder, `independent_verification.json` | `cb67bcc60966b19f04c1882acb5f9497f08b66fec1a51987312f9f045d38d875` |
| Supported V2 registry: `dataset/detector_supported_review_v2/manifest.json` | `3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e` |
| Dataset audit: `outputs/mendeley_supported_dataset_validation_v1/verification.json` | `67adbbcafc34002345ea1fc2995a5291c83b797cd3123873dca1c6eb12461c41` |
| Camera cache: `outputs/real_camera_pairs_v1/manifest.json` | `872b1e0d4cb75835892d4df2b6a17fd7dafde954a076724e10426ea0e71a047c` |
| Camera audit: same folder, `independent_verification.json` | `7fd28b5c9d91322ac762f659046a56b8a44b52cd34fcf3f138a353333ac3a408` |

Next: the separately frozen three-arm diagnostic in
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_VM.md` (VM
`~/forensic-dgp/real_camera_vm_bundle/REAL_CAMERA_VM.md` after bundle transfer)
isolates native83 versus camera83, then camera91 source addition. It uses identical
starting weights/moments and replay, 112 updates per branch, 336 total. It must
run only on the L4 VM. Returning its checkpoints is followed by independent audit,
mask review and a decision about original-gate/end-to-end evaluation. Historical
gate failures and the current application detector remain unchanged.
