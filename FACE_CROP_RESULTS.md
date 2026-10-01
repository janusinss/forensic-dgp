# Fixed face-crop result — 2 October 2026

The predeclared crop rule failed. Do not add a crop sweep or evaluate this failed
rule on held-out images. More real reflection examples need native mask review
before another VM training recipe is justified. No checkpoint is promoted.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; audited source states originated in
`~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_vm/`.
This diagnostic ran locally on CPU with frozen states, zero optimizer updates
and zero held-out forward passes. Actual training remains VM-only.

## Executed rule and independent verification

The unchanged specification is `FACE_CROP_DIAGNOSTIC.md`. The protocol froze all
73 real training examples and 280 reflection fixtures, source/final checkpoints,
code and inputs before execution. The rule used one input-only SCRFD face box,
confidence 0.6, square context 1.25 and uniform scaling to 256 pixels. Original
predictions remain outside the ROI; missing/unreliable/no-zoom cases fall back.

Of 353 cases, 171 received crops, 181 had no reliable zoom and one had no valid
single face. Each frozen model performed 171 crop forwards. The independent
saved-mask audit verifies all 706 new masks, original mask hashes, 353 image/
target/affine mappings, exact fallback masks, unchanged outside-ROI pixels and
logged whole-image, valid-support and lens-only metrics. It performs no inference.
The original executed runner checked unchanged model tensors; the saved-mask
auditor does not independently rerun that state check.

Two saved-mask counterexamples pass. The earlier eight geometry/inference and
three protocol contracts passed before execution. Tool tests are separate from
model-quality evidence. The initial sandbox dependency read failed before outputs;
the authorized CPU rerun completed without reinstalling dependencies. SCRFD emitted
static output-shape metadata warnings at 256 input. Their cause was not independently
isolated; do not treat completed inference as proof that all boxes are accurate.

## Original-coordinate results

| Frozen state / input | Real training IoU | Real visible FP | Fixture IoU | Lens recovery |
| --- | ---: | ---: | ---: | ---: |
| Source 30 / ordinary | 0.91719 | 0.001791 | 0.18159 | 1,208 / 2,031 (59.48%) |
| Source 30 / crop | 0.89769 | 0.001967 | 0.17387 | 940 / 2,031 (46.28%) |
| Reflective 42 / ordinary | 0.92536 | 0.002081 | 0.80937 | 1,369 / 2,031 (67.41%) |
| Reflective 42 / crop | 0.90770 | 0.002184 | 0.78964 | 1,018 / 2,031 (50.12%) |

All 26 real and 56 fixture clear examples remain empty. The final reflective
state was the predeclared advancement model. It failed lens gain, whole real IoU,
visible-FP retention and fixture IoU; clear-case checks passed. Source 30 was an
ancestry diagnostic, never an alternative selected after the result.

Lens counts include only V3-minus-V2 additions on training cases 15 and 46; mouth
mask recovery cannot count as lens recovery. Final crop lens recovery is 730/1,345
and 288/686 respectively. Their input-derived crop sides are 243 and 248 pixels,
so enlargement is only about 1.054 and 1.032. No native detail is created.

The fixed six-row preview was inspected: both lens rows plus four clear controls.
Clear predictions remain empty; final crop predictions lose lens coverage, while
the medical-mask region is mostly retained. This is assistant review, not expert
annotation. The result rejects this one rule; it does not exclude every possible
face-scale method. No independent generalization or completion gain is established.

## Preserved artifacts

Local protocol:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face_crop_protocol_v1\protocol.json`.
Local result directory:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face_crop_diagnostic_v1\`.
It contains `results.json`, `cases.json`, 706 masks, `preview.png` and independent
`verification.json`. There is no new VM counterpart or upload for these CPU artifacts.

| Artifact | SHA256 |
| --- | --- |
| Protocol | `de64663b607f34f65f105029d4862d29cc78e799c8b993ff10ad36d9280c31a3` |
| Executed result | `cc3ee1f39690a7cd79c5c6a2cfdfdb380e49b21e90641ed95c08c7b201554585` |
| Cases | `9bc744ac40dd906ba803b59ea6a28e85b7961e13af9113481cb68c46d1287f31` |
| Independent verification | `c670613b4830daa150f716ef35dad37ac2802f57237abeb285906f6365116261` |
| Preview | `f6ca32b86d7ddef612d4c8477198b30cade59bdddb894d8d43267f1803c4d096` |

Auditor: `scripts/audit_face_crop_diagnostic.py`, SHA256
`1553da4e29033969e4c17decd929a7ffc12e08cd9717e16f231f1fc18e3676cf`.
It refuses to overwrite completed verification. Do not rerun completed inference
over preserved outputs. Executed code/protocol/inputs stay unchanged.

Next: qualify additional real reflections in a separate training-only source
review, exclude all current reviewed/held-out/benchmark sources, and inspect native
covering boundaries before generating proposed masks. Existing eyewear queue
contains candidates and clear-control proposals, not approved training labels.
The subsequent qualification is now complete: 24 sources remain after excluding
three already used; fresh 4,210-reference screening has no overlap flags. Native
mask and clear-label approval remain pending; see `REAL_REFLECTION_REVIEW.md`.
Retention repair also requires a distinct hypothesis. Original real/synthetic
selection gates and generator/application baselines remain unchanged. No new
training recipe, commit or push occurred. Hidden features remain plausible estimates.
