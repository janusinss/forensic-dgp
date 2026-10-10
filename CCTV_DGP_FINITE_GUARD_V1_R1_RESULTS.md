# Finite-guard V1 R1 return: all three proposals rejected

The manual L4 study completed its prescribed diagnosis and stopped correctly.
It accepted **zero parameter changes, zero optimizer updates and zero epochs**;
there is no new trained checkpoint. All three reset proposals fail delivered
PNG preservation in at least one of the two exposed TRAIN cohorts. Separately,
all measured structure gains remain below the unchanged 1% quality requirement.
This return does not justify another epoch run under the same recipe.

The 876,945,200-byte returned archive has SHA256
`7be2c395dfe5517a736c78ae18fc5775554318067d6f9dae861e3e07c2f56b32`.
Its sidecar and export receipt agree. Export completion means files were
packaged; it does not mean model improvement. The VM study took 256.150 seconds.
It made 320 autograd queries, calculated one common direction for 64 objectives,
and tested the three predeclared displacement fractions. Those objectives
initially predict improvement, but prediction cannot override actual outputs.
No accepted training state was committed; all original model/recognizer/buffer
state fingerprints were restored exactly.

| Proposed relative displacement | Raw structure gain: first / second TRAIN cohort | PNG structure gain: first / second TRAIN cohort | PNG preservation failures: first / second |
|---|---:|---:|---:|
| 0.000020 | 0.029153% / 0.032728% | 0.043497% / 0.039478% | 0 / 1 |
| 0.000010 | 0.015068% / 0.016774% | 0.033851% / 0.018666% | 0 / 3 |
| 0.000005 | 0.007637% / 0.008828% | 0.026661% / 0.005028% | 2 / 6 |

The table reports percentage reduction of paired landmark high-frequency error,
not identification accuracy or visible sharpness. Every raw preservation check
passes. All retained failures in these delivered outputs are fixed ArcFace
similarity regressions against the paired targets. The 12 original quality
decisions (three proposals, two cohorts, raw/PNG) all fail; four of six PNG
preservation comparisons fail. The 12 failing group/metric entries are not a
count of people. Original- and preceding-state comparisons duplicate here
because no preceding change was accepted; they are not independent observations.

For example, the largest proposal improves second-cohort asian_faces low-light
raw similarity by 0.000416985, but its PNG similarity drops from
0.349221548 to 0.349088025 (a 0.000133523 regression). Source structure gains
remain positive and brightness-only shares stay below the retained limit.
The delivered recognition failure still blocks acceptance. This isolates a
raw-to-delivered discrepancy that can be checked locally with the saved pixels
before another manual VM experiment.

The **unchanged prospective return checker passed**: 1,629 regular members,
all 400 raw/PNG case records, 64 saved aggregate gradient vectors, the common
direction arithmetic, three proposals and all 24 original/preceding comparisons
were verified. Eighty fresh CPU replay cases agree with VM output within the
predeclared inference tolerances: raw maximum 2.391636e-6, PNG maximum one byte,
embedding maximum 3.147870e-7. The independent audit took 185.655 seconds inside
a 189.008-second supervised run. Replay tolerances do not relax a quality gate.
Individual autograd query vectors and their aggregation were not independently
replayed locally; the checker explicitly retains that limitation.

All 20 gallery pages were actually inspected at original 256-cell resolution,
covering all 400 unique model outputs. The separate pixel checker verifies 600
exact cells and binds each of the 20 observations. Clear and motion profiles
usually retain coarse facial arrangement better than blur/compound profiles.
The three proposals look essentially like the baseline; convincing added eye,
nose, mouth, hair or tooth definition is not demonstrated. Visible glasses,
ordinary hair, hands and camera padding remain in their original role. There
is no completion generation in this study. This is primary-assistant development
review, not an independent final reviewer. The original audit's pending visual
flag is closed by the later bound visual-review receipt.

Both 50-case cohorts are exposed photographic TRAIN data: 20 references with
five synthetic profiles each. The formerly cross-check TRAIN cohort now guides
fitting, as prospectively declared; no independent-generalization claim is made.
The asian_faces and FFHQ-thumbnail labels identify acquisition sources, not
ethnicity. Full 3,905-case TRAIN capacity, native CCTV development, paired
development and reserved final evaluation are not tested by this study.
No Zamboanga or real CCTV performance follows from these paired measurements.

The fixed saved-output conversion diagnostic is documented separately in
CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_DESIGN.md. It compares floor and nearest
rounding fairly for both baseline and proposal, with fresh fixed CPU recognition
measurements and unchanged gates. It performs no DGP forwards, autograd,
parameter updates, training or VM launch. It cannot turn these small gains into
qualified training or supply missing detail. Its newly rounded images require
separate review before any future adoption.

The current own-trained DGP checkpoint remains SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The retained Phase 3 checkpoint, app sources/design, splits, caches, provenance,
terms and all failed gates remain exact. No stopped recipe is resumed. Additional
epochs require a justified finite manual L4 pilot from the current checkpoint,
with actual preservation and useful development evidence before app promotion.

The full goal remains active/incomplete: useful native development restoration,
separate independent final identities/reviewer, DGP primary/Auto/override flow,
all seven automatic/assisted covering families and bundled inline Playwright
verification remain required. Visible appearance, clear glasses and ordinary
hair must remain; insufficient structure needs a clearer or less-covered crop.
Completion is a plausible estimate with original/mask/result and downloads,
not recovery of exact hidden identity.

Evidence:

- [Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_v1_r1_independent_audit.json>)
- [All 400 outputs: visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_v1_r1_return_review_v1/visual_review.json>)
- [Exact gallery and observation audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_v1_r1_return_review_v1/gallery_independent_audit.json>)
- [Original VM scientific decisions](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_v1_r1_vm_return/outputs/results.json>)
- [Fixed local conversion design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_DESIGN.md>)
