# Varied-covering pilot results — 3 October 2026

The V2 return and fixed practical mask comparison are independently verified.
Adding the reviewed COFW cohort improves all five new covering groups in the
exposed training measurements, but retention and complete-hair-detection checks
fail. Neither checkpoint is selected for the application. Practical hand/scarf
proposals improve; the gallery hair source and nearly hidden rejection still fail.
All 20 actual face estimates are reconstructed and visually inspected separately
from detector metrics. Central coverings are often replaced, but hair, covering
remnants and several generated joins prevent full-scope readiness.

| Artifact | Windows local | Linux VM original or intended counterpart |
| --- | --- | --- |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_RESULTS.md` | `~/forensic-dgp/VARIED_COVERING_RESULTS.md` after future transfer |
| Return | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-results.tar.gz` | `~/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz` |
| Returned models | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_varied_covering_v1\outputs\varied_covering_vm\<arm>\last.pth` | `~/forensic-dgp/varied_covering_vm_bundle/outputs/varied_covering_vm/<arm>/last.pth` |
| Independent return audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_results_validation_v1\verification.json` | Intended `~/forensic-dgp/outputs/varied_covering_results_validation_v1/verification.json` after transfer |
| Practical audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_practical_validation_v1\verification.json` | Intended `~/forensic-dgp/outputs/varied_covering_practical_validation_v1/verification.json` after transfer |

## Verified execution

The 108,064,323-byte archive matches its LF sidecar, SHA256
`63014d50269bb0b50da0c5b5d1f54cadee1bd180634f0a51ac8b502f0ec9488c`.
All 1,653 members, 1,638 saved masks and 256 step records pass independent checks.
Each branch has 184 finite checkpoint tensors, 92 unchanged reference-head/BN
states, 128 fresh updates and 1,122 cumulative model updates. Recorded L4
training/measurement time is 43.45 seconds, excluding loading/export. This is a
finite detector comparison, not full-dataset generator/identity training.

Checkpoint SHA256: `existing91`
`ce6fa1b54e7322b50c75df82e61c16eedd9022b261ae5ce8e4dc4041b9d37df6`;
`varied133` `e3a087f55e248e64f93c96ad50aa3cb73bd08d0ea923583ac11f02aa85179b29`.
Final optimizer tensors remain on the VM; their logged hashes/counters are not
independent local moment inspection. Training inputs are 133 native/degraded
source pairs plus 280 reflection fixtures. Unknown boundaries/padding are
excluded from supervised reductions. These are approximate exposed training
labels, not hidden facial ground truth or population performance.

## New-family training fit

Pooled supported-pixel IoU, matched `existing91` control versus `varied133`:

| Family | Native control | Native treatment | Degraded control | Degraded treatment |
| --- | ---: | ---: | ---: | ---: |
| hand | 0.5754 | 0.7130 | 0.4879 | 0.5893 |
| hair | 0.0319 | 0.2931 | 0.1037 | 0.2256 |
| cloth | 0.5848 | 0.7564 | 0.5670 | 0.7948 |
| object | 0.4228 | 0.8331 | 0.3299 | 0.8502 |
| eyewear_mask | 0.6244 | 0.8748 | 0.6811 | 0.8883 |

All new native/degraded group IoUs improve and every final clear control stays
empty. Three other frozen fit checks fail: old-cohort retention, reflection
retention and every new covered view having a nonempty mask. Four treatment hair
views stay empty (`real/207`, `211`, `212`, `213`); all ten preview rows were
inspected and recorded in the return-audit folder. Native/degraded hair IoU gains
still leave substantial missing coverage. A pooled gain is not full removal.

Old native IoU drops from0.9244 to0.8993 against the matched control, old degraded
from0.9158 to0.8840, and reflection from0.9133 to0.8997. Supervised visible false
positives increase on all three retained groups. Original425 qualification gates
were not evaluated, weakened or waived. No `best.pth` or app swap follows.

## Fixed practical transfer

The comparison retains all36 prior cases from18 previously inspected sources,
including four clear controls, two nearly hidden inputs and two difficult
three-quarter diagnostic cases outside the first-version pose scope. Models are
the cached retained app default and camera91 source, plus both new checkpoints.
Threshold0.5, 3px proposal expansion, inputs/references/protected wires and the
conditional visibility guard remain unchanged. No source-aware routing or
threshold/margin sweep occurs.

The two returned detectors perform106 CPU image forwards in10.26 seconds
excluding loading: 34 fixed return reproductions and72 new practical predictions.
All34 reproduction masks match their VM PNGs exactly. All106 probabilities,
144 metric records and290 artifacts pass independent verification. State
fingerprints remain unchanged; no optimizer, completion or restoration runs in
this detector comparison. All six full preview sheets were inspected.

Mean degraded proposal recall below uses the declared development references;
it is not hidden-face quality or an unbiased accuracy estimate.

| Covering | Camera91 source | Matched control | Varied treatment |
| --- | ---: | ---: | ---: |
| face_mask (3 cases) | 80.2% | 88.1% | 86.1% |
| hand (2 cases) | 57.0% | 73.0% | 74.0% |
| scarf (2 cases) | 40.8% | 76.7% | 80.0% |
| other_object (2 cases) | 62.6% | 74.0% | 76.4% |
| obstructing_hair (1 cases) | 0.0% | 0.0% | 0.0% |

The three mask cases include one three-quarter diagnostic. Per-case and pose
metadata remain in the audit; this table does not qualify first-version pose
coverage. Treatment improves degraded hands/scarves/objects slightly over the
matched continuation, but more nearby visible edits occur on some hand/object
cases. Sunglasses improve modestly; white glare loses degraded coverage relative
to camera91. Both returned detectors keep every clear control empty. The retained
app-default cached mask falsely marks11,683 pixels on the native uncovered control;
that baseline defect remains recorded, rather than hidden by the newer results.

Both new detectors still produce empty masks for the gallery hair source in native
and degraded conditions. Training-cohort hair improvement did not transfer to
this source. Both nearly hidden sources still miss the automatic rejection guard.
They are held from generation based on source review. Hands, mask straps, camera,
flower/leaf edges and scarf boundaries still require optional correction.

## Downstream output review

A ten-case degraded comparison completed through unchanged
CodeFormer completion, Auto visible restoration and palette preservation.
It produced 20 automatic-mask outputs (two branches), with 18 completion forwards,
two empty-hair bypasses and 20 restoration forwards in 174.15 CPU seconds excluding
loading. There were zero detector forwards or optimizer updates; the frozen cap
was 300 processing seconds after loading. It includes cloth
mask, dark sunglasses, white glare, two hands, hair, two scarves and two objects.
Nearly hidden sources and the two three-quarter diagnostics are excluded.

Protocol/results location is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_completion_review_v1\`
(intended `~/forensic-dgp/outputs/varied_covering_completion_review_v1/` only after
transfer). All 20 final PNGs reconstruct exactly from the captured float stages.
The independent audit verifies 59 artifacts and unchanged model fingerprints.
Its historical `visual_review_pending=true` records the audit-time state; the
subsequent `visual_review.json` records inspection of all ten rows and 20 outputs.

| Covering | Observed automatic output |
| --- | --- |
| Cloth mask / hands | Central covering mostly replaced; cloth straps and finger edges remain. Estimated hand-eye features are visibly stylized. |
| Sunglasses / glare | Treatment clears dark lens interiors but retains or generates spectacle frames. Most white glare is replaced while clear frames remain. |
| Obstructing hair | Both masks are empty; hair remains and no hidden eye completion occurs. |
| Scarves | Plausible lower-face structure on the knit scarf, with a cloth strip remaining. Scarf/glove output has conspicuous wool/color contamination and mouth/chin artifacts. |
| Flower / leaf | Core objects mostly replaced. Pink petal remnants and flower-mouth artifacts remain; the leaf estimate is more plausible but still has boundary remnants. |

This is assistant review of exposed development cases, not expert/user acceptance
or proof of the person's hidden identity. No manually assisted comparison was
rerun, no checkpoint was promoted, and the existing app/models remain unchanged.

Downstream results SHA256:
`6feceb94e9c21c87f88c1b9fbdbfa6bc6f02ebe9745b35df061f60c02df841cb`;
independent audit `5cfb067b650c1069f68749a2d58ae0882d5b76e18de18e424e59ae41af117ece`;
visual review `e08298b5f213c8546d4435c7d0c2451a99672c9a4aa0e8f3ee09a596bd430ae3`.

## Footprint diagnosis and next pilot

The next saved-pixel diagnosis completed144 proposal recounts,72 probability
summaries and seven comparisons with verified cached assisted outputs in 1.28
seconds. No new model forwards or training updates occurred. All 449 bindings are
unchanged. Results/visual review are at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_footprint_diagnostic_v1\`
(only JSON receipts are bundled for the next VM experiment).

Both hair masks are empty, with treatment crop maxima0.0000349 native and0.0000713
degraded. Expanding an empty mask cannot repair this failure. Correcting the area
in the cached assisted path yields a plausible estimated eye while keeping exterior
hairstyle. The precise training/domain cause is unproven.

Reviewed nearly hidden masks cover 85.9% of the approximate facial region and both
are rejected by the unchanged guard. The treatment's automatic areas cover 44.5%
native/53.9% degraded, so it misses rejection. All 36 reviewed-mask guard decisions
match their declared gallery scope; this conditional check does not qualify a
general visibility detector. Corrected masks reduce finger/cloth/petal remnants;
scarf/glove generated texture and stylized anatomy remain limitations.

Next: the separately prepared RGB/grayscale exposure experiment retains V2's
existing family sampler and tests grayscale input with equal512-update branches,
including exact reproduction at RGB128. It is ready for VM preflight, with actual
execution pending. Frozen commands:
`C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_VM.md` ↔
`~/forensic-dgp/gray_covering_vm_bundle/GRAY_COVERING_VM.md` after bundle extraction.
Preparation evidence is in `GRAY_COVERING_READINESS.md` under the workspace root.
No generator training or app checkpoint selection follows from these diagnoses.
