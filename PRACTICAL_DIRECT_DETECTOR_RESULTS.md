# Direct covering detector: completed development comparison

Updated 2 October 2026. The returned reflective epoch 42 detects more native
coverings than the retained baseline, but still misses obstructing hair, degrades
under blur/noise and misses both nearly hidden hand cases. It is not selected by
the application. Its original synthetic-retention failure remains unchanged.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM counterpart after explicit transfer: `~/forensic-dgp/`.

## Fixed experiment and evidence

The comparison uses the same 36 native/degraded inputs as the completed XSeg
experiment: the ten-source practical cohort plus eight RealOcc sources. The
operator references are reviewed removal proposals, not hidden-face truth.
Both three-quarter case 02 inputs are flagged as pose diagnostics; do not count
them as evidence for the frontal/mildly turned operating scope. The six existing
V3 proposals are reused without revision. None of these exposed development
sources becomes new training data or an unseen validation set.

The ResNet18 U-Net predicts covering probability directly from RGB/255. A fixed
0.5 threshold and 3-pixel dilation at 256 scale are applied without an ellipse
complement. Ten previously verified native raw masks are reused; 26 detector
forwards are new. No completion, restoration or optimizer updates occur. CPU
inference takes 4.31 seconds after loading with four threads. State fingerprints
before and after agree. The independent audit verifies 26 probability arrays,
108 binary masks, conversion, cache identity, source hashes, metrics and guards.
Execution counts are checked against recorded results, not replayed by the audit.
All six preview sheets, containing 36 rows, were inspected by the assistant.
The result file's original `visual_review_pending` field is historical; this
document and the separate visual-review record complete that inspection.

| Artifact | Windows path relative to root / same VM-relative path | SHA256 |
| --- | --- | --- |
| Returned checkpoint | `outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42.pth` | `a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25` |
| Protocol | `outputs/practical_direct_detector_v1/frozen_protocol.json` | `69c6a75e4dc1af099a3cb1a621082a870966170108e9a2d2255d614b708a8db8` |
| Results | `outputs/practical_direct_detector_v1/results.json` | `4b07e9a7765486d588f025392f7cc3d2adc84e3a731a8012b4037fa7885975b0` |
| Independent audit | `outputs/practical_direct_detector_v1/independent_verification.json` | `6e2867507a20252704ff453b8c1d0b58a333fe4dbc7367180884c52351e4615b` |

Local preview directory is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_direct_detector_v1\preview\`;
intended VM counterpart is `~/forensic-dgp/outputs/practical_direct_detector_v1/preview/`.
Models and outputs under ignored `outputs/` require explicit transfer, not Git.

## Findings

These are per-case means against operator removal references, after dilation.
Small case counts and prior exposure limit generalization; averages do not select
a production checkpoint or establish the identity of generated features.

| Covering family | Native reference recall | Degraded reference recall | Native / degraded IoU |
| --- | ---: | ---: | ---: |
| Face mask (3 sources, including pose diagnostic) | 87.66% | 52.40% | 0.874 / 0.505 |
| Sunglasses (2) | 66.76% | 45.78% | 0.666 / 0.445 |
| Strong white lens glare (1) | 51.67% | 41.77% | 0.501 / 0.369 |
| Mirrored lenses and glare (1) | 61.88% | 75.44% | 0.619 / 0.750 |
| Hand over mask (1) | 94.64% | 59.32% | 0.925 / 0.582 |
| Standalone hands (2) | 31.21% | 5.32% | 0.309 / 0.053 |
| Obstructing hair (1) | 0% | 0% | 0 / 0 |
| Scarf (2) | 86.24% | 34.49% | 0.822 / 0.342 |
| Other object (2) | 78.85% | 36.08% | 0.773 / 0.333 |

Both uncovered and clear-glasses controls have empty native/degraded predictions,
with no unexpected geometry rejection. These four passing controls do not prove
universal false-positive protection. The nearly hidden hand pair is marked only
19.29% / 11.32% of its reference, and neither automatic proposal triggers the
visibility rejection. Correct reviewed masks still trigger the existing guard.

Preview inspection agrees with the numeric failure pattern: native mask bodies
are better captured, but lens rims, one/both glare regions and some scarf/glove
boundaries remain incomplete. Both hair proposals are empty. Hand-mouth,
patterned-scarf and flower degraded proposals are empty or fragmented. Native
leaf/flower proposals are useful but incomplete. Strong degradation is a distinct
failure mode rather than proof that more identical native-only training will help.

## Next training hypothesis

Prepare reviewed training examples covering standalone hands, hair obstructing
features, scarves and general objects. Existing 83 real training sources are
dominated by medical masks and clear controls. Add native/degraded input pairs
that share the same reviewed covering target and valid support, with a documented
boundary tolerance. Keep original source splits, controls, synthetic replay and
historical retention criteria. Broader native data and degradation augmentation
must be separable experiment arms; a new dataset alone cannot establish their
individual effects.

The newly user-uploaded Mendeley source is being checked for actual resolution,
duplicates and label availability. It must not be treated as paired clean-face
supervision or covering masks merely because the publisher describes occlusions.
RealOcc author-validation sources remain evaluation/development only.

A new VM recipe is pending source/annotation readiness and frozen criteria.
No further training, checkpoint promotion or application detector swap has occurred.
All actual optimizer work must run on the user's L4 VM, never locally.
