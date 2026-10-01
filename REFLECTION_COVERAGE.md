# Fixed reflection-coverage detector pilot — 1 October 2026

This bounded Track2 detector experiment tests whether reviewed partial lens
reflections improve detection relative to same-source clear-frame supervision.
It does not train the completion generator or Track1 restoration model. Previous
control/focus candidates all fail the original synthetic retention guard; no
eligible detector or better end-to-end output has been established.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`.
VM execution root: `~/forensic-dgp/coverage_vm_bundle/`.
The same relative code/data layout is installed in that execution root. Exact
SSH commands and return paths are in `REFLECTION_COVERAGE_VM.md`.

## Frozen sources and data

Twenty-eight training-only uncovered bases are admitted: fourteen from the
Asian source pool and fourteen from FFHQ. Their IDs are
`0,4,7,9,10,13,15,17,18,19,20,21,22,26,101,102,111,113,114,116,117,119,120,123,124,128,131,132`.
The original four prototypes were retained. A fixed additional44 native-aspect
training-clear proposals were inspected;24 were admitted and20 deferred/unpaired.
Selection used source IDs and native visibility/pose, without candidate errors
or validation/test photos. Native references remain low resolution. Assistant
review is not independent expert adjudication or a verified identity split.

CPU eye proposals used the existing InsightFace `det_10g.onnx`, SHA256
`5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`.
Images were uniformly fitted into256px, passed in BGR, then proposed eyes mapped
back into native coordinates. All44 native-aspect views and all28 lens overlays
were inspected. This auxiliary landmark model is not the trained occlusion
detector. Its Windows file is
`C:\Users\janus\.insightface\models\buffalo_l\det_10g.onnx`; the new VM runner
uses frozen tensors and does not require that external file or InsightFace.

Each base has a clear-frame control and four partial lens styles: white patch,
white streak, scene reflection and blue glare. Each appears in clear/degraded
camera conditions:28×5×2 =280 cases,224 positive and56 clear. Same-source/style
variants share camera seed and parameters. Clean fixtures preserve all pixels
outside their known reflection target; degraded fixtures apply blur, resizing,
noise and JPEG across the image, so visible pixels can change as intended.

Uniform affine scaling preserves native aspect ratio; neutral padding is96.
This preserves geometry and does not establish facial alignment. Degraded
support excludes an8px padding-influence boundary; the supervision target uses
a declared4px reflection-effect dilation, clipped to valid support. These
conservative morphologies are modeling assumptions, not a measured physical
camera support. Camera parameters are stored per case. Padding contributes no
supervised BCE, Dice or hard-visible term. Tests verify zero padding gradients
and invariance to logits in unsupported pixels.

The registry remains180 sources:28 paired uncovered,149 pending,3 likely intrinsic
occlusions. Sources160/177 and new118 have no accepted pixel masks. Eight uncertain
sources `[67,78,107,118,121,122,160,177]` are quarantined from **both arms' core
exposures**. Forty-nine replacements preserve real examples, safe replay slots,
covered/clear strata and degraded/clear conditions. Original cached targets and
images are retained unchanged. This quarantine is not a relabeling decision.

All280 cases independently hash/recount;224 unique positives and56 clear cases
are present in the schedule. Four seven-row lens sheets and a four-row camera
sheet were visually reviewed. Frames and scene textures are simplified. Real
transfer, hidden-feature accuracy and improved model output remain unproven.

| Artifact | SHA256 |
| --- | --- |
| `outputs/reflection_coverage_data_v1/manifest.json` | `6696cee3a3acf48049f2f121a2a6328625c4351f558deabb5475da7fe948baf9` |
| `outputs/reflection_coverage_data_v1/pixels.pth` | `ab47a88d79fe8d300ca92d572a2c1a05b92fa273b9577cb209df99e692009cc1` |
| `outputs/reflection_coverage_data_v1/audit.json` | `21a77f47d2cdea74b4d04e0aef7433664422a5c8a923d4ac05e5f9468016fbb8` |
| `outputs/reflection_coverage_data_v1/visual_review.json` | `17cd6e9758a1f82ab4a8b5820ab52794d0e4ee1284fd353fdb16217dcd091420` |

The preparation-stage manifest's `training_ready:false` is preserved. Later
bounded-pilot readiness is recorded separately in `source_check.json` and the
package inventory/build report; it does not certify output quality. Do not rerun
preparation/audit defaults or edit their hash-bound inputs; use a new version.

## Matched optimization specification

Both arms independently start from the audited pretrained epoch30 weights and
the exact saved AdamW states. The source has630 lifetime model updates; saved
moments have step420, reflecting the earlier fresh optimizer initialization.
The already trained source is a diagnostic starting point, not an eligible
application checkpoint. Original parent predictions anchor core synthetic replay.

| Item | Fixed value |
| --- | --- |
| Starting model | `outputs/face_occlusion_continuation_vm/epoch_30.pth`; SHA `9ba74a30719f01bfafd7ee9c060dc090c507c26eeeecb117b4cb342a8bf4df42` |
| Starting moments | Same directory `final_optimizer.pth`; SHA `258a01cc4698668333e191d6d01bd2866bd1e847b1f9df84e1a33c8ec8204b00` |
| Original parent | `outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth`; SHA `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93` |
| Per-arm budget |12 epochs ×21 updates =252 updates;504 total across independent arms |
| Core batch |8: original balanced2 real covered/2 real clear/2 synthetic covered/2 synthetic clear |
| Supplemental batch |2; one scheduled reflection slot plus one scheduled clear slot |
| Control |Replace the reflection slot with its exact same-source, same-camera clear frame |
| Reflective |Use the scheduled reflection in that slot; second clear slot identical to control |
| Objective |Core BCE+Dice+0.25 hard-visible term +1.0 original-parent Bernoulli KL on core synthetic examples +0.25 valid-support supplemental loss |
| New fixture teacher |None; their introduced occlusions have explicit procedural targets |
| AdamW |Encoder1e-5; decoder/head1e-4; weight decay1e-4; original betas/eps/flags; clip norm1 |
| Seed/runtime |42; source frozen BN statistics/reference head; no AMP; TF32 off; deterministic cuDNN settings |
| Evaluation |Fixed experiment epochs6/12; global labels36/42; threshold0.5; no sweep or early stopping |

The core schedule is the registered ten-epoch schedule followed by its first two
epochs as a declared fixed tail, sanitized identically in both arms. The252-step
supplement schedule includes all224 positive combinations; each of eight
style/condition strata has31 or32 exposures. Clear cases are deterministically
cycled. New fixture scores are training diagnostics, excluded from selection.
No adaptive source selection, repeated budget or continuation between arms.

At global36:126 new updates,756 model updates,546 optimizer step.
At global42:252 new updates,882 model updates,672 optimizer step.
Identical global labels identify forks from source30, not sequential training
of control into reflective. Checkpoints bind these counters and the recipe.

## Validation and advancement

Original real validation is25 cases; human and mannequin subsets are reported
separately. The400 synthetic benchmark and its target policy stay fixed. Lens-only
491-pixel validation recall must be independently recounted after return; a whole
glare-image IoU must not substitute for small reflection recovery. The previously
viewed test split is not a pristine final generalization claim and is unscored.

`best_detector.pth` is eligible only through the unchanged `selection()` function:
real IoU must strictly exceed the arm's previous eligible best/original parent;
visible false-positive fraction, empty-mask cases and clear-case false positives
must not exceed the original real baseline. Real missed fraction is reported,
but the unchanged real gate does not separately constrain it. All five original
synthetic safeguards must pass together:

| Synthetic guard | Required |
| --- | ---: |
| IoU |≥0.9746899906463458 |
| Missed fraction |≤0.016476187160583314 |
| Visible false-positive fraction |≤0.0013327836915128428 |
| Empty covered cases |≤1 |
| Clear-case false positives |0/80 |

An absent best file means no candidate qualifies. Training fit, new fixture
accuracy and successful script completion cannot override those gates. The
runner exports full raw binary masks, reports ignored padding positives
separately, and requires source/baseline CUDA counts to match prior audited
outputs before updating weights. New source fixtures never become validation.

After return: independently verify all504 logged updates/losses, package hashes,
source/final optimizer bindings and frozen states; recount raw masks and guard
decisions; reproduce checkpoints on CPU without fitting. Inspect the fixed ten
real rows `[0,1,5,6,2,3,4,17,8,23]`, lens zoom and selected training fixture cases.
Only an eligible detector advances to reviewed end-to-end completion against
the retained baseline. Plausible hidden facial features remain estimates.

## Local verification and VM status

Seventeen new contracts pass. The measured read-only CPU check returned finite
core/control/reflective/padded-support losses of0.048945/0.000221/0.684187/0.314697.
All92 source optimizer states match their parameter shapes and step420. Model
and original parent tensors stay unchanged; no optimizer was constructed, no
backward pass or fit occurred. CPU Torch is2.13.0+cpu with pinned SMP0.5.0.
The intended existing VM environment is Torch2.9.1+cu129 on NVIDIA L4; actual new
CUDA execution is unverified and requires preflight.

```powershell
Set-Location 'C:\xampp\htdocs\YEAR 4\Testing'
venv/Scripts/python.exe -m unittest tests.test_reflection_source_cohort tests.test_reflection_coverage tests.test_reflection_coverage_data tests.test_reflection_coverage_audit tests.test_reflection_coverage_vm tests.test_reflection_coverage_package
```

Model forward, data checks and unit-test gradients are not new training epochs.
Track1 Phase3 `checkpoints/dgp_zamboanga_final.pth`, full Phase5 identity training,
the Track2 generator and application retain their separate status. No automatic
promotion or deployment is performed by this pilot.
