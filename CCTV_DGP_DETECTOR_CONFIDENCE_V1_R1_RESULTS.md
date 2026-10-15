# Detector confidence diagnostic V1 R1 — 14 October 2026

**The diagnostic completed; automatic covering detection remains inadequate.**
The unchanged detector made empty proposals on 4/14 eligible original-photo
covering views and 11/14 matched synthetic-degraded views. An uncovered original
also received an incorrect 11,683-pixel proposal. More mask margin cannot fix
empty predictions. No new detector, restorer, completed face or training update
was created. [Independent saved-array audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_confidence_v1_r1/audit.json>).

## Evidence and input scope

One frozen CPU inference run recorded 36 score maps, 36 raw masks and 36 app
proposals in **10.20 seconds**. These are 18 original photographs
and their matched synthetic degradations, not 36 independent people. Thirty-two
views are input-usable, including four clear controls; only 28 covered views
(14 original/degraded pairs) enter the covering counts. Four excluded views
retain their input-only out-of-scope or needs-clearer decisions. The legacy
`_native` filenames denote original photographs here, **not native CCTV**.
They provide no Zamboanga performance, ethnicity or untouched final evidence.

The reviewed operator masks are approximate removal footprints, shared between
matched views. They are not segmentation ground truth or an aligned clean hidden
face. These sources are already exposed development/training material; external
pretraining overlap is unknown. Current scores are uncalibrated classifier
outputs, not validated probabilities. Small exposed counts describe these exact
cases and cannot establish general population accuracy.

| View condition | Eligible covered views | Empty proposals |
|---|---:|---:|
| Original photographs | 14 | 4 |
| Matched synthetic degradations | 14 | 11 |

[All counts, pairs, exclusions and recorded family tags](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_confidence_v1_r1/summary.json>).
Composite tags such as hand-over-mask and sunglasses-with-glare are retained;
the agreed seven covering families are still required separately.

## Where detection fails

The .5 threshold and documented three-pixel margin at 256×256 were unchanged.
All 36 proposals equal the prior app masks by pixels, and the saved measurements
also reproduce the original evidence. Thus this result explains existing masks;
it does not introduce a new segmentation policy or improve generation.

The four empty original covering cases are the two dark-sunglasses examples,
white lens glare and flower-over-mouth. Eleven degraded cases are empty, spanning
pink mask, sunglasses, glare, hands, obstructing hair, scarves and flower. Their
raw threshold masks are already empty before margin processing. For example,
the operator-region maximum score is 4.6871e-8 on the degraded pink mask and
1.0080e-7 on the degraded knit scarf. These specific misses are not predictions
just below .5. No threshold was fitted or swept on these exposed images.

Inference from the exact masks: binary dilation needs selected seed pixels;
it cannot recover these empty predictions. The low region scores and extra
selections on visible features demonstrate detector scoring/localization
limitations on these cases. They do not establish one unique architectural,
loss or dataset root cause, nor justify guessing a lower threshold. This
diagnostic does not test the completion generator or explain all generated
central defects.

The original obstructing-hair view has a nonempty 70-pixel proposal but **zero
raw selected pixels inside its reviewed obstruction footprint**. A nonempty
proposal alone is not evidence that the covered feature was detected. Mirrored
lenses and the degraded leaf have larger proposals, while other covered views
receive fragments or extra selections on visible eye, mouth, hair and background
regions. All six sheets were directly reviewed at 1280×1752, including all 180
input/score/raw/proposal/operator cells.
[Every-sheet observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_confidence_v1_r1/visual_review.json>).

| Clear control | Proposal pixels |
|---|---:|
| 08_uncovered_native | 11683 |
| 08_uncovered_degraded | 0 |
| 09_clear_glasses_native | 0 |
| 09_clear_glasses_degraded | 0 |

The uncovered original's false proposal is 17.826843% of its 256×256 canvas,
including background and visible facial pixels. The clear-glasses pair is
correctly unselected in this small review. Zero overlap with the limited supplied
protected masks does not prove preservation of every visible pixel; this run
generated no faces. Clear glasses and ordinary hair remain protected by scope.

## Location, cause and corrected startup

Location: the original V1 recorder, `scripts/record_dgp_detector_confidence_v1.py:142`.
Cause: unavailable optional `psutil`, before any model import or neural forward.
Fix: a separate R1 recorder uses the built-in Windows process working-set API.
A touched 64 MiB allocation verified that reader. No environment package was
installed. The original frozen protocol, source, traceback, supervisor and zero-
call failure remain intact; R1 keeps the same images, checkpoint, threshold,
mask processing and budgets. The supervisor enforces 210 seconds, with an
internal 180-second/4 GiB memory/32 MiB output limit.
[Preserved startup failure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_confidence_v1/failure_audit.json>).
[Separate repair binding](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_detector_confidence_v1_r1_startup_review.json>).
[Microsoft process-memory API](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo).

The detector checkpoint is `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`, trained
container epoch 2; this is **not the primary DGP's epoch count**. The local runtime
was `2.13.0+cpu` on CPU. The CompletionNet container loads segmenter and
unused generator tensors; only the segmenter executes. The primary restorer and
separate app completion backend were not loaded. Counts are detector 36,
completion 0, restorer 0, gradients 0 and optimizer updates 0.
Before/after state receipts match the checkpoint tensor schema exactly.

The independent checker reconstructs thresholding and the seven-pixel-square
dilation with SciPy, recalculates region statistics and all prior app proposal
comparisons, and reconstructs every image cell. All 118 source and 114 output
bindings pass. It does not repeat neural inference. Prospective array fixtures
reject eight saved-score/mask corruptions; the cohort checker rejects seven
false claims, including relabeling photographs as CCTV and counting excluded
or clear images as covered cases. Completion of these audits does not qualify
the detector or model. Result-time pending flags remain historical; the later
audit and visual sidecars record their actual completion.

## Consequence for the full goal

The automatic detector needs a justified representation/label/degradation
review before any new finite training design. Assisted masks and completion
generator quality remain separate obligations: correcting an area does not
establish a plausible hidden face. No detector training recipe, threshold
change, mask expansion, new restorer, app adoption or epoch extension is selected
by this diagnostic. All automatic and assisted covering families remain
unqualified; original/mask/result previews, downloads and insufficient-
information requests remain required.

The joint, RAW-only and nearby-gradient restoration recipes remain failed.
The required reconstruction-path-versus-gradient design question is still
pending; no fourth training recipe is prepared while awaiting that discussion.
[Restorer design boundary](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md>).
The original DGP checkpoint remains the permanent baseline. It is a fine-tuned
DeblurGAN-v2-compatible restorer; the external frozen StyleGAN2 bank is not an
own-trained prior or qualified replacement. All five milestones remain active,
with native unpaired, synthetic paired and final evidence kept separate.
Actual training/gradients stay human-manual on the L4 in tmux; direct agent VM
access stays inventory-first hash-bound maintenance. Goal incomplete.
