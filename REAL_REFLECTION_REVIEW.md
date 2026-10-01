# Real reflection source review — 2 October 2026

24 previously queued native images are qualified for further annotation. At this
qualification milestone none had an approved new label. The subsequent ten native
annotations are now reviewed in `REAL_REFLECTION_PROPOSALS.md`; fourteen remain
unlabelled. No new training manifest or GPU recipe is ready.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; existing executed bundle:
`~/forensic-dgp/coverage_vm_bundle/`.
This is local source analysis only, without model forwards or optimizer updates.
No source review artifacts have been uploaded to the VM. Actual training remains
VM-only; original gates and generator/application baselines are unchanged.

## Fresh membership and overlap check

The old 27-source eyewear queue predates the current 105 reviewed examples.
Indices 13, 46 and 49 are already used in real training and were excluded rather
than counted as new examples. The remaining 24 retain their original Phase4
training membership. No validation, test or benchmark source was reassigned.

The new tool verifies queue, candidate, role, split and previous duplicate-screen
bindings, current image/mask hashes, native source bytes and the failed crop result.
A fresh screen compares all 24 candidates against 4,210 current source/crop,
original-validation and benchmark references using raw-byte hashes, EXIF-aware
decoded RGB hashes and 63-bit DCT distance at most 6. There are no exact/decoded
matches or perceptual flags and no flagged candidate pairs. Reference signatures
are saved for reproducibility. This does not certify identity separation or detect
every alternate crop; no independence claim follows from the screen.

Three counterexample tests pass after the observed missing-module failure:
exclude a newly used source despite old queue membership; exclude held-out and
benchmark sources; reject duplicate candidates and already-labelled/enabled queues.
These tests qualify tooling, not model quality. Actual native source outputs and
input/artifact hashes were also checked after execution.

## Source inspection and remaining annotation work

Five native-aspect pages covering all 24 sources were inspected. Nearest-neighbor
magnification preserves the low-resolution pixel structure; it adds no detail.

| Proposed review role | Count | Status |
| --- | ---: | --- |
| Covering / reflective eyewear | 8 | Native polygons pending |
| Transparent eyewear control | 10 | Empty targets not approved |
| Ambiguous covering boundary | 6 | Deferred; no pixel labels |

The inspection found important distinctions: source 119 contains a background
face, source 106 includes foreground food, and 310 has tinted lenses plus a hand
near the chin. Some eye detail appears behind source 218's tinted lenses; do not
label the entire lens automatically. Sources 157, 219 and 232 need close inspection
before any empty control target. Sources 171/348 show strong scene/color reflection;
216/374 show dark covering. Trace the actual hidden region and inspect other facial
coverings under `OCCLUSION_POLICY_V3.md`, preserving transparent visible details.

The existing six deferred cases include skin overexposure, blur, colored spots
and small/partial glints. Brightness alone cannot approve an occlusion target.
The accepted thesis scope includes strong white, dark scene and blue reflections;
it does not establish precise boundaries for these new photos. All findings are
assistant source review, without independent expert adjudication.

## Preserved evidence

Local source proposal directory:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real_reflection_review_v4\`.
It contains `manifest.json`, `reference_signatures.json`, `visual_review.json`,
24 native RGB review images and five pages. There are no mask files or new split
assignments. These artifacts have no new VM counterpart.

| Artifact | SHA256 |
| --- | --- |
| Source manifest | `76bcfe8061b331f09eaa78ae8f2c2e6a372e9da1042d5b4723454526b81d846c` |
| Reference signatures | `9915bc00c497b98a6b154eacd9d84f1eeae8a7d182955f6eab20a3f73378f1f0` |
| Visual review | `c0de12206894d2c04f23e0090dd1dadd69e00bb2816fa812d901ee662139f1fb` |
| Executed qualifier | `2cc85e8fd11669e43b088ca3d5c793b89375e784a4c83cac8d1d4abb71c0d177` |
| Existing signature helper | `46f823a2c9129cd26ca6ff58db24e27396f23266c8082aac0e247a8833d48c5d` |

The tool is `scripts/prepare_real_reflection_review.py`; it preserves existing
completed output. No sent/executed VM recipe or accepted dataset was edited.
No training, commit, push or new VM upload occurred.

Next: trace native polygons for clearly obscured regions, review overlays and
uncertain/mixed-covering cases, and validate a separate proposed dataset version
before accepting additions. Include explicit geometry and padding support in any
new training recipe. More low-resolution opaque-lens examples alone do not resolve
small/partial real reflection transfer. Synthetic retention also needs a distinct
predeclared repair hypothesis. An eligible detector plus inspected end-to-end
completion improvement is still required; hidden features remain plausible estimates.

Later annotation milestone: four covering and six clear labels now have source/
overlay review and independent geometry/support checks. The source171 lower
helmet/strap region remains unsupervised; fourteen other sources stay unlabelled.
The subsequent explicit supported reader and separate115-record registry are now
independently verified. See `C:\xampp\htdocs\YEAR 4\Testing\SUPPORTED_REAL_DATA.md`.
Next is a distinct retention-repair specification informed by prior failures.
