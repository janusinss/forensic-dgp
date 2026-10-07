# Mask-import review ordering: verified application fix

7 October 2026. Selecting an imported removal mask previously left generation
enabled until the PNG finished loading. A controlled browser reproduction sent
one request with the old mask, then displayed its result after the new mask had
cleared review. The submitted old mask was empty; the intended imported mask was
not. This violated the requirement to review the actual removal area first.

The frontend now enters its existing busy state and invalidates review immediately
when import begins. A successful import requires fresh confirmation. Invalid
imports recover the controls while keeping approval cleared. An older import
cannot unlock detection or replace the proposal for a subsequently uploaded face.
The original frontend source is retained under
`outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js` (SHA256
`81edfc86fcd16d976f1ed98d146680b38ada9af7504f6bdc4d60a5471cb249b0`).
Current frontend SHA256:
`9be6cfab57e0041f87571cd38ba6d0d1e0850742b8fafd498f1b1d126137a060`.
Backend, checkpoints, masking margins, numerical gates and visual design retain
their previous source bindings. Historical integration evidence refers to the
archived frontend; it is not rewritten to describe the new source.

Three bundled inline Playwright regressions pass: slow import, invalid dimensions
and a new upload while import is pending. Their controlled transport creates no
neural outputs and makes no image-quality claim. The independent pixel check
confirms that the later submitted mask exactly matches the imported mask.

A separate plan was frozen before real-model testing. In 66.29 seconds the actual
local app completed 11 generation requests and 11 automatic-proposal requests.
The seven covering families are masks, sunglasses, strong lens glare, hands,
obstructing hair, scarves and other objects. Two cases are uncovered/clear-glasses
controls; two unsupported or insufficient-structure inputs submit zero generation
requests. Default Auto and explicit On/Off overrides work. Imported corrections
require renewed approval before each family is generated. PNG downloads and a
review bundle succeed. Responsive checks at 375, 768 and 1280 pixels record no
horizontal overflow, page exceptions or console errors.

The separate saved-output audit verifies all 11 downloaded PNGs against the
unchanged, previously audited processing outputs. It checks 1,521,399 visible Off
bytes, native/prepared originals, native/prepared masks, result, processing
metadata and exact saved raw DGP floats in the bundle. The raw floats and completion
stage reproduce the displayed PNG composition. This audit performs no model calls
or autograd. The baseline own-trained DGP remains SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

These are exposed development photographs. Their legacy `_native` suffix means
original photo, not native CCTV. All covering generations use previously reviewed
operator areas; automatic proposals are displayed before correction. This does
not qualify automatic covering generation, independent performance, hidden identity
accuracy or Zamboanga performance. Completion/visible-restoration defects remain
as recorded in `CCTV_DGP_APP_V3_COVERING_RESULTS.md`. Screenshot review confirms
the existing interface and blocked/reviewed states; it is not a new quality review
of the unchanged neural outputs.

Evidence: `outputs/dgp_mask_review_race_fix_v1/` and
`scratch/mask-import-race-v1/`. Inline audit source is recorded as JSON; no scratch
JavaScript scripts were created. The initial missing audit-directory startup and
sandbox Chromium-spawn limitations did not change app code or launch any VM task.

V31 remains a distinct finite manual VM experiment. Its transfer archive, protocol
and training gates are unchanged; no returned V31 archive is present locally at
this milestone. The user's subsequent launch failed during transfer verification
because the root schedule helper was absent from Python's script import path.
Two read-only maintenance checks retained its terminal failures and verified a
command-only PYTHONPATH correction with zero neural or optimizer calls. No agent
launched training or changed the packet. The corrected command in
`CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md` supersedes step4 of the original
five-step guide; its upload, installation, tmux and separate downloads still apply.
Useful native DGP restoration, automatic/assisted whole-family quality and the
independent final review remain required. Goal active/incomplete.
