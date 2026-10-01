# Native real-mask annotations — 2 October 2026

Ten annotations are reviewed for a future supported detector dataset: four
covering examples and six transparent controls. Fourteen sources remain unlabelled.
The labels are approximate assistant annotations, not independent expert ground
truth. The subsequent separate supported registry is independently verified;
see `C:\xampp\htdocs\YEAR 4\Testing\SUPPORTED_REAL_DATA.md`. No new GPU recipe is
ready and no model quality gain is claimed from annotation checks.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; existing execution root:
`~/forensic-dgp/coverage_vm_bundle/`.
The artifacts below are local only; there is no new VM copy/upload. Training
remains VM-only. Existing accepted datasets, gates, generator/application and
Phase3 restoration baseline remain unchanged.

## Source-only tracing and reviewed scope

The preceding 24-source qualification binds original Phase4 training membership
and a fresh 4,210-reference duplicate screen. No held-out source or model score
selected a polygon. Native coordinates refer to the oriented source RGB, with
uniform pixel-center scaling to 256 and conservative padding support.

| Covering source | Native positive pixels | 256px positive pixels | Annotation scope |
| --- | ---: | ---: | --- |
| 171 | 3,529 | 5,666 | Scene/blue reflective lenses; lower helmet/strap support unresolved |
| 216 | 1,321 | 5,284 | Dark lens footprints conceal eyes |
| 348 | 1,207 | 4,828 | Green scene-reflective lenses conceal eyes |
| 374, refined | 1,011 | 4,044 | Dark lenses; visible cheek spill reduced |

The six empty controls are 207, 208, 217, 230, 240 and 241. Visible eye/face details
through transparent lenses remain outside the covering target; blur and normal
glints are not automatic occlusion labels. The fourteen other sources have no
mask/image/loss-support artifact for training. Uncertain tint, small reflections,
multiple faces and mixed foreground objects need further native review.

The first overlay inspection found source374's mask extending below the glasses
onto visible cheek. Native pixel-grid review refined that source in a separate
output version: 1,174→1,011 native positive pixels. Nine other proposals and the
fourteen pending records retain their exact first-draft bytes/metadata. Both drafts
are preserved; no validation or model score guided the correction.

Both covering overlays and all six clear controls were inspected. The final
annotation decision sidecar records assistant review and explicitly retains the
approximation caveat. These are unpaired real occlusion labels; there is no
uncovered-face or high-resolution reconstruction target.

## Unknown regions must not become negative labels

Source171's lower helmet/strap boundary remains unresolved. A conservative native
band at y156–201 contains 7,268 ignored pixels. Its projected loss support is39,400
pixels versus51,200 source-supported pixels. The ignored observed RGB remains in
the input as context; it is not painted over. Only padding is neutral RGB96.
All target reductions must exclude unknown regions and padding. No mapped known
positive overlaps either region; no positives were discarded at source borders.

The old `ReviewedMasks` yields only image/mask and ignores `valid` support. It is
unsuitable for these new annotations. Inspection also confirms that the legacy
manifest reader does not enforce format/readiness metadata. A separate explicit
reader/registry must prevent this partial annotation from becoming a false clear
target. That explicit reader and registry are now verified and reject the legacy
pair-only consumer; all original held-out support remains full. Do not pass the
proposal manifest to an existing training runner.

## Verification and artifacts

Three native geometry/label counterexamples and two saved-support counterexamples
pass after observed missing-module failures. The independent auditor verifies all
ten original RGB inputs, native polygon rasters, binary labels, affine mappings,
source/loss support, hashes and exact unchanged records. It confirms all fourteen
pending records stay unlabelled and unknown RGB context is preserved. No model
forward or optimizer update occurred. Tool tests do not establish generalization.

Initial draft:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real_reflection_proposals_v4\`.
Reviewed refinement:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real_reflection_proposals_v4_refined\`.
The latter contains `manifest.json`, `verification.json`, `annotation_decisions.json`,
native/output masks, support maps, inputs and two previews. The proposal/verification
files preserve their historical pre-acceptance state; the later decision sidecar
records ten reviewed annotations while training remains disabled.

| Artifact | SHA256 |
| --- | --- |
| First-draft manifest | `368cdf82cfeef3088c225f2b70cd797697419875989fe1d7bfd0482d9d95d24a` |
| Refined manifest | `4d75620863314e7ff26e9c3a4abf3e28a99be46744f7f08730e9af697b9ffe91` |
| Independent verification | `e820314b23913fbb59d8aa837ff776b15d0736f9ad4e3ffdbf7cd760b449e0bc` |
| Annotation decisions | `e6eef67417fa9a152cf7277969a5cae83b8949f5f1038cdadb8bd140c08186ea` |
| Refined covering preview | `1a27d3591bbe5d216ecb5e8cc28546c3cc6b9eb138c9ee641d3e9fc9677483ee` |

Tools: `scripts/prepare_real_reflection_proposals.py`,
`scripts/refine_real_reflection_proposals.py`,
`scripts/audit_real_reflection_proposals.py`. Completed outputs are preserved.
The independent auditor SHA256 is
`592b85c6d8401e471731d1c0514927a4d1f8ef6aeff0f027f57585e60913331c`.
The native-coordinate review grids are temporary, ignored `scratch/` artifacts.
No accepted dataset, executed VM recipe, checkpoint or original split changed.
No commit/push or VM upload occurred.

The subsequent supported-data reader and separate115-record registry pass an
independent252-file audit. All105 original records and73 training tensor pairs
remain unchanged; all ten reviewed additions use explicit valid support. See
`C:\xampp\htdocs\YEAR 4\Testing\SUPPORTED_REAL_DATA.md` for evidence and reader usage.
Next: review previous retention failures before specifying a distinct bounded repair.
A distinct retention-repair hypothesis is still required before another VM pilot.
Four opaque/scene-covering examples do not resolve the small/partial real-glare
gap. Goal completion still requires both original selection safeguards and reviewed
end-to-end completion improvement; hidden facial features remain plausible estimates.
