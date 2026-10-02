# Practical facial completion scope — user decisions, 2 October 2026

This records the requested output behavior. It is not a trained-model claim or a
change to any executed experiment. Windows workspace:
`C:\xampp\htdocs\YEAR 4\Testing\`; VM repository: `~/forensic-dgp/`.
The intended VM counterpart is `~/forensic-dgp/PRACTICAL_OUTPUT_SCOPE.md`; this
clarification has not been uploaded. All actual training remains VM-only.

## Confirmed first-version workflow

The user confirmed the first-version workflow on 2 October 2026:

1. School staff or thesis researchers review generated images manually.
2. Input is an already cropped face image; full-scene, multi-face and video input
   are outside the confirmed first-version upload workflow.
3. Show the automatically detected removal area for optional correction before
   generating the face estimate.
4. Run image processing locally and use the Google Cloud VM only for training.
5. Prioritize useful facial output over processing speed.

The second round resolved application behavior:

| Decision | Confirmed behavior |
| --- | --- |
| Application | Combine restoration and covering removal in the existing main application; preserve its existing UI |
| Output | One estimated face alongside the original and removal area |
| Restoration | Automatically restore blur/noise when needed, with a user override |
| Downloads | Final image and an optional bundle containing the original, removal mask and result |
| Readiness | Useful results across a fixed review gallery, allowing manual mask correction |

The final round resolved execution constraints:

1. Start with existing data; find and prepare suitable additional public research
   datasets if requested covering families are missing. Check provenance, usage
   terms and overlap before incorporation.
2. Support frontal and mildly turned faces first. Full side profiles and strongly
   tilted faces are outside the initial demonstrated pose scope.
3. Actual training uses the existing NVIDIA L4 / g2-standard-4 VM (4 vCPUs,
   16 GB RAM). The user has credits and imposed no fixed GPU-hour cap. Keep each
   experiment bounded and report its cost in runtime; do not repeat failed recipes
   unchanged or use this answer as permission to provision additional resources.

An uncovered reference image of the same person is not required. A generated
estimate is not an automatic identity/attendance decision. All discovery questions
are resolved. The actionable specification is in `SYSTEM_WORKFLOW_AND_GOAL.md`.

Read-only code/runtime inspection finds two current apps: `app.py` for the main
Phase3 restoration page and `completion_web.py` for experimental completion.
The latter supports face cropping, mask painting/erasing, pretrained CodeFormer
inpainting and a manual visible-restoration toggle. This is not yet the agreed
combined workflow. No app/browser interaction was needed for this inspection.
The local machine reports an RTX3050 Laptop GPU, but the project's current venv
uses `torch 2.13.0+cpu`, CUDA unavailable. A future local inference setup must
account for that distinction; no CUDA package replacement occurred in discovery.

## Confirmed goal

Use a single uploaded face image. Detect whatever covers facial features, replace
the covering with plausible facial content, and restore observed blur/noise where
needed. The output does not have to recover the person's exact hidden appearance.
Completion/inpainting generates missing facial regions; restoration improves
degraded visible features. The visible person's appearance should remain coherent.

| User decision | Intended behavior |
| --- | --- |
| Covering types | Include masks, sunglasses, strong lens glare, hands, obstructing hair, scarves and other objects covering the face |
| Ordinary glasses | Keep ordinary clear frames and transparent lenses; replace sunglasses and strong obscuring glare |
| Boundaries | Complete removal may regenerate a small surrounding skin margin; exact pixel-perfect boundaries are not required |
| User assistance | Automatic detection with optional painting/adjustment of the removal region |
| Hair | Leave hair untouched unless it obstructs facial features; preserve ordinary hairstyle and facial hair |
| Nearly hidden face | Request a less-covered image when nearly the entire face is hidden |

For a clear face with a mouth mask, estimate the hidden lower face and preserve
the visible upper appearance. For a degraded masked face, also improve visible
blur/noise. For a hand or scarf, replace its overlap with the facial area; this
does not imply erasing the person's hand, clothing or hair elsewhere in the image.
Glare over otherwise transparent glasses targets the obscuring reflection rather
than automatically removing all visible clear frames.

## Clarification completed

All six product decisions are resolved. The user's hair instruction is to leave
hair untouched unless it covers the face. Treat obstructing strands as completion
regions; do not automatically erase normal hairstyle, eyebrows, beard or moustache.
The selected behavior for a nearly fully hidden face is to request a less-covered
image. No extra uncovered reference photograph is required for supported inputs.

The current inpainting adapter rejects removal masks covering85% or more of the
square crop. That is an existing implementation constraint, not a decision newly
made for the user. Any changed behavior needs a separately reviewed implementation.

## Proposed practical output review

Freeze the examples and review rules before evaluating a new pipeline. Prioritize
real images for each covering family, transparent-glasses/uncovered controls, and
clear/degraded conditions. The scope is broad; aggregate scores dominated by
medical masks cannot demonstrate hand, scarf, hair or glare handling.

Review these five aspects separately:

1. The covering is substantially removed within the intended facial region.
2. Generated facial features are plausible without requiring the actual hidden appearance.
3. Visible features outside the intended removal margin retain their appearance.
4. The replacement joins the surrounding face without conspicuous covering remnants or seams.
5. Clear faces/glasses avoid unnecessary generation; failed detection offers manual correction.

Record automatic and manually corrected results separately. Manual success proves
the assisted completion path, not fully automatic detection accuracy. Retain input,
raw predicted mask, any expanded/corrected removal mask, output and restoration
setting. An intentional surrounding margin is a documented composition policy;
do not relabel visible pixels as genuinely occluded to make detector scores pass.

Numerical metrics remain regression evidence. Record original gate outcomes and
preserve completed protocols; do not retrospectively rename failed experiments
as successes. A new practical acceptance protocol may differ from earlier strict
mask-selection assumptions, but its criteria and margins must be versioned before
evaluation. None is silently changed in this clarification.

## Current implementation and next decision

The experimental page already supports supplied/painted masks, a CodeFormer
inpainting backend and optional Phase3 visible restoration. It does not establish
reliable automatic detection of every covering family. CodeFormer and restoration
weights exist locally; their existence is not output-quality evidence.

The prepared `NATIVE_EXPERT_VM.md` pilot targets native lens coverage using83 real
training images and280 reflection fixtures. Its fixed package is immutable. It
does not by itself address all newly requested covering types. Before continuing
VM work, check real-example coverage by family and prepare a fixed output review
that distinguishes detection failure from
completion failure using the same images with automatic and reviewed masks.

A metadata-only check of the83 supported real training records finds68 without
an `occlusion_stratum`, four opaque-lens cases, seven transparent controls, one
profile respirator, one hand-over-mask and two scene-reflective lens cases.
Untyped records cannot establish or exclude hair/scarf/other-object coverage;
native visual classification is needed. No inference/training occurred in this
check. Preserve existing split membership when adding these descriptive tags.

Do not repeat unchanged generator training solely because a covering remains in
the image. Establish whether the selected area or generated facial content causes
the failure. Training, application promotion and a new margin policy have not been
started by this clarification. Implementation remains incomplete. Follow the
current Goal and milestone status at the top of `PROJECT_HANDOFF.md`.
