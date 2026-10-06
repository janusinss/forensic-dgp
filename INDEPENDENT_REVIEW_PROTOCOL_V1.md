# Independent final review protocol V1 — prepared 5 October 2026

**Preparation only. No reserved pixels, model outputs or reviewer verdicts have
been collected by this protocol. Final execution is not ready:** useful own-DGP
development outputs and covering-family acceptance are unqualified, the final
candidate is not frozen, final reviewers are unassigned, and a separate final
covering cohort has not been selected. The native-softness diagnostic question
remains pending. This document does not waive a historical gate or authorize a
closed training recipe.

Forms and coordinator metadata are in
`outputs/dgp_independent_review_protocol_v1/`. The frozen native manifests remain
the authoritative case/label definitions. No new source, identity, country or
ethnicity is inferred from their appearance.

## Prerequisites before opening reserved inputs

Record the exact final model sources, weights, normalization, cropping, mask
margin, automatic selection rule, completion component and raw/display delivery
in a separately immutable execution binding. Document our learned contribution
and every pretrained component. Use evidence of useful development outputs to
justify that candidate; archive completion or a training trace cannot do so.

Assign at least two independent human reviewers and record their roles, dates
and independence from candidate development. This is the V1 review design, not
a claim that the user has already recruited reviewers. The implementing
assistant's development reviews cannot fill independent final verdicts. A
third reviewer adjudicates disagreements without rewriting the individual rows.
Do not contact or message reviewers without the user's separate authorization.

Before generation, freeze final covering examples, source terms, hashes, masks,
pose/input-only decisions, condition and identity-overlap limits. The existing
36 covering cases were exposed in development and some originate in detector
training. They may support a separately labeled development review; they cannot
be silently relabeled as unused final examples. Known final-cohort gaps remain
explicit until resolved.

Final execution must have a finite forward count, wall-time cap, timing check and
stop/export policy bound before outputs are seen. Any actual fitting remains on
the existing L4 VM and follows verified transfers/pasteable manual commands.
Nothing in this review protocol authorizes assistant VM access or execution.

## Native reserved cohort and limits

| Source | Reserved cases | Release-labeled people | Current exposure |
| --- | ---: | ---: | --- |
| QMUL-SurvFace V1, original frozen subset | 32 | 32 | Native bytes/headers only; pixels remain unviewed |
| ChokePoint P1E_S1 C1, source-specific split | 26 | 13 | Two metadata-selected frames/person; native crops not decoded/rendered |

These are 58 cases and 45 source-qualified person labels, not 58 independent
people. ChokePoint's two temporal views are clustered by person in reporting.
They are not clean/degraded restoration pairs. Report sources separately; do
not pool person counts into a claim of population representativeness. ChokePoint
is not an Asian or Philippine proxy. QMUL's aggregate source list does not assign
capture country or ethnicity to each image. There are no Zamboanga samples.

Development/reserved label separation is checked within each release namespace.
Cross-source person overlap, unlabeled historical photo overlap and pretrained
data overlap remain unknown. Exact different bytes or dataset names cannot prove
different people. Future training must exclude all reserved case/identity roles;
do not use final outcomes to tune thresholds, losses, preprocessing or checkpoints.
Any subsequent change requires a new declared evaluation rather than reuse of
these outcomes as an untouched final test.

## Stage A: input-only review

Review inputs before displaying generated outputs, arm names, model quality
scores or another reviewer's decision. Retain every selected case and rejection;
no replacement of difficult cases. Keep the previously frozen criteria:

1. One already cropped frontal or mildly turned face; unsupported pose/framing is out of scope.
2. Visible eyes, nose, mouth/lower-face relation and contour have usable rough structure; insufficient dark, washed-out or blurred detail requests a clearer crop.
3. Native face dimensions describe the case and do not themselves classify usability.
4. Clear glasses and non-obstructing hair remain visible appearance, not removal targets.
5. A nearly hidden face requests a less-covered image; generic generated anatomy cannot establish that the input was usable.

`native_input_review.csv` supplies two empty reviewer slots per case (116 rows).
Allowed decisions are `usable`, `needs_clearer` and `out_of_scope`; reasons are
required. Cases with conflicting or uncertain eligibility go to recorded
adjudication before inference. Retain counts of all three outcomes per source
and native-size stratum. A rejected input is a workflow outcome, not an omitted
sample or successful restoration. The later runtime must reproduce the agreed
insufficient-input behavior before making an output-quality claim.

## Stage B: blinded native restoration review

Only the final input-only qualified cases proceed. Compare resize, retained
Phase 3, the final own-trained DGP, a declared pretrained restoration baseline
and the frozen DGP Auto result on identical prepared 256×256 inputs. Raw arrays,
delivered PNGs, support masks and all crop/padding/quantization operations remain
separate and auditable. Do not conceal a raw failure behind output-specific
sharpening, contrast adjustment or a preferred display transform.

The five arms receive deterministic per-case labels A–E in separate coordinator
metadata. Reviewers see the native/prepared input and anonymized outputs, not
the mapping, checkpoint names or numeric ranking. A later execution binding must
pin actual candidate files and its unchanged pretrained baseline settings before
rendering. The prepared mapping is a presentation assignment, not a generated
result or a selected checkpoint.

`native_output_review.csv` has 580 blank rows: 58 cases × five presentation arms
× two reviewer slots. It asks for five separate observations:

1. Visible facial structure and appearance remain coherent relative to the input.
2. Restoration provides useful improvement over resizing; brightness/sharpness alone is insufficient.
3. Artifacts, changed visible expression, glasses or non-obstructing hair are absent or explicitly described.
4. Softness is acceptable while rough facial structure remains useful.
5. The recommendation is to use the result, prefer resizing, request a clearer crop or record no usable result.

Record `yes`, `no` or `uncertain` with reasons where applicable; do not fill
unknown fine identity details with a positive judgment. If an input is excluded,
record `NOT_GENERATED_INPUT_EXCLUDED` and its input-review reference for every
planned output arm. Do not replace excluded rows with invented scores.

Native footage without an aligned clean reference is unpaired evidence. No
native PSNR/SSIM, hidden-feature accuracy or recovered-identity score is allowed.
Report case verdicts, disagreement, source/size/quality strata and clustered
person summaries. No automatic numerical percentage in this template declares
the full goal complete. Independent reviewers must record a scope-specific
verdict with its successes, failures and limitations; the requested own-DGP
benefit cannot be replaced by pretrained-baseline success or Auto's resize alias.

## Stage C: automatic and assisted covering-family review

The final covering cohort is **not yet frozen**. `covering_review_template.csv`
provides 54 empty slots: nine required families/controls × Auto/On/Off × two
reviewer slots. Fill actual case IDs and conditions only through a separate
pre-inference cohort binding. Multiple cases/subtypes may require more rows;
these slots are not an adequate sample-size claim or a substitute for that cohort.

| Required family/control | Scope and necessary distinctions |
| --- | --- |
| Face masks | Cloth/medical variants; preserve the visible upper face |
| Sunglasses | Estimate obscured eyes; plausible anatomy without exact hidden identity |
| Strong lens glare | Replace obscuring reflections while retaining visible clear frames where appropriate |
| Hands | Face overlap at eyes/mouth and hand-over-mask; hands outside the face are not removal targets |
| Obstructing hair | Remove only obstructing regions; keep ordinary hairstyle and facial hair |
| Scarves | Face-overlapping scarf/glove variants; clothing outside the face remains |
| Other objects | Separate object types and facial overlap; one successful object is not universal coverage |
| Clear glasses/uncovered | Avoid unnecessary generation; preserve visible frames, transparent lenses and hair |
| Nearly hidden/unsupported pose | Request less-covered/clearer input or reject unsupported pose before generation |

Show the automatic proposal first and retain its raw prediction and documented
margin. Review/correct the removal area before generation; store each edit and
the final reviewed mask. A painted/imported/corrected mask produces assisted
evidence unless it is exactly the same input-bound proposal. Do not count manual
completion success as automatic detection success. Score automatic and assisted
removal, generated content and preservation independently on the same case.

Review the five aspects already agreed in `PRACTICAL_OUTPUT_SCOPE.md`: substantial
removal inside intended facial overlap, plausible estimated facial content,
preserved visible appearance outside the small declared margin, acceptable joins
without conspicuous remnants/seams, and controls avoiding unnecessary generation.
Boundary failures, unchanged covering, changed visible gaze and uncertain anatomy
remain failures/uncertainty even if a hidden estimate looks sharp.

Keep the final removal mask fixed after approval; a new mask requires renewed
review. Record visible-feature support before asserting preservation. An empty
protected mask cannot prove protected-eyewear or hair preservation. Actual whole
visible-support checks, nonempty relevant control regions and independent visual
verdicts provide distinct evidence. Do not treat a deliberately added margin as
genuinely occluded ground truth when scoring automatic proposals.

Return one plausible estimate alongside the original and removal mask, a PNG and
the optional original/mask/result bundle. Hidden identity is unknown. Report
original photographs, synthetic degraded derivatives and any native CCTV
separately. Paired synthetic PSNR/SSIM apply only where aligned known pixels
exist, with removal/support/valid-window rules stated; they remain regression
evidence rather than proof of accurate completion in genuinely hidden regions.

## Stage D: evidence, application flow and final verdict

`readiness_and_evidence.csv` maps the full five-milestone scope to authoritative
evidence and current gaps. A source manifest, green archive audit or historical
test does not alone prove the relevant output requirement. Preserve original
failed gates and report a different prospective practical verdict separately.

After the final candidate is bound, run meaningful model/processing regressions
and bundled inline Playwright from the project root. Resolve the actual target
with `helpers.resolveTargetUrl()` using user/project configuration; do not guess
a port. Check 375/768/1280 overflow, console/page exceptions and real upload,
input qualification, mask preview/correction/renewed review, Auto/override, one
output, PNG/bundle download and insufficient-input rejection. Keep screenshots
and diagnostics in git-ignored `scratch/`; no scratch JavaScript or delegated
browser verification. Match download content to saved original/mask/raw/result.

Reviewers date/sign their source-specific and family-specific findings before
unblinding the presentation arms. Preserve each individual verdict and adjudication.
The coordinator then connects blind labels to frozen components and reports the
actual learned contribution, baseline comparison, automatic/assisted differences,
all exclusions and all limitations. Reviewers' identities remain unassigned in
this preparation; no document here claims their participation.

Final completion requires useful own-DGP native development and independent
reviewed results, the full requested covering-family workflow, meaningful
regressions and the verified DGP-led local app. Public acquisition, fitting,
integrity checks, template creation, a clear-glasses Off control or one plausible
completion cannot replace the rest of that scope. The goal remains active.
