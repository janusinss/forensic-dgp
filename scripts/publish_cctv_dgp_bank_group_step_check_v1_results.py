"""Publish audited signed-step findings; preserve status history and used guide."""
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_step_check_v1_return import ROOT, RETURN, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link


def main():
    assert not (ANALYSIS/'publication.json').exists()
    proof_names = ['import.json', 'independent_audit_r1.json', 'finite_metrics.json',
        'signed_derivative_review.json', 'actual_step_direction_review.json',
        'quantization_review.json', 'visual_review.json', 'review_binding_audit.json', 'r1_checker_contract.json']
    assert all(read(ANALYSIS/name)['complete'] for name in proof_names)
    audit = read(ANALYSIS/'independent_audit_r1.json')
    imported = read(ANALYSIS/'import.json')
    bound = read(ANALYSIS/'review_binding_audit.json')
    assert bound['all_four_proposals_rejected_for_adoption'] and bound['quantization_case_recounts'] == 400
    assert read(ANALYSIS/'visual_review.json')['all20_sheets_and800_cells_inspected']
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == digest for name, digest in protected.items())
    app = read(ROOT/'outputs/dgp_app_flow_audit_20261012/publication.json')
    storage = read(ROOT/'outputs/google_drive_offload_20261012_v1/closure_20261013_v1/publication_audit.json')
    previous = read(ROOT/'outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/publication.json')
    names = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
             'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    guide = 'CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md'
    expected = {name: (storage['published_documents'][name]['published_sha256'] if name in storage['published_documents']
                      else app['documents'][name]['after_sha256']) for name in names}
    expected[guide] = previous['new_documents_sha256'][guide]
    assert all(sha(ROOT/name) == digest for name, digest in expected.items()), 'Preserve concurrent status edits'
    report_name = 'CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_RESULTS.md'
    design_name = 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md'
    assert not (ROOT/report_name).exists() and not (ROOT/design_name).exists()
    table = '\n'.join('| '+row['proposal'].removeprefix('balanced_')+' | '+
        f"{row['raw_structure_gain_percent']:.6f}% | {row['PNG_structure_gain_percent']:.6f}% | "+
        str(row['raw_preservation_failure_count'])+' | '+str(row['PNG_preservation_failure_count'])+' | '+
        ('Diagnosis only' if row['negative_step_diagnostic_only'] else 'Rejected')+' |'
        for row in bound['summary'])
    design = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        scope='Design review only; no sealed transfer, training recipe execution or new model',
        source_audit_sha256=sha(ANALYSIS/'independent_audit_r1.json'),
        signed_slopes_sha256=sha(ANALYSIS/'signed_derivative_review.json'),
        actual_parameter_step_review_sha256=sha(ANALYSIS/'actual_step_direction_review.json'),
        delivered_png_review_sha256=sha(ANALYSIS/'quantization_review.json'),
        human_direction='apply best approach',
        invalidated_assumption='A raw-gradient descent prediction at the initial state guarantees preservation after a finite change and PNG delivery',
        raw_central_slope_signs_are_favorable_but_not_full_gradient_correctness_proof=True,
        chosen_design='Keep both raw and exact delivered-image losses in learning, use small finite checked changes, and refresh directions at changed weights',
        retained_initial_B_and_original_app_baseline=True,
        reconstruction_and_conditioning_weights_may_learn_frozen_external_prior_remains_declared=True,
        exact_png_floor_forward_with_explicit_surrogate_backward_is_only_a_hypothesis=True,
        surrogate_is_not_derivative_of_discontinuous_PNG_or_quality_evidence=True,
        acceptance='Actual raw and PNG preservation, brightness and 1% structure requirements stay unchanged; no invalid step commits',
        stage_before_more_epochs='Verify matched initial raw/PNG forward losses and a delivered-loss-aware actual finite step on the manual L4 VM',
        no_unchanged_repeat_of_current_fixed_direction_probe=True,
        full_TRAIN_and_native_DEV_and_independent_final_review_required=True,
        finite_limits_must_be_frozen_and_enforced_before_a_new_packet_is_launch_ready=True,
        transfer_packet_prepared=False, training_launch_ready=False, new_neural_calls=0,
        gradients=0, optimizer_updates=0, new_checkpoint=False, VM_launch=False,
        app_adoption=False, more_epochs_qualified=False, goal_complete=False)
    write(ANALYSIS/'next_design_review.json', design)
    report = f'''# Signed small-step result — 13 October 2026

The manual diagnostic completed successfully in **{audit['VM_seconds']:.2f} seconds**.
It produced useful learning-path evidence, but **no qualified improved model**.
There were zero optimizer updates, gradients or training epochs and no new
checkpoint. The original initial model was restored. All four proposals remain
unqualified; the two negative steps are diagnosis only.

| Additive step | Raw structure gain | PNG structure gain | Raw failed checks | PNG failed checks | Decision |
|---:|---:|---:|---:|---:|---|
{table}

Failure counts are source/profile group–metric checks, including the brightness
requirement where applicable; they are not subjects. The 1% structure requirement
and both-source raw/PNG requirements remain unchanged. At +.0001 the source raw
gains are0.009976% for `dataset/asian_faces` and0.009959% for
`dataset/thumbnails128x128`; both are below1%. Dataset names are source labels,
not ethnicity or native CCTV evidence.

## What the signed experiment established

All32 measured central slopes are negative at both .001 and .0001. Raw MSE,
SSIM loss and feature-loss slope magnitudes are close to their saved first-order
predictions. ArcFace central-slope error falls on the smaller radius for all10
groups, but the largest remaining relative difference is about34.7%. Accounting
for actual float32 parameter rounding leaves that discrepancy substantially
unchanged. These results support a local descent tendency; they do not prove
every backward path correct or identify a specific broken layer.

Finite changes can still reverse a favorable local prediction. For example,
`dataset/thumbnails128x128/clear/MSE` at +.001 has a central linear contribution
of−9.18650e−8 and a positive midpoint departure of1.49353e−7, giving an actual
increase of5.74876e−8. This is measured behavior along this direction, not an
explanation of every layer. Simply increasing a fixed step or adding epochs
to the failed recipe is unsupported.

At +.0001 **raw preservation has zero failed checks**, while the PNG has six.
The delivered checks include clear-source MSE/SSIM/ArcFace regressions, two
degraded ArcFace regressions, and brightness fraction0.279498 above0.2.
The raw brightness fraction is0.077726. Both views remain required.

The unchanged converter is `floor(float32*255)`. At +.0001 it changes159,154
observed channel values across156,922 pixels in100 images, by at most one byte
level. About18.67million raw channel values changed before quantization.
Two individual cases improve raw MSE while worsening PNG MSE. This traces a
delivery difference; it is not a claim that quantization alone causes all model
defects, a reason to weaken the gates, or proof of visible identity preservation.

{link('Signed slopes','outputs/cctv_dgp_bank_group_step_check_v1_analysis/signed_derivative_review.json')}.
{link('Actual rounded-step gradient recount','outputs/cctv_dgp_bank_group_step_check_v1_analysis/actual_step_direction_review.json')}.
{link('All400 PNG/raw processing recounts','outputs/cctv_dgp_bank_group_step_check_v1_analysis/quantization_review.json')}.

## Independent saved-evidence checks and visual outcome

Archive bytes:{imported['archive_bytes']:,}; SHA256
`{imported['archive_sha256']}`. All596 returned files and5,852 parent
bindings passed. The separate R1 saved-output audit verifies100 exact control
receipts,400 raw arrays/400 exact floor PNGs,800 metric rows,17 source/profile
groups, four reconstructed disposable state hashes, and20 sheets/800 RGB cells.
Peak allocated CUDA memory:{audit['VM_peak_VRAM']/1024**3:.3f} GiB. This is an
independent audit of saved data, not a fresh neural reproduction of the proposals.

All20 sheets were actually inspected at requested original detail. Visible
eyes, lips, teeth, ordinary glasses and hair remain soft. The four small changes
are visually close to the retained restorer; there is no convincing useful
whole-face clarity gain. Narrow observed canvases preserve their side padding.
This is primary-assistant development review, not independent final human review.

The original prospective checker stopped on exact equality of derived squared
parameter-change sums. All148 parameter-byte hashes match; 835 of888 statistics
match exactly. The other53 differ only in three float64 reduction summaries,
by at most two representable rounding steps. The original source and failure
remain. A separate R1 checker permits that two-ULP reduction difference only;
counts, maxima, parameter/state hashes, image bytes and every quality gate remain
unchanged. Its inverse source/AST and rejection fixtures pass. NumPy explains
that summation precision depends on reduction details.
[NumPy summation reference](https://numpy.org/doc/stable/reference/generated/numpy.sum.html).

{link('R1 complete saved-output audit','outputs/cctv_dgp_bank_group_step_check_v1_analysis/independent_audit_r1.json')}.
{link('Original checker stop','outputs/cctv_dgp_bank_group_step_check_v1_analysis/original_parameter_statistics_audit_stop/failure.json')}.
{link('All20 actual visual observations','outputs/cctv_dgp_bank_group_step_check_v1_analysis/visual_review.json')}.
{link('Independent review readback','outputs/cctv_dgp_bank_group_step_check_v1_analysis/review_binding_audit.json')}.

## Next design and the user's five-item summary

The selected next design accounts for both raw and delivered losses, uses small
actual checked changes and refreshes directions as weights move. It retains the
current app checkpoint and the frozen initial replacement state; no random-weight
restart is justified. A prospective exact-PNG-forward/surrogate-backward design
needs a finite manually run L4 test before extra epochs. It is design work only,
not a sealed packet or launch-ready training recipe.
{link('Concrete next learning design review',design_name)}.

The user's summary accurately describes model diagnosis, fair comparison inputs,
functional app testing, cold-backup offload and a finite VM diagnostic. The
recorded storage offload freed52,028,568,639 logical bytes (**52.03 GB/48.45 GiB**)
locally; it did not change VM storage. Its recorded Drive audit retains4,457 files
and all active assets. No new cloud inventory is claimed by this model audit.
{link('Verified storage closure','GOOGLE_DRIVE_OFFLOAD_COMPLETE_20261013.md')}.

Fresh prior app checks verified14 actual PNG/ZIP downloads,10 backend refusals
and required responsive layouts. This is functional evidence, not model usefulness
or covering-family qualification.
{link('Bounded app flow','CCTV_DGP_APP_FLOW_AUDIT_20261012.md')}.

The separate completion saved-evidence recount verifies2,529 bindings/1,182 files
and exact outside-mask preservation across five retained comparisons. Among14
eligible covering photographs per condition, the retained automatic detector is
empty on4 original photos and11 degraded photos. An uncovered original control
also has11,683 false-marked pixels. Masks are approximate operator footprints,
not expert segmentation truth. Generated anatomy and outside-mask remnants remain
distinct problems; all seven automatic/assisted families stay unqualified.
{link('Saved completion readiness recount','outputs/dgp_covering_readiness_20261012_v1/audit.json')}.

## Goal and evidence boundaries

The full goal is unchanged and incomplete. The permanent app baseline SHA256 is
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
It is an own fine-tuned DeblurGAN-v2-compatible restorer. The proposed replacement
has an own conditioning/reconstruction path with an externally pretrained frozen
StyleGAN2 prior; there is no own-trained-prior claim. No model or app is promoted.
Original checkpoints, splits, full states, caches, provenance, rollback versions
and failures remain retained.

All five milestones still apply: native DEV/separate final identities and source
contracts; identical-input model comparisons with raw/display separation;
demonstrated finite improvements; qualified primary DGP with Auto/override;
independently useful DEV/final review, regressions and full app verification.
These100 paired photographic TRAIN cases are not native CCTV. No native PSNR/SSIM,
ethnicity, Zamboanga-performance or exact hidden-identity claim is made. Reserved
final identities remain outside tuning and their pixels remain unopened.

Separately retain completion for masks, sunglasses, strong glare, hands,
obstructing hair, scarves and objects. Show automatic removal support for optional
correction; preserve clear glasses, ordinary hair and visible appearance with a
documented small margin. Request clearer/less-covered input when insufficient.
Return original/mask/one plausible estimate with PNG and optional bundle. Automatic
and assisted results and independent final review remain separate and outstanding.

All actual training, gradients and live model proposals remain human-manual L4
tmux work. Agent VM access remains inventory-first hash-bound maintenance only.
This turn runs saved-data audits; it launches no VM or local neural job. Every
future pilot needs finite updates/epochs, timing/projection, VRAM/RSS, free-space,
reserve, output and export stops. Credits/migration notifications and the
November2026 thesis deadline remain. More epochs require demonstrated preserved
learning; the conditional five-epoch plan is not activated by this result.
'''
    design_text = f'''# Delivered-image learning design review — 13 October 2026

**Selected for design work; not a sealed VM packet or a launch-ready pilot.**
The signed test shows a favorable local raw descent tendency, finite curvature,
and delivery regressions at an otherwise raw-preserving +.0001 step. Extending
the old fixed direction or changing the model-quality gates is unsupported.
{link('Verified signed-step findings',report_name)}.

1. Retain the permanent app restorer and the pinned initial replacement state.
2. Include raw and exact delivered-image forward losses in the learning design.
3. Verify a matched initial raw/PNG forward before any new gradient measurement.
4. Accept an actual change only after the unchanged raw/PNG preservation checks.
5. Permit more epochs only after finite useful improvement passes the retained gates.

For the delivered branch, one design hypothesis is an exact floor-to-PNG forward
with explicitly declared surrogate backward. That backward is an approximation;
it is not the derivative of a discontinuous PNG converter. Straight-through
estimators are heuristic gradient constructions in the primary literature; that
work does not establish efficacy for our face-restoration workflow.
[Bengio, Léonard and Courville](https://arxiv.org/abs/1308.3432).
The choice must be tested rather than treated as a successful fix.

The candidate design must preserve clear-image replay, both sources and all
visible features. Raw and PNG source/profile MSE, SSIM and fixed observed ArcFace
checks remain separate; brightness fraction stays at most0.2, and structure gain
stays at least1% globally and per source at the retained qualification checkpoint.
An intermediate favorable derivative or loss cannot bypass the final requirements.
No fixed direction is assumed valid after weights change. Failed proposals are
retained and rolled back; no negative diagnostic state becomes an incumbent.

A future finite manual L4 packet must freeze its permitted gradients/updates,
input cases and seeds, checkpoint/source hashes, exact baseline controls, timing
projection and outer timeout, VRAM/RSS, free-space/reserve/output/export stops,
checkpoint/export contents and prospective return checker. If a delivered-aware
actual step cannot preserve the frozen evidence, retain the stop before a longer
run. No automatic historical pilot, unchanged failed rerun or random-weight
training is permitted. Training remains manual at `~/forensic-dgp`.

Only audited clean TRAIN references and replay may train; native CCTV DEV without
an aligned reference stays unpaired, and final identities remain outside tuning.
Full TRAIN and useful native DEV qualification, independent final review and the
existing DGP-led Auto/override app still follow any successful finite learning test.
The frozen StyleGAN2 component remains external; only our declared reconstruction
and conditioning weights may learn. Completion remains a separate seven-family
task with reviewed masks and visible preservation, not a reason to claim exact
hidden identity or to enlarge a removal area from model output.

No new inference, gradient, optimizer update, checkpoint, VM launch, model adoption
or extra-epoch qualification follows from this review.
{link('Bound next-design record','outputs/cctv_dgp_bank_group_step_check_v1_analysis/next_design_review.json')}.
'''
    new(ROOT/report_name, report)
    new(ROOT/design_name, design_text)
    prefix = f'''# Latest model evidence — 13 October 2026; signed diagnostic audited

The manual L4 signed test completed in58.14s with100 candidate/160 recognizer
forwards, four disposable proposals, zero gradients/updates/epochs and no new
checkpoint. All596 returned/5,852 parent bindings,100 controls,400 raw/400 PNG
outputs,800 metrics and20 sheets/800 cells pass the retained saved-evidence audit.
All20 sheets were inspected; convincing whole-face clarity gain is absent.
The original exact-summary checker stop is retained. A distinct R1 handles only
two-ULP float64 squared-sum reductions; parameter/image hashes and gates are exact.

At +.0001 raw preservation has0 failed checks and0.009962% structure gain;
the delivered PNG has6 failed checks and0.018381% gain. At +.001 raw/PNG gains
are0.096543%/0.107710% with6/9 failed checks. Every proposal remains unqualified.
All32 central raw slopes favor descent at both radii, while finite curvature and
PNG quantization still matter. No specific broken layer or qualified epoch
continuation follows. {link('Complete findings',report_name)}.

Next design: include both raw and delivered forward losses, use actual small
checked changes and refresh directions as weights move. A surrogate backward is
only a hypothesis needing manual finite verification. No new packet is sealed
or launch-ready. {link('Next learning design review',design_name)}.
The used signed-test guide is historical; do not rerun its immutable recipe.

The recorded Drive offload remains52.03GB/48.45GiB of LOCAL cold-backup copies;
VM storage was unchanged. Active checkpoints/splits/caches/provenance/failures
remain retained. {link('Storage closure','GOOGLE_DRIVE_OFFLOAD_COMPLETE_20261013.md')}.
The bounded app checks are functional evidence; useful restoration and all seven
automatic/assisted covering families remain unqualified. The saved completion
recount has4/14 original and11/14 degraded covering cases with empty proposals.

The full goal remains unchanged and incomplete: permanent baseline SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b; all five
milestones; primary qualified DGP Auto/override; preserved visible structure and
design; separate unpaired native and paired synthetic evidence; untouched final
identities and independent final review. No ethnicity, Zamboanga or exact hidden
identity claim. All seven families, editable automatic mask, automatic/assisted
separation, clear glasses/ordinary hair, documented margin, clearer/less-covered
requests, original/mask/one estimate and PNG/optional bundle stay required.
Training/gradients/live proposals stay human-manual L4 tmux; agent VM access stays
hash-bound maintenance only. Finite stops, credits/migration policy and November
deadline remain. Conditional extra epochs are not activated. Original bytes follow.

'''
    backup = ANALYSIS/'publication_before'
    backup.mkdir()
    documents = {}
    for name in names:
        documents[name] = prepend_expected(ROOT/name, prefix, backup/name, expected[name])
    guide_prefix = f'''# Completed signed diagnostic — historical commands, do not rerun

The user manually completed this immutable test. Its returned archive is audited;
all four proposals remain unqualified. +.0001 preserves raw checks but fails6
delivered checks and remains below1% structure. No training epoch or checkpoint
was created. {link('13 October result and next design',report_name)}.
The original prepared/unrun status and commands below remain as historical bytes.

'''
    documents[guide] = prepend_expected(ROOT/guide, guide_prefix, backup/guide, expected[guide])
    write(ANALYSIS/'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__), proof_sha256={name:sha(ANALYSIS/name) for name in proof_names},
        next_design_review_sha256=sha(ANALYSIS/'next_design_review.json'), documents=documents,
        new_documents_sha256={report_name:sha(ROOT/report_name), design_name:sha(ROOT/design_name)},
        latest_storage_publication_sha256=sha(ROOT/'outputs/google_drive_offload_20261012_v1/closure_20261013_v1/publication_audit.json'),
        original_app_checkpoint_protected_sha256=protected,
        no_model_or_app_source_changes=True, actual_training_run=False, new_packet_sealed=False,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, result_report=report_name, design_review=design_name,
               preserved_status_documents=len(documents), model_qualified=False))


if __name__ == '__main__':
    main()
