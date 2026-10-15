"""Publish an audited failed pilot and the evidence-backed design decision."""
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage = ANALYSIS / 'publication'
    assert not (stage / 'publication.json').exists()
    imported = read(ANALYSIS / 'import.json')
    audited = read(ANALYSIS / 'failure_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    prediction = read(ANALYSIS / 'gradient_prediction_review.json')
    decision = read(ANALYSIS / 'diagnostic_decision.json')
    regression = read(ANALYSIS / 'receipt_check_green.json')
    assert all(v['complete'] for v in [imported, audited, visual, prediction, decision, regression])
    assert audited['completed_trials'] == 3 and audited['actual_update_count_reported'] == 0
    assert visual['all20_sheets_and600_cells_directly_viewed'] and len(prediction['real_group_trial_comparisons']) == 192
    assert len(regression['corruptions_rejected']) == 18
    assert decision['human_answer'] == 'apply best approach'
    failure = audited['failure']
    assert failure['counts'] == dict(B=100, recognizer=200, gradient_queries=640,
                                      backwards=0, accepted_updates=0, parameter_proposals=3)
    assert failure['message'] == 'No actual raw/PNG preserving conditioning step; retain stop'
    assert not list((RETURN / 'outputs/conditioning_learning_v1_r4').glob('accepted*'))
    previous = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_analysis/publication/publication.json')
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    used = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_VM.md'
    expected = {name: previous['documents'][name]['after_sha256'] for name in docs}
    expected[used] = previous['new_documents_sha256'][used]
    for name, expected_sha in expected.items():
        assert sha(ROOT / name) == expected_sha or ((stage / 'before' / name).exists()
               and sha(stage / 'before' / name) == expected_sha), 'Preserve concurrent edits'
    report_name = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_RESULTS.md'
    design_name = 'CCTV_DGP_POST_R4_GRADIENT_DESIGN_REVIEW_V1.md'
    table = []
    for row in prediction['trials']:
        raw, png = [row['detail_and_gates'][s] for s in ['raw', 'png']]
        table.append(f"| {row['trial']} | {raw['actual_global_detail_gain'] * 100:.6f}% | "
                     f"{png['actual_global_detail_gain'] * 100:.6f}% | "
                     f"{raw['actual_preservation_failed_checks']} / {png['actual_preservation_failed_checks']} | Rejected |")
    report = f'''# Conditioning learning R4 returned result — 14 October 2026

**No trial qualified and no training update was accepted.** R4 completed the
previously failing scientific-loss alignment stage, all 640 gradient queries,
and three actual temporary conditioning changes. Every change failed retained
RAW/PNG preservation. The worker stopped at
`scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py:364` with
`No actual raw/PNG preserving conditioning step; retain stop`.

| Trial | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG | Decision |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table)}

The largest trial still missed the 1% global and per-source endpoint. Its two
source gains were 0.635030% / 0.701935% RAW and 0.639203% / 0.722234% PNG.
Reducing the step improved preservation but left 13 RAW and 8 PNG failed checks
at the smallest trial. Brightness fractions stayed below 0.2 in all trials.
The failures include clear-image preservation, SSIM and the fixed-recognizer
appearance comparison. Recognizer similarity is not verified identity.

The manual L4 worker reports {failure['seconds']:.2f} seconds, 100 candidate
forwards, 200 recognizer forwards, 640 gradient queries, three parameter
proposals and zero accepted updates/epochs. The export took 80.13 seconds;
`complete: true` means the failure archive was created. No accepted component
checkpoint or new incumbent exists. All trial components and outputs remain.
The exception handler reports no rollback error; no fresh post-failure RAM-state
hash is claimed.

All {imported['returned_file_bindings']} returned files and 5,852 required parent
bindings pass hash checks. Archive size is {imported['archive_bytes']:,} bytes;
SHA256 `{imported['archive_sha256']}`. The separate R1 checker independently
recomputes four snapshots/400 RAW and PNG model outputs/800 metric rows, visible
support compositing, selected-component state schemas, all 64 saved derivative
vectors, Gram/KKT arithmetic, rounded changes, rates and all three rejections.
Initial RAW controls equal the permanent DGP exactly. No neural derivative or
embedding inference was repeated locally.
{link('Independent returned audit', 'outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/failure_audit.json')}.

The original checker stopped on exact equality of derived receipt floats.
Four-snapshot replay reproduced differences up to 4.8225313e-13 while scientific
image-score replay passed its original 1e-13 bound. R1 binds the stored receipt
exactly to stored scientific scores and requires every independent quality
branch/failure member to agree exactly. It adds no scientific or replay
tolerance. Three real receipt regressions pass; 18 corrupted records are
rejected. Original checker, stopped check and failed regression remain retained.
{link('Checker repair evidence', 'outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/return_audit_r1_preparation.json')}.

All 20 full-resolution sheets and 600 cells were directly inspected. The clear
images retain softened eye/lip/teeth/hair detail; degraded features remain
diffuse. The three trials supply no convincing whole-face clarity advantage.
This is 20 exposed photographic TRAIN references and 100 synthetic paired
cases, not native CCTV, independent final identities or completion evidence.
No ethnicity, Zamboanga performance or hidden-identity claim is supported.
{link('Every-sheet visual review', 'outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/visual_review.json')}.

The invalid assumption was that a feasible local descent certificate and a
small relative parameter change establish finite image preservation. After the
third stop, the diagnostic question was asked; the human answered “apply best
approach.” The saved-data review is now complete: float32 parameter rounding
did not reverse any predicted descent sign, yet actual losses worsened.
The joint 64-gradient Gram condition number is 884,961,456, compared with
4,164 for RAW32 and 4,138 for PNG32. This identifies a loss/step-design hypothesis,
not proof of a unique cause or a new passing model.
{link('192 saved prediction-versus-output comparisons', 'outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/gradient_prediction_review.json')}.
{link('Design decision and primary research', design_name)}.

Original checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
R4 is used and stopped; do not rerun or resume its failed recipe. New training,
longer epochs, app promotion and completion qualification are not authorized by
archive completion. Actual learning stays human-manual on the existing L4.
All five milestones and seven automatic/assisted covering families remain
required; the full goal is active and incomplete.
'''
    design = f'''# Post-R4 gradient and learning design review — 14 October 2026

The immediate improvement target is the update rule's finite preservation,
before additional epochs. The permanent original model remains the incumbent.
The human's “apply best approach” response selected a diagnosis of prediction
versus actual output; no new training was launched during this review.

## What the saved experiment establishes

The 192 saved comparisons cover 64 source/profile/stage derivative groups at
three executed rates. Independent arithmetic uses each actual rounded component
change, not just rate times the direction. No predicted descent sign changes
after rounding. Relative rounding error is 0.00175%, 0.00350% and 0.00699% of the
intended step. At the smallest step, 9 RAW and 6 PNG derivative-group predictions
reverse beyond the original metric regression tolerances. These counts differ
from the 13/8 scientific gate failures because derivative groups and reporting
groups have different membership.

At the largest step, global structure prediction is 1.630645% RAW and 1.629864%
PNG, versus 0.688521% / 0.705580% actually delivered. At the smallest step the
prediction is 0.407661% / 0.407466%, versus 0.345093% / 0.359206%. Smaller steps
reduce this gap but do not establish preserved learning or an untested safe
step. All three trial states remain rejected.

The RAW32 and PNG32 Gram condition numbers are approximately 4,164 and 4,138;
joint64 is approximately 885 million. Paired RAW/PNG gradient cosines range
0.941681–0.999999658. The joint certificate itself passes its stated rank,
KKT and float32-direction checks. Conditioning, approximate derivatives and
finite nonlinear response remain distinct hypotheses. An ill-conditioned Gram
does not by itself prove a wrong derivative or justify weakening image gates.
{link('Recomputed vectors, rounded changes and actual scores', 'outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/gradient_prediction_review.json')}.

## Research boundary

Straight-through estimators supply surrogate gradients through quantization.
Theoretical usefulness depends on the estimator and assumptions; the cited
two-layer/Gaussian-input analysis does not validate our deep face model or its
PNG surrogate. Its findings support checking finite outputs rather than
treating a surrogate certificate as image preservation.
[Yin et al., ICLR 2019](https://arxiv.org/abs/1903.05662).

PyTorch documents numerical-versus-analytic gradient checks, warns that default
checks assume double precision, and describes failures at nondifferentiable
points. A blanket default gradcheck on our FP32 model or quantized PNG loss
would not diagnose this saved result reliably. Any future neural derivative
check stays manual on the L4 and must record precision, displacement, stochastic
state and the scientific forward being tested.
[PyTorch 2.9 gradcheck documentation](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.gradcheck.gradcheck.html).

## Next finite design to verify before training

The leading hypothesis is that jointly constraining the approximate RAW/PNG
derivatives produces an unnecessarily sensitive update direction. Compare
optimization of the existing 32 RAW loss constraints with the retained R4 joint
route, while keeping delivered PNG as a hard actual acceptance test. This is a
proposed optimizer comparison, not a proved repair or a runnable packet.

Keep the same observed inputs, target references, masks, 12 conditioning
tensors, initial state, frozen original decoder/readouts/StyleGAN2 prior/
recognizer, source separation and 256 geometry. The external prior remains
attributed; it is not an own-trained face prior. The old R4 “retain all64 loss
dimensions” recipe remains immutable and failed. A new recipe must explicitly
declare the changed optimizer constraint set rather than claiming R4 unchanged.

For every actual proposed change, retain all 17 scientific groups for RAW and
all 17 for PNG, clear-image/appearance preservation, positive global and
per-source detail, brightness fraction <=0.2, exact protected pixels and the
cumulative 0.1% parameter-distance bound. The finite endpoint still requires
1% global and per-source structure gain in both delivered forms. No PNG
measurement or acceptance gate may be removed or relaxed.

Before issuing a manual transfer, construct a separate immutable protocol,
compare useful exposure with the corrected-current-DGP route, verify the new
solver on declared synthetic arithmetic fixtures, reject the three actual R4
states again, check initial parity and namespaces, and build an independent
returned-array/state/RAW/PNG checker. Runtime/resource limits must be grounded
in measured R4 timings and include finite updates/trials, projection, VRAM,
storage/reserve/output and export stops. No new solver, direction or parameter
proposal was computed locally in this saved-data review.

If the finite comparison cannot preserve useful improvement, stop it and
review the data/conditioning assumptions. Do not append epochs to R4. Only a
qualified development version becomes the next incumbent; then separately
freeze the conditional maximum-five-epoch study already recorded in the plan.
Full TRAIN's 10% gate, audited native DEV, reserved final identities and
independent final/app review remain later requirements. Photographic paired
metrics do not establish unpaired CCTV performance or local performance.

Completion remains separate: masks, sunglasses, glare, hands, obstructing hair,
scarves and objects need automatic removal preview, optional correction,
preservation, original/mask/one estimate and downloads, with automatic versus
assisted qualification reported separately. Neither a better gradient nor extra
restoration epochs establishes this scope. Goal remains active/incomplete.
'''
    retained_new(ROOT / report_name, report)
    retained_new(ROOT / design_name, design)
    prefix = f'''# R4 actual preservation stop audited; gradient design reviewed — 14 October 2026

R4 completed 640 gradient queries and three temporary conditioning changes in
237.05 seconds, accepting zero updates. Best gain is 0.688521% RAW / 0.705580%
PNG, below 1%; preservation failures are 31/31, 25/25 and 13/8. All 663 returned
files and 5,852 parent bindings pass. Four snapshots, component states, all64
saved derivative vectors/KKT arithmetic and every RAW/PNG row are checked.
All20 sheets/600 cells were directly reviewed; no convincing whole-face gain.
{link('Audited results and retained checker stop', report_name)}.

An exact-float receipt-comparison stop is retained. The separate R1 checker
binds receipts exactly to saved scores and keeps all independent gate decisions
exact. Scientific replay bounds and quality thresholds are unchanged;18 corrupt
records are rejected. It verifies the pilot failure, not model usefulness.

After the third stop, the diagnostic question was asked and the human answered
“apply best approach.” Saved-data review finds no predicted-descent sign reversal
from parameter rounding, but finite outputs still regress. Joint RAW/PNG Gram
conditioning is about885 million versus about4,164 for RAW32. This is a design
hypothesis, not proof. Next compare RAW optimization with actual RAW/PNG gates
in a separately verified finite recipe; no new runnable packet or VM launch yet.
{link('Design review, scientific boundaries and primary research', design_name)}.

R4 is used/stopped; do not rerun or continue it. Original baseline/checkpoints,
app design, splits/caches/provenance/full states/failures and rollback versions
remain. No accepted incumbent or longer epochs are qualified. Paired TRAIN,
native unpaired evidence and reserved final identities remain separate. All five
milestones and seven automatic/assisted covering families remain required.
Training stays human-manual L4 tmux; no agent VM action or neural training
occurred in this review. Full goal active/incomplete. Exact history follows.

'''
    banner = f'''# Used R4 pilot — failed actual preservation; do not rerun

Returned results are independently audited:640 gradient queries,three rejected
changes,zero accepted updates. All RAW/PNG gates and original checkpoints remain.
{link('Result and why it stopped', report_name)}.
{link('Next optimizer-design review; not a runnable packet', design_name)}.
The prior guide is retained below as exact historical command evidence.

'''
    changed = {}
    (stage / 'before').mkdir(exist_ok=True)
    for name in docs:
        changed[name] = finish_prepend(ROOT / name, prefix, stage / 'before' / name, expected[name])
    changed[used] = finish_prepend(ROOT / used, banner, stage / 'before' / used, expected[used])
    evidence_names = ['import.json', 'failure_audit.json', 'visual_review.json', 'gradient_prediction_review.json',
                      'diagnostic_decision.json', 'original_checker_stop.json', 'receipt_equality_red.json',
                      'receipt_check_red.json', 'receipt_check_green.json', 'return_audit_r1_preparation.json']
    write(stage / 'publication.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), publisher_sha256=sha(__file__), documents=changed,
        new_documents_sha256={n: sha(ROOT / n) for n in [report_name, design_name]},
        evidence_sha256={str((ANALYSIS / n).relative_to(ROOT).as_posix()): sha(ANALYSIS / n) for n in evidence_names},
        returned_archive_sha256=imported['archive_sha256'], returned_protocol_sha256=PIN,
        returned_accepted_updates=0, actual_rejected_trials=3, actual_manual_gradient_queries=640,
        all20_sheets_and600_cells_directly_reviewed=True, saved_group_trial_predictions_reviewed=192,
        diagnostic_question_asked_and_human_best_approach_answer_recorded=True,
        proposed_RAW_optimization_and_actual_PNG_gates_is_hypothesis_not_runnable=True,
        original_recipe_and_checker_failure_retained=True, original_scientific_bounds_unchanged=True,
        no_new_incumbent_or_epochs_or_app_adoption=True, no_agent_VM_actions_or_new_neural_training=True,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, returned_updates=0, rejected_trials=3, reviewed_sheets=20,
               history_documents_preserved=len(changed), new_runnable_packet=False, goal_complete=False))


if __name__ == '__main__':
    main()
