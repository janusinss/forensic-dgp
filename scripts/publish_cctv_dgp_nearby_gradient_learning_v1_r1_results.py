"""Publish fully audited rejected proposals, image review and the required design question."""
from datetime import datetime,timezone

from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import ROOT,OUT,RETURN,ANALYSIS,PIN,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend,retained_new


def main():
    stage=ANALYSIS/'publication';(stage/'before').mkdir(parents=True,exist_ok=True)
    assert not (stage/'publication.json').exists()
    files=[ANALYSIS/n for n in ['import.json','failure_audit.json','finite_response_review.json','visual_review.json','retained_failure_regression.json']]
    imported,audited,analysis,visual,regression=[read(f) for f in files]
    assert all(read(f)['complete'] for f in files)
    assert audited['accepted_trials']==[] and audited['trials_replayed']==3 and not audited['completed_pilot']
    assert audited['complete_gradient_sites']==['initial','plus02','minus02'] and audited['certificate_replay']['complete']
    assert visual['all20_sheets_and900_cells_directly_viewed'] and len(analysis['initial_RAW_group_trial_comparisons'])==96
    assert not audited['all_original_initial_parity_checks_pass'] and regression['actual_original_parity_miss_reproduced']['mismatched_elements']==1
    failure=audited['failure'];assert failure['message']=='No actual raw/PNG preserving nearby-state proposal; retain stop'
    assert failure['counts']==dict(B=200,recognizer=340,gradient_queries=960,backwards=0,accepted_updates=0,parameter_proposals=3,derivative_sampling_displacements=2)
    out=RETURN/'outputs/nearby_gradient_learning_v1';assert not list(out.glob('accepted*'))
    prior=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_publication/publication.json')
    expected={n:b['after_sha256'] for n,b in prior['documents'].items()};expected.update(prior['new_documents_sha256'])
    assert len(expected)==10 and all(sha(ROOT/n)==h for n,h in expected.items())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    question='The joint, RAW-only and nearby-gradient recipes all failed preservation. Before another VM pilot, which cause should the design review investigate first? Both directions retain every visible feature and the original quality gates.'
    question_path=ANALYSIS/'diagnostic_question.json'
    write(question_path,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),question=question,
        options=['Review the reconstruction path and target detail (Recommended)','Continue investigating loss gradients and PNG conversion'],
        user_question_sent=True,human_answer_pending=True,human_answer=None,
        source_rule_path='AGENTS.md',source_rule_sha256=sha(ROOT/'AGENTS.md'),
        source_rule='If an issue remains unresolved after 3 consecutive attempts, STOP modifying code. Name the invalid assumption and ask one diagnostic question.',
        related_unsuccessful_recipes=['joint RAW/PNG conditioning R4','RAW-only conditioning V1 R1','nearby-gradient conditioning V1 R1'],
        invalid_assumption='Certified derivative descent in the current conditioning path is sufficient to yield a finite preserving structure improvement.',
        training_recipe_modification_stopped=True,evidence_audit_and_publication_completed=True,
        original_goal_not_redefined=True,goal_complete=False))
    files.append(question_path)
    report_name='CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_R1_RESULTS.md'
    design_name='CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md'
    table=[]
    for t in analysis['trials']:
        r,g=t['stages']['raw'],t['stages']['png']
        table.append(f"| {t['trial']} | {100*r['global_structure_gain']:.6f}% | {100*g['global_structure_gain']:.6f}% | {r['failed_checks']} / {g['failed_checks']} | Rejected |")
    best=analysis['trials'][0];raw=best['stages']['raw'];png=best['stages']['png']
    sources=[]
    for s in sorted(raw['source_structure_gains']):
        sources.append(f"| {s.removesuffix('/degraded')} | {100*raw['source_structure_gains'][s]:.6f}% | {100*png['source_structure_gains'][s]:.6f}% |")
    counts=[t['counts']['actual_RAW_preservation_losses_failed'] for t in analysis['trials']]
    parity=next(r for r in audited['original_initial_gradient_parity_reports'] if not r['original_parity_check_pass'])
    point=parity['violations'][0];exported=read(OUT/'cctv-dgp-nearby-gradient-learning-v1-r1-export.json')
    report=f'''# Nearby-gradient learning V1 R1 result — 14 October 2026

**The VM completed its finite computation, but all three proposals were
rejected. Zero updates were accepted and no improved checkpoint exists.** The
best measured structure gain was **0.687495% RAW / 0.706917% PNG**, below the
required 1% global and both-source endpoint. It also failed 32 preservation
checks in each representation. These are group–metric checks, not 32 faces.

| Trial | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG | Decision |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table)}

Largest-trial source results remain separate:

| Photographic TRAIN source | RAW structure gain | PNG structure gain |
| --- | ---: | ---: |
{chr(10).join(sources)}

This percentage is reduction in mean degraded landmark high-frequency error
on the same 100 exposed paired photographic TRAIN cases. It is not a fraction
of identity recovered, native CCTV performance or Zamboanga evidence. Twenty
references / five profiles were fixed; no native crop or reserved-final
identity entered optimization. Dataset names do not establish ethnicity.

## Location, cause and corrective action

Location: returned `scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py:394`,
`No actual raw/PNG preserving nearby-state proposal; retain stop`.
Cause of the recorded stop: none of the three actual outputs passed all
original preservation requirements. The largest proposal failed four pixel-
error, twelve SSIM and sixteen fixed-recognizer appearance checks on each
stage. The smallest still failed 14 RAW / 9 PNG checks. All proposals stayed
inside the original physical distance and brightness bounds. No unique
architectural, numerical or loss-gradient root cause is proved by this stop.
Corrective action: retain rejection and rollback; stop further training-code
changes for the required diagnostic discussion before another recipe. Adding
epochs or changing only the learning rate is not justified by these outputs.

The manual L4 worker took **{failure['seconds']:.2f}s** and export took
**{exported['seconds']:.2f}s**. It reports 200 restoration forward batches,
340 recognizer forward batches, 960 derivative queries, two derivative-sampling
displacements and three new proposals. Zero accepted updates, full epochs,
native pixels or final-identity pixels. No continuation or app change ran.
The archive's `complete: true` means export completion, not pilot success.

## What the nearby derivatives changed

All 96 original normalized dot constraints and the full Gram/dual/float32
certificate pass independent saved-vector replay. Numerical span rank is 96;
no inequality, source or loss was removed and no ridge was used. The Gram
condition number is approximately {analysis['normalized_Gram_resolved_condition_number']:.0f}.
That sensitivity does not itself prove a wrong derivative or explain all
finite failures. The actual solve returned a certified direction in 3.865s.

The new direction's cosine with the failed RAW-only direction is
**{analysis['previous_RAW32_direction_cosine']:.9f}**. Inference: the three sampled
regions changed the tested update direction only slightly. The new sampling
recipe therefore did not produce a materially different direction in this
run, although the design and measured derivative inputs were new.

For the largest proposal, the initial derivative predicted **1.630269% RAW**
structure gain; actual RAW gain was **0.687495%**. Actual rounded displacement
dots predict descent for all 96 sampled constraints in each trial. Yet
{counts[0]}/{counts[1]}/{counts[2]} of the 30 initial RAW preservation-loss groups
regress beyond their original tolerances. These derivative-group counts differ
from the overlapping 17-group scientific reporting counts in the table. Dots
from a nearby sampling state are local directional information, not predictions
of the entire initial-to-proposal finite output change.
{link('All saved predictions and actual outputs','outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/finite_response_review.json')}.

The cited gradient-sampling paper uses random nearby gradients, a quadratic
program and a line search under stated assumptions. Our deterministic,
multi-constraint experiment differs; its theorem is not a RAW/PNG face-
preservation guarantee. The result rejects this particular finite recipe,
not the paper or every possible sampling approach.
[Burke, Lewis and Overton, primary author-hosted paper](https://cims.nyu.edu/~overton/papers/pdffiles/gradsamp.pdf).

## Independent verification and retained repeatability failure

All **{imported['returned_file_bindings']:,} return-file**, **{imported['parent_file_bindings']:,} bank**
and **{imported['response_file_bindings']:,} response** bindings pass. Return archive:
{imported['archive_bytes']:,} bytes, SHA256 `{imported['archive_sha256']}`.
Returned source was not executed locally. The prospective checker replays
seven snapshots, 700 RAW / 700 PNG outputs, 1,400 stage metric rows, observed
pixel composition, scientific aggregates, saved embeddings, component byte
schemas, three complete derivative rounds, all 96 gradient aggregates, the
certificate, actual rounded rates/proposals and exact decisions. Neural
derivatives and embedding inference were not repeated locally.
{link('Independent failed-pilot audit','outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/failure_audit.json')}.

One original cross-run elementwise derivative bound still fails, now in
`{parity['key']}` at index {point['index']:,}. Actual difference is
{point['absolute_difference']:.10g}, above its unchanged
{point['original_bound']:.10g} bound. All 32 vector-norm checks pass; the failed
vector's relative difference is {parity['relative_vector_difference']:.10g}.
The prospective checker reports this separately; its overall audit completion
is not an all-parity pass or model qualification. No bound was relaxed and no
cause was assigned to CUDA nondeterminism. A saved-data regression reproduces
the exact miss and rejects erased gate failures or false promotion for all
three real trials.
{link('Actual failures remain rejected','outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/retained_failure_regression.json')}.

Initial and rollback RAW arrays equal the original DGP output by bits; PNG
pixels and cached control scores also agree. Saved Torch CPU/CUDA RNG arrays
restore exactly. Fresh VM receipts record original, recognizer, frozen and
buffer hashes unchanged and successful failure restoration. These runtime
hash receipts are distinguished from independently replayed saved arrays.
The permanent original checkpoint and twelve protected local files remain
exact. No accepted full-state file exists.

## All images reviewed and next requirement

All **20 original-resolution sheets / 900 cells** were directly inspected:
input, target, initial, both derivative sampling states, all three proposals
and rollback; five rows each. No convincing useful whole-face clarity gain
over initial is established. Eye, mouth, teeth, hair and transparent-glasses
detail remain soft. Some lower-resolution source targets are already soft;
the sharp-target examples still show missing visible detail in the outputs.
Hands, ordinary hair and head coverings in these restoration inputs are
observed content; they are not hidden-region completion evaluations.
{link('Every-sheet visual review','outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/visual_review.json')}.

The joint, RAW-only and nearby-gradient recipes have not established a
preserving useful update. The invalid assumption to retire is that a certified
local direction in this conditioning path suffices for useful finite outputs.
The required diagnostic question was sent; its answer is pending. No next
training recipe is selected, prepared or launched from this result.
{link('Post-result discussion boundary',design_name)}.

Keep the original app DGP as the permanent baseline. Its architecture is
DeblurGAN-v2-compatible; the comparison's StyleGAN2 bank is externally
pretrained and frozen. No own-trained prior or replacement qualification is
claimed. Original checkpoints, splits, caches, provenance, full states and
failures remain. Real native CCTV stays unpaired; synthetic paired evidence
and final identities stay separate. All five milestones and seven automatic/
assisted covering families remain required. More epochs and app promotion
remain unqualified. Actual training/gradients/updates stay human-manual L4
tmux; direct agent VM access stays inventory-first, hash-bound maintenance
only. Detector-confidence recording remains prepared and UNRUN. Goal incomplete.
'''
    design=f'''# Post nearby-gradient learning review — 14 October 2026

**This learning recipe is closed and failed. Stop training-code changes for
the required diagnostic discussion.** The audit and direct image review are
complete; no accepted update, replacement checkpoint or epoch continuation
exists. The original DGP remains protected and the full goal is active.
{link('Measured result and exact failures',report_name)}.

The latest best gain is 0.687495% RAW / 0.706917% PNG, with 32 preservation
failures per representation. Two smaller proposals also fail. All 96 sampled
local dot constraints pass, yet the actual finite output gains and appearance
do not. The direction remains almost parallel to the failed RAW-only direction
(cosine 0.999539308). The invalid assumption is that collecting these nearby
derivatives would yield a sufficiently different, preserving update.

The original question, assumptions and failed experiments remain binding.
The same 100 paired TRAIN exposures and targets cannot establish native CCTV
or independent final performance. No new automatic selection or blending
rule may be used merely to hide failed face preservation; future selection
requires its own input-only development evidence.

## Required diagnostic question

{link('Workspace rule','AGENTS.md')} requires: “If an issue remains unresolved
after 3 consecutive attempts, STOP modifying code. Name the invalid assumption
and ask one diagnostic question.” The three related failed recipes are joint
RAW/PNG R4, RAW-only V1 R1 and nearby-gradient V1 R1. The human's earlier “apply
best approach” responses authorized earlier reviews; the new three-attempt
discussion must still be asked and its actual answer retained.

The question sent is:

> {question}

Recommended discussion direction: review the reconstruction path and target
detail. Alternative: continue investigating loss gradients and PNG conversion.
The human answer is currently pending. This is a diagnostic discussion, not
approval of a concrete new training recipe or model. No fourth recipe is
prepared while it is pending. Auditing, preserving and publishing the already
returned result are completed independently of that choice.
{link('Bound question record','outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/diagnostic_question.json')}.

## Evidence a subsequent design must resolve

1. Establish whether the current learned path can represent useful visible
   detail while preserving appearance, using retained architecture and saved
   signal/target evidence; do not infer this solely from nonzero gradients.
2. Audit genuine target resolution and visible detail across both existing
   sources before claiming more data is required. Acquire only for a measured
   gap with terms, provenance and overlap checks.
3. Compare against prior spatial/decoder failures as well as the recent bank
   conditioning route, so a new name does not repeat a failed recipe.
4. Keep all original actual RAW/PNG gates, brightness/distance limits, original
   checkpoint, failure records and TRAIN/DEV/final separation unchanged.
5. Freeze a materially justified finite design with source/component origins,
   meaningful regressions, resource/export limits and independent audit before
   issuing human-manual L4 commands.

The gradient-sampling paper motivates nearby derivatives and finite line
search under its assumptions. It does not prove our deterministic face-loss
recipe or hidden identity. Its primary text was refreshed for this review.
[Burke, Lewis and Overton](https://cims.nyu.edu/~overton/papers/pdffiles/gradsamp.pdf).
No universal derivative error, impossible target, or unique architectural
bottleneck has been proved by the latest finite failure.

No app, model, loss, mask, checkpoint or threshold was changed by this review.
Preserve visible structure and appearance, existing app design and insufficient-
information handling. Native unpaired/synthetic paired evidence, untouched
final identities, all five milestones and all seven automatic/assisted
covering families remain. Training and neural parameter changes stay manual
in tmux on the existing L4; agent VM access remains maintenance only. Goal
incomplete; no automatic historical retry or new epoch launch is authorized.
'''
    retained_new(ROOT/report_name,report);retained_new(ROOT/design_name,design)
    prefix=f'''# Nearby-gradient V1 R1 audited and rejected — 14 October 2026

The manual L4 run completed its finite computations, but all three proposals
failed actual preservation. Best structure gain 0.687495% RAW / 0.706917% PNG,
below 1%, with 32 RAW / 32 PNG failed group–metric checks. Smaller trials fail
24/25 and 14/9. Zero accepted updates, full epochs or new incumbent.
{link('Result, cause of stop and full evidence',report_name)}.

All 1,144 return / 5,852 bank / 1,246 response bindings, seven snapshots,
700 RAW / 700 PNG outputs, 1,400 metric rows, all 96 local derivative constraints
and actual gate decisions pass saved-evidence auditing. One original
elementwise cross-run derivative bound still fails and remains reported;
all32 vector norms pass. Audit completion is not a model-quality pass.
All20 sheets/900 cells were directly reviewed; no convincing useful new clarity.
Initial/rollback outputs and saved RNG match; original/protected hashes remain.

New direction cosine with the failed RAW-only direction is 0.999539308.
Certified nearby descent has not established finite preserving improvement.
Joint, RAW-only and nearby learning attempts now require a diagnostic question
under AGENTS.md. Question sent; human answer pending. Stop training-code
changes while continuing completed evidence preservation and publication.
{link('Discussion boundary; no next training recipe',design_name)}.
USED / FAILED: retain this run; do not rerun it unchanged or append epochs.

No accepted checkpoint, app adoption, extra epochs or next manual pilot is
qualified. Preserve original checkpoints, splits, caches, provenance, full
states and gates. Actual training/gradients stay human-manual L4 tmux; agent
VM access stays inventory-first hash-bound maintenance only. Native unpaired,
paired TRAIN and final roles, original app design, all five milestones and
seven automatic/assisted covering families remain. Detector-confidence work
is prepared only and UNRUN. Goal incomplete. Exact prior document follows.

'''
    bindings={}
    for n,h in expected.items():bindings[n]=finish_prepend(ROOT/n,prefix,stage/'before'/n,h)
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        protocol_sha256=PIN,documents=bindings,new_documents_sha256={n:sha(ROOT/n) for n in [report_name,design_name]},
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},returned_pilot_used_and_failed=True,
        human_diagnostic_question_pending=True,next_training_recipe_not_selected_or_prepared=True,
        original_goal_preserved=True,local_neural_or_training_calls=0,automatic_VM_launch=False,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=len(bindings),new_reports=2,accepted_updates=0,human_diagnostic_question_pending=True))


if __name__=='__main__':main()
