"""Publish the rejected manual comparison, retaining checker and research history."""
from datetime import datetime, timezone

from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage = ANALYSIS / 'publication'; stage.mkdir(exist_ok=True)
    assert not (stage / 'publication.json').exists()
    imported = read(ANALYSIS / 'import.json')
    audited = read(ANALYSIS / 'failure_forensic_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    prediction = read(ANALYSIS / 'gradient_prediction_review.json')
    stopped = read(ANALYSIS / 'original_checker_stop.json')
    regression = read(ANALYSIS / 'cross_run_report_check.json')
    assert all(v['complete'] for v in [imported,audited,visual,prediction,stopped,regression])
    assert audited['actual_update_count_reported']==0 and audited['completed_trials']==3
    assert not audited['cross_run_derivative_parity_pass'] and not audited['original_prospective_checker_passed']
    assert visual['all20_sheets_and600_cells_directly_viewed'] and len(prediction['RAW_group_trial_comparisons'])==96
    failure=audited['failure']
    assert failure['counts']==dict(B=100,recognizer=180,gradient_queries=320,backwards=0,accepted_updates=0,parameter_proposals=3)
    assert failure['message']=='No actual raw/PNG preserving conditioning step; retain stop'
    assert not list((RETURN/'outputs/raw_conditioning_learning_v1_r1').glob('accepted*'))
    previous=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
          'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    used='CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_VM.md'
    expected={n:previous['documents'][n]['after_sha256'] for n in docs}
    expected[used]=previous['new_documents_sha256'][used]
    for name,h in expected.items():
        backup=stage/'before'/name
        assert sha(ROOT/name)==h or (backup.exists() and sha(backup)==h), 'Preserve concurrent edits'
    report_name='CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_RESULTS.md'
    design_name='CCTV_DGP_POST_RAW_CONDITIONING_REVIEW_V1.md'
    table=[]
    for t in prediction['trials']:
        r,g=t['stages']['raw'],t['stages']['png']
        table.append(f"| {t['trial']} | {r['actual_global_feature_gain']*100:.6f}% | {g['actual_global_feature_gain']*100:.6f}% | {r['failed_checks']} / {g['failed_checks']} | Rejected |")
    report=f'''# RAW conditioning R1 returned result — 14 October 2026

**All three trials were rejected; zero updates were accepted.** The manual L4
worker completed 320 RAW derivative queries and tested three finite changes of
the 12 style/spatial conditioning tensors. Every trial failed actual RAW/PNG
preservation. The stop is at `scripts/cctv_dgp_raw_conditioning_learning_v1_r1_vm.py:363`:
`No actual raw/PNG preserving conditioning step; retain stop`.

| Trial | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG | Decision |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table)}

The best trial also misses the retained 1% global and per-source endpoint.
Source gains are 0.616700% / 0.682537% RAW and 0.624365% / 0.696822% PNG,
reported under their dataset names rather than inferred ethnicity. Its failures
include 12 SSIM, 16 fixed-recognizer appearance and 4 pixel-error checks on each
stage. The smallest trial still has 14 RAW and 10 PNG failures. Brightness
fractions stay below 0.2. A fixed-recognizer comparison is not verified identity.

The worker took {failure['seconds']:.2f} seconds, with 100 restoration forward
batches, 180 recognizer forward batches, three temporary component proposals,
zero accepted updates and zero full epochs. Export took 50.67 seconds.
`complete: true` describes the archive export; the pilot did not pass. No
accepted checkpoint or new development incumbent exists. Saved trial components
and failure records remain. No fresh post-failure RAM state hash is claimed.

All {imported['returned_file_bindings']} return-file bindings and 5,852 required
parent bindings pass path/size/hash checks. Archive is {imported['archive_bytes']:,}
bytes, SHA256 `{imported['archive_sha256']}`. The failure-only supplemental audit
replays four snapshots, 400 RAW outputs and 400 delivered PNGs, 800 stage metric
rows, observed-pixel composition, component byte schemas, all 32 saved gradient
vectors, Gram/KKT arithmetic, rates, rounded saved changes and exact rejections.
Initial RAW outputs equal the original DGP. Gradients and embeddings have not
been independently recomputed through neural models locally.
{link('Failure-only forensic audit','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/failure_forensic_audit.json')}.

## Retained prospective-checker stop

The original returned checker **did not pass** cross-run gradient parity. One
of 82,587,648 compared elements fails its unchanged `rtol=1e-4, atol=1e-8` bound:
the RAW `dataset/thumbnails128x128/lowlight_lr32/ArcFace_loss` vector at index
2,485,488 differs by 2.3003668e-8. All 32 norm comparisons pass; the affected
vector's relative L2 vector error is about 4.9630e-7. The original checker,
traceback facts and failed criterion remain. This is distinct from the VM's
image-preservation stop. Its cause is not established as CUDA nondeterminism.

The separate supplement is restricted to an already rejected zero-update
pilot. It reports the parity failure and verifies the remaining saved-output
arithmetic. It does not loosen the cross-run bound, any scientific preservation
gate, the 1e-13 image metric replay bound or qualification conditions. Regression
checks keep the actual mismatched value rejected and reject NaN/Inf/shape errors.
{link('Original checker stop','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/original_checker_stop.json')}.
{link('Supplement preparation and exact source differences','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/forensic_checker_preparation.json')}.

## Images and the next decision

All 20 sheets and 600 cells were directly reviewed at original resolution.
Visible eye, mouth, tooth and hair details remain soft, particularly on the
degraded rows. No convincing whole-face clarity gain over initial is established.
Some targets from the existing lower-resolution photo source are already soft;
this is not evidence that the ten audited HQ FFHQ targets lack detail. These are
100 paired photographic TRAIN cases, not native CCTV, final identities or a
covering-family completion evaluation.
{link('Every-sheet visual review','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/visual_review.json')}.

Removing the approximate PNG constraints improved Gram conditioning but did not
solve finite preservation. The new direction's cosine with R4 is 0.998685.
The largest trial predicted 1.642168% RAW structure gain but delivered 0.669337%.
No rounded-step dot product changes the predicted descent sign, yet 19 of the
30 RAW source/profile preservation losses regress beyond their retained bounds.
These derivative-group counts differ from the 17-group scientific reporting
counts above. No new direction, solver or model proposal was computed locally.
{link('96 saved prediction-versus-output comparisons','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/gradient_prediction_review.json')}.

The recorded next decision is to review data support and the conditioning
path's measured finite response before issuing another training recipe.
Adding epochs, smaller untested rates, a random reset or a lower quality gate is
not justified by this result. A future manual derivative/response diagnostic
must address that uncertainty explicitly; it is not a runnable packet today.
{link('Design review and primary research',design_name)}.

Original checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
R1 is used and stopped; do not rerun or resume its failed recipe. Checkpoints,
app design, splits/caches/provenance/full states and gate failures are retained.
No native CCTV, ethnicity, Zamboanga or exact hidden-identity claim follows.
Training remains human-manual L4 tmux. All five milestones and seven automatic
and assisted covering families remain required. The complete goal is incomplete.
'''
    design=f'''# Post-RAW conditioning design review — 14 October 2026

**RAW-only optimization has not solved the image-quality limitation.** Keep the
permanent original restorer and every stopped version. The next task is finite
response and data-support diagnosis before a new training recipe; no automatic
epochs or runnable replacement are established by this review.

## What changed and what the returned comparison establishes

R1 chose its direction using 32 RAW constraints, compared with R4's 64 combined
RAW/approximate-PNG constraints. Both use the same 20 exposed TRAIN references,
100 prepared paired cases, exact initial conditioning arrays, scientific
baseline, original decoder, frozen external StyleGAN2 bank and deterministic
readouts. The selected 12 conditioning tensors and all actual RAW/PNG quality
requirements remain. R1 uses 320 gradient queries versus R4's 640, so compute
exposure differs even though the finite output opportunity is the same three
trials. Runtime is 168.55s versus 237.05s. Rates are separately bounded; this is
an optimizer comparison, not a unique architectural-cause proof.

The RAW32 Gram condition number is {prediction['RAW32_Gram_condition_number']:.6f},
versus about 885 million jointly in R4. The current/prior direction cosine is
{prediction['current_previous_direction_cosine']:.9f}; removing PNG optimizer
constraints changes the direction little. At the largest rate, actual RAW/PNG
gain is 0.669337% / 0.682289%, compared with R4's 0.688521% / 0.705580%.
All trials fail preservation. At the smallest rate RAW predicts 0.410542%
feature gain and delivers 0.346369%, with 14 RAW and 10 PNG reporting failures.
Improved numerical conditioning alone has not yielded useful preserved learning.

All 96 saved-vector comparisons use the actual rounded component displacement.
Parameter rounding changes no predicted descent sign. Finite image losses still
regress. This does not uniquely identify a wrong derivative, finite curvature,
activation changes, or a data/conditioning limitation. One cross-run gradient
element also fails the original prospective check, which remains failed.
{link('Saved comparisons','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/gradient_prediction_review.json')}.
{link('Every original cross-run bound result','outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/failure_forensic_audit.json')}.

## Data and path review

Ten selected references have separately audited HQ FFHQ counterpart targets;
their ancestral 128-pixel source metadata is not the current target resolution.
The other ten use the retained source-photo targets, with ancestral sizes from
153–226 pixels. Every prepared target is 256x256. The direct sheets show mixed
fine-detail support in that source, and severe synthetic inputs where eye/mouth
definition is mostly lost. No new acquisition, target substitution, severity
selection, final-identity exposure or ethnicity inference is justified here.
Keep the audited terms, overlap limits and existing roles; document an actual
target/support gap before acquisition. Native CCTV review remains unpaired.

The current branch keeps the existing image encoder, reconstruction decoder and
feature readouts frozen. It learns global style and three spatial-conditioning
scales through the frozen generative bank. At initialization a frozen branch of
the same observed input cancels the added feature path exactly. The trial tests
that conditioned branch, not more epochs of the original restorer or a new
own-trained generative face prior. The attribution boundary is unchanged.

Before another solver or training schedule, measure whether the existing
conditioning path gives repeatable, correctly directed changes at declared
finite displacements. Separate exact RAW MSE/detail forwards, explicitly
approximate SSIM backward and rounded PNG scientific acceptance. Retain clear,
degraded and source-specific comparisons. A trustworthy sampled direction does
not establish all-dimensional derivatives or full TRAIN/native usefulness.

## Evidence-backed next diagnostic design

Prefer a zero-accepted-update manual L4 response diagnostic to another
essentially equivalent optimizer or extra epochs. First repeat the identical
initial forward and selected loss-gradient measurements with backend/precision,
batch order and RNG state recorded. Then compare saved directional predictions
with actual two-sided finite changes at a fixed, predeclared set of magnitudes.
Return actual rounded parameter differences, raw/PNG outputs, all scores,
per-reference records, state hashes and rollback proof. No optimizer, candidate
selection, new solve, rate search, plateau bypass or promotion belongs to that
diagnostic. Temporary neural parameter displacements stay manual on the L4.

Use the exact existing failure direction rather than deriving a new direction
locally. The new finite protocol must include update/displacement/gradient/
forward counts, timing/projection, VRAM/RSS, storage/reserve/output and export
limits based on measured returned work, plus a prospective independent checker.
No such verified transfer packet or commands are issued by this document.

If repeated or finite derivatives disagree, isolate loss precision/backward or
state effects before optimizing again. If directional predictions agree only
at smaller scales but required appearance/structure cannot be preserved within
the retained finite limits, review the conditioning/decoder route and supported
training targets; do not relabel a failed checkpoint as an incumbent. Only
audited preserved useful learning can justify the conditional later epoch study.

PyTorch describes comparison of analytical derivatives with central finite
differences. This supports the proposed diagnostic, not its unrun outcome.
FP32 and nondifferentiable/quantized paths need explicit interpretation rather
than a blanket default all-parameter gradcheck.
[PyTorch gradcheck mechanics](https://docs.pytorch.org/docs/2.9/notes/gradcheck.html).

PyTorch documents finite-precision differences across computation arrangements
and devices. That makes backend/precision evidence relevant, but does not prove
that CUDA noise caused our one failed comparison or the preservation failure.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

All original quality thresholds and scientific replay bounds remain. No
unexecuted safe rate or future success is inferred. The new RAW-only recipe is
stopped, and existing failures remain binding. New training remains manual in
tmux on the verified L4; agent VM access stays maintenance-only. Original app
design/model/checkpoints/splits/caches/provenance/full states are retained.

The independent completion work is still separate: seven covering families,
automatic proposal preview/correction, exact visible appearance, insufficient
information refusal, original/mask/one estimate and downloads. The prepared
local detector-confidence recorder has not run and establishes no new mask or
completion result. Native DEV, final identities, five milestones and full app
flow remain required. The goal is incomplete.
'''
    retained_new(ROOT/report_name,report); retained_new(ROOT/design_name,design)
    prefix=f'''# RAW-only conditioning pilot rejected — 14 October 2026

The returned R1 completed320 derivative queries/three finite trials in168.55s,
accepting zero updates. Best gain is0.669337% RAW/0.682289% PNG, below1%;
preservation failures are32/32,24/25 and14/10. All632 return-file and5,852 parent
bindings pass. Four snapshots/800 stage metric rows/saved-state and32-vector
KKT arithmetic are checked; all20 sheets/600 cells were directly reviewed,
with no convincing whole-face clarity gain.
{link('Audited failure and review',report_name)}.

Original prospective cross-run gradient parity remains FAILED: one element
differs by2.3003668e-8 beyond its original bound. A failure-only supplement
records this and verifies rejected outputs without relaxing any criterion.
Original checker/stop and all scientific preservation gates remain intact.

RAW-only optimization improves Gram conditioning but supplies a direction
close to R4(cosine0.998685) and does not solve preservation. Review data support
and the actual finite conditioning response before another training recipe.
No more epochs,new incumbent,app adoption or runnable next packet is qualified.
{link('Next diagnosis and primary research',design_name)}.

R1 is USED/STOPPED; do not rerun/resume it. All original checkpoints,app design,
splits/caches/provenance/full states/failures remain. Paired TRAIN/native unpaired
and reserved-final roles stay separate. All five milestones and seven automatic/
assisted covering families remain required. Training stays human-manual L4 tmux;
no agent VM action or new local neural training occurred. Full goal incomplete.
The separate detector-confidence recorder is prepared only and UNRUN.
Exact previous document follows.

'''
    records={}
    (stage/'before').mkdir(exist_ok=True)
    for name in docs:
        records[name]=finish_prepend(ROOT/name,prefix,stage/'before'/name,expected[name])
    used_prefix=f'''# USED AND STOPPED — RAW conditioning R1, 14 October 2026

Do not rerun the historical commands below. The returned manual pilot accepted
zero updates; all three proposals failed RAW/PNG preservation and1% structure.
The original prospective gradient-parity check also remains failed; a separate
failure-only forensic review retains that discrepancy and verifies the outputs.
{link('Returned results and all-sheet review',report_name)}.
{link('Next design review; no new runnable packet',design_name)}.

Exact historical guide follows.

'''
    records[used]=finish_prepend(ROOT/used,used_prefix,stage/'before'/used,expected[used])
    evidence=['import.json','failure_forensic_audit.json','visual_review.json','gradient_prediction_review.json',
              'original_checker_stop.json','forensic_checker_preparation.json','cross_run_report_check.json']
    evidence_hashes={(ANALYSIS/n).relative_to(ROOT).as_posix():sha(ANALYSIS/n) for n in evidence}
    for n in ['analyze_cctv_dgp_raw_conditioning_learning_v1_r1_prediction.py',
              'record_cctv_dgp_raw_conditioning_learning_v1_r1_visual_review.py',
              'cctv_dgp_cross_run_gradient_report_v1.py']:
        evidence_hashes['scripts/'+n]=sha(ROOT/'scripts'/n)
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        returned_protocol_sha256=PIN,returned_archive_sha256=imported['archive_sha256'],returned_accepted_updates=0,
        actual_rejected_trials=3,original_prospective_checker_passed=False,original_cross_run_bound_not_relaxed=True,
        documents=records,new_documents_sha256={n:sha(ROOT/n) for n in [report_name,design_name]},evidence_sha256=evidence_hashes,
        more_epochs_qualified=False,new_runnable_packet=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=6,new_documents=[report_name,design_name],accepted_updates=0,new_runnable_packet=False))


if __name__=='__main__':main()
