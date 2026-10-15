"""Publish the audited manual measurement and retain every earlier status byte."""
from datetime import datetime, timezone
from pathlib import Path

from cctv_dgp_decoder_activation_v1_contract import CHECKPOINT, read, write, sha
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
ANALYSIS = OUT/'cctv_dgp_decoder_activation_v1_analysis'
RETURN = OUT/'cctv_dgp_decoder_activation_v1_return'
PREP = OUT/'cctv_dgp_decoder_activation_v1_preparation'
PIN = 'b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481'


def link(label, path):
    return f'[{label}](<{(ROOT/path).as_posix()}>)'


def main():
    stage = ANALYSIS/'publication'
    assert not (stage/'publication.json').exists()
    imported = read(ANALYSIS/'import.json')
    audit = read(ANALYSIS/'independent_audit_r2.json')
    repair = read(ANALYSIS/'independent_audit_r2_schema_r2.json')
    causal = read(ANALYSIS/'independent_causal_review.json')
    closure = read(ANALYSIS/'independent_readback.json')
    assert all(item['complete'] for item in [imported,audit,repair,causal,closure])
    assert imported['protocol_sha256'] == audit['protocol_sha256'] == repair['protocol_sha256'] == causal['protocol_sha256'] == PIN
    assert audit['exact_retained_RAW_and_floor_PNG_cases'] == 400
    assert audit['batch5_objective_rows_checked'] == 80
    assert audit['returned_run_counts']['gradient_queries'] == 70
    assert audit['returned_run_counts']['optimizer_updates'] == 0
    assert repair['original_checker_passed_unmodified'] is False
    assert repair['derived_R2_numerical_checks_complete'] is True
    assert repair['scientific_thresholds_changed'] is False
    assert sha(OUT/'cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == CHECKPOINT
    archive = OUT/'cctv-dgp-decoder-activation-v1-results.tar.gz'
    assert sha(archive) == imported['archive_sha256']
    assert archive.stat().st_size == imported['archive_bytes']
    previous = read(PREP/'publication/publication.json')
    names = ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
             'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_NEXT_PILOT.md',
             'CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md','CCTV_DGP_DECODER_ACTIVATION_V1_VM.md',
             'CCTV_DGP_DECODER_ACTIVATION_V1_PREPARATION.md']
    expected = {name:previous['documents'][name]['after_sha256'] if name in previous['documents']
                else previous['new_document_sha256'][name] for name in names}
    assert all(sha(ROOT/name) == digest for name,digest in expected.items()), 'Retain concurrent document edits'
    states = causal['state_component_summary']
    assert [row['state'] for row in states] == ['initial','probe_1_16','probe_1_4','fit_u001']
    table = '\n'.join(f"| {row['state']} | {row['mean_objective']:.9f} | {row['objective_worsened_references']}/20 |" for row in states)
    result_name = 'CCTV_DGP_DECODER_ACTIVATION_V1_RESULTS.md'
    decision_name = 'CCTV_DGP_POST_ACTIVATION_V1_REVIEW.md'
    result = f'''# Decoder activation V1: measurement completed, model stop retained

The manual L4 measurement completed in **86.30 seconds** with **70 gradient
queries, zero optimizer updates and no new candidate direction or checkpoint**.
The returned archive is fully hash-verified: {imported['archive_bytes']:,} bytes,
SHA256 `{imported['archive_sha256']}`. Its 535 regular members were checked
before extraction and rehashed afterward. Protocol: `{PIN}`.

The original VM saved-data checker stopped on missing gate metadata. The first
local checker also stopped on exact norm-reduction equality. A separately
reviewed local R2 checker completes the numerical audit; the original and R1 failures,
sources, protocol and returned files remain intact. This is a completed
measurement/audit, not a successful restoration model.

| Retained state | Mean actual batch-five objective | References with worse objective |
|---|---:|---:|
{table}

Lower objective is better. All 400 RAW/floor-PNG outputs equal the previously
audited, visually reviewed hinge-repair controls. All 80 fresh batch-five
objective records, 70 saved parameter gradients and exact before/after model
and CPU/CUDA RNG states are checked. Saved gradient arithmetic was independently
recomputed; no fresh neural or autograd replay occurred locally. The first
fitting-reference clipped gradient equals its retained original bytes exactly.

At the smallest retained placement, reconstruction plus anchors improve by
0.018446591, but regression penalties add 0.029102959 and brightness penalties
add 0.007709694. The total mean objective therefore worsens by 0.018366062.
The actual fitting reference has a negative initial selected slope, yet its
finite objective rises. Both sampled references improve all four degraded
landmark training slopes while worsening low-light and compound RGB MSE slopes.
Those positive preservation slopes exhaust the tiny regression allowances far
before the tested 1/16 placement. Re-evaluating the actual hinge offsets in a
finite affine scalar model explains the sign change substantially better than
extrapolating the infinitesimal slope alone. This is explanatory arithmetic,
not a certified finite neural or PNG prediction.

The largest batch-five versus historical batch-fifteen objective difference is
4.1781e-7. That observed discrepancy is too small to explain these finite
objective rises; exact batch parity is not claimed. Brightness activation
alone also does not reverse the fitting reference's initial descent prediction.
There is no evidence here that more epochs or scaling this same direction
will preserve the face.

Historical quality gates remain failed: at the full retained update the RAW
structure-error reduction is **0.113913%**, PNG **0.128470%**, with **30 RAW and
30 PNG group/metric preservation failures**. The smaller placements also fail.
Structure means landmark-supported high-frequency luminance MSE reduction;
recognizer similarity is a separate metric. These are exposed photographic
TRAIN results, not native CCTV, final review or identity recovery.

R2 changes only declared checker boundaries. Four missing historical metadata
fields match the frozen parent gate exactly. Exact dyadic integer arithmetic
proves the unchanged parameter radius; the saved approximate norm remains
explicitly unverified rather than rewritten or accepted with a ULP tolerance.
Only the derived `finite_arithmetic.objective` scalar uses the already frozen
absolute 1e-10 arithmetic bound: 14/80 rows differ, at most 8.8818e-16. Every
other field in that comparison remains exact. The original bitwise scalar
assertion did not pass. RAW/PNG scoring and scientific gates are unchanged.
Stored VM scalar differences are checked exactly against their stored operands,
separately from bounded local recomputation. Typed nested-record comparisons
retain exact keys, lengths and nonnumeric types with the existing per-call
numeric bounds. Actual records were prechecked before the full output audit.

{link('Full local R2 saved-data audit', 'outputs/cctv_dgp_decoder_activation_v1_analysis/independent_audit_r2.json')}.
{link('Explicit R2 changes, exact radius proof and original failures', 'outputs/cctv_dgp_decoder_activation_v1_analysis/independent_audit_r2_schema_r2.json')}.
{link('Independent causal arithmetic', 'outputs/cctv_dgp_decoder_activation_v1_analysis/independent_causal_review.json')}.
{link('Independent return readback', 'outputs/cctv_dgp_decoder_activation_v1_analysis/independent_readback.json')}.
{link('Selected next design requirements', decision_name)}.

Only two references have these underlying gradients; the other 18 have finite
replay evidence. No new native/final images, completion masks or visual sheets
were used. No new visual review is claimed because all output pixels equal the
already reviewed controls. The original app model and design remain unchanged.
All five milestones and seven automatic/assisted covering families remain open.
'''
    decision = f'''# Post-activation decision: constrain preservation before another trajectory

The invalid assumption was that the corrected initial objective gradient could
be extrapolated to the retained update while its preservation barriers stayed
inactive. The completed measurement disproves that for the tested direction:
positive profile-level RGB slopes exhaust real, tiny regression allowances.
Fresh batch-five evidence confirms the finite loss increase. It does not prove
that every direction in the current decoder is infeasible.

{link('Audited measurement and limits', result_name)}.

The best-supported next design is **one finite feasibility experiment in the
current A reconstruction decoder**, using the saved underlying gradients.
It must select a different direction that addresses the measured preservation
conflicts before any new optimizer trajectory. The experiment is not prepared
or authorized for automatic execution by this review. Its solver, numerical
checks, exact transfer assets and timing/storage limits still need a separate
prospective contract and independent review.

1. Reuse the original A0 state and saved 70 gradients from the two existing
   scheduled TRAIN references. Solve at most one nonzero direction in a
   declared saved-gradient span, manually on the L4. Require simultaneous
   per-profile RGB/SSIM/recognizer preservation and progress in both references'
   degraded structure terms, with actual regression slack and nested
   photometric/mean-control clipping represented. A favorable weighted total
   alone is insufficient; no new gradient, optimizer update or epoch is needed
   to establish this bounded feasibility question.
2. Freeze the solver and at most three placements before inspecting new output.
   Keep the original-relative radius, normalization, reconstruction weights and
   scientific gates. Exclude zero, unchanged and vanishing proposals. An
   infeasible solve is a result to retain, not permission to loosen constraints,
   enlarge the radius or add more search attempts.
3. Evaluate every allowed placement on all 20 fixed references/100 cases.
   Independently reconstruct the exact RAW and floor-PNG metrics for all 17
   groups. Require at least 1% global and each-source structure improvement,
   original visible-appearance/preservation checks, brightness fraction at
   most 0.2 and exact exterior pixels. Only two references supply gradients;
   the other 18 provide essential finite checks rather than fitted derivatives.
4. Bind and inspect every planned visual sheet for useful eyes, brows, nose,
   mouth, jaw and visible appearance. Tiny loss changes or PNG-unchanged images
   do not establish useful detail. A TRAIN pass still needs the frozen native
   DEV comparisons, original/pretrained comparators and independent final
   evaluation before any app promotion or extra-epoch continuation.
5. Stop this declared experiment if the solve is infeasible or every allowed
   placement fails. Return to reconstruction/data design review; do not add a
   fourth scale or resume the failed Adam recipe. Infeasibility concerns this
   sampled span, not the whole architecture or all possible restoration models.

This differs from the earlier seven-loss PCGrad, aggregated B-direction and
conditioning-only searches: the current 15-tensor A decoder's underlying
profile losses and exact corrected batch-five path are now exposed. It must
not reproduce those failed methods under a new name. A solver that cannot
represent the required constraints must report that limitation before any
finite neural proposal is made.

Route A remains a DeblurGAN-v2-compatible conditional restorer; cached external
GAN features are not injected into this A path. This is not a newly trained GAN
or qualified generative face prior. The goal's justified replacement comparison
remains required. The original baseline `{CHECKPOINT}` stays permanent.
No local gradients, parameter proposals, VM connection or new app change was
performed for this review. Completion remains separate: preserve reviewed
visible pixels and report all seven automatic/assisted covering families.
'''
    for name,text in [(result_name,result),(decision_name,decision)]:
        retained_new(ROOT/name,text)
    prefix = f'''# USED: activation measurement audited; preservation conflict isolated — 15 October 2026

**Do not rerun the activation packet or resume the failed optimizer.** The
downloaded manual diagnostic completed 70 gradient queries, zero updates and
no new direction. A separate local R2 audit verifies 400 exact retained RAW/PNG
controls, 80 actual batch-five objectives, saved gradients and exact restored
model/RNG states. The original VM/local and R1 checker stops remain preserved;
explicit metadata/schema repairs do not change scientific gates. The user's
diagnostic answer selects the best approach: an explicit saved-evidence schema.

The mean objective rises from 5.200000 to 5.218366 at the smallest tested
placement. Low-light/compound preservation losses increase despite improving
landmark slopes; tiny regression margins activate well before that placement.
Batch-five versus batch-fifteen differences are too small to explain this
failure. All retained nonzero proposals still fail RAW/PNG preservation.
No model qualifies, replaces the baseline or earns additional epochs.

{link('Audited results and checker limitations', result_name)}.
{link('Next bounded feasibility design requirements', decision_name)}.

Next work must establish a distinct preservation-aware direction before a new
training trajectory. No new execution packet is certified by this decision.
All five milestones, independent native DEV/final review and seven covering
families remain active. Checkpoints, splits, caches, failed states and exact
historical instruction bytes stay retained. Prior PREPARED/UNRUN commands below
are historical and superseded by this USED status.

'''
    (stage/'before').mkdir(parents=True,exist_ok=True)
    docs = {name:finish_prepend(ROOT/name,prefix,stage/'before'/name,digest)
            for name,digest in expected.items()}
    evidence = [ANALYSIS/name for name in ['import.json','audit_r0.log','audit_r1.log','independent_audit_r2.json',
        'independent_audit_r2_schema_r2.json','independent_causal_review.json',
        'finite_arithmetic_leaf_differences.json','independent_readback.json']]
    protected_names = ['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
        'static/face_workflow.js','static/face_workflow.css','templates/face_workflow.html',
        'dgp_face_workflow_v3.py','face_workflow_web.py']
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='progress: fixed mask fidelity, audited native exclusion and sealed the distinct manual diagnostic',
        current_goal_turn_classification='progress: imported manual return, retained original checker failures, independently audited saved evidence and isolated finite preservation conflict',
        documents=docs,new_document_sha256={name:sha(ROOT/name) for name in [result_name,decision_name]},
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence},
        protected_sha256={name:sha(ROOT/name) for name in protected_names},
        protocol_sha256=PIN,archive_sha256=imported['archive_sha256'],archive_bytes=imported['archive_bytes'],
        status='MEASUREMENT_AUDITED_VIA_EXPLICIT_LOCAL_R2; ORIGINAL_STOPS_RETAINED; MODEL_UNQUALIFIED',
        worker_optimizer_updates=0,worker_gradient_queries=70,new_local_gradient_queries=0,
        new_local_optimizer_updates=0,new_parameter_proposals=0,agent_VM_calls=0,
        new_manual_execution_packet_certified=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,new_documents=2,preserved_documents=len(docs),model_qualified=False,goal_complete=False))


if __name__ == '__main__':
    main()
