"""Publish completed measurement and unqualified images, retaining exact history."""
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_response_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new


def main():
    stage=ANALYSIS/'publication';stage.mkdir(exist_ok=True);(stage/'before').mkdir(exist_ok=True)
    assert not (stage/'publication.json').exists()
    proof=[ANALYSIS/n for n in ['import.json','independent_audit.json','response_review.json',
        'saved_embedding_precision_review.json','PNG_change_review.json','visual_review.json']]
    imported,audited,response,precision,png,visual=[read(n) for n in proof]
    assert all(v['complete'] for v in [imported,audited,response,precision,png,visual])
    assert audited['full_diagnostic_returned'] and audited['failure'] is None
    assert visual['all20_sheets_and1000_cells_directly_viewed'] and precision['all_frozen_embeddings_exact']
    assert len(response['signed_points'])==6 and len(response['central_summaries'])==12
    assert len(audited['central_response_comparisons'])==96
    assert response['diagnostic_counts']==dict(B=200,recognizer=360,gradient_queries=640,temporary_displacements=6,optimizer_updates=0,accepted_updates=0)
    assert response['prior_smallest_trial_comparison']['maximum_absolute_score_difference']==0
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    previous=read(OUT/'cctv_dgp_conditioning_response_v1_publication/publication.json')
    expected={n:r['after_sha256'] for n,r in previous['documents'].items()}
    expected.update(previous['new_documents_sha256'])
    assert len(expected)==8
    for n,h in expected.items():
        assert sha(ROOT/n)==h or ((stage/'before'/n).exists() and sha(stage/'before'/n)==h),'Preserve concurrent document edits'
    result_name='CCTV_DGP_CONDITIONING_RESPONSE_V1_RESULTS.md'
    review_name='CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md'
    table=[]
    for point in response['signed_points']:
        raw,delivered=point['stages']['raw'],point['stages']['png']
        table.append(f"| {point['label']} | {point['signed_rate']:.10g} | {100*raw['structure_gain_fraction']:.6f}% | {100*delivered['structure_gain_fraction']:.6f}% | {raw['preservation_failures']} / {delivered['preservation_failures']} |")
    report=f'''# Conditioning-response V1 returned result — 14 October 2026

**The diagnostic completed successfully; no improved model qualified.**
The manual L4 run measured the original conditioning path twice and six fixed
signed parameter changes. It constructed no optimizer, solved no new direction,
accepted zero updates and ran zero training epochs. The sampled positive
changes reduce structure error, but none meet the original RAW/PNG preservation
and 1% global/per-source endpoint requirements together.

| Fixed diagnostic point | Signed rate | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

These are relative reductions of degraded landmark high-frequency error;
they are not percentages of identity recovered. The largest positive point
is the previous RAW R1 pilot's smallest rejected trial: all {response['prior_smallest_trial_comparison']['compared_group_scalars']}
group scalars match exactly. This repeat measurement establishes no new gain
over that result. Negative points are response probes, not training proposals.
No candidate was selected, even at the smallest positive point where RAW
preservation passes and PNG still fails five checks.

At `plus01`, RAW fails one appearance check for
`dataset/thumbnails128x128/compound_lr24`; PNG fails one for
`dataset/asian_faces/blur_lr24`. At `plus02`, RAW has no preservation failure,
but PNG fails two MSE and three fixed-recognizer appearance checks. Source
labels describe the retained datasets, not ethnicity or local CCTV performance.
{link('Every signed point and retained decisions','outputs/cctv_dgp_conditioning_response_v1_analysis/response_review.json')}.

## What the completed measurement explains

All six source-specific RAW structure central comparisons fall within the
predeclared 5% diagnostic comparison bound. At the smallest magnitude the two
relative slope errors are below 0.202%. MSE passes 29/30 such comparisons and
SSIM passes 29/30. Appearance passes only 12/30: its disagreements are 9/10,
5/10 and 4/10 from largest to smallest magnitude. Every comparison is resolved
under the original declared diagnostic noise rule. A directional comparison
is not an all-dimensional derivative proof, a preservation gate, or training
qualification; the 5% bound does not relax a single image-quality criterion.

The small-step pixel/detail direction is informative, but its first-order
appearance prediction does not reliably describe the actual finite response.
Even response and PNG rounding also matter. This result supports locating that
response boundary before another optimizer or epoch schedule; it does not
uniquely prove a faulty backward operator or a failed architecture.

The saved baseline and target recognizer embeddings are exactly unchanged in
every snapshot. Fresh initial double-precision group scores match both recorded
gradient-forward group values within 1.11e-16. Recomputing scalar dots from the
same saved vectors in float64 leaves the same 9/5/4 central disagreements;
the largest float-versus-double two-sided difference is 2.96e-8. Scalar dot
rounding alone does not explain the remaining sampled disagreements. This
is saved-array reporting only: the network was not re-executed in float64,
and no scientific metric or quality threshold was changed.
{link('Saved embedding precision review','outputs/cctv_dgp_conditioning_response_v1_analysis/saved_embedding_precision_review.json')}.

The smallest positive point changes 22,124 RGB channel values across all 100
PNGs, by at most one byte level each. Its correct delivered conversion still
fails five preservation checks. Correct PNG export is independently verified;
neither a RAW-only pass nor approximate PNG derivatives guarantee delivery
preservation. Pixel-change counts do not establish a visible improvement.
{link('All six delivered pixel recounts','outputs/cctv_dgp_conditioning_response_v1_analysis/PNG_change_review.json')}.

## Retained reproducibility limit

All 32 vector-norm comparisons between the two repeats pass. Their original
elementwise parity requirement passes 31/32 groups; one element in
`RAW/dataset/thumbnails128x128/lowlight_lr32/ArcFace_loss` still misses it.
At index 2,451,098 its difference is 1.1781230568e-8 against the unchanged
1.0507631339e-8 bound. Repeat00 versus prior R1 passes 32/32; repeat01 versus
prior R1 passes 31/32 with the same index missing that comparison's bound.
The previously retained R1 parity failure remains. No numerical tolerance was
relaxed and no blanket claim that every derivative reproduces is made.

## Audit and images

All {imported['returned_file_bindings']} return-file bindings and
{imported['parent_file_bindings']} required parent bindings passed path/size/hash
checks. Archive: {imported['archive_bytes']:,} bytes, SHA256
`{imported['archive_sha256']}`. The original prospective returned checker passed
its declared saved-evidence checks while explicitly reporting the parity misses;
no failure-only replacement checker was needed for this return.

The audit independently replays 800 RAW outputs, 800 PNGs and 1,600 stage metric
rows, saved embeddings, exact observed-support composition, all 64 aggregate
vectors, contribution arithmetic, six rounded parameter differences, 96 central
responses, unchanged scientific decisions and protected model/app files.
Initial and rollback outputs match the original DGP exactly; selected component
weights, frozen full states, buffers and saved Torch CPU/CUDA RNG states are
restored. This is rollback proof, not a full optimizer-resume checkpoint.
No neural derivatives or embeddings were re-executed locally.
{link('Prospective independent audit','outputs/cctv_dgp_conditioning_response_v1_analysis/independent_audit.json')}.

All 20 planned sheets and 1,000 cells were directly viewed at original
2560x1400 resolution. Eye, mouth, tooth, skin and hair details remain soft on
degraded rows. None shows convincing new whole-face detail over initial.
Some retained source-photo targets also limit available fine detail; this does
not invalidate the separately audited HQ targets from the other source.
{link('Per-sheet visual observations','outputs/cctv_dgp_conditioning_response_v1_analysis/visual_review.json')}.

The worker took {response['runtime_seconds']:.2f}s, with 200 restorer and 360
recognizer forward batches, 640 derivative queries and six temporary component
displacements. Export took 100.64s. `complete: true` describes export, and
`complete_diagnostic: true` now has an independent audit; neither qualifies
training success. USED/DIAGNOSTIC COMPLETE: retain this run and do not rerun it.

These are 100 already exposed paired photographic TRAIN cases, not native
CCTV, reserved final identities, or a completion evaluation. Original app
checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The original restorer is DeblurGAN-v2-compatible; the comparison branch's
StyleGAN2 bank is externally pretrained and frozen, not our trained face prior.
All checkpoints, splits, caches, provenance, full states, failures and app design
remain. All five milestones and seven automatic/assisted covering families are
still required. More epochs and model promotion remain unqualified.
{link('Next bounded diagnosis and primary research',review_name)}.
'''
    design=f'''# Appearance-response boundary review — 14 October 2026

**Isolate the reconstruction-to-recognizer response before training more.**
The conditioning-response diagnostic completed and its original checker passed
the saved-evidence audit. Its measurement result does not qualify a replacement
restorer. Keep the permanent checkpoint, rejected states and original quality
requirements. {link('Audited result',result_name)}.

## Supported findings and remaining uncertainty

The selected direction correctly predicts sampled small RAW detail changes;
it fails several actual appearance response comparisons even after reducing
the magnitude 100-fold. Fresh initial gradient-forward group scores agree with
saved double-precision scores, and all frozen baseline/target embeddings stay
exact. Double dot arithmetic leaves the same disagreements. There is also one
retained elementwise gradient repeatability miss. No single backward bug,
activation transition, precision source or architectural incapacity is proved.

The response is measured through a chain: conditioning tensors, frozen bank,
fixed feature readouts, original reconstruction decoder, observed-support
composition, fixed crop/recognizer and score. The current parameter-space
comparison combines these boundaries. MSE/detail agreement and appearance
disagreement narrow the diagnosis but do not identify which boundary causes it.
The delivered PNG branch adds byte quantization; its preservation failure is
separate from a RAW derivative check. Neither mean-loss descent nor a tiny
parameter distance substitutes for actual output preservation.

## Next manual diagnostic design

Use the same 100 exposed TRAIN cases and existing initial/signed RAW images.
No new target, final identity, native CCTV pixel, candidate checkpoint, model
parameter displacement or restoration forward is needed to examine this boundary.
Record fixed-recognizer derivatives with respect to the prepared initial RGB
image tensor, twice in the same batch order. Compare their saved pixel-space
dot products with the actual saved plus/minus RGB differences, then compare
those predictions with both the existing parameter-space predictions and
actual appearance-score changes. Retain per-reference records as well as the
source/profile means so cancellation is visible.

This distinguishes an error already present in the fixed recognizer's finite
input response from one introduced earlier in the reconstruction-to-image
linearization. It is a directional localization test; if disagreement remains,
it does not automatically prove an incorrect derivative. The input-space
finite interval still requires numerical and nonlinearity interpretation.

The prospective scope is two passes of 20 five-case batches: at most 40 frozen
recognizer forward batches and 200 RGB-input derivative queries, zero restorer
forwards, zero optimizer updates and zero neural-parameter changes. Save every
RGB-input gradient, exact input/output embeddings, scalar score, precision and
batch/RNG/state receipt. Reuse prior saved output hashes and original gates;
accept no training update or candidate during this diagnostic.

A transfer packet and its independent checker still need preparation. Before
issuing commands, enforce finite timing/projection, VRAM/RSS, storage/reserve,
output/export limits and Windows manual-execution guards, with meaningful
saved-array regressions. This document is a design decision, not a verified
runnable packet or authorization for agent neural training. Actual derivatives
stay human-manual on the existing L4 in tmux. Agent VM access stays maintenance
only. No new VM launch or local neural execution occurred in this return audit.

## Why this precedes another optimizer or epoch schedule

Reusing the failed direction at smaller rates already demonstrates the tradeoff:
RAW gain shrinks to 0.040293% and then 0.004101%, while actual PNG checks still
fail. Repeating it with more epochs or extrapolating a safe accumulation is not
justified. A new optimizer needs trustworthy boundary evidence and actual
RAW/PNG verification at each retained finite proposal. Changing architecture
or acquiring targets without a demonstrated limitation would add confounders.
Future training remains conditional on useful preserved learning and independent
development review; the 1% requirement is an early learning gate, not final
model usefulness. No quality requirement or failed historical result changes.

PyTorch describes comparing backward derivatives with central finite
differences. Its methodology supports examining this boundary, not a claim
that our entire FP32 network has passed gradcheck.
[Primary gradcheck mechanics](https://docs.pytorch.org/docs/2.9/notes/gradcheck.html).

The original GEM paper derives gradient-angle constraints under a local
linearity assumption. That supports retaining actual finite-loss checks in
addition to projected derivative checks; its classification results do not
establish face-restoration performance or prove a new recipe here.
[Lopez-Paz and Ranzato, GEM](https://arxiv.org/pdf/1706.08840).

PyTorch documents limitations of finite-precision and bitwise reproducibility.
This motivates preserving our failed original bound; it does not establish
that CUDA numerical variation caused the image-preservation failures.
[Primary numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

The trained main app model stays pinned. Component origins remain declared:
the retained restorer is DeblurGAN-v2-compatible, the separate comparison bank
is externally pretrained StyleGAN2, and the fixed recognizer is a measurement
component, not an identity guarantee. Native CCTV stays unpaired and reserved
final identities stay outside tuning. Seven completion families, automatic
mask preview/correction, visible appearance, insufficient-information refusal,
original/mask/one plausible result and downloads remain a separate requirement.
The detector-confidence recorder remains prepared only and UNRUN. All five
milestones, independent final review and bundled inline Playwright app flow
remain required; the full goal is incomplete.
'''
    retained_new(ROOT/result_name,report);retained_new(ROOT/review_name,design)
    prefix=f'''# Conditioning response completed; model still unqualified — 14 October 2026

The manual diagnostic completed in366.59s plus100.64s export:640 derivative
queries,six fixed signed changes,zero optimizer updates/accepted updates/epochs.
All1,247 return and5,852 parent bindings,8 snapshots/1,600 metric rows and
20 sheets/1,000 directly reviewed cells are checked. Rollback weights,outputs
and saved Torch CPU/CUDA RNG are exact; original model/app files are unchanged.
{link('Audited result and every failed decision',result_name)}.

Positive RAW/PNG structure gains are0.346369/0.361388%,0.040293/0.053512%,
and0.004101/0.006602%;preservation failures are14/10,1/1 and0/5. None meets
the retained preservation and1% global/per-source endpoint together. Small RAW
detail predictions agree,while appearance comparisons disagree9/5/4 groups;
saved double-dot reporting does not remove those disagreements. One original
elementwise repeatability bound still fails and remains reported,not relaxed.

Next: isolate the reconstruction-to-fixed-recognizer input response before
another optimizer or epoch schedule. No next verified packet or new incumbent
is established. {link('Bounded design and primary research',review_name)}.
USED/DIAGNOSTIC COMPLETE; do not rerun the returned packet. Native unpaired,
paired TRAIN and final roles,all five milestones and seven automatic/assisted
covering families remain. Training/derivatives stay human-manual L4 tmux;
agent VM access is maintenance only. Detector-confidence work remains prepared
only and UNRUN. Goal incomplete. Exact previous document follows.

'''
    guide_prefix=f'''# USED — diagnostic completed and independently audited, 14 October 2026

This packet has already run. Retain its outputs and do not repeat the historical
launch commands below. Zero optimizer updates or training epochs; no model
qualified. {link('Current result and retained limits',result_name)}.
The next input-response diagnostic is a design review, with no verified transfer
packet yet. Exact previous preparation and commands follow as history.

'''
    docs={}
    for n,h in expected.items():
        docs[n]=finish_prepend(ROOT/n,guide_prefix if n in previous['new_documents_sha256'] else prefix,stage/'before'/n,h)
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__),protocol_sha256=PIN,return_archive_sha256=imported['archive_sha256'],
        documents=docs,new_documents_sha256={n:sha(ROOT/n) for n in [result_name,review_name]},
        evidence_sha256={n.relative_to(ROOT).as_posix():sha(n) for n in proof},
        diagnostic_completed=True,original_prospective_return_audit_completed=True,
        original_elementwise_parity_all_pass=False,all20_sheets_and1000_cells_reviewed=True,
        optimizer_updates=0,accepted_updates=0,next_diagnostic_is_design_only=True,
        automatic_VM_launch=False,local_neural_or_training_calls=0,model_qualified=False,
        more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,documents_updated=len(docs),new_documents=2,diagnostic_completed=True,model_qualified=False))


if __name__=='__main__':main()
