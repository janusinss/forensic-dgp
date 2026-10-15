"""Publish measured precision limits and the next bounded training design, with history."""
from datetime import datetime, timezone

from import_cctv_dgp_recognizer_precision_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new


def main():
    stage = ANALYSIS / 'publication'; (stage / 'before').mkdir(parents=True, exist_ok=True)
    assert not (stage / 'publication.json').exists()
    files = [ANALYSIS / n for n in ['import.json', 'independent_audit.json', 'precision_activation_review.json']]
    imported, audited, reviewed = [read(n) for n in files]
    assert all(v['complete'] for v in [imported, audited, reviewed])
    assert audited['full_diagnostic_returned'] and not audited['failure'] and audited['all_original_parity_checks_pass']
    assert reviewed['sampled_response_failure_counts'] == dict(FP32=6, FP64=5)
    prior = read(OUT / 'cctv_dgp_recognizer_precision_v1_publication/publication.json')
    expected = {n: b['after_sha256'] for n, b in prior['documents'].items()}
    expected.update(prior['new_documents_sha256']); assert len(expected) == 9
    assert all(sha(ROOT / n) == h for n, h in expected.items()), 'Preserve concurrent documents'
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    report_name = 'CCTV_DGP_RECOGNIZER_PRECISION_V1_RESULTS.md'
    design_name = 'CCTV_DGP_NEARBY_GRADIENT_LEARNING_REVIEW_V1.md'
    table = []
    for r in reviewed['paired_precision_responses']:
        table.append(f"| {r['id']} | {100*r['FP32_relative_residual']:.4f}% / {'pass' if r['FP32']['central_comparison_within_diagnostic_bound'] else 'fail'} | {100*r['FP64_relative_residual']:.4f}% / {'pass' if r['FP64']['central_comparison_within_diagnostic_bound'] else 'fail'} |")
    report = f'''# Recognizer precision V1 result — 14 October 2026

**The measurement completed and passed independent auditing. Full FP64
computation does not resolve the sampled appearance-response disagreement.**
Six of ten float32 comparisons miss the declared diagnostic bound; five of ten
float64 comparisons miss it. All 30 original gradient parity checks pass.
No restorer ran, no training update was accepted, and no new image gain was
produced. The existing trained model and scientific gates remain unchanged.

| Selected exposed TRAIN case | FP32 relative prediction residual / status | FP64 relative prediction residual / status |
| --- | ---: | ---: |
{chr(10).join(table)}

These values are directional prediction errors, not identity recovery or image
restoration gains. The comparison uses the predeclared 5% diagnostic bound and
noise rule. All 20 responses are resolved; no noise tolerance was changed. The
two references were selected from exposed failure evidence, so these ten cases
are not an independent population or final evaluation. Each source contributes
one reference with all five clear/degraded profiles. Dataset names are
provenance labels, not ethnicity or local CCTV claims.

## What changed in the measured computation

The float64 branch promotes the same saved RGB values, fixed float32 grids
and neutral fill, uses a disposable double recognizer copy, and keeps crop,
encoder, normalization and scalar arithmetic in double. It round-trips exactly
to the original frozen float32 weight state. All floating encoder outputs pass
their declared dtype checks. This is a genuine full-computation precision
comparison, unlike recalculating only saved float32 vector dots.

Float64 changes the thumbnail clear comparison from fail to pass. The five
other failed float32 cases remain failures. In the selected Asian-source
lowlight case, residual error is 9.6853% in FP32 and 9.8010% in FP64. In the
thumbnail compound case it is 6.7857% and 6.8584%. Double precision is therefore
not a general repair for these sampled finite responses. Original scientific
FP32/PNG scores and all previous rejected restoration decisions are retained.

Packed activation-map replay finds 1,838 and 1,852 sign changes for plus/minus
FP32 inputs and 1,845 and 1,844 for FP64. Each count spans 37,130,240 units in
50 module/reference maps. Both signed endpoints change signs in every such
map; repeated initial inputs change zero signs. These are PReLU region
transitions, not visible pixels or generated facial regions. They support
finite nonlinear behavior as a contributor but do not prove exclusive
causation, a wrong backward kernel, or correctness of every derivative.
{link('Paired scores and activation recount', 'outputs/cctv_dgp_recognizer_precision_v1_analysis/precision_activation_review.json')}.

## Independent audit and preserved state

All {imported['returned_file_bindings']} return-file bindings, 192 stable input-response,
1,246 conditioning-response and 5,852 bank bindings pass. The {imported['archive_bytes']:,}-byte
archive SHA256 is `{imported['archive_sha256']}`. Manual measurement took
{reviewed['runtime_seconds']:.2f}s and export {reviewed['export_seconds']:.2f}s.
The run completed 16 recognizer forwards and 40 RGB-input derivative queries,
with zero restoration forwards, learned-weight changes, optimizer/accepted
updates or full epochs. Frozen states, buffers and saved Torch CPU/CUDA RNG
are preserved/restored. All ten FP32 repeats, ten FP64 repeats and ten FP32
versus prior-input comparisons meet the original elementwise and norm bounds.

The original prospective auditor passed without a returned-checker repair.
It independently replays complete gradient arrays and actual saved pixel
projections, embedding/scalar arithmetic, packed sign maps, transition counts,
repeat receipts and finite response classifications. Neural state and off-case
zero-gradient assertions remain VM receipts; no neural forward or derivative
was re-executed locally. Original checkpoint and protected app hashes pass.
{link('Independent audit', 'outputs/cctv_dgp_recognizer_precision_v1_analysis/independent_audit.json')}.

No new face images or visual sheets were planned or generated. The earlier
20 conditioning-response sheets/1,000 cells were directly reviewed and remain
soft/unqualified. Archive integrity and measurement success are not trained
model improvement. USED/DIAGNOSTIC COMPLETE: preserve this run and do not rerun.

## Next model-improvement decision

The assumption that higher recognizer precision alone would restore reliable
finite predictions is unsupported. Repeating precision diagnostics or adding
epochs to the failed single-initial-gradient recipe is not justified. The
next finite training design should consider gradients at the initial state
and nearby measured states, with every actual proposal checked under the
unchanged scientific RAW/PNG gates. This is a change to the learning calculation,
not promotion of the two selected references or any existing failed state.
{link('Bounded nearby-gradient training design', design_name)}.

The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The original restorer is DeblurGAN-v2-compatible; the comparison's StyleGAN2
bank is externally pretrained/frozen, not an own-trained face prior. All
checkpoints, splits, caches, provenance, full states and failures are retained.
Native CCTV remains unpaired and final identities remain outside tuning.
All five milestones and seven automatic/assisted covering-family requirements
remain incomplete. Training, derivatives and parameter changes remain manual
on the existing L4 in tmux; agent VM access remains maintenance only.
'''
    design = '''# Nearby-gradient learning review V1 — 14 October 2026

**The next finite training comparison should use nearby-state derivatives and
actual proposal acceptance, rather than the initial derivative alone.** The
precision diagnostic is complete: full FP64 leaves five of ten sampled finite
prediction misses, while packed PReLU maps show changed regions at both signed
endpoints in both precisions. This supports testing a learning direction that
accounts for more than one local region. It does not prove a unique root cause
or guarantee a safe improvement.

## Changed hypothesis and retained model

The earlier RAW R1 and joint RAW/PNG pilots already used actual backtracking;
their gradients came only from the current initial state and all proposals
failed preservation or the 1% structure requirement. Adding another backtrack
factor, changing only the learning rate, or using float64 reporting would be
an unchanged failed approach. This design instead changes the gradients used
to propose a direction, while keeping actual endpoint acceptance mandatory.

Start from the exact retained comparison state that reproduces the original
DGP, preserving the original app checkpoint permanently. Reuse the same 100
exposed paired TRAIN controls from both audited sources and all five profiles.
Keep all reference/mask/grid/target roles fixed; use no native CCTV or final
identity for this training comparison. The original reconstruction decoder,
bank and recognizer stay frozen. Only the declared 12 conditioning tensors
(2,580,864 parameters) may change in a disposable VM copy.

Collect fresh RAW group derivatives at three states: initial, and the exact
previously measured smallest plus02/minus02 conditioning states. These two
failed states are derivative sampling locations only and cannot be accepted
as training updates or bypass their historical failures. Fresh RAW outputs
must equal their saved counterparts, with the same batch order, scalar
alignment and frozen state checks, before gradients are used. Do not replace
scientific appearance scores with a high-precision diagnostic.

Keep all 32 source/profile RAW loss groups at each location: 96 inequality
rows. Fit a bounded common descent proposal against this local collection.
State any normalized targets, numerical representation, solver, feasibility
and certificate rules before execution. Preserve every original inequality
in the final actual-dot certificate; rank deficiency must not silently remove
losses or justify ridge-based gate relaxation. If a common certified direction
is infeasible, stop and retain the evidence. This is a finite constrained
experiment inspired by nearby-gradient methods, not an exact implementation
of the published random-sampling algorithm or a convergence guarantee.

## Finite proposed training scope

One direction solve, at most three actually evaluated proposals and at most
one accepted research update. Measure at most 960 derivative queries across
three complete 20-batch gradient rounds. No epoch continuation. Keep the
cumulative parameter-distance limit at 0.001 of the original all-37 norm and
the existing brightness-only-gain bound at 0.2. More epochs remain conditional
on useful preserved learning and independent development review.

Every proposal must preserve all 17 groups in both RAW and delivered PNG,
with the exact original MSE/SSIM/appearance bounds. Any provisional accepted
state must pass preservation and positive global/per-source structure progress;
the endpoint still needs at least 1% global and both-source structure gain in
RAW and PNG. Preserve provisional states and failures even if that endpoint
fails. Do not call a small accepted update a qualified model. Do not weaken
gates, crop out difficult faces, change masks or reinterpret PNG rounding as
an exemption. Restore frozen states and save full component state/RNG/code/
environment bindings needed for a declared continuation; never alter the
permanent checkpoint.

Prepare prospective timing/projection, VRAM/RSS, storage/reserve, output and
export stops before issuing manual commands. A planning ceiling is 1,350s
measurement, 1,500s worker, 600s per gradient round and 420s export, with finite
external termination. Reserve 9 GiB free before a run, a retained 1 GiB reserve
and at most 3.5 GiB output. Count the planned restoration/recognizer forwards
from the actual worker, not guessed estimates, before sealing a transfer.
These are proposed limits and are not a prepared runnable packet or fresh
VM capacity/timing evidence.

Save and independently replay every RAW/PNG output, embeddings, loss record,
gradient aggregate, actual rounded parameter displacement, solver certificate,
gate decision and rollback. All planned visual sheets for every new proposal
must be inspected. Native DEV review and final held-out review are separate
later requirements. The transfer, guard regressions, independent return checker
and exact Windows upload/tmux/single-source download guide remain to prepare.
No new actual training has been launched by this review.

## Research grounding and limitations

Burke, Lewis and Overton form a direction from gradients at nearby locations
and use a line search for nonsmooth objectives. Their theory has assumptions
and random sampling; it does not prove our deterministic three-state,
multi-constraint RAW/PNG face-restoration proposal. It motivates testing nearby
derivatives when measured activation regions change.
[Original SIAM paper, author-hosted copy](https://cims.nyu.edu/~overton/papers/pdffiles/gradsamp.pdf).

PReLU is piecewise linear at an individual activation. Observing sign changes
shows a region crossing, but the complete recognizer also contains normalization
and other operations. Do not claim the entire network is piecewise affine or
attribute every mismatch exclusively to PReLU.
[PyTorch 2.9 PReLU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.PReLU.html).
Finite-precision limitations remain distinct from finite nonlinear response.
[PyTorch 2.9 numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

All actual training/gradients/parameter changes stay human-manual L4 tmux.
Direct agent VM access is inventory-first, hash-bound maintenance only.
Local saved-array audits are allowed. Preserve original checkpoints, splits,
caches, provenance, full states, failed recipes and the app design. Retain
declared component origins. No acquisition without a demonstrated gap and
terms/overlap review. The user deadline and migration/credit instructions
remain as recorded; current remaining credits are not known. All five
milestones and seven automatic/assisted covering families remain required.
The full goal is incomplete.
'''
    retained_new(ROOT / report_name, report); retained_new(ROOT / design_name, design)
    prefix = f'''# Precision diagnostic complete; nearby-gradient training design — 14 October 2026

Recognizer precision V1 completed in 48.69s plus 2.98s export. All 462 return,
192 stable input, 1,246 response and 5,852 bank bindings pass. All 30 original
gradient-parity checks pass; frozen states/buffers and saved Torch CPU/CUDA RNG
are preserved. Zero restorer forwards, learned-weight/optimizer/accepted updates
or epochs. {link('Audited result', report_name)}.

Full FP64 changes sampled appearance misses from six to five of ten. It does
not resolve the two selected degraded failure cases. Both precisions show
PReLU region transitions at the signed endpoints; initial repeats show none.
No new restored images or structure gain. Original RAW/PNG gate failures stand.
USED/DIAGNOSTIC COMPLETE: do not rerun the precision packet.

Next design: collect derivatives at initial and two already measured nearby
states, retain all loss constraints, and require actual RAW/PNG proposal gates.
One solve / at most three proposals / one accepted research update; no epochs.
{link('Finite changed learning design; not yet a verified transfer', design_name)}.
Packet, regressions and return checker remain to prepare. Training/derivatives
and parameter changes stay human-manual L4 tmux; agent VM access stays maintenance
only. No new training or app change was launched.

All original checkpoints, splits, caches, provenance, full states and failures
remain. No new incumbent, model promotion or more epochs qualify. Native
unpaired / paired TRAIN / reserved final roles, all five milestones and seven
automatic/assisted covering families remain. Detector-confidence recording
remains prepared only and UNRUN. Goal incomplete. Exact previous document follows.

'''
    documents = {n: finish_prepend(ROOT / n, prefix, stage / 'before' / n, h) for n, h in expected.items()}
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(stage / 'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__), protocol_sha256=PIN, documents=documents,
        new_documents_sha256={n: sha(ROOT / n) for n in [report_name, design_name]},
        evidence_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in files},
        diagnostic_used=True, next_training_is_design_only=True, local_neural_or_training_calls=0,
        automatic_VM_launch=False, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, history_documents=9, reports=2, next_training_design_only=True))


if __name__ == '__main__':
    main()
