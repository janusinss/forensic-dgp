"""Publish the audited input-boundary result while preserving exact document history."""
from datetime import datetime, timezone

from import_cctv_dgp_appearance_input_response_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new


def main():
    stage = ANALYSIS / 'publication'
    (stage / 'before').mkdir(parents=True, exist_ok=True)
    assert not (stage / 'publication.json').exists()
    files = [ANALYSIS / n for n in ['import.json', 'independent_audit.json', 'boundary_review.json']]
    imported, audited, review = [read(n) for n in files]
    assert all(v['complete'] for v in [imported, audited, review])
    assert audited['full_diagnostic_returned'] and not audited['failure'] and audited['all_original_parity_checks_pass']
    assert [v['input_canonical_fail'] for v in review['summaries'] if v['repeat'] == 0] == [9, 6, 3]
    assert [v['input_double_fail'] for v in review['summaries'] if v['repeat'] == 0] == [9, 6, 3]
    old = read(OUT / 'cctv_dgp_appearance_input_response_v1_publication/publication.json')
    addition = read(OUT / 'cctv_dgp_appearance_input_response_v1_checker_addendum/publication.json')
    expected = {n: b['after_sha256'] for n, b in old['documents'].items()}
    expected.update(old['new_documents_sha256'])
    expected.update({n: b['after_sha256'] for n, b in addition['documents'].items()})
    assert len(expected) == 8
    assert all(sha(ROOT / n) == h for n, h in expected.items()), 'Preserve concurrent document changes'
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    result = 'CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_RESULTS.md'
    design = 'CCTV_DGP_RECOGNIZER_PRECISION_REVIEW_V1.md'
    table = []
    for v in review['summaries']:
        if v['repeat']:
            continue
        table.append(f"| {v['pair']} | {v['input_canonical_pass']} / {v['input_canonical_fail']} | {v['input_double_pass']} / {v['input_double_fail']} | {v['parameter_pass']} / {v['parameter_fail']} |")
    text = f'''# Appearance-input response V1 result — 14 October 2026

**The diagnostic completed and passed independent auditing. No model was trained
or improved by this run.** All 100 repeated RGB-input gradients meet the original
elementwise and norm bounds. The finite appearance predictions still disagree
with several actual scores, even after the reconstruction decoder is removed
from the differentiation path.

| Retained signed magnitude, largest to smallest | RGB-input canonical pass / fail | Saved double-dot reporting pass / fail | Previous parameter prediction pass / fail |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

Each row contains ten source/profile groups. Both repeats produce these same
classifications. All 60 group comparisons are resolved under the predeclared
diagnostic noise rule. The retained 5% comparison bound is for response diagnosis;
it does not change any restoration, appearance, RAW/PNG or preservation gate.

At the smallest magnitude, input-space failures remain in
`dataset/asian_faces/lowlight_lr32`,
`dataset/thumbnails128x128/motion_lr48`, and
`dataset/thumbnails128x128/compound_lr24`. The latter two groups cancel roughly
91.91% and 94.74% of the signed individual predictions. Asian-source lowlight
cancels roughly 94.25%. Per-case records still contain real prediction misses;
cancellation is a measured complication, not a complete explanation or a reason
to drop those cases. Dataset names describe source provenance, not ethnicity.

The smallest thumbnail clear-group prediction now agrees in RGB space while
its parameter-space prediction disagrees. At the middle magnitude, Asian-source
motion disagrees in RGB space while the parameter prediction agrees. Earlier
reconstruction-to-RGB behavior therefore also contributes in some comparisons.
The evidence does not support attributing every problem exclusively to either
the decoder or the recognizer.

## What is established

The measurement uses the same 100 exposed paired photographic TRAIN cases and
the exact initial and six signed RAW arrays from the prior completed response
diagnostic. It constructs only the frozen measurement recognizer. There are
40 recognizer forward batches, 200 RGB-input derivative queries, zero restoration
forwards, zero neural-weight displacements, zero optimizer or accepted updates,
zero full epochs, and zero native CCTV or reserved-final pixels.

All 193 returned-file bindings, 1,246 stable response bindings and 5,852 bank
bindings pass. Archive: {imported['archive_bytes']:,} bytes, SHA256
`{imported['archive_sha256']}`. Measurement took {review['runtime_seconds']:.2f}s;
export took {review['export_seconds']:.2f}s. Original frozen weights, buffers and
saved Torch CPU/CUDA random state are preserved/restored.
{link('Independent saved-evidence audit', 'outputs/cctv_dgp_appearance_input_response_v1_analysis/independent_audit.json')}.

The local audit independently replays gradient shapes, byte hashes and norms,
embeddings and scalar arithmetic, frozen 2×3 affine grids, exact actual saved
RGB-displacement projections, repeat comparisons and group response receipts.
Off-case zero derivatives and frozen neural state are VM receipts; those neural
computations were not repeated locally. The original local checker and its
pre-run zero-reference reporting repair remain retained. The repair changes
only the undefined relative ratio for a zero reference; this actual run contains
no zero RGB gradients. No original parity or scientific bound was relaxed.

Maximum absolute repeat-gradient difference is
{review['maximum_repeat_absolute_gradient_difference']:.10g}; all 100 original
bounds pass. Saved double-dot arithmetic leaves the same 9/6/3 group failures.
This recalculation uses already saved float32 embeddings; it is not a network
forward or backward in float64. Correct full-network high-precision behavior,
the cause of finite disagreements, and all-dimensional derivative correctness
are still unproved.
{link('All 60 group and 600 case response records', 'outputs/cctv_dgp_appearance_input_response_v1_analysis/boundary_review.json')}.

## Effect on the model goal

No new restored images or visual sheets were planned or generated. The earlier
20 sheets/1,000 cells were directly reviewed and still show soft details and
no convincing whole-face gain. All prior signed-point preservation failures,
including delivered PNG failures, remain binding. This run supplies no new
structure improvement percentage and does not change the previous rejected
gain figures. USED/DIAGNOSTIC COMPLETE: retain the run; do not rerun it.

Next, separate full-network precision and activation behavior from simple
saved-score reporting before preparing another optimizer or epoch schedule.
{link('Bounded next design', design)}. A completed diagnostic is not a qualified
restorer or authorization to bypass a failed stop.

The permanent app checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The retained restorer is DeblurGAN-v2-compatible; the comparison's frozen
StyleGAN2 bank is externally pretrained. No own-trained replacement prior or
new incumbent is claimed. Checkpoints, splits, caches, provenance, full states,
failures and the app design remain. Native unpaired evidence and reserved final
identities remain separate. All five milestones and seven automatic/assisted
covering families remain required. More epochs and the full goal remain
unqualified/incomplete.
'''
    review_text = '''# Recognizer precision and activation review V1 — 14 October 2026

**Compare the complete frozen recognizer computation in float32 and float64
before another training update.** Appearance-input V1 passed independent auditing
and all 100 repeat-gradient bounds, but finite input responses still miss 9, 6
and 3 source/profile groups as the saved displacement shrinks. Saved double-dot
reporting does not remove those misses. This places a demonstrated disagreement
at the recognizer input boundary without proving a particular backward bug.

Two explanations remain distinguishable: finite intervals crossing nonlinear
activation regions, and precision effects earlier than the final scalar dot.
Cancellation among individual cases can amplify a small group residual. The
current fixed recognizer explicitly performs its crop, encoder output and
normalization in float32. A final double dot therefore does not test a complete
float64 forward/backward. The higher-precision branch must be declared separately
and must not replace the official scientific score or its quality gates.

## Bounded next measurement

Use one problematic TRAIN reference from each retained source and all five
profiles for each: ten already exposed cases. Freeze selection from the saved
smallest-magnitude input-response failures, before any new execution. Compare
the exact initial, plus02 and minus02 RGB arrays, with an initial repeat in
each precision. No new restoration, target, alignment or parameter direction.

Construct a disposable recognizer precision copy. All parameters remain frozen;
double tensors must round-trip to the original float32 state exactly. Promote
the existing RGB values, fixed grid and float32 neutral fill exactly. The
float64 branch must retain double crop, encoder output, normalization and loss,
rather than silently casting back to float. Keep the original float32 branch
and official saved scores unchanged. Unsupported kernels or mixed precision
must stop and export evidence; no fallback, tolerance relaxation or auto retry.

Record complete RGB-input gradients, output embeddings, scalar scores and
repeat receipts for both precisions. Record packed activation sign maps for
PReLU/ReLU modules to expose crossed regions, with saved shapes and hashes so
transition counts can be independently recounted. Numerical agreement or sign
transitions in this small sample are diagnostic evidence, not a universal
derivative proof or image-quality improvement.

Planned maximum: 16 recognizer forward batches and 40 RGB-input gradient
queries; zero restoration forwards, optimizer updates, accepted updates,
learned-weight updates, new restored images or training epochs. Measure only
the two frozen references. Retain original failures and exact source binding.
Require finite timing/projection, RSS/VRAM, free-space/reserve, output and export
limits before sealing a packet. Preserve original checkpoints and rollback
the saved random state. The packet, importer, independent replay and exact
manual commands still need preparation; this document is a design, not a
verified runnable transfer.

If float64 improves derivative/finite-response agreement, treat numerical
precision as a supported contributor and assess a separately declared learning
calculation while still requiring the original actual float32/PNG checks.
If discrepancies remain and activation regions change, design against the
measured finite response rather than extrapolating first-order predictions.
If neither explains them, retain the unresolved diagnostic and investigate
the specific converted operators. No outcome automatically launches training
or qualifies more epochs. The original conditional epoch comparison still
requires useful preserved learning first.

## Primary research and limits

PyTorch's gradcheck documentation uses central finite differences to compare
analytical derivatives. This motivates the directional comparison; it does
not establish that our network has passed a full gradcheck.
[PyTorch 2.9 gradcheck mechanics](https://docs.pytorch.org/docs/2.9/notes/gradcheck.html).
Its numerical-accuracy documentation describes finite precision and the limits
of exact reproducibility. This supports isolating precision, not assigning it
as the cause before measurement.
[PyTorch 2.9 numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

All derivatives and any new precision-copy experiment stay human-manual on
the existing L4 in tmux. Agent VM access is inventory-first, hash-bound
maintenance only. Local saved-array audits are allowed. No agent neural
training, gradient query, temporary neural-weight proposal or VM launch
occurred during the appearance-input return audit. Paired TRAIN, native
unpaired and reserved final roles remain separate. Source labels do not imply
ethnicity or Zamboanga performance. The app design and permanent DGP remain.
Completion's seven families, automatic mask preview/correction, exact visible
appearance, clearer/less-covered refusals and original/mask/one plausible
estimate with downloads remain separate. All five milestones and independent
final/app reviews remain; the full goal is incomplete.
'''
    retained_new(ROOT / result, text)
    retained_new(ROOT / design, review_text)
    prefix = f'''# Appearance-input diagnostic completed; training still unqualified — 14 October 2026

The manual frozen-recognizer diagnostic completed in 77.61s plus 6.27s export:
40 forward batches, 200 RGB-input derivatives and zero optimizer/accepted
updates, restoration forwards, neural-weight displacements or epochs.
All 193 return / 1,246 stable response / 5,852 bank bindings pass. All 100
repeat-gradient checks pass the original bounds; maximum absolute difference
is 4.66e-10. Frozen weights/buffers and saved Torch CPU/CUDA RNG are preserved.
{link('Audited result', result)}.

Finite RGB-input appearance responses still disagree in 9/6/3 of ten groups
from largest to smallest magnitude; saved double-dot reporting leaves the same
misses. Individual improvements/regressions cancel heavily in some groups.
This locates a disagreement at the recognizer input without uniquely proving
a backward bug. Earlier reconstruction-to-RGB behavior also contributes in
some cases. No new images, useful detail gain or candidate was produced.

USED/DIAGNOSTIC COMPLETE: do not rerun this returned packet. Next design:
compare a disposable full-recognizer precision copy and activation transitions
on ten exposed TRAIN cases before another optimizer/epoch schedule.
{link('Bounded next design; not yet a runnable packet', design)}.
All original RAW/PNG gates, previous failures, original checkpoints, splits,
caches, provenance, full states and app design remain. No more epochs qualified.
Training/derivatives stay human-manual L4 tmux; direct agent VM access remains
maintenance only. Native unpaired / paired TRAIN / reserved final roles,
all five milestones and seven automatic/assisted covering families remain.
Detector-confidence work remains prepared only and UNRUN. Goal incomplete.
Exact previous document follows.

'''
    records = {n: finish_prepend(ROOT / n, prefix, stage / 'before' / n, h) for n, h in expected.items()}
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(stage / 'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__), protocol_sha256=PIN, documents=records,
        new_documents_sha256={n: sha(ROOT / n) for n in [result, design]},
        evidence_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in files},
        diagnostic_used=True, model_qualified=False, more_epochs_qualified=False,
        next_manual_diagnostic_is_design_only=True, local_neural_or_training_calls=0,
        automatic_VM_launch=False, goal_complete=False))
    print(dict(complete=True, documents=len(records), new_reports=2, model_qualified=False))


if __name__ == '__main__':
    main()
