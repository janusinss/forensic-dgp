"""Publish the audited quality stop; retain every prior instruction byte."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
ANALYSIS = OUT/'cctv_dgp_decoder_hinge_repair_v1_analysis'
RETURNED = OUT/'cctv_dgp_decoder_hinge_repair_v1_return'
PIN = 'a4d57c933214345a6d1cab5639007d15599f85e0db0450a0aad11aea5d715518'
ARCHIVE_SHA = '7c06ddca900866797f31974e2c5349b1e2d1bc2f8a25ad34a4444fbb17f83b22'
BASELINE_SHA = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def link(label, relative):
    return f'[{label}](<{(ROOT/relative).as_posix()}>)'


def main():
    publication = ANALYSIS/'publication'
    receipt = publication/'publication.json'
    assert not receipt.exists()
    stage = RETURNED/'outputs/decoder_hinge_repair_v1'
    evidence = {name: ANALYSIS/name for name in [
        'import.json', 'independent_audit_r1.json', 'projection_metadata_r1.json',
        'projection_reserve_oracle.json', 'visual_observations.json',
        'visual_binding_audit.json', 'boundary_activation_independent_review.json']}
    records = {name: read(path) for name, path in evidence.items()}
    assert all(record['complete'] for record in records.values())
    audit = records['independent_audit_r1.json']
    boundary = records['boundary_activation_independent_review.json']
    visual = records['visual_binding_audit.json']
    assert audit['compositions_checked'] == 600 and audit['training_updates_audited'] == 1
    assert audit['model_qualified'] is False and visual['sheets'] == 20 and visual['cells'] == 800
    assert len(audit['guard_boundary']) == 20 and all(
        row['legacy_gradient_nonzero'] and row['corrected_gradient_exact_zero']
        for row in audit['guard_boundary'])
    assert boundary['new_parameter_proposals'] == boundary['optimizer_updates'] == boundary['agent_VM_calls'] == 0
    for relative, expected in boundary['evidence_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    archive = OUT/'cctv-dgp-decoder-hinge-repair-v1-results.tar.gz'
    assert archive.stat().st_size == 587914575 and sha(archive) == ARCHIVE_SHA
    assert sha(RETURNED/'protocol.json') == PIN
    baseline = OUT/'cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    assert sha(baseline) == BASELINE_SHA
    assert sha(ROOT/'scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py') == \
        'fa6f84b30690bcf43bd478370c1406d70d74d43f1af110904a3d8a8703f3e105'
    run = read(stage/'run_results.json')
    assert run['stop_reason'] == 'quality_stop' and run['training_updates'] == 1
    outcomes = {row['label']: row for row in audit['outcomes']}
    fit = outcomes['fit_u001']
    assert not any(row['continuation_eligible'] for row in audit['outcomes'])
    rows = []
    for label, title in [('legacy_A1', 'Retained legacy first step'),
                         ('fit_u001', 'Corrected first step'),
                         ('probe_1_16', 'Diagnostic 1/16 change'),
                         ('probe_1_4', 'Diagnostic 1/4 change')]:
        gates = outcomes[label]['gates']
        rows.append(f"| {title} | {100*gates['raw']['global_structure_gain']:.6f}% | "
                    f"{100*gates['png']['global_structure_gain']:.6f}% | "
                    f"{len(gates['raw']['preservation_failures'])} | "
                    f"{len(gates['png']['preservation_failures'])} |")
    table = '\n'.join(rows)
    components = boundary['component_scopes']
    probe = components['probe_1_16']
    delta = probe['global_component_deltas_from_baseline']
    fit_delta = probe['fit_reference_deltas_from_baseline']
    result_name = 'CCTV_DGP_DECODER_HINGE_REPAIR_V1_RESULTS.md'
    review_name = 'CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md'
    results = f'''# Decoder hinge repair: verified correction, rejected model — 15 October 2026

**The manual run stopped correctly after one optimizer update. No candidate
qualifies, no extra epoch is justified, and the application baseline is unchanged.**
The inactive brightness-gradient defect is repaired, but useful restoration and
preservation did not follow. Export completion packages the failed experiment.

## Audited comparison

| Saved state | RAW structure gain | PNG structure gain | RAW failures | PNG failures |
|---|---:|---:|---:|---:|
{table}

Structure gain is the relative reduction in landmark-supported high-frequency
luminance MSE against the baseline, not a percentage of human-visible clarity.
ArcFace recognition similarity is a separate preservation check. The unchanged
qualification threshold is at least 1%
globally and in each source, plus all preservation and brightness requirements.
Failures count overlapping group/metric decisions, not distinct people or images.
The corrected step fails 12 MSE, 10 SSIM and 8 recognizer comparisons in each
RAW/PNG path. All sampled states also have zero brightness-gain fractions;
passing that check does not compensate for the other failures. Diagnostic probes
are evidence only and never replace the full optimizer state or select a model.

Corrected source gains remain separate:

| TRAIN source namespace | RAW gain | PNG gain |
|---|---:|---:|
| `dataset/asian_faces/degraded` | {100*fit['gates']['raw']['source_structure_gains']['dataset/asian_faces/degraded']:.6f}% | {100*fit['gates']['png']['source_structure_gains']['dataset/asian_faces/degraded']:.6f}% |
| `dataset/thumbnails128x128/degraded` | {100*fit['gates']['raw']['source_structure_gains']['dataset/thumbnails128x128/degraded']:.6f}% | {100*fit['gates']['png']['source_structure_gains']['dataset/thumbnails128x128/degraded']:.6f}% |

Clear-image MSE worsens by 7.077% RAW and 14.738% PNG. Degraded-image MSE worsens
by 7.504% RAW and 7.608% PNG. This does not establish useful improvement over the
baseline. The root agent inspected all 20 eight-column, five-row sheets. Their
800 image cells match independently recomputed output pixels exactly. Facial
detail remains soft; severe compound cases remain insufficient for confident
restoration. This is an agent visual review, not independent final human review.

{link('Complete numerical audit', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/independent_audit_r1.json')}.
{link('Per-reference visual observations', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/visual_observations.json')}.
{link('Visual file and pixel bindings', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/visual_binding_audit.json')}.

## What the correction proves

All 20 initial legacy penalty-gradient arrays are finite and nonzero. All 20
corrected inactive arrays are exactly zero; scalar penalties and metadata remain
identical. The worker used the intended corrected helper. Adam remained at
1e-4, with the same 15 decoder tensors, 609,219 elements, loss weights, anchors,
reference order, original-relative radius, splits and quality requirements.
The first update changed direction, so this return is not an unchanged rerun.
It also shows that the old derivative was not the only cause of failure.

Local auditing checked 600 RAW/PNG compositions, 300 exact baseline/initial/
legacy controls, all 17 groups in each path, 40 saved gradient arrays, two full
states, the saved first Adam update, projected vector bytes, frozen tensors,
optimizer step and exact probe-state/RNG restoration. Saved embeddings and CUDA
derivatives remain VM evidence; the local auditor did not run a fresh recognizer
or neural/gradient/optimizer call. The successful R1 audit took
{audit['seconds']:.2f} seconds.

## Retained local audit stop and strict repair

**Location:** original local checker `verify_projection_journal`, line 90.
**Cause:** a CPU NumPy norm and VM reserve metadata differ by one binary64 ULP:
`7.041221370338014e-6` versus `7.041221370338013e-6`. The saved VM value matches
the independently derived correctly rounded exact reference. This scalar audit
portability issue did not change model weights or explain the quality failures.
**Fix:** the separate R1 wrapper uses an exact rational squared-spacing sum and
high-precision decimal reference. It requires exact equality, rejects even a
one-ULP metadata change, and retains exact actual-vector/radius checks. It does
not widen a quality or numerical tolerance. Six saved-vector regression tests
pass. The original checker, its failed receipt, all output bytes and protocol
remain unchanged.

{link('Preserved original failure', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/initial_audit_stop/failure.json')}.
{link('Independent exact reserve proof', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/projection_reserve_oracle.json')}.
{link('R1 metadata receipt', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/projection_metadata_r1.json')}.

## Run, provenance and scope

The human-manual L4 run made 1 optimizer update, 42 gradient queries and 1
backward call. It saved 20 original-restorer batches, 20 prior-feature batches,
103 decoder batches and 243 recognizer batches. The worker quality-stopped
after 169.54 seconds; its external exit was code 2 without timeout or kill.
The saved audit and export succeeded. No worker, new parameter proposal or
training was launched by the agent in this return audit.

The 587,914,575-byte archive SHA256 is
`{ARCHIVE_SHA}`.
Protocol SHA256: `{PIN}`.
The import verified 828 regular members, 410 source, 203 parent and 136 replay
bindings before the full audit. Original checkpoint SHA256 remains
`{BASELINE_SHA}`.

This is 20 exposed photographic TRAIN references with 100 synthetic views,
ten references per source. The historical `thumbnails128x128` namespace here
uses audited HQ FFHQ counterpart targets; it is not evidence of 128px clean
targets. Dataset labels do not identify ethnicity or capture country. No native
CCTV, final identity, Zamboanga sample or covering-removal claim follows.
Paired errors stay separate from native unpaired evidence. The retained baseline
is a fine-tuned DeblurGAN-v2-compatible conditional restorer; the frozen external
StyleGAN2 prior and separate pretrained completion component remain declared.

{link('Selected next causal review', review_name)}. The full five-milestone goal,
seven automatic/assisted covering families and independent final review remain
active and unqualified. This packet is USED; do not resume or rerun it.
'''
    review = f'''# Post-hinge-repair decision: examine boundary activation — 15 October 2026

**The selected next approach is to test preservation-aware update feasibility
before another training trajectory.** The two spatial routes and corrected
penalty run failed preservation. The user's latest instruction, “apply the best
approach to achieve the goal,” answers the required diagnostic question. The
invalid assumption is that correcting an inactive derivative and reducing the
same update's amplitude is sufficient to produce preserved learning.

## Evidence that selects this cause

The old inactive brightness gradients are now exactly zero in all 20 corrected
controls. The corrected first parameter change differs from the legacy change
(direction cosine 0.692262), so that mechanical defect was isolated. Nevertheless,
the full step has 30 RAW and 30 PNG failures. Its 1/16 and 1/4 amplitude controls
also fail. More epochs, a higher rate or another unchanged smaller-step recipe
are not supported by these results.

For the smallest tested change, the saved fixed-cohort objective decomposes as:

| Component change from baseline | All 20 references | Actual fitting reference |
|---|---:|---:|
| Reconstruction terms plus anchors | {delta['reconstruction_and_anchors']:+.9f} | {fit_delta['reconstruction_and_anchors']:+.9f} |
| Newly active preservation/photometric penalties | {delta['regression_and_photometric_penalties']:+.9f} | {fit_delta['regression_and_photometric_penalties']:+.9f} |
| Total saved diagnostic objective | {delta['objective']:+.9f} | {fit_delta['objective']:+.9f} |

Lower is better. The small reconstruction/anchor improvement is outweighed by
new penalties. Fourteen of 20 reference objectives rise at 1/16; all 20 rise
for the full corrected step. The actual fitting reference is `tr_asian_00176`.
These values use saved diagnostic recognizer embeddings from a 15-image batch;
they are not a fresh post-update five-image training loss. No sole cause of all
historical failures is claimed.

The saved clipped gradient has a negative dot product with the actual parameter
change (-0.006914795), while finite saved objective values rise. This describes
the selected autograd linearization and tested states, not proof that no feasible
decoder direction exists. All values above were independently recomputed from
retained arrays and component arithmetic.

{link('Independent boundary-activation evidence', 'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/boundary_activation_independent_review.json')}.
{link('Authoritative quality-stop result', result_name)}.

## Why the update direction needs review

At an inactive boundary, an implementation can choose a zero derivative for
`max(0, r(theta)-r(theta0))`. Yet a direction that increases `r` activates a
positive penalty immediately. Choosing a derivative at equality does not make
that direction a finite descent direction. For a smooth residual, its one-sided
directional penalty is `max(0, grad(r) dot d)` at equality. This equality example
applies to the zero-margin brightness hinges. The actual regression hinges have
1e-12 MSE and 1e-6 SSIM/recognizer offsets: their derivative remains zero inside
that slack, until a finite change exhausts it. Their activation distance must be
measured in metric units, not replaced by an equality formula. Actual neural
branches, batch-dependent recognizer behavior and PNG quantization require
separate checks. The tested positive penalties do not prove every infinitesimal
step ascends or that no feasible restoration model exists.

Stanford's primary optimization notes explain that a chosen negative subgradient
need not be a descent direction for a nonsmooth objective. That general
observation motivates the diagnostic; their convex convergence statements do
not establish convergence or feasibility for this neural model.
[Primary notes, section 2.1](https://web.stanford.edu/class/ee364b/lectures/subgrad_method_notes.pdf).
The [pinned PyTorch derivative source](https://github.com/pytorch/pytorch/blob/v2.9.1/tools/autograd/derivatives.yaml)
documents the implementation boundary already tested by the saved controls.

Previous work already tested PCGrad, constrained parameter directions and
signed small steps. A saved small step passed RAW preservation but failed six
PNG checks. Repeating those methods under a new name would not resolve this
question. The distinct scope must be the current 15-tensor A decoder, its exact
corrected full objective and actual Adam-preconditioned direction, including
finite activation margins and the unchanged `floor(float32*255)` delivery path.
{link('Earlier PCGrad method', 'CCTV_DGP_PCGRAD_FIT_V41_VM.md')}.
{link('Earlier constrained-direction study', 'CCTV_DGP_BANK_HINGE_BOUNDARY_V1_RESULTS.md')}.
{link('Earlier signed-step and PNG failures', 'CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_RESULTS.md')}.

## Next diagnostic requirements; no execution packet certified

1. Keep the original checkpoint and the exposed TRAIN cohort fixed. Reconcile
   the exact five-image fitting loss and 15-image saved diagnostic components
   before using either to predict descent; compare derivatives and finite TRAIN
   losses on the same batch-five path and retain the separate batch-fifteen audit.
2. Examine underlying preservation residuals, slack and activation distances for
   each reference/profile, plus the exact-boundary nested brightness hinges.
   A zero penalty gradient must not hide a finite crossing of an inactive margin.
3. Test whether an activation-aware direction can reduce structure error while
   respecting the existing preservation constraints. Use a finite feasibility
   diagnostic with retained per-constraint evidence; do not merely retune penalty
   weights, repeat PCGrad, scale the failed direction or enlarge the radius. Exclude
   zero/unchanged directions and require declared progress in both sources; a
   vanishing or PNG-unchanged proposal does not establish useful learning.
4. Check actual finite RAW and delivered PNG outputs against the original gates.
   A favorable derivative, loss prediction or TRAIN-only pass cannot authorize
   useful-model, native CCTV, final-review or app promotion claims.
5. Freeze any subsequent manual VM packet only after independent source/asset
   review, exact transfers and finite query/update, timing, storage and stop
   limits are verified. If no preserved improvement direction is demonstrated,
   retain the stop and review the reconstruction design instead of adding epochs.
   Any infeasible result concerns the declared local subspace and constraints.

This is a completed causal review and selected next question, not a completed
feasibility test, new model design, prepared VM packet or permission to resume
the failed optimizer. No new gradient or parameter proposal was computed
locally. The original five milestones, native DEV/final separation, all visible
features and seven covering families remain required. Automatic and assisted
completion remain separately unqualified. The app and permanent baseline stay
unchanged while this research decision is resolved.
'''
    retained_new(ROOT/result_name, results)
    retained_new(ROOT/review_name, review)
    prefix = f'''# USED: hinge repair audited; quality stop retained — 15 October 2026

The downloaded manual return is fully imported and audited. The inactive
brightness-gradient correction works in all 20 saved controls. The new model
still fails after **one Adam update**: **0.113913% RAW / 0.128470% PNG** structure
proxy gain, with **30 RAW / 30 PNG preservation failures**. Both smaller
diagnostic changes also fail. All 20 visual sheets / 800 cells are reviewed
and bound to independently audited pixels. No model qualifies or replaces the
original baseline; no extra epoch, unchanged rerun or optimizer resume follows.

The first local audit's one-ULP reserve-metadata stop is preserved. A separate
strict R1 checker completes all 600 compositions without widening tolerances
or changing vectors, radius or quality gates. Six regression tests pass.

The user's diagnostic answer selects the best supported approach. The next
review examines **one-sided preservation-penalty activation and feasible update
directions**: the tested smaller change reduces reconstruction terms but adds
larger regression penalties. This is saved TRAIN evidence, not native CCTV or
final qualification. No new VM execution packet or training is certified.

{link('Audited result and limitations', result_name)}.
{link('Selected next causal review', review_name)}.
The full five-milestone goal, seven automatic/assisted covering families,
manual-training boundary and original checkpoints/splits/failures remain intact.
The instructions below are retained historical state; this USED status takes
precedence over their PREPARED/UNRUN labels and upload/run commands.

'''
    previous = read(OUT/'cctv_dgp_decoder_hinge_repair_v1_design/publication/publication.json')
    expected = {name: record['after_sha256'] for name, record in previous['documents'].items()}
    expected.update(previous['new_document_sha256'])
    assert len(expected) == 10
    assert all(sha(ROOT/name) == digest or
               ((publication/'before'/name).is_file() and sha(publication/'before'/name) == digest)
               for name, digest in expected.items()), 'Preserve concurrent instruction edits'
    (publication/'before').mkdir(parents=True, exist_ok=True)
    documents = {name: finish_prepend(ROOT/name, prefix, publication/'before'/name, digest)
                 for name, digest in expected.items()}
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        documents=documents, new_document_sha256={name: sha(ROOT/name) for name in [result_name, review_name]},
        evidence_sha256={name: sha(path) for name, path in evidence.items()},
        original_checkpoint_sha256=sha(baseline), protocol_sha256=PIN,
        results_archive_sha256=ARCHIVE_SHA, results_archive_bytes=archive.stat().st_size,
        original_checker_unchanged=True, training_updates_audited=1,
        compositions_audited=600, sheets_reviewed=20, cells_reviewed=800,
        status='USED / AUDITED / QUALITY_STOP / NOT_QUALIFIED',
        selected_review='one-sided preservation boundary activation and finite update feasibility',
        new_packet_prepared=False, model_qualified=False, goal_complete=False,
        new_local_optimizer_updates=0, new_parameter_proposals=0, agent_VM_calls=0)
    publication.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True, superseded_documents=len(documents), new_reports=2,
                          status=record['status'], new_packet_prepared=False)))


if __name__ == '__main__':
    main()
