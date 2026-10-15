"""Publish audited replay outcomes without changing immutable research evidence."""
from datetime import datetime, timezone
from pathlib import Path

from cctv_dgp_spatial_path_replay_v1_contract import read, write, sha, CHECKPOINT
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_analysis'


def link(label, name):
    return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def main():
    publication = ANALYSIS/'publication'
    assert not (publication/'publication.json').exists(), 'Retain prior publication'
    proofs = ['import.json', 'independent_audit.json', 'curve_summary.json',
              'visual_observations.json', 'visual_binding_audit.json', 'hinge_import_regression_audit.json']
    evidence = {name:read(ANALYSIS/name) for name in proofs}
    assert all(row['complete'] for row in evidence.values())
    imported, audit, curves, visual, binding, hinge = [evidence[n] for n in proofs]
    assert audit['compositions_checked'] == 1500 and binding['cells'] == 1800
    assert visual['sheets_reviewed'] == binding['sheets'] == 40
    assert curves['no_sampled_qualifying_state'] and hinge['exact_scalar_and_metadata_parity']
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == CHECKPOINT
    protected_path = ANALYSIS/'protected_before.json'
    protected = read(protected_path)['files']
    # Instruction documents will receive versioned prefixes; protected model/app
    # assets must still match their exact pre-import hashes.
    for name, digest in protected.items():
        assert sha(ROOT/name) == digest, 'Preserve concurrent research edit: '+name
    publication.mkdir(exist_ok=True)
    (publication/'before').mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for route, color in [('A', '#1565c0'), ('B', '#d14900')]:
        rows = [r for r in curves['states'] if r['route'] == route]
        steps = [r['update'] for r in rows]
        axes[0].plot(steps, [r['raw_gain_percent'] for r in rows], '--', color=color, alpha=.65, label=route+' RAW')
        axes[0].plot(steps, [r['png_gain_percent'] for r in rows], 'o-', color=color, label=route+' PNG')
        axes[1].plot(steps, [r['png_failures'] for r in rows], 'o-', color=color, label=route+' PNG')
    axes[0].axhline(1, color='#555555', linestyle=':', label='1% structure requirement')
    axes[0].set(xlabel='Saved training update', ylabel='Structure error reduction (%)', title='Paired photographic TRAIN; no qualifying state')
    axes[1].axhline(0, color='#555555', linestyle=':')
    axes[1].set(xlabel='Saved training update', ylabel='Failed group–metric checks', title='Preservation requires zero failures')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    plot = ANALYSIS/'learning_path.png'
    assert not plot.exists()
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    rows = '\n'.join('| '+str(r['route'])+' | '+str(r['update'])+' | '+f"{r['raw_gain_percent']:.6f}%"+' | '+f"{r['png_gain_percent']:.6f}%"+' | '+str(r['raw_failures'])+' / '+str(r['png_failures'])+' |' for r in curves['states'])
    result_name = 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_RESULTS.md'
    review_name = 'CCTV_DGP_POST_PATH_REPLAY_REVIEW_V1.md'
    result = f'''# Saved learning-path replay results — 15 October 2026

**USED / MEASUREMENT COMPLETE / QUALITY FAILED.** The manually executed replay
completed successfully. All 14 sampled A/B states failed the unchanged full
quality gate; no accepted checkpoint was found. This run performed zero new
optimizer updates, gradients, backwards calls or additional epochs.
Original app checkpoint: `{CHECKPOINT}`.

## Measured path

| Route | Saved update | RAW structure gain | PNG structure gain | Failed RAW / PNG checks |
|---|---:|---:|---:|---:|
{rows}

The highest sampled PNG gain is **0.392688% at B30**, with 13 RAW / 14 PNG
failed preservation checks. B30 source PNG gains are 0.268559% for Asian-source
replay and 0.423832% for HQ FFHQ counterparts. The requirement remains ≥1%
globally and in each source, with zero preservation failures across all 17 RAW
and 17 PNG groups and brightness fraction ≤0.2. Failed checks overlap; they
are not counts of distinct people or damaged images. A30 has fewer failures
(3 RAW / 4 PNG) but only 0.115815% PNG structure gain.

Both routes already fail preservation at the **first sampled update, 1**.
There is no qualifying sampled intermediate checkpoint. Unsampled updates were
not measured; this does not assert that every individual unsampled state fails.
The results reject a purely late-overtraining explanation. They do not isolate
learning rate, loss weighting, radius, data or architecture as the sole cause.
No retrospective TRAIN checkpoint selection, extra epochs or app promotion follows.

Structure gain is a reduction of the fixed facial high-frequency error, not
recognition accuracy, recovered identity or a visible-quality percentage.
Approximately 92.94% of B30's aggregate PNG MSE gain is explainable by the
constant-mean-shift control. That fraction is about MSE gain, not image area.

{link('Saved curves and exact source reductions', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/curve_summary.json')}.
![Audited learning path](<{plot.as_posix()}>)

## Direct visual review and evidence scope

All **40 sheets / 1,800 cells** were directly reviewed at original resolution.
The states remain close to the soft original. Eyes, lids, nose edges, lips,
teeth and ordinary hair show no convincing useful whole-face detail improvement
at the sampled states. B changes tone more than facial detail. Clear inputs are
already softened by forced original restoration. Clear glasses remain visible,
but blurred eye/lens details remain unresolved. Padding stays unchanged.
Hands, hair and headwear in these restoration sheets do not establish completion.
Request a clearer or less-covered crop where usable structure is insufficient.

The evidence uses **20 exposed photographic TRAIN references / 100 synthetic
views**, ten references per source. `dataset/thumbnails128x128` is the historical
key; the targets are audited HQ FFHQ counterparts. These are paired training
diagnostics, separate from native unpaired CCTV, final identities and covering
qualification. Source labels support no ethnicity or Zamboanga performance claim.
The current restorer is fine-tuned DeblurGAN-v2-compatible; the frozen external
StyleGAN2 bank is declared. No own-trained GAN or exact published DGP is claimed.

{link('Per-reference direct observations', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/visual_observations.json')}.
{link('Exact reviewed sheet/cell bindings', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/visual_binding_audit.json')}.

## Import and independent verification

Archive: **1,270,358,295 bytes**, SHA256
`6ce7e5b4129a995c6efc830ab31b1a098bc8874c02bf711fd4e5a081cf412787`.
The importer verifies all 1,921 regular members, sizes and hashes before
extraction and rehashes the extracted files. It verifies 410 source and 203
parent file bindings. The independent local audit recomputes 1,500 saved RAW/
PNG compositions, all 17+17 group decisions, 14 state byte streams and 500 exact
original/initial/endpoint control cases. All 40 returned sheets match the
independently recomputed 1,800 image cells. Original checkpoint/app assets remain.

Worker time is 350.81 seconds, VM saved audit 243.81 seconds and export 56.97
seconds; all supervised stages exit zero without timeout. Local saved-array/
state audit takes 158.77 seconds. Local work uses no neural calls, gradients,
optimizer updates, parameter installation or agent VM execution. Recognizer
embeddings are saved VM evidence, not a fresh independent local recognizer run.
This is an independent agent audit and direct review, not final human review.

{link('Import receipt', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/import.json')}.
{link('Independent numerical audit', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/independent_audit.json')}.

## Demonstrated processing limitation found during design review

Location: the bound spatial training worker's brightness-guard import at line129.
Cause: it imports the legacy parent helper containing three `clamp_min(0)`
hinges. The previously audited corrected helper uses ReLU at those boundaries.
The older helper has a nonzero boundary derivative when the scalar penalty is
zero; the current A/B worker therefore reintroduced a known gradient defect.
This was a code-selection mistake. The review must fix that import before
another learning-rate or extra-epoch comparison.

The actual legacy helper hash is
`7474a7688872a130bf60ced51eff37e25017f295f73a459b06d082c23d4ca393`;
the corrected helper hash is
`8368f97c13b4703834c6ea249a85a6970322dce45416bfecd23d8a299f6706f8`.
Only the three boundary operators differ. On this returned replay, all 100
tested five-view blocks / 500 saved compositions retain exact penalty values
and metadata under the corrected helper. No local derivatives were computed.
The historical manual L4 hinge audit established zero corrected initial guard
gradients on all 20 references. Fresh A-path component gradients and finite
repair effects still need the new manual study; this defect is not proven to
be the sole cause of weak structure or preservation failure.

PyTorch's pinned implementation sends the clamp-min gradient through equality
at its boundary. [PyTorch 2.9.1 derivative source](https://github.com/pytorch/pytorch/blob/v2.9.1/tools/autograd/derivatives.yaml).
{link('Hash-bound import and scalar audit', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/hinge_import_regression_audit.json')}.
{link('Post-return decision', review_name)}.

All five milestones, native development/final review, automatic selection and
override, app regressions and seven automatic/assisted covering families remain
required. No model qualification or goal completion is claimed.
'''
    review = f'''# Post-path-replay design decision — 15 October 2026

**The replay is complete; its sampled states remain rejected.** The original
app baseline remains `{CHECKPOINT}`. The human's instruction to apply the best
approach is carried forward. No higher learning rate, extra epoch or unchanged
failed optimizer continuation is authorized by this result.

## Correct a demonstrated limitation before changing the training scale

The invalid assumption was that the new spatial worker retained the previously
corrected brightness derivative. It actually imports the legacy helper. A rate-
only comparison would knowingly keep this defect. The selected next factor is
the scalar-equivalent **hinge derivative correction**, on a separate fresh A
decoder copy. Keep Adam at1e-4, the same 15 tensors /609,219 elements, schedule,
anchors, objective weights, clipping, original-relative radius, data and RAW/
PNG gates. Changing the boundary derivative is the only intended objective change.
The corrected module must be loaded by its verified exact file, avoiding name
resolution back to the old helper. Historical failed workers remain immutable.

Before any optimizer update, require exact original/A0 and retained old-A1
RAW/PNG controls. On all20 initial reference blocks, save old/corrected penalty
values, metadata and pixel-gradient arrays. Require exact scalar equality and
exactly zero corrected inactive gradients; retain the legacy evidence. Measure
two fresh combined parameter-gradient controls with all15 tensors connected.
These operations stay human-manual on the L4, never local neural/autograd work.

Use one fresh corrected Adam update. Evaluate its full projected state and two
diagnostic-only smaller changes at1/16 and1/4 along the corrected first change.
They test amplitude without selecting a model or changing the learning rate.
Restore the exact full first-update state after the probes. Continue only if
that full state has positive global and each-source structure gains and zero
RAW/PNG preservation or brightness failures. The diagnostic probes do not
authorize training and never replace the initial state or optimizer trajectory.

A conditional continuation has at most50 new updates and fixed snapshots at
1,10,20,30,40,50. Stop at the first scheduled preservation regression, nonpositive
global/source structure, nonfinite value or resource limit. Full qualification
still requires≥1% globally and in both sources, all preserved groups and the
brightness gate, followed by independent visual and development review. A
positive sub-1% intermediate value permits only the bounded trajectory; it is
not a relaxed acceptance rule, useful model or reason for extra epochs.

Save every actual gradient/proposal/projected update, full model/optimizer/RNG
states, RAW arrays, PNG outputs, embeddings, controls and failures. Never resume
the failed old A/B optimizer or auto-launch a historical worker. This study
does not freeze new native/final selection or change app/completion behavior.

**Design decision only at this publication.** A new packet still requires
prospective independent checking, boundary regression tests, source/assets/
archive hashes, exact transfer/tmux/download commands and enforced finite
limits. No new transfer or actual training is certified by this document.
The proposed worker cap is2400s, external2430s, audit/export600s each with
external630s and30s kill grace; free disk/RAM8GiB, reserve1GiB, output3GiB,
VRAM20GiB and RSS12GiB. Final budgets must match the verified execution protocol.

{link('Audited replay result', result_name)}.
{link('Bound import-regression evidence', 'outputs/cctv_dgp_spatial_path_replay_v1_analysis/hinge_import_regression_audit.json')}.
All goal milestones and covering-family requirements remain; new improvement
and correction effectiveness are unproven until returned evidence is audited.
'''
    retained_new(ROOT/result_name, result)
    retained_new(ROOT/review_name, review)
    common = f'''# Saved path audited; known loss import regression found — 15 October 2026

**USED / MEASUREMENT COMPLETE / QUALITY FAILED.** All14 sampled states fail.
Best sampled PNG structure gain is0.392688% (B30), with14 PNG preservation
failures. Both routes first fail preservation at update1. All1,500 saved
compositions,14 state byte streams,500 exact controls and40 reviewed sheets /
1,800 cells pass the independent evidence audit. This replay adds zero updates.
No new checkpoint is accepted; the original app baseline remains retained.

The training worker reimported the legacy brightness helper, reintroducing a
previously corrected zero-penalty boundary-gradient defect. This is a confirmed
processing limitation, not proof of the sole quality cause. The next design
isolates that derivative repair at the same Adam1e-4 before more epochs or a
rate change. A new manual-only finite packet is still being prepared; this
status does not certify it or launch training. All milestones, native/final
identities, RAW/PNG gates and seven automatic/assisted families remain required.
{link('Results and independent evidence', result_name)}.
{link('Repair-first decision and prospective stops', review_name)}.

Earlier UNRUN statements below describe preserved historical preparation;
this dated status supersedes them. Exact previous document follows.

'''
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_NEXT_PILOT.md',
        'CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md', 'CCTV_DGP_EXPERIMENT_PROGRESS_20261014.md']
    used = ['CCTV_DGP_SPATIAL_PATH_REPLAY_V1_VM.md', 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_PREPARATION.md',
            'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_LOCAL_RETURN.md']
    expected = {name:sha(ROOT/name) for name in docs+used}
    write(publication/'before_hashes.json', dict(complete=True, documents=expected))
    changed = {}
    for name in docs+used:
        prefix = common if name in docs else f'''# USED — saved replay imported and audited, 15 October 2026

The manual replay and independent local audit are complete. All sampled
quality gates fail; no new optimizer update or model qualification resulted.
Do not rerun this immutable packet. Historical commands below are retained
for provenance. {link('Results and next decision', result_name)}.

Exact previous document follows.

'''
        changed[name] = finish_prepend(ROOT/name, prefix, publication/'before'/name, expected[name])
    write(publication/'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        documents=changed, new_document_sha256={name:sha(ROOT/name) for name in [result_name, review_name]},
        evidence_sha256={name:sha(ANALYSIS/name) for name in proofs}, plot_sha256=sha(plot),
        baseline_sha256=CHECKPOINT, model_qualified=False, agent_VM_calls=0, optimizer_updates=0))
    print(dict(complete=True, documents=len(changed), result=result_name, next_decision='hinge repair; packet not yet certified'))


if __name__ == '__main__':
    main()
