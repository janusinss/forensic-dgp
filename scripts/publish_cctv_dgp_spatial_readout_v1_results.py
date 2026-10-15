"""Publish the independently audited quality stop, preserving exact history."""
from datetime import datetime, timezone
import json
from pathlib import Path

from cctv_dgp_spatial_readout_v1_contract import read, write, sha
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_readout_v1_analysis'
RETURN = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
OUT = RETURN/'outputs/spatial_readout_v1'


def link(label, name):
    return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def prose(text):
    # Explicit prose-only replacements, never executable names or hashes.
    for old, new in [('completed50', 'completed 50'), ('All20', 'All 20'), ('all20', 'all 20'),
        ('sheets /500', 'sheets / 500'), ('all17', 'all 17'), ('and17', 'and 17'),
        ('are20', 'are 20'), ('references /100', 'references / 100'), ('The50', 'The 50'),
        ('equal2.5', 'equal 2.5'), ('exits2', 'exits 2'), ('above the0.2', 'above the 0.2'),
        ('approximately100%', 'approximately 100%'), ('above1', 'above 1'),
        ("is0", "is 0"), ('All15', 'All 15'), ('and25', 'and 25'),
        ('All50', 'All 50'), ('spends52.381%', 'spends 52.381%'), ('and47.619%', 'and 47.619%'),
        ('on20/20', 'on 20/20'), ('on19/20', 'on 19/20'), ('on19', 'on 19'),
        ('worse on\n19/20', 'worse on\n19/20'), ('arrays and15', 'arrays and 15'), ('used5', 'used 5'),
        ('verifies781', 'verifies 781'), ('all15', 'all 15'), ('and203', 'and 203'),
        ('recomputes500', 'recomputes 500'), ('all100', 'all 100'), ('Every500', 'Every 500'),
        ('reports258.21s', 'reports 258.21 s'), ('are30.45s', 'are 30.45 s'), ('and86.04s', 'and 86.04 s'),
        ('verifies\n20 cases in4.14s', 'verifies\n20 cases in 4.14 s'), ('took133.03s', 'took 133.03 s'),
        ('at the authorized maintenance inventory,14', 'at the authorized maintenance inventory, 14'),
        ('October13:31:39', 'October 13:31:39'), ('had84.343', 'had 84.343'),
        ('all15/25', 'all 15/25'), ('PNG0.441%', 'PNG 0.441%'), ('and0.029%', 'and 0.029%'),
        ('at0,1,10,20,30,40,50', 'at 0, 1, 10, 20, 30, 40, 50'),
        ('Exact0/50', 'Exact 0/50'), ('a1/2/5-', 'a 1/2/5-'), ('A:0.', 'A: 0.'), ('B:0.', 'B: 0.'),
        ('RAW /0.', 'RAW / 0.'), ('PNG,8/', 'PNG, 8/'), ('PNG,18/', 'PNG, 18/'),
        ('about100%', 'about 100%'), ('the20%', 'the 20%'), ('the1%', 'the 1%'),
        ('All781', 'All 781'), ('members,15', 'members, 15'), ('protocol,203', 'protocol, 203'),
        ('assets,\n500', 'assets,\n500'), ('measured84.343', 'measured 84.343'),
        ('at13:31:39', 'at 13:31:39'), ('gain0.', 'gain 0.'), ('with8', 'with 8'), ('B0.', 'B 0.'),
        ('with18', 'with 18'), ('manual50-', 'manual 50-')]:
        text = text.replace(old, new)
    return text


def main():
    stage = ANALYSIS/'publication'
    stage.mkdir(exist_ok=True)
    (stage/'before').mkdir(exist_ok=True)
    assert not (stage/'publication.json').exists(), 'Retain published receipt'
    names = ['import.json', 'independent_audit.json', 'learning_attribution.json',
             'gallery_binding_audit.json', 'visual_observations.json', 'saved_objective_audit.json']
    values = {name: read(ANALYSIS/name) for name in names}
    assert all(v['complete'] for v in values.values())
    imported, audit = values['import.json'], values['independent_audit.json']
    learning, visual = values['learning_attribution.json'], values['visual_observations.json']
    objective = values['saved_objective_audit.json']
    assert audit['paired_RAW_PNG_compositions_checked'] == visual['cells_viewed'] == 500
    assert len(audit['sheets']) == visual['sheets_reviewed'] == 20
    assert len({v['id'] for v in visual['references']}) == 20
    assert all(not v['useful_new_detail_demonstrated'] for v in visual['references'])
    assert values['gallery_binding_audit.json']['cells'] == 500
    assert sha(RETURN/'protocol.json') == imported['protocol_sha256']
    expected = read(ANALYSIS/'protected_before.json')['files']
    checkpoint = 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    assert sha(ROOT/checkpoint) == expected[checkpoint]
    for name, digest in expected.items():
        assert sha(ROOT/name) == digest or ((stage/'before'/name).is_file() and sha(stage/'before'/name) == digest)
    outcomes = {o['route']: o['gates'] for o in audit['outcomes']}
    assert all(not g['pass'] for r in outcomes.values() for g in r.values())
    assert [len(outcomes[r]['png']['preservation_failures']) for r in ['A','B']] == [8,18]
    assert all(v['projection_applied_updates'] == 50 for v in learning['routes'].values())
    obj = objective['source_summaries']['all']['summary']
    gain = lambda route, mode: 100*outcomes[route][mode]['global_structure_gain']
    rows = '\n'.join(f"| {r} | {gain(r,'raw'):.6f}% | {gain(r,'png'):.6f}% | "
        f"{len(outcomes[r]['raw']['preservation_failures'])} / {len(outcomes[r]['png']['preservation_failures'])} | Rejected |"
        for r in ['A','B'])
    source_rows = '\n'.join(f"| {r} | {'Asian-source replay' if 'asian' in key else 'HQ FFHQ counterparts'} | "
        f"{100*outcomes[r]['raw']['source_structure_gains'][key]:.6f}% | "
        f"{100*outcomes[r]['png']['source_structure_gains'][key]:.6f}% |"
        for r in ['A','B'] for key in outcomes[r]['raw']['source_structure_gains'])
    report_name = 'CCTV_DGP_SPATIAL_READOUT_V1_RESULTS.md'
    review_name = 'CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md'
    report = f'''# Spatial reconstruction V1 returned results — 14 October 2026

**USED / QUALITY FAILED.** Both routes completed50 manual L4 updates, but
neither meets the unchanged structure and preservation requirements. There is
no accepted development update, full-corpus extra epoch, new incumbent or app
promotion. All20 planned comparison sheets /500 cells were directly inspected.
The original app checkpoint remains `{expected[checkpoint]}`.

## Actual endpoint results

| Route | RAW structure gain | Delivered PNG gain | Failed RAW / PNG group–metric checks | Decision |
|---|---:|---:|---:|---|
{rows}

The early requirement is ≥1% globally and in each source, with all17 RAW and
17 PNG groups preserved. These percentages measure relative reduction of the
fixed facial high-frequency error; they are not recognition accuracy or proof
of recovered identity. Failed checks are overlapping group–metric records,
not counts of distinct bad faces. Neither RAW nor delivered PNG qualifies.

| Route | Source | RAW gain | PNG gain |
|---|---|---:|---:|
{source_rows}

These are20 already exposed photographic TRAIN references /100 synthetic views,
ten references per source. `dataset/thumbnails128x128` is the historical key;
the actual targets are the audited HQ FFHQ counterparts. Asian-source replay
is reported separately, without ethnicity or capture-location inference. No
native CCTV, unexposed DEV, final identity or completion evidence was used.
The50 updates per route equal2.5 passes through this small sample, **zero full-
corpus additional epochs**. Archive completion cannot change those roles.

## Location, cause and next action

Location: the route endpoints in returned `outcome_A.json` and `outcome_B.json`;
the supervised training phase exits2 intentionally for the retained quality stop.
Cause of rejection: insufficient global and source structure gain plus actual
preservation regressions. This is a model-quality failure after computations,
not a missing-file, disconnected-gradient or CUDA-loading traceback.
Next action: retain both failed states and review the saved learning path before
choosing a new finite training change. Do not rerun this recipe or add epochs.

A regresses low-light RGB errors in both sources and some compound-view RGB,
SSIM and fixed-recognizer checks. B adds clear-image and blur/motion regressions;
its delivered clear aggregate MSE worsens
{learning['routes']['B']['group_deltas']['png']['clear']['MSE_relative_percent']:.3f}%.
B's RAW/PNG brightness-control ratio is
{outcomes['B']['raw']['brightness_gain_fraction']:.6f} /
{outcomes['B']['png']['brightness_gain_fraction']:.6f}, above the0.2 limit.
The constant-channel-shift counterfactual accounts for approximately100% of
its aggregate MSE gain; values slightly above1 mean that counterfactual improves
the score more than the candidate. This is not a percentage of all image changes.
A's aggregate brightness ratio is0, but its other failures remain.

All15 A and25 B selected tensors changed. Every tensor has a finite nonzero
derivative in the recorded manual preflight, and all five B kernels acquired
nonzero neighboring coefficients. Fixed observation features, normalization,
conditioning, external StyleGAN2, anchor readouts and recognizer stay unchanged.
The bank remains an external pretrained face prior; this is not an own-trained
GAN or an exact reproduction of a published DGP method.

All50 updates in each route hit the common original-relative distance boundary
`{learning['common_radius_L2']:.17g}`. A projection scales range
{learning['routes']['A']['projection_scale_range'][0]:.6f}–{learning['routes']['A']['projection_scale_range'][1]:.6f};
B scales range{learning['routes']['B']['projection_scale_range'][0]:.6f}–{learning['routes']['B']['projection_scale_range'][1]:.6f}.
B spends52.381% of final squared parameter displacement on the new spatial
connections and47.619% on the original decoder. This demonstrates an active
constraint and differing allocation, not proof that relaxing it would be safe
or sufficient. The scientific bound and appearance gates remain unchanged.

Saved-array evaluation of the declared weighted training formula gives mean
objective{obj['baseline']['objective']:.6f} initially,
{obj['A50']['objective']:.6f} for A and{obj['B50']['objective']:.6f} for B.
A lowers this objective on20/20 references, with much of the improvement in
the clear RGB anchor, while the structure term barely improves. B is worse on
19/20 references, with added anchor, regression and photometric costs. A lower
weighted objective is therefore insufficient evidence of the actual gate.
This uses fixed saved arrays and15-image embeddings; training used5-image
embedding batches. No new GPU objective or derivative was replayed locally.

## Visual evidence and audit coverage

A is very close to the original in all20 sheets. B mainly shifts tone; degraded
eyes, nose, lips and teeth stay diffuse. Clear forced restorations often retain
less texture than the input/target. Neither route demonstrates a convincing
useful new whole-face detail gain. Severe compound views remain insufficient
for readable features; request a clearer crop in that setting. Existing visible
glasses, ordinary hair, cloth and hands remain in these restoration comparisons;
their presence does not qualify automatic covering removal or hidden completion.

The2,107,676,784-byte archive SHA256 is
`{imported['archive_sha256']}`. Import verifies781 archive members,
all15 declared packet files plus protocol and203 parent assets. The prospective
local checker recomputes500 RAW/PNG compositions, all source/profile reductions,
original/initial parity, exact observed exterior pixels, both17-group gate
decisions, all100 projection formulas, final vectors, frozen states and
optimizer receipts. Every500 gallery cells exactly matches its pinned source.
The original checkpoint and documents stayed hash-identical throughout auditing.

The manual worker reports258.21s including cache/parity/preflight/endpoints;
actual fits are30.45s for A and86.04s for B. Manual fresh CUDA replay verifies
20 cases in4.14s. Local saved-output audit took133.03s. These are finite measured
times, not a promise for a larger study. Independent CUDA gradient replay and
independent final human review remain absent; the local work performs zero
learned-model construction, neural forwards, gradients or optimizer updates.

At the authorized maintenance inventory,14 October13:31:39 UTC
(21:31:39 Asia/Manila), the VM had84.343 GiB free. Cleanup was unnecessary;
no files were deleted and no agent launched training. Training was already
running under the human's manual workflow, later completed/exported, and the
human downloaded this return. Original checkpoints, caches, splits, states,
provenance, masks and every quality stop remain retained.

{link('Independent saved-output audit', 'outputs/cctv_dgp_spatial_readout_v1_analysis/independent_audit.json')}.
{link('Parameter and source attribution', 'outputs/cctv_dgp_spatial_readout_v1_analysis/learning_attribution.json')}.
{link('Fixed-array objective review', 'outputs/cctv_dgp_spatial_readout_v1_analysis/saved_objective_audit.json')}.
{link('Every directly reviewed sheet', 'outputs/cctv_dgp_spatial_readout_v1_analysis/visual_observations.json')}.
{link('Next evidence review', review_name)}.

The goal remains active/incomplete: useful native DEV, untouched final review,
qualified primary integration, app-flow regressions and all seven automatic/
assisted covering families remain required. No current model is declared best
possible or successful merely because this finite comparison completed.
'''
    retained_new(ROOT/report_name, prose(report))
    review = f'''# Post-spatial reconstruction evidence review — 14 October 2026

**Both new reconstruction routes are USED / FAILED.** The original incumbent
remains. The human already answered the previous design discussion with
“apply the best  approach to improve”; that question is resolved. This new
finite A/B comparison has two rejected routes, not three new unchanged reruns.
No automatic pilot, epoch extension, app change or relaxed gate is selected.

## What the evidence changes

Connectivity is no longer the missing evidence for these selected paths:
all15/25 selected tensors changed, and every spatial kernel's neighbors learned.
Adding spatial bank connections under this optimizer did not improve the
delivered result. Its source gain is especially uneven: PNG0.441% on Asian-
source replay and0.029% on HQ FFHQ, plus brightness and clear-image regressions.
This does not prove that spatial priors or a larger architecture cannot work.

The projection constraint is active from the first update through the last in
both routes. Additional B parameters split the same allowance. The worker
copies the projected live values back into its shadow parameters after every
Adam proposal, so an unbounded accumulating shadow base is not a demonstrated
bug. Do not modify historical code or inflate the radius to make a pass.

A improves its mean saved-array objective and clear anchor while facial
structure barely changes. B worsens the mean objective and19/20 references.
The weighted soft loss and per-reference bounded updates have not established
the exact delivered group acceptance. A dataset, step-size, batch or capacity
cause is not uniquely isolated. More epochs are not yet justified by this run.

## Next diagnostic direction

Use the already retained parameter journals to inspect **the finite learning
path**, with no new gradients, optimizer updates or epoch extension. Freeze
reconstruction snapshots at0,1,10,20,30,40,50 for both routes. Compare all100
same-input paired TRAIN cases at each snapshot with the original RAW/PNG
controls and unchanged17-group/source/brightness criteria. Keep every failure
and avoid picking an incumbent from a retrospective exposed-TRAIN curve.

New candidate construction, parameter replay and fresh neural evaluations
remain human-manual on the L4 in tmux. A separate hash-bound packet, CPU
boundary checks, prospective return checker and explicit time/call/storage
stops are required before transfer. No such new packet is certified by this
document; **it is not uploaded or launched by the agent**. Never invoke the
historical training worker or optimizer merely to reconstruct saved states.

The diagnostic should distinguish an endpoint problem from an already weak
whole path, and record which source/profile first fails. Exact0/50 parity with
this audited return is a control. If the path never improves structure while
preserving appearance, retain that evidence before redesigning the loss or
reconstruction. Do not restart the failed optimizer or extrapolate a1/2/5-
epoch benefit. Any subsequent design must explain how it differs from the
retained original-feature, loss-balance, V9, conditioning and bank failures.

This diagnostic is selected because the needed journal states already exist;
it adds no new training. It cannot by itself qualify useful native CCTV
restoration, an own-trained face generator or any covering family. The full
five-milestone/seven-family scope, final identities and original app design
remain unchanged.

{link('Audited result and exact evidence', report_name)}.
{link('Saved learning attribution', 'outputs/cctv_dgp_spatial_readout_v1_analysis/learning_attribution.json')}.
'''
    retained_new(ROOT/review_name, prose(review))
    prefix = f'''# Current spatial comparison: audited, USED / FAILED — 14 October 2026

Both reconstruction routes completed50 manual L4 updates, but neither qualified.
A:0.115752% RAW /0.132699% PNG,8/8 failed group–metric checks.
B:0.095199% RAW /0.111965% PNG,18/18 failed checks, plus brightness-only
contribution about100%, above the20% limit. All sources miss the1% requirement.
These are20 exposed photographic TRAIN references /100 synthetic views,
zero full-corpus extra epochs and zero accepted development updates.
{link('Current independently audited result', report_name)}.

All781 return members,15 declared packet files plus protocol,203 parent assets,
500 RAW/PNG compositions,100 projection steps and all20 sheets/500 exact cells
are checked. Every selected tensor changed; the new spatial neighbors learned.
All steps hit the unchanged original-relative radius. A lowers the weighted
objective mainly through clear RGB fitting; B worsens it on19/20 references.
No convincing useful new facial detail is established. Original checkpoint
`{expected[checkpoint]}` and app remain unchanged.

{link('Next diagnosis: retained learning-path replay, no new training', review_name)}.
A new finite manual diagnostic packet still requires preparation/verification;
this status is not a launch command or certified new transfer. Do not rerun the
USED recipe, append failed epochs, select a new incumbent or relax thresholds.
The prior diagnostic discussion was answered and remains resolved. Earlier
UNRUN/preflight-pending statements below are historical and superseded.

Authorized maintenance measured84.343 GiB free at13:31:39 UTC; no cleanup,
deletion or agent training launch occurred. Retain all original checkpoints,
splits, caches, full states, journals, source terms, masks and failures.
Actual training/derivatives/parameter replay remain human-manual L4 tmux;
agent VM connections remain inventory-first, hash-bound maintenance only.
All five milestones, native unpaired versus synthetic paired evidence,
untouched final identities, clearer/less-covered requests, original app design
and all seven automatic/assisted covering families remain. Goal incomplete.
Exact previous document follows.

'''
    docs = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_NEXT_PILOT.md', 'PROJECT_HANDOFF.md']
    prefixes = {name: prefix for name in docs}
    prefixes['CCTV_DGP_SPATIAL_READOUT_V1_VM.md'] = f'''# USED / FAILED — historical commands retained, do not rerun

The manual run and export finished. Independent return review confirms A PNG
gain0.132699% with8 failed checks; B0.111965% with18, both below1%.
All20 sheets are reviewed. There is no new incumbent or epoch continuation.
{link('Audited result and exact evidence', report_name)}.
{link('Next diagnostic direction; separate packet still unprepared', review_name)}.
The upload/launch commands below are preserved historical instructions for the
USED experiment. Do not use them to repeat this failed recipe. Return files
are already downloaded and independently imported. Original app unchanged.
Exact prior guide follows.

'''
    prefixes['CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md'] = f'''# This selected comparison is now USED / FAILED — 14 October 2026

Both manual50-update routes were independently audited and visually reviewed.
The spatial path learned but did not supply qualified structure or preservation.
{link('Actual result; original checkpoint unchanged', report_name)}.
{link('Selected next diagnosis; no new certified packet yet', review_name)}.
The original decision below remains exact historical evidence, with its
scientific requirements intact. No unchanged rerun, epoch extension or app
promotion follows from its completed preparation. Exact prior review follows.

'''
    changes = {}
    for name, header in prefixes.items():
        header = prose(header)
        prefixes[name] = header
        changes[name] = finish_prepend(ROOT/name, header, stage/'before'/name, expected[name])
    assert sha(ROOT/checkpoint) == expected[checkpoint]
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        status='USED_FAILED_INDEPENDENTLY_AUDITED_VISUALLY_REVIEWED',
        evidence_sha256={name: sha(ANALYSIS/name) for name in names},
        analysis_scripts_sha256={name: sha(ROOT/'scripts'/name) for name in [
            'audit_cctv_dgp_spatial_readout_v1_return.py', 'review_cctv_dgp_spatial_readout_v1_return.py',
            'audit_cctv_dgp_spatial_readout_v1_saved_objective.py']},
        documents=changes, prefix_sha256={name: __import__('hashlib').sha256(v.encode('utf-8')).hexdigest() for name,v in prefixes.items()},
        new_reports_sha256={name: sha(ROOT/name) for name in [report_name,review_name]},
        original_checkpoint_sha256=expected[checkpoint], model_qualified=False,
        accepted_development_updates=0, goal_complete=False, new_packet_prepared=False,
        agent_neural_training_or_gradient_calls=0)
    write(stage/'publication.json', record)
    print(json.dumps(dict(complete=True, documents=len(changes), reports=2, original_checkpoint_unchanged=True,
                         model_qualified=False, new_packet_prepared=False)))


if __name__ == '__main__':
    main()
