"""Publish an evidence-bound rejected diagnostic milestone with byte-exact backups."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_multiscale_return_milestone_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def link(name, label):
    return f'[{label}](<{(ROOT/name).as_posix()}>)'


def main():
    assert not OUT.exists()
    protected_path = ROOT/'outputs/cctv_dgp_multiscale_archive_cleanup_v1/local_protected_sha256.json'
    protected = read(protected_path)
    assert len(protected) == 346
    for name, digest in protected.items():
        assert (ROOT/name).resolve().is_relative_to(ROOT)
        assert sha(ROOT/name) == digest, name
    audit_path = ROOT/'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_path)
    visual_path = ROOT/'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1/visual_review.json'
    visual = read(visual_path)
    quality = read(ROOT/'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1/quality_summary.json')
    analysis_root = ROOT/'outputs/cctv_dgp_multiscale_calibration_v1_first_step_analysis'
    analysis = read(analysis_root/'analysis.json')
    analysis_audit = read(analysis_root/'independent_audit.json')
    color_root = ROOT/'outputs/cctv_dgp_multiscale_color_detail_v1_r2'
    color = read(color_root/'results.json')
    anchor_root = ROOT/'outputs/cctv_dgp_multiscale_active_anchor_v1'
    anchor = read(anchor_root/'results.json')
    anchor_audit = read(anchor_root/'independent_audit.json')
    assert audit['complete'] and audit['all_gate_decisions_and_failure_names_unchanged']
    assert visual['complete'] and visual['viewed_sheets'] == 64 and visual['exact_saved_PNG_cells'] == 2280
    assert quality['arms_passing_sampled_requirements'] == 0
    assert analysis_audit['complete'] and analysis_audit['analysis_sha256'] == sha(analysis_root/'analysis.json')
    assert color['complete'] and color['all_decisions_and_failure_locations_unchanged']
    assert anchor_audit['complete'] and anchor_audit['results_sha256'] == sha(anchor_root/'results.json')
    assert sha(ROOT/'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py') == 'a2e92da64ae224b1ec2f5b0d3bfaec0582e16e300c85f22b0d1c8f3124bbe7c9'
    OUT.mkdir()
    evidence_files = [audit_path, visual_path, analysis_root/'analysis.json', analysis_root/'independent_audit.json',
        color_root/'plan.json', color_root/'results.json', anchor_root/'plan.json', anchor_root/'results.json',
        anchor_root/'independent_audit.json', ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json']
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in evidence_files}
    write(OUT/'publication_plan.json', dict(complete=True, prepared_before_document_changes=True,
        evidence_bindings=bindings, protected_bindings_sha256=sha(protected_path), protected_count=346,
        publisher_sha256=sha(Path(__file__)), model_qualification=False, goal_complete=False))
    rows = []
    for arm in quality['arms']:
        raw, png = arm['comparisons']['raw'], arm['comparisons']['png']
        rows.append(f"| {arm['partition']} | {arm['lr']:g} | {arm['pool']} | {100*raw['relative_feature_gain']:.8f}% | {100*png['relative_feature_gain']:.8f}% | {len(raw['preservation_failures'])}/{len(png['preservation_failures'])} | Fail |")
    full = [r for r in analysis['arms'] if r['partition']=='decoder15' and r['lr']==.001]
    assert len(full)==2
    report = f'''# Multiscale DGP calibration V1: execution completed, quality rejected

10 October 2026. The downloaded packet is hash-verified and independently audited.
All 64 planned saved-PNG comparison sheets were actually inspected. Every one
of the twelve independent trials fails the unchanged quality requirements.
Keep the retained app DGP; do not resume or promote any calibration state.
The complete restoration and seven-family completion goal remains active.

## What ran

The user manually ran the finite diagnostic on the existing NVIDIA L4. Twelve
independent arms each start from the same retained checkpoint and make one Adam
update after balanced accumulation of ten reference batches. Total: 12 optimizer
updates, 120 backwards, 280 component-gradient queries, 600 fitting exposures,
zero complete epochs. Model work took 362.287 seconds, approximately six minutes.
The trainer exited successfully; the experiment's quality outcome is rejection.
Export completion establishes evidence packaging, not model qualification.

The two trainable partitions are the repaired deepest branch (3 tensors/147,456
elements) and the complete original RGB decoder (15 tensors/609,219 elements).
The MobileNet/FPN feature extractor, five stored normalization layers, original
checkpoint and fixed recognizer stay frozen. This is not whole-model training.

All arms evaluate the same 100 historically exposed paired photographic TRAIN
cases from 20 references, covering clear, blur, low light, motion and compound
profiles. These cases are not held-out validation. All 24 frozen ChokePoint C1
native development crops are reported separately as unpaired evidence. They
have no aligned clean face target. Final identity pixels enter neither fitting
nor tuning. Source namespace names do not establish ethnicity, capture country
or performance in Zamboanga City.

The return archive contains 1,639,351,615 bytes; SHA256:
`61cfd317f18b0fd99b6a05a48c03b67c5fbf42e751c6b31cef9c3762de511769`.
Protocol SHA256:
`c0239c2eebab3891cd4f42d34b93a5c7c2b07c8597bf93247df2cc16239dd491`.
The complete GZIP stream passes its integrity check.

## Identical-input results

Structure gain measures the reduction of the frozen landmark high-frequency
error against paired TRAIN targets. It is not an identity-recovery percentage
or a useful-image rating. Raw floats and delivered PNGs remain separate.
Failure counts below are group/metric entries, not counts of faces.

| Partition | Learning rate | Fitting pool | Raw structure gain | PNG structure gain | Raw/PNG preservation failures | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

Only the complete decoder at 0.001 exceeds the 1% early structure threshold.
Its clear-image raw pixel error rises from 0.00067381223 to 0.01668709704 in
pool0 ({full[0]['clear_raw_MSE_multiplier']:.3f} times) and 0.02124276589 in
pool1 ({full[1]['clear_raw_MSE_multiplier']:.3f} times). Clear-image SSIM and
fixed observed-face embedding similarity also worsen. The original 1% early,
10% later capacity, source/profile appearance and 20% brightness-share limits
stay unchanged. A zero brightness-only fraction when overall pixel error
worsens does not demonstrate preserved color or recovered detail.

## Independent audit and retained checker failure

R1 stopped on derived brightness-share arithmetic: 0.9264908100902142 versus
0.9264908100876813, a difference of 2.5328628e-12 above its 2e-12 allowance.
Its original source, traceback and unsuccessful execution receipt remain.
R2 permits 5e-11 only for that derived ratio and its recorded failure-value
field. Pixel, SSIM, embedding, structure thresholds, failure group names and
all Boolean decisions stay unchanged; gates recalculated from stored rows
are checked exactly. Nine negative controls reject changes to scientific
thresholds, preservation metrics, gate decisions and excessive ratio drift.

The successful R2 audit verifies 2,276 manifest files, 1,300 paired raw/PNG
records, 312 unpaired native records, 280 saved gradient vectors and all twelve
full model/optimizer/scheduler/RNG states. Ten fresh CPU inference replays
across two arms differ in raw pixels by at most 2.03e-6. Initial saved-gradient
accumulation and first-Adam arithmetic are independently checked. Local audits
make no new gradients or optimizer updates. This is independently implemented
arithmetic/integrity checking; the goal's independent final reviewer is still
required.

{link(audit_path.relative_to(ROOT), 'Independent return receipt')}.

## Completed visual review

All 40 paired TRAIN and 24 native development sheets were inspected: 64/64,
2,280 unscaled 256×256 RGB cells. A separate gallery audit verifies every
cell byte-for-byte against its saved PNG and binds every page hash. Sixteen
explicit visual batches bind page IDs, case IDs and observations.

The deep branch and lower-rate full decoder add little convincing whole-face
clarity. Severe synthetic profiles remain soft around eyes, noses and mouths.
The highest full-decoder rate produces blue/purple or gray/green washes,
lifted skin/backgrounds and changed highlights, without useful corresponding
facial definition. Clear glasses and ordinary hair generally remain recognizable;
that does not override changed visible appearance or preservation failures.
Native outputs remain soft, with visible tonal changes in the high-rate arms.
No native reference PSNR/SSIM, recovered identity or local-CCTV performance
claim follows.

All 24 native inputs retain their original input-only usable labels. Model
failure cannot relabel a usable input as insufficient. Clearer-crop requests
remain tied to the frozen input review, rather than which treatment looks best.
This implementing-assistant development inspection is distinct from the
independent final visual review required by the full goal.

{link(visual_path.relative_to(ROOT), 'Complete 64-sheet visual ledger')}.

## Demonstrated learning limitation

The saved-step analysis replays 3,360 signed products: each of twelve actual
parameter changes against twenty reference derivatives and fourteen signals.
All 120 pre-update regression-barrier values are exactly zero. Because every
trial starts with candidate output equal to its retained anchor, the ReLU
regression penalty contributes no preservation gradient to that first update.
Actual clear-image failures show that the soft barrier does not protect the
first finite change. This is a demonstrated limitation, not proof of a unique
cause of all earlier failures.

The current negative combined gradient also predicts increasing clear-image
pixel error. Consequently an optimizer-name change alone is not a supported
solution. First-order products do not predict later epochs, finite clipping,
Adam curvature or delivered-image preservation.

{link(analysis_root.relative_to(ROOT)/'independent_audit.json', 'Saved-step arithmetic audit')}.

## Additional diagnostic closure

Four fixed color/detail controls process 400 returned paired TRAIN outputs
without models, new gradients or training. Removing constant RGB differences
retains over 1% filter gain but still gives approximately 3.5–4 times the clear
pixel error and fails preservation. Retaining the baseline low-frequency band
reduces raw filter gain to 0.553–0.874%, with remaining preservation failures.
They do not establish useful corrected images, native performance or a new
restorer. The separate report keeps processing distinct from trained output.

The fixed active-anchor analysis tests the current objective and two added
direct RGB reconstruction terms using only saved derivatives. Clear/blur
fitting-mean slopes become favorable, but some reference-level slopes worsen.
No coefficient search, model assignment or local optimizer occurs. This
supports investigating active preservation supervision in a finite matched
test; it does not select qualified weights or demonstrate an optimal recipe.

## Next action and full-goal boundary

{link('CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md', 'The next training design')}
tests active clear/blur reconstruction supervision in a separate copy of the
current DGP's complete original decoder. Keep the failed rates, states,
source roles, exact baseline, full-TRAIN gates and raw/PNG evidence. More
epochs of the rejected objective are not justified by this diagnostic.
Additional epochs 1/2/5 remain conditional on a finite recipe passing
preservation, useful native development review and the original capacity gates.
No new executable VM packet or next training launch is claimed here.

The retained app checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`;
original Phase3 checkpoint, research caches, splits, provenance and failures
remain. Current app design/model selection is unchanged. All five milestones,
seven automatic/assisted covering families, independent final review and
bundled inline Playwright full-flow verification remain required. No automatic
historical pilot or failed-state resume is permitted. Direct VM access remains
maintenance-only; all actual training stays under manual transfer/tmux workflow.
The latest storage/workload values are historical, not fresh VM availability.
'''
    color_rows = []
    for control in color['controls']:
        raw, png = control['comparisons']['raw'], control['comparisons']['png']
        pool = control['arm'][-1]
        color_rows.append(f"| {pool} | {control['variant']} | {100*raw['relative_feature_gain']:.6f}% | {100*png['relative_feature_gain']:.6f}% | {raw['clear_MSE_multiplier']:.6f}× | {len(raw['preservation_failures'])}/{len(png['preservation_failures'])} | Fail |")
    color_report = f'''# Color/detail counterfactual V1: no processing remedy qualified

10 October 2026. Four predeclared arithmetic controls inspect the two returned
full-decoder arms with raw structure gain above 1%. No model, gradient,
parameter update or learned output selector is run locally. Both fitting pools
and all 100 exposed paired TRAIN cases are retained, giving 400 transformed
outputs. Aligned targets are used for scoring only; the transforms use the
retained baseline and failed candidate arrays. Native and final pixels are not
decoded. No app change or model qualification follows.

The constant-RGB control removes the observed-support mean of candidate minus
baseline. The low-band control subtracts a fixed sigma2, 13tap Gaussian-filtered
difference, adding the remaining difference to the baseline. This sigma is
inherited from the existing detail measurement; no parameter sweep or fitting
occurs. Values are clipped to [0,1], converted to float32, retain camera pixels
outside observed support, and use the original floor-to-PNG policy. These are
counterfactual processing diagnostics, not raw neural outputs or recommended
delivery processing.

| Pool | Fixed control | Raw feature gain | PNG feature gain | Clear raw MSE multiplier | Raw/PNG pixel preservation failures | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(color_rows)}

The independent V1 implementation reconstructs 400 transformations, 800 metric
records and all 400 exact saved PNGs, checking 209 source bindings. Transform
replay is exact; its independently implemented high-pass filter differs by at
most 8.67e-19. MSE/SSIM thresholds remain 1e-12/1e-6 and the necessary early
filter-gain requirement remains 1%. All four controls fail these necessary
pixel requirements. ArcFace is not recomputed and these new transformed images
have no visual-usefulness review; the original neural-return gallery is the
separate completed 64-sheet review. Therefore no full preservation or visual
qualification claim is possible.

A subsequent precision review corrects PNG-detail conversion to float64 from
saved uint8, matching the original frozen PNG metric. V1's float32 conversion
had compared slightly different filter arithmetic with that baseline. Retain
V1 and its receipt. R2 changes only the 400 PNG-detail values and their derived
group/gain summaries. Maximum changed per-case detail error is 3.86e-11;
independent OpenCV versus SciPy discrepancy is at most 1.31e-18. All raw metrics,
all other PNG metrics, failure locations and decisions remain exact. No image,
scientific threshold or model changes. The table uses R2.

The initial preparation's incorrect assumption of per-case raw NPY files also
remains recorded. Returned paired arrays actually use five-case lossless NPZ
packs; both revised decoders verify each recovered float32 array against its
original stored pixel hash. Candidate XOR storage changes no raw value.

These findings reject a display-only correction as the sufficient remedy for
these two failed arms. They do not establish that every possible training-time
color constraint is ineffective or identify a unique cause. The selected next
investigation is active reconstruction supervision before the first update,
with finite raw/PNG acceptance and source/profile protection still required.

{link('outputs/cctv_dgp_multiscale_color_detail_v1/plan.json', 'Original frozen arithmetic plan')},
{link('outputs/cctv_dgp_multiscale_color_detail_v1/independent_audit.json', 'Independent V1 transformation audit')},
{link('outputs/cctv_dgp_multiscale_color_detail_v1_r2/results.json', 'R2 metric correction and all decisions')}.
'''
    anchor_rows = []
    for contrast in anchor['contrasts']:
        mean = contrast['fitting_means']
        anchor_rows.append(f"| {contrast['name']} | {contrast['pool']} | {mean['MSE_clear']:.9f} | {mean['MSE_blur']:.9f} | {mean['HF_degraded']:.9f} | {contrast['positive_clear_MSE_reference_slopes']}/20 | {contrast['positive_HF_reference_slopes']}/20 |")
    anchor_plan = read(anchor_root/'plan.json')
    next_review = f'''# After multiscale calibration: active reconstruction before extra epochs

10 October 2026. Selected direction: investigate direct clear/blur RGB
preservation supervision that is active before the first optimizer update,
in an isolated copy of the current DGP's complete original reconstruction path.
The multiscale study is closed and all twelve treatments remain rejected.
This is a documented design and arithmetic review, not a prepared executable
VM packet, successful capacity pilot or qualified training recipe.

## Evidence determining the choice

The full original 15-tensor decoder now demonstrates over 1% paired TRAIN
filter gain, but its largest finite steps visibly alter appearance and increase
clear-image pixel error 24.8–31.5 times. The smaller repaired branch remains too
weak in this test. All 120 first-update hinge barriers are inactive at exact
initial parity. A current combined negative gradient also worsens clear RGB
loss initially, so changing the optimizer name alone is insufficient evidence.

Fixed color-only removal leaves 3.5–4 times clear pixel error. Fixed low-band
retention falls below 1% structure and still regresses source/profile pixel
metrics. Do not conceal these defects through display processing or promote
the strongest filter-gain result. More epochs of this rejected objective are
not a demonstrated solution.

V41 gradient surgery, V33 loss-cone proposals and the earlier finite-guard
directions already failed quality. They remain separate retained treatments.
The three-proposal finite guard accepted zero changes and missed 1% structure,
despite raw preservation; delivered PNG embedding regressions still mattered.
Do not repeat an unchanged cone/projection or infer a finite guarantee from
gradient compatibility. This next hypothesis changes supervised reconstruction
pressure, not only learning rate, tensor count or a display formula.

## Fixed saved-gradient contrast

The current positive normalized weights are MSE0.2, SSIM loss1, fixed observed
identity loss1 and degraded landmark HF loss1. Its MSE normalizer is the all-case
mean, approximately 0.02218; the clear baseline error is approximately 0.000674.
The new arithmetic contrasts add an always-present clear RGB reconstruction
term normalized by that exact clear TRAIN mean; a second contrast also adds
blur RGB normalized by its exact blur TRAIN mean. This gives the existing
clear/blur target errors direct learning pressure before regression occurs.
It is not a zero-gradient equality-to-current-output distillation term.

The two frozen additional scales are {anchor_plan['baseline_clear_RGB_MSE']:.17g}
and {anchor_plan['baseline_blur_RGB_MSE']:.17g}. The contrasts have fixed unit
coefficients, no coefficient search and the same four original objective terms.
They use only the saved genuine L4 derivatives, with directions normalized to
unit parameter L2. No neural forward, new gradient, optimizer, parameter
assignment or trained checkpoint is created locally.

| Fixed negative-gradient contrast | Pool | Fitting clear-MSE slope | Fitting blur-MSE slope | Fitting HF slope | Clear positive slopes | HF positive slopes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(anchor_rows)}

Negative slopes indicate predicted initial error decrease only. Although
fitting averages become favorable, 3–6 of 20 clear-reference slopes and 8–9
detail-reference slopes remain adverse under added anchors. These are
reference-level predictions, not the existing aggregate quality gates or
observed finite failures. Adam scaling, changing activations, clipping and PNG
rounding remain untested. The earlier favorable-average failure means this
arithmetic cannot select a successful recipe.

An independently implemented blockwise checker reconstructs all six directions
and replays 1,680 signed products. Maximum product discrepancy is 1.56e-14,
with every sign unchanged. This is arithmetic evidence for a targeted finite
test, not permission to weaken any appearance requirement.

{link(anchor_root.relative_to(ROOT)/'plan.json', 'Predeclared contrasts')},
{link(anchor_root.relative_to(ROOT)/'independent_audit.json', 'Independent contrast arithmetic')}.

## Next finite design, not executable yet

Compare the two materially changed objectives (active clear; active clear plus
blur) in the complete original RGB decoder at 1e-4 and 1e-3, in both original
balanced fitting pools: eight independent one-update arms, 400 fitting
exposures and zero complete epochs. Each starts anew from the retained current
checkpoint, with frozen encoder/normalization/recognizer and exact initial
parity. The old failed objective is a retained matched historical comparator;
it is not relaunched. Do not resume a failed arm.

Proposed limits inherit the verified machinery: 280 genuine derivative queries,
80 backwards, eight updates, 1800 seconds model work, 600 seconds export,
20GiB allocated VRAM, 7GiB free after installation, 1GiB disk reserve and
2.5GiB retained output cap. Independent scope/transfer/source/data tests and a
prospective returned-output checker must pass before issuing a manual command.
Storage, runtime projection and current workload must be freshly checked;
the preceding cleanup snapshot is historical. Model work remains manual in
tmux on the user's existing L4; direct connection remains maintenance-only.

Retain all 100 sampled TRAIN outputs at both raw and PNG stages, all 24 native
development crops separately, every original mask and every accepted/rejected
full state. Review every prospectively planned comparison sheet. The original
1%/10% structure, MSE/SSIM/embedding source/profile preservation, brightness
limit, visible facial appearance and native input-only usability remain.
An eight-arm diagnostic cannot promote the app or complete an epoch study.

If this materially changed first-step mechanics passes the original necessary
checks and visual review, freeze a separate finite full-TRAIN capacity pilot.
That pilot must evaluate all 3,905 cases/781 references and native development,
including the unchanged update50 stop. Only then consider additional epochs
1, 2 and 5 from a declared qualified initializer, maximum five in the first
study. Export full weights/optimizer/scheduler/RNG/schedule on every stop and
before any user-announced VM migration. Do not turn a weights-only fine tune
into a claim of exact optimizer resume. No failed gate is bypassed.

## Data and research limits

The completed full input/target inventory already verifies 3,905 TRAIN inputs,
781 canonical targets, 391 genuine HQ FFHQ counterparts and retained photographic
replay. The partial geometric overlap with native input eye spacing supports
later less-severe synthetic grid/framing coverage. It does not make native CCTV
paired ground truth or justify source-based clear routing: existing Auto flags
336/390 photographic replay clear cases and 0/391 HQ FFHQ clear cases.
Retain legacy replay, source provenance/terms, identity roles and overlap limits.
No new acquisition or source/ethnicity inference occurs in this review.

GEM motivates explicit loss-preservation constraints while allowing improvement;
its experiments concern continual classification, not CCTV faces. PCGrad addresses
conflicting task gradients, not this project's delivered finite-step guarantee.
These papers support distinguishing gradient compatibility from required
actual-image checks; they do not warrant rerunning rejected recipes.
[Original GEM paper](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf),
[Original PCGrad paper](https://arxiv.org/html/2001.06782v4).

NAFNet's released architecture uses multiscale paths and residual learning.
That informs later input-conditioned spatial design if the active-supervision
test fails. Our original DGP already has multiple FPN scales and an input skip;
another generic skip or a larger parameter count is not a demonstrated remedy.
No external restoration weights are substituted for our DGP.
[Official NAFNet architecture](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py).

The five milestones, native usefulness, separate final identities, independent
final reviewer, DGP-primary Auto/override flow, seven automatic/assisted completion
families and inline bundled Playwright verification remain required. Clear glasses,
ordinary hair and visible appearance remain protected. Insufficient-input requests
follow frozen input criteria. No hidden-identity or Zamboanga performance claim.
Current checkpoint selection and app design remain unchanged; the full goal is
active and incomplete.
'''
    documents = {
        'CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md': report,
        'CCTV_DGP_MULTISCALE_COLOR_DETAIL_V1_RESULTS.md': color_report,
        'CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md': next_review,
    }
    for name, text in documents.items():
        with (ROOT/name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    prefix = '''**Latest verified status — 10 October 2026: multiscale return audited; all64 visual sheets reviewed; all12 treatments rejected. Full goal incomplete.**

The manual L4 diagnostic completed12 independent one-update arms,600 fitting
exposures and zero complete epochs. Full original-decoder LR0.001 gives
1.170–1.884% raw structure gain, but changes visible appearance and increases
clear pixel error24.8–31.5 times. Lower rates/deep-only learning do not supply
enough useful detail. Every treatment fails unchanged quality requirements;
no stopped state is resumed or promoted. Current checkpoint/app design remain.

Independent R2 integrity/arithmetic checks preserve the original checker failure
and every scientific gate decision. All64 planned sheets/2280 exact saved PNG
cells were inspected; native24 ChokePoint DEV cases remain separately unpaired
and retain their input-only usable labels. This implementing-assistant review
is not the goal's independent final review. Final identity pixels remain outside
tuning. Report: CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md.

400 fixed color/detail controls also fail necessary pixel checks. Six fixed
saved-gradient contrasts support examining active clear/blur RGB reconstruction
supervision before the first update, while retaining adverse reference slopes.
These are arithmetic diagnostics, not trained or qualified models. Selected
next design: CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md. No new executable VM
packet or additional-epoch run is prepared here; the epoch1/2/5 study stays
conditional on a passing finite recipe and useful native development review.

All five milestones and seven covering families remain required. Current model,
original checkpoints, caches, splits, provenance and failed gates are preserved.
Training remains manual through verified transfers/tmux on the existing L4;
direct VM access remains maintenance-only. Current guest storage/workload are
not verified by this local return audit. Historical ready/pending statuses and
closed commands below are superseded, not authorization to rerun them.

---

'''
    mutable = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
               'CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md']
    backups = {}
    for name in mutable:
        assert name not in protected and not (ROOT/name).is_symlink()
        original = (ROOT/name).read_bytes()
        backup = OUT/(name+'.original')
        with backup.open('xb') as stream:
            stream.write(original)
        new_prefix = prefix if name != mutable[-1] else '''**CLOSED — 10 October 2026: returned and independently audited; all12 calibration arms rejected. Do not rerun or resume these historical commands.**

The trainer/export completed, but no arm passes the unchanged restoration
requirements. All64 planned visual sheets are reviewed. Read
CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md and
CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md for the returned result and selected
next design. No next executable packet is ready in this guide. The prior ready
status, disk/workload values and transfer/launch commands below are historical.
Current checkpoint and original research/failure evidence remain preserved.

---

'''
        (ROOT/name).write_bytes(new_prefix.encode('utf-8')+original)
        backups[name] = dict(backup=backup.relative_to(ROOT).as_posix(),
                            original_sha256=sha(backup), updated_sha256=sha(ROOT/name),
                            prefix_bytes=len(new_prefix.encode('utf-8')),
                            original_suffix_bytes_preserved=True)
        assert (ROOT/name).read_bytes()[backups[name]['prefix_bytes']:] == original
    for name,digest in protected.items():
        assert sha(ROOT/name) == digest,name
    for name,digest in bindings.items():
        assert sha(ROOT/name) == digest,name
    write(OUT/'publication.json',dict(complete=True, publication_plan_sha256=sha(OUT/'publication_plan.json'),
        documents={name:sha(ROOT/name) for name in documents}, backups=backups,
        protected_bindings_verified=346, original_checker_unchanged=True,
        returned_checkpoint_app_selection_unchanged=True, visual_sheets_completed=64,
        all12_treatments_rejected=True, color_controls_rejected=4,
        next_recipe_is_design_not_executable_packet=True, new_VM_connection_or_launch=False,
        local_optimizer_updates=0, local_gradient_queries=0,
        goal_status='active', model_qualification=False, goal_complete=False,
        UTC=datetime.now(timezone.utc).isoformat()))
    print(dict(complete=True, reports=3, canonical_updates=3, historical_guide_closed=True,
               protected_bindings=346, goal_complete=False),flush=True)


if __name__=='__main__':
    main()
