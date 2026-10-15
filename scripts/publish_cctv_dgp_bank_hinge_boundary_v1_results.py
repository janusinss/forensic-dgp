"""Publish audited hinge evidence and independently checked finite forward R1."""
from datetime import datetime, timezone
from pathlib import Path
from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write
from publish_cctv_dgp_bank_AB_and_guard_v1_review import prepend_locked


def link(label, name):
    return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(value)


def main():
    receipt=ANALYSIS/'publication.json'
    assert not receipt.exists()
    direction_root=OUT/'cctv_dgp_bank_saved_direction_v1_analysis'
    evidence=[ANALYSIS/'import.json',ANALYSIS/'independent_audit.json',ANALYSIS/'loss_direction_review.json',
        OUT/'cctv_dgp_bank_hinge_boundary_v1_filename_recovery.json',
        direction_root/'direction_review.json',direction_root/'independent_audit_arithmetic_order.json',
        direction_root/'png_policy_repair.json',
        OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_preparation.json',
        OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_transfer_audit.json',
        OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_bash_syntax.json',
        OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_return_checker_preparation.json']
    assert all(read(path)['complete'] for path in evidence)
    imported,audit,review,recovery,direction,arithmetic,png,prep,transfer,bash,checker=[read(path) for path in evidence]
    assert audit['diagnostic_complete'] and audit['inactive_initial_guard_zero_gradient_all20_independently_checked']
    assert recovery['source_sha256']==recovery['target_sha256']==imported['archive_sha256']
    assert checker['protocol_sha256']==transfer['protocol_sha256']==prep['protocol_sha256']
    packet=OUT/prep['name']
    assert not (packet/'outputs').exists()
    assert sha(packet/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py')==png['corrected_R1_worker_sha256']
    assert bash['checked_shell_sha256']==sha(packet/'scripts/run_forward.sh') and bash['exit_code']==0
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    states=review['group_summaries']['all']
    scales=review['retained_scalar_loss_scales']
    report_name='CCTV_DGP_BANK_HINGE_BOUNDARY_V1_RESULTS.md'
    guide_name='CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_VM.md'
    table=[]
    for state,label in [('initial','Pinned initial'),('joint50','Rejected B50')]:
        r=states[state]
        table.append(f"| {label} | {r['nonzero_guard_gradient_rows']}/20 | {r['nominal_single_reference_Adam_detail_ascents']}/20 | {r['nominal_single_reference_Adam_clear_MSE_ascents']}/20 | {r['nominal_single_reference_Adam_degraded_MSE_ascents']}/20 |")
    report=f"""# Completed brightness-hinge diagnostic — 11 October 2026

**The inactive-brightness-gradient defect is corrected on the manual L4 run.**
Every initial guard gradient is exactly zero on all 20 sampled TRAIN references,
with exact scalar penalty values retained. This is a verified loss correction;
the run has zero optimizer updates and creates no improved checkpoint.

The run took **{audit['VM_seconds']:.2f} seconds**, with 80 B forwards,
40 fixed-recognizer forwards, 360 gradient queries, zero backwards calls and
zero optimizer updates/epochs. Peak allocated VRAM was {audit['VM_peak_VRAM']/1024**3:.3f} GiB;
export took 20.69 seconds. The {imported['archive_bytes']:,}-byte archive SHA256 is
`{imported['archive_sha256']}`. The file was downloaded as `outputs/.5`; an
exclusive-read, hash-verified copy recovered its expected name. Both originals
are retained; nothing was deleted. The independent importer checked 135 return
files and 5,852 required parent assets (1,282,410,271 bytes).

All 40 frozen control blocks/200 case outputs are bit-exact to retained initial
and B50 outputs. All 40 scalar rows were independently recomputed. Four full
saved gradient samples, 370 arrays each, passed NumPy Gram/clipping/Adam checks.
These checks verify saved derivative arithmetic; they do not independently
re-run all 40 neural derivatives. Old/new scalar components are exact. Pure
reconstruction/anchor derivatives are exact in the four saved cross-run samples.
Small ArcFace/active-barrier gradient variation is retained, rather than called
bit-exact. Maximum sampled ArcFace difference is 2.384185791015625e-07;
the active barrier sample differs by at most 3.5762786865234375e-07.
The identity path uses CUDA `grid_sample`, whose backward can be nondeterministic;
this is a source-supported explanation, not a freshly isolated replay of that
variation. [PyTorch 2.9 grid_sample](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.grid_sample.html),
[reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html).

| State | Nonzero guard gradients | Nominal Adam detail ascents | Clear-MSE ascents | Degraded-MSE ascents |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

These are **20 photographic TRAIN references per state**, five synthetic profiles
each. Directions are hypothetical single-reference Adam directions, not the
historical accumulated five-reference update or measured finite image changes.
B50 is a rejected state used for diagnosis only; it is not resumed.

The boundary correction reduces initial degraded-MSE ascents from 20/20 to
15/20. It does not resolve the whole objective. Unconditional clear/blur absolute
anchor coefficients are {scales['clear_absolute_anchor_coefficient']:.2f} and
{scales['blur_absolute_anchor_coefficient']:.2f}, compared with a per-profile
weighted reconstruction MSE coefficient of {scales['weighted_MSE_per_profile_coefficient']:.4f}.
That is **{scales['clear_anchor_to_weighted_MSE_per_profile_ratio']:.2f}×** and
**{scales['blur_anchor_to_weighted_MSE_per_profile_ratio']:.2f}×** per-profile weighting,
respectively; coefficient ratios alone are not gradient-norm ratios.

The combined anchor negative gradient opposes average degraded MSE on 16/20
initial references and 20/20 B50 references. The blur anchor alone opposes that
average on all 20 initial references. This exposes a tradeoff between one blur
profile and the four degraded profiles, not a claim that blur restoration is
unimportant. Removing the absolute anchors analytically reduces initial plain
negative-gradient degraded-MSE ascents to 3/20, but predicts clear-MSE ascent on
10/20. The two saved initial Adam examples also do not establish a safe recipe:
omitting all anchors worsens the clear-MSE direction on both, and worsens degraded
MSE on the first. All visible facial features and preservation gates stay required.

**Next decision: test an explicitly constrained parameter direction before
training more epochs.** Two saved initial reference batches from
`dataset/thumbnails128x128` and `dataset/asian_faces` supply derivatives. These
are source labels, not ethnicity or native Asian CCTV capture claims. Their
ordinary direction averages reconstruction, regression barrier and corrected
brightness losses without the conflicting absolute anchors. A convex projection
after Adam preconditioning imposes detail and preservation directions on both
sampled references. The ten first-order targets use a 1% predicted detail reduction
and 0.01% reductions in clear/degraded MSE, SSIM loss and ArcFace loss at nominal
LR 1e-4. These are proposed local directional targets, not changed scientific
output gates or observed improvements.

The saved two-reference constrained direction passes independent Gram, target,
primal/dual and complementarity arithmetic checks. Its norm is
{direction['direction_norm_ratio']:.9f}× the ordinary direction; no local model
parameters were displaced. It is not the old five-reference accumulated step.
Applicability to all profiles, native faces and finite outputs remains unproven.

The **{prep['archive_bytes']:,}-byte R1 packet** compares ordinary/constrained
directions at two prospectively fixed rates, 1e-4 and 1e-5, from the exact initial
state. It plans four temporary model displacements, 100 B forwards, 160 fixed
recognizer forwards, zero new derivatives/backwards/optimizer updates and no
checkpoint. Each proposal is reset before the next; the initial state is restored
afterward. It saves all 400 finite raw case outputs and exact floor-policy PNGs,
with source/profile gates and 20 comparison sheets for independent visual review.
Successful arithmetic or export does not qualify any proposal or more epochs.

Prepublication checking caught the draft's rounding PNG policy. R1 uses the exact
retained `floor(RGB*255)` policy with input preservation outside observed pixels;
its actual export expression matches all 100 retained baseline PNGs. The draft
changed {png['old_rounding_policy_different_channels']:,} channels across all 100 cases;
its original sealed packet is preserved, unrun and superseded. The future metric
checker uses float64 conversion from PNG bytes for the retained high-frequency
metric and passed all 200 saved raw/PNG control rows. Six synthetic boundary
fixtures independently match the retained 1%/20%/MSE/SSIM/ArcFace logic; they are
checker fixtures, not model results. A float32 sum-order audit retains 49
subnormal direction differences (maximum 1.4013178358747879e-36) from moving
division before summation; exact original arithmetic is used, without loosening
raw parity or scientific gates.

The R1 archive/source/checksum/Bash/Python-3.10 AST checks passed. Its Windows
run rejects before neural imports. Require **4 GiB free after installation**;
worker/measurement/export caps are 1200/900/300 seconds, external 1230/330 with
30-second kill grace. VRAM cap 20 GiB, output 768 MiB, reserve 1 GiB; conservative
payload forecast {prep['forecast_payload_bytes']:,} bytes. The new runtime is
unmeasured; estimate 2–6 minutes measurement plus 1–3 minutes export. No agent
VM connection, compute, training, power or cleanup action occurred.

{link('Five exact R1 upload, tmux and download steps',guide_name)}.
{link('Independent returned hinge audit','outputs/cctv_dgp_bank_hinge_boundary_v1_analysis/independent_audit.json')}.
{link('Audited loss conflict review','outputs/cctv_dgp_bank_hinge_boundary_v1_analysis/loss_direction_review.json')}.

Goal **ACTIVE**. The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
This fine-tuned DeblurGAN-v2-compatible restorer is the original comparator. B
uses our conditioning/reconstruction with a declared external frozen StyleGAN2
prior, not a newly own-trained StyleGAN2 or exact published-DGP reproduction.
The matched corrected-current-versus-replacement comparison remains unresolved.
No new incumbent, extra epoch qualification or app promotion is claimed.

Native CCTV remains unpaired; paired photographic TRAIN PSNR/SSIM and labeled
final identities remain separate. No real Zamboanga CCTV samples, ethnicity,
local performance or exact hidden identity are claimed. The input-only usable
criteria, all visible features, native/final independent reviews, design-preserving
Auto/override app flow and bundled inline Playwright verification remain required.
All seven covering families remain: masks, sunglasses, strong glare, hands,
obstructing hair, scarves and objects; automatic area preview/correction,
visible appearance/clear glasses/non-obstructing hair preservation, original/mask/
one plausible estimate, PNG and optional bundle downloads, and separate automatic
versus assisted evidence. All five milestones remain the completion criterion.
"""
    stem=prep['stem']
    guide=f"""# Saved-direction finite forward R1: five exact manual steps

**Use R1 only.** The earlier V1 draft stays archived and unrun because its PNG
rounding policy differed from the retained baseline. This R1 packet passed
independent transfer and return-checker preparation checks. Its actual VM
forward proposals are **unrun**. {link('Audited hinge results and reason for this test',report_name)}.

Upload **{prep['archive_bytes']:,} bytes** (about 19.7 MiB). Require **4 GiB free
after installation** and idle L4. Estimate **2–6 minutes measurement plus 1–3
minutes export**, unmeasured for this packet. Worker/measurement/export stops
1200/900/300 seconds; external 1230/330 with 30-second kill grace. VRAM cap 20 GiB,
output 768 MiB, reserve 1 GiB. Preserve the parent and historical evidence.

This compares four temporary parameter proposals on 100 paired TRAIN cases,
with zero new gradient queries or optimizer steps. It restores the initial state
between proposals and at completion, saves no checkpoint and launches no training.
Raw/PNG metrics and all 20 visual sheets must be reviewed afterward.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\\r' < {stem}-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{prep['name']} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

Archive SHA256: `{prep['archive_sha256']}`.
Protocol SHA256: `{prep['protocol_sha256']}`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_saved_direction_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {prep['protocol_sha256']} --verify-transfer &&
bash scripts/run_forward.sh {prep['protocol_sha256']}
```

Expected transfer: `complete: true`, `packet_assets: 4`, `parent_assets: 5852`,
with zero neural/gradient/optimizer calls in that transfer check. Successful
finite execution has 100 B and 160 recognizer forwards, four temporary proposals,
zero gradient queries/backwards/optimizer updates and no checkpoint. Export
`complete` confirms packaging; passing restoration is assessed after return.
Detach with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_saved_direction_v1_r1`. Stops are retained/exported.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "{stem}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "{stem}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "{stem}-export.json"
```

The explicit local filenames prevent another accidental `.5` destination. Keep
all three files together. Do not rerun a used root or resume a failed historical
state. Independent audit and visual review precede any new training decision.
The full native/final/app and seven-family completion goal remains active.
"""
    new(ROOT/report_name,report)
    new(ROOT/guide_name,guide)
    prefix=f"""# Latest verified status — 11 October 2026: brightness-hinge fix audited; finite direction forward R1 prepared

The human manual hinge diagnostic completed in {audit['VM_seconds']:.2f} seconds:
80 B/40 recognizer forwards, 360 queries, zero optimizer updates/backwards/epochs
and no checkpoint. All 135 return files/5,852 parent assets, 40 bit-exact raw
controls/200 case replays and sampled gradient arithmetic passed independent checks.
All 20 initial guard gradients are now zero; scalar penalties are preserved.
ArcFace/active-barrier cross-run derivative variation is retained separately;
derivatives were not independently regenerated for all40. Downloaded `outputs/.5`
and a hash-verified expected-name copy both remain; no deletion occurred.

The loss conflict persists: unconditional clear/blur anchor coefficients are
{scales['clear_anchor_to_weighted_MSE_per_profile_ratio']:.2f}×/{scales['blur_anchor_to_weighted_MSE_per_profile_ratio']:.2f}× per-profile weighted reconstruction MSE.
Their negative gradient opposes degraded MSE on 16/20 initial references;
corrected nominal single-reference Adam still predicts degraded ascent on15/20.
Removing anchors alone worsens clear-MSE directions in both saved examples.
These are photographic TRAIN/first-order diagnostics, not applied updates,
native quality or additional-epoch justification. Failed B50 is diagnosis only.

A saved two-reference post-Adam constrained direction passed independent
Gram/target/primal/dual arithmetic, with no local model displacement. The next
R1 compares four prospectively fixed temporary parameter proposals on100 cases,
ordinary/constrained at1e-4/1e-5, with 100 B/160 recognizer forwards and zero new
gradients/backwards/optimizer updates. Initial state resets between proposals;
no checkpoint is saved. Its actual VM behavior and image usefulness are unrun.

The {prep['archive_bytes']:,}-byte R1 archive/source/LF/Python/Bash/Windows checks
passed. Its floor PNG policy matches100 retained PNGs exactly. Earlier rounding
V1 is preserved/unrun/superseded. Independent return metrics/gates were checked
on200 saved raw/PNG rows and six declared synthetic checker fixtures. Return
auditing must assess all400 raw outputs/PNGs and all20 visual sheets before
further training. No scientific thresholds or input-only criteria were changed.

Require4GiB free after installation, idle L4, manual tmux. Worker/measurement/
export caps1200/900/300 seconds; external1230/330 with30-second kill grace;
VRAM20GiB/output768MiB/reserve1GiB. No agent VM connection/compute occurred.
{link('Audited hinge result and finite decision',report_name)}.
{link('Five exact R1 upload/tmux/download commands',guide_name)}.

Goal ACTIVE. Permanent baseline checkpoint
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`,
existing app design and all visible facial features remain preserved. The
matched corrected-current/replacement comparison remains unresolved. No new
incumbent, more-epoch qualification, app promotion or failed-state resume.
Native CCTV remains unpaired; paired photographic TRAIN and reserved final
identities remain separate. No ethnicity, Zamboanga performance or hidden-identity
claim. All five milestones, seven automatic/assisted covering families,
native/final independent reviews and bundled inline Playwright app flow remain
required. Original historical text is retained exactly below.

---

"""
    backups=ANALYSIS/'publication_before'
    backups.mkdir()
    documents={}
    for name in ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        documents[name]=prepend_locked(ROOT/name,prefix,backups/name)
    old_guide='CCTV_DGP_BANK_HINGE_BOUNDARY_V1_VM.md'
    warning='# Hinge diagnostic completed and audited — historical commands\n\nThis immutable diagnostic is complete; do not rerun it unchanged.\n'+link('Audited hinge results',report_name)+'.\n'+link('Use the prepared finite-forward R1 guide',guide_name)+'.\n\n---\n\n'
    documents[old_guide]=prepend_locked(ROOT/old_guide,warning,backups/old_guide)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence},documents=documents,
        new_documents_sha256={name:sha(ROOT/name) for name in [report_name,guide_name]},
        actual_hinge_derivative_correction_audited=True,remaining_anchor_conflicts_retained=True,
        finite_forward_R1_manual_unrun=True,original_unrun_rounding_V1_preserved=True,
        strict_future_metrics_and_visual_checker_prepared=True,protected_bindings_unchanged=True,
        no_local_model_gradient_optimizer_or_parameter_displacement=True,no_agent_VM_launch=True,
        all_five_milestones_seven_families_and_manual_workflow_preserved=True,goal_status='active',goal_complete=False))
    print(dict(complete=True,status_documents=5,new_report_and_guide=2,next_manual_R1_forward_pending=True))


if __name__=='__main__':
    main()
