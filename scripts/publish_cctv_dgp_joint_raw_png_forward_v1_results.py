"""Publish audited rejection and one changed manual diagnostic; preserve history."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage=ANALYSIS/'publication';assert not (stage/'publication.json').exists()
    names=['independent_audit.json','finite_metrics.json','finite_learning_review.json',
           'spatial_patch_review.json','quantization_review.json','visual_review.json','review_binding_audit.json']
    proofs=[ANALYSIS/n for n in names]+[OUT/('cctv_dgp_live_conditioning_v1_'+suffix+'.json')
        for suffix in ['preparation','transfer_audit','gate_fixtures','bash_syntax']]
    assert all(read(path)['complete'] for path in proofs)
    bound=read(ANALYSIS/'review_binding_audit.json');audit=read(ANALYSIS/'independent_audit.json')
    imported=read(ANALYSIS/'import.json');finite=read(ANALYSIS/'finite_metrics.json')
    assert bound['all_four_proposals_rejected_for_adoption'] and bound['all20_visual_sheets_and800_source_cells_bound']
    assert read(ANALYSIS/'visual_review.json')['all20_sheets_and800_cells_inspected']
    packet=OUT/'cctv_dgp_live_conditioning_v1_vm';stem='cctv-dgp-live-conditioning-v1'
    p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    prep=read(OUT/'cctv_dgp_live_conditioning_v1_preparation.json')
    transfer=read(OUT/'cctv_dgp_live_conditioning_v1_transfer_audit.json')
    fixture=read(OUT/'cctv_dgp_live_conditioning_v1_gate_fixtures.json')
    shell=read(OUT/'cctv_dgp_live_conditioning_v1_bash_syntax.json')
    assert all(record['protocol_sha256']==pin for record in [prep,transfer,fixture,shell])
    assert sha(packet/'scripts/run_conditioning.sh')==shell['checked_shell_sha256'] and shell['exit_code']==0
    assert sha(ROOT/'scripts/audit_cctv_dgp_live_conditioning_v1_return.py')==fixture['future_auditor_sha256']==prep['future_auditor_sha256']
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    archive=OUT/(stem+'-execution.tar.gz');assert sha(archive)==prep['archive_sha256']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    previous=read(OUT/'cctv_dgp_png_group_gradients_v1_r2_analysis/publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
          'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    used='CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_VM.md'
    expected={name:previous['documents'][name]['after_sha256'] for name in docs}
    expected[used]=previous['new_documents_sha256'][used]
    stage.mkdir(exist_ok=True);backup=stage/'before';backup.mkdir(exist_ok=True)
    assert all(sha(ROOT/n)==h or ((backup/n).exists() and sha(backup/n)==h) for n,h in expected.items())
    result='CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_RESULTS.md'
    guide='CCTV_DGP_LIVE_CONDITIONING_V1_VM.md'
    table='\n'.join('| '+row['proposal'].removeprefix('balanced_')+' | '+
        f"{row['raw_structure_gain_percent']:.6f}% | {row['PNG_structure_gain_percent']:.6f}% | "+
        str(row['raw_preservation_failure_count'])+' | '+str(row['PNG_preservation_failure_count'])+' | Rejected |'
        for row in bound['summary'])
    report=f'''# Joint raw/PNG comparison and spatial-path decision — 13 October 2026

**The VM diagnostic completed correctly; every proposal remains unqualified.**
The largest step improves the pooled paired TRAIN detail proxy by **0.865441%
raw / 0.886155% PNG**, below the retained 1% requirement, with **37 raw and
37 PNG overlapping group/metric preservation failures**. Counts are not failed
people. Neither the permanent app checkpoint nor the output conversion changed.

| Positive step | Raw detail gain | PNG detail gain | Raw failures | PNG failures | Decision |
|---:|---:|---:|---:|---:|---|
{table}

Both sources also remain below 1% independently. At .02, source label
`dataset/asian_faces` has **0.559452% raw / 0.557613% PNG** gain;
`dataset/thumbnails128x128` has **0.942176% raw / 0.968584% PNG**.
The .0005 step has zero raw preservation failures but two PNG failures: clear
thumbnail-source RGB MSE and blur-source observed similarity. Its detail gain is
only 0.049125% raw / 0.061905% PNG. A small raw-safe step does not qualify a model.

The L4 took **{audit['VM_seconds']:.2f} seconds**:100 B and160 fixed recognizer
forwards, four temporary parameter proposals, and **zero optimizer updates,
gradients, epochs or checkpoints**. Each proposal reset to the initial B.
All {audit['files_checked']} return members,5,852 parent bindings,100 raw
control replays,400 raw outputs and400 exact floor-converted PNGs pass the
independent saved-evidence audit. All four parameter/state hashes were rebuilt
from saved arrays. No full neural replay or fresh post-run RAM inspection is
claimed. Archive size is **{imported['archive_bytes']:,} bytes**; SHA256:
`{imported['archive_sha256']}`.
{link('Return audit','outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/independent_audit.json')}.

## Why the favorable predictions did not qualify the result

All64 saved raw/PNG directional constraints predict initial descent, but the
finite model response and discontinuous byte conversion do not follow that
linear approximation sufficiently. For example, PNG clear thumbnail-source
MSE at .0005 was predicted to decrease by4.57310e-8 using the actual stored
parameter step, but increased by2.71190e-8. At .02 the same error increases
6.1321%. The separate saved-array review checks256 actual-step derivative dots;
it does not compute new neural derivatives or prove a unique nonlinear cause.
{link('Finite learning evidence','outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/finite_learning_review.json')}.

The new review's initial exact baseline-loss guard stopped because differentiable
float64-moment SSIM and scientific skimage float32 SSIM differ by up to5.04591e-7;
recognizer replay reductions differ by up to1.52402e-8. That preparation stop and
its source are retained. The completed review records these offsets and uses
the existing derivative-alignment bound only for those representations. Exact
arrays, PNGs and every scientific acceptance check remain unchanged. These
derivatives remain approximate; an arithmetic check is not output qualification.

## All visible features still matter

All20 sheets were displayed at original2048x1460 resolution; all800 source
cells are bound to image hashes. Small steps have no convincing whole-face
clarity advantage. The largest step adds extra curved nose and lip outlines
in several degraded cases, while eyes, mouths, teeth and ordinary hair remain
soft. Clear glasses and non-obstructing hair remain present. The assistant's
development review is not independent final human review.
{link('Every visual observation','outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/visual_review.json')}.

An additional4,000-row diagnostic separates the five existing landmark patches.
At .02, pooled PNG image-left eye error decreases1.716538%, but nose error
increases0.012112% and image-right mouth error increases0.112131%. Nose error
increases in47 of80 degraded TRAIN cases. For the Asian source label, pooled
PNG nose error increases1.581780% and image-right eye error increases0.504818%.
These are overlapping photographic patches, not all visible features, independent
subjects, or new acceptance thresholds. They reinforce the visible result:
a pooled score near1% is not adequate whole-face restoration.
{link('Separate patch evidence','outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/spatial_patch_review.json')}.

## Best next approach: verify a live spatial-conditioning path

The raw-only, signed small-step and joint raw/PNG comparisons did not meet the
combined requirements. AGENTS.md's three-attempt question was asked; the user
answered **apply best approach**. The invalid assumption was that a jointly
descending initial direction would yield enough preserved detail at a finite
step. Another unchanged fixed-direction or larger-step trial is not selected.

The current reset tests leave **all12 style/spatial tensors**, representing
**2,580,864 values**, unchanged. All64 saved derivatives are zero in those
partitions. Zero bank readouts initially block their output gradients. This is
a limit of these one-step reset experiments; the earlier audited B50 training
changed all37 tensors and still failed. It is not evidence that conditioning can
never learn or the sole cause of every earlier failure.

The selected separate prototype uses nonzero, declared deterministic channel
readouts and subtracts a **frozen same-input initial feature path**. Its intended
initial difference is zero, so the original output stays the control while
derivatives can reach live style/spatial conditioning immediately. The reference
uses the observed image, not a clean target or recovered identity. Original
observation/decoder weights and the declared external frozen StyleGAN2 bank are
reused. Its new readouts are untrained initializers; no new trained model exists.
This is a feature-path architecture hypothesis, not sharpening or display mixing.

GLEAN's experiments motivate using multiscale spatial encoder conditioning and
decoder fusion; they do not validate our anchored construction. It also reports
non-aligned examples, so copying a recognizer affine into the generative path
is not justified without a separate geometric test.
[Primary GLEAN paper, sections3 and5](https://arxiv.org/html/2012.00739).
FFHQ's official documentation describes aligned/cropped source images; that
does not prove misalignment is the unique cause here.
[Official FFHQ source](https://github.com/NVlabs/ffhq-dataset).

The new **verified, UNRUN manual diagnostic** uses the same100 paired TRAIN
cases,32 raw loss groups and320 gradient queries. It requires exact starting
RGB parity and finite nonzero gradients in all12 conditioning tensors for both
sources' detail and clear-MSE groups. Its frozen/live feature branches each run
40 times; no optimizer, displacement, committed update, epoch or checkpoint is
produced. Original failed recipes and full states remain. Gate fixtures reject
the actual old closed-branch gradients and zero/negative/nonfinite scalar cases.
CPU fixtures prove only matrix/state arithmetic; CUDA parity and live neural
gradients require the manual VM run. {link('Exact next commands',guide)}.

Only after that return is independently audited can a separately frozen finite
training design be prepared. It must refresh learning at changed weights and
evaluate actual raw/PNG outputs, source/profile preservation, brightness,1%
global/each-source structure and all visible features. Passing activation alone
does not authorize additional epochs or application promotion.

All data here remain exposed paired photographic TRAIN evidence. The existing
HQ FFHQ counterpart targets remain audited; a thumbnail source label alone does
not justify new acquisition. No native CCTV or reserved final pixels were used,
and no ethnicity, Zamboanga performance or hidden identity is claimed. The
original checkpoint remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
All five milestones and seven separate automatic/assisted completion families
remain required. The full goal stays active and incomplete.
'''
    retained_new(ROOT/result,report)
    guide_text=f'''# Live conditioning diagnostic — manual L4 commands

**Verified and UNRUN. This is an architecture/gradient diagnostic, not training.**
It tests a changed feature path after the rejected joint comparison.
{link('Audited results and design rationale',result)}.

Same100 paired TRAIN cases;32 loss groups;320 gradient queries. Zero optimizer
updates, displacements, committed updates, epochs or new checkpoints. Starting
outputs must match the original DGP exactly. All12 style/spatial tensors must
receive finite nonzero detail and clear-preservation gradients for each source.
Activation alone is not useful restoration, PNG preservation or epoch approval.

Require **4 GiB free after installation**. Estimate **3–6 minutes** plus export.
Measurement/worker600/900s; outer930s. Export300s/outer330s plus30s kill grace.
VRAM20GiB/RSS12GiB/output1280MiB/reserve1GiB. Existing workloads and old outputs
are preserved. Upload **{prep['archive_bytes']:,} bytes**. No current free-space
or power-state claim is made; the VM checks them at execution.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c {stem}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{packet.name} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_live_conditioning_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{packet.name} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_live_conditioning_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_conditioning.sh {pin}
```

Progress visits20 reference batches and ends with320 queries. Export runs after
completion or a retained stop. Its `complete: true` certifies the archive, not
the diagnostic or model. Detach **Ctrl+B,D**; reattach with
`tmux attach -t dgp_live_conditioning_v1`. No automatic training follows.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "{stem}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "{stem}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "{stem}-export.json"
```

Each PuTTY download has one remote source. Return all three files for an
independent audit of baseline parity, states, gradients, controls and resource
stops. Do not rerun this used diagnostic, delete failure records or bypass a
stop. The named L4/g2-standard-4 is required; a new VM needs migration/identity
verification. Agent access remains storage maintenance only.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
{link('Transfer/state evidence','outputs/cctv_dgp_live_conditioning_v1_transfer_audit.json')};
{link('Actual old-gradient/adversarial gate fixtures','outputs/cctv_dgp_live_conditioning_v1_gate_fixtures.json')}.
'''
    retained_new(ROOT/guide,guide_text)
    design_path=ANALYSIS/'next_design_review.json'
    if not design_path.exists():
        write(design_path,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
            human_diagnostic_answer='apply best approach',consecutive_finite_comparisons_rejected=3,
            invalid_assumption='Joint initial descent predictions guarantee an adequate preserved finite output',
            selected='Separate anchored same-input multiscale conditioning path; test parity and live derivatives before training',
            basis=bound['evidence_sha256'],prototype_protocol_sha256=pin,
            earlier_B50_changed_all37_tensors_and_failed=True,unique_cause_not_claimed=True,
            diagnostic_transfer_verified_and_unrun=True,training_launch_ready=False,optimizer_updates=0,
            all_quality_requirements_and_app_checkpoint_preserved=True,automatic_launch=False,
            model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    prefix=f'''# Latest model evidence — 13 October 2026; joint comparison rejected

Manual joint raw/PNG comparison completed in57.52s. Best detail gain is
0.865441% raw /0.886155% PNG, below1%, with37 raw/37 PNG overlapping
group-metric preservation failures. Both sources remain below1%. All596 return
members,5,852 parent bindings,100 initial controls,400 raw/400 exact PNG outputs
and all20 visual sheets/800 cells pass saved-evidence/review binding checks.
This is a diagnostic: zero optimizer updates, gradients, epochs or checkpoints.
{link('Complete findings',result)}.

Separate4,000-patch checks show nose error rises in47/80 degraded PNG cases;
the largest step adds central contours without convincing whole-face clarity.
All12 style/spatial tensors stay unchanged in these reset one-step tests.
Earlier B50 did change all37 tensors and failed; no unique universal cause is
claimed. Scientific raw/PNG/source preservation and1% requirements remain exact.

After three unsuccessful finite comparisons, the required diagnostic question
was asked; the user answered apply best approach. New separate anchored
same-input feature-path prototype is verified and UNRUN. Nonzero readouts plus
a frozen initial reference aim to retain starting output and activate spatial/
style derivatives. Same100 TRAIN cases/32 groups/320 queries; zero updates,
displacements, epochs or checkpoints. CUDA parity/activation are unproven.
4GiB free; projection600s/worker900s/outer930s/export300s/outer330s;
20GiB VRAM/12GiB RSS/1280MiB output/1GiB reserve. No automatic follow-on.
{link('Exact manual upload, tmux and download commands',guide)}.

Original app/model checkpoints, full states, splits, caches, failures, research
terms, migration backups and design remain. All five milestones and seven
automatic/assisted covering families remain; photographic paired TRAIN and
unpaired native CCTV evidence stay separate and final identities stay outside
tuning. Agent VM access remains maintenance only. More epochs and app adoption
are not qualified. Full goal remains ACTIVE and incomplete. Exact history follows.

'''
    published={}
    for name in docs:published[name]=finish_prepend(ROOT/name,prefix,backup/name,expected[name])
    history=f'''# Used diagnostic — completed and rejected; do not rerun

The joint comparison completed and all four proposals failed the retained
requirements. Best raw/PNG gains0.865441%/0.886155% still have37 failures each.
{link('Audited results and new design decision',result)}.
{link('Fresh changed manual diagnostic',guide)}.
The original commands below are retained history, not an instruction to relaunch.

'''
    published[used]=finish_prepend(ROOT/used,history,backup/used,expected[used])
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__),evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in proofs+[design_path]},
        documents=published,new_documents_sha256={result:sha(ROOT/result),guide:sha(ROOT/guide)},
        all_four_actual_proposals_rejected=True,all20_visual_sheets_reviewed=True,
        new_manual_diagnostic_verified_unrun=True,new_protocol_sha256=pin,new_archive_sha256=sha(archive),
        original_checkpoint_and_app_preserved=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,result=result,manual_guide=guide,canonical_documents=len(docs),history_preserved=True,
        VM_uploaded_or_launched=False,training_qualified=False))


if __name__=='__main__':main()
