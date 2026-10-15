"""Publish completed measurement and an independently checked finite next step."""
from datetime import datetime,timezone
from pathlib import Path
from import_cctv_dgp_png_group_gradients_v1_r2_return import ROOT,OUT,RETURN,ANALYSIS,PARENT,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected,new,link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new,finish_prepend


def main():
    stage=ANALYSIS/'publication';assert not (stage/'publication.json').exists()
    joint=ANALYSIS/'joint_raw_png_direction_v1_r2'
    base='cctv_dgp_joint_raw_png_forward_v1';name=base+'_vm';stem='cctv-dgp-joint-raw-png-forward-v1'
    proof=[ANALYSIS/'import.json',ANALYSIS/'independent_audit.json',joint/'direction_review.json',joint/'independent_audit.json',
        OUT/(base+'_preparation.json'),OUT/(base+'_transfer_audit.json'),OUT/(base+'_return_checker_preparation.json'),
        OUT/(base+'_source_gate_audit.json'),OUT/(base+'_bash_syntax.json')]
    assert all(read(path)['complete'] for path in proof)
    imported,audit,direction,independent,prep,transfer,checker,gates,bash=[read(path) for path in proof]
    packet=OUT/name;pin=sha(packet/'protocol.json');archive=OUT/(stem+'-execution.tar.gz')
    assert pin==prep['protocol_sha256']==transfer['protocol_sha256']==checker['protocol_sha256']==gates['protocol_sha256']
    assert sha(archive)==prep['archive_sha256'] and bash['checked_shell_sha256']==sha(packet/'scripts/run_forward.sh') and bash['exit_code']==0
    assert checker['auditor_sha256']==gates['future_auditor_sha256']==sha(ROOT/'scripts'/('audit_'+base+'_return.py'))
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/file)==digest for file,digest in protected.items())
    previous=read(OUT/'cctv_dgp_png_group_gradients_v1_r2_publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected={file:previous['documents'][file]['after_sha256'] for file in docs}
    used='CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_VM.md';expected[used]=previous['new_documents_sha256'][used]
    assert all(sha(ROOT/file)==digest or ((stage/'before'/file).exists() and sha(stage/'before'/file)==digest) for file,digest in expected.items())
    result_name='CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_RESULTS.md'
    guide_name='CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_VM.md'
    report=f'''# PNG-aware learning measurement and joint direction — 13 October 2026

**The conversion repair worked and the manual R2 measurement completed.**
This establishes usable diagnostic evidence; it has not trained or qualified an
improved checkpoint. The permanent app restorer and every old failure remain.

The L4 run took **{audit['VM_seconds']:.2f} seconds**:40 candidate forwards,
20 fixed-recognizer forwards and320 gradient queries. There were zero optimizer
updates, parameter displacements or epochs. Its source reports unchanged model,
buffer, original and recognizer states. The return has132 files and5,852 parent
bindings, independently checked by path/size/hash; no neural replay is claimed.
Its **{imported['archive_bytes']:,}-byte** archive SHA256 is
`{imported['archive_sha256']}`.
{link('Complete return audit','outputs/cctv_dgp_png_group_gradients_v1_r2_analysis/independent_audit.json')}.

All100 raw controls and100 PNG controls match the frozen baseline. Saved exact
RGB32/feature64 byte values pass the256-level CUDA table. The legacy CUDA division
differs on126 levels, by at most5.96046448e-8 and one float32 ULP. Those are actual
returned table values, unlike the previous CPU illustration. The strict forward
comparison now passes without loosening its equality or scientific quality gates.
All320 scalar/scope records,32 mean gradient arrays and four sampled full vectors
pass saved arithmetic checks. The derivatives remain explicitly approximate
straight-through constructions, not derivatives of discontinuous PNG conversion.

## What changes the improvement direction

The previous raw-only direction initially worsens one measured delivered loss:
`PNG/dataset/thumbnails128x128/clear/MSE`. Its saved directional dot is positive,
5.9090397524e-5. This is one demonstrated initial conflict; it is not the entire
explanation for the37 raw/37 PNG finite preservation failures at the earlier
0.862655% best raw gain. Curvature and quantization still require actual checks.
{link('Earlier finite limits','CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md')}.

A new minimum physical-norm direction includes **32 raw and32 PNG losses**, with
both sources, all five profiles, clear replay and each source's degraded feature
loss. Design targets are local fractional descent1 per unit rate for four feature
groups and0.1 for60 preservation groups; they are not replacement output gates.
The new float32 vector predicts descent for all64 saved derivatives. All64 rows
and all64 positive eigendimensions remain; no ridge, group deletion, app change,
quality threshold change or pretrained-restorer substitution is used.

The first arithmetic attempt stopped at a conservative full-rank guard because
the smallest Gram eigenvalue is about4.74455e-10. It exceeds the measured matrix's
machine-precision rank bound; raw/PNG gradients are nearly collinear. The revised
solve uses a full orthonormal derivative basis and a nonnegative dual KKT witness.
A separate metadata-writing stop is retained with its partial unqualified array;
older raw records lacked the newer `balanced` dot key. The completed review
computes that comparison directly from the verified saved vectors.
{link('Retained numerical stop','outputs/cctv_dgp_png_group_gradients_v1_r2_analysis/joint_raw_png_direction_v1_stop/numeric_rank_review.json')};
{link('Retained metadata stop','outputs/cctv_dgp_png_group_gradients_v1_r2_analysis/joint_raw_png_direction_v1/metadata_stop.json')}.

Independent different-chunk Gram/full-vector reconstruction verifies the convex
KKT witness and all64 float32 directional constraints. Norm is
{direction['direction_norm']:.9f}; the largest proposed step is about0.02018% of
the selected initial parameter norm, within the retained0.1% bound. These are
saved-array arithmetic results, not improved faces or a training trajectory.
{link('Joint direction and predictions','outputs/cctv_dgp_png_group_gradients_v1_r2_analysis/joint_raw_png_direction_v1_r2/direction_review.json')};
{link('Independent joint check','outputs/cctv_dgp_png_group_gradients_v1_r2_analysis/joint_raw_png_direction_v1_r2/independent_audit.json')}.

Multi-objective literature motivates investigating conflicting losses; this
inequality design is not the paper's MGDA algorithm or its convergence result.
[Sener and Koltun](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html).
The dual witness uses nonnegative least-squares arithmetic, whose primary
documentation describes its convex problem and KKT conditions.
[SciPy NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html).
Neither source establishes useful face restoration from these derivatives.

## The next manual test changes weights in a disposable copy

The new verified, UNRUN packet compares positive steps **.0001,.0005,.005,.02**.
The first two probe preservation below the previously failing larger steps;
the other two probe structure gain. Every proposal resets to the same pinned
initial B, with37 selected tensors and the frozen original/prior/recognizer.
It saves100 baseline controls,400 raw outputs,400 delivered PNGs, identity
embeddings and exact proposed parameter/state hashes. No optimizer, gradient,
committed update, epoch or promoted checkpoint is produced by this comparison.

All unchanged raw/PNG source/profile preservation and brightness checks remain.
Structure must reach1% globally and for each source in both representations;
every planned20-sheet/800-cell comparison needs visual review. A small preserving
step below1% remains diagnostic. A favorable derivative cannot approve more
epochs. Any successful finite output would still require a separately justified
manual training continuation, full TRAIN/native DEV review and final separation.
No automatic follow-on, historical rerun or unchanged failed recipe is allowed.
{link('Exact next manual commands',guide_name)}.

Scope is the same20 previously exposed photographic TRAIN references/100 cases;
there is no native CCTV or reserved final tuning, independent subject count,
ethnicity, Zamboanga performance or hidden identity claim. The original DGP
checkpoint remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
All five milestones, independent useful native/final assessment, DGP-primary
Auto/override and seven separate automatic/assisted covering families remain.
The recent covering-record inspection was read-only; no detector or app changed.
'''
    retained_new(ROOT/result_name,report)
    guide=f'''# Joint raw/PNG weight-change comparison — manual L4 commands

**Verified and UNRUN.** This tests the new64-loss direction on actual outputs.
Four disposable positive weight changes: .0001,.0005,.005,.02, each reset to the
same initial model. Zero optimizer updates, gradients, committed updates, epochs
or checkpoints. {link('Completed measurement and purpose',result_name)}.

Require **4 GiB free after installation**. Estimate **2–5 minutes**. Worker/
measurement1200/900s; outer1230s. Export300s/outer330s plus30s kill grace per stage.
VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. Export reserves the full payload;
current VM storage/workload is checked at execution. Upload **{prep['archive_bytes']:,} bytes**.

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
test ! -e ~/forensic-dgp/{name} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_joint_raw_png_forward_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{name} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{name}.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_forward.sh {pin}
```

Hash-only preflight expects4 packet assets,5,852 parent assets,100 fixed affines
and zero neural/gradient/update calls. Progress visits20 batches for each of four
proposals. Export runs after completion or a retained stop; its `complete: true`
is not a model-quality pass. Detach **Ctrl+B,D**; reattach with
`tmux attach -t dgp_joint_raw_png_forward_v1`.
Existing GPU workloads are preserved. Do not delete old outputs or bypass stops.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "{stem}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "{stem}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "{stem}-export.json"
```

Each Windows PuTTY download has one remote source. Return all three files for
independent parameter/image hashes,400 raw/PNG outputs,800 metric rows, each
source's1% structure requirement and all20 visual sheets. No automatic more
training or app adoption. The existing named L4 VM is required; changing VM needs
identity/migration verification. All research assets and failures stay retained.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
LF checksum and future audit fixtures pass.
{link('Transfer evidence','outputs/'+base+'_transfer_audit.json')}.
'''
    retained_new(ROOT/guide_name,guide)
    stage.mkdir(exist_ok=True);backup=stage/'before';backup.mkdir(exist_ok=True)
    prefix=f'''# Latest learning evidence — 13 October 2026; PNG measurement completed

Manual R2 completed in119.13s:40 B/20 recognizer forwards,320 approximate-gradient
queries, zero optimizer updates, parameter displacements, epochs or checkpoints.
All132 returned/5,852 parent bindings,100 raw/PNG controls,320 records,32 groups
and four samples pass saved-evidence audits. Canonical256-byte CUDA parity passed;
legacy CUDA division differs on126 levels by at most one float32 ULP.
The previous conversion stop is fixed. {link('Complete findings',result_name)}.

The old raw-only direction initially worsens one PNG clear-MSE group. A new
joint32 raw/32 PNG direction passes independent full-vector/KKT checks with no
loss-group removal, ridge or quality-gate change. It remains a local approximate
prediction, not improved output or epoch qualification. Numerical and metadata
arithmetic stops are preserved. The quoted0.863% old gain still has37 raw/37 PNG
preservation failures and remains unqualified.

New manual finite packet is verified and UNRUN:four positive steps
.0001/.0005/.005/.02 on the same100 paired TRAIN cases,100 initial controls,
400 raw/400 PNG outputs, zero optimizer/gradient/committed updates or checkpoints.
Each step resets; all raw/PNG preservation/brightness checks and1% global and
each-source structure requirements remain. All20 sheets need returned review.
{link('Exact upload,tmux and download commands',guide_name)}.

The permanent app checkpoint, frozen components, splits, caches, migration
backups, failures and app design remain. All five milestones and seven covering
families stay required, with separate unpaired native and paired evidence,
untouched final identities and independent final review. Agent VM access remains
maintenance only; new runs stay manual L4 tmux with finite stops. Full goal
ACTIVE and incomplete; conditional more epochs are not activated. Exact history follows.

'''
    documents={file:finish_prepend(ROOT/file,prefix,backup/file,expected[file]) for file in docs}
    used_prefix=f'''# Completed and audited R2 measurement — do not rerun

The manual measurement completed; its return and canonical CUDA repair are
audited. Zero weight updates or new checkpoint. {link('Results',result_name)}.
Next is a distinct finite actual-weight comparison: {link('New manual guide',guide_name)}.
The earlier prepared/unrun text and commands below are preserved history.

'''
    documents[used]=finish_prepend(ROOT/used,used_prefix,backup/used,expected[used])
    assert all(sha(ROOT/file)==digest for file,digest in protected.items())
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        proof_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in proof},documents=documents,
        new_documents_sha256={result_name:sha(ROOT/result_name),guide_name:sha(ROOT/guide_name)},
        next_protocol_sha256=pin,next_archive_sha256=sha(archive),protected_app_files=protected,
        R2_measurement_completed=True,new_actual_VM_comparison_unrun=True,no_agent_VM_connection_or_launch=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,results=result_name,next_manual_guide=guide_name,new_actual_VM_run=False))


if __name__=='__main__':main()
