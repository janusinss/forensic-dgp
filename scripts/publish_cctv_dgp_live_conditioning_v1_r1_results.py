"""Publish the verified activation result and finite, unrun manual pilot."""
from datetime import datetime,timezone
from pathlib import Path
from import_cctv_dgp_live_conditioning_v1_r1_return import ROOT,OUT,ANALYSIS,RETURN,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new,finish_prepend


def main():
    stage=ANALYSIS/'publication';assert not (stage/'publication.json').exists()
    imported=read(ANALYSIS/'import.json');audit=read(ANALYSIS/'independent_audit.json')
    sub=read(ANALYSIS/'learning_subspace_audit.json');review=read(ANALYSIS/'learning_subspace_review.json')
    prep=read(OUT/'cctv_dgp_conditioning_learning_v1_r2_preparation.json')
    transfer=read(OUT/'cctv_dgp_conditioning_learning_v1_r2_transfer_audit.json')
    assert all(r['complete'] for r in [imported,audit,sub,review,prep,transfer])
    packet=OUT/'cctv_dgp_conditioning_learning_v1_r2_vm'
    pin=sha(packet/'protocol.json');assert pin==prep['protocol_sha256']==transfer['protocol_sha256']
    assert sha(OUT/'cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz')==prep['archive_sha256']==transfer['archive_sha256']
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    shell=packet/'scripts/run_learning.sh'
    # Git Bash -n was executed read-only successfully before publication.
    syntax=OUT/'cctv_dgp_conditioning_learning_v1_r2_bash_syntax.json'
    assert read(syntax)['complete'] and read(syntax)['checked_shell_sha256']==sha(shell)
    run=read(RETURN/'outputs/live_conditioning_v1_r1/run_results.json')
    norms=read(RETURN/'outputs/live_conditioning_v1_r1/conditioning_gradient_norms.json')['groups']
    normrows=[]
    for source in ['dataset/asian_faces','dataset/thumbnails128x128']:
        for suffix in ['clear/MSE','degraded/feature_MSE']:
            values=norms[source+'/'+suffix]
            assert len(values)==12 and all(v>0 for v in values.values())
            normrows.append(f"| {source} | {suffix} | {min(values.values()):.8g} | {max(values.values()):.8g} |")
    report_name='CCTV_DGP_LIVE_CONDITIONING_V1_R1_RESULTS.md'
    guide_name='CCTV_DGP_CONDITIONING_LEARNING_V1_R2_VM.md'
    inventory_dir=OUT/'cctv_dgp_conditioning_learning_v1_storage/inventory'
    inventory=read(inventory_dir/'guest_inventory.json');inventory_receipt=read(inventory_dir/'inventory_receipt.json')
    assert inventory['complete'] and inventory_receipt['complete'] and inventory_receipt['files_removed']==0
    assert sha(inventory_dir/'guest_inventory.json')==inventory_receipt['inventory_sha256']
    assert inventory['disk']['free_bytes']>=9*1024**3 and not inventory['gpu_processes']['stdout'].strip()
    free=inventory['disk']['free_bytes']/1024**3
    report=f'''# Live conditioning R1 returned audit — 13 October 2026

**The initial gradient blockage is resolved on the fixed photographic TRAIN
diagnostic. Image quality and additional epochs remain unqualified.** All12
style/spatial conditioning tensors have finite nonzero aggregate gradients for
clear-preservation MSE and degraded-detail MSE in each sampled source. The100
initial rawRGB controls match the original DGP bit for bit. No optimizer update,
parameter change, new checkpoint or epoch was performed by this diagnostic.

The manual L4 measurement took **{run['seconds']:.2f}s**. It completed40 candidate
forwards,40 live plus40 frozen feature-branch passes,20 recognizer forwards and320
gradient queries. Peak allocated VRAM was{run['peak_VRAM']/1024**3:.3f}GiB and peak
host RSS{run['peak_RSS']/1024**3:.3f}GiB. The full candidate state/buffers and original
DGP/recognizer remained unchanged. The all12 requirement passed in all four
source/task combinations:

| Photographic source label | Task | Minimum tensor gradient norm | Maximum |
|---|---|---:|---:|
{chr(10).join(normrows)}

These source labels identify retained photographic archives, not ethnicity,
native capture origin or Zamboanga performance. Scope is20 TRAIN references,
100 paired cases and32 raw loss groups. Native CCTV and final identity pixels
were excluded. No new improved PNG output or covering completion was produced.
{link('Independent return audit','outputs/cctv_dgp_live_conditioning_v1_r1_analysis/independent_audit.json')}.

The archive is{imported['archive_bytes']:,}bytes, SHA256
`{imported['archive_sha256']}`. Every one of92 returned file bindings and5,852
parent assets was rechecked. The independent audit recalculates saved norms,
direction dots, contribution scopes and controls; it does not rerun all320 neural
derivatives. Two clear-MSE decoder/readout sample derivatives were compared to
the prior measured gradients. This result establishes activation, not a finite
image-quality gain. Earlier B50 already changed all37 tensors and still failed;
the new finding does not explain away every historical failure.

## Selected finite learning test

The saved raw derivative-span review found a common local descent direction in
both the full37-tensor space and the12-tensor conditioning space while retaining
all32 losses. The latter has direction norm{review['subspaces']['conditioning12']['direction_norm']:.6f}
and Gram condition number{review['subspaces']['conditioning12']['condition_number']:.2f},
compared with{review['subspaces']['all37']['condition_number']:.2f} for all37.
These are arithmetic results, with prospective finite and PNG behavior pending.
{link('Independent subspace audit','outputs/cctv_dgp_live_conditioning_v1_r1_analysis/learning_subspace_audit.json')}.

The next manual pilot trains only the12 now-active conditioning tensors. The
current decoder, readouts, same-input frozen reference, external generative bank,
recognizer and buffers stay fixed. It permits **at most2 accepted full-cohort
gradient-descent updates**, with at most3 actual backtracking trials per update.
It refreshes all32 raw and32 approximate PNG loss derivatives at the current
weights after an accepted update. Each accepted change must preserve all17 raw
and17 PNG metric groups, satisfy the0.2 brightness limit and improve degraded
detail globally and in both sources. The finite checkpoint still requires1%
structure gain globally and per source on raw and PNG. Full TRAIN's10% structure
requirement and useful native DEV qualification remain separate later gates.

Accepted research components and every rejected trial are retained with true
update counts, raw/PNG images, gradients and state receipts. The cumulative
parameter-distance limit remains0.1% of the initial37-tensor norm. No fixed old
direction is reused after weights change. No longer epoch run or app adoption
follows automatically. The100-case test is a finite training-method pilot; it
does not cover a full epoch, reserved final identities or seven completion families.
{link('Verified manual upload, tmux and download commands',guide_name)}.

The first two local drafts remain retained. Final preparation restored the
retained float32 scientific identity-dot scoring (maximum draft difference
7.7264e-8 on200 saved rows), freezes the original cached baseline scoring while
checking live embedding replay, checks the internal measurement cap at every
clock/final verification, and explicitly rejects nonfinite scientific metrics.
All original quality thresholds remain. R2 is the only published launch guide.

The design hypothesis uses a frozen same-observed-input feature reference and
active multiscale conditioning. GLEAN supplies primary evidence for an
encoder–bank–decoder using a frozen generative bank; our adaptation remains a
separate hypothesis, not a reproduced result or established identity recovery.
[GLEAN paper](https://arxiv.org/abs/2012.00739).
Separate loss constraints are motivated by documented conflicts in multi-objective
learning. Our full-span constrained minimum-norm direction is not claimed as
the paper's exact MGDA method or a finite preservation guarantee.
[Sener and Koltun](https://arxiv.org/abs/1810.04650).
The exact PNG forward uses an explicitly approximate identity backward; its
heuristic nature follows the general straight-through-estimator literature.
[Bengio, Léonard and Courville](https://arxiv.org/abs/1308.3432).

## Storage and remaining goal

The authorized fresh maintenance inventory at{inventory['UTC']} found
**{free:.2f}GiB free** and an idle GPU. No files were removed and no training was
launched by the agent. The named L4/g2-standard-4 instance and SSH host key were
verified. The pilot requires9GiB free after installation and rechecks its own
reserve; this snapshot is not a future free-space guarantee.
{link('Fresh inventory','outputs/cctv_dgp_conditioning_learning_v1_storage/inventory/guest_inventory.json')}.

The original app checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The DGP-led app, Auto/override design, native development/final separation,
paired/unpaired evidence boundary, original checkpoints/splits/failures and all
seven completion families remain in scope. All actual training is human-manual
on the existing L4. Direct agent VM access remains storage maintenance only.
Milestone3 advances from verified activation to a prepared finite learning pilot;
all five milestones and independent useful-output/final/app qualification remain
ACTIVE and incomplete.
'''
    guide=f'''# Conditioning learning V1 R2 — finite manual L4 pilot

**Verified and UNRUN. This is actual training, capped at2 accepted updates.**
{link('Returned activation result and design rationale',report_name)}.

It learns only12 conditioning tensors on the fixed20 photographic TRAIN
references/100 cases. The original decoder and frozen generative bank stay fixed.
Every update needs actual raw and delivered PNG preservation, brightness<=0.2
and positive detail progress in both source groups. At most3 trials per update;
all rejected trials are saved. The finite endpoint still needs1% structure gain
globally and per source, on raw and PNG. It stops early if that endpoint passes.
Full epochs, useful native CCTV, final identities and covering-family qualification
require subsequent independent evidence. No automatic follow-on or app promotion.

Require **9GiB free after installation**. Estimate **8–15 minutes plus export**.
The fresh maintenance snapshot found{free:.2f}GiB free; launch checks current space.
Learning1350s/internal worker1500s/outer1530s; export420s/outer450s, each outer
timeout has30s kill grace. Total outer allowance34minutes. Gradient round600s,
solver60s; VRAM20GiB/RSS12GiB/output3.5GiB/reserve1GiB. Maximum180 candidate and360
recognizer forwards and1,280 gradient queries. The upload is{prep['archive_bytes']:,}bytes.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_conditioning_learning_v1_r2_vm &&
tar -xzf cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux** from the VM SSH shell:
```bash
tmux new-session -A -s dgp_conditioning_learning_v1_r2
```
If an older tmux session is attached, detach with **Ctrl+B,D** first.

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_conditioning_learning_v1_r2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_learning.sh {pin}
```
Detach **Ctrl+B,D**; reattach with `tmux attach -t dgp_conditioning_learning_v1_r2`.
`review.log` retains live progress. The shell prints it after the finite worker
ends. Export also runs after a retained stop. Archive `complete: true` certifies
transfer only; `accepted_updates` reports true learning counts. A stopped run can
contain partial accepted research components. Return failures for audit and keep
the used directory. Original checkpoints, splits and gate records stay retained.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r2-results.tar.gz" "cctv-dgp-conditioning-learning-v1-r2-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r2-results.tar.gz.sha256" "cctv-dgp-conditioning-learning-v1-r2-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r2-export.json" "cctv-dgp-conditioning-learning-v1-r2-export.json"
```
PuTTY downloads use one remote source each. Return all3 files for independent
raw/PNG metrics, gradients, state, rollback and every visual-sheet review before
another training decision. Preserve the original unqualified app restorer.
Use this R2 guide; earlier local drafts were superseded during preparation.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
{link('Transfer/state and retained-gate audit','outputs/cctv_dgp_conditioning_learning_v1_r2_transfer_audit.json')}.
'''
    retained_new(ROOT/report_name,report);retained_new(ROOT/guide_name,guide)
    previous=read(OUT/'cctv_dgp_joint_raw_png_forward_v1_analysis/publication_live_conditioning_v1_r1/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
          'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected={n:previous['documents'][n]['after_sha256'] for n in docs}
    old_guide='CCTV_DGP_LIVE_CONDITIONING_V1_R1_VM.md'
    expected[old_guide]=previous['new_documents_sha256'][old_guide]
    assert all(sha(ROOT/n)==h for n,h in expected.items())
    stage.mkdir(exist_ok=True);backup=stage/'before';backup.mkdir(exist_ok=True)
    prefix=f'''# Returned activation verified; finite conditioning learning prepared — 13 October 2026

Live conditioning R1 passed the independent returned audit: all12 conditioning
tensors have finite nonzero preservation/detail gradients for both sampled
sources, and all100 initial raw controls match the original DGP exactly. The
manual measurement took127.94s:40 candidate/20 recognizer forwards,320 gradient
queries, zero updates/epochs or new checkpoints. Gradient activation is now
verified; useful image quality and longer epochs remain unqualified.
{link('Audited findings and next design',report_name)}.

The next **verified, UNRUN** pilot trains only those12 conditioning tensors,
freezing the original decoder/readouts/bank/reference. It allows at most2
accepted full-cohort updates with3 trials each, refreshing all64 raw/approximate
PNG derivatives after an accepted change. Actual raw/PNG preservation, positive
per-source detail, brightness<=0.2 and cumulative0.1% parameter-distance checks
apply before acceptance. The endpoint retains1% raw/PNG gain globally and per
source; full TRAIN/native qualification and10% full TRAIN structure remain later.
All failures and partial accepted components are retained with true counts.
{link('Exact manual upload, tmux and single-source download commands',guide_name)}.

Fresh authorized VM maintenance found99.93GiB free and idle GPU; no files were
removed or training launched. The new pilot needs9GiB after installation, with
finite timing/storage stops. Agent VM access stays maintenance only; all actual
training remains human-manual on the existing L4. Original checkpoints, app,
splits/provenance/gate failures, final identities, paired/unpaired boundaries,
all five milestones and seven covering families remain. Full goal ACTIVE and
incomplete. The exact previous document follows.

'''
    published={n:finish_prepend(ROOT/n,prefix,backup/n,expected[n]) for n in docs}
    banner=f'''# Returned and audited — completed diagnostic; do not rerun

Live conditioning R1 passed activation and100 exact starting controls with zero
training updates. {link('Returned audit',report_name)}.
{link('Next finite manual learning pilot',guide_name)}.
The completed diagnostic commands below are retained historical instructions.

'''
    published[old_guide]=finish_prepend(ROOT/old_guide,banner,backup/old_guide,expected[old_guide])
    evidence=[ANALYSIS/'import.json',ANALYSIS/'independent_audit.json',ANALYSIS/'learning_subspace_review.json',ANALYSIS/'learning_subspace_audit.json',
        OUT/'cctv_dgp_conditioning_learning_v1_r2_preparation.json',OUT/'cctv_dgp_conditioning_learning_v1_r2_transfer_audit.json',syntax,
        OUT/'cctv_dgp_conditioning_learning_v1_preparation_review.json',OUT/'cctv_dgp_conditioning_learning_v1_r1_scoring_review.json',
        inventory_dir/'inventory_receipt.json']
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        documents=published,new_documents_sha256={n:sha(ROOT/n) for n in [report_name,guide_name]},
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in evidence},
        returned_activation_passed=True,returned_training_updates=0,next_manual_protocol_sha256=pin,
        next_manual_archive_sha256=prep['archive_sha256'],next_training_pilot_unrun=True,
        VM_maintenance_inventory_only=True,VM_files_removed=0,no_agent_training_launch=True,
        milestone3_partial_progress_only=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,activation_passed=True,next_manual_guide=guide_name,history_documents_preserved=len(published),training_unrun=True))


if __name__=='__main__':main()
