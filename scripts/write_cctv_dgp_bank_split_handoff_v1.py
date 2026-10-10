"""Publish the reviewed resource stop and verified manual next steps."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
ANALYSIS=OUT/'cctv_dgp_bank_comparison_v1_analysis'
CLEAN=OUT/'cctv_dgp_bank_split_offload_v1_r1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,text):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    date=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    audit=read(ANALYSIS/'independent_audit.json');vis=read(ANALYSIS/'visual_review.json')
    transfer=read(OUT/'cctv_dgp_bank_split_transfer_v1_audit.json')
    cleanup=read(CLEAN/'independent_audit.json');space=read(CLEAN/'inventory_after.json')
    assert audit['complete'] and audit['optimizer_updates_in_VM']==0
    assert vis['complete'] and vis['all_26_planned_sheets_viewed'] and not vis['candidate_comparison_possible']
    assert transfer['complete'] and cleanup['complete'] and cleanup['exact_deletions_checked']==930
    assert space['manual_A_space_requirement_met']
    prep=read(OUT/'cctv_dgp_bank_comparison_v1_r2_preparation.json');A,B=prep['routes']
    aroot=A['name'];astem=A['stem'];apin=A['protocol_sha256']
    report=f'''# Bank comparison v1: returned resource stop, {date}

The manual VM run stopped before its first optimizer update. It completed a
baseline snapshot of 3,905 paired TRAIN cases and 24 native DEV crops, then
projected 10,568,596,813 bytes (9.84 GiB) of outputs against a 6 GiB result cap.
This is a resource-layout failure. It supplies no trained A/B candidate and
does not establish whether either proposed training design improves quality.
The original archive, protocol, traceback, storage projection and supervisor
receipt remain intact. Export completion establishes packaging only.

The 2,924,238,572-byte return is SHA256
9fc3a0d7f62e1ae9d2b51eaf567ea0aaaca0236865d4ab1ad5c010caff1f1dcc.
The importer checked 6,549 regular members and all 5,611 original prepared
bindings. The independent audit rechecked 6,548 export-manifest bindings,
3,905 saved raw/PNG paired outputs and 24 native compositions. No candidate
gate, new full model state, CPU/CUDA prior fixture or neural replay was present
to audit. Neither import nor audit ran optimizer updates or gradients.

All 26 planned visual sheets were inspected at original resolution: 20 paired
TRAIN sheets and six native sheets, representing 124 input/baseline cases.
The 372 candidate tiles are explicitly missing. Clear paired examples retain
broad appearance but soften eyes, hair, skin, lips and teeth; strongly degraded
examples remain unresolved. Native ChokePoint t033 crops are heavily softened;
t067 crops retain more broad structure with soft fine detail. This is baseline
review by the implementing assistant, not independent final evaluation.
Native CCTV is unpaired: no PSNR/SSIM, ethnicity or Zamboanga performance claim.
Clear glasses/non-obstructing hair are preservation observations; visible hands
or head coverings on paired sheets do not constitute completion evaluation.

## Resource-only remedy

The comparison is split into separate immutable manual A and B roots. Each
starts its original initializer and retains the identical objective, Adam,
normalizers, active clear/blur anchors, lossless float storage, delivered-PNG
policy, data, schedule and structure/preservation/brightness thresholds.
Both use 50 updates, 250 distinct TRAIN references (125 per source) and 1,250
profile-image exposures; zero complete epochs, about 0.3201 of the 781-reference
TRAIN set. The 24 native DEV crops remain unpaired; final identities are unused.
No failed trained state is resumed: the parent resource stop had zero updates.

A is the corrected current restorer control. B adapts input conditioning and
reconstruction around a separately declared frozen external StyleGAN2 bank.
Our conditioning/reconstruction are trainable; the external generator is not
claimed as trained by us or as an exact published-DGP reproduction. Pretrained
restoration models remain comparison baselines. Pretraining overlap is not
fully excluded. A restoration improvement does not qualify covered completion.

The measured output forecast is now {transfer['routes'][0]['output_bound_GiB']:.3f}
GiB for A and {transfer['routes'][1]['output_bound_GiB']:.3f} GiB for B, each below
the unchanged 6 GiB cap. Historical baseline growth has a 5% allowance and is
checked before neural work; fresh baseline/output/export space is rechecked
before the optimizer. This is a static bound, not a measured future run.
The strict compression bound follows upstream [zlib deflate.c]
(https://raw.githubusercontent.com/madler/zlib/v1.3.1/deflate.c); no image values
or scientific gates changed. The first independent-check stop remains retained:
a generic 16 MiB metadata allowance was inappropriately applied to the 100-case
ablation. Its unchanged fixed JSON schema has a conservative 242,563-byte bound;
the corrected checker reserves 8 MiB for it. Sealed packets were not edited.

Independent checks compared computational ASTs, all inherited assets, all
5,616 members of each archive, exact ASCII LF checksums and transfer CLI runs.
Six unsafe mutations per route were rejected (weaker gate, final-role leakage,
schedule duplicate, removed anchor, wrong route, extra updates). No model,
gradient or optimizer call occurred in packet preparation/checking.

## Storage maintenance and recovery

Exact cleanup removed 929 backed-up RGB image output files from the stopped
bank run plus its redundant VM home result archive: 930 singly linked files.
Every retained local backup was rehashed before removal and independently
rechecked afterwards. Reclaimed allocation: {cleanup['allocated_reclaim_bytes']/1024**3:.3f}
GiB; fresh VM free space: {space['free_GiB']:.3f} GiB. Conservative free space
after uploading/installing A: {space['conservative_A_post_install_GiB']:.3f} GiB,
above its 16 GiB minimum. Future launch still performs live checks.

Retained-state audit compared {cleanup['retained_metadata_entries']:,} research
metadata entries and {cleanup['retained_bank_byte_hashes']:,} retained bank-file
hashes. Checkpoints, scientific caches, inputs/splits, source/instructions,
metrics, provenance, logs, failures and gates remained. All scientific cache
bytes were not rehashed; exact singly linked deletions and retained metadata
are the scope of that preservation proof. Original app/checkpoint bindings
are unchanged. No model work or new VM training was launched by maintenance.
The initial local receipt-key error is retained in the original offload folder;
the repaired preparation correctly binds compressed bytes to import evidence
and shares the independent audit's export-manifest fingerprint.

Recover the offloaded files from the complete local results archive or the
hash-verified extracted return. This return is a research packet, not a full
boot-disk/environment backup. Prior scientific-cache/output backups remain.

Evidence: outputs/cctv_dgp_bank_comparison_v1_analysis/import.json,
independent_audit.json, visual_review.json; outputs/cctv_dgp_bank_split_transfer_v1_audit.json;
outputs/cctv_dgp_bank_split_offload_v1_r1/independent_audit.json and inventory_after.json.

## Next decision

Run A manually with CCTV_DGP_BANK_COMPARISON_V1_R2_VM.md. Return and independently
audit its files before installing/running B; offload new images if needed.
B remains prepared, unrun. Learning/CUDA/timing/quality can still stop either
route. A failed quality gate is retained; B starts independently. Review both
under equal exposure before choosing any further finite epoch study. The
proposed additional 1/2/5-epoch checkpoints remain conditional, assistant-proposed
numbers; no extra epochs are automatically authorized or scheduled by this fix.

The permanent baseline remains SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
No new checkpoint is adopted. Full goal ACTIVE: useful native development
acceptance, independent reserved final review, DGP-primary app integration,
Auto/override, regressions, bundled inline Playwright and all seven separately
reported automatic/assisted completion families remain required.
'''
    report=report.replace('[zlib deflate.c]\n(', '[zlib deflate.c](')
    report_path=ROOT/'CCTV_DGP_BANK_COMPARISON_V1_RETURN_REVIEW.md';save(report_path,report)
    guide=f'''# Bank comparison v1 r2: manual route A

Run the new **A** root only. The original v1 stopped for storage before training;
keep its history. A and B now run separately with unchanged scientific checks.
The local transfer audits passed; new CUDA/training/quality checks remain pending.

Latest verified VM free space: **{space['free_GiB']:.2f} GiB**. Conservative space
after A installation: **{space['conservative_A_post_install_GiB']:.2f} GiB**.
Launch requires **16 GiB free after installation**. Baseline alone took 9.4 minutes
in the parent run; full revised route timing is unmeasured. Hard worker cap:
90 minutes plus at most 10.5 minutes supervised export. Fit cap 10 minutes,
snapshot cap 20 minutes, preflight cap five minutes, peak VRAM 20 GiB, disk
reserve 1 GiB, outputs 6 GiB. Each route is limited to 50 updates, not more epochs.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{astem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{astem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\\r' < {astem}-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{aroot} &&
tar -xzf {astem}-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256: `{A['archive_sha256']}`.
The CR filter handles Windows checksum line endings without altering the upload.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_comparison_A
```

4. Launch **inside that tmux session**:

```bash
cd ~/forensic-dgp/{aroot} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha {apin} --verify-transfer &&
bash scripts/run_bank_comparison.sh {apin}
```

Detach without stopping the run with **Ctrl+B**, then **D**. Reopen the same
tmux session to see progress. Do not repeat step 4 in a used/partial run root.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{astem}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{astem}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{astem}-export.json" "."
```

Each SCP command has one remote source, as required by Windows/PuTTY.
An export `complete: true` means that evidence was packaged. A trainer error,
gate stop or missing candidate remains a failure even if export succeeds.
Return all three files; independent raw/PNG and visual audit precedes B.

B is already frozen locally as `{B['stem']}-execution.tar.gz`, protocol
`{B['protocol_sha256']}`. It will start its own initializer after A is audited
and VM capacity is checked again. No automatic B launch or app promotion occurs.
Full goal and all seven completion families remain incomplete.
'''
    guide_path=ROOT/'CCTV_DGP_BANK_COMPARISON_V1_R2_VM.md';save(guide_path,guide)
    handoff=ROOT/'PROJECT_HANDOFF.md';backup=ANALYSIS/'documents_before';backup.mkdir()
    shutil.copyfile(handoff,backup/handoff.name)
    header=f'''# Latest return and maintenance — {date}: bank v1 storage stop audited; split-route A ready for manual run

The downloaded bank-comparison v1 run stopped before training: zero optimizer
updates, 9.84 GiB projected result output versus the unchanged 6 GiB cap.
The original archive/protocol/stop remain. Independent checks covered all
3,905 paired raw/PNG outputs and 24 native unpaired compositions. All 26
planned sheets were viewed at original resolution; 372 candidate tiles are
missing. This baseline review supplies no new candidate improvement or final
qualification. Export completion is packaging, not training/quality success.

The resource-only revision uses separate immutable A and B roots. Both models,
objective, data, equal exposures, active clear/blur anchors and scientific
quality gates are unchanged. Each retains 50 updates / 250 distinct TRAIN
references / 1,250 profile exposures / zero complete epochs. Independent source,
AST, full archive, LF and transfer-CLI checks passed. A/B output forecasts are
5.540 / 5.665 GiB. Original checker metadata-allowance stop is retained; the
corrected fixed-schema bound changed no sealed packet or scientific threshold.

Authorized maintenance offloaded exactly 929 inactive bank RGB output files
and one redundant home result archive after complete local backup verification.
Independent closure checked 930 exact deletions, {cleanup['retained_metadata_entries']:,}
retained research metadata entries and {cleanup['retained_bank_byte_hashes']:,}
retained bank byte hashes. Reclaimed allocation {cleanup['allocated_reclaim_bytes']/1024**3:.3f} GiB;
fresh VM free space {space['free_GiB']:.3f} GiB; conservative A post-install free
space {space['conservative_A_post_install_GiB']:.3f} GiB. All scientific cache bytes
were not rehashed. Checkpoints, caches, inputs/splits, metrics, provenance,
instructions, failure/gate/log records and app/checkpoint bindings remain.
Original local preparation receipt-key failure was retained; no deletion occurred
on that attempt. Current VM was not started, stopped or switched by the agent.
No model/gradient/training work was launched by the agent.

Manual next: CCTV_DGP_BANK_COMPARISON_V1_R2_VM.md, route A only. B is prepared
but requires A's independent return review and another storage check first.
New actual training stays human/manual tmux. Do not rerun original v1 or resume
a rejected trained state. Return bytes and restoration/covering-family quality
are distinct. Full goal ACTIVE and incomplete; no checkpoint is adopted.

Detail: CCTV_DGP_BANK_COMPARISON_V1_RETURN_REVIEW.md.
Evidence: outputs/cctv_dgp_bank_comparison_v1_analysis/import.json,
independent_audit.json, visual_review.json; outputs/cctv_dgp_bank_split_transfer_v1_audit.json;
outputs/cctv_dgp_bank_split_offload_v1_r1/independent_audit.json and inventory_after.json.
Exact previous PROJECT_HANDOFF.md bytes: analysis/documents_before/PROJECT_HANDOFF.md.
All older sections below are historical.

---

'''
    old=handoff.read_text(encoding='utf-8');handoff.write_text(header+old,encoding='utf-8',newline='\n')
    save(ANALYSIS/'handoff_publication.json',json.dumps(dict(complete=True,date_Asia_Manila=date,guide_sha256=sha(guide_path),report_sha256=sha(report_path),handoff_before_sha256=sha(backup/handoff.name),handoff_after_sha256=sha(handoff),manual_training_pending=True,goal_complete=False),indent=2)+'\n')
    print(dict(complete=True,guide=str(guide_path),VM_free_GiB=space['free_GiB'],manual_A_pending=True))

if __name__=='__main__':main()
