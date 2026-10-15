"""Publish audited R1 conversion failure and verified unrun R2 commands."""
from datetime import datetime, timezone
from pathlib import Path
from import_cctv_dgp_png_group_gradients_v1_r2_return import ROOT, OUT, NAME, STEM, PIN, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link


def retained_new(path,text):
    if path.exists():
        assert path.read_bytes()==text.encode('utf-8'), 'Preserve changed staged report: '+str(path)
    else:
        new(path,text)


def finish_prepend(path,prefix,backup,expected):
    if backup.exists():
        old=backup.read_bytes()
        assert sha(backup)==expected and path.read_bytes()==prefix.encode('utf-8')+old
        return dict(before_sha256=expected,after_sha256=sha(path),backup=str(backup),
            prefix_bytes=len(prefix.encode('utf-8')),old_bytes_preserved_as_exact_suffix=True,
            previous_locked_write_verified_without_repetition=True)
    return prepend_expected(path,prefix,backup,expected)


def main():
    stage=OUT/'cctv_dgp_png_group_gradients_v1_r2_publication';assert not (stage/'publication.json').exists()
    analysis=OUT/'cctv_dgp_png_group_gradients_v1_r1_analysis'
    prep_path=OUT/'cctv_dgp_png_group_gradients_v1_r2_preparation.json'
    audit_path=OUT/'cctv_dgp_png_group_gradients_v1_r2_transfer_audit.json'
    forward_path=OUT/'cctv_dgp_png_group_gradients_v1_r2_forward_audit.json'
    bash_path=OUT/'cctv_dgp_png_group_gradients_v1_r2_bash_syntax.json'
    proof=[analysis/'import.json',analysis/'failure_audit.json',analysis/'canonical_boundary_review.json',prep_path,audit_path,forward_path,bash_path]
    assert all(read(p)['complete'] for p in proof)
    imported,failure,boundary,prep,audit,forward,bash=[read(p) for p in proof]
    packet=OUT/NAME;archive=OUT/(STEM+'-execution.tar.gz')
    assert PIN==sha(packet/'protocol.json')==prep['protocol_sha256']==audit['protocol_sha256']
    assert sha(archive)==prep['archive_sha256']==audit['archive_sha256']
    assert bash['exit_code']==0 and sha(packet/'scripts/run_png_groups.sh')==bash['checked_shell_sha256']
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    protected=read(analysis/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    previous=read(OUT/'cctv_dgp_png_group_gradients_v1_r1_publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected={name:previous['documents'][name]['after_sha256'] for name in docs}
    used='CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R1_VM.md';expected[used]=previous['new_document_sha256'][used]
    assert all(sha(ROOT/name)==digest or ((stage/'before'/name).is_file() and sha(stage/'before'/name)==digest)
        for name,digest in expected.items()), 'Preserve concurrent status edits'
    result_name='CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R1_RESULTS.md'
    guide_name='CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_VM.md'
    report=f'''# PNG-gradient R1 conversion stop and R2 repair — 13 October 2026

**R1 stopped before its first gradient query. No structure improvement, training
update, parameter proposal or new checkpoint was produced.** Its source, archive,
logs and exact initial controls are retained. The export's `complete: true` means
an archive was created; it does not mean this diagnostic succeeded.

Location: returned `scripts/cctv_dgp_png_group_gradients_v1_r1_vm.py:200`.
The CUDA RGB32 tensor did not exactly equal NumPy's byte/255 values. The helper's
own CUDA-value equality assertion had passed, so the demonstrated boundary is
that this internal CUDA conversion did not reproduce the authoritative CPU
conversion. The failed CUDA float values were not exported; their mismatch
count, ULP magnitude and precise kernel behavior are not independently known.

The audited run reports **{boundary['VM_seconds']:.2f} seconds**, two B forwards,
zero recognizer forwards, zero gradient queries, zero optimizer updates and no
parameter displacement. All **{imported['returned_file_bindings']}** returned and
**{imported['parent_file_bindings']}** required parent hashes pass. Five saved raw
controls equal the retained raw arrays by bits; all five PNG controls equal
retained PNG pixels. No new candidate images or visual sheets were produced.
These are photographic TRAIN controls, not native CCTV or final-review evidence.
The model was not displaced; no fresh post-failure model-state hash is claimed.

The7,311,160-byte archive SHA256 is
`{imported['archive_sha256']}`.
{link('Independent failure audit', 'outputs/cctv_dgp_png_group_gradients_v1_r1_analysis/failure_audit.json')}.
{link('Canonical boundary and exact controls', 'outputs/cctv_dgp_png_group_gradients_v1_r1_analysis/canonical_boundary_review.json')}.

The previous preparation checked CPU primitives but missed this CUDA boundary.
That was an implementation-check limitation. PyTorch documents that mathematically
equivalent floating computations, including CPU versus GPU, need not match bit
for bit. This supports a numerical-boundary diagnosis, not a specific observed
GPU error magnitude. [Primary numerical-accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
A retained CPU illustration finds419,362 channel differences, at most
5.96046448e-8, between division and multiplication by a float32 reciprocal on
these saved PNGs. That illustration is explicitly **not a GPU reproduction**.

## Why0.863% is still unqualified

The earlier all-group step .02 achieved **0.862655% raw /0.877397% PNG** mean
degraded landmark high-frequency error reduction. This is one saved metric,
not a percentage of facial identity recovered. It also failed **37 raw and37
PNG group–metric preservation checks**. Source-specific raw gains were0.556856%
for `dataset/asian_faces` and0.939342% for `dataset/thumbnails128x128`;
both must reach1%. Dataset labels are not ethnicity or native-capture evidence.
Increasing the same step to .04 made structure worse. Being0.137345 percentage
points below the global raw threshold does not establish safe continued learning.
The much smaller signed proposals also remain unqualified.
{link('Retained finite comparison', 'CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md')}.

The next improvement must reduce structural error while preserving clear faces,
visible appearance and both raw and delivered outputs. Passing1% is an early
learning requirement, not full model, native CCTV, app or covering qualification.
More epochs remain conditional on a finite useful, preserving actual change.

## Fresh repair and next manual measurement

R2 computes authoritative NumPy RGB32 and feature64 byte/255 values once without
any CPU neural or gradient calls, then copies those values to CUDA. Exactly-zero
straight-through terms preserve the forward bits and retain the explicitly
approximate identity backward. Scientific SSIM keeps its exact pixel evaluator
and approximate analytic backward with the same zero-valued assembly. Scientific
loss definitions, masks, baseline states, thresholds and scalar tolerances stay
unchanged. No claim of a true derivative of discontinuous PNG conversion is made.

Before loading models, the manual VM worker must pass all256 byte levels in both
precisions and export the values plus a legacy CUDA division comparison. It then
checks the same100 frozen raw/PNG cases before querying gradients. All local100
saved-value forward checks and256-byte fixtures passed with no neural/autograd
calls. Actual CUDA repair execution remains **unrun**.

The fresh packet retains320 queries,32 groups,40 B/20 recognizer forwards,
zero updates/proposals/epochs/checkpoints and the same timing/storage stops.
Only after its returned gradient evidence is audited can a joint raw/PNG direction
and a separate small actual-change comparison be justified. A gradient dot or
completed archive cannot qualify more training or bypass preservation failures.
{link('Exact manual R2 commands', guide_name)}.

Original app checkpoint, model components, splits, masks, caches, migration
backups, previous failures and app design remain preserved. All five milestones,
separate native/unpaired and paired evidence, independent final identities and
all seven automatic/assisted covering families remain required. No Zamboanga,
ethnicity or exact hidden-identity claim is made. Agent VM access remains
maintenance only; training/gradient work stays human-manual L4 tmux.
'''
    retained_new(ROOT/result_name,report)
    guide=f'''# Canonical PNG-gradient measurement V1-R2 — manual L4 commands

**Verified locally and unrun on the VM.** R1 stopped at the CPU/CUDA conversion
boundary before gradients; preserve that used packet. R2 changes canonical
forward construction, adds a pre-model256-byte CUDA test, and keeps all losses,
cases, scientific gates and320-query limits. Zero optimizer updates, model
proposals, epochs or new checkpoints. {link('Audited cause and structure status',result_name)}.

Require **4 GiB free after installation**. Estimated runtime3–8min. Measurement
cap600s/worker900s/outer930s; export300s/outer330s;30s kill grace per stage.
First two batches must project below600s. VRAM20GiB/RSS12GiB/output1,280MiB/
reserve1GiB; export additionally reserves the full payload. Current VM free space
is checked at run time. Upload **{prep['archive_bytes']:,} bytes**.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_png_group_gradients_v1_r2
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_png_groups.sh {PIN}
```

Hash-only preflight:6 packet assets,5,852 parent assets,32 groups and320 planned
queries. The run must pass256-byte CUDA parity before any model. Progress reaches
20 reference batches/320 queries only on measurement success. Export also runs
after a retained failure; export completion is not diagnostic or model success.
Detach with **Ctrl+B,D**; reattach with
`tmux attach -t dgp_png_group_gradients_v1_r2`.
The worker preserves competing GPU jobs and refuses used packets. Do not delete
previous outputs, rerun a failed recipe unchanged or bypass a stop.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "{STEM}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "{STEM}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "{STEM}-export.json"
```

Each PuTTY download has one remote source. Return all three files for independent
pixel-table, exact raw/PNG control, scientific scalar/embedding, saved-gradient
group and timing checks. No follow-on training is automatic. The existing named
L4 VM is required; a different VM needs identity and migration verification.

Protocol SHA256: `{PIN}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
Checksum text is verified LF. {link('Transfer audit',audit_path.relative_to(ROOT).as_posix())}.
'''
    retained_new(ROOT/guide_name,guide)
    stage.mkdir(exist_ok=True);backup=stage/'before';backup.mkdir(exist_ok=True)
    prefix=f'''# Latest model step — 13 October 2026; PNG conversion stop audited

The returned PNG-gradient R1 stopped in10.56s at canonical CPU/CUDA RGB32 parity:
two B forwards, zero recognizer forwards, gradients, optimizer updates or model
displacements. All15 return/5,852 parent bindings and five raw/PNG controls pass.
No new structure gain or checkpoint was produced. The CPU-only preparation
missed the device conversion boundary. {link('Audited findings',result_name)}.

The quoted0.863% is the earlier0.862655% raw gain with37 raw/37 PNG preservation
failures; it remains unqualified. Bigger versions of that step already regressed.
Improvement must preserve visible appearance, raw and PNG outputs and each source.

Fresh R2 is sealed and verified, still UNRUN: canonical CPU byte values copied
to CUDA, an all256-byte pre-model test,100 controls/32 groups/320 approximate
gradient queries and zero updates/proposals/epochs/checkpoints. Scientific loss
definitions and gates, timing/storage stops and original states are unchanged.
The repaired CUDA execution needs the human manual run; no automatic launch or
more-epoch qualification is made. {link('Exact upload/tmux/download guide',guide_name)}.

Original checkpoint, splits, caches, migration backups, used packets, failures
and app design remain. All five milestones and seven covering families remain
required; native/unpaired and synthetic/paired evidence stay separate, final
identities stay outside tuning, and independent final/app review is pending.
The full goal remains active and incomplete. Historical exact bytes follow.

'''
    documents={name:finish_prepend(ROOT/name,prefix,backup/name,expected[name]) for name in docs}
    used_prefix=f'''# R1 completed with a retained conversion stop — do not rerun

The user returned R1. It stopped before gradients at the canonical CPU/CUDA PNG
conversion boundary. Zero updates or new structure evidence. Preserve this
packet; use the changed R2 only. {link('R1 findings',result_name)}.
{link('Fresh R2 manual guide',guide_name)}. Original commands follow as history.

'''
    documents[used]=finish_prepend(ROOT/used,used_prefix,backup/used,expected[used])
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=sha(archive),proof_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in proof},
        documents=documents,new_documents_sha256={result_name:sha(ROOT/result_name),guide_name:sha(ROOT/guide_name)},
        protected_app_files=protected,no_VM_connection_or_launch=True,manual_R2_unrun=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,results=result_name,manual_guide=guide_name,model_qualified=False))


if __name__=='__main__':main()
