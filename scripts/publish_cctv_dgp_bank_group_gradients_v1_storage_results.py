"""Publish verified storage recovery while retaining the failed packet and history."""
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_gradients_v1_return_r1 import ROOT, OUT, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link


def main():
    receipt=ANALYSIS/'publication.json';assert not receipt.exists()
    evidence=[ANALYSIS/'import.json',ANALYSIS/'failure_audit.json',
        OUT/'cctv_dgp_bank_group_gradients_v1_local_audit_repair_r1.json',
        OUT/'cctv_dgp_bank_group_gradients_v1_storage_cleanup/independent_audit.json',
        OUT/'cctv_dgp_bank_group_gradients_v1_r1_preparation.json',
        OUT/'cctv_dgp_bank_group_gradients_v1_r1_transfer_audit.json',
        OUT/'cctv_dgp_bank_group_gradients_v1_r1_bash_syntax.json']
    assert all(read(p)['complete'] for p in evidence)
    imported,failure,repair,cleanup,prep,transfer,bash=[read(p) for p in evidence]
    assert failure['scope']=='pre-model storage preflight stop' and failure['neural_gradient_optimizer_and_parameter_displacement_calls']==0
    assert cleanup['exact_deleted_archives']==1 and cleanup['free_GiB_live']>=4
    packet=OUT/prep['name'];p=read(packet/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256']==transfer['protocol_sha256']
    assert not (packet/'outputs').exists() and bash['exit_code']==0
    assert bash['checked_shell_sha256']==sha(packet/'scripts/run_groups.sh')
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert sha(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_r1_return.py')==transfer['future_auditor_sha256']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    report_name='CCTV_DGP_BANK_GROUP_GRADIENTS_V1_RESULTS.md'
    guide_name='CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_VM.md'
    report=f'''# Source/profile diagnostic storage stop and verified recovery

**The returned V1 packet stopped at its storage preflight.** It did not load
models, measure gradients, update weights or generate images. This return
provides no new structure, clarity or preservation result. The existing rejected
finite proposals remain rejected, and no new incumbent is qualified.

**Location:** the sealed worker's `run` function, line 89, before Torch import.
**Cause:** available VM space was below the unchanged **4 GiB** requirement.
The exact failed-run free byte count was not recorded. A fresh authorized VM
inventory measured **3.588 GiB** afterward; it is a later observation.
**Fix:** independently verified cleanup of one duplicate archive, followed by a
fresh R1 directory with explicit storage reporting and structured storage-stop
evidence. R1 keeps the measurement scope, quality requirements and every resource
limit. Its actual measurement remains **unrun**.

The return archive is **{imported['archive_bytes']:,} bytes**, SHA256
`{imported['archive_sha256']}`. All **10 returned file bindings**, **5,852 parent
hashes** and protected local app/checkpoint bindings were verified. The traceback
and exit code1 bind to the exact sealed source. There are zero neural forwards,
gradient queries, optimizer updates, parameter displacements or epochs in this
failed run, and no new checkpoint or image output. No visual review is reported
for outputs that do not exist.

The exception happened before the original worker's output creation and
`try`/failure logger. Consequently `failure_present:false` means `failure.json`
was absent. `complete_diagnostic:false` and the retained traceback establish the
stop. Successful export means the packet was packaged; it is not diagnostic or
training success.

{link('Independent retained-failure audit','outputs/cctv_dgp_bank_group_gradients_v1_analysis/failure_audit.json')}.

## Cleanup closure

The existing VM identity, L4 hardware, GPU activity, user processes, tmux panes,
disk and exact archive metadata were inventoried before mutation. The GPU was
idle and tmux panes were shells. No workload was stopped or launched.

Only `/home/janusdominic0/cctv-dgp-bank-photometric-guard-v1-results.tar.gz`
was removed: **{cleanup['full_local_backup_bytes']:,} logical bytes**,
**{cleanup['allocated_reclaim_bytes']:,} allocated bytes**
({cleanup['allocated_reclaim_bytes']/1024**3:.3f} GiB). Its complete local backup
was hashed before planning, immediately before deletion and again at closure.
The remote file was regular, owned by the intended user, singly linked and
matched the backup's hash, size and fixed inode/mtime metadata.

{link('Retained full local archive','outputs/cctv-dgp-bank-photometric-guard-v1-results.tar.gz')}.
Backup SHA256: `{cleanup['full_local_backup_sha256']}`.

Independent receipt checks verified the exact deletion ledger and absence of
the remote duplicate. **{cleanup['retained_research_metadata_entries']:,}
research namespace/metadata entries** and **{cleanup['retained_packet_byte_hashes']:,}
byte hashes** across the parent and failed diagnostic packets matched before and
after. **{cleanup['critical_byte_bindings_checked']:,}** critical hashes bind to
the locally retained packets. The active packets, trained/original checkpoints,
splits, scientific caches, provenance, logs and failed gates are retained. The
finalized export log is checked against live before/after bytes; it is not
substituted for the earlier archived empty export log.

The observed immediate free-space change was
**{cleanup['measured_free_change_bytes']/1024**3:.3f} GiB**. Receipt creation and
other filesystem writes affect observed free space, so it is distinct from exact
allocated deletion bytes. The final live audit measured
**{cleanup['free_GiB_live']:.3f} GiB free**. This is a snapshot; R1 checks current
space again before loading models. All scientific cache bytes were not rehashed:
full research metadata and the specified packet byte hashes were checked.

{link('Independent cleanup closure','outputs/cctv_dgp_bank_group_gradients_v1_storage_cleanup/independent_audit.json')}.
{link('Plan, backup proof, transport and returned receipts','outputs/cctv_dgp_bank_group_gradients_v1_storage_cleanup')}.

## R1 preparation and retained goal

The earlier local importer retained five stale folder/archive scope names from
its template. The earlier transfer check missed that inconsistency. Corrected
local importer/auditor copies retain their original sources and hashes, check
the real V1 scope and audit this pre-model traceback without inventing a
failure receipt. R1's transfer checker now also checks the prospective importer
name, archive stem, return directory and analysis directory explicitly.

R1 prints available/required storage in transfer verification and writes an
explicit pre-model `failure.json` with available/required bytes and zero calls
if storage falls below 4 GiB. It uses a fresh root and archive names; original
V1 records and the immutable used directory remain intact. This recovers an
unexecuted measurement after an infrastructure stop; it does not repeat an
unchanged failed learning recipe or lower a scientific gate.

R1 measures the same **20 photographic TRAIN references /100 cases**, **32
source/profile metric groups**, **320 gradient queries**, at most 40 B and 20
recognizer forwards, with zero optimizer updates, backward accumulation,
parameter displacements or epochs. Source names do not imply ethnicity. Native
and reserved final identities remain outside this diagnostic. Saved arithmetic
checks do not independently regenerate every neural derivative.

Upload **{prep['archive_bytes']:,} bytes**; require 4 GiB free after installation.
Estimated measurement/export time remains3–8 minutes and is not a newly measured
neural runtime. Worker/measurement/export caps remain900/600/300 s; external
worker/export930/330 s with 30s kill grace, VRAM 20GiB, RSS 12GiB, output 1.25GiB,
disk reserve 1GiB. Python3.10, asset/affine/scope/arithmetic checks, a local Windows
execution rejection before neural import and Bash syntax verification passed.

{link('Five exact manual recovery steps',guide_name)}.

Goal ACTIVE. Permanent baseline SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
and app design remain retained. The baseline is DeblurGAN-v2-compatible;
replacement B contains an externally trained frozen StyleGAN2 prior, not an
own-trained prior. This maintenance/import/preparation work does not qualify
new restoration quality, extra epochs, native CCTV performance or app adoption.

All five milestones and independent final/app review remain completion criteria.
Native unpaired evidence stays separate from paired synthetic metrics. The seven
completion families remain masks, sunglasses, strong glare, hands, obstructing
hair, scarves and objects, with automatic and assisted results reported
separately. Clear glasses, ordinary hair and visible appearance remain protected.
Preview/editable removal area, original/mask/one plausible estimate and downloads
remain required; insufficient crops need clearer or less-covered input. No exact
hidden identity, ethnicity or Zamboanga performance is claimed. Training,
gradients and displacements stay human-manual on the existing L4. Agent VM work
in this turn was only the authorized inventory/hash-bound maintenance. Full
browser-flow verification remains pending; the interrupted browser preflight is
not a new Playwright qualification.
'''
    pin=prep['protocol_sha256'];stem=prep['stem'];name=prep['name']
    guide=f'''# Storage-recovered source/profile diagnostic R1: five manual steps

**Prepared and independently transfer-checked; actual VM measurement unrun.**
V1 stopped before neural work because space was below 4 GiB. Authorized cleanup
is independently closed; final live free-space snapshot is
**{cleanup['free_GiB_live']:.2f} GiB**. R1 uses a fresh directory and records any
pre-model storage stop. It retains all measurement, timing and quality limits.

{link('Audited V1 cause and cleanup proof',report_name)}.

Upload **{prep['archive_bytes']:,} bytes** (about19.7 MiB). Require **4 GiB free
after installation** and an idle existing L4. Estimate **3–8 minutes** for the
new measurement/export; actual R1 neural runtime is unmeasured. Hard caps:
worker/measurement/export 900/600/300 s, external 930/330 s with 30s kill grace,
VRAM 20GiB, RSS 12GiB, outputs1.25GiB and disk reserve 1GiB. Approximately948MiB
of uncompressed arrays are planned. No optimizer updates or epochs are planned.

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
test ! -e ~/forensic-dgp/{name} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

The checksum is verified LF-only. Archive SHA256:
`{prep['archive_sha256']}`. A missing or differing parent stops verification;
no historical pilot is launched or silently reacquired.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_group_gradients_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{name} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_groups.sh {pin}
```

Transfer verification prints available and required free bytes and 32 groups /
320 planned queries, with zero actual neural/gradient calls in that check.
The worker checks current storage again, then measures20 batches. A successful
measurement/export prints `Diagnostic exit code:0` and `Export exit code:0`.
An export success alone does not prove measurement or training success. Detach
with **Ctrl+B**, release, then **D** to keep tmux running. Leave failed records
intact; do not relaunch a used directory unchanged.

5. Download from **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "{stem}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "{stem}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "{stem}-export.json"
```

Each download uses one remote source and an explicit local filename. Return
import/audit must check all saved controls, group scopes, gradient arithmetic,
states and stops before a new direction is derived. No native/final inputs,
app promotion or more-epoch qualification follows this diagnostic alone. The
permanent baseline, splits, caches, failures and seven-family completion scope
remain preserved; the full goal stays active.
'''
    new(ROOT/report_name,report);new(ROOT/guide_name,guide)
    prefix=f'''# Latest verified status: source/profile V1 stopped before neural work; storage recovered

The returned V1 packet failed its unchanged 4 GiB storage guard before model
loading, gradients, updates or image generation. All 10 return bindings and 5,852
parent hashes passed independent saved-evidence checks. The traceback is
retained; `failure_present:false` meant no structured failure file existed.
It did not mean success and provides no new image-quality result. The local
importer's stale template scope names were repaired in new checker copies;
R1 now also verifies future importer scope names explicitly.

Authorized inventory-first, hash-bound maintenance removed one exact completed
home archive after verifying its full local backup. Reclaimed allocated space:
{cleanup['allocated_reclaim_bytes']/1024**3:.3f} GiB; final live free snapshot:
{cleanup['free_GiB_live']:.3f} GiB. All {cleanup['retained_research_metadata_entries']:,} research
metadata entries and {cleanup['retained_packet_byte_hashes']:,} packet byte hashes agreed before/after.
Original checkpoints, research caches, splits, provenance, logs and gate failures
are retained. All cache bytes were not rehashed. No workload was stopped and
no model/training job was launched by the agent.

Fresh R1 transfer/geometry/arithmetic/Python/Bash checks passed. It reports
available/required storage and records an early structured storage failure;
the original used packet remains intact. Actual R1 measurement is UNRUN.
Same 20 paired photographic TRAIN references/100 cases, 32 groups/320 queries,
40 B/20 recognizer forward limits, zero optimizer updates/displacements/epochs.
Upload {prep['archive_bytes']:,}B; require 4 GiB free after install. Estimate 3–8min;
worker/measurement/export 900/600/300 s; external 930/330 s plus 30s grace;
VRAM 20GiB/RSS 12GiB/output 1.25GiB/reserve 1GiB. Current space is checked again.

{link('Audited failure and cleanup',report_name)}.
{link('Five exact new manual recovery steps',guide_name)}.

Goal ACTIVE. Permanent baseline SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
and app design remain preserved. Useful visible-structure/appearance and raw/PNG
qualification, all five milestones, independent final/app review and all seven
automatic/assisted covering families remain required. Native unpaired evidence
stays separate from paired synthetic metrics and reserved final identities.
Clear glasses, ordinary hair, visible appearance, editable removal preview,
original/mask/one plausible estimate/downloads and clearer or less-covered input
requests remain required. No exact hidden-identity, ethnicity or Zamboanga claim.
Training/gradients/displacements remain human-manual L4 work. Agent VM access
remains maintenance-only. Historical bytes below are retained and superseded.

'''
    previous=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/publication.json')
    names=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    expected={name:previous['documents'][name]['after_sha256'] for name in names}
    expected['CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md']=previous['new_documents_sha256']['CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md']
    assert all(sha(ROOT/name)==h for name,h in expected.items())
    backups=ANALYSIS/'publication_before';backups.mkdir()
    records={}
    for name,digest in expected.items():
        block=prefix if name in names else f'''# Superseded used V1 packet: pre-model storage stop retained

V1 stopped before neural measurement at the unchanged 4 GiB guard. Authorized
cleanup is independently closed; use the fresh prepared/unrun R1 manual guide.
{link('Audited V1 result and cleanup',report_name)}.
{link('Five R1 recovery steps',guide_name)}.
Original commands below are historical and must not be relaunched unchanged.

'''
        records[name]=prepend_expected(ROOT/name,block,backups/name,digest)
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={x.relative_to(ROOT).as_posix():sha(x) for x in evidence},documents=records,
        new_documents_sha256={report_name:sha(ROOT/report_name),guide_name:sha(ROOT/guide_name)},
        failure_classification='pre-model storage, zero neural/gradient/update calls',
        exact_archive_cleanup_independently_closed=True,fresh_manual_recovery_prepared_unrun=True,
        scientific_and_resource_limits_and_original_checkpoint_preserved=True,
        all_five_milestones_seven_families_and_independent_final_app_requirements_preserved=True,
        local_neural_gradient_optimizer_or_displacement_calls=0,agent_VM_maintenance_only=True,
        new_training_or_gradient_measurement_launched=False,more_epochs_qualified=False,
        model_qualified=False,goal_status='active',goal_complete=False))
    print(dict(complete=True,updated_documents=len(records),new_documents=2,
        free_GiB=cleanup['free_GiB_live'],manual_recovery_unrun=True,goal_status='active'),flush=True)


if __name__=='__main__':
    main()
