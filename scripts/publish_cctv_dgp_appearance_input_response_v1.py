"""Publish verified UNRUN manual commands and retain exact research history."""
from datetime import datetime,timezone

from import_cctv_dgp_appearance_input_response_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new,finish_prepend


def main():
    stage=OUT/'cctv_dgp_appearance_input_response_v1_publication';stage.mkdir(exist_ok=True);(stage/'before').mkdir(exist_ok=True)
    assert not (stage/'publication.json').exists()
    files=[OUT/f'cctv_dgp_appearance_input_response_v1_{n}.json' for n in ['preparation','transfer_audit','bash_syntax','batch_fixture',
        'contract_red','contract_green','guard_red','guard_green','archive_red','archive_green']]
    values=[read(n) for n in files];assert all(v['complete'] for v in values)
    prepared,check,bash=values[:3];packet=OUT/NAME;archive=OUT/(STEM+'-execution.tar.gz')
    assert prepared['protocol_sha256']==check['protocol_sha256']==PIN==sha(packet/'protocol.json')
    assert prepared['archive_sha256']==check['archive_sha256']==sha(archive) and bash['exit_code']==0
    assert bash['checked_shell_sha256']==sha(packet/'scripts/run_input_response.sh')
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    before=read(OUT/'cctv_dgp_conditioning_response_v1_analysis/publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
          'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected={n:before['documents'][n]['after_sha256'] for n in docs}
    expected['CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md']=before['new_documents_sha256']['CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md']
    for n,h in expected.items():assert sha(ROOT/n)==h or ((stage/'before'/n).exists() and sha(stage/'before'/n)==h),'Preserve concurrent document edits'
    guide_name='CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_VM.md';report_name='CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_PREPARATION.md'
    upload=f'''cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"'''
    install=f'''cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_conditioning_response_v1_vm &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp'''
    tmux='tmux new-session -A -s dgp_appearance_input_response_v1'
    launch=f'''cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_appearance_input_response_v1_vm.py --root . --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_input_response.sh {PIN}'''
    download='cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n'+'\n'.join(
        f'gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{n}" "."'
        for n in [STEM+'-results.tar.gz',STEM+'-results.tar.gz.sha256',STEM+'-export.json'])
    guide=f'''# Frozen appearance-input response V1 — manual L4 commands

**Verified locally and UNRUN on the VM.** This diagnoses why the predicted
appearance changes differ from actual outputs. It trains no weights and
generates no new face images. Preserve the completed conditioning-response run.

Require **3 GiB free after installation**. Estimated measurement **2–5 minutes**,
or **3–7 minutes with export**, based on the previous returned workload.
The enforced external maximum is **12 minutes 30 seconds**, including both
termination graces. The VM's current free space and CUDA execution have not
been measured by this local preparation.

Packet: **{prepared['archive_bytes']:,} bytes**, SHA256 `{prepared['archive_sha256']}`.
Protocol: `{PIN}`.
{link('Design, fixtures and exact limits',report_name)}.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
{upload}
```

2. Install in **VM SSH**:

```bash
{install}
```

3. Open **tmux**:

```bash
{tmux}
```

4. Launch **inside tmux**:

```bash
{launch}
```

The first receipt checks hashes only: 100 cases, at most 200 RGB-input
derivative queries and 40 recognizer forward batches; zero neural calls in
that receipt. The later measurement freezes every recognizer weight. Retain
any stop; do not delete outputs or repeat a used directory to bypass a guard.
The wrapper prints `review.log` when measurement ends. Detach with Ctrl+B,
release, then D. In a second VM SSH window, monitor with
`tail -F ~/forensic-dgp/{NAME}/review.log ~/forensic-dgp/{NAME}/export.log`.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:

```bat
{download}
```

Each download has one remote source, compatible with Windows PuTTY-backed SCP.
`complete: true` verifies export completion; the returned derivatives still
need independent audit. No new model, epoch continuation, native CCTV result
or covering-family result follows from this measurement alone.
'''
    report=f'''# Appearance-input response V1 preparation — 14 October 2026

**A new finite diagnostic is verified and UNRUN.** It localizes the demonstrated
appearance-response mismatch without applying another failed training direction.
The conditioning-response diagnostic completed, but none of its tested changes
qualified; all preservation failures and original parity misses remain binding.
{link('Audited prior result','CCTV_DGP_CONDITIONING_RESPONSE_V1_RESULTS.md')}.
{link('Evidence-based boundary review','CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md')}.

The same 100 already exposed paired photographic TRAIN cases and six saved
signed RAW outputs are reused. Fixed-recognizer derivatives are taken only with
respect to each initial RGB input tensor, twice in the same 20 five-case batch
order. All recognizer parameters remain frozen; no restorer or StyleGAN2 bank
is constructed. There are no optimizer updates, neural-parameter displacements,
new targets, new face images, rates, training epochs or candidate selection.

The prospective return stores 40 five-case RGB-gradient packs containing 200
per-case gradients, fresh embeddings, exact crop grids, scalar scores, actual
saved-RGB projections, batch/precision/RNG/state records and failure evidence.
Its independent checker compares 60 source/profile/pair/repeat pixel-space
responses with the prior parameter-space responses and actual scientific loss
changes. Per-reference records retain cancellation that source means can hide.

An input-space disagreement points to a limitation already present at the
fixed-recognizer finite input boundary. Agreement there with disagreement in
parameter space narrows attention to the earlier reconstruction-to-RGB path.
Neither outcome proves a unique wrong backward operator or every derivative
dimension; activation changes, finite intervals and numerical uncertainty still
need interpretation. No diagnostic pass qualifies restoration or more epochs.

Original RAW/PNG preservation, visible appearance, brightness, scientific
metric definitions, source/role separation and 1% global/per-source early
structure requirement remain. The 5% central comparison is a diagnostic bound,
not an image-quality gate. Prior elementwise `rtol=1e-4, atol=1e-8` and norm
`rtol=1e-4, atol=1e-10` bounds remain explicitly reported for repeated gradients.
The scalar alignment and independent replay bounds remain 1e-6 and 1e-13.

## Finite resource limits

| Limit | Value |
| --- | ---: |
| Measurement / worker / each derivative round | 420s / 450s / 180s |
| Export | 180s |
| External measurement / export / each kill grace | 480s / 210s / 30s |
| Free space after install / retained reserve | 3 GiB / 1 GiB |
| Output payload / peak VRAM / peak RSS | 512 MiB / 12 GiB / 8 GiB |

After the first two batches in each round, project `max(batch times) ×20×1.5`
and stop above 180s. External ceilings total 750s. Measurement counts stop at
200 input derivative queries and 40 recognizer forward batches; restorer
forwards, neural-parameter changes, optimizer/accepted updates and epochs stay
zero. No failed-run resume or automatic follow-on exists. Restore saved Torch
CPU/CUDA RNG and verify frozen recognizer state; retain all failure records.

The 2–5 minute measurement / 3–7 minute total estimates use the prior
366.59s/640-query/360-recognizer-batch workload as context. They are not a new
CUDA measurement, and the smaller component workload still needs manual timing.

## Preparation evidence and scope

All **17 archive bindings**, **1,246 stable response-source bindings** and
**5,852 bank-parent bindings** pass. The response VM's `export.log` grows after
the archived export log was hashed empty. It is retained and explicitly excluded
from live dependency hashes; the source export manifest and scientific outputs
stay bound. No checkpoint, split, cache, failure, provenance or log is removed.

The actual local hash-only CLI receipt agrees with protocol counts and imports
no neural model. Its available-space observation is Windows, not VM evidence.
ASCII/LF checksum and wrapper bytes pass; Bash `-n` parses without launching.
The real Windows measurement CLI rejects before neural loading or metadata
requests. Static checks restrict the sole autograd target to the RGB leaf and
reject optimizer/restorer construction. Python 3.10 syntax is checked.

Three expected RED/GREEN sequences cover image-displacement arithmetic,
manual-only execution and unsafe return paths. The arithmetic checks include
wrong sign, asymmetric endpoints, zero gradient and nine invalid inputs;
16 unsafe archive member sets are rejected. A separate saved five-case fixture
checks 30 projections and rejects a corrupted gradient file and incorrect dot
receipt, restoring its scratch bytes afterward. Its coefficients are explicitly
synthetic analytical arrays, not neural gradients. No local model forward,
autograd, optimizer, temporary neural-weight proposal or VM call occurred.
{link('Independent transfer audit','outputs/cctv_dgp_appearance_input_response_v1_transfer_audit.json')}.
{link('Saved-data fixture','outputs/cctv_dgp_appearance_input_response_v1_batch_fixture.json')}.
{link('Read-only Bash syntax receipt','outputs/cctv_dgp_appearance_input_response_v1_bash_syntax.json')}.

The strict return importer validates paths, regular-file types, case-insensitive
collisions, archive membership, sizes, all hashes, sealed protocol, dependencies,
space and protected model/app files before publication. It executes no returned
source. The prospective checker replays saved derivatives and embedding/score
arithmetic without neural re-execution locally. It reports parity misses rather
than weakening their bounds. Off-case zero gradients are VM-reported, not
independently recomputed through neural models on Windows.

No new visual sheets are generated: the same prior 20 sheets/1,000 cells were
already directly reviewed and did not show convincing new whole-face detail.
Paired TRAIN calibration stays separate from native unpaired CCTV, reserved
final identities and all seven completion families. Source labels are not
ethnicity or Zamboanga performance evidence. The fixed pretrained recognizer
is a measurement component, not an identity guarantee. The original app DGP
and design remain pinned and preserved; all five milestones and seven
automatic/assisted covering-family requirements still apply. The separate
detector-confidence work remains prepared only and UNRUN. Goal incomplete.
{link('Exact new manual commands',guide_name)}.
'''
    retained_new(ROOT/guide_name,guide);retained_new(ROOT/report_name,report)
    prefix=f'''# Appearance-input diagnostic verified, UNRUN — 14 October 2026

The prior response diagnostic completed but qualified no model. Its small RAW
detail prediction agrees while several appearance responses disagree; saved
double-dot reporting leaves those disagreements. A new bounded manual L4
measurement isolates the frozen recognizer's RGB-input response against the
same actual saved RAW changes before more training.
{link('Exact upload, tmux and single-source downloads',guide_name)}.
{link('Design, limits and preparation evidence',report_name)}.

Same100 exposed paired TRAIN cases;at most40 recognizer forward batches and200
RGB-input derivative queries. All neural weights stay frozen;zero restorer
forwards,weight changes,optimizer/accepted updates,epochs,new images or targets.
Require3GiB free after install with1GiB reserve;finite projection,measurement,
VRAM/RSS,output and export stops. External ceiling12min30s. Estimates2–5min
measurement/3–7min with export are not fresh CUDA evidence.

17 archive/1,246 stable response/5,852 bank bindings,real hash-only receipt,
manual guard,unsafe-return tests,saved five-case corruption fixture and read-only
Bash syntax pass. No new neural or VM run occurred. Original gates,failures,
checkpoints,splits,caches,provenance,app design and paired/native/final roles
remain. No new incumbent or more epochs qualified. All five milestones and
seven automatic/assisted covering families remain required. Detector-confidence
work stays prepared only and UNRUN. Goal incomplete. Exact history follows.

'''
    bindings={n:finish_prepend(ROOT/n,prefix,stage/'before'/n,h) for n,h in expected.items()}
    protected=read(OUT/'cctv_dgp_conditioning_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=prepared['archive_sha256'],documents=bindings,
        new_documents_sha256={n:sha(ROOT/n) for n in [guide_name,report_name]},
        exact_command_blocks=dict(upload=upload,install=install,tmux=tmux,launch=launch,download=download),
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},
        manual_diagnostic_unrun=True,automatic_VM_launch=False,local_neural_gradient_or_training_calls=0,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,new_documents=2,history_documents=len(bindings),manual_UNRUN=True))


if __name__=='__main__':main()
