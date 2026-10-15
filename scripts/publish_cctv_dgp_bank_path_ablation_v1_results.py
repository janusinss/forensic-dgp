"""Publish verified returned diagnosis and a separately unrun manual packet."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from publish_cctv_dgp_bank_AB_and_guard_v1_review import prepend_locked

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
ANALYSIS=OUT/'cctv_dgp_bank_path_ablation_v1_analysis'
PARTITION=OUT/'cctv_dgp_bank_partition_v1_analysis'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def new(path,text):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


def link(label,name):return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def main():
    receipt=ANALYSIS/'publication.json';assert not receipt.exists()
    evidence=[ANALYSIS/n for n in ['import.json','independent_audit.json','limited_frozen_replay.json','visual_review.json','partition_state_audit.json']]
    evidence += [PARTITION/n for n in ['run_results.json','independent_audit.json','visual_review.json']]
    evidence += [OUT/n for n in ['cctv_dgp_bank_loss_alignment_v1_math_audit.json','cctv_dgp_bank_loss_alignment_v1_preparation.json','cctv_dgp_bank_loss_alignment_v1_transfer_audit.json']]
    assert all(read(p)['complete'] for p in evidence)
    imported,audit,visual=[read(ANALYSIS/n) for n in ['import.json','independent_audit.json','visual_review.json']]
    counter=read(PARTITION/'independent_audit.json');counter_visual=read(PARTITION/'visual_review.json')
    prep,transfer=read(evidence[-2]),read(evidence[-1]);math=read(evidence[-3])
    assert visual['viewed_sheets']==counter_visual['viewed_sheets']==26
    assert visual['sheet_cells_independently_verified']==1216 and counter_visual['sheet_cells_independently_verified']==720
    assert audit['saved_paired_raw_PNG_compositions_checked']==700 and audit['native_compositions_checked']==168
    assert counter['new_paired_raw_PNG_cases_independently_checked']==100 and counter['new_native_compositions_checked']==24
    assert sha(OUT/(prep['stem']+'-execution.tar.gz'))==prep['archive_sha256']
    assert transfer['preparation_sha256']==sha(evidence[-2]) and transfer['protocol_sha256']==prep['protocol_sha256']
    assert not (OUT/prep['name']/'outputs').exists()
    assert not any(g['raw']['pass'] or g['png']['pass'] for g in audit['partial_preview_gate_diagnostics'].values())
    assert not any(g['raw']['pass'] or g['png']['pass'] for g in counter['partial_preview_gate_diagnostics'].values())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    report_name='CCTV_DGP_BANK_PATH_ABLATION_V1_RESULTS.md';guide_name='CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_VM.md'
    rows=[]
    for label,gates in audit['partial_preview_gate_diagnostics'].items():
        a,b=gates['raw'],gates['png'];rows.append(f"| {label} | {100*a['gain']:.6f}% / {100*b['gain']:.6f}% | {len(a['failures'])} / {len(b['failures'])} | {100*a['brightness']:.6f}% / {100*b['brightness']:.6f}% |")
    counter_rows=[]
    for label,gates in counter['partial_preview_gate_diagnostics'].items():
        a,b=gates['raw'],gates['png'];counter_rows.append(f"| {label} | {100*a['gain']:.6f}% / {100*b['gain']:.6f}% | {len(a['failures'])} / {len(b['failures'])} |")
    run=read(PARTITION/'run_results.json');drop=audit['variant_groups']['drop128']['raw']['clear']
    enabled=audit['variant_groups']['enabled']['raw']['clear']
    detail_tradeoff=100*(drop['landmark_high_frequency_MSE']/enabled['landmark_high_frequency_MSE']-1)
    stem,pin=prep['stem'],prep['protocol_sha256']
    report=f"""# Frozen bank-path return and training diagnosis — 11 October 2026

The downloaded diagnostic **completed successfully**. Its saved outputs were
independently checked, but **none of the seven variants qualifies as a model**.
Removing the 128-scale readout produces the strongest preview detail score and
still misses detail and preservation requirements. More epochs are not yet justified.

## Returned run and independent audit

The L4 diagnostic took **{audit['VM_seconds']:.2f} seconds** (about 2 minutes43 seconds)
and used **{audit['VM_peak_VRAM']/(1024**3):.3f} GiB peak allocated VRAM**.
It made 312 B forwards and 140 fixed-recognizer forwards, with **zero gradients,
backwards, optimizer updates and epochs**. Archive export took 24.86 seconds.
The {imported['archive_bytes']:,}-byte archive has SHA256
`{imported['archive_sha256']}`. All **1,192 returned files** and
**5,852 retained-parent files** were hash checked. Export success is packaging
evidence and is separate from useful restoration.

Independent saved-array checks covered **700 paired raw/PNG compositions**,
100 baseline cases and **168 native compositions**. The enabled and disabled
controls replayed **200 paired RGB/embedding cases and 48 native raw/PNG outputs
exactly** against the parent L4 run. Zeroing all five readouts equaled disabling
the bank exactly on two source batches. A separate frozen CPU replay covered
every variant on those two batches, **70 compositions**; CPU/CUDA tolerances
apply only to that numerical replay, never to scientific preservation gates.
The rejected B50 state and all buffers remained unchanged. No checkpoint was created.

All **26 planned sheets** were displayed at original resolution, with actual
observations and **1,216 independently checked source cells**. Clearer native
faces retain texture better than original DGP, consistent with the earlier user
feedback. Soft, dark and turned faces still lack eye, nose and mouth detail.
Individual readout removals mostly produce subtle tone changes. No single removal
demonstrates convincing whole-face clarity improvement. Clear glasses and ordinary
visible hair remain; completion was not tested. The implementing assistant's
review is not independent final review. Input-only usable labels stay fixed;
model softness cannot remove a usable DEV case from the review.

## Seven fixed readout comparisons

These are the **same 100 paired photographic TRAIN previews** (20 references,
ten per source, five profiles each), not the full 3,905-case pool. Nine of the
twenty references were exposed in the earlier 50-update run. Unexposed TRAIN
references remain TRAIN evidence. The detail metric is **landmark high-frequency
MSE**, not recognizer feature MSE or recovered identity. Failures overlap groups
and metrics, and are not counts of failed people.

| Frozen variant | Raw / PNG detail error reduction | Raw / PNG group-metric failures | Raw / PNG brightness-only contribution |
|---|---:|---:|---:|
{chr(10).join(rows)}

All fail the unchanged partial preview criteria. The early detail requirement is
**1%**, with a **20% maximum** brightness-only contribution and retained source/
profile preservation requirements. Drop128 reaches **0.739311% raw / 0.746189%
PNG**, but brightness contribution remains **23.166817% / 22.335946%** and six
overlapping group-metric failures remain. Clear detail error worsens by about
**{detail_tradeoff:.2f}%** relative to enabled, with worse clear MSE, SSIM and
similarity. Dropping this path is not selected for app adoption or training.
The older parent's full-pool gain of 0.680385% uses a different evaluation scope;
it must not be confused with these preview-subset scores.

## Isolating the learned decoder and bank parameters

Path removal alone did not identify a qualified architecture. A fixed-weight
cross-composition separated the **15 learned current-decoder tensors** from the
**22 learned style/spatial/fusion tensors**; all **755 frozen tensors** remained
exact. It uses the pinned initial B state and rejected B50 state, with no learning.
Initial equals the retained baseline, joint equals the saved enabled model, and
trained-decoder/initial-bank equals the already audited disabled control.
Only the missing **initial-decoder/trained-bank** composition was computed locally.

Frozen CPU inference took **{run['seconds']:.2f} seconds**, with 50 B forwards,
20 recognizer forwards and zero gradients or updates. Six fresh control batches
cover ten cases per initial/joint/decoder control across the two sources, bounding
CPU/CUDA raw differences at 1e-4; this is not full fresh control replay. All 100
fixed target embedding rows stayed identical to the controls. The new bank-only
outputs are CPU inference while saved controls are L4 output, with that limited
parity evidence declared. Saved pixel/PNG/metric checks covered all **100 new
paired cases and 24 native cases**. All **26 cross-composition sheets** were viewed
and **720 cells** independently checked.

| Frozen cross-composition | Raw / PNG detail error reduction | Raw / PNG group-metric failures | Raw / PNG brightness-only contribution |
|---|---:|---:|---:|
{chr(10).join(counter_rows)}

Bank-only gives **0.169222% / 0.182174%** detail gain and **20 / 19** overlapping
failures. Its pooled degraded MSE worsens from 0.027559660 to 0.028247220; its
zero brightness fraction is not success when total MSE does not improve.
Decoder-only reaches **0.716408% / 0.739873%**, but brightness contributes
**76.247268% / 76.646960%** and ten failures remain. The bank helps some clear
preservation when combined with decoder learning, but the full learned combination
still misses the retained requirements. Visually, isolating bank changes follows
the original smoothing, while joint/decoder retain more clear texture; blurred
features remain soft. These are fixed-weight counterfactuals, **not independently
trained decoder-only or bank-only experiments**, and do not prove either path's
eventual training capacity.

## Selected next diagnostic before new training

The evidence supports reviewing **loss directions and their interaction** before
changing another architecture or allowing extra epochs. Large combined gradients
and changed weights do not establish useful detail learning. The exact cause is
not proven by these output comparisons. Freeze a manual L4 diagnostic on the same
20 TRAIN references at the **initial and joint B50 states**, with no DEV/final
model inputs, no parameter updates and no optimizer construction.

It separately measures the seven retained objective components (MSE, SSIM,
recognizer, detail, anchors, preservation barrier and brightness guard), plus
clear and degraded MSE reference gradients. It records norms and pairwise products
for the decoder, style, spatial and five fusion scales. A mathematical direction
uses the retained Adam/clipping/weight-decay configuration and saved moments
without mutating them. It is a **hypothetical one-reference first-order direction**,
not the actual accumulated five-reference optimizer update, and cannot prove
finite-step image quality or convergence. Diagnostic use of rejected B50 gradients
is not failed-state training continuation.

Finite scope: **360 gradient queries**, 80 model/40 recognizer forwards, zero
backwards/optimizer updates/epochs, 20 TRAIN references, 100 distinct TRAIN cases,
two measured states, no native/final pixels and no checkpoint. Retain full gradient
arrays and mathematical directions for **four reference-state batches**, one per
source per state, so those summaries can be independently recomputed; the other
36 summaries have a narrower audit scope. Do not claim all 40 raw gradients were
independently replayed. No automatic follow-on training or hyperparameter search.

The **{prep['archive_bytes']:,}-byte packet** passed independent source, archive,
all 5,852 parent hashes, LF checksum, readonly Bash syntax, Python3.10 hashing
compatibility and actual Windows rejection before neural/gradient/output work.
**100 cached metric rows** passed scalar loss-arithmetic checks; fixed synthetic
arrays passed independent NumPy direction/Gram checks. These are arithmetic checks,
not actual L4 gradient verification. Sampled raw-gradient upper bound is
**{math['uncompressed_sampled_gradient_upper_bound_bytes']:,} bytes**, including
metadata reserve. Actual CUDA gradients, runtime and results remain **unrun**.

Require **4 GiB free after installation**, idle L4 and manual tmux. Unmeasured
estimate: **3–10 minutes measurement plus 1–5 minutes export**. Worker cap15 minutes,
measurement stop10 minutes checked between reference batches, export5 minutes;
external caps15.5/5.5 minutes with30-second kill grace. Projection uses the first
two batches with1.5 safety factor. VRAM20 GiB; output/reserve1 GiB each. No current
VM free space, power state or cloud balance is claimed.

{link('Five exact manual upload/tmux/download steps',guide_name)}.
{link('Returned path audit','outputs/cctv_dgp_bank_path_ablation_v1_analysis/independent_audit.json')};
{link('Path visual closure','outputs/cctv_dgp_bank_path_ablation_v1_analysis/visual_review.json')};
{link('Cross-composition audit','outputs/cctv_dgp_bank_partition_v1_analysis/independent_audit.json')};
{link('Cross-composition visual closure','outputs/cctv_dgp_bank_partition_v1_analysis/visual_review.json')};
{link('Prepared transfer audit','outputs/cctv_dgp_bank_loss_alignment_v1_transfer_audit.json')}.

## Goal and evidence boundaries

The current app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
It is the permanent fine-tuned DeblurGAN-v2-compatible baseline. The custom bank
replacement learns our conditioning/reconstruction with a declared external frozen
StyleGAN2 bank; it is not an own-trained StyleGAN or an exact published-DGP replica.
No incumbent is accepted here. Extra epochs and failed-state continuation remain
unqualified. No app/model binding, checkpoint, research cache, split or historical
stop record changed. No agent VM connection, cleanup, training, gradient or power
action occurred; local frozen inference and audits are allowed.

Native ChokePoint DEV remains unpaired, with no native clean-reference PSNR/SSIM.
Paired photographic TRAIN measurements stay separate. Source labels do not establish
ethnicity or Zamboanga performance; no real Zamboanga samples exist. Final identities
remain outside tuning, and independent final review is pending. All five milestones
remain required, including useful reviewed outputs, retained baseline/pretrained/
resize comparisons, qualified primary DGP Auto/override integration and bundled
inline Playwright app verification preserving the existing design.

Completion remains separate for masks, sunglasses, strong glare, hands, obstructing
hair, scarves and objects. Require automatic removal-area preview/optional correction,
preserved clear glasses/ordinary hair/visible appearance, a documented small margin,
original/mask/one plausible estimate, PNG and optional bundle downloads, clearer or
less-covered requests when information is insufficient and separate automatic versus
assisted review. None of these seven covering families is qualified by this work.
Goal **ACTIVE and incomplete**. Reported credits/migration notice and November2026
deadline remain governed by the canonical plan; no live financial claim is added.
"""
    guide=f"""# Loss-direction diagnostic: exact manual VM commands

**Prepared and independently transfer-verified; actual L4 gradients are unrun.**
This is diagnosis before new training: **zero optimizer updates and no new model**.
It compares fixed initial and rejected B50 states; it does not resume the failed run.
{link('Audited results and rationale',report_name)}.

Upload only **{prep['archive_bytes']:,} bytes**. Preserve
`~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm` and its 5,852 bound assets.
Require **4 GiB free after installation** and an idle L4. Estimate **3–10 minutes
measurement plus1–5 minutes export**, unmeasured. Worker15 minutes, measurement
stop10 checked between reference batches, export5; external15.5/5.5 with30-second
kill grace. VRAM20 GiB, output/reserve1 GiB each. Raw sampled-gradient storage
bound530 MiB. Launch checks exact source/data/state hashes before model work.

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

Expected archive SHA256: `{prep['archive_sha256']}`.
Expected protocol SHA256: `{pin}`.
Installation deletes nothing. Keep any stopped run and failure evidence.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_loss_alignment_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_alignment.sh {pin}
```

Expected transfer receipt: `complete: true`, `packet_assets: 3`,
`parent_assets: 5852`, `neural_calls: 0`, `gradient_queries: 0`,
`optimizer_updates: 0`. The diagnostic prints one line per reference/state.
Successful measurement finishes with360 gradient queries and **zero optimizer
updates**. Detach safely with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_bank_loss_alignment_v1`. Export `complete` confirms packaging
only. The shell attempts to export a diagnostic failure too; no automatic training.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "."
```

Keep the three files together. They will be independently hash checked and the
retained gradient samples recomputed before selecting a new finite training recipe.
Model selection, extra epochs, final identities, native unpaired evidence and all
seven automatic/assisted completion families remain separate and unqualified here.
"""
    new(ROOT/report_name,report);new(ROOT/guide_name,guide)
    status=f"""# Latest verified status — 11 October 2026: frozen path return and parameter separation audited; loss-direction diagnostic prepared

The returned frozen path diagnostic completed in162.77 seconds, with zero
gradients, updates and epochs. All1,192 returned files and5,852 parent hashes,
700 paired raw/PNG and168 native compositions passed independent checks.
Enabled/disabled controls replayed exactly;70 compositions passed limited frozen
CPU replay. All26 planned sheets and1,216 cells were reviewed/checked.
**No readout variant qualifies.** Drop128 reaches0.739311% raw /0.746189% PNG
detail gain, below1%, with23.166817% /22.335946% brightness contribution above20%,
six overlapping failures and about{detail_tradeoff:.2f}% worse clear detail than enabled.

A frozen parameter cross-composition retained755 frozen tensors and separated15
decoder from22 bank tensors. Three controls reused audited outputs; the missing
bank-only composition ran locally in193.93 seconds with zero gradients/updates.
All100 new paired and24 native outputs were independently checked; all26 new
sheets and720 cells were reviewed/checked. Bank-only detail gain0.169222% /0.182174%
has20 /19 failures and worsened degraded MSE. Decoder-only gain0.716408% /0.739873%
has ten failures and76.247268% /76.646960% brightness contribution. These are fixed
weight counterfactuals, not independently trained controls or full3,905-case gates.
No new incumbent, extra epochs, failed-state resume or app promotion is qualified.

Next: **manual L4 loss-direction diagnostic**, at initial and rejected B50 states
on20 TRAIN references:360 gradient queries, zero optimizer updates/backwards/epochs,
80 model/40 recognizer forwards, no native/final input pixels and no checkpoint.
Sampled full gradients cover four reference-state batches; hypothetical Adam
directions are not actual accumulated updates or finite-step quality proof.
The {prep['archive_bytes']:,}-byte transfer passed source/archive/LF/Python310,
5,852 parent hashes, readonly Bash and Windows pre-neural rejection checks.
100 cached scalar loss rows and synthetic direction-array math passed; actual
CUDA gradients, timing and results are **unrun**. Require4 GiB free; worker/
measurement/export stops15/10/5 minutes, external15.5/5.5 with30-second kill grace,
VRAM20 GiB, output/reserve1 GiB. Measurement stop is checked between batches.
Manual tmux only. No agent VM, cleanup, training, gradient or power action occurred.
{link('Verified return and diagnosis',report_name)}.
{link('Five exact manual transfer/tmux/download steps',guide_name)}.

Goal ACTIVE. Permanent app checkpoint
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`,
existing design, five milestones and seven covering families remain. Native CCTV
is unpaired; paired TRAIN, final identities and completion evidence stay separate.
No ethnicity, Zamboanga performance or exact hidden-identity claim. Input-only
usability labels remain; model softness cannot exclude usable DEV inputs.
Historical sections below retain exact bytes and authorize no reruns.

---

"""
    history=f"""# Executed and audited — 11 October 2026

This historical frozen diagnostic completed successfully and all returned outputs
were independently audited and viewed. No variant qualified; **do not rerun it**.
Its original prepared commands below are retained as exact historical bytes.
{link('Current evidence and diagnosis',report_name)}.
{link('New unrun manual loss-direction diagnostic',guide_name)}.

---

"""
    backups=ANALYSIS/'publication_before';backups.mkdir()
    documents={}
    for name in ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        documents[name]=prepend_locked(ROOT/name,status,backups/name)
    documents['CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md']=prepend_locked(ROOT/'CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md',history,backups/'CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md')
    for name,expected in protected.items():assert sha(ROOT/name)==expected
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in evidence},documents=documents,
        new_documents_sha256={n:sha(ROOT/n) for n in [report_name,guide_name]},
        returned_diagnostic_executed_and_reviewed=True,next_manual_gradient_diagnostic_unrun=True,
        protected_app_checkpoint_and_examples_unchanged=True,no_VM_or_local_optimizer_execution=True,
        all_five_milestones_and_seven_covering_families_retained=True,goal_status='active',goal_complete=False)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,status_documents=5,new_documents=2,history_bytes_preserved=True,manual_VM_run_pending=True))


if __name__=='__main__':main()
