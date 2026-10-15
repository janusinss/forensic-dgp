"""Publish the verified, unrun diagnostic and exact human-manual VM commands."""
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_response_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new,finish_prepend


def main():
    stage=OUT/'cctv_dgp_conditioning_response_v1_publication';stage.mkdir(exist_ok=True)
    assert not (stage/'publication.json').exists()
    evidence=['preparation.json','transfer_audit.json','bash_syntax.json','contract_green.json','guard_green.json','archive_members_green.json','snapshot_fixture.json']
    values=[read(OUT/('cctv_dgp_conditioning_response_v1_'+n)) for n in evidence];assert all(v['complete'] for v in values)
    prep,transfer,bash=values[:3];packet=OUT/NAME;p=read(packet/'protocol.json')
    assert PIN==prep['protocol_sha256']==transfer['protocol_sha256']==sha(packet/'protocol.json')
    assert sha(OUT/(STEM+'-execution.tar.gz'))==prep['archive_sha256']==transfer['archive_sha256']
    assert bash['exit_code']==0 and bash['checked_shell_sha256']==sha(packet/'scripts/run_response.sh')
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    previous=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_analysis/publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    review='CCTV_DGP_POST_RAW_CONDITIONING_REVIEW_V1.md'
    expected={n:previous['documents'][n]['after_sha256'] for n in docs};expected[review]=previous['new_documents_sha256'][review]
    for n,h in expected.items():assert sha(ROOT/n)==h or ((stage/'before'/n).exists() and sha(stage/'before'/n)==h),'Preserve concurrent status edits'
    guide_name='CCTV_DGP_CONDITIONING_RESPONSE_V1_VM.md';report_name='CCTV_DGP_CONDITIONING_RESPONSE_V1_PREPARATION.md'
    guide=f'''# Manual conditioning-response diagnostic V1 — 14 October 2026

**Verified and UNRUN.** This measures why predicted learning differs from real
image changes. It accepts no training updates and selects no checkpoint.
The previous RAW R1 pilot remains failed. New neural work stays human-manual
inside tmux on the existing NVIDIA L4 VM.

Require **9 GiB free after installation**. Estimated diagnostic **6–12 minutes**,
or **8–15 minutes including export**; those are estimates from prior measured
work. External stop limits total **29.5 minutes including both termination
graces**. Storage, VRAM, RSS, projection and output stops also apply.
Packet: **{prep['archive_bytes']:,} bytes**, SHA256 `{prep['archive_sha256']}`.
Protocol: `{PIN}`.

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
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_conditioning_response_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_conditioning_response_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_response.sh {PIN}
```

The first receipt is hash-only: 100 cases, maximum640 derivative queries,
six temporary displacements, zero optimizer updates and zero neural calls in
that receipt. The wrapper writes progress into `review.log`; its full log prints
when the diagnostic ends. In a second VM SSH window, monitor with
`tail -F ~/forensic-dgp/{NAME}/review.log ~/forensic-dgp/{NAME}/export.log`.
Stop that monitor with Ctrl+C. Detach tmux with Ctrl+B, release, then D.
Do not rerun a used directory or remove a failure to get past the guard.

5. Download from **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each download has one remote source, as required by PuTTY-backed gcloud SCP.
Keep all three files together in `outputs`. Return them for the independently
prepared audit. `Diagnostic exit code: 0` means the measurements finished;
`complete: true` in export means the archive exists. Neither is a model-quality
or epoch-training success. All complete outputs and all planned sheets need
review before any next learning decision.

Original checkpoints, app design, splits, provenance, caches, full states and
failed gates remain. This is photographic paired TRAIN evidence only. Native
CCTV, final identities and seven automatic/assisted completion families remain
separate and unqualified by this diagnostic.
{link('Design, evidence and exact limits',report_name)}.
'''
    report=f'''# Conditioning-response V1 preparation — 14 October 2026

**Verified and UNRUN; no new trained image or accepted checkpoint exists.**
RAW R1's three proposals remain rejected at best0.669337% RAW/0.682289% PNG,
with32/32 best-trial preservation failures and zero accepted updates. Its one
cross-run gradient mismatch also remains failed. This packet measures finite
response and repeatability; it does not resume or re-run that optimizer.
{link('Retained RAW R1 evidence','CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_RESULTS.md')}.

## What the measurement resolves

Repeat all32 RAW loss derivatives over the same20 exposed photographic TRAIN
references/100 prepared256 cases twice. Record backend/precision, source/batch
order, original output parity, individual scalar values, contributions, vector
norms, saved arrays, directional dots and RNG state. Use the exact previously
executed failed direction, copied by hash, without a new solve or optimizer.

Then measure six fixed, signed changes: plus/minus the prior smallest trial
magnitude `{p['base_signed_magnitude']}`, plus/minus one tenth and one hundredth
of it. Every state starts from the exact initial component. All stay within
the original cumulative0.1% parameter-distance bound. Restore the original
weights and RNG after the last response; verify the full state and original
raw/PNG outputs again. On a handled failure, retain fresh rollback facts where
available rather than asserting a rollback that was not observed.

The returned audit compares the actual rounded parameter differences with the
saved repeated vectors and scientific source/profile losses. Central-response
agreement has an explicitly diagnostic5% comparison fraction with a measured
repeat/alignment resolution floor. Unresolved signals remain unresolved.
This is not an image preservation or model acceptance threshold. SSIM's
float64 backward remains explicitly approximate. Cached baseline scientific
appearance scores are kept; fresh recognizer scores are saved separately.
No identity recovery or all-dimensional derivative validation is claimed.

Both17-group RAW/PNG preservation checks, MSE/SSIM/fixed-recognizer tolerances,
brightness<=0.2 and the later1% global/per-source structure gate remain intact.
The six measurements are retained even if they fail those gates; none is
selected or accepted for training/app use. The prospective audit retains old
cross-run bounds rather than forcing repeatability to pass. No untested rate,
more epochs, random reset or new incumbent follows automatically.

## Finite protocol and component origins

Maximum640 derivative queries,200 restoration-forward batches,360 recognizer
forward batches and six temporary component displacements. Zero optimizer
updates, accepted updates or full epochs. Eight complete100-case snapshots
produce800 RAW outputs/800 PNGs/1,600 stage metric rows. The independent gallery
has20 sheets and1,000 cells; every sheet must be directly viewed after return.
No native CCTV/final-identity pixels or completion generation are used.

The same12 style/spatial tensors contain2,580,864 parameters. The permanent
original DGP decoder/encoder, feature readouts, same-observed-input anchor,
pretrained StyleGAN2 bank, fixed recognizer and buffers stay frozen. The bank
remains attributed and is not our own trained generative prior. Initial state
and scientific controls match the last diagnostic. This is not additional
epochs of the app restorer. Component origins and exact hashes are in protocol.

Worker1200s, internal measurement1170s, gradient round600s with a first-two-batch
projection and safety factor1.5, export450s. External bounds are1230s/480s plus
30s termination grace each, totaling29.5 minutes. Require9GiB free after install;
reserve1GiB, cap outputs at3.5GiB, peak VRAM20GiB and RSS12GiB. A whole export
must fit along with the retained output tree; no automatic deletions or resume.
Estimate6–12 minutes diagnostic/8–15 with export uses the measured168.55s prior
work scaled to two repeats and eight snapshots; new CUDA timing is unmeasured.

## Preparation and evidence boundaries

All20 archive bindings and5,852 required parent-file bindings pass. The archive
is{prep['archive_bytes']:,} bytes, SHA256 `{prep['archive_sha256']}`. Protocol
SHA256 `{PIN}`. ASCII/LF checksum, matching upload/export/import namespaces,
actual hash-only CLI receipt, fixed counts, source origins and finite stops are
checked. Windows execution rejects before Torch or protocol reads. Git Bash
`-n` passes. No agent VM action, local network inference, gradient, model-weight
proposal or optimizer update occurred.

Six declared synthetic scalar fixtures cover correct/rounded predictions,
curvature, wrong sign, unresolved noise/stationarity; invalid NaN/Inf/magnitude/
noise inputs are rejected. Tar boundary tests reject15 unsafe namespaces.
The returned-array checker replays100 retained real RAW/PNG cases as an explicit
extended-receipt fixture and rejects an altered PNG; it creates no new inference.

Two preparatory checker stops are retained. The image fixture initially required
bitwise-equal independently replayed group floats after the scientific replay
had passed. Its corrected fixture uses the original1e-13 replay bound and records
every actual difference. The transfer checker attempted JSON parsing of the
worker's Python-literal receipt; a separate R1 checker safely uses
`ast.literal_eval`. Neither changes the sealed packet, worker, protocol,
scientific gates or any failed model qualification.
{link('Prepared transfer audit','outputs/cctv_dgp_conditioning_response_v1_transfer_audit.json')}.
{link('Original snapshot checker stop','outputs/cctv_dgp_conditioning_response_v1_snapshot_fixture_stop/stop.json')}.
{link('Original receipt checker stop','outputs/cctv_dgp_conditioning_response_v1_preparation_checker_stop/stop.json')}.

Future returned gradients/embeddings are checked as saved arithmetic and stated
manual-worker evidence; no independent neural replay is pretended locally.
Full TRAIN/native DEV, independent final review and all five milestones remain.
Completion has seven separate automatic/assisted families and remains pending;
the local detector-confidence recorder is prepared only and UNRUN. The DGP-led
app, preservation, less-covered/clearer-input refusal and downloads still need
qualified useful outputs. Goal incomplete.
{link('Exact Windows upload, tmux launch and downloads',guide_name)}.
'''
    retained_new(ROOT/guide_name,guide);retained_new(ROOT/report_name,report)
    prefix=f'''# Finite conditioning-response diagnostic verified, UNRUN — 14 October 2026

RAW R1's failure remains binding: best0.669337% RAW/0.682289% PNG, all three
proposals rejected, zero accepted updates and retained cross-run parity failure.
The next verified packet measures the current path's actual response before
another optimizer or epoch schedule. It repeats32 RAW gradient groups twice,
then measures six signed changes along the same exact saved failure direction.
No optimizer,solve,rate search,candidate selection or checkpoint adoption.
{link('Exact manual upload, tmux and single-source downloads',guide_name)}.

Same100 paired TRAIN cases/12 conditioning tensors/initial state/frozen origins
and all original RAW/PNG quality requirements. Maximum640 derivative queries,
200 restorer/360 recognizer forwards,six temporary displacements;zero accepted
updates/epochs. Eight snapshots/800 RAW/800 PNG/1,600 metric rows and20 planned
sheets/1,000 cells. Restore initial weights and RNG; retain all failed evidence.
{link('Design, evidence and finite resource limits',report_name)}.

Require9GiB free after install,1GiB reserve,3.5GiB output cap and finite timing/
projection/VRAM/RSS/export stops. Estimate6–12 minutes measurement/8–15 with
export;external limits total29.5 minutes.20 archive/5,852 parent bindings,
real100-case saved-output fixtures,unsafe-return tests,Windows manual guard,
actual hash-only receipt and read-only Bash syntax pass. Preparatory checker
stops remain preserved;no scientific gate or failed qualification is relaxed.

UNRUN means no new CUDA result or improved image is established. Actual neural
measurement stays human-manual L4 tmux;no agent VM launch or local training.
Original model/app/checkpoints/splits/caches/provenance/full states/failures,
native unpaired/paired TRAIN/final roles,all five milestones and seven automatic/
assisted covering families remain. Goal incomplete. Separate detector-confidence
recorder remains prepared only and UNRUN. Exact previous document follows.

'''
    (stage/'before').mkdir(exist_ok=True);records={}
    for n in docs+[review]:records[n]=finish_prepend(ROOT/n,prefix,stage/'before'/n,expected[n])
    hashes={('outputs/cctv_dgp_conditioning_response_v1_'+n):sha(OUT/('cctv_dgp_conditioning_response_v1_'+n)) for n in evidence}
    for n in ['outputs/cctv_dgp_conditioning_response_v1_snapshot_fixture_stop/stop.json','outputs/cctv_dgp_conditioning_response_v1_preparation_checker_stop/stop.json']:
        hashes[n]=sha(ROOT/n)
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=prep['archive_sha256'],documents=records,
        new_documents_sha256={n:sha(ROOT/n) for n in [guide_name,report_name]},evidence_sha256=hashes,
        archive_bindings=20,parent_bindings=5852,manual_diagnostic_unrun=True,automatic_VM_launch=False,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,guide=guide_name,history_documents=6,manual_unrun=True,protocol_sha256=PIN))


if __name__=='__main__':main()
