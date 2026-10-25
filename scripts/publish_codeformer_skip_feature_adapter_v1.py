"""Publish only a sealed and independently reviewed manual-pilot preparation."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/codeformer_skip_feature_adapter_v1_preparation'
PUB=PREP/'publication_v1'
DOCS=('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
GUIDE=ROOT/'CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_VM.md'
REPORT=ROOT/'CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_PREPARATION.md'


def pin(p):
    p=Path(p)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'bytes':p.stat().st_size,'sha256':sha}


def new(p,b):
    assert p.resolve(strict=False).is_relative_to(ROOT) and not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())


def main():
    sealed=json.loads((PREP/'sealed_packet_v1.json').read_bytes())
    ready=json.loads((PREP/'source_readiness_v1.json').read_bytes())
    closed=json.loads((PREP/'independent_archive_closure_v1.json').read_bytes())
    assert sealed['complete'] and ready['ready_for_seal'] and closed['complete'] and closed['archive_verified']
    assert ready['stage']['protocol']['sha256']==sealed['protocol_sha256']
    assert sealed['source_review']==pin(PREP/'source_readiness_v1.json')
    assert sealed['training_run'] is False and sealed['local_training_neural_gradient_optimizer_VM_calls']==0
    arc=sealed['archive'];sha=sealed['protocol_sha256'];basename=Path(arc['path']).name
    assert pin(ROOT/arc['path'])=={k:arc[k] for k in ('bytes','sha256')}
    assert closed['archive_sha256']==arc['sha256'] and closed['protocol_sha256']==sha
    assert (ROOT/(arc['path']+'.sha256')).read_bytes()==(arc['sha256']+'  '+basename+'\n').encode('ascii')
    assert not PUB.exists() and not GUIDE.exists() and not REPORT.exists()
    guide=f'''# Skip-feature adapter V1: five-step manual L4 guide

Prepared and verified, **not trained or accepted yet**. New trainable adapters
start from retained R2. All515 existing R2 tensors stay frozen. This experiment
tests whether learned features can approach part of the clean-reference diagnostic.

Allow **24GiB free after installation**, preserving4GiB. Estimated L4 runtime:
25–45minutes; the changed graph is unmeasured until manual preflight. Worker
work cap80minutes; external worker cap88minutes20seconds, with bounded shutdown.
Export cap10minutes. Stops retain failures and partial states; never rerun or
append epochs unchanged. Use the existing NVIDIA L4 VM and existing venv.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{basename}" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{basename}.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {basename}.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_codeformer_skip_feature_adapter_v1_vm &&
tar -xzf {basename} -C ~/forensic-dgp
```

Checksum expected: `{arc['sha256']}`. The supplied checksum file uses ASCII/LF.
The new directory check preserves any earlier installation or partial run.

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_skip_adapter_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_codeformer_skip_feature_adapter_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -m research.codeformer_skip_feature_adapter_v1.worker --root . --protocol-sha {sha} --verify-transfer &&
bash run_skip_adapter.sh {sha}
```

Verification performs zero neural calls. The manually invoked supervisor then
starts one finite run:110 Adam updates/440 microbatches, capped3,644 forwards
and four gradient queries. It proves identity initialization and image-loss
connectivity before Adam; failed preservation/resource checks retain the stop.
Training or export completion does not qualify the model.

Detach safely with **Ctrl+B**, release, **D**. Reattach with the step3 command.
For another SSH tab, progress is in:
`~/forensic-dgp/cctv_codeformer_skip_feature_adapter_v1_vm/outputs/supervision/worker.log`.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-supervisor-final.json" "."
```

Each remote source has its own command, matching Windows/PuTTY limitations.
Keep all four returned files. Independent local model-state, saved-image and
visual review are required before changing the app or choosing more training.
If you move to another VM, update its project/name/zone and restore the runtime
before using this guide; this packet is self-contained for model/data sources.
'''
    report=f'''# Skip-feature adapter V1: verified preparation, manual training unrun

19 October2026. The preceding clean-reference SFT oracle produced a visible fine
detail advantage with49.94% lower paired PNG MAE on four readable strong TRAIN
cases. That unavailable-reference result is diagnostic, not a learned model.
This new route adds four residual feature translators after CodeFormer's stock
skip detach and before its frozen SFT blocks. All515 original R2 tensors remain
frozen;24 new tensors contain51,328 trainable parameters. CodeFormer remains the
credited architecture; untouched pretrained CodeFormer remains the benchmark.

## Actual completed preparation

The physical stage and compressed archive were independently checked. The
archive contains **{sealed['stage_files']:,} files /{sealed['uncompressed_bytes']:,}
uncompressed bytes**. Its compressed size is **{arc['bytes']:,} bytes**, SHA256
`{arc['sha256']}`. Protocol SHA256 is `{sha}`. The checksum is ASCII/LF; every
compressed member was streamed and hashed. No new model execution, gradient,
optimizer update, VM connection or actual training was performed for this pilot.

Source reviews, pure fixtures and staged pixel/namespace audits are bound in
`outputs/codeformer_skip_feature_adapter_v1_preparation/`. These establish
preparation, not actual CUDA runtime, useful restoration or qualification.
The VM must measure identity parity, connected gradients and resources before
the first optimizer step. Previous checkpoints and stopped experiments remain.

## Frozen experiment

All55 previously admitted TRAIN photographs contribute55 Gaussian2.4 inputs,
55 clear controls and55 linear-sRGB exposure0.35 brightness-retention controls.
Old411-role membership, holds and optimization flags remain immutable in
provenance. A new explicit165-role subset maps historical filter_a/filter_b
slots to the new filter role; no new training photos or strong inputs were added.
These55 photos are not established unique identities. Terms, attribution and
pretraining/person overlap limits remain documented. The dark proxy is not a
calibrated native-camera corruption.

110 new Adam updates use two blur, one clear and one dark microbatch each:
440 microbatches; each blur input4 times, each clear/dark input2 times. This is
not110 epochs. Feature loss uses mean four-site L1 normalized by each source's
fixed clear-feature RMS clamped1e-6, coefficient0.1, plus paired RGB L1 and0.25
adjacent residual L1. Clear controls target original CodeFormer behavior and
dark controls target R2 behavior, each coefficient2. Each slot divides by4.
Adam LR1e-4 with10-update warmup, betas(.9,.99),eps1e-8,weightdecay0 and gradient
clipnorm1 are fixed prospective choices; they are not proven optimal settings.

Initial exact identity is a control. R2 already inherits some original-based
preservation failures; clear controls therefore target original CodeFormer,
rather than imposing unchanged R2 output after training. Full478 comparisons
are retained at0/55/110:406 historical DEV, including48 unpaired native CCTV,
plus72 separately labeled mixed-role filter probes.24 refusal controls make no
model calls; FINAL stays unused. The18 auxiliary observations remain repeated
TRAIN diagnosis only, with16 readable/two borderline and four readable strong
cases reported separately. They do not become an independent holdout or new
optimization data. Paired scores never become native or Zamboanga performance.

The finite ceiling is3,644 top-level forwards,440 backwards,110 optimizer
attempts and FOUR gradient queries. The earlier three-query proposal is
retained; before sealing, a fourth query was added to prove the isolated image
loss reaches all four adapters, separately from combined image/feature loss.
Zero early-layer gradients from the zero final projection are expected at
initialization; nonzero last-projection gradients and ordinary decoder autograd
are required. Witness state, modes, RNG and gradient fields must remain exact.

Snapshots and full unchanged gates are saved at0/55/110. At55 the additional
no-new-regression check plus both parent/R2 dark requirements enforce a stop;
the original qualification status remains visible. At110 all original
preservation and both dark requirements must pass. Repeated TRAIN guards at
0/20/55/110 add stops. No inherited failure, export receipt or fitted TRAIN
score qualifies the model. Failures, current/stopped states, incomplete writes,
source pins and actual operation records remain exportable; no automatic retry.

The source-derived envelope requires24GiB after installation and4GiB reserve,
with9GiB whole payload/9.25GiB archive caps.55 four-site feature caches total
about2.42GiB and remain research evidence. RSS10GiB and CUDA-reserved18GiB are
hard bounds. Worker stages and outer5300s limit are finite.25–45minutes is an
estimate from prior L4 work; actual changed-graph timing is still unmeasured.

[Exact five-step upload/tmux/download guide](CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_VM.md).
Manual finite L4 training remains the next action. No main-app default changed;
user acceptance, independent final review and full app qualification still
remain. Completion development stays deferred. The earlier storage cleanup is
closed, with no new deletion or reclaimed-space claim here. The full goal stays active.
'''
    header='''# Skip-feature adapter V1 packet verified; manual training unrun

19 October2026. A distinct CodeFormer adaptation packet is physically staged,
sealed and independently verified. It freezes all515 R2 tensors and trains24
new feature-adapter tensors, using55 admitted TRAIN photos and165 role inputs.
110 planned updates/440 microbatches are four blur exposures per photo, not110
epochs. All478 comparisons, source-separated native/unpaired evidence, original
preservation/dark gates, held sources, failures and reserved FINAL remain intact.

[Preparation and limits](CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_PREPARATION.md).
[Five-step manual L4 upload/tmux/download guide](CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_VM.md).
No actual training was launched, no learned improvement is claimed and app
defaults are unchanged. Audit returned states/pixels and all planned sheets
before acceptance. Completion stays deferred; the full goal remains active.
Earlier storage cleanup is closed and is not counted again.

Earlier status below remains historical evidence.

---

'''.encode()
    originals={n:(ROOT/n).read_bytes() for n in DOCS};before={n:pin(ROOT/n) for n in DOCS}
    for n,b in originals.items():
        backup=PUB/'source_backups'/(n+'.before');new(backup,b);assert pin(backup)==before[n]
    for n,b in originals.items():new(PUB/(n+'.staged'),header+b)
    assert all(pin(ROOT/n)==p for n,p in before.items())
    new(GUIDE,guide.encode());new(REPORT,report.encode());records={}
    for n,b in originals.items():
        assert pin(ROOT/n)==before[n] and pin(PUB/'source_backups'/(n+'.before'))==before[n]
        with (PUB/'journal.jsonl').open('ab') as f:
            f.write((json.dumps({'event':'replace_intent','path':n,'before':before[n]})+'\n').encode());f.flush();os.fsync(f.fileno())
        os.replace(PUB/(n+'.staged'),ROOT/n);assert (ROOT/n).read_bytes()==header+b
        records[n]={'before':before[n],'after':pin(ROOT/n),'suffix_offset_bytes':len(header),
            'backup':str((PUB/'source_backups'/(n+'.before')).relative_to(ROOT)).replace('\\','/')}
        with (PUB/'journal.jsonl').open('ab') as f:
            f.write((json.dumps({'event':'replace_complete','path':n,'after':records[n]['after']})+'\n').encode());f.flush();os.fsync(f.fileno())
    new(PUB/'publication_v1.json',(json.dumps({'complete':True,'documents':records,
        'guide':{'path':GUIDE.name,**pin(GUIDE)},'report':{'path':REPORT.name,**pin(REPORT)},
        'sealed_receipt':pin(PREP/'sealed_packet_v1.json'),'archive_closure':pin(PREP/'independent_archive_closure_v1.json'),
        'source':pin(Path(__file__)),'training_run':False,'model_qualified':False,'goal_complete':False},indent=2)+'\n').encode())
    print(json.dumps({'complete':True,'guide':GUIDE.name,'report':REPORT.name,'documents_updated':4,'training_run':False}))


if __name__=='__main__':main()
