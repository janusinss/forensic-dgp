"""Publish verified manual training commands and retain exact prior document bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'outputs/codeformer_role_preserving_training_v1_preparation'
PREFIX = 'cctv-codeformer-role-preserving-training-v1'
VM = 'cctv_codeformer_role_preserving_training_v1_vm'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def main():
    seal = json.loads((PREP/'seal_v1.json').read_text(encoding='utf-8'))
    physical_path = PREP/'independent_sealed_transport_v1.json'
    physical = json.loads(physical_path.read_text(encoding='utf-8'))
    assert seal['complete'] is True and physical['complete'] is True
    assert physical['archive_sha256'] == seal['archive_sha256']
    assert physical['protocol_sha256'] == seal['protocol_sha256']
    archive = Path(seal['archive'])
    assert archive.stat().st_size == seal['archive_bytes'] and sha(archive) == seal['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes() == (seal['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    protocol = seal['protocol_sha256']
    guide = f'''# CodeFormer restoration training: five manual steps

**VERIFIED TRANSFER PREPARED / NOT RUN — 18 October 2026.**
[Why this is the next model change](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_PREPARATION.md).

This pilot trains **350 tensors in CodeFormer's restoration network** for up to
**50 new Adam updates / 200 microbatches**. It starts from the retained full-coarse
research checkpoint, uses a fresh optimizer and trains no brightness head.
Both normal exposure paths preserve the untouched model's full RGB/token behavior;
dark paths use clean-reference supervision plus frozen-parent retention. This is
our fine-tuned adaptation of CodeFormer. Its quality has not yet been demonstrated.

Require **16 GiB free after installation**, with a 4 GiB reserve. The immutable
parent `{VM.replace('role_preserving_training', 'adain_gradient_isolation')}`
must remain under `~/forensic-dgp`. Preflight verifies **3,362 existing file
bindings**. No automatic cleanup, download, retry or resume is included.
Preparation did not contact the VM or measure its current free space.

Upload: **{seal['archive_bytes']:,} bytes**. SHA-256:
`{seal['archive_sha256']}`.
Protocol SHA-256: `{protocol}`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{PREFIX}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{PREFIX}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {PREFIX}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
test ! -e ~/forensic-dgp/{VM} &&
tar -xzf {PREFIX}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_role_preserving_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{VM} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
TASK_ROOT="$(pwd -P)" &&
PYTHON_VM="$(python -c 'import sys; print(sys.executable)')" &&
"$PYTHON_VM" -B -u "$TASK_ROOT/scripts/supervise_codeformer_role_preserving_training_v1_vm.py" --root "$TASK_ROOT" --protocol-sha {protocol}
```

The supervisor runs preflight, training and export once. Expect approximately
**15–35 minutes**, an estimate for this unrun recipe. Enforced limits: worker
**3,000 seconds**, optimization **900 seconds**, initialization **600 seconds**,
gradient witness **180 seconds** and each 406-case snapshot **600 seconds**.
Preflight/export are capped at **300/600 seconds**, with **30 seconds** termination
grace per stage. Each phase remains within the worker cap. Maximum **50 updates,
200 backward calls and five gradient queries**; **1,397 neural forwards** maximum.
Detach using **Ctrl+B, then D**. An existing competing workload stops this launch.

5. Download the **four return files** from **Windows Google Cloud SDK Shell** after the supervisor finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-supervisor-final.json" "."
```

One remote source per SCP command avoids the Windows PuTTY multiple-source error.
The checksum file is ASCII with LF line endings. An export marked complete only
means the evidence was packaged; it can include a stopped or quality-failed run.
Preserve errors, partial outputs and checkpoints. Do not erase or relaunch the
folder unchanged. Preflight failure leaves its log and does not start training.

The actual model checkpoint is retained at
`~/forensic-dgp/{VM}/outputs/role_preserving_training_v1/snapshot_0050/checkpoint.pt`
after all 50 updates and the final comparison bank finish. Earlier failure keeps
the actual stopped state or partial checkpoint evidence instead. A final-state
alias can point to the same verified checkpoint without another large copy.

Optional monitoring from another **VM SSH** terminal:

```bash
tail -F ~/forensic-dgp/{VM}/outputs/supervision/diagnostic.log ~/forensic-dgp/{VM}/outputs/supervision/export.log
```

Returned state, Adam/RNG/sampler, actual updates, every RAW/PNG, all planned visual
sheets and the unchanged preservation/dark-retention gates require independent
audit. App acceptance, reserved FINAL review and user approval of image quality
remain pending. No automatic app switch or model qualification occurs.
'''
    preparation = f'''# Direct restoration training preparation — 18 October 2026

**Sealed and independently verified; user manual L4 launch pending.**
[Five upload/tmux/download steps](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_VM.md).

The last fixed-state measurement established unwanted darkening on all 100 fitted
dark and 32 unused-reference dark observations. Root reviewed all 24 sheets.
[Exact returned evidence](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_RETURN_REVIEW.md).
Its scalar head changes tone and cannot repair spatial facial detail. A lower
combined loss concealed worse dark behavior; neither more head epochs nor a
brightness threshold is justified by that result.

The new pilot changes the actual restoration objective. It initializes from
the retained full-coarse CodeFormer model (515 tensors; 350 trainable and 165
frozen) and creates fresh Adam state. Both normal slots use only untouched-model
RGB L1 and token KL targets, at weights 2 and 0.1. Dark slots retain clean token
CE/latent/coarse/eligible pixel/LPIPS supervision, weights 0.5/1/4/1/1, plus
frozen-parent RGB/token retention at 2/0.1. Role labels select TRAIN targets and
losses; inference receives only the face crop. There is no tone head, normal
brightness-centering term, runtime brightness switch, weight search or old
optimizer/RNG installation. This is a prospective design, not a proven unique
cause or a guarantee of improvement.

Fresh Adam uses the retained base rates 0.00001 for encoder/token parameters and
0.00002 for fusion, with a new 50-update linear warmup. The pilot commits at most
50 updates and 200 microbatches; it does not append epochs automatically. Genuine
initial/final checkpoints, Adam, RNG, sampler, rates, reserved/committed counters,
gradient witnesses, 200 actual training observations/targets and complete 406-case
development outputs are retained. Numerical forward/target parity and real
normal/dark gradient connectivity are checked before optimizer construction.

Normal preservation still compares all 61 clear and 53 normal-blur cases, in RAW
and delivered PNG, against untouched CodeFormer. The unchanged mean/worst
regression limits are 0.01/0.03. Initial full-coarse failures are recorded without
pretending they are the original baseline. At update50, both dark validation
cohorts (53 lowlight and 53 compound) must retain or improve their parent's mean
PNG MAE and SSIM. Native CCTV has no clean-reference metric. No quality threshold
was relaxed, and a trained/exported model is not automatically accepted.

CPU preparation closed 3,362 immutable physical file bindings: 1,434 selected
source/assets/data, 1,624 retained RAW/PNG outputs, and 304 HQ/runtime/metric
witnesses. Actual CPU decoding checked 200 training inputs, 100 direct clean512
targets/HQ array sets and 406 development inputs/targets/support masks. Source
metadata retains the original 391 TRAIN/53 validation split; only 100 selected
TRAIN source pixels/HQ targets are needed for this finite run. Photograph/source
IDs do not prove separate human identities. FINAL images are unused.

Separate source-bound CPU receipts contain 61 tests run, 60 passed and one
unsupported Windows physical-hard-link check skipped. Four additional sealer
fixtures checked canonical gzip/USTAR and rejected an extra zero record, second
gzip frame and bad CRC. Independent model, data, worker, transport and actual
stage reviews found no unresolved issue. These are preparation checks: no local
neural forward, autograd, optimizer, training or VM connection occurred.

Upload: {seal['archive_bytes']:,} bytes, SHA-256 `{seal['archive_sha256']}`.
Protocol: `{protocol}`. All {seal['static_members']} staged members and physical
single-frame archive boundaries passed independent readback. Require 16 GiB
free after installation and retain 4 GiB reserve. The source-derived ledger
bounds payload at about 3.6 GB within a 4 GiB cap and archive within 4.25 GiB;
it is a conservative projection, not measured VM peak usage. Actual worker
limits are documented in the guide and remain authoritative.

The full goal stays useful 256x256 restoration with visible appearance preserved,
clearer-input refusal when insufficient, credited untouched CodeFormer and old
DGP/checkpoints/caches/splits/failures preserved, audited source terms/overlap,
native unpaired and paired synthetic evidence separated, independent FINAL and
user review, then full existing-app Auto/override/PNG/Playwright verification.
No ethnicity, Zamboanga performance or exact hidden identity is established.
Separate covering detection, reviewable removal masks and seven-family plausible
completion remain deferred until restoration acceptance. Storage offload is
already complete. This preparation does not complete the research goal.
'''
    guides = {'CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_VM.md':guide,
              'CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_PREPARATION.md':preparation}
    for name, value in guides.items():
        new(ROOT/name, value.encode('utf-8'))
    prefix = f'''# Direct CodeFormer restoration training ready — 18 October 2026

**CURRENT: VERIFIED NEW TRAINING PACKAGE / MANUAL L4 LAUNCH PENDING / NOT QUALIFIED.**
[Five upload/tmux/download steps](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_VM.md).
[Design, evidence and finite limits](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_PREPARATION.md).

The fixed-state return is audited: 656 forwards, zero updates; the final tone
head darkens all 264 observations, including all 100 fitted dark cases. All 24
comparison sheets were reviewed. Its genuine trained composite stays preserved
and unqualified. [Actual return review](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_RETURN_REVIEW.md).

The new package trains CodeFormer's 350 restoration tensors directly from the
retained full-coarse parent, using fresh Adam. Both normal exposure paths retain
untouched-model full RGB/token behavior; dark paths retain full clean supervision
and frozen-parent behavior. There is no tone head or old optimizer continuation.
Finite maximum: 50 new updates / 200 microbatches, initial/final checkpoints and
full 406-case DEV RAW/PNG review. Existing quality gates remain in force.

Upload {seal['archive_bytes']:,} bytes, SHA-256 `{seal['archive_sha256']}`;
protocol `{protocol}`. All {seal['static_members']} source members and 3,362 existing
parent bindings passed readback; actual CPU data joins passed. Require 16 GiB
free after installation, with a 4 GiB reserve. Only the user's manual L4/tmux run
is authorized. No local neural work, VM connection or automatic launch occurred.

The full restoration-first goal remains active: useful256 output, preservation,
insufficient-input refusal, user acceptance, independent FINAL and existing-app
Auto/override/PNG/Playwright verification. Retain original CodeFormer/DGP, trained
states, caches/splits/provenance/terms and failures. Native CCTV remains unpaired
and source-separated; no ethnicity/Zamboanga/exact hidden-identity claim follows.
Separate covering/completion stays deferred. Storage offload is already complete.
Earlier commands/status below are historical; the new model has not yet trained.

---

'''.encode('utf-8')
    publication = []
    backups = PREP/'publication_v1'
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
                 'PROJECT_HANDOFF.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        path = ROOT/name
        before = path.read_bytes()
        new(backups/name, before)
        path.write_bytes(prefix+before)
        assert path.read_bytes()[len(prefix):] == before
        publication.append({'path':name,'old_sha256':hashlib.sha256(before).hexdigest(),
            'prefix_bytes':len(prefix),'new_sha256':sha(path),'old_bytes_exact_suffix':True})
    result = {'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'seal_sha256':sha(PREP/'seal_v1.json'),'physical_review_sha256':sha(physical_path),
        'archive_sha256':seal['archive_sha256'],'protocol_sha256':protocol,
        'guides_sha256':{name:sha(ROOT/name) for name in guides},'publications':publication,
        'publisher_sha256':sha(Path(__file__)),'manual_training_not_started':True,
        'model_qualified':False,'goal_complete':False}
    new(backups/'publication.json',(json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'complete':True,'guides':list(guides),'manual_training_not_started':True}))


if __name__ == '__main__':
    main()
