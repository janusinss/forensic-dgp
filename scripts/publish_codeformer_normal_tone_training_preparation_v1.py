"""Publish verified manual training steps without launching a model or VM."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_training_v1 import contract as c


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(text.encode('utf-8'))


def main():
    review = ROOT / 'outputs/codeformer_normal_tone_training_v1_preparation'
    stage = ROOT / 'outputs' / c.VM_NAME
    sealed = c.read_json(review / 'sealed_packet_v1.json')
    ready_path = review / 'readiness_v1.json'
    ready = c.read_json(ready_path)
    protocol = c.read_json(stage / 'protocol.json')
    c.verify_protocol(protocol)
    ps, ah = sealed['protocol_sha256'], sealed['archive_sha256']
    archive = Path(sealed['archive'])
    c.require(sealed['complete'] is True and ready['complete'] is True and ready['tests_failed'] == 0 and
              c.sha256(ready_path) == sealed['readiness_receipt_sha256'] and
              c.sha256(stage / 'protocol.json') == ps == ready['protocol_sha256'] and
              c.sha256(archive) == ah, 'Closed matching packet/readiness required')
    checksum = Path(str(archive) + '.sha256')
    c.require(checksum.read_bytes() == (ah + '  ' + archive.name + '\n').encode('ascii'),
              'Exact ASCII basename/single-LF checksum required')
    publication = review / 'publication_v1'
    c.require(not publication.exists(), 'Preserve previous publication')
    publication.mkdir()
    guide = f'''# CodeFormer normal-tone training V1: five manual steps

17 October 2026. **ACTUAL TRAINING PILOT PREPARED / UNRUN.**

This new run fine-tunes untouched CodeFormer for at most **50 optimizer updates**
using **200 training observations**. It writes real model/Adam/RNG checkpoints.
It is not50 epochs, an extra-epoch resume or a disposable-step-only diagnostic.
Only normal TRAIN coarse-brightness supervision changes. Dark supervision,
other losses, preserved data/optimizer/architecture and quality gates stay fixed.
The proposed loss remains unproven until its returned outputs are reviewed.

Expected trained state after update50:
`~/forensic-dgp/{c.VM_NAME}/{c.OUTPUTS}/snapshot_0050/checkpoint.pt`.
The initial state is in`snapshot_0000/checkpoint.pt`; failures retain
`stopped_checkpoint.pt`when the state/write reserve allows. Checkpoint files
alone do not establish useful restoration or permit app promotion.

Source-only upload: **{sealed['archive_bytes']:,} bytes**, {sealed['readback']['members']} regular members.
Archive SHA-256: `{ah}`.
Protocol SHA-256: `{ps}`.

Require **16 GiB free after installation** with4 GiB retained reserve. Estimated
runtime **15–30 minutes**, unmeasured for this changed objective and excluding
transfers. Enforced limits: preflight300 seconds, worker3,000 seconds, export600
seconds; termination grace30/60/30 seconds. Within worker: initialization600,
witness180, benchmark600, training900 and each snapshot600 seconds, plus a
20-update timing/projection stop. These caps do not guarantee completion.
RSS8GiB, CUDA reserved22GiB, result3.5GiB, archive3.6GiB, temporary checkpoint1GiB.

The existing AdaIN parent at`~/forensic-dgp/{c.PARENT_NAME}`must remain complete.
The packet references2,011 hash-bound files without copying weights/caches.
Its current VM availability has not been checked remotely here. Keep the frozen
environment; do not upgrade its image/numerical packages. Previous failed
diagnostic/training folders and all original research evidence stay preserved.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{c.PREFIX}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{c.PREFIX}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c {c.PREFIX}-execution.tar.gz.sha256 &&
printf '%s  %s\\n' '{ah}' '{c.PREFIX}-execution.tar.gz' | sha256sum -c - &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
printf '%s  %s\\n' '{c.PARENT_PROTOCOL_SHA}' "$HOME/forensic-dgp/{c.PARENT_NAME}/protocol.json" | sha256sum -c - &&
python3 -B -c 'import shutil; from pathlib import Path; free=shutil.disk_usage(Path.home()/"forensic-dgp").free; need=16*1024**3+32*1024**2; print({{"free_bytes":free,"required_before_install_bytes":need}}); assert free>=need, "Need 16 GiB plus 32 MiB free before installation"' &&
test ! -e ~/forensic-dgp/{c.VM_NAME} &&
tar -xzf {c.PREFIX}-execution.tar.gz -C ~/forensic-dgp
```

If a check stops, retain its output. Do not delete an earlier run, rerun a failed
historical recipe or recreate missing parent evidence automatically.

## 3. Open tmux

```bash
tmux new-session -A -s cf_normal_tone_training_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/{c.VM_NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_normal_tone_training_v1.py --root . --protocol-sha {ps}
```

The supervisor checks workloads and hashes, runs only this new finite pilot and
exports terminal evidence. Maximum1,614 neural forwards,11 gradient queries
(two are tiny non-neural formula checks),200 backward calls and50 Adam attempts.
All training gradients are fresh; no cached-gradient bit-identity gate is used
as training supervision. Each snapshot has406 cases. Both original61-clear and
53-normal-blur RAW/PNG gates remain. It saves the trained snapshot before checking
those gates and preserves a failed checkpoint as failed evidence.

Detach with **Ctrl+B**, release, **D**. Reattach using
`tmux attach-session -t cf_normal_tone_training_v1`.

Wait for export and the final supervisor companion. Export`"complete": true`
and`"archive_sha256"`mean the files were packaged. Read committed update count,
failure status and both quality guards separately. A failed run can still export
successfully. There is no automatic retry, resume, extra epoch or app promotion.

## 5. Download from Windows Google Cloud SDK Shell

Each command has one remote source, compatible with PuTTY on Windows.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{c.PREFIX}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{c.PREFIX}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{c.PREFIX}-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{c.PREFIX}-supervisor-final.json" "."
```

The fourth file binds exact final export-log bytes plus parsed finished receipt/
hash and archive/export identities. Retain failures/partial files if export or
closure stops. Returned checkpoints and images must be independently audited;
all planned visual sheets and native source groups remain required. This is our
trained adaptation of CodeFormer, not a new architecture trained from scratch.
User acceptance, full app verification and independent final review remain
outstanding. The main app has not been switched during preparation.
'''
    put(ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md', guide)
    prep = f'''# CodeFormer normal-tone training V1 preparation

17 October 2026. **VERIFIED SOURCE/DATA PREPARATION / UNRUN.**

[Five manual steps](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md).
[Prospective design](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md).
[Independent source review](outputs/codeformer_normal_tone_training_v1_preparation/independent_source_review_v1.md).

This packet trains an actual50-update hypothesis candidate from untouched
CodeFormer with fresh full-objective gradients and saves genuine checkpoints.
It supersedes the unexecuted treatment branch of the stopped cached-gradient
diagnostic, not its preserved failed verdict. Only normal TRAIN coarse loss
changes; all image-preservation gates remain. The new loss is not yet validated.

Upload{sealed['archive_bytes']:,} bytes/{sealed['readback']['members']} regular members,
archive SHA`{ah}`, protocol SHA`{ps}`.
All2,011 dependencies/{ready['dependency_bytes']:,} logical bytes freshly hashed;
{ready['tests_run']} CPU checks passed with{ready['tests_skipped']} skips and no failures.
Python3.10 syntax and single-gzip/canonical-USTAR/member/padding/EOF readback pass.
No weights, caches, splits or original checkpoints were copied into the upload.

Data stay200 first50 TRAIN observations/100 references,391 HQ parity entries,
50 behavior anchors,318 saved validation inputs and88 comparisons. Snapshot0/50
each406 cases; both61-clear and53-normal-blur gates are unchanged. Planned caps:
1,614 neural forwards,11 gradient queries,200 backward and50 Adam attempts.
The600-position horizon is schedule arithmetic, not authorization for600updates.
Pure-formula prerequisites test the actual shared differentiated mean helper,
not a face-quality outcome. The normal loss uses accumulation/4 during training.

Checkpoint initial/update50/stopped names are explicit; failed/partial states
remain evidence, never an accepted model. All full-model/Adam/RNG/sampler state
and actual update/partial counters remain available for independent review.
RAW/PNG outputs, paired/native source separation and historical comparisons stay
distinct. Exact source/input/target/HQ controls precede use; no cached gradients,
retroactive tolerance change, failed resume or historical replay is introduced.

Manual L4/tmux only.16GiB free/4GiB reserve; worker3,000s, training900s,
initialization600s, witness180s, benchmark600s, snapshots600s, preflight300s,
export600s; RSS8GiB/CUDA22GiB/result3.5GiB/archive3.6GiB/checkpointtmp1GiB.
Estimate15–30minutes excluding transfers is unmeasured for this new objective.
Runtime records actual versions/driver/RSS/CUDA/sampled disk and phase time.

No local/remote neural execution, gradient/backward/optimizer call, upload,
training launch, app/model promotion, cleanup, Drive transfer or deletion was
performed during preparation. Overall goal remains incomplete. After the user
runs this packet, independent returned model/output review and user-visible
improvement are required before the app uses a new candidate. Completion/detector
development remains deferred until restoration acceptance; all later seven-family
requirements, input-only refusal and final app checks remain intact.
'''
    put(ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_PREPARATION.md', prep)
    prefix = f'''# CodeFormer normal-tone training V1 prepared — 17 October 2026

**CURRENT: ACTUAL CHECKPOINT-PRODUCING PILOT PREPARED / UNRUN.**
[Five manual steps](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md).
[Preparation](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_PREPARATION.md).
[Design](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md).
[Previous diagnostic stop](CCTV_CODEFORMER_NORMAL_TONE_V1_RETURN_REVIEW.md).

Latest user priority is creating an improved trained model promptly. This new
finite50-update/200-observation pilot starts untouched CodeFormer and writes
real initial/update50/stopped checkpoints. Normal-only coarse luminance centering
is a prospective hypothesis, not a proven cause/fix. All losses and gradients
are fresh; dark supervision, other objectives, data/optimizer/architecture/
isolated route and complete61-clear/53-blur RAW/PNG guards remain. No cached
gradient training, unchanged failed recipe, old optimizer resume or historical
training replay. Prior cached-gradient control failure stays failed and retained.

Upload{sealed['archive_bytes']:,} bytes/{sealed['readback']['members']} regular source members;
archive SHA`{ah}`, protocol SHA`{ps}`.
Fresh2,011parent files/{ready['dependency_bytes']:,}logical bytes and{ready['tests_run']}CPUchecks
pass. Maximum1,614neural forwards,11queries,200backward/50Adam attempts;
600-position schedule horizon authorizes no extra updates. Two tiny formula
queries precede Adam, without a neural forward. Checkpoints save before quality
assessment and remain failed evidence if a guard stops. No candidate is promoted.

Manual user L4/tmux launch only.16GiB free/4GiB reserve; finite3,000sworker,
900straining,300spreflight/600sexport and resource/write guards. Estimated
15–30minutes excluding transfer is unmeasured. No training or VM run occurred
here. Independent returned checkpoints/outputs/all planned sheets, user acceptance
and independent final review remain required. The main app stays unchanged.

The full goal remains restoration first: accurately credited CodeFormer adapted
by us for useful256x256 of one cropped frontal/mild-turn face, preserved visible
appearance and existing app design. Retain untouched benchmark, original DGP/
trained checkpoints, caches, provenance, terms, splits, failures and migration
backups. Native CCTV is unpaired/source-separated; paired synthetic/untouched
FINAL stay separate. No ethnicity, exact recovered identity or Zamboanga claim.
Input-only insufficiency refusal, Auto/override/PNG and bundled inline Playwright
full-app verification remain outstanding. Later separate seven covering-family
mask-review/completion/visible-pixel/download and automatic-versus-assisted scope
stays deferred until restoration acceptance. Storage offload remains complete
and separate; no deletion was repeated. Overall goal incomplete.

Earlier preparation/test commands below are historical. Preserve every old byte.

---

'''
    names = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
             'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
    changes = {}
    encoded = prefix.encode('utf-8')
    for name in names:
        path = ROOT / name
        before = path.read_bytes()
        c.require(c.sha256(path) == protocol['files_sha256']['instructions/' + name],
                  'Core instructions changed after staging: ' + name)
        backup = publication / (name + '.before')
        with backup.open('xb') as stream:
            stream.write(before)
        path.write_bytes(encoded + before)
        c.require(path.read_bytes()[len(encoded):] == before, 'Historical suffix changed')
        changes[name] = {'before_sha256': c.sha256(backup), 'after_sha256': c.sha256(path),
                         'preserved_suffix_offset_bytes': len(encoded)}
    docs = ('CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md',
            'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_PREPARATION.md',
            'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md')
    c.write_new_json(publication / 'receipt.json', {'complete': True, 'status': 'PREPARED_UNRUN',
        'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': ps, 'archive_sha256': ah,
        'archive_bytes': archive.stat().st_size, 'prefix_bytes': len(encoded), 'instructions': changes,
        'documents_sha256': {name: c.sha256(ROOT / name) for name in docs},
        'readiness_sha256': c.sha256(ready_path), 'seal_sha256': c.sha256(review / 'sealed_packet_v1.json'),
        'publisher_sha256': c.sha256(__file__), 'actual_training_launched': False,
        'local_neural_gradient_optimizer_VM_calls': 0, 'candidate_checkpoint_created': False,
        'model_qualified': False})
    print({'complete': True, 'manual_guide': docs[0], 'protocol_sha256': ps,
           'archive_bytes': archive.stat().st_size, 'training_launched': False})


if __name__ == '__main__':
    main()
