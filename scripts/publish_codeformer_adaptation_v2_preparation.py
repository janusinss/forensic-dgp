"""Publish a verified manual V2 packet without training or editing frozen bytes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_training_readiness_v2/publication_v1'
DOCUMENTS = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
             'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def main():
    assert not OUT.exists(), 'Preserve prior publication'
    archive = json.loads((ROOT / 'outputs/codeformer_adaptation_v2_archive.json').read_text())
    audit_path = ROOT / 'outputs/codeformer_training_readiness_v2/root_review_v1/independent_archive_review.json'
    audit = json.loads(audit_path.read_text())
    assert archive['complete'] and audit['complete'] and audit['all_members_streamed_and_verified']
    assert archive['archive_sha256'] == audit['archive_sha256']
    assert archive['protocol_sha256'] == audit['protocol_sha256']
    pin = archive['protocol_sha256']
    checksum = archive['archive_sha256']
    guide = f'''# CodeFormer V2: brightness correction — five manual VM steps

**VERIFIED / PREPARED / UNRUN.** This changed fine-tune starts from untouched
CodeFormer and addresses V1's brightness drift. It preserves V1's promising
outputs and its failed clear-preservation evidence. Do not rerun V1.

Require **20 GiB free after installation**; the worker preserves a **4 GiB** reserve.
Planning estimate: **20–40 minutes** including snapshots/export. Actual V2 timing
is unmeasured; first 20 updates measure feasibility. Maximum 600 updates, reviews
at 0/50/200/600; optimization cap 60 minutes, worker cap 120 minutes, export cap
10 minutes. A preservation/resource stop still exports evidence.

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adaptation-v2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adaptation-v2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install — VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-adaptation-v2-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm &&
tar -xzf cctv-codeformer-adaptation-v2-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum result: `cctv-codeformer-adaptation-v2-execution.tar.gz: OK`.
The supplied checksum file uses Linux LF endings. If the directory already exists,
retain it and inspect its evidence before any new version; do not overwrite it.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s codeformer_adaptation_v2
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
bash scripts/run_adaptation_v2.sh {pin}
```

The supervisor verifies transfer bytes, tmux, idle workloads and finite execution;
the worker checks the L4, packages, space, initial parity, gradients and quality.
An idle tmux session alone is allowed. Existing training/results are not resumed.
Do not start a second run. Detach with **Ctrl+B**, release, then **D**.
Reattach using `tmux attach -t codeformer_adaptation_v2`.

After export, `complete: true` confirms a packaged archive. Check `Trainer exit
code`, `failure_present` and `run_results_present`; export completion alone is
not training success or model acceptance. Preserve the console output and files
even if a guard stops the run. Do not append epochs to a failed recipe.

## 5. Download — Windows Google Cloud SDK Shell, after export

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v2-export.json" "."
```

Each command has one remote source, compatible with Windows PuTTY. The downloaded
files will be in `C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs`.
The return must be independently audited before choosing a model or app default.

Execution archive SHA256: `{checksum}`.
Protocol SHA256: `{pin}`.
[Verified preparation](CCTV_CODEFORMER_ADAPTATION_V2_PREPARATION.md).
'''
    prep = f'''# CodeFormer V2 preparation verified — 15 October 2026

**PREPARED / UNRUN.** The next manual restoration fine-tune targets excessive
brightening while preserving useful facial-detail gains. No V2 training, neural
forward, gradient or optimizer update was run during local preparation.

The user's personal review favored most V1 outputs over untouched CodeFormer,
while explicitly identifying increased brightness. This is positive feedback,
not acceptance of its failed preservation case or every source/profile.
[Exact user feedback](CCTV_CODEFORMER_ADAPTATION_V1_USER_REVIEW_20261015.md).
[Audited V1 results and all comparisons](CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md).

## What changes

The separate candidate starts from untouched CodeFormer. Matched normal/dark
TRAIN pairs balance exposure at each selected severity. A coarse RGB loss applies
even at 24/28-pixel observations, while fine losses retain their 32-pixel eligibility.
A separate frozen original CodeFormer teacher anchors normal clear/light TRAIN
behavior. This is a joint data/loss/schedule experiment; it cannot identify the
causal contribution of each change. Architecture and inference policy stay fixed:
prepared256 → internal512, fidelity1, AdaINTrue → delivered256.

Adam remains explicit: encoder/token LR1e-5, fusion2e-5, 50-update warmup and
cosine to2e-6 at update600. There are at most 600 updates / 2400 image exposures,
with snapshots0/50/200/600. Both old clear RAW/PNG guards remain, with additional
prospective normal-blur brightness/coarse-error regression guards. They are abort
checks, not proof of useful structure or a new interpretation of historical gates.

The same391 audited HQ TRAIN references and53 separate validation sources are
retained. All318 validation input PNGs and pair records are exact V1 VM bytes;
the88 fixed comparisons include48 native/unpaired and40 synthetic/paired views.
Reserved final identities remain excluded. Every generated TRAIN input and
behavior-teacher anchor is retained for return auditing. Source terms, FFHQ
pretraining/subject-overlap uncertainty and repeated development exposure remain
explicit. Native CCTV receives no aligned-clean PSNR/SSIM or Zamboanga claim.

## Verified transfer and limits

- **59 combined local tests passed**, plus eight independent CPU formula fixtures.
- **1,410 archive members / 1,333,451,654 payload bytes** were checked against
  retained originals and exact namespaces. The archive was independently streamed.
- Twelve original comparison checkpoints and V1's update0/update200/stopped states
  and failure were independently checked unchanged.
- Require20GiB after install, retain4GiB reserve and22GiB reserved-VRAM ceiling.
  Initialization600s, first20 benchmark600s, optimization3600s, each snapshot600s,
  worker7200s and export600s; external termination grace is separately declared.
- Conservative output planning includes all training images, teacher anchors,
  RAW/PNG snapshots, complete-or-incomplete checkpoint evidence, metadata and
  archive overhead. Runtime measured byte caps remain authoritative.

Archive: `outputs/cctv-codeformer-adaptation-v2-execution.tar.gz`.
Compressed bytes: **{archive['bytes']:,}**.
SHA256: `{checksum}`.
Protocol SHA256: `{pin}`.

[Five upload/install/tmux/run/download steps](CCTV_CODEFORMER_ADAPTATION_V2_VM.md).
[Frozen prospective design](CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md) retains its
historical design-only wording and exact approved bytes; this verified preparation
supersedes that packaging status without editing the frozen design or packet.

## Evidence and acceptance

[Independent worker/model review](outputs/codeformer_adaptation_v2_design_review/worker_review_v1/INDEPENDENT_WORKER_MODEL_REVIEW.md),
[independent packet readback](outputs/codeformer_training_readiness_v2/root_review_v1/independent_packet_review_v1.json),
[combined source tests](outputs/codeformer_training_readiness_v2/root_review_v1/source_readiness.json),
and [independent archive audit](outputs/codeformer_training_readiness_v2/root_review_v1/independent_archive_review.json)
establish preparation integrity. L4 execution, resulting image quality and the
new recipe's usefulness are unmeasured. Do not auto-launch, repeat V1 or change
the app default. Actual training remains the user's manual tmux action.

After download, independently audit archive bytes, weights/Adam/RNG, exact inputs,
teacher anchors, actual exposures, RAW/PNG conversions and profile metrics, then
review all intended images against untouched CodeFormer and retained V1.
User acceptance and independent reserved-final review remain required before
integration and bundled inline Playwright app verification. Restoration remains
first; detector/completion improvement stays deferred. The full goal is active.
'''
    put(ROOT / 'CCTV_CODEFORMER_ADAPTATION_V2_VM.md', guide.encode('utf-8'))
    put(ROOT / 'CCTV_CODEFORMER_ADAPTATION_V2_PREPARATION.md', prep.encode('utf-8'))
    prefix = ('''# CodeFormer V2 prepared for manual brightness-correction training — 15 October 2026

**CURRENT: VERIFIED / PREPARED / UNRUN.** The user finds most first fine-tune
outputs better than original CodeFormer but observes excess brightness. V1's
200-update preservation stop remains binding; no failed recipe is resumed.

[Five manual V2 steps](CCTV_CODEFORMER_ADAPTATION_V2_VM.md).
[Verified changes, limits and acceptance boundaries](CCTV_CODEFORMER_ADAPTATION_V2_PREPARATION.md).
59 local tests and independent source/packet/archive audits passed. A changed
600-update comparison uses balanced normal/dark TRAIN pairs, coarse brightness
supervision at all severities and frozen original-behavior anchors. All original
checkpoints, V1 states, splits and failures remain retained. No new training,
app promotion or useful native/final qualification is claimed.

Actual training remains manual on the L4 in tmux, with finite stops and exact
transfers. Restoration comes first; further detector/completion development
remains deferred until accepted restoration. The full thesis goal is active.
Earlier preparation statuses and commands below are retained history.

---

''').encode('utf-8')
    records = []
    for name in DOCUMENTS:
        path = ROOT / name
        old = path.read_bytes()
        put(OUT / 'prior_documents' / name, old)
        assert path.read_bytes() == old, 'Concurrent publication edit'
        path.write_bytes(prefix + old)
        assert path.read_bytes()[len(prefix):] == old
        records.append({'file': name, 'previous_sha256': sha(old),
                        'current_sha256': sha(path.read_bytes()), 'prefix_bytes': len(prefix),
                        'prior_bytes_preserved': True})
    report = {'complete': True, 'status': 'prepared_unrun_not_qualified',
              'archive_sha256': checksum, 'protocol_sha256': pin,
              'documents': records, 'archive_audit_sha256': sha(audit_path.read_bytes()),
              'guide_sha256': sha(guide.encode()), 'preparation_sha256': sha(prep.encode()),
              'VM_calls': 0, 'neural_calls': 0, 'optimizer_updates': 0}
    put(OUT / 'receipt.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'guide': 'CCTV_CODEFORMER_ADAPTATION_V2_VM.md',
                      'prepared_not_run': True, 'history_preserved': True}))


if __name__ == '__main__':
    main()
