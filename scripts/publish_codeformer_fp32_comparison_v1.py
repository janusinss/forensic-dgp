"""Publish reviewed manual commands and preserve all prior workflow text."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/codeformer_fp32_comparison_v1_preparation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main():
    archive = read(EVIDENCE / 'execution_archive.json')
    verification = read(EVIDENCE / 'independent_packet_archive_readback_v1.json')
    assert archive['complete'] and verification['complete'] and verification['approved']
    assert verification['archive_sha256'] == archive['archive_sha256']
    assert verification['protocol_sha256'] == archive['protocol_sha256']
    protocol_sha, archive_sha = archive['protocol_sha256'], archive['archive_sha256']
    assert sha(ROOT / archive['archive']) == archive_sha
    prefix = 'cctv-codeformer-fp32-comparison-v1'
    packet = ROOT / 'outputs/cctv_codeformer_fp32_comparison_v1_vm'
    p = read(packet / 'protocol.json')
    assert sha(packet / 'protocol.json') == protocol_sha
    dependency_bytes = sum(p['dependency_files_bytes'].values())
    publication = EVIDENCE / 'publication_v1'
    publication.mkdir(exist_ok=False)
    (publication / 'prior').mkdir()
    guide = f'''# FP32 CodeFormer comparison V1 — five manual steps

**VERIFIED / PREPARED / UNRUN.** This trains a fresh copy of untouched
CodeFormer for at most **50 optimizer updates**, using V2's exact first
200 observations and unchanged preservation checks. This is a precision
comparison, not 50 epochs or a proven model improvement.

Upload: **{archive['bytes']:,} bytes ({archive['bytes'] / 1e6:.2f} MB)**.
The installer copies **{dependency_bytes:,} bytes ({dependency_bytes / 1024**3:.2f} GiB)**
of verified immutable dependencies from the existing V2 folder. It requires
that copy space **plus 16 GiB free before copying**, and **16 GiB free after
installation**. Runtime retains a **4 GiB reserve**. Original folders and
checkpoints remain protected; no automatic cleanup or package installation.

Planning estimate: **10–25 minutes** for training, comparisons and export;
FP32 optimization timing/memory remain unmeasured until this run. Hard limits:
15 minutes dependency copy; 5 minutes preflight; 50 minutes worker including
at most 15 minutes optimization; 10 minutes export, plus termination grace.
A resource or quality stop retains the evidence. No precision fallback,
automatic retry, resume or continuation beyond update50.

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install — VM SSH

```bash
cd ~ &&
sha256sum -c {prefix}-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -f ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm/protocol.json &&
test ! -e ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
tar -xzf {prefix}-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
timeout --signal=TERM --kill-after=30 900 python3 -B scripts/prepare_codeformer_fp32_comparison_v1.py --stage-dependencies --root . --protocol-sha {protocol_sha}
```

Expected: checksum `OK`, followed by a dependency receipt with `complete: true`.
That receipt confirms installation only. If a dependency, space or existing-file
check stops, preserve the message and partial files; do not delete the folder
or rerun an old pilot. The installer has no neural or training calls.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s codeformer_fp32_v1
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_fp32_comparison_v1.py --root . --protocol-sha {protocol_sha}
```

The supervisor immediately prints a child PID and log path. It verifies all
installed hashes, checks live GPU/workload availability, runs this one finite
training comparison and exports retained quality/resource-stop evidence too.
An idle tmux shell alone is allowed. No V1/V2 training is relaunched.

The training output goes to a log, so the tmux terminal can stay quiet during
the stage. Read progress from a **second VM SSH terminal**:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm/outputs/supervision/trainer.log
```

Leave the viewer with **Ctrl+C**. Detach tmux with **Ctrl+B**, release, then
**D**. Reattach with `tmux attach -t codeformer_fp32_v1`. Do not paste the launch
command into a second terminal while its child is still running.

At completion, the supervisor prints each stage's `exit_code`. Its
`outputs/supervision/finished.json` separately records `training_process_exit_code`
and `export_exit_code`. The file below reports archive status:

```bash
cat ~/{prefix}-export.json
```

`complete: true` confirms an exported archive. Separately check
`run_results_present` and `failure_present`; archive completion never means
the model passed quality review. Stop at 50 even if guards pass.

## 5. Download — Windows Google Cloud SDK Shell, after export

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-export.json" "."
```

Each SCP command has one remote source for Windows PuTTY compatibility. All
three files land in the existing local `outputs` folder. Download stopped
results too. The result archive has a hard 3.6 GiB bound; actual size is unknown.
Independent review must check inputs, actual weight/Adam changes, all RAW/PNG
outputs, preserved frozen tensors, calls/timing/stops and visual comparisons
before deciding whether to continue with a separately designed longer run.

Execution archive SHA-256: `{archive_sha}`.

Protocol SHA-256: `{protocol_sha}`.

[Preparation evidence](CCTV_CODEFORMER_FP32_COMPARISON_V1_PREPARATION.md).
[Frozen design](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md).
'''
    prep = f'''# FP32 CodeFormer comparison V1 prepared — 16 October 2026

**VERIFIED / PREPARED / UNRUN.** No new local neural, gradient, optimizer,
training or VM execution occurred during preparation. This package tests
FP32 optimization against the retained V2 first 50 BF16 experiment. It does
not establish improved restoration or qualify a replacement checkpoint.

[Five exact upload/install/tmux/launch/download steps](CCTV_CODEFORMER_FP32_COMPARISON_V1_VM.md).

The same 200 observations/100 TRAIN reference IDs, first 50 learning rates,
Adam settings, losses, six trainable groups and quality gates are retained.
Distinct reference IDs do not establish 100 unique people. The frozen 600
schedule/data horizon is separate from the 50-update execution cap. No more
epochs are appended to a failed state; untouched CodeFormer is the initializer.

All 200 input PNG/float32 and clean512 bindings, all 50 saved FP32 teacher
anchors, 444 photographic sources and 406 development inputs were independently
audited. The 48 native CCTV comparisons remain unpaired; reserved native/final
pixels were not used. Existing overlap, provenance and terms limits remain.
No ethnicity or Zamboanga-performance claim follows.

Exact retained V2 behavior arrays provide the loss targets. The 50 budgeted
online FP32 teacher passes are measured separately, with both array sets saved;
small numerical logit differences cannot silently change the control targets.
All 391 new HQ-cache index/feature pairs are retained for return audit without
extra forwards. Old/new HQ-cache equality is unproven because V2 did not export
its cache. FP32 actual backward/Adam memory and timing remain unmeasured.

**48 CPU tests passed**, including Linux/tmux/exact-root rejection, unchanged
quality guard behavior, exact data/LR/target bindings, finite calls, strict
checkpoint/file limits, failed-result export, live-writer refusal and corrupted
archive boundaries. All 11 new code/test sources parse as Python 3.10. Independent
source reviews approved the current bytes before protocol/archive creation.
Earlier preparation-checker assumptions and source drafts remain retained.

The resource ledger budgets 3,421,371,648 bytes of final artifacts within 3.5 GiB:
two 406-output RAW/PNG snapshots, three bounded checkpoint states, 391 cache
pairs, both 50-pair teacher sets, 200 inputs/tokens and metadata/source records.
Writers enforce 384 MiB initial and 832 MiB populated-Adam checkpoints, with a
separate 1 GiB temporary reservation and 7,000-file ceiling. Require 16 GiB free
after installing 1.24 GiB of verified copies; retain 4 GiB during execution.
Worker/optimization/export limits are 3,000/900/600 seconds; peak/current CUDA
reserved-memory bound is 22 GiB. Resource or quality failures retain their stops.

Upload archive: **{archive['bytes']:,} bytes**; exact member readback passed.
Its unchanged large dependencies remain external locally and are copied as
verified regular files on the VM. No parent worker, old output, overwrite or
hardlink is used. LF checksum avoids the earlier Windows CRLF issue.

Archive SHA-256: `{archive_sha}`.

Protocol SHA-256: `{protocol_sha}`.

[Data audit](outputs/codeformer_fp32_comparison_v1_preparation/independent_data_resource_v3/preparation_audit.json),
[resource refresh](outputs/codeformer_fp32_comparison_v1_preparation/independent_resource_refresh_v4.json),
[source readiness](outputs/codeformer_fp32_comparison_v1_preparation/source_readiness.json),
[independent packet/archive readback](outputs/codeformer_fp32_comparison_v1_preparation/independent_packet_archive_readback_v1.json).

Moderate helpful brightening remains accepted; examine clarity, appearance,
contrast and clipping separately. V1/V2 historical quality failures remain
unchanged. User acceptance, independent final review, useful native evidence,
app integration/Auto/override/PNG verification and later covering-family work
are outstanding. Restoration stays first; the overall goal remains active.
'''
    for name, content in (('CCTV_CODEFORMER_FP32_COMPARISON_V1_VM.md', guide),
                          ('CCTV_CODEFORMER_FP32_COMPARISON_V1_PREPARATION.md', prep)):
        write_new(ROOT / name, content.encode('utf-8'))
    top = '''# FP32 CodeFormer comparison V1 ready for manual training — 16 October 2026

**CURRENT: VERIFIED / PREPARED / UNRUN; no new optimizer updates.**
The audited precision mismatch now has a new finite 50-update FP32 comparison
package. It starts from untouched CodeFormer, retains V2's exact 200 observations,
first 50 learning rates, losses and RAW/PNG quality checks, and stops at 50 even
if they pass. The original 600 schedule horizon remains separate. All prior
checkpoints, splits, provenance and failed gates remain protected.

[Five exact manual steps](CCTV_CODEFORMER_FP32_COMPARISON_V1_VM.md).
[Verified preparation and limits](CCTV_CODEFORMER_FP32_COMPARISON_V1_PREPARATION.md).
The package is not uploaded or run. Actual training remains human-launched
on the L4 in tmux. Require 16 GiB free after dependency installation, retaining
4 GiB. FP32 backward memory/timing are unmeasured until its finite benchmark.
No automatic fallback, retries, old pilots or appended epochs are supported.

Independent audits verified all 200 TRAIN inputs/targets, 50 teacher anchors,
444 photographic sources, 406 development inputs and exact first 50 rates.
48 CPU tests, source reviews and complete archive readback passed. A passing
preparation does not establish model usefulness. After return, independently
review actual RAW/PNG outputs, states/Adam, frozen tensors, all planned sheets,
native unpaired evidence and unchanged stops before considering further work.

Moderate helpful brightening remains accepted; preserve visible features,
contrast and appearance. Restoration remains first, with detector/completion
development deferred until user acceptance. Native/final usefulness, app
adoption and full-flow verification remain outstanding. No cleanup or Drive
transfer occurred during this preparation; the earlier offload stays complete.
The full thesis goal remains active. Earlier DESIGN ONLY and return statuses
below are preserved unchanged history, superseded by this prepared-package status.

---

'''
    documents = []
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
                 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        path = ROOT / name
        before = path.read_bytes(); prefix_bytes = top.encode('utf-8')
        write_new(publication / 'prior' / name, before)
        staged = ROOT / (name + '.fp32-preparation-staged')
        write_new(staged, prefix_bytes + before)
        assert path.read_bytes() == before
        os.replace(staged, path)
        assert path.read_bytes()[len(prefix_bytes):] == before
        documents.append({'path': name, 'before_sha256': hashlib.sha256(before).hexdigest(),
                          'after_sha256': sha(path), 'retained_suffix_offset': len(prefix_bytes)})
    result = {'complete': True, 'status': 'verified_prepared_unrun', 'documents': documents,
        'guide_sha256': sha(ROOT / 'CCTV_CODEFORMER_FP32_COMPARISON_V1_VM.md'),
        'preparation_sha256': sha(ROOT / 'CCTV_CODEFORMER_FP32_COMPARISON_V1_PREPARATION.md'),
        'protocol_sha256': protocol_sha, 'archive_sha256': archive_sha,
        'independent_readback_sha256': sha(EVIDENCE / 'independent_packet_archive_readback_v1.json'),
        'source_sha256': sha(__file__), 'actual_training_run': False, 'VM_actions': 0,
        'model_qualified': False, 'goal_complete': False}
    write_new(publication / 'receipt.json', (json.dumps(result, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'complete': True, 'guide': 'CCTV_CODEFORMER_FP32_COMPARISON_V1_VM.md',
                      'prepared_unrun': True}))


if __name__ == '__main__':
    main()
