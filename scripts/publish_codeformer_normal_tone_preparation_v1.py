"""Publish a verified unrun guide and preserve exact prior instruction bytes."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c


def new_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value.encode('utf-8'))


def main():
    review = ROOT / 'outputs/codeformer_normal_tone_v1_preparation'
    seal_path = ROOT / 'outputs/codeformer_normal_tone_v1_design/sealed_packet_v1.json'
    sealed = c.read_json(seal_path)
    ready_path = review / 'readiness_v1.json'
    ready = c.read_json(ready_path)
    stage = ROOT / 'outputs' / c.VM_NAME
    protocol = c.read_json(stage / 'protocol.json')
    c.verify_protocol(protocol)
    ps, archive_sha = sealed['protocol_sha256'], sealed['archive_sha256']
    archive = Path(sealed['archive'])
    c.require(sealed['complete'] is True and ready['complete'] is True and
              c.sha256(ready_path) == sealed['readiness_receipt_sha256'] and
              c.sha256(stage / 'protocol.json') == ps and c.sha256(archive) == archive_sha and
              ready['protocol_sha256'] == ps and ready['tests_failed'] == 0,
              'Closed matching readiness/seal/actual archive required')
    checksum = Path(str(archive) + '.sha256')
    c.require(checksum.read_bytes() == (archive_sha + '  ' + archive.name + '\n').encode('ascii'),
              'Exact portable checksum required')
    publication = review / 'publication_v1'
    c.require(not publication.exists(), 'Preserve existing publication')
    publication.mkdir()
    prefix_name = 'cctv-codeformer-normal-tone-v1'
    vm_guide = f'''# CodeFormer normal-tone V1: five manual steps

17 October 2026. **VERIFIED SOURCE PACKET / PREPARED / UNRUN.**

This is the next short objective test toward an improved trained CodeFormer
checkpoint. It keeps useful dark-face supervision and tests whether normal-image
coarse brightness supervision contributes to appearance drift. It commits zero
training updates and exports no new model checkpoint. Eight real disposable Adam
steps are controls, not additional epochs or an accepted model.

Upload {sealed['archive_bytes']:,} bytes; {sealed['readback']['members']} regular source members.
Protocol SHA-256: `{ps}`.
Archive SHA-256: `{archive_sha}`.

Estimated runtime **8–18 minutes**, unmeasured and excluding transfers. Enforced
worker limit1,800 seconds, preflight300 and export600, with grace60/30/30 seconds.
Initialization300/source600/probe900 are cumulative phase caps within the worker
total. Require **10 GiB free after installation**, retaining4 GiB reserve. Caps:
RSS8 GiB, reserved CUDA22 GiB, result2 GiB, archive2.25 GiB. These are limits,
not measured peaks or guaranteed completion. Final external closure has30 seconds.

Keep both existing parent folders and the frozen environment. Their present VM
availability was not checked remotely during this preparation. The new preflight
verifies4,630 read-only dependencies:2,011 AdaIN and2,619 loss-response files,
including old source/control outputs and the exact eight metric targets. No
weights or caches are uploaded again. Do not upgrade image/numerical packages.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix_name}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix_name}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c {prefix_name}-execution.tar.gz.sha256 &&
printf '%s  %s\\n' '{archive_sha}' '{prefix_name}-execution.tar.gz' | sha256sum -c - &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
printf '%s  %s\\n' '{c.ROOTS['adain']['protocol_sha256']}' "$HOME/forensic-dgp/{c.ROOTS['adain']['directory']}/protocol.json" | sha256sum -c - &&
printf '%s  %s\\n' '{c.ROOTS['loss_response']['protocol_sha256']}' "$HOME/forensic-dgp/{c.ROOTS['loss_response']['directory']}/protocol.json" | sha256sum -c - &&
python3 -B -c 'import shutil; from pathlib import Path; free=shutil.disk_usage(Path.home()/"forensic-dgp").free; need=10*1024**3+32*1024**2; print({{"free_bytes":free,"required_before_install_bytes":need}}); assert free>=need, "Need 10 GiB plus 32 MiB free before installation"' &&
test ! -e ~/forensic-dgp/{c.VM_NAME} &&
tar -xzf {prefix_name}-execution.tar.gz -C ~/forensic-dgp
```

If a check stops, retain its output. Do not delete previous run folders, replay
historical pilots or recreate missing parent evidence automatically.

## 3. Open tmux

```bash
tmux new-session -A -s cf_normal_tone_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/{c.VM_NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_normal_tone_v1.py --root . --protocol-sha {ps}
```

Maximum224 new CodeFormer forwards,48 autograd queries and8 completed disposable
Adam steps. There are no teacher/LPIPS/benchmark forwards, committed updates,
checkpoint exports or automatic follow-on training. Each branch restores its
exact base; neither measurement state resumes the failed historical recipe.

The supervisor checks workloads, verifies hashes, runs this one finite test and
exports terminal evidence. A parity/resource/quality stop remains a failure;
no automatic retry follows. Detach with **Ctrl+B**, release, **D**; reattach using
`tmux attach-session -t cf_normal_tone_v1`.

Wait for export and the final supervisor companion to finish. Export reports
`"complete": true` and `"archive_sha256"`; a full measurement additionally has
`"complete_measurement": true` and `"actual_disposable_Adam_steps": 8`.
An exported failure can have fewer steps. Export completion is not model success.

## 5. Download from Windows Google Cloud SDK Shell

Each command has one remote source, compatible with Windows PuTTY.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix_name}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix_name}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix_name}-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix_name}-supervisor-final.json" "."
```

The fourth file carries exact final export-log bytes, the parsed finished receipt
and its hash, plus archive/export bindings outside the archive it describes.
Preserve partial/failure evidence if
export or closure fails. Returned files must be independently audited and all
planned visual sheets reviewed before selecting the separate checkpoint-producing
training pilot. The main app is unchanged; no new restoration option is installed.
'''
    prep = f'''# CodeFormer normal-tone V1 preparation

17 October 2026. **PREPARED / UNRUN / NO NEW TRAINED CHECKPOINT.**

[Five manual steps](CCTV_CODEFORMER_NORMAL_TONE_V1_VM.md).
[Prospective design](CCTV_CODEFORMER_NORMAL_TONE_V1_DESIGN.md).
[Independent source review](outputs/codeformer_normal_tone_v1_preparation/independent_source_review_v1.md).
[Fresh readiness receipt](outputs/codeformer_normal_tone_v1_preparation/readiness_v1.json).

The source packet is{sealed['archive_bytes']:,} bytes/{sealed['readback']['members']} regular members,
SHA-256 `{archive_sha}`. Protocol SHA-256 `{ps}`.
All4,630 parent bindings were freshly rehashed locally, spanning
{sum(ready['dependency_bytes'].values()):,} logical bytes. They are references,
not new upload copies. All{ready['tests_run']} CPU fixtures passed, with zero
failures and{ready['tests_skipped']} skips. Python3.10 syntax checks passed.
The sealed archive passed single-gzip CRC and exact USTAR header/order/member/
payload/padding/EOF readback. ASCII basename checksum uses exactly one LF.

The only proposed loss change is normal-only, differentiable luminance-offset
centering on the original native512 pooled16x16 RGB grid. Dark coarse, LPIPS,
behavior anchors, other objective bases, rates/moments, frozen decoder/codebook,
cases and isolated route stay fixed. Omission of normal coarse is a falsifier:
normal supervision may be protective. This does not repeat the rejected inference
brightness gate, change deployment inputs or claim a solved cause.

The224/48/8 finite ledger includes all source/probe/repeat neural calls and real
disposable Adam steps. Six common component bases are explicitly cached. Exact
fresh all-coarse, source-output, full/momentum and target controls must pass;
there is no new direct-full-autograd proof. Preserve any failure without relaxing
equality or appending epochs. Full61-clear/53-blur guards remain binding and are
not replaced by16 same-source TRAIN probes. Native/FINAL/app qualification remains
outstanding, as does the input-only insufficient-structure threshold implementation.

Save32 source witnesses,160 probes,32 repeat native/RAW/token sets,720 fresh coarse
basis arrays and576 actual fusion-delta arrays. Record actual runtime versions,
driver/UUID, per-case camera/target parity, RSS/CUDA peaks and sampled worker-root
free disk. HOME writes have separate reserve guards. Parent/checkpoint/cache/
split/failure preservation remains unchanged. Source-bound autograd/runtime
limits stay distinct from independent saved-array inspection.

Manual L4/tmux only:10GiB free,4GiB reserve; worker1,800/preflight300/export600
seconds, RSS8GiB/CUDA22GiB/result2GiB/archive2.25GiB. These are prospective caps.
Estimated8–18 minutes excludes transfers and is unmeasured for this packet.
No upload, VM connection, neural/gradient/optimizer execution, actual training,
app change, cleanup, Drive transfer or deletion occurred during preparation.
Prior storage offload remains complete and separate.

After returned review supports a loss choice, prepare a separate finite pilot
that writes actual candidate/stopped checkpoints and runs full development and
preservation review. No new accepted model is claimed now; restoration first,
user acceptance and independent final review remain required. Later detector/
completion development stays deferred. The overall goal is incomplete.
'''
    new_text(ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_V1_VM.md', vm_guide)
    new_text(ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_V1_PREPARATION.md', prep)
    prefix = f'''# CodeFormer normal-tone V1 prepared for manual objective test — 17 October 2026

**CURRENT: VERIFIED SOURCE PACKET / UNRUN / NO NEW MODEL CHECKPOINT.**
[Five manual steps](CCTV_CODEFORMER_NORMAL_TONE_V1_VM.md).
[Preparation and limits](CCTV_CODEFORMER_NORMAL_TONE_V1_PREPARATION.md).
[Prospective design](CCTV_CODEFORMER_NORMAL_TONE_V1_DESIGN.md).
[Independent source review](outputs/codeformer_normal_tone_v1_preparation/independent_source_review_v1.md).

The next bounded step toward the requested improved trained CodeFormer tests
normal-only coarse luminance centering against full, normal-coarse omission,
baseline and momentum at original0/stopped50. Dark coarse, other objectives,
LPIPS/anchors, rates, frozen decoder/codebook, sources and isolated routing stay
fixed. Normal coarse may be protective; the omission control can falsify the
hypothesis. No inference brightness gate or automatic recipe selection is added.

Upload{sealed['archive_bytes']:,} bytes/{sealed['readback']['members']} regular source members;
archive SHA-256 `{archive_sha}`,
protocol SHA-256 `{ps}`.
All4,630 parent bindings/{sum(ready['dependency_bytes'].values()):,} logical bytes freshly
rehash locally;{ready['tests_run']} CPU fixtures pass. Exact input/metric-target,
fresh-coarse/cache and prior full/momentum controls must pass on the manual VM.
Local checks establish source/data preparation, not CUDA feasibility or quality.

Caps224 new neural forwards,48 queries,8 real disposable Adam steps; zero
committed updates and no candidate checkpoint. New source/probe/repeat/coarse/
fusion-delta arrays are retained. All cached full-gradient proof stays explicitly
historical/source-bound; no new full-autograd check is claimed. Manual L4/tmux
only,10GiB free with4GiB reserve; worker1,800/preflight300/export600 seconds.
Runtime8–18 minutes is an unmeasured estimate excluding transfer. Preserve any
failure; no historical replay, unchanged resume, epoch extension or relaxed gate.

The checkpoint-producing training pilot follows independent returned-array and
all-sheet review. The prior AdaIN50 clear/normal-blur failure remains binding;
this16-TRAIN source/probe diagnostic does not qualify DEV, FINAL, native CCTV,
user acceptance or the app. No actual neural/training/VM run or app promotion
occurred. Storage offload remains complete and separate; no deletion was done.

The full restoration-first goal stays active: accurately credited CodeFormer
fine-tuned by us, useful256x256 output of one cropped frontal/mild-turn face,
preserved visible appearance and app design, moderate useful brightening allowed,
original CodeFormer/DGP checkpoints and research evidence retained. Native CCTV
stays unpaired/source-separated with audited provenance/terms/resolution/overlap;
paired synthetic evidence and untouched FINAL stay separate. No ethnicity,
exact recovered identity or Zamboanga performance claim follows. User acceptance,
independent final review and bundled inline Playwright Auto/override/PNG app
verification remain required.

The input-only refusal threshold for insufficient visible structure remains
required and unimplemented/uncalibrated. Judge visible relationships, effective
resolution, pose, blocking, clipping and coverage from the input; brightness or
blur alone must not reject usable faces. Preserve DEV cases and input labels;
no output-based relabeling or FINAL tuning. Confirmed hard insufficiency cannot
be bypassed by On/Off or mask edits. Request clearer/more frontal/less-covered
crops; covered mouth/nose alone does not reject enough visible upper-face context.

Detector/completion improvement remains deferred until restoration acceptance.
Retain later masks, sunglasses, glare, hands, obstructing hair, scarves/objects;
automatic removal preview and optional correction; visible appearance, clear
glasses/non-obstructing hair preservation; one plausible estimate, original/
mask/result and PNG/optional bundle downloads; separate automatic/assisted
evidence and no exact hidden-identity claim. All actual training remains the
user's finite manual L4 workflow with verified transfers and independent returns.
The overall goal remains incomplete. Earlier bytes below and in publication
backups are preserved; the completed loss-response commands are historical.

---

'''
    core = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
    changes = {}
    for name in core:
        target = ROOT / name
        before = target.read_bytes()
        c.require(c.sha256(target) == protocol['files_sha256']['instructions/' + name],
                  'Current instructions changed after staging: ' + name)
        backup = publication / (name + '.before')
        with backup.open('xb') as stream:
            stream.write(before)
        target.write_bytes(prefix.encode('utf-8') + before)
        c.require(target.read_bytes()[len(prefix.encode('utf-8')):] == before, 'Historical suffix changed')
        changes[name] = {'before_sha256': c.sha256(backup), 'after_sha256': c.sha256(target),
                        'preserved_suffix_offset_bytes': len(prefix.encode('utf-8')), 'backup': str(backup)}
    docs = ('CCTV_CODEFORMER_NORMAL_TONE_V1_VM.md', 'CCTV_CODEFORMER_NORMAL_TONE_V1_PREPARATION.md',
            'CCTV_CODEFORMER_NORMAL_TONE_V1_DESIGN.md')
    c.write_new_json(publication / 'receipt.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'PREPARED_UNRUN', 'protocol_sha256': ps, 'archive_sha256': archive_sha,
        'archive_bytes': archive.stat().st_size, 'prefix_bytes': len(prefix.encode('utf-8')),
        'instructions': changes, 'documents_sha256': {name: c.sha256(ROOT / name) for name in docs},
        'evidence_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): c.sha256(path)
                           for path in (seal_path, ready_path, archive, checksum)},
        'publisher_source_sha256': c.sha256(__file__), 'neural_gradient_optimizer_VM_calls': 0,
        'actual_training_launched': False, 'new_trained_model_created': False, 'model_qualified': False})
    print({'complete': True, 'manual_guide': docs[0], 'instructions_updated': len(core),
           'old_bytes_preserved': True, 'protocol_sha256': ps, 'model_created': False})


if __name__ == '__main__':
    main()
