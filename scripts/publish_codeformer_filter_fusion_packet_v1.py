"""Publish exact sealed manual commands and preserve prior milestone documents."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import transport as t

PREP = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation'
GUIDE = ROOT / 'CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_VM.md'


def main():
    seal = t.read_json(PREP / 'packet_seal_v1.json')
    t.require(seal.get('complete') is True and seal['physical_readback']['complete'] is True
              and seal['neural_gradient_optimizer_VM_calls'] == 0,
              'Complete inert physical archive seal required')
    archive = t.no_links(ROOT / seal['archive']['path'])
    t.require(t.file_pin(archive) == {k: seal['archive'][k] for k in ('bytes', 'sha256')},
              'Actual sealed archive changed before publication')
    checksum = ROOT / seal['checksum']['path']
    t.require(checksum.read_bytes() == t.checksum_bytes(archive.name, seal['archive']['sha256']),
              'Exact portable ASCII/LF checksum required')
    psha, asha = seal['protocol_sha256'], seal['archive']['sha256']
    guide = t.command_guide(psha, asha)
    required = seal['payload_bytes'] + t.LIMITS['initial_free_bytes']
    install = ("cd ~ &&\n"
        f"printf '%s  %s\\n' '{asha}' '{archive.name}' | sha256sum -c - &&\n"
        f"sha256sum -c {archive.name}.sha256 &&\n"
        "test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&\n"
        f"test ! -e ~/forensic-dgp/{t.VM_NAME} &&\n"
        f"python3 -c \"import shutil; free=shutil.disk_usage('/home/janusdominic0/forensic-dgp').free; "
        f"print('Available bytes:', free, 'Required before extraction:', {required}); "
        f"assert free >= {required}, 'Need20GiB free after installation; stop here'\" &&\n"
        f"tar -xzf {archive.name} -C ~/forensic-dgp")
    text = f'''# CodeFormer filter-fusion training V1 — manual VM guide

**Sealed preparation only; training has not run.** The experiment starts from our
retained R2 adaptation and targets readable blur, mosaic and repeated-line inputs.
It trains existing fusion branches while retaining clear-face and brightening
controls. No model is promoted by upload, training or export completion.

The packet contains **{seal['static_files']:,} files / {seal['payload_bytes']:,} bytes**.
The archive is **{seal['archive']['bytes']:,} bytes**. Require **20GiB free after
installation** and preserve the enforced4GiB reserve. The install command checks
space before extraction. Stop if it fails; this guide authorizes no deletion.

The ceiling is **151 new optimizer updates /604 microbatches**, not151 epochs.
The pilot uses301 distinct filter cases,55 clear and55 brightness-only anchors;
one filter case is explicitly repeated for the four-slot update schedule.
Expect progress every50 baseline/snapshot inputs, every10 teacher inputs and each
training update. GPU runtime is not measured yet. Internal work is capped at80
minutes; the outer worker cap is5300seconds and export10minutes, with finite
verification/closure. Allow approximately110minutes for the bounded session;
quality/resource failures can stop it much earlier. No automatic retry/resume.

1. Upload in **Windows Google Cloud SDK Shell**:

```bat
{chr(10).join(guide['upload_Windows_SDK_Shell'])}
```

2. Install in **VM SSH**:

```bash
{install}
```

3. Open **tmux**:

```bash
{guide['tmux_VM_SSH']}
```

4. Launch **inside tmux**:

```bash
{guide['launch_inside_tmux']}
```

5. Download after the supervisor finishes, using **Windows Google Cloud SDK Shell**:

```bat
{chr(10).join(guide['download_Windows_SDK_Shell_one_remote_source_per_command'])}
```

Use each SCP command separately; Windows PuTTY does not support multiple remote
sources in one SCP call. The export receipt can say `complete: true` after a
training failure: it means the evidence archive was created. Actual update counts,
gate failures and physical checkpoint status remain separate. Preserve all four
downloaded files for independent local review.

To detach safely, press **Ctrl+B**, release, then **D**. Reopen with the same tmux
command. In another VM SSH terminal, monitor the saved log:

```bash
tail -F ~/forensic-dgp/{t.VM_NAME}/outputs/supervision/worker.log
```

If verification, storage, initialization, gradient or quality checks fail, retain
the traceback and export. Do not change epochs, gates, source or checkpoints and
rerun this immutable packet. Earlier research runs and failure evidence remain.

**Protocol SHA256**: `{psha}`

**Upload archive SHA256**: `{asha}`

[Preparation and evidence](CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_PREPARATION.md)
and [finite design](research/codeformer_filter_fusion_training_v1/DESIGN_PILOT_V1.md)
retain source terms, split/overlap limits, exact counters and original preservation
rules. Native CCTV remains unpaired; wild1–wild5 have no clean originals and remain
visual stress tests outside training. No automatic input cutoff, real Zamboanga
performance or exact reconstructed identity is established. Keep the current app
unchanged until returned outputs, user-visible usefulness and required independent
evaluation qualify a replacement. Completion development remains deferred.
'''
    t.require(not GUIDE.exists(), 'Preserve previous guide; no overwrite')
    with GUIDE.open('xb') as handle:
        handle.write(text.encode('utf-8'))
    folder = PREP / 'instruction_publication_sealed_v1'
    folder.mkdir(exist_ok=False)
    header = f'''# Filter-fusion151-update manual packet sealed; training unrun — 18 October 2026

[Exact five-step manual guide](CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_VM.md).
The independently reviewed physical packet contains{seal['static_files']:,} files /
{seal['payload_bytes']:,} bytes, with3 exact input model states and full data/provenance.
Canonical streamed archive readback and ASCII/LF checksum pass. Source/data/math/
failure fixtures passed122 checks. This proves transfer preparation, not GPU runtime
or model usefulness. No neural/gradient/optimizer/VM call, app/default change or
new trained model occurred. Actual training remains the user's manual L4/tmux run.

Separate admission is301 filter +55 clear +55 brightness-retention inputs,411
role inputs from55 photographs, with whole-source holds21374/45780 retained.
Cap151 updates/604 microbatches, one fixed03212 padding duplicate,3,639 forwards
and7 gradient queries. Full original RAW/PNG preservation gates remain unchanged;
24 repeated TRAIN guards add an early20-update stop. Snapshots0/50/151 include
all478 comparison slots:406 historical DEV plus72 mixed TRAIN/DEV filter probes.
Keep48 native CCTV inputs unpaired,24 refusals without forwards and FINAL unused.

Require20GiB VM free after installation and preserve4GiB. Retain failure/partial
states and original checkpoints; no automatic retry or resume. Partial gradients
are not serialized. Independently audit returned evidence and every planned visual
sheet before acceptance. User-visible restoration and independent FINAL/app flow
still remain. Unknown wild1–wild5 remain unpaired visual stress tests; insufficient
usable structure calls for clearer input. Completion stays deferred. Overall goal
remains active. Earlier storage offload is complete; no cleanup repeated here.

Earlier status below remains retained.

---

'''.encode('utf-8')
    records = []
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md',
                 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        path = ROOT / name
        before = path.read_bytes()
        with (folder/name).open('xb') as saved: saved.write(before)
        after = header + before
        path.write_bytes(after)
        t.require(path.read_bytes()[len(header):] == (folder/name).read_bytes() == before,
                  'Exact historical instruction suffix not retained')
        records.append({'path': name, 'old_bytes': len(before),
            'old_sha256': hashlib.sha256(before).hexdigest(), 'new': t.file_pin(path),
            'backup': (folder/name).relative_to(ROOT).as_posix(),
            'retained_suffix_offset': len(header), 'exact_suffix_readback': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'source': t.file_pin(Path(__file__)), 'sealed_receipt': t.file_pin(PREP/'packet_seal_v1.json'),
        'guide': {'path': GUIDE.relative_to(ROOT).as_posix(), **t.file_pin(GUIDE)},
        'files': records, 'training_not_run': True, 'model_qualified': False,
        'actual_neural_gradient_optimizer_VM_calls': 0}
    t.write_new_json(folder/'publication.json', receipt)
    print(t.json_bytes(receipt).decode(), flush=True)


if __name__ == '__main__': main()
