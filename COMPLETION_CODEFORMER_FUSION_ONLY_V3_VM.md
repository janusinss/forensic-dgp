# HISTORICAL V3 RUN - DO NOT LAUNCH AGAIN

26 October 2026: the manual V3 run has returned and was independently audited.
All64 fusion cycles and a valid recovery checkpoint completed, but both fixed
quality screens failed. Keep trained R1; do not append epochs or repeat V3
unchanged. The commands below are historical evidence, not the next action.
See COMPLETION_CODEFORMER_FUSION_ONLY_V3_RESULTS.md in the FACE_MASK_COMPLETION
workspace and outputs/completion_codeformer_fusion_only_v3_return/review.html.

---

# CodeFormer completion fusion-only V3: manual VM route-isolation test

RELEASED FOR MANUAL TRANSFER ONLY. No training has been launched by preparation.
Start from the retained trained R1 state and its complete Adam/RNG/sampler evidence.
Original CodeFormer and every prior state/failure remain preserved.

The source upload is 82400258 bytes and contains 831 pinned
payload files plus its protocol. It reuses original weights and full R1 terminal
state already on the same L4 VM. No duplicate large checkpoint is uploaded.

## 1. Upload from Windows Google Cloud SDK Shell

Run these on Windows, where the prompt starts with C:, not inside VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-fusion-only-v3-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-fusion-only-v3-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Archive SHA256: f2ef14a22a50756ac31825aa14caeb861a76994e75499fce6498bf28df3a07e0.

## 2. Verify and safely install in VM SSH

This checks every regular member and its hash while extracting to a fresh staging
folder, then publishes the new directory. It preserves prior experiments and any
failed installation staging. Requires 7 GiB plus the small installation payload.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-fusion-only-v3-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test -f ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth &&
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile, uuid

def need(ok, message):
    if not ok:
        raise ValueError(message)
base = Path.home() / 'forensic-dgp'
name = 'cctv_completion_codeformer_fusion_only_v3_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-fusion-only-v3-execution.tar.gz'
need(base.is_dir() and (not destination.exists()), 'Preserve any existing experiment; do not reinstall')
need(archive.stat().st_size <= 256 * 1024 ** 2, 'Execution archive size cap')

def safe(value):
    p = PurePosixPath(value)
    need(value and (not p.is_absolute()) and ('..' not in p.parts) and (':' not in value) and ('\\' not in value) and (str(p) == value), 'Unsafe member path')
    return p
with tarfile.open(archive, mode='r|gz') as stream:
    iterator = iter(stream)
    first = next(iterator)
    need(first.isreg() and first.name == name + '/protocol.json' and (0 < first.size <= 2 * 1024 ** 2), 'Expected first protocol member')
    body = stream.extractfile(first).read(2 * 1024 ** 2 + 1)
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == '3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-fusion-only-v3' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 256 * 1024 ** 2 and shutil.disk_usage(base).free >= 7 * 1024 ** 3 + total, 'Need7GiB plus installation payload free')
    stage = base / (name + '.installing-' + uuid.uuid4().hex)
    stage.mkdir()
    with (stage / 'protocol.json').open('xb') as output:
        output.write(body)
    seen = {first.name}
    for member in iterator:
        need(member.isreg() and member.name in expected and (member.name not in seen), 'Unlisted/duplicate/nonregular member')
        parts = safe(member.name).parts
        need(parts[0] == name, 'Wrong member root')
        row = expected[member.name]
        need(member.size == row['bytes'], 'Member size differs')
        path = stage.joinpath(*parts[1:])
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with stream.extractfile(member) as incoming, path.open('xb') as output:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
                count += len(block)
                need(count <= row['bytes'], 'Oversized member')
                digest.update(block)
                output.write(block)
        need(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Member SHA differs')
        seen.add(member.name)
    need(seen == set(expected) | {name + '/protocol.json'}, 'Missing payload members')
    need(not destination.exists(), 'Experiment appeared during installation; preserve staging')
    stage.rename(destination)
print({'complete': True, 'installed': str(destination), 'protocol_sha256': '3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033', 'training_launched': False})
PY
```

Expected complete:true, protocol 3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033, training_launched:false.
If installation refuses, retain the error and existing folders. Do not repeat an
old pilot or bypass hashes. A different VM requires the verified parent state,
original checkpoint and environment before these commands apply.

## 3. Open tmux

If already inside tmux, use the current session. Otherwise:

```bash
tmux new-session -A -s completion_fusion_only_v3
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_fusion_only_v3_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_fusion_only_v3_vm.py --root . --protocol-sha 3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033 --verify-transfer &&
bash scripts/run_completion_codeformer_fusion_only_v3.sh 3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033
```

Transfer verification performs no neural or training work. The next command starts
the distinct finite fusion-only continuation. Both named R1 Adam states are restored;
code Adam remains unchanged at step 32, and fusion Adam advances at most 32 to 96.
R1 encoder/code prediction plus generator/codebook (443 tensors) stay exact; only
the 54 existing fusion tensors train. Sixty-four cycles give 64 new Adam updates
and 256 backwards: 128 cumulative Adam updates across R1 and this new phase. This matches case exposure,
not the earlier joint arm's training compute. No unchanged V2 recipe is rerun.

256 FIT pairs include all 128 original clear anchors and 128 reviewed appearance/
context/mask augmentations. The original 32-case TRAIN monitor remains fixed.
16 new challenges use the same eight adaptive TRAIN monitor sources and are
reported separately; all 32 public cases remain unpaired, without clean targets.
Both input and clean targets receive the declared gray appearance transformation.
Synthetic mild degradation changes only observed input, never the clean target.
These cases are not measured native CCTV, independent identities or FINAL data.

No teacher, CE or latent-matching work runs in this arm. Planned work is 480 whole
model/typed forwards, capped at 600. Four fresh R1 app parity pairs must pass 1e-6.
All 48 initial clear/challenge outputs must also match historical R1 raw 256 values
within 1e-6 and PNG bytes exactly before any update. Historical joint V2 outputs are only a separate
comparison; all quality requirements remain against R1. Fusion HQ512 hole plus inner-
boundary loss, learning rate 1e-5 and clipping remain unchanged.

Worker hard cap: 30 minutes, with a 28-minute cooperative stop for state retention.
Start free disk: 7 GiB; stop floor: 2 GiB. CUDA allocation cap: 20 GiB. Returned
output and both raw/compressed export cap: 2304 MiB. Export cooperative cap:
15 minutes. Failed partial states remain evidence; no automatic resume or earlier
snapshot rescue. Limits are enforced stops, not promises of training completion.

Monitor from a second SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_fusion_only_v3_vm/trainer.log
```

Detach safely with Ctrl+B then D. At the end:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_fusion_only_v3_vm/supervisor.json
cat ~/cctv-completion-codeformer-fusion-only-v3-export.json
```

A complete export means packaging succeeded. Quality requires the predeclared
incremental gain versus frozen R1 on both separately reported monitor cohorts,
nonworsening declared RAW/PNG groups, independent pixel/state audit, all planned
visual reviews and user acceptance. No hidden-identity or full-family qualification
follows from numeric screening. Original/source visible pixels remain exact.

Successful export keeps its successful .partial pathname as a hard link to the same archive
inode; the receipt explicitly records it, and it consumes no duplicate payload
storage. Failed archive/state partials also remain retained; do not delete them.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command. Run on Windows, not inside VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-fusion-only-v3-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-fusion-only-v3-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-fusion-only-v3-export.json" "."
```

Tell this chat when all three files are downloaded. Keep original R1 and CodeFormer
available while the returned candidate is independently reviewed before adoption.
