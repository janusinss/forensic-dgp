# Unified CodeFormer Completion R2 V1: Manual Finite L4 VM Guide

RELEASED FOR MANUAL TRANSFER ONLY. Preparation has launched no training or VM action.
R1 remains selected until all post-return gates pass.
Source archive: `73376283` bytes, `652` pinned payloads plus `protocol.json`.
Archive SHA-256: `83ee54a8a29b9a92bdb26ee1533957638a9c9a2dfa4065895354f690ca59f672`.
Protocol SHA-256: `60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347`.

## 1. Upload from Windows Google Cloud SDK Shell

Run on Windows, not inside VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-unified-r2-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-unified-r2-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and Safely Install in VM SSH

Requires 8 GiB plus installation payload free, the original inpainting checkpoint,
and full trained R1 state already present on the VM.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-unified-r2-v1-execution.tar.gz.sha256 &&
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
name = 'cctv_completion_codeformer_unified_r2_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-unified-r2-v1-execution.tar.gz'
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
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == '60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-unified-r2-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 256 * 1024 ** 2 and shutil.disk_usage(base).free >= 8 * 1024 ** 3 + total, 'Need 8 GiB plus installation payload free')
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
print({'complete': True, 'installed': str(destination), 'protocol_sha256': '60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347', 'training_launched': False})
PY
```

Expected output: `complete: True`, `protocol_sha256: '60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347'`, `training_launched: False`.

## 3. Open tmux

```bash
tmux new-session -A -s completion_unified_r2_v1
```

## 4. Verify and Launch Inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_unified_r2_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_unified_r2_v1_vm.py --root . --protocol-sha 60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347 --verify-transfer &&
bash scripts/run_completion_codeformer_unified_r2_v1.sh 60549f671b5be8e260460e3787ec460dd5cd3aa404a09f3eefc5cf9e6c9b3347
```

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_unified_r2_v1_vm/trainer.log
```

At completion:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_unified_r2_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-unified-r2-v1-export.json
```

## 5. Download from Windows Google Cloud SDK Shell

One remote source per command; run on Windows, not in VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-unified-r2-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-unified-r2-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-unified-r2-v1-export.json" "."
```
