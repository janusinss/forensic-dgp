> **HISTORICAL_EXECUTED_DO_NOT_RELAUNCH — completed 26 October 2026.**
> This probe has already run. Commands below are retained as historical evidence; do not upload, reinstall or launch it again.
> [Current results and next-design status](<C:/xampp/htdocs/YEAR 4/Testing_FACE_MASK_COMPLETION/COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md>).

---

# SAME-codebook completion probe V1: manual frozen-state diagnostic

RELEASED FOR MANUAL TRANSFER ONLY. No training or VM work was launched by preparation.
This creates no new model. It diagnoses the current R1 code representation while all 497
model tensors remain frozen. Original/R1 and every failed training state stay preserved.

Archive: 30037439 bytes, 254 payload files plus protocol.
SHA256: 099e3cac01b107258be8d1af4d5bd2eb72485bdb34b087553c46f0749654e2c7.

## 1. Upload from Windows Google Cloud SDK Shell

Run on Windows, where the prompt starts C:, not inside VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and safely install in VM SSH

The exact original inpainting checkpoint and complete trained R1 parent must already
exist on this VM. A new VM requires verified copies first. No duplicate weights are
uploaded by this packet. Requires 7 GiB plus installation payload free. The installer
checks every closed regular member and refuses existing experiment directories.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz.sha256 &&
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
name = 'cctv_completion_codeformer_same_codebook_probe_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz'
need(base.is_dir() and (not destination.exists()), 'Preserve any existing experiment; do not reinstall')
need(archive.stat().st_size <= 128 * 1024 ** 2, 'Execution archive size cap')

def safe(value):
    p = PurePosixPath(value)
    need(value and (not p.is_absolute()) and ('..' not in p.parts) and (':' not in value) and ('\\' not in value) and (str(p) == value), 'Unsafe member path')
    return p
with tarfile.open(archive, mode='r|gz') as stream:
    iterator = iter(stream)
    first = next(iterator)
    need(first.isreg() and first.name == name + '/protocol.json' and (0 < first.size <= 2 * 1024 ** 2), 'Expected first protocol member')
    body = stream.extractfile(first).read(2 * 1024 ** 2 + 1)
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == '51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-same-codebook-probe-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 128 * 1024 ** 2 and shutil.disk_usage(base).free >= 7 * 1024 ** 3 + total, 'Need7GiB plus installation payload free')
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
print({'complete': True, 'installed': str(destination), 'protocol_sha256': '51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec', 'training_launched': False})
PY
```

Expected complete:true, protocol 51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec, training_launched:false.
If it refuses, retain the error and all existing/staging folders. Do not bypass hashes.

## 3. Open tmux

Use the current session if already inside tmux. Otherwise:

```bash
tmux new-session -A -s completion_same_codebook_probe_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py --root . --protocol-sha 51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec --verify-transfer &&
bash scripts/run_completion_codeformer_same_codebook_probe_v1.sh 51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec
```

The first command verifies source/assets without model imports or calls. The shell
starts this distinct frozen-state diagnostic. There are zero optimizer objects,
updates, gradients, backwards or teacher-network calls. No checkpoint is created.

The 48 fixed paired adaptive TRAIN cases include 32 clear and 16 synthetic challenges.
Four ordinary app parity pairs run first (8 calls). All 48 actual-A outputs must match
saved R1 raw256 within 1e-6 and PNG bytes exactly before the other arms proceed.
Each of 4 arms then produces 48 outputs, for 200 planned whole-network calls including
parity (220 hard cap). Arm A: actual predicted codes with R1 fusion w1. Arm B: actual
predicted codes without fusion w0. Arm C: ORACLE clean-context codes with fusion w1.
Arm D: ORACLE clean-context codes without fusion w0. Every C/D filename and receipt
labels unavailable clean information. They cannot be generated from a real covered
input alone and are not a new deployable model or exact hidden identity.

Only 16 already retained labels from the SAME original inpainting codebook are reused.
Official HQ512 labels are incompatible and are not substituted. Observed context,
reviewed mask and copied outside pixels stay the same; clean RGB is never pasted.
All 192 outputs, 4 parity arrays, predicted/used codes and before/after 497 tensor hashes
are retained for independent review. Public/unpaired and FINAL data are not used.

Worker hard cap 15 minutes; cooperative stop 14 minutes. Start disk 7 GiB; floor 2 GiB;
CUDA allocation 20 GiB. Each raw output payload, raw tar stream and gzip archive has
its own 1536 MiB cap. Export cooperative cap 15 minutes. Maximum 196 NPZ and 1000 return
members; failed partial arrays/evidence stay retained. No automatic resume.
Worker exit 0 means the diagnostic finished, regardless of the oracle comparison
scores. A complete export means packaging succeeded. Neither means quality passed.

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/trainer.log
```

Detach safely with Ctrl+B then D. At the end:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-same-codebook-probe-v1-export.json
```

The successful archive.partial remains a hard link to the same inode, consuming no
duplicate payload storage. Keep all successful or failed evidence until audited.
No app adoption or further training follows automatically from this diagnosis.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command. Run on Windows, not in VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-same-codebook-probe-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-same-codebook-probe-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-same-codebook-probe-v1-export.json" "."
```

Return all three files for transport, saved-array/code/state-invariance and all-case
visual review before choosing a distinct next design. Trained R1 stays unchanged.
