# Spatial code intervention V1: manual zero-update diagnostic

PREPARED, UNRUN. R1 remains the working completion model. No automatic training.
Execution archive 13610328 bytes; 166 pinned payloads plus protocol.
ArchiveSHA256 `24f480add44e4fe8176a68a3b67321b11b4baee0ebb43d0a3790fefc434549a5`; protocolSHA256 `b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123`.

This distinct diagnostic separates whether clean-context code information helps
the mask-touched latent positions or the visible-context positions. Both H/V
use UNAVAILABLE CLEAN INFORMATION and cannot be deployed on covered inputs alone.
No model is trained. It does not replace R1 or repeat the rejected CE recipe.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

Require6GiB free after installation, plus the exact packet payload during extraction.
The existing original weights, full R1 state and admitted earlier same-codebook
probe A/C outputs must remain on this VM. No duplicate models or A/C arrays are
uploaded. If a source/control is absent, stop and retain the error; do not skip it.
The installer checks every closed regular member and refuses an existing target.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz.sha256 &&
command -v timeout >/dev/null &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test -f ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth &&
test -d ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs &&
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile, uuid
def need(ok, message):
    if not ok: raise ValueError(message)
base = Path.home() / 'forensic-dgp'
name = 'cctv_completion_codeformer_spatial_code_intervention_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz'
need(base.is_dir() and not destination.exists(), 'Preserve existing experiment; do not reinstall')
need(archive.stat().st_size <= 128 * 1024 ** 2, 'Execution archive size cap')
def safe(value):
    p = PurePosixPath(value)
    need(value and not p.is_absolute() and '..' not in p.parts and ':' not in value and '\\' not in value and str(p) == value and len(value.encode('utf-8')) <= 320, 'Unsafe member path')
    return p
with tarfile.open(archive, mode='r|gz') as stream:
    iterator = iter(stream)
    first = next(iterator)
    need(first.isreg() and first.name == name + '/protocol.json' and 0 < first.size <= 2 * 1024 ** 2, 'Expected first protocol member')
    body = stream.extractfile(first).read(2 * 1024 ** 2 + 1)
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == 'b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-spatial-code-intervention-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(row['path']).as_posix(): row for row in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum(row['bytes'] for row in p['files'])
    need(total <= 128 * 1024 ** 2 and shutil.disk_usage(base).free >= 6 * 1024 ** 3 + total, 'Need6GiB plus exact installation payload free')
    stage = base / (name + '.installing-' + uuid.uuid4().hex)
    stage.mkdir()
    with (stage / 'protocol.json').open('xb') as output: output.write(body)
    seen = {first.name}
    for member in iterator:
        need(member.isreg() and member.name in expected and member.name not in seen, 'Unlisted/duplicate/nonregular member')
        parts = safe(member.name).parts
        need(parts[0] == name, 'Wrong member root')
        row = expected[member.name]
        need(member.size == row['bytes'], 'Member size differs')
        path = stage.joinpath(*parts[1:]); path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(); count = 0
        with stream.extractfile(member) as incoming, path.open('xb') as output:
            for block in iter(lambda: incoming.read(8 * 1024 ** 2), b''):
                count += len(block); need(count <= row['bytes'], 'Oversized member')
                digest.update(block); output.write(block)
        need(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Member SHA differs')
        seen.add(member.name)
    need(seen == set(expected) | {name + '/protocol.json'}, 'Missing payload members')
    need(not destination.exists(), 'Experiment appeared; preserve staging')
    stage.rename(destination)
print({'complete': True, 'installed': str(destination), 'protocol_sha256': 'b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123', 'training_launched': False})
PY
```

Expected complete:true, protocolb27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123, training_launched:false.
Keep failed install staging and all existing evidence; never bypass the checks.

## 3. Open tmux

Use your current session if already inside tmux. Otherwise:

```bash
tmux new-session -A -s completion_spatial_code_intervention_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_spatial_code_intervention_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_spatial_code_intervention_v1_vm.py --root . --protocol-sha b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123 --verify-transfer &&
bash scripts/run_completion_codeformer_spatial_code_intervention_v1.sh b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123
```

Verify-transfer checks files and196 reused A/C control payloads without importing
the neural model. The new run uses exact original and R1 weights; all497tensors
stay frozen. Four ordinary app parity pairs use8calls. Four fresh full-C bridge
cases use4calls and must match retained C raw/PNG outputs and A/C predicted
indices. H and V each produce
48outputs, making108plannedwholemodelcalls/120hard. Typed limits prevent any arm
from silently using another arm's allowance. No optimizer objects/steps, gradients,
backwards, teacher-network calls, new labels or ordinary-success checkpoint writes.

H substitutes the existing oracle indices only at mask-touched16x16positions;
V substitutes them at the complementary visible-context positions. Both keep
observed inputs, exact reviewed masks and R1 fusionw1 fixed. These labels come
from the SAME original inpainting codebook; the incompatible HQ512prior is not used.
The full48-case A/C cohorts are hash-verified and reused rather than rerun; only
the four declared C bridge calls are new. The48pairedcases are adaptiveTRAIN
(32clear/16syntheticchallenges), not
public/unpaired hidden truth, nativeCCTV performance or independentFINAL review.

All96newPNGs,104NPZs, actual predicted/used indices, selectors and before/after
497statehashes are retained. This is a bounded diagnosis, not a quality-selected
new model. No result is generated from unavailable clean information in the app.

Worker15minhard/14mincooperative with60ssave reserve;6GiBstart/2GiBfloor;
20GiBCUDAallocated cap. Wholeoutputs1GiB; each rawtar/gzip1280MiB. Export has
15-minute cooperative checks and an owned external900-second GNUtimeout KILL
deadline. If export is killed, retain its partial archive and all outputs; an
export timeout is not successful packaging and does not authorize relaunching
the model or repeating this diagnostic. The shell checks timeout exists before
any model work.
No ordinary checkpoint is saved. Only an unexpected model mutation may retain a
bounded actual497-tensor+RNG failure state or partial, at most384MiB cumulative;
that is evidence, not an adopted/resumable model. Failed arrays/partials remain.
Never auto-resume, repeat a historical pilot or promote from exportcomplete.

The worker intentionally writes receipts rather than progress lines; trainer.log
may stay empty until termination. Let the foreground run finish within its bounded
15-minute limit. An empty tail is not evidence that it stopped. Do not launch a
second process or repeat the command because the log is quiet. After the normal
prompt returns, inspect supervisor.json and the export receipt shown below.

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_spatial_code_intervention_v1_vm/trainer.log
```

Detach with Ctrl+B thenD. When finished:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_spatial_code_intervention_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-spatial-code-intervention-v1-export.json
```

Exit0 means the diagnostic finished, regardless of measured improvement; an
exportcomplete receipt means packaging succeeded. Neither qualifies a model.
The successful archive.partial is a hardlink to the same inode, not a second
payload copy. Retain all successful/failed evidence until independently audited.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command, on Windows rather than inside VM SSH:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-spatial-code-intervention-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-spatial-code-intervention-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-spatial-code-intervention-v1-export.json" "."
```

Return all three files for independent transport/state/indices/RAW/PNG and all-case
visual review. R1, RESTORATION and IMPLEMENTATION remain unchanged.
