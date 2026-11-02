# HISTORICAL COMPLETED PILOT — DO NOT RELAUNCH

27 October 2026. This exact manual pilot ran and failed final quality at32updates.
Read COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md in FACE_MASK_COMPLETION.
R1 remains selected. Preserve the actual state, archive and original commands;
do not rerun, resume or append epochs. No successor is launched by this guide.
The original guide bytes follow unchanged as historical evidence.

---

# Predictor-only CodeFormer completion V1: manual finite L4 pilot

RELEASED FOR MANUAL TRANSFER ONLY. Preparation has launched no training or VM action.
R1 remains selected; no candidate is qualified by packet creation or training completion.
Source archive: 73748641 bytes, 670 pinned payloads plus protocol.
Archive SHA256: 1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3.

## 1. Upload from Windows Google Cloud SDK Shell

Run on Windows, not inside VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and safely install in VM SSH

Require 8 GiB plus exact installation payload free, the original inpainting checkpoint,
and full trained R1 state already present on this VM. This packet duplicates neither
large checkpoint. A different VM needs verified dependencies first. The fresh
installer refuses existing roots and retains failed staging; do not bypass a refusal.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz.sha256 &&
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
name = 'cctv_completion_codeformer_predictor_ce_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz'
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
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == '15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-predictor-ce-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 256 * 1024 ** 2 and shutil.disk_usage(base).free >= 8 * 1024 ** 3 + total, 'Need8GiB plus installation payload free')
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
print({'complete': True, 'installed': str(destination), 'protocol_sha256': '15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b', 'training_launched': False})
PY
```

Expected complete:true, protocol 15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b, training_launched:false.

## 3. Open tmux

Use your current session if already inside tmux. Otherwise:

```bash
tmux new-session -A -s completion_predictor_ce_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_predictor_ce_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py --root . --protocol-sha 15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b --verify-transfer &&
bash scripts/run_completion_codeformer_predictor_ce_v1.sh 15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b
```

Transfer verification uses no model calls. The shell starts one distinct finite
predictor-only CE test from full R1: maximum 32 Adam updates /128 backwards on exactly
128 selected FIT observations. This is not an epoch over the retained 256-case pool.
Only 114 existing predictor tensors train ;383 encoder/fusion/prior tensors stay exact.
R1's 114 predictor Adam moments map by name from step 32 to 64; frozen 164 encoder and 54 fusion
slots remain unchanged at 32. LR1e-5 and existing optimizer settings remain fixed.
No optimizer reset, latent/image/GAN loss, fusion/encoder learning or oracle inference.

Reuse 80 exact cached original same-codebook label sets: 64 FIT targets and 16 monitor
diagnostics. The 16 monitor sets never optimize. No new teacher call/label generation.
All 32 clear/16 challenge paired cases read out at 0,8,16,32; 32 public unpaired cases at 0/32.
Four fresh ordinary app parity pairs and all 48 initial R1 raw/PNG matches must pass
before updates. Expected 256 evaluation outputs +4 parity arrays +128 FIT logits arrays,
plus 80 retained teacher NPZ copies. Public outputs have no hidden clean-target metrics.
If a quality stop occurs at8/16, its stopped-cycle public readout replaces the planned
final32 readout; no third public readout or extra work after a resource stop is allowed.

Planned 384 network calls =128 code-only FIT  +248 decoded evaluation  +8 parity decoded.
Fullgeneration class cap 256, code-only 128; global hard 420 checked before dispatch.
168 empty bypasses remain exact with no neural call. Empty saved raw/codes are marked
absent. Save every FIT logit/label/case/cycle/microbatch for independent CE replay.
Snapshots 8/16 are safety checks, not selectable winners. Final 32 must pass separate
clear/challenge>=1%PNGholeerror gains, all oldRAW/PNGhole/boundary subgroup nonworse,
exact outside-source preservation, and the new strict-gray macro/fourgeometry channel-
spread guard. Gray color errors, lower CE or higher code agreement do not imply quality.

Finite worker 1800 seconds hard/1680 cooperative with 120 seconds save reserve;20 GiB CUDA.
Require 8 GiB free after installation, 2 GiB floor. Whole output 2.5 GiB, terminal or failed
partial 1 GiB, teacher 24 MiB, metadata 8 MiB including 4 MiB stdout, PNG 256 KiB each.
Streamed rawtar and gzip each 2.75 GiB; export 900 seconds. No second rawtar disk copy.
All source/parent checkpoints and stopping evidence remain preserved. Save once; internal ZIP/member/primitive-pickle/CRC checks run on the closed bounded
partial before commit. The total 1 GiB cap is enforced before every write. A rejected
partial remains evidence and is never automatically retried. Actual failed-prior 165 tensors and partial 114 gradients can be retained within cap.
A hardkill/storagefailure can still prevent recoverability; no automatic resume.

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_predictor_ce_v1_vm/trainer.log
```

Detach with Ctrl+B then D. At completion:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_predictor_ce_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-predictor-ce-v1-export.json
```

Worker exit 2 is a quality stop, not a qualified new model. Complete export means
packaging only. Independent weights/Adam/RNG/logit/RAW/PNG audits and all-case visual
review, broader development, user acceptance and final review precede any adoption.
The successful archive.partial is one same-inode hardlink, not duplicate disk data.
Rare same-inode terminal commit aliases are recorded and preserved at source while
one payload is archived; distinct checkpoint files are never hash-only deduplicated.

## 5. Download from Windows Google Cloud SDK Shell

One remote source per command; run on Windows, not in VM SSH.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-predictor-ce-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-predictor-ce-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-predictor-ce-v1-export.json" "."
```

Return all three files for independent admission. Do not automatically repeat or
continue this historical pilot. The working R1 app and restoration remain unchanged.
