"""Write exact commands from an independently audited, unrun transfer file."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_preparation'
GUIDE = ROOT / 'CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md'


def main():
    build = json.loads((PREP / 'build_receipt_v1.json').read_bytes())
    audit = json.loads((PREP / 'archive_review_v1.json').read_bytes())
    if not (build['complete'] is True and audit['complete'] is True
            and audit['archive_review_pass'] is True and audit.get('unresolved_blockers') == []):
        raise ValueError('Complete sealed-archive independent admission required')
    archive = ROOT / build['archive']['path']
    payload = archive.read_bytes()
    if len(payload) != build['archive']['bytes'] or hashlib.sha256(payload).hexdigest() != build['archive']['sha256']:
        raise ValueError('Reviewed transfer bytes changed')
    stage = 'cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
    prefix = 'cctv-codeformer-u64-gate-aligned-pilot-v1'
    protocol = build['protocol']['sha256']
    size = f"{len(payload):,}"
    text = f'''# Original trained u64: four finite gate-aligned updates

**Prepared for your manual L4 run; not launched.** This distinct pilot continues
the exact original trained u64, including its frozen R2/U96 base. It tests a
joint fidelity/structure update direction instead of the failed mixed Adam loss.
U96 remains in the app. No new model is qualified by preparing this file.

[Diagnosis](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md).
[Design](CCTV_CODEFORMER_U64_GATE_ALIGNED_V1_DESIGN.md).

Transfer: **{size} bytes**. Archive SHA256:
`{build['archive']['sha256']}`.
Protocol SHA256: `{protocol}`.

Require **8 GiB free** before launch; retain 4 GiB throughout. Keep the completed
parent `cctv_codeformer_u64_cleananchors_continuation_v1_vm`, its 469 static
files and both installed weight files. The pilot reads them without resuming
its failed training. Local audit aliases are not accepted on the VM.
No VM connection or live-space check was performed during preparation.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c {prefix}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
test ! -e ~/forensic-dgp/{stage} &&
tar -xzf {prefix}-execution.tar.gz -C ~/forensic-dgp
```

The checksum file uses Linux line endings. If any check fails, retain the exact
message; do not delete a previous or partial stage to force installation.

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_u64_gate_aligned_v1
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/{stage} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u64_gate_aligned_pilot_v1.py --root . --protocol-sha {protocol} --verify-transfer &&
bash scripts/run_gate_aligned.sh {protocol}
```

The worker deliberately uses the completed parent's directory as its read-only
model-loading context. Every new output is written to the new pilot stage.

Detach using **Ctrl+B**, then **D**. Reattach with step 3. From another SSH
terminal, inspect the live log without starting another worker:

```bash
tail -F ~/forensic-dgp/{stage}/run.log
```

Estimate **8–16 minutes**, unmeasured for this route. Finite limits: at most
**four updates, 540 full forwards and 512 gradient queries**. Worker stop:
1800 seconds; external watchdog: 1860 seconds plus at most 20 seconds cleanup.
Each phase is limited to 180 seconds. Export is separate: 300 seconds internally,
330 externally plus bounded cleanup. Limits: 10 GiB RAM, 18 GiB reserved CUDA,
2 GiB new outputs and 2 GiB return archive, with the 4 GiB free-space reserve.

There is no Adam object or momentum. This is still actual training: one fixed
global scale applies a certified gradient-descent direction to 36 head tensors.
Its formula is `alpha = 0.99 * 2.5e-5 / maxabs(direction)`: the 0.99 factor
provides numerical headroom, fixed before metrics. There is no line search.
Maximum actual coordinate step is 2.5e-5; cumulative change from original u64
is capped at 1e-4. FP32-rounded signs and bounds are checked before each update.
All 539 R2/U96 tensors must remain exact. A failed check stops without retries.

After every update, all 56 quality cases are saved and checked. RAW and PNG
fidelity/structure means must not increase against original u64 or the preceding
accepted state. All inherited guards remain in force, including the original
1% blur-structure requirement at terminal update 4. Clear and low-light MAE
must then improve. A failure rejects the pilot; an earlier state is not selected
afterward. Preserve the failed state and traceback.

Final supervisor fields:

- `worker_exit: 0`: the finite TRAIN screen passed; independent return review is
  still required before useful-restoration or model-promotion claims.
- `worker_exit: 2`: a quality gate stopped training. Download the retained evidence.
- `worker_exit: 1` or another nonzero code: an execution/resource failure needs
  diagnosis. Do not relaunch or append updates.
- `export_exit: 0`: the return archive was exported. This does not imply training
  success. Unknown writer cleanup skips export; the download files can be absent.

## 5. Download from Windows Google Cloud SDK Shell

Wait for the supervisor's export result. Use **one remote source per command**
because Windows PuTTY rejects multiple remote sources in one invocation.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-export.json" "."
```

If export failed or was skipped, retain the logs and supervisor receipts; do not
repeat training to manufacture an archive. Return review must check actual
rounded updates, gradients, all states, RNG/sampler records, saved RAW/PNG metrics,
individual regressions and every planned visual sheet.

This adaptive TRAIN pilot does not qualify native CCTV, wild images, FINAL or
Zamboanga performance. Broader DEV failures stay binding. Original u64 and all
predecessors remain retained. Reviewed Protect/Apply/bypass is a separate input
preparation study; automatic matching and completion development remain deferred.
The full goal remains ACTIVE/incomplete.
'''
    with GUIDE.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps({'complete': True, 'guide': str(GUIDE), 'bytes': GUIDE.stat().st_size,
                      'manual_training_launched': False}))


if __name__ == '__main__':
    main()
