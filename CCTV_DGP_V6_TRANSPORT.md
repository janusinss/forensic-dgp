# Smaller lossless V6 transfer — 4 October 2026

## Current outcome: VM regeneration failed; exact-byte recovery succeeded

This transport passed local reconstruction, then correctly stopped on the VM
before any training when its NumPy/OpenCV/Pillow stack changed camera input pixels.
Do not use the historical materialization commands below on the current VM.
The failure inventory and original wrong derivative remain preserved. Recovery V2
uploaded 1,430 original PNGs in 100,123,557 bytes, restored all 2,709 asset hashes
and independently confirmed all 391 reduced-target pixels on the VM. No package
versions, target roles, camera inputs, model/loss or gate changed.

Recovery archive SHA256:
`2fd8719652ee3b6dbdf0be1f33c5a9c40100597f648eb1dbbbc89f7433f59c00`.
Recovery evidence is local at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_recovery_v2\`
and VM `~/forensic-dgp/cctv_dgp_targets_vm_v6/transport_recovery_v2.json`.
The original incorrect input is retained under VM `transport_failures_v1/`.

The portable driver binds the independent local source/input derivation receipt,
uses the original V6 verifier including all asset hashes/reduced pixels, and invokes
the unchanged trainer and output auditor. It does not rerun cross-platform camera
generation. Standalone preflight passed in 27.40 seconds, and the actual comparison
completed 196 updates in 259.18 seconds. All four trained epochs were rejected;
VM/local output and execution audits plus all five preview reviews are complete.
These supersede the old next-step snapshot. The VM has returned to its initial
stopped state after V6/V7 result collection; the full Goal remains active.
See local `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_TARGETS_V6.md`
↔ intended VM document `~/forensic-dgp/CCTV_DGP_TARGETS_V6.md` after sync.

## Preserved transport V1 procedure and local evidence

This changes transport only. The frozen V6 protocol remains
`0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c`.
All targets, inputs, teachers and runtime sources must reproduce the original
2,709 asset fingerprints before CUDA preflight or training.

The initial full upload averaged about 300 KB/s with roughly 19 minutes remaining.
That transfer was explicitly cancelled; its remote partial file is preserved.
The full local archive/checksum remain intact. The new transport is 63,567,794
bytes versus 376,810,544 full bytes (83% reduction). It ships 458 assets, reuses
821 exact cached assets and regenerates 1,039 prepared camera inputs plus 391
reduced targets. One cached camera module is bootstrapped before the materializer
imports it, accounting for 459 already-present verified files in its receipt.
Cache files are read only; no data, role, model recipe or quality gate changes.

The local independent materialization check completed in 30.00 seconds and
verified all original assets and byte-identical recipe, with zero model forwards,
backward calls or updates. Cross-platform VM reconstruction must pass the same
hashes; codec/version differences must stop it rather than silently changing data.
The materializer has a 180-second cap and refuses any existing file with wrong
bytes. Safe reuse of correct data files is not a training resume.

Transport archive SHA256:
`16fbe1bcbbe2a54839a858221e1efa5f6b58b3a3f0409ee38f9e043d46c4183a`.
Manifest SHA256:
`e7f12655f74fb94f5b43243acc0fd761f5326dfc519cfcabf01ad49bdda5c0d0`.
Materializer SHA256:
`b7050bcffa14ebfe94b9e0ca038c5cc32a0a484fc39f6b3bbe864c7f70834b09`.

Windows PowerShell upload:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing\outputs'
$env:CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE = 'C:\xampp\htdocs\YEAR 4\Testing\scratch\gcloud-windows-roots-v1.pem'
$env:CLOUDSDK_CORE_DISABLE_FILE_LOGGING = 'true'
gcloud compute scp cctv-dgp-targets-v6-transport-v1.tar.gz cctv-dgp-targets-v6-transport-v1.tar.gz.sha256 'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E
```

The SSH pin above matches six previously trusted instance keys and the offered
key on the API-verified current VM IP. Do not substitute another key on a mismatch.
The VM guest host-key attribute is unavailable; no API host-key match is claimed.
Windows pscp does not expand `~/` in this destination; use the absolute home path.

VM SSH, for a new extraction only:

```bash
cd ~
sha256sum -c cctv-dgp-targets-v6-transport-v1.tar.gz.sha256
test ! -e ~/forensic-dgp/cctv_dgp_targets_vm_v6 && tar -xzf cctv-dgp-targets-v6-transport-v1.tar.gz -C ~/forensic-dgp
cd ~/forensic-dgp/cctv_dgp_targets_vm_v6
source ../cctv_dgp_vm_bundle/.venv/bin/activate
```

Bootstrap the exact cached camera module, then materialize the fixed recipe:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
import shutil
root = Path.cwd()
name = 'scripts/cctv_camera_stress.py'
source = root.parent/'cctv_dgp_vm_bundle'/name
pin = json.loads((root/'targets_protocol_v6.json').read_text())['assets_sha256'][name]
assert hashlib.sha256(source.read_bytes()).hexdigest() == pin, 'Cached camera module changed'
destination = root/name
assert not destination.exists(), 'Preserve existing bootstrap'
shutil.copyfile(source, destination)
PY
timeout --signal=INT --kill-after=20s 200s python -u scripts/materialize_cctv_dgp_targets_v6_transport_v1.py --cache-root ../cctv_dgp_vm_bundle --manifest-sha256 e7f12655f74fb94f5b43243acc0fd761f5326dfc519cfcabf01ad49bdda5c0d0
```

After `all_frozen_asset_sha256_verified: 2709`, follow the zero-update CUDA
preflight and fixed launcher in `CCTV_DGP_TARGETS_V6.md`. Preserve failures and
existing output directories; do not delete evidence to repeat training.

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| Transport archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-targets-v6-transport-v1.tar.gz` and `.sha256` | `~/cctv-dgp-targets-v6-transport-v1.tar.gz` and `.sha256` |
| Transport/source/check receipts | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_transport_v1\` | Manifest/materializer inside `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |
| Local full reconstruction check | `C:\xampp\htdocs\YEAR 4\Testing\scratch\cctv_dgp_targets_v6_transport_check_v1\` | Not transferred |
| Materialized recipe | Original `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_vm_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |

**Historical next step at preparation:** VM checksum/reconstruction, CUDA preflight,
bounded comparison and return audit. The current outcome above supersedes this;
transfer completion does not establish a trained output improvement.
