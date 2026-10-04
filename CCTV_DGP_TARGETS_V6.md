# V6 controlled DGP target-bandwidth pilot — 4 October 2026

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing L4 VM root: `~/forensic-dgp/`.
The full DGP-first Goal remains active; this is one bounded development experiment.

## What changes

Two arms start from the same independently audited V2 identity epoch-2 tensors.
`reduced_target` reconstructs a target made from canonical HQ256 downsampled to
128 and enlarged to 256 with PIL LANCZOS. `hq_target` reconstructs canonical HQ256
downsampled from its verified official 1024 source. Common input bytes, identity
target embeddings, historical affine geometry, optimizer, normalization, order,
evaluation targets and selection guards remain matched. The factor is target
bandwidth, not a simultaneous new loss/architecture or an exact legacy-thumbnail
replication. Source review was frozen before these model outputs.

Training: 391 approved FFHQ references in their historical training role.
Validation: 53 approved HQ FFHQ references plus 51 unchanged Asian-source
regression sentinels, five fixed camera profiles each (520 cases). Asian sentinels
are not HQ targets. Report their metrics separately. No role is reassigned, no
reserved native image is used, and earlier exposure/full identity overlap remain
unestablished. Public/synthetic data cannot establish real Zamboanga performance.

Budget: two epochs per arm, batch eight with a seven-image tail, 49 updates per
epoch, 98 per arm, 196 total. Fresh Adam per arm uses backbone LR `2e-6`, other
parameters `1e-5`, weight decay `1e-5`, clip norm `1`; no EMA/AMP. Reconstruction
loss and identity weight `0.1` retain the prior postactivation VGG configuration.
The same canonical HQ identity target is used in both arms, including the reduced
reconstruction arm. Frozen InstanceNorm adapters preserve stored statistics.

The main pilot has a 1,200-second cap, a measured projection after 16 updates and
no automatic resume or repeat. Standalone zero-update CUDA preflight has a
180-second launch timeout. Nonfinite/empty gradients, buffer/teacher changes,
source/checksum mismatch, budget overrun or unexpected traversal stop the run.
Preserve failure receipts and partial states before changing a recipe version.

## Current verification

Preparation finished in 76.10 seconds: 2,709 pinned assets, zero model operations.
The independent local data audit passed in 33.67 seconds, rebuilding 1,302 input
PNGs, 391 reduced targets and 444 canonical sources. Five offline controls pass;
five Python sources parse for Python 3.10. Standalone CUDA preflight passed on the
L4 in 27.40 seconds, with zero updates. Both arms completed 196 total updates in
259.18 seconds; none passed the unchanged selection guards. VM and independent
local output/execution audits passed, and all five paired grids were reviewed.
The return is local; see `CCTV_DGP_TARGETS_RESULTS_V6.md`. Data quality
and training completion do not establish improved native output.

Protocol SHA256: `0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c`.
Archive SHA256: `d829012e9a6c2c109335914da207c1fca6c205c7409b756345d41057b064e4c9`.
Archive size: 376,810,544 bytes. The archive contains canonical targets, prepared
inputs, teachers, V2 starting checkpoint, source provenance and pinned runtime.
The original 1024 sources remain locally preserved; their target derivation was
independently verified before transfer.

The latest user instruction restores VM-only training, including small pilots,
after the assistant's configured ability to start the VM was explained. The brief
permission for small local training is superseded. Preparation, inference and
audits remain local; no local CUDA installation is required for this work.

## Transfer and run

Transport history is in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_V6_TRANSPORT.md`
↔ intended document copy `~/forensic-dgp/CCTV_DGP_V6_TRANSPORT.md`.
The 63.6-MB transport passed locally but failed before training on different VM
camera-library pixels. Exact-byte recovery V2 then restored all 2,709 original
asset hashes and preserved the wrong regenerated file. The current VM is already
materialized and trained: do not repeat extraction, preflight or fitting there.
The full-archive method below is a reference for a fresh root only.

Only one execution is allowed in the new output directory. Reuse completed
extraction after checking the package; never delete prior evidence to rerun it.
The assistant can execute these operations with the existing CLI connection.
The commands also support manual operation.

Windows PowerShell transfer, using the existing verified public trust bundle:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing\outputs'
$env:CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE = 'C:\xampp\htdocs\YEAR 4\Testing\scratch\gcloud-windows-roots-v1.pem'
$env:CLOUDSDK_CORE_DISABLE_FILE_LOGGING = 'true'
gcloud compute scp cctv-dgp-targets-v6.tar.gz cctv-dgp-targets-v6.tar.gz.sha256 cctv_dgp_targets_v6_local_preparation_audit.json '..\scripts\run_cctv_dgp_targets_v6_portable_v1.py' 'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E
```

Inside Google Cloud SSH, after the existing instance is running:

```bash
cd ~
sha256sum -c cctv-dgp-targets-v6.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/cctv_dgp_targets_vm_v6 &&
tar -xzf cctv-dgp-targets-v6.tar.gz -C ~/forensic-dgp &&
cp -n ~/run_cctv_dgp_targets_v6_portable_v1.py ~/forensic-dgp/cctv_dgp_targets_vm_v6/scripts/ &&
cp -n ~/cctv_dgp_targets_v6_local_preparation_audit.json ~/forensic-dgp/cctv_dgp_targets_vm_v6/local_source_derivation_audit_for_v6.json &&
tmux new-session -A -s dgp_training_v6
```

Inside tmux, first perform the zero-update CUDA check:

```bash
cd ~/forensic-dgp/cctv_dgp_targets_vm_v6
source ../cctv_dgp_vm_bundle/.venv/bin/activate
nvidia-smi
python -u scripts/run_cctv_dgp_targets_v6_portable_v1.py --preflight-only
```

After the preflight completion message in a genuinely fresh execution, the portable
driver verifies unchanged CUDA packages and all frozen assets, binds the independent
local input/source derivation receipt, then invokes the unchanged trainer/auditor:

```bash
python -u scripts/run_cctv_dgp_targets_v6_portable_v1.py
```

Use the already configured CUDA environment; no pip/torch reinstall is needed.
The original shell launcher is preserved as historical source; do not use it on
this VM because it regenerates camera pixels with a different library stack. The
portable driver changes transport verification only, with original model/loss/gates
unchanged. Driver SHA256: `7e8f08473f417c6e5254d9e05a5c884225e4821cd121ad8af1ca0bc34fc6dc4b`.
The standalone preflight and pilot are different bounded executions. The main
pilot records two zero-update autograd calls plus 196 training backward/update
calls; the separate standalone preflight performs two more autograd calls and
zero updates. Its evidence remains in its separate output directory.

## Selection and return

`best.pth` uses the unchanged strict source/profile PNG guard: no group MSE
regression, no SSIM/embedding-similarity degradation beyond existing numerical
tolerance, identical case/pair counts and at least 0.1 dB aggregate degraded
PSNR gain over the incumbent. Evaluate every candidate on the common new targets;
do not compare PSNR numbers across different GT versions as a training gain.
If none qualifies, retain the starting baseline. The audit reconstructs actual
PNG metrics, saved-embedding cosines, 196 update traces, checkpoint changes,
50 raw-float previews and five fixed 10-row grids. It does not independently
replay CUDA gradients or establish identity/usefulness.

No checkpoint is promoted into the application by this pilot. Review paired
images and source/profile failures first. A metric-eligible research candidate
may then undergo the frozen 24-case native development comparison; keep all
32 reserved cases untouched until the final evaluation protocol. Useful visible
structure and the full app/covering workflow still require verification.

Windows PowerShell download after export is complete:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing\outputs'
gcloud compute scp 'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_targets_vm_v6/cctv-dgp-targets-v6-results.tar.gz' . --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E
gcloud compute scp 'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_targets_vm_v6/cctv-dgp-targets-v6-results.tar.gz.sha256' . --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E
```

Windows PuTTY accepts one remote source per SCP invocation. Use the two calls
above; a combined archive/checksum remote-source call is unsupported.

Audit once per new archive in local PowerShell:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_dgp_targets_v6.py --root outputs/cctv_dgp_targets_vm_v6 --archive outputs/cctv-dgp-targets-v6-results.tar.gz --extract-to outputs/cctv_dgp_targets_return_v6 --receipt outputs/cctv_dgp_targets_return_v6/local_independent_audit.json
```

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| Fixed recipe/data/runtime | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_vm_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |
| Standalone CUDA preflight | Local receipt after transfer | `~/forensic-dgp/cctv_dgp_targets_vm_v6/outputs/cctv_dgp_targets_v6_preflight/` |
| Completed pilot | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_return_v6\outputs\cctv_dgp_targets_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/outputs/cctv_dgp_targets_v6/` |
| Return archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-targets-v6-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/cctv-dgp-targets-v6-results.tar.gz` and `.sha256` |

**Next:** follow the audited V7 diagnostic in `CCTV_DGP_CAPACITY_V7.md`, then prepare
isolated blur fitting before another full-cohort recipe. No trained V6 epoch is
eligible for native forwarding/default promotion under the frozen gates. Training
completion alone does not complete the active Goal. The VM has returned to its
initial stopped state after result collection; existing outputs must be preserved.
