# CCTV DGP: matched calibrated perceptual-feature pilot V3

Prepared 4 October 2026 for the active DGP-first Goal. Windows project:
`C:\xampp\htdocs\YEAR 4\Testing\`. Existing L4 VM bundle:
`~/forensic-dgp/cctv_dgp_vm_bundle/` (absolute path
`/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/`).
All actual training and gradient probes run on that VM. Local work is preparation,
frozen inference and independent audit. This pilot does not select an app default.

## Evidence and hypothesis

The completed V2 identity pilot and its independent audit are recorded in
local `CCTV_DGP_PILOT_RESULTS.md` (VM counterpart
`~/forensic-dgp/CCTV_DGP_PILOT_RESULTS.md` after explicit transfer). Its selected
identity epoch2 improves paired synthetic degraded PSNR from13.23 to16.22dB and
fixed ArcFace similarity from0.277 to0.327. These are paired photograph results.
The fixed24-case native CCTV review still finds soft faces and color shifts;
native useful restoration is not demonstrated. Seven input-selected insufficient
cases should request a clearer crop. The32 reserved native crops remain untouched.

The current VGG19 feature loss samples after ReLU. The
[ESRGAN paper](https://arxiv.org/abs/1809.00219) motivates sampling before activation
for brightness/texture supervision. Applying that loss idea to our DGP is a
hypothesis, not a demonstrated cause or guaranteed CCTV improvement. The existing
FPN/MobileNet student remains our trained restorer. The
[DeblurGAN-v2 paper](https://arxiv.org/abs/1908.03826) provides architecture context;
this pilot does not add its adversarial discriminator or replace the student.

A frozen10-case diagnostic found preactivation loss roughly7 times the control
magnitude. A separate frozen calibration uses64 original training references,
32/source, and zero validation/native cases. It sets per-tap preactivation scales:
`[0.4226472984724232, 0.2982341503327365, 0.18339591271073197, 0.08725710628575495]`.
This matches each tap's mean loss magnitude at the starting state; it does not
establish equal gradients. Signed taps are cloned before the following in-place
ReLU. The control retains the historical postactivation feature values.

Local evidence lives under
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_perceptual_diagnostic_v3\`
and `outputs\cctv_dgp_perceptual_calibration_v3\`. The VM overlay includes pinned
`perceptual_scales_v3.json` and `calibration_protocol_v3.json` at the bundle root.
These completed checks have zero optimizer updates and backward calls.

## Matched finite experiment

Both arms start from the audited V2 identity epoch2 checkpoint:
local `outputs\cctv_dgp_normfix_return_v2\outputs\cctv_dgp_pilot\camera_identity\best.pth`
and VM `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_pilot/camera_identity/best.pth`.
SHA256: `b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916`.
Fresh optimizers start new diagnostic epochs1–2; optimizer state is not resumed.

| Arm | Perceptual supervision | Identity coefficient |
| --- | --- | --- |
| `continued_postactivation` | Original post-ReLU taps, scales1 | 0.1 |
| `calibrated_preactivation` | Signed pre-ReLU taps, frozen training-only scales | 0.1 |

The continuation control measures the effect of additional updates. The changed
arm tests the feature policy rather than interpreting an identical historical
recipe as a new fix. Both arms retain902 training references (451/source),
110 validation references/550 fixed cases, prepared camera degradations, original
order/seed, batch8, Adam learning rates2e-6 backbone/1e-5 other parameters, weight
decay1e-5, and the original pixel/color/Sobel/identity weights. All five InstanceNorm
adapters retain the verified cloned-statistic correction.

Budget:226 optimizer updates/arm,452 total, two epochs/arm;90-minute cap from
model setup through the final validation. The launcher also enforces a90-minute
process timeout, allowing30 seconds for termination. The V2 run took5m12s on this
L4; allow approximately6–12 minutes for V3, subject to its measured timing.
The timing check at update32 stops if the remaining work and validation reserve
cannot fit. Stops also cover nonfinite loss/gradients, changed frozen state,
wrong lineage, unmatched starting tensors, output reuse and unexpected counts.
Partial files/logs stay in their distinct V3 locations; no automatic resume.

Each feature policy gets a VM-only batch8 zero-update backward probe with finite,
nonzero identity and perceptual input gradients. A batch6 evaluation verifies
exact starting state preservation. These are preflight probes, not trained epochs.
Do not run the VM runner locally without its `--verify` option.

Selection retains the unchanged strict source/profile, clear-input, paired
MSE/SSIM and identity safeguards against the audited starting model. `best.pth`
can be epoch0 fallback; inspect `best_selection.json`. A qualifying trained epoch
must still pass the same input-selected native development review and visual
inspection before any app default change. No native PSNR/SSIM/identity accuracy
is inferred from unpaired footage. Independent final review remains pending.

## 1. Upload from Windows Google Cloud SDK Shell

The additive archive reuses the existing data, weights, CUDA runtime and venv.
It contains only new files under `cctv_dgp_vm_bundle/`. No git pull or package
reinstallation is needed for this experiment. These local changes are not pushed.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-perceptual-v3.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-perceptual-v3.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Transfer one file per SCP command because the configured PuTTY backend rejects
multiple remote sources. Select the actual VM zone if prompted; it is not recorded.

## 2. Extract in Google Cloud SSH

```bash
cd ~ &&
sha256sum -c cctv-dgp-perceptual-v3.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_pilot/camera_identity/best.pth &&
test ! -e ~/forensic-dgp/cctv_dgp_vm_bundle/perceptual_protocol_v3.json &&
tar -xzf cctv-dgp-perceptual-v3.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

The guard deliberately stops if V3 was already extracted. Attach separately with
`tmux new-session -A -s dgp_training` if extraction completed earlier; do not delete
existing results or repeat extraction. Checksum files use Linux LF line endings.

## 3. Run inside tmux

```bash
cd ~/forensic-dgp/cctv_dgp_vm_bundle &&
source .venv/bin/activate &&
nvidia-smi &&
bash scripts/run_cctv_dgp_perceptual_vm_v3.sh
```

The script verifies every pinned source/data asset, both parent evidence records,
the starting checkpoint and current CUDA versions before the bounded new run.
It refuses an existing V3 output, log, environment or archive. Detach with Ctrl+B,
then D; reconnect with `tmux attach -t dgp_training`. Do not kill other tmux jobs.
There is no configured automated SSH connection from this workspace.

## 4. Download both completed files on Windows

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-perceptual-v3-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-perceptual-v3-results.tar.gz.sha256" .
```

The return contains `outputs/cctv_dgp_perceptual_v3/`, exact new protocol,
environment, CUDA version record and pilot log. The original failed attempt and
completed V2 directory remain separate. Preserve all returned files and checkpoint
selections, including rejected epochs.

## 5. Independently audit the return in local PowerShell

```powershell
Set-Location 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 scripts/audit_cctv_dgp_perceptual_results_v3.py --root outputs/cctv_dgp_vm_bundle_v1 --bundle-dir outputs/cctv_dgp_perceptual_vm_v3 --parent-return outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot --archive outputs/cctv-dgp-perceptual-v3-results.tar.gz --extract-to outputs/cctv_dgp_perceptual_return_v3 --verify-recognizer --receipt outputs/cctv_dgp_perceptual_return_v3/local_independent_audit.json
```

Next: independently reconstruct paired metrics, embeddings, update traces, loss
weights, checkpoint changes and selection; review ten-row previews and repeat the
unchanged24-case native development comparison only for eligible trained epochs.
Keep the32 reserved crops untouched until the method is frozen. DGP local-app
integration, input qualification, covering-family usefulness and final independent
assessment remain part of the active Goal; preparation or training alone is not
completion.
