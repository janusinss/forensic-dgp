# Matched CCTV DGP diagnostic: VM commands

Prepared 3 October 2026. The active Goal prioritizes our DGP for degraded CCTV
restoration; completion remains a separate supporting component. This is a new
bounded diagnostic, not the historical Phase 5 script and not proof of useful
output. Retain `dgp_zamboanga_final.pth` as the application baseline until an
audited candidate passes native/visual review. No local training is permitted.

## Frozen pilot

| Item | Prepared specification |
| --- | --- |
| Training | 902 references, 451 per source, original Phase 4 training memberships; two precomputed camera epochs with approximately 20% clear anchors |
| Validation | 110 inherited development references: 59 FFHQ, 51 Asian-source; 550 fixed clear/blur/low-light/motion/compound cases |
| Arms | Same Phase 3 starting tensors, inputs, order, optimizer and loss; identity coefficient 0 versus 0.1 |
| Budget | Batch 8; 2 epochs and 226 updates per arm; 452 updates total; 90-minute execution cap including preflight/validation; setup capped separately at 20 minutes for pip |
| Return | Starting and four raw epoch checkpoints, two diagnostic `best.pth` files, 2,750 prediction PNGs, 50 raw float previews, ten-row grids, source/profile metrics, embeddings, exact update trace, timings and fingerprints |

References are processed photos, mostly below 256 captured pixels. Enlarging them
does not create high-resolution truth. Some original dim/noisy/grayscale or soft
validation photos remain; only inspected input exceptions and measured geometry
are excluded. The fixed full-context gate V1 is retained as failed assumption
evidence; V2 checks observed feature support. Not all references were visually
reviewed, and there is no full historical identity overlap audit. Source names
are not ethnicity labels. No Zamboanga footage or reserved native crops are used.

Both arms use observed Charbonnier + 0.05 pooled color + 0.1 frozen VGG + 0.05
Sobel. There is no FAN/FFT term, EMA or AMP. Normalization running statistics stay
fixed. Adam uses backbone LR 2e-6, other LR 1e-5, weight decay 1e-5 and gradient
clip 1. Identity uses the same reference-derived five-point affine per case with
unsupported 112-crop context filled identically with RGB 128. The diagnostic is
`ArcFace_observed_fixed`, not identification accuracy. CUDA `grid_sample`
backward can remain nondeterministic despite common seeds/settings.

All inputs are prepared PNGs. Codec/random degradation differences between the
Windows preparation runtime and VM do not change the two arms' data. VM pixel
metrics use the exact exported RGB8 PNG. PSNR is derived from mean per-case MSE;
it is **not** the historical mean per-image PSNR. Zero-MSE PSNR is null with a
perfect-match flag. Selection requires at least 0.1 dB degraded agreement gain
against the current best, while every source/profile and clear group preserves
starting-baseline MSE, SSIM and reference cosine within frozen tolerances.
Epoch 0 remains eligible. A diagnostic `best.pth` can therefore contain unchanged
starting weights. Selection does not automatically replace the app checkpoint.

## 1. Upload from Windows Google Cloud SDK Shell

Local project: `C:\xampp\htdocs\YEAR 4\Testing\`.
Prepared package: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-vm-bundle.tar.gz`.
VM destination: `/home/janusdominic0/` (`~/`). Upload one remote file per command
because the Windows PuTTY backend rejected multiple remote sources previously.
The zone is not recorded; use the actual Console zone if gcloud prompts for it.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-vm-bundle.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-vm-bundle.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

These files are self-contained. A `git pull` alone cannot retrieve this prepared
data/weight bundle or local uncommitted code. If separately synchronizing already
pushed repository changes, use `cd ~/forensic-dgp && git pull --ff-only origin main`
outside the bundle. Do not assume the current new scripts have been pushed.

## 2. Verify and extract in Google Cloud SSH

```bash
cd ~
sha256sum -c cctv-dgp-vm-bundle.tar.gz.sha256 &&
mkdir -p ~/forensic-dgp &&
test ! -e ~/forensic-dgp/cctv_dgp_vm_bundle &&
tar -xzf cctv-dgp-vm-bundle.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

The checksum is ASCII with LF, avoiding the earlier Windows CRLF filename error.
If the destination already exists, stop and preserve its evidence rather than
extract over it. The source package directory locally is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_bundle_v1\`; its archive
prefix deliberately extracts as `~/forensic-dgp/cctv_dgp_vm_bundle/`.

## 3. Set up and run inside tmux

```bash
cd ~/forensic-dgp/cctv_dgp_vm_bundle
if [ -d ../venv ]; then source ../venv/bin/activate; fi
nvidia-smi
bash scripts/setup_cctv_dgp_vm.sh &&
source .venv/bin/activate &&
bash scripts/run_cctv_dgp_vm.sh
```

The runner first verifies all frozen files and performs one batch-8 CUDA
forward/backward with **zero optimizer updates**. It checks converted recognizer
parity against ONNX, finite identity-input gradients, frozen teachers and
unchanged student tensors/buffers. Only a passed preflight proceeds to baseline
validation and the matched training. It refuses CPU, Windows, another host/GPU,
changed assets, existing outputs and automatic resume. Timing at update 32
projects the remaining budget. On failure, preserve logs/partial state; report
the error before choosing a changed recipe. No blind retry.

If `ensurepip` is absent on the existing Python 3.10 VM, install its matching
venv package with `sudo apt-get install python3.10-venv` before setup. Stop if
the detected version differs. Setup uses the existing CUDA torch/torchvision;
it verifies that their version strings stay unchanged. It does not install new
CUDA torch weights or download model weights. Dependencies still require pip
network access. Actual CUDA behavior has not been verified locally.

Optional preflight-only command, if inspecting compatibility before committing
the full pilot, uses a separate directory:

```bash
python -u scripts/run_cctv_dgp_pilot_vm.py --preflight-only --output outputs/preflight_only
```

The full run repeats that zero-update compatibility batch in its own directory
before any training. Detach with Ctrl+B, then D. Reattach with
`tmux attach -t dgp_training`; do not start another copy of the runner.

## 4. Download the return from Windows Google Cloud SDK Shell

On successful completion and automatic file audit, the script exports both
files in `~/forensic-dgp/cctv_dgp_vm_bundle/`. The local receiving directory is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\`.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-results.tar.gz.sha256" .
```

The local forward-only return audit uses the original frozen preparation bundle:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 scripts/audit_cctv_dgp_pilot_results.py --root outputs/cctv_dgp_vm_bundle_v1 --archive outputs/cctv-dgp-results.tar.gz --extract-to outputs/cctv_dgp_return_v1 --verify-recognizer
```

The audit verifies archive paths/checksum, exact cohort/update order, exported
pixels, every saved embedding cosine, source/profile selection, unchanged norm
buffers and changed student tensors. Its optional ONNX preview check is bounded
to 52 local recognition forwards, never training. No completed archive exists
until the VM run succeeds; do not substitute an older detector return.

Next after return: review both arms' ten-row grids and all source/profile/clear
metrics; compare qualified candidates on the existing 24-case native development
gallery, retaining the 32 reserved images for final frozen evaluation. Retain
Phase 3 if no useful candidate passes. Broader training and DGP integration
depend on that evidence. Independent final review and the full app workflow
remain Goal requirements.
