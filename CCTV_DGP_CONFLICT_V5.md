# CCTV DGP V5: identity weight versus gradient conflict

Prepared 4 October 2026. V4 has arrived and its independent audit reproduces.
V4 used 40 training cases, 50 CUDA autograd traversals, 21.75 seconds and zero
updates. Both sources showed reconstruction/identity opposition for blur and
low light. This motivates the two changed training recipes below; it does not
prove causation or establish useful restored output.

## Fixed comparison

| Item | Specification |
| --- | --- |
| Start | Audited V2 identity epoch 2, SHA256 `b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916` |
| Arm 1 | `identity_weight04`: identity coefficient 0.4, ordinary weighted gradient sum |
| Arm 2 | `two_objective_pcgrad`: identity coefficient 0.1, symmetric projection of reconstruction and identity gradients when their global parameter dot product is negative |
| Data/optimization | Original 902 training/110 validation references, fixed cases/order/seed, postactivation VGG, Adam backbone 2e-6/head 1e-5, weight decay 1e-5, clipping at 1, batch 8, no AMP/EMA; frozen normalization and teachers |
| Budget | Two epochs/226 updates per arm, 452 total; 8 zero-update preflight and 904 training autograd traversals; 30-minute training-process cap |

The 0.4 coefficient is a predeclared training-only choice: the largest V4
four-case multiplier needed to make the ordinary gradient's identity dot
nonnegative was 3.3865, relative to coefficient 0.1. Rounding to 4 is a small
ablation, not proof of a generally suitable weight. PCGrad groups the existing
pixel/color/VGG/Sobel objective into reconstruction and keeps ArcFace separate.
It projects BOTH original gradients, then sums; nonconflicting gradients stay
unchanged. This is the two-objective form of the
[original PCGrad method](https://arxiv.org/abs/2001.06782), applied to this DGP as
an experimental adaptation. Adam, clipping, stochastic batches and finite steps
can still regress identity or reconstruction. No preservation guarantee is made.

Expected training-process time is approximately 8–15 minutes, estimated from
V3's 5.55 minutes and the extra gradient traversal; V5 has not run yet. Timing
at update 32 must fit the 30-minute limit. Preverification, CPU return audit and
archive compression are additional. No environment reinstall is required.

The original, V2, V3 and V4 assets/returns stay intact. The 32 reserved native
CCTV crops are not read. No native inputs are used in this pilot. The application
default remains unchanged until independently reviewed native outputs support it.

## 1. Upload in Windows Google Cloud SDK Shell

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM bundle: `~/forensic-dgp/cctv_dgp_vm_bundle/`.
Upload these two new additive files separately. They have not been pushed to Git;
`git pull` alone cannot install them.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-conflict-v5.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-conflict-v5.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Select the actual VM zone if requested. The project has not recorded the zone.

## 2. Extract in Google Cloud SSH

```bash
cd ~ &&
sha256sum -c cctv-dgp-conflict-v5.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_objective_diagnostic_v4/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_vm_bundle/conflict_protocol_v5.json &&
tar --keep-old-files -xzf cctv-dgp-conflict-v5.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

The archive contains only new V5 assets. `--keep-old-files` also prevents
overwriting a partial extraction. If extraction previously completed, use
`tmux new-session -A -s dgp_training` directly. Preserve partial files and
diagnose extraction failures rather than deleting evidence to repeat a run.

## 3. Run inside tmux

```bash
cd ~/forensic-dgp/cctv_dgp_vm_bundle &&
source .venv/bin/activate &&
nvidia-smi &&
bash scripts/run_cctv_dgp_conflict_vm_v5.sh
```

The launcher verifies unchanged torch/torchvision/CUDA, at least 8 GiB VRAM,
all pinned parent/V2/V3/V4 assets, the completed V4 report and reproduced audit.
It checks both gradient policies on one eight-image training batch each with
zero updates, verifies a six-image forward preserves normalization, runs the
matched pilot, audits exact exported PNG metrics/checkpoint changes/selection/
trace arithmetic and 52 recognizer preview forwards, then exports the return.
All actual backward probes/training are guarded to `forensic-dgp-thesis` and L4.

Stop on nonfinite or empty gradients, changed state/teachers/buffers, changed
sources/cohort, unexpected traversal/update counts or deadline. On failure,
preserve `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_conflict_v5/` and
`pilot_conflict_v5.log`. Local equivalents after transfer live under
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_return_v5\`.
No automatic resume or unchanged rerun is permitted. Detach with Ctrl+B, D;
reattach with `tmux attach -t dgp_training`.

## 4. Download after the completion message

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-conflict-v5-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-conflict-v5-results.tar.gz.sha256" .
```

Expected return size is approximately 350 MB, inferred from V3's 347 MB;
the full return includes checkpoint candidates and all 2,750 validation PNGs.

## 5. Independent local audit and output review

From Windows CMD or PowerShell at `C:\xampp\htdocs\YEAR 4\Testing\`:

```text
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_dgp_conflict_results_v5.py --root outputs/cctv_dgp_vm_bundle_v1 --bundle-dir outputs/cctv_dgp_conflict_vm_v5 --perceptual-bundle-dir outputs/cctv_dgp_perceptual_vm_v3 --diagnostic-bundle-dir outputs/cctv_dgp_objective_vm_v4 --parent-return outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot --v3-return outputs/cctv_dgp_perceptual_return_v3/outputs/cctv_dgp_perceptual_v3 --v4-return outputs/cctv_dgp_objective_return_v4/outputs/cctv_dgp_objective_diagnostic_v4 --archive outputs/cctv-dgp-conflict-v5-results.tar.gz --extract-to outputs/cctv_dgp_conflict_return_v5 --receipt outputs/cctv_dgp_conflict_return_v5/local_independent_audit.json --verify-recognizer
```

Local extracted result:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_return_v5\outputs\cctv_dgp_conflict_v5\`.
VM result: `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_conflict_v5/`.
The auditor rebuilds exported PNG metrics, embedding cosines, gradient receipt
arithmetic, exact cohort/update/traversal traces, frozen-buffer/checkpoint changes
and `best.pth` decisions. It does not replay CUDA gradients locally or independently
prove every recorded training operation. No local restoration training is run.

An epoch qualifies only if every source/profile retains baseline MSE, SSIM and
fixed ArcFace within the original numerical tolerances, and aggregate degraded
PSNR gains at least 0.1 dB over the retained best. `best.pth` selects epoch 0
when no trained candidate passes. Review all five fixed ten-row grids. Only an
eligible, visually credible candidate proceeds to the frozen 24-case native
development comparison. Keep all 32 reserved cases untouched. Synthetic gains
alone do not establish useful CCTV restoration or Zamboanga-specific performance.

Next after return: independently audit V5, review paired outputs and evaluate
eligible candidates on native development cases before any app/default promotion.
The full DGP-led workflow and independent final review remain incomplete.
