# V15 fresh-image generalization probe — 4 October 2026

Completed and independently audited; all five grids reviewed. **Negative
generalization result:** fresh-image parity passed, but the fitted head degraded
new-face structure and all fixed-group embedding comparisons. No adoption.
L4 runner212.12s/full296.54s; local audit55.49s. Zero training calls; VM restored
to TERMINATED after collection and idle checks. See
[CCTV_DGP_GENERALIZATION_V15_RESULTS.md](CCTV_DGP_GENERALIZATION_V15_RESULTS.md).
The frozen execution recipe below is historical; do not rerun it unchanged.

Windows `C:\xampp\htdocs\YEAR 4\Testing\` ↔ VM `~/forensic-dgp/`.
Prepared local `outputs\cctv_dgp_generalization_vm_v15\` ↔ executed VM
`~/forensic-dgp/cctv_dgp_generalization_vm_v15/`.
This runbook's intended VM counterpart is `~/forensic-dgp/CCTV_DGP_GENERALIZATION_V15.md`
after document sync. Source/docs have not been committed or published.

## Frozen comparison

First reproduce all50 update1000 training-case outputs using newly computed
DGP RGB, original-input prior encoder features and base logits. Require raw
float difference≤2e-6 and exact delivered PNG equality with the audited cached
V14 r2 no-stat outputs. Failure stops before any validation inference.

Then evaluate **104 development validation references /520 cases**, unchanged
from V9:51 Asian-source references and53 FFHQ references, each with clear,
blur_lr24, lowlight_lr32, motion_lr48 and compound_lr24 inputs. Exact own
train/validation IDs, source bytes and target RGB overlap are checked. This is
held out from our conditioner fitting, not proof of unseen pretrained identities
or a blind final test; the same development cohort informed earlier pilots.

| Arm | Behavior |
|---|---|
| Input | Exact camera crop and its measured starting error |
| Retained DGP V2 | Frozen retained DGP restoration |
| Starting prior, no statistics | Original-input CodeFormer codes, frozen w0 decoder, no AdaIN/stat transfer |
| Trained conditioner, no statistics | Our V14 r2 update1000 code prediction, same frozen w0 decoder |

The prior comparison is an explicitly modified rendering diagnostic; it is not
the official default CodeFormer pipeline. No-stat rendering is frozen from the
training-only V14 review before validation. No predicted-stat mode, alternative
fidelity selection, teacher codes or target-conditioned rendering is used.
Targets enter pixel scoring and fixed-alignment recognizer scoring only.

## Budget and guardrails

Batch1, no AMP, zero optimizers/backwards/updates. L4 runner cap1200seconds;
training-free supervisor cap1800seconds including independent VM audit/export;
VRAM cap20GiB. A20-case timing estimate must fit the runner budget. No resume or
overwrite of partial runs. Existing Python3.10/CUDA runtime is reused and checked;
no dependency installer runs. Idle GPU/tmux checks precede launch.

Every delivered PNG and embedding is audited:1,560 model outputs plus520 input
cosines.150 predeclared raw previews and50 raw parity renders are independently
checked. Five ten-row grids use the first five sorted references per source,
chosen before execution;250 original256-cell pixels are compared to the saved
images. Other raw images are not exported, keeping transfer bounded.

All frozen module states must match. Expected actual forwards:570 DGP,570 prior
encoder,570 prior transformer classifier,570 direct conditioner,1,090 generator,
2,184 recognizer, zero unused V11 conditioner. No clean teacher model is loaded.
Native24 and reserved32 images remain unused.

Diagnostic guards preserve the historical source/profile rule: no MSE, SSIM or
fixed ArcFace degradation in any clear/degraded/source/profile group, plus at
least0.1dB pooled degraded PSNR gain. Report against both retained DGP and
starting prior. These flags never select/promote a checkpoint. Original-cell
visual structure review and an independent final assessment remain necessary.

## Execution and return

Protocol SHA256 `6c0e1de5bcb56b755b82a3289adb4905c591f49908f02365ad6ad5fbd151f9da`.
Verified execution archive:44,662,961 bytes /743 regular-file members,
SHA256 `5684c7d4a64f8913893cf179d02faefe9a96b35845f0831a5f9cba4c61fc3a19`.
Four local safety tests passed without importing a neural library.

The authorized assistant uses configured gcloud SSH/SCP to launch once:

```bash
cd ~/forensic-dgp/cctv_dgp_generalization_vm_v15
~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python -u \
  scripts/supervise_cctv_dgp_generalization_v15.py \
  --root "$PWD" \
  --parent ~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2 \
  --capacity ~/forensic-dgp/cctv_dgp_direct_codes_vm_v14_r2/outputs/cctv_dgp_direct_codes_v14_r2 \
  --expected-sha 6c0e1de5bcb56b755b82a3289adb4905c591f49908f02365ad6ad5fbd151f9da
```

Run in the dedicated `dgp_generalization_v15` tmux session after verified fresh
extraction. Collect VM-root `cctv-dgp-generalization-v15-results.tar.gz` and
`.sha256` to Windows `outputs\`, plus `supervisor_completion.json` to
`outputs\cctv_dgp_generalization_v15_completion.json`. Independently import once:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_generalization_v15.py `
  --archive outputs/cctv-dgp-generalization-v15-results.tar.gz `
  --completion outputs/cctv_dgp_generalization_v15_completion.json `
  --extract-to outputs/cctv_dgp_generalization_return_v15
```

The return audit is arithmetic on saved outputs and execution receipts; it does
not replay neural models or certify identity. Review all five grids, compare
source/profile and clear behavior, update the handoff, verify idle state and
restore the previously stopped VM. If generalization fails, prepare a separately
justified broader training recipe; do not repeat or adopt the ten-face fit.
