# V6 target-bandwidth comparison: audited results, 4 October 2026

The L4 completed 196 updates in 259.18 seconds. All four trained epochs fail the
frozen selection safeguards. Both `best.pth` files retain the V2 starting tensors;
no checkpoint was promoted. Higher-resolution supervision improves the clear-face
example but does not resolve degraded-face softness in this bounded comparison.
The full DGP-first Goal remains active.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
Linux runtime: `~/forensic-dgp/cctv_dgp_targets_vm_v6/`.
The latest user instruction is VM-only training, including short experiments.
Local preparation, inference and audits remain permitted.

## Verified execution and transport

The assistant started the initially stopped L4 through the existing Google Cloud
CLI, verified idle GPU/unchanged CUDA packages, and pinned its previously trusted
SSH key. A slow full upload was cancelled and its remote partial preserved.
Transport V1 reproduced every original byte locally but correctly failed on the
VM's different NumPy/OpenCV/Pillow camera pixels before training. Recovery V2
transferred 1,430 original PNGs, restoring all 2,709 asset fingerprints; all 391
reduced-target pixels also match under the original verifier.

A separately pinned portable supervisor binds the independent local derivation
receipt and invokes the unchanged V6 verifier/trainer/auditor on exact prepared
inputs. No data roles, model/loss, installed packages or selection gates changed.
Standalone CUDA preflight passed in 27.40 seconds with zero updates. The main pilot
records two additional preflight autograd calls, 196 backward calls and 196 steps.
Training, VM audit and export together took 385.92 seconds within the supervisor's
1,800-second cap; the original pilot cap remained 1,200 seconds.

The local independent audit passed once: 2,600 actual PNGs, 50 raw float previews,
3,120 embedding-cosine rebuilds, 196 update records and all checkpoint/selection
bindings. Each candidate changed 158 unique named parameter tensors; model buffers
and teacher states stayed frozen. The execution audit additionally binds exact-byte
recovery, the standalone preflight and the portable driver. These are arithmetic,
state and source audits, not independent CUDA-gradient or recognizer-forward replay.

| Evidence | Windows local | Linux VM |
| --- | --- | --- |
| Returned results/checkpoints | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_return_v6\outputs\cctv_dgp_targets_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/outputs/cctv_dgp_targets_v6/` |
| Result archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-targets-v6-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/cctv-dgp-targets-v6-results.tar.gz` and `.sha256` |
| Passed local output audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_return_v6\local_independent_audit.json` | VM counterpart `~/forensic-dgp/cctv_dgp_targets_vm_v6/outputs/cctv_dgp_targets_v6/independent_audit_vm.json` |
| Transport/execution proof | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_execution_evidence_v1\` | Original receipts under `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |
| Metrics/assistant preview review | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_review_v6\` | Local review; not yet transferred |

Protocol SHA256: `0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c`.
Return archive SHA256: `3a12b481a4950ed551c98da6955b463f33f63ee48273c7e1e3dfe20245d13c77`.
Returned results SHA256: `6d0709d4ae4880858466a56a8ea36f7d6fae618fa64d44a057f0a2d7ab223aa7`.
Local output receipt SHA256: `2f8003d9ef0b71115cbfc82d1986f037d59cee91bf5e72d8a90b5b43cf4887f7`.
Local execution receipt SHA256: `bfc6c61bfa92fd87a2437d167d41cf4caff5eabc1052cf078a0bf26aa4b0ce7b`.

## Common-target paired measurements

All stages share 53 approved HQ FFHQ validation references and 51 unchanged
Asian-source sentinels, five profiles each. Training uses 391 approved FFHQ
references. PSNR below comes from mean observed-support MSE across 416 degraded
cases, excluding the 104 clear controls. It is synthetic development evidence,
not real Zamboanga CCTV accuracy. The embedding score is a limited similarity
proxy, not identity verification. Do not compare these PSNR values with older
target/cohort versions as a training gain.

| Stage | Degraded PSNR | SSIM | Embedding similarity | Selected |
| --- | ---: | ---: | ---: | --- |
| Common V2 starting tensors | 16.0267 | 0.61930 | 0.33011 | Retained |
| Reduced target, epoch 1 | 16.4211 | 0.62134 | 0.32848 | No |
| Reduced target, epoch 2 | 16.6432 | 0.62394 | 0.32717 | No |
| HQ target, epoch 1 | 16.3891 | 0.62104 | 0.32881 | No |
| HQ target, epoch 2 | 16.5727 | 0.62289 | 0.32645 | No |

Aggregate pixel gains coexist with source/profile regression. For HQ epoch 2:

| Source/profile | PSNR delta | SSIM delta | Embedding delta |
| --- | ---: | ---: | ---: |
| Asian sentinel / blur | -0.9039 | -0.00362 | -0.00674 |
| Asian sentinel / motion | -1.1741 | -0.00526 | -0.00485 |
| HQ FFHQ / blur | -0.4069 | -0.00002 | +0.00471 |
| HQ FFHQ / motion | -0.4253 | -0.00086 | -0.00135 |
| Asian sentinel / low light | +0.7280 | +0.00571 | -0.01555 |

Clear HQ FFHQ PSNR rises from 30.2521 to 32.8542 dB, but Asian clear-control
PSNR falls from 32.2871 to 31.0569 dB. The strict original per-source/profile
MSE/SSIM/embedding guard rejects all epochs despite aggregate PSNR gains.
`best.pth` serialization has SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`;
its tensors match starting state
`d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3`.
Serialization bytes differ from the original starting file; this is not learning.

## Visual review and follow-up diagnostic

All five fixed ten-row grids were inspected at their actual 256-pixel cells.
They cover two preview references under five profiles; all 520 cases are checked
numerically. The HQ arm recovers noticeably more clear-input eyes/hair/skin detail
in the FFHQ example. Degraded outputs remain soft, especially blur and compound
cases; smoothing motion artifacts does not restore fine facial detail. Tone
differences remain. This is assistant development review, not independent final
review or proof of native usefulness. No V6 epoch was forwarded to native data.

A separate training-only arithmetic audit checked 782 existing training inputs
against the pinned DGP output range `clamp(input + 0.5*tanh(residual), 0, 1)`.
It used zero model/gradient/update operations and completed in 19.77 seconds.
Within the fixed face footprint, unavoidable MSE floors are 0.00000588 for blur,
0.00000179 for motion, 0.00001872 for low light and 0.00014632 for compound cases.
These small floors do not justify changing the residual range alone. They do not
measure actual saturation or identify the cause of training regression. Receipt:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_output_range_v7\training_range_audit.json`
↔ intended VM evidence copy only after explicit transfer.

**Follow-up status:** V7 audit/preview review and the isolated V8 fit diagnostic
are complete. V8 fits two blurred examples, but other faces regress; no new
production model is qualified. Current next step is source-only Asian replay
review before a broader finite mixed-source pilot. See
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BLUR_FIT_V8.md` ↔ intended VM
`~/forensic-dgp/CCTV_DGP_BLUR_FIT_V8.md` after document sync. Preserve these rejected
epochs and unchanged guards. The 32 reserved native crops remain untouched; DGP app
integration, useful native restoration, covering-family checks and independent final
review remain unfinished.
