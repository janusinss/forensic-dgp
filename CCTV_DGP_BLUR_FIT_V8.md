# V8: isolated blur fitting diagnostic

**Current status, 4 October 2026:** training, VM audit/export, download, independent
local audit and four-grid visual review are complete. Training took 45.95 seconds
for 1,000 updates; complete execution took 67.87 seconds. The assistant restored
the initially stopped VM after collection and competing-job checks; Cloud state
is `TERMINATED`. The full Goal remains active, with all actual training on the VM.

Frozen 4 October 2026. Actual training uses the existing NVIDIA L4
`forensic-dgp-thesis` in `us-central1-a`; local preparation and arithmetic audits
perform zero model forwards/backward/optimizer updates. The user restored
VM-only actual training after learning that configured Cloud CLI access permits
the assistant to start the VM. Direct start/SSH/SCP/stop are available.

## Purpose and finite protocol

V7 mixed ten training pairs across five degradation profiles. Both objectives
failed its fixed blur/motion criterion, despite lower error for dark examples.
V8 isolates the same two blurred training pairs to test fitting at matched and
increased exposure. This cannot establish generalization or identify one sole
cause: V7 also differs in its case mixture and training exposure.

| Fixed factor | Value |
| --- | --- |
| Blur training cases | `e1_tr_ffhq_00178`, `e1_tr_ffhq_01210` |
| Collateral evaluation | All ten existing V7 training pairs; no validation/native forwards |
| Initial tensors | V2 identity starting state `d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3` |
| Objective | Per-image observed RGB MSE; identical to V7 pixel-only arm |
| Optimizer | Fresh Adam; backbone `2e-5`, other `1e-4`; weight decay `1e-5`, clipping `1` |
| Execution | Batch 2, 1,000 updates; fixed snapshots at 20, 100, 1,000 |
| Timing | Trainer cap 600 seconds; trainer/audit/export supervisor cap 900 seconds |
| Preflight | VM host/device guard; finite nonzero CUDA gradients, unchanged starting state/buffers, zero updates |
| Training-fit criterion | Final blur mean MSE at most 80% of the common baseline |

At update 20 each blur pair has the same number of exposures as in V7's
100-update mixed arm. The final endpoint increases exposure to 1,000 per pair.
These are fixed diagnostic checkpoints: no `best.pth`, native testing,
production promotion or early choice of a flattering intermediate output.
All five frozen normalization adapters, teacher weights and current VM package
versions remain unchanged. Exact prepared input/target bytes are copied;
camera images are not regenerated across Windows/Linux library versions.

## Locations and fingerprints

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| Prepared runtime | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_vm_v8\` | `~/forensic-dgp/cctv_dgp_blur_fit_vm_v8/` |
| Package/checksum | Local `outputs\cctv-dgp-blur-fit-v8.tar.gz` and `.sha256` | `/home/janusdominic0/cctv-dgp-blur-fit-v8.tar.gz` and `.sha256` after upload |
| Training output | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_return_v8\outputs\cctv_dgp_blur_fit_v8\` | `~/forensic-dgp/cctv_dgp_blur_fit_vm_v8/outputs/cctv_dgp_blur_fit_v8/` |
| Return archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-blur-fit-v8-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_blur_fit_vm_v8/cctv-dgp-blur-fit-v8-results.tar.gz` and `.sha256` |
| Runbook | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BLUR_FIT_V8.md` | Intended `~/forensic-dgp/CCTV_DGP_BLUR_FIT_V8.md` after document sync; not claimed copied |

Executable protocol SHA256:
`d58cf74c41d4b87db294286fd04c04292d00730ec8aa97d86af93169c5bc1d2b`.
Initial design SHA256:
`1537890efea27da622e495d2b071cbd770c54a5e160a4f847b6a8af39a76e9ac`.
Package SHA256:
`841cf4f80c32b6c2ad885a4051ef54ba0b00104a62048db21c0a42784ac8c68b`.
Package size: 2,070,573 bytes. Three original large weight assets are reused only
after checking their existing VM copies against the frozen asset fingerprints.
All 42 inherited exact asset copies, archive members, three Python 3.10 source
parses and the local training guard passed independent offline verification.

## Execution and return review

The assistant uses a dedicated `dgp_blur_fit_v8` tmux session, the existing
`~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python`, and the bundled bounded
supervisor. The launcher refuses an existing V8 root or another active GPU job.
Do not repeat/resume the same frozen execution automatically.

The separate result auditor checks all 40 output PNGs and raw float predictions,
50 saved embedding arrays, all 1,000 update records, the three changed model
states, actual preview cells, fixed endpoint arithmetic and exact ten-image
baseline equality with V7. It does not replay CUDA gradients or recognizer
inference locally. Inspect all four ten-row grids before choosing the next
experiment. Training fit alone cannot qualify the thesis model or finish the
active DGP-first Goal. Keep all prior gate failures and the 32 reserved native
cases untouched.

## Audited outcome and next decision

Mean PNG MSE below uses two cases per profile. Only the blur pair was fitted in
V8; all ten cases are historical training-role examples, not validation.

| Training profile | Baseline | Update 20 | Update 100 | Fixed final 1,000 |
| --- | ---: | ---: | ---: | ---: |
| Blur | 0.0095371 | 0.0066236 | 0.0040778 | 0.0018928 |
| Motion | 0.0050750 | 0.0048443 | 0.0071293 | 0.0134428 |
| Low light | 0.0592767 | 0.0758448 | 0.0791535 | 0.0790062 |
| Compound | 0.0561817 | 0.0762839 | 0.0783256 | 0.0778644 |
| Clear | 0.0009495 | 0.0013547 | 0.0055669 | 0.0180006 |

The fixed final blur ratio is `0.1984673389595779`, passing the declared fit
criterion. Mean blur SSIM rose from 0.53227 to 0.78881; recognizer cosine rose
from 0.23504 to 0.83317. These describe memorized training examples, not recovered
identity or generalization. At matched 20 exposures, V8 blur MSE is 0.0066236
versus V7 pixel-only 0.0201019; case/task mixture differences prevent attributing
this to one cause. The early V8 recognizer cosine actually decreases to 0.19041.

Both fitted faces become visibly clearer, especially eyes, mouth and hair.
Other eight examples develop severe contrast/texture/color artifacts; clear and
motion performance regresses substantially. The model can fit these images,
but the resulting checkpoint is unsuitable for production. No `best.pth`,
validation/native pass, app/default promotion or model improvement is claimed.

Return: 85,445,552 bytes; SHA256
`96a80124de9115ed95995c1e6a7c9acc075185380c06841580e244ee95e3b21d`.
Returned results SHA256:
`b5a10bd131bccbde8388d866b0231e1eb6a7eb5bcc7abb6ec58edc9fe7b1951f`.
Passed local audit: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_return_v8\local_independent_audit.json`,
SHA256 `39838f5bb7f3cfe05b7f1764bc600efcc7e125d0f9a945eb3a421109e1f42a6b`;
VM counterpart: `~/forensic-dgp/cctv_dgp_blur_fit_vm_v8/outputs/cctv_dgp_blur_fit_v8/independent_audit_vm.json`.
Assistant development review/source binding: local return
`assistant_preview_and_execution_review.json`, SHA256
`ef93411b4c3c6eacd614d39eca912fd909d08c1d751c3211df25df49515e49d8`;
this review remains local. Terminal completion and VM control receipts are local
`outputs\cctv_dgp_blur_fit_v8_terminal_completion.json` and
`outputs\cctv_dgp_vm_control_v8_20261004.json`; VM terminal receipt is at the
runtime root `supervisor_completion_v8.json`.

**V9 follow-up:** the Asian source review is complete (390 accepted, 28 covering
and 33 quality exclusions from 451 originals). The distinct balanced mixed-source
V9 executable and all 6,232 archive member hashes are verified. Its 781 training
references cover all five profiles with unchanged 104-reference validation and
bounded 7,820 updates. VM start succeeded; exact transfer/remote preflight precede
training. See `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_MIXED_V9.md` after document sync. V8's overfit remains
unpromoted. The next paragraph is the preserved earlier preparation decision.

Next: review the original Asian replay sources before freezing a broader finite
mixed-source pilot with clear/degradation replay and unchanged validation guards.
The zero-model coverage audit at local
`outputs\cctv_dgp_training_coverage_v9_audit.json` checked 1,353 Asian assets,
451 target pixel hashes and 1,684 prepared camera inputs in 6.53 seconds. V6
trained only 391 HQ FFHQ references; the original split also contains 451 Asian
training references. Their native minimum edges are all below 256, and they
require separate covering/quality review. Keep the 53 HQ FFHQ / 51 Asian
validation references unchanged. No direct source-hash collision was found;
subject identity separation is unestablished. The next training recipe is not
executable or started. Start the VM only after its new verified package is ready.
