# CCTV DGP objective diagnostic: audited V4 return

Verified 4 October 2026 in `C:\xampp\htdocs\YEAR 4\Testing\`.
The user supplied the completed local audit. A second audit invocation reproduced
the existing receipt exactly without overwriting returned files. The archive's
SHA256 and exact LF checksum/filename match.

| Evidence | Windows local | L4 VM |
| --- | --- | --- |
| Archive/checksum | `outputs\cctv-dgp-objective-v4-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-objective-v4-results.tar.gz` and `.sha256` |
| Extracted report | `outputs\cctv_dgp_objective_return_v4\outputs\cctv_dgp_objective_diagnostic_v4\results.json` | `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_objective_diagnostic_v4/results.json` |
| Separate local receipt | `outputs\cctv_dgp_objective_return_v4\local_independent_audit.json` | Local evidence; not a separate VM file until transferred |
| Detached analysis | `outputs\cctv_dgp_objective_v4_review\analysis.json` | Optional `~/forensic-dgp/outputs/cctv_dgp_objective_v4_review/analysis.json` after explicit transfer |

All local table entries resolve beneath `C:\xampp\htdocs\YEAR 4\Testing\`.
The archive is 45,787 bytes, SHA256
`3052c9a7f0e5ddefc5be26236382525b48e4b9bfcbbb2d30e6c51baafa3816d8`.
Returned report SHA256:
`e73b789f437483d046763cca60b07c8f67d81238dd993eab1976d689b5ff4ebb`.
Local receipt SHA256:
`1a4a1bd1c904b2d9c08f42837b91bf95828931b821d6832b19ff03aeaa113b05`.
Detached analysis SHA256:
`d1752be85b64a3ad86d8856f0542a20cc441e187d273384078a1d5595a10d90a`.

The L4 report records 21.75 seconds, ten four-image training-only forwards,
40 distinct training references, 50 autograd traversals and zero optimizer
updates. Peak allocated CUDA memory was 1,994,058,240 bytes. Student state
`d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3`
and both frozen teachers are unchanged. No validation/native/reserved cases,
checkpoint export, trained improvement or production promotion.

## Finding and its limits

Reconstruction means the weighted sum of pixel, color, postactivation VGG and
Sobel gradients. ArcFace is the separate weighted identity gradient. Parameter
gradients oppose identity for blur and low light in both sources; the full
ordinary weighted gradient also has a negative identity dot product for all
four groups. The other six parameter groups do not show this opposition.

| Source/profile | Reconstruction–identity parameter cosine | Identity multiplier needed for a zero total dot, relative to 0.1 |
| --- | ---: | ---: |
| Asian / blur lr24 | −0.2958 | 2.5266 |
| Asian / low light lr32 | −0.3477 | 3.3865 |
| FFHQ / blur lr24 | −0.3261 | 2.7326 |
| FFHQ / low light lr32 | −0.4720 | 1.6648 |

The analogous restoration-image gradients do not oppose identity in these ten
groups. Image-gradient magnitude alone would miss the parameter-space issue.
These are four-case batches at one student state, not evidence of every training
batch, optimizer step, curvature or a causal explanation of native failures.
Adam's preconditioning/momentum/weight decay, clipping and finite steps can
change actual behavior. The audit rebuilds the twenty serialized Gram summaries
and pins cohort/source/state claims; it does not replay CUDA gradients locally.

## Changed next experiment

The predeclared V5 comparison retains the V2 starting state, original split,
degradation/order/seed, Adam settings, postactivation VGG and selection guards.
One arm raises identity weight from 0.1 to 0.4; the other keeps 0.1 and applies
the symmetric two-objective form of
[PCGrad](https://arxiv.org/abs/2001.06782). The paper projects conflicting original
task gradients before summing; grouping this DGP's reconstruction and identity
terms is our adaptation. Its published experiments are not evidence of DGP CCTV
output improvement.

Detached V4 Gram arithmetic gives a nonnegative combined identity dot for both
declared methods in all ten groups. This is a rationale for a bounded test, not
a prediction of training success. Use 226 updates/arm, 452 total and a 30-minute
training-process limit. Reuse existing data/environment; preserve all failed
V3 checkpoints and the executed V4 protocol. No unchanged failed arm is rerun.
Exact transfer/run/return/audit commands:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_CONFLICT_V5.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_CONFLICT_V5.md` after overlay transfer.

Next: run the verified V5 pilot on the L4, independently audit returned outputs,
then review paired previews and eligible native development candidates. No useful
new model is established yet. The 32 reserved native crops remain untouched;
the DGP-led application workflow and independent final review remain required.
