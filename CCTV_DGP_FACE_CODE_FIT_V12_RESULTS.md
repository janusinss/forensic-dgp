# V12 r2: trained face-code conditioner, no useful upgrade

Status, 4 October 2026: the bounded L4 pilot, transfer, independent local audit
and all ten original-cell preview reviews are complete. **Do not adopt these
conditioner checkpoints or start a larger version of the same recipe.** Actual
training occurred, but it did not establish useful restoration of degraded faces.
The active CCTV restoration Goal remains incomplete.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; Linux VM root:
`~/forensic-dgp/`. Document counterpart after explicit sync:
`~/forensic-dgp/CCTV_DGP_FACE_CODE_FIT_V12_RESULTS.md`.

## What actually ran

Only our 455,072-parameter conditioner trained. The retained V2 DGP, CodeFormer
encoder/transformer/codebook/renderer, clean VQGAN teacher and recognizer remained
frozen. Ten original **training** references, five per source, generated fifty
fixed cases across clear, blur, low light, motion and compound degradation.
Twelve epochs, batch two, produced **300 optimizer updates / 600 exposures**.
The CUDA preflight performed one backward call and zero optimizer updates;
training added 300 backward calls. All ten conditioner parameter tensors changed.

Trainer: **125.22 seconds**. VM audit: **15.75 seconds**. Full supervised
training/audit/export: **177.28 seconds**. Peak allocated VRAM: **1,428,103,680
bytes**. The small number of training images explains the short runtime; this
was a fitting diagnostic rather than a full-data training run.

Initial zero-conditioner CUDA output deviation was exactly zero. Recorded
frozen states and actual component-call counts remained stable. No validation,
native development or reserved input was accessed. No `best.pth` was created,
no checkpoint was selected, and no application default was replaced.

The initial V12 execution failed after 8.49 seconds because its loader used
Python's newer `hashlib.file_digest` API on Python 3.10. It made zero backward
calls and updates. Its source, protocol and audited failure remain intact.
Separate r2 changed the loader compatibility only, retaining the same data and
fitting recipe. The original pilot was not resumed.

## Independent verification and output review

The successful archive is **468,092,580 bytes**, SHA256
`f22261197d26f316ff2fdfb005dd147c0ddb3cfff0e5b4e17ce00a2e007c0614`.
r2 protocol SHA256:
`06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846`.
Results JSON SHA256:
`95fb1993e93aba1972ae89d024df1ad8d50574d1fefd7eacf5dc977b6c9c4796`.

The **21.64-second local arithmetic audit** rebuilt 300 PNG/raw compositions,
300 saved-embedding cosines, 150 code-loss/feature probes, 2,560 teacher labels,
500 grid cells and all 300 update traces / 600 exposures. It also checked both
changed snapshots. It constructed no neural model and made zero local backward
calls or updates. CUDA gradients and encoder/recognizer forwards were not
replayed locally. A separately downloaded terminal source proof verified all
116 pinned assets, including sources absent from the return archive.

Windows sandboxed extraction initially failed at an audit-file write. That
partial directory is preserved. Authorized extraction to a **fresh** directory
and the audit completed; the successful evidence path uses `_verified`.

All ten 10-row grids were inspected at their original 256-pixel cells: w0 and
w1 for each profile. Clear faces stay broadly coherent but some eyes, lips,
teeth and expressions change. In degraded cases, the conditioner does not remove
the baseline's displaced or repeated facial features, invented glasses, changed
mouths, hair-like artifacts and mismatched skin detail. Some FFHQ rows are
coherent; several Asian-source and compound rows retain conspicuous artifacts.
Neither source establishes a useful trained upgrade. This is an assistant
development review, not independent final assessment.

## Learning diagnosis: loss is not output quality

Numbers below come from a separate no-forward analysis of already audited
saved logits. They are **training-fit** measurements. Mean code statistics give
each case equal weight; they are not a pooled token count.

| Training cases | Code CE, initial → update 300 | Correct observed codes | Mean top-1 probability | Code entropy |
| --- | --- | --- | --- | --- |
| All 50 | 6.93786 → 5.61599 | 5.8868% → 4.3847% | 0.12933 → 0.03988 | 4.01253 → 5.44407 |
| Degraded 40 | 7.86234 → 6.07713 | 2.0214% → 2.0361% | 0.09143 → 0.02828 | 4.34420 → 5.69249 |
| Clear 10 | 3.23997 → 3.77145 | 21.3483% → 13.7792% | 0.28097 → 0.08628 | 2.68583 → 4.45039 |

Degraded accuracy is effectively unchanged. Clear controls lose code accuracy
and increase CE. The aggregate CE decrease accompanies reduced confidence and
higher entropy, rather than reliable correct top-1 predictions. Across observed
tokens, 260 initially wrong codes become correct while 457 correct codes become
wrong. Only 22.91% of initial/final top-1 choices agree. These observations do
not prove a particular causal bottleneck or that a different prior would work.

Delivered w1 aggregate PSNR from group mean MSE changes **14.7878 → 14.8047 dB**;
SSIM **0.59403 → 0.59943**; fixed observed ArcFace **0.35109 → 0.34735**.
Asian-source similarity changes **0.34181 → 0.32502**; FFHQ-source similarity
changes **0.36037 → 0.36968**. A tiny photometric gain does not resolve the
visible structural failures. These metrics are not CCTV identity verification.

## Evidence locations and VM state

| Evidence | Windows below the root | Linux VM |
| --- | --- | --- |
| Actual execution bundle | `outputs\cctv_dgp_face_code_fit_vm_v12_r2\` | `~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2/` |
| Result archive and `.sha256` | `outputs\cctv-dgp-face-code-v12-r2-results.tar.gz` | Bundle root, `cctv-dgp-face-code-v12-r2-results.tar.gz` |
| Independently audited results | `outputs\cctv_dgp_face_code_fit_return_v12_r2_verified\outputs\cctv_dgp_face_code_fit_v12\` | Bundle `outputs/cctv_dgp_face_code_fit_v12/` |
| Local audit | Return root `local_independent_audit.json` | Bundle output `independent_audit_vm.json`; local audit is not synced |
| Learning analysis and visual review | `outputs\cctv_dgp_face_code_fit_analysis_v12\analysis.json`; `outputs\cctv_dgp_face_code_fit_review_v12.json` | Intended same relative evidence paths after explicit sync |

The verified output contains `update0`, `update100` and `update300/conditioner.pth`.
These are **diagnostic adapters**, not complete DGP models or selected upgrades.
Keep the required original DGP and declared prior weights separate.

All results and terminal receipts are collected. GPU, tmux and project-process
checks found no competing work. The assistant restored the VM's prior stopped
state; Cloud confirms **TERMINATED**, last stop
`2026-10-04T06:33:01.149-07:00`. Stop receipt:
Windows `outputs\cctv_dgp_face_code_v12_r2_vm_stop.json`.
No git publication or document sync is claimed.

## Next concrete step

Completed afterward: V13 isolated wrong code prediction and degraded rendering
statistics. See `CCTV_DGP_FACE_CODE_CONTROLS_V13_RESULTS.md` beneath both document
roots after sync. The paragraph below records the original V12 next step.

Separate **code prediction** from **rendering** before more fitting. Prepare an
inference-only training-cohort control using saved initial/learned code choices
and clean teacher code labels, with versus without original-input AdaIN/fidelity
features. Teacher-label arms deliberately consume clean targets and are oracle
diagnostics, never deployable outputs or evidence of learned restoration. Freeze
that protocol before evaluating it; preserve these checkpoints and all old gates.
Its outcome should identify whether the code mapping, transfer of degraded input
features, or both require a distinct repair. Do not repeat the failed recipe or
move directly to held-out/native evaluation of it.
