# V19 partial failure audit — 5 October 2026

**Latest closure:** the separately corrected V19 r2 has now completed all 520
development cases. Independent audit and original-sheet review pass; canonical
normalization parity is restored, while 30 automatic preservation checks still
fail. Both jobs are closed and preserved. See
[V19 r2 returned results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).
The following retains the original failure and correction-preparation record.

**Location:** the first development case, `va_ffhq_28544_clear`, in the frozen
V19 inference runner at line136. The exact error is
`Fresh DGP PNG differs from declared V15 baseline`.

**Cause confirmed by the separate returned diagnostic:** NumPy float32 division
before GPU transfer and CUDA scalar division produce slightly different input
floats. The first development case matches the V15 PNG exactly with the former;
the latter differs at four colour-channel values by one byte. Maximum raw
difference between the routes is 1.07288361e-6. The fixed V18 training case instead
matches CUDA scalar division exactly. Both conventions must remain explicit.

**Fix prepared:** separate immutable inference-only V19 r2 retains V18's spatial
path and adds a V15-encoded canonical DGP comparison forward per development
case. Save raw/PNG outputs before parity checks, export the internal DGP base
separately, and retain every original quality/parity gate. Eight regressions,
independent transfer audit and actual frozen-package preflight pass. Manual VM
inference and full development review remain pending:
[V19 r2 runbook](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md>).
Original V19 and the diagnostic are closed; no historical repeat or model
adoption is supported.

## Original partial-return finding, before the diagnostic

**Cause established from the original failure alone:** a fresh baseline PNG failed exact equality.
The worker passed all50 fresh training parity cases first. It did not save the
first failing development raw/PNG before raising. The returned evidence cannot
establish the size or numerical cause of that difference.

**Diagnostic prepared at that earlier milestone:** compare only the two historical normalization paths on
three fixed cases, saving normalized tensors/raw RGB/PNGs before comparison.
Use the separate, bounded, inference-only diagnostic described in
[CCTV_DGP_INPUT_SELECTION_V19_PARITY_DIAGNOSTIC_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_PARITY_DIAGNOSTIC_VM.md>).
No correction, relaxed parity requirement, retraining or V19 resume is authorized
by this result. The original frozen sources, protocol, weights and failure remain.

## Verified downloaded evidence

The167,982,054-byte failure archive matches its checksum, sidecar and exported
receipt. SHA256:
`f6847aa742acee19c21a16df7f44dce2456b052793cf7fa154a62cd59abcb16b`.
Protocol SHA256:
`2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2`.
Safe import retains all144 original assets byte-identically in
`outputs/cctv_dgp_input_selection_failure_return_v19/`.

The original L4 supervisor reports31.591647246s before exporting the failure.
No complete development prediction, timing projection, grid, final state/count
receipt or full output audit exists. Original `execution.json` records inference
only, no optimizer and zero backwards/updates. Six initial state hashes match
the audited lineage. Final state/counter verification is unavailable because
the run stopped before the final receipt. The source contains no training path.

The separate5.5873s local partial audit verifies:

- Fifty ordered training decisions: ten retained-DGP and forty fixed-terminal.
- All100 fresh training raw arrays and observed PNG compositions: maximum raw difference0.
- Four training cache probe schemas and all104 fresh target embeddings; maximum baseline difference2.1234154701e-7, within the frozen2e-6 requirement.
- All728 camera/target/support images against V15: exact pixel equality and matching development case metadata.
- Original protocol, all144 assets, six initial state hashes, failure stage and absence of completed development outputs.

This audit uses zero neural forwards, training updates or backwards. It proves
partial evidence integrity, not full V19 execution or useful restoration.
Receipts:
`outputs/cctv_dgp_input_selection_v19_failure_audit.json` and
`outputs/cctv_dgp_input_selection_failure_return_v19/local_import_and_audit.json`.

## Earlier normalization hypothesis and required check

The original V15 tensor helper performs NumPy float32 division by255 before
GPU transfer. V19 performs PyTorch scalar division after GPU transfer, matching
V18's training path. In [PyTorch2.9.1's CUDA scalar-division kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu),
the scalar case multiplies by a precomputed reciprocal. A CPU arithmetic
simulation differs from true float32 division for126 of256 possible byte
values, by at most5.9604644775e-8. Ordinary CPU Torch division matched NumPy for
all256 values; a CPU test therefore does not reproduce the CUDA operation.

This is a concrete processing hypothesis. Small float changes can cross PNG
flooring boundaries, but the missing failed output prevents attribution from
the original archive alone. No first-case pixel difference has been measured.

The separate diagnostic uses exactly:

| Role | Fixed case |
| --- | --- |
| First stopped development case | `va_ffhq_28544_clear` |
| First previously declared clear preview with V15 raw output | `va_asian_00209_clear` |
| First V18 training parity case with original raw output | `v9_tr_ffhq_00084_clear` |

Eight frozen DGP forwards cover both normalization paths and two exact repeat
checks on the stopped case. No prior, code head, spatial decoder or recognizer
forward, target fitting, full validation, native/reserved image or training.
Worker120s, supervisor240s, export60s within the overall limit, with process-group
interrupt/kill stops. Fresh diagnostic root/session; original failed root is read
only. Every raw/PNG/reference difference is saved. Hypothesis confirmation
requires exact V15 PNG recovery, existing2e-6 raw preview/training parity,
stable repeated forwards and measured CUDA reciprocal behavior.

Four diagnostic regressions pass in0.011s. Standalone21,214-byte upload and sidecar
are separately audited in0.2922s; all144 original V19 assets remain unchanged.
Script SHA256:
`cad6da856e74ae9dcaa28d463304c93997d7e21ebee49fb65e5fa8a6d030e729`.
No diagnostic launch is claimed. The prepared local importer independently checks
eight raw arrays, six normalized tensors/compositions, reference differences,
repeat stability and148 protected original bindings after return. It performs
no local whole-network replay; source-bound VM counters remain reported evidence.

## Original milestone decision

Close the original V19 commands as failed; do not repeat or modify that package.
The decision at that point was to gather the small diagnostic before selecting a correction. If the hypothesis
fails, preserve that result and investigate the invalid assumption. A confirmed
processing correction needs a separate documented protocol with unchanged
scientific preservation gates before another development inference run.

V18's unconditional clear-preservation failures remain unchanged. No V19
generalization, useful native CCTV, local/Zamboanga performance, model adoption
or final-review claim is supported. The DGP-led app, insufficient-input behavior,
covering-family scope and independent final review remain incomplete.
