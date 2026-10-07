# V25 gradient diagnostic — confirmed reference-processing discrepancy

6 October 2026. The fixed-state L4 diagnostic completes in18.5350s with **zero
optimizer updates**,140 component gradient calls,20 head batches and120 frozen
recognizer forwards. All original DGP/head/recognizer evidence remains unchanged.
This completes a measurement; the original V25 restoration failure remains closed.

Archive1,695,519 bytes, SHA256
`3b849fa23336b42d765fe811f6cf4fe5b968d561ad4a4f2d589e09c48c50046a`.
Sidecar/export/source/protocol agree. Strict safe import checks10 regular members.
Supervisor19.7094s and1,379,948,544 allocated VRAM bytes remain within frozen bounds.
Both saved heads have exact original hashes; no new checkpoint is created.

The independent1.63s readback verifies235 original assets,114 original saved inputs,
250 frozen features, two7x53,781 float64 gradient matrices, fourteen component
rows and26 named parameter groups per state. Norms, Gram matrices, cosines,
per-parameter statistics, ten-batch scalar sums, log/counters/timing and state
receipts agree. Maximum fixed CPU/CUDA scalar difference1.3932586e-7 is within
the prospective compatibility bounds. No local model, derivative, backward or
optimization call occurs. This audit recomputes saved-gradient arithmetic and
checks source provenance; it does not independently rerun L4 derivatives or
reconstruct AdamW moments/trajectory.

| Saved state | Landmark-detail gradient norm | Identity-penalty gradient norm | Identity/landmark ratio | Their cosine |
| --- | ---: | ---: | ---: | ---: |
| Initial0 | 0.00706145 | 0.01024891 | 1.45139 | +0.21542 |
| Stopped50 | 0.00698367 | 0.01201631 | 1.72063 | −0.33243 |

**All50 initial raw outputs equal their cached baseline exactly.** Nevertheless,
the legacy identity penalty has mean1.3932586e-7 and a nonzero gradient larger
than the landmark-detail gradient. The source compares baseline scores evaluated
one image at a time without gradient tracking against predictions evaluated in
a five-image gradient-enabled call. A rounding-scale positive difference crosses
the zero-margin ReLU, whose active slope retains the full penalty weight5. The
large gradient is therefore not evidence that the identical output lost identity.
The discrepancy belongs to reference processing rather than missing input detail.

At stopped50, the combined preservation gradient norm0.01327472 is similar to the
degraded reward norm0.01373153, with cosine−0.40168. SSIM and identity terms oppose
some structural gradients there. This is a fixed-state observation; it does not
establish that all real penalties are numerical artifacts or that their competition
alone caused the failed training trajectory. Initial identity-gradient correction
is justified, but useful restoration remains unproven.

V26 compares baseline and prediction in **one frozen recognizer call**, with the
baseline embedding detached as the fixed reference. The same target, observed
support, affine geometry, loss coefficient5 and zero margin remain. Other six
objective terms, own-DGP encoder/spatial head,50 exposed TRAIN cases, schedule,
optimizer, original quality gates and all finite limits stay unchanged. It adds
no epsilon allowance, identity-term removal or pretrained restoration model.

Before creating an optimizer, the L4 preflight must reproduce the legacy numerical
discrepancy, confirm all50 initial outputs are exact baseline, and prove the new
identity penalty and every one of26 parameter gradients are **exactly zero** in
ten batches of five. State and frozen-feature checks remain exact. Failure stops
the fresh pilot before any training; preserve its source/log/receipts and do not
rerun unchanged. Successful preflight permits the otherwise identical finite
capacity test, retaining1% structure at50 and10% at final800 plus all preservation
and brightness requirements. No checkpoint is promoted automatically.

Eleven new archive/source/matrix and arithmetic-reference regressions pass without
local neural derivatives or learned-model forwards. The49,537-byte thin V26 packet,
Python3.10 source, actual Windows rejection guards and Bash syntax are verified.
Actual L4 processing correction and learned useful output remain pending.

No native, reserved-final or new covering pixels are used. Paired photographic
TRAIN evidence remains separate from unpaired CCTV. All visible facial features
remain required together; previously useful native inputs remain usable despite
model softness. Original failures, checkpoints, splits and app design remain.
Useful native output, canonical candidate app parity/full flow, insufficient-info
handling, all seven automatic/assisted covering families and independent final
review remain required. Goal active/incomplete.

[Independent gradient audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v25_gradient_diagnostic_v1_independent_audit.json>) ·
[V26 finite design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_PLAN.md>) ·
[V26 manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md>)
