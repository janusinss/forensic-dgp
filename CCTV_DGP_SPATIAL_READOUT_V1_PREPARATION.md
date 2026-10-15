# Spatial reconstruction V1 preparation — 14 October2026

**Verified and UNRUN.** The human diagnostic answer resolved the required design
discussion. The selected comparison changes reconstruction capacity, instead
of extending a rejected conditioning-only state or repeating a gradient solver.
The original app DGP SHA256 remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.

Route A has15 decoder tensors /609,219 elements. B adds five anchored3×3 feature
connections for25 tensors /2,176,419 elements. Existing decoder/readout weights
are reused. Observation features, input-to-prior conditioning, normalization,
external pretrained StyleGAN2 and recognizer stay frozen. The bank is declared
external; it is not an own-trained GAN or a pretrained restoration substitution.
The base feature-provider state hash is98675d10be6f3405dd4bd0950d9251a61b26b9b161ae62a9ce48f775bc43e2bd;
this is the original B initializer, not the later anchored wrapper's state hash.

Both routes start at identical original outputs, subject to exact manual CUDA
parity on all100 controls. Each then gets at most50 bounded Adam steps on the
same20 exposed TRAIN references /100 views. All selected tensors need finite
nonzero objective derivatives before any optimizer. Adam computes proposals on
shadow values; only the projected result enters the live decoder. The common
0.001 distance allowance cannot grow with B's added parameters. Full states,
RNG, optimizer moments and all100 parameter-journal steps remain in the return.
This is research fitting, not accepted development updates or full-corpus epochs.

All17 RAW and17 PNG groups, original RGB/SSIM/recognizer tolerances, ≤20%
brightness-only contribution, ≥1% global and both-source structure gate and the
later full-TRAIN10% requirement remain. Original float32 embedding scoring and
cached baseline controls are retained. Actual output preservation, useful native
DEV images and independent final review remain necessary; local derivative
predictions are not acceptance. The previous V6/V9/original-feature/bank failures
are explicitly reconciled in the design review, rather than being treated as
untried unfreezing ideas.

The independent preparation audit checked all16 archive members,203 required
parent file bindings,20 canonical targets,100-case membership and unchanged
historical block ordering,5 readout rank/shape checks,14 CPU boundary regressions,
11 altered-protocol rejections, actual hash-only CLI, Windows pre-neural training
denial, Python3.10 AST and read-only host Bash syntax. No model construction,
forwards, derivatives, optimizer updates or VM calls ran during preparation.
Local tensor deserialization was read-only. New CUDA behavior/runtime remains
unverified and is explicitly guarded in the manual test.

The first preparer assumed an HQ-only metadata field existed on replay records.
The first checker assumed case-ID and block lists shared an order; all100 unique
members actually match, while the exact block/schedule order is separately pinned.
Git Bash then failed to spawn within the Windows sandbox; the unchanged launcher
passed `bash -n` with normal host permissions. All three preparation/checker
failure records are retained. None is a training or model-quality result.
The sealed archive/protocol/executable sources were not changed for checker fixes.

Require8 GiB free and8 GiB available RAM before imports; estimate20–40 minutes
including checks/export with finite caps in the guide. This is not current VM
inventory. No cleanup, automatic VM access or automatic training occurred.
Checkpoint/caches/splits/provenance/full states/failures and original app design
remain. Native unpaired, synthetic paired, source-specific and untouched final
roles stay separate. The seven automatic/assisted covering families and detector
limitations remain independent; reconstruction fitting does not qualify completion.
Goal active/incomplete; no new incumbent, extra epochs or app promotion.

[Exact upload, tmux and download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_READOUT_V1_VM.md>).
[Chosen hypothesis and research sources](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md>).
[Independent preparation audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_reconstruction_design_review_v2/independent_preparation_audit.json>).
Protocol `3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e`; execution archive `83033e7207a9ce5bce5b7554d19a53202ace9375b8a74b5e9ba5bfe26903e3b5`, 99,256 bytes.
