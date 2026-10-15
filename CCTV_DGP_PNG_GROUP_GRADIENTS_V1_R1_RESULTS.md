# PNG-gradient R1 conversion stop and R2 repair — 13 October 2026

**R1 stopped before its first gradient query. No structure improvement, training
update, parameter proposal or new checkpoint was produced.** Its source, archive,
logs and exact initial controls are retained. The export's `complete: true` means
an archive was created; it does not mean this diagnostic succeeded.

Location: returned `scripts/cctv_dgp_png_group_gradients_v1_r1_vm.py:200`.
The CUDA RGB32 tensor did not exactly equal NumPy's byte/255 values. The helper's
own CUDA-value equality assertion had passed, so the demonstrated boundary is
that this internal CUDA conversion did not reproduce the authoritative CPU
conversion. The failed CUDA float values were not exported; their mismatch
count, ULP magnitude and precise kernel behavior are not independently known.

The audited run reports **10.56 seconds**, two B forwards,
zero recognizer forwards, zero gradient queries, zero optimizer updates and no
parameter displacement. All **15** returned and
**5852** required parent hashes pass. Five saved raw
controls equal the retained raw arrays by bits; all five PNG controls equal
retained PNG pixels. No new candidate images or visual sheets were produced.
These are photographic TRAIN controls, not native CCTV or final-review evidence.
The model was not displaced; no fresh post-failure model-state hash is claimed.

The7,311,160-byte archive SHA256 is
`cd9a118d6f85dcc914c58bae37d19726a3aa5ed029a349543b20455b1cd93e27`.
[Independent failure audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_png_group_gradients_v1_r1_analysis/failure_audit.json>).
[Canonical boundary and exact controls](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_png_group_gradients_v1_r1_analysis/canonical_boundary_review.json>).

The previous preparation checked CPU primitives but missed this CUDA boundary.
That was an implementation-check limitation. PyTorch documents that mathematically
equivalent floating computations, including CPU versus GPU, need not match bit
for bit. This supports a numerical-boundary diagnosis, not a specific observed
GPU error magnitude. [Primary numerical-accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
A retained CPU illustration finds419,362 channel differences, at most
5.96046448e-8, between division and multiplication by a float32 reciprocal on
these saved PNGs. That illustration is explicitly **not a GPU reproduction**.

## Why0.863% is still unqualified

The earlier all-group step .02 achieved **0.862655% raw /0.877397% PNG** mean
degraded landmark high-frequency error reduction. This is one saved metric,
not a percentage of facial identity recovered. It also failed **37 raw and37
PNG group–metric preservation checks**. Source-specific raw gains were0.556856%
for `dataset/asian_faces` and0.939342% for `dataset/thumbnails128x128`;
both must reach1%. Dataset labels are not ethnicity or native-capture evidence.
Increasing the same step to .04 made structure worse. Being0.137345 percentage
points below the global raw threshold does not establish safe continued learning.
The much smaller signed proposals also remain unqualified.
[Retained finite comparison](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md>).

The next improvement must reduce structural error while preserving clear faces,
visible appearance and both raw and delivered outputs. Passing1% is an early
learning requirement, not full model, native CCTV, app or covering qualification.
More epochs remain conditional on a finite useful, preserving actual change.

## Fresh repair and next manual measurement

R2 computes authoritative NumPy RGB32 and feature64 byte/255 values once without
any CPU neural or gradient calls, then copies those values to CUDA. Exactly-zero
straight-through terms preserve the forward bits and retain the explicitly
approximate identity backward. Scientific SSIM keeps its exact pixel evaluator
and approximate analytic backward with the same zero-valued assembly. Scientific
loss definitions, masks, baseline states, thresholds and scalar tolerances stay
unchanged. No claim of a true derivative of discontinuous PNG conversion is made.

Before loading models, the manual VM worker must pass all256 byte levels in both
precisions and export the values plus a legacy CUDA division comparison. It then
checks the same100 frozen raw/PNG cases before querying gradients. All local100
saved-value forward checks and256-byte fixtures passed with no neural/autograd
calls. Actual CUDA repair execution remains **unrun**.

The fresh packet retains320 queries,32 groups,40 B/20 recognizer forwards,
zero updates/proposals/epochs/checkpoints and the same timing/storage stops.
Only after its returned gradient evidence is audited can a joint raw/PNG direction
and a separate small actual-change comparison be justified. A gradient dot or
completed archive cannot qualify more training or bypass preservation failures.
[Exact manual R2 commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_VM.md>).

Original app checkpoint, model components, splits, masks, caches, migration
backups, previous failures and app design remain preserved. All five milestones,
separate native/unpaired and paired evidence, independent final identities and
all seven automatic/assisted covering families remain required. No Zamboanga,
ethnicity or exact hidden-identity claim is made. Agent VM access remains
maintenance only; training/gradient work stays human-manual L4 tmux.
