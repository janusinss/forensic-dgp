# V23 result — independently audited failed capacity pilot

6 October 2026. **Close V23.** Its original one-percent early structure stop fails
at update 50, and all 50 reviewed training cases show no convincing visible gain
over the retained own-DGP output. Preserve the partial run and original gates.
No checkpoint is promoted to the local application.

The Windows return contains 76,481,832 bytes; SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
The checksum and export receipt match. All 370 safe regular returned members,
209 frozen original assets, exact source and stopped-head state, two complete
50-case snapshots, 100 raw/PNG pairs and 100 saved metric rows are checked.
The export's `complete` flag describes packaging; the trainer exited 1.

| Evidence | Independently checked result |
|---|---|
| L4 execution | 50 optimizer updates, 51 backward calls; fresh head only |
| Delivered degraded landmark-HF MSE | 0.0019965976532523044 → 0.0019966422385438174 |
| Delivered structure gain | −0.0022330634%; original requirement ≥1% at update 50 |
| Raw degraded structure gain | −0.0022518731%; PNG quantization does not explain the failure |
| Execution timing | Worker 19.5523 s, fitting 6.6122 s, supervisor 25.2792 s |
| Timing stop | 19 exact steady step samples; 152.9801 s projected within 1,500 s cap |
| Memory receipt | Peak torch-allocated memory 846,010,880 bytes, below 20 GiB; not total device memory |
| Direct-path gradient | Nonzero sum of squares 0.006192748643339683 |
| Original frozen components | Four cached/fresh DGP CUDA checks match; recognizer state unchanged |
| Final 800-update capacity test | Unexecuted; no final result exists |

The independent audit took 32.30 s with 100 fixed CPU head forwards and 110
frozen recognizer forwards. Maximum raw replay difference is 5.9604645e−8;
maximum saved metric difference is 7.1054274e−15. Original GPU receipts determine
the failure. The predeclared CPU replay tolerances remain unchanged. Seeded
nonzero initialization across torch 2.9/2.13 uses the already documented 1e−8
compatibility bound; exact saved VM state, zero output tails, fixed buffers and
all initial baseline arrays are preserved. Ten return-auditor regression checks
pass. No local backward call, optimizer update or assistant VM action occurred.

Every original-resolution review sheet is viewed: ten TRAINING references, five
profiles each (clear, blur, low light, motion and compounded degradation). Clear
eyes, mouths and transparent frames remain soft; degraded eyes and facial edges,
motion ringing and low-light/compound artifacts show no convincing improvement.
The 153,767 changed canvas pixels out of 3,276,800 differ by at most two bytes;
degraded changes are at most one byte. Nonidentical files do not establish useful
structure. These are exposed photographic training examples, not native CCTV,
independent evaluation or verification of reconstructed identity.

A known-layer trace uses 50 fixed CPU head forwards. All eight trainable tensors
change, but the median corrective-band RMS is 3.80554e−5. Independent saved-data
checks cover 375 bindings and 200 exact review cells. Seven original preservation
conditions also fail when applied diagnostically at update 50, including SSIM/
recognizer groups and brightness-only fraction 0.2084449 above 0.2. That diagnostic
does not relabel the unexecuted final gate as evaluated.

## Demonstrated objective mismatch

The exact original loss is decomposed on both saved raw states for all 50 cases.
Its compiled, source-pinned definition matches assembled terms within 1e−6.
The fixed CPU check uses 210 recognizer forwards, no head prediction, optimization
or backward call. A separate arithmetic auditor verifies 174 bindings, 100 case
states and six cohort states. These checks do not prove the GPU gradient trajectory.

| Cohort | Original baseline objective | Stopped-head objective |
|---|---:|---:|
| All 50 | 1.2999999523 | 1.2954864430 |
| Clear 10 | 1.2999999523 | 1.2771114230 |
| Degraded 40 | 1.2999999523 | 1.3000801980 |

Clear cases contribute 101.4223% of the net objective reduction; the degraded
cohort offsets part of that reduction. Per-case HF normalization can also prefer
a different direction from the required absolute degraded-cohort error. Thus the
old overall objective can improve while the intended degraded restoration fails.
Clear delivered feature error improves 2.30449%, while degraded error worsens.

The next distinct finite experiment is **V24**: retain the same own-model head,
data and all output gates, but remove clear target reward, use frozen degraded-
cohort HF normalizers and treat clear inputs as preservation controls. CPU
contracts establish this policy's arithmetic; actual learning remains a manual
L4 experiment. If it fails, stop blind changes to this head recipe and revisit
the architecture/data/loss assumptions before another training protocol.

Useful native restoration, canonical app parity, all seven automatic/assisted
covering families and independent final review remain open. The user's shown
useful native crop still needs clearer structure; its usable input is not
reclassified as insufficient. Original checkpoints, splits, failures and the
existing DGP-led application/design are unchanged.

Evidence:

- [Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_detail_skip_v23_independent_audit.json>)
- [All-case visual ledger](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_detail_skip_v23_diagnostic/visual_review.json>)
- [Independent saved loss check](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/independent_saved_loss_audit.json>)
- [V24 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md>)
- [V24 VM commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>)
