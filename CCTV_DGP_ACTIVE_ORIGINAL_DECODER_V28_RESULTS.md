# V28 return: training completed; acceptance failed

V28 completed all 800 updates/80 epochs on the existing NVIDIA L4. The downloaded
291,926,255-byte archive matches SHA256
`3196e8ab797763e7ced75a57411b973afc3b1b12654632b7662fab3e2e168aac`.
The independent audit passes execution and evidence checks. **The final model
fails the prospectively fixed acceptance requirements and is not adopted.**

The audit verifies 838 regular return files, 246 original assets, 38,394,279
saved gradient values and every checkpoint/raw/PNG/metric at updates
0/50/400/800. It performs 50 original-DGP, 200 candidate-DGP and 250 fixed
recognizer CPU forwards without local derivatives or updates. The 12 selected
decoder tensors were trained; the original checkpoint, encoder/FPN, inactive
head4 and stored evaluation buffers remain unchanged. The historical all14 R2
failure remains failed. The audit takes 139.150 seconds locally.

The VM receipt records 800 backwards/updates, 80 epochs, 70 initial component
gradient queries, 10 reference-DGP, 850 candidate-DGP and 860 recognizer calls.
Fitting takes 91.850 seconds; the execution receipt's worker takes 124.967
seconds, and the terminal result takes 125.789 seconds. Peak allocated VRAM is
1,240,671,744 bytes. The finite timing, state and selected12 initial parity/
gradient requirements pass. Export completion is packaging evidence.

| Fixed check | Final result | Decision |
| --- | --- | --- |
| Final degraded landmark-HF MSE gain | 18.0595% against10% minimum | Pass |
| Both photograph source gains | 29.4282% and15.8564% | Pass |
| `dataset/asian_faces/clear` SSIM | 0.971355783 → 0.970622678; tolerated drop1e-6 | Fail |
| Same clear group, fixed ArcFace cosine | 0.977832592 → 0.977572036; tolerated drop1e-6 | Fail |
| Constant RGB-mean shift fraction of degraded pixel-MSE gain | 71.3988% against20% maximum | Fail |

The early structure check passes with 2.373% gain at50 against1% minimum.
The brightness fraction describes pixel-MSE improvement, not a percentage of
the landmark-HF gain or recovered facial identity. ArcFace is a frozen
recognition-feature proxy, not proof of identity. The source labels are dataset
paths; no ethnicity, Zamboanga performance or native-CCTV result is inferred.
All metrics here use paired photographic TRAIN cases, not held-out identities.

All 50 final faces were reviewed in ten exact-size sheets: input, unchanged
DGP, V28 final800 and paired training target. Saved-cell readback verifies all
200 source arrays without resizing or enhancement. Several degraded faces have
stronger coarse eye/nose/mouth boundaries and a more readable overall facial
arrangement. Finer eyes, glasses, gaze and expression remain unresolved or
altered in strong degradation. Clear glasses, hair and visible objects remain
part of the preservation scope. These training faces do not establish
generalization. The primary assistant's development review is separate from
the still-required independent final review.

A single fixed processing control subtracts the candidate's observed-support
RGB mean change relative to the baseline, clips and exports the same PNG
policy. It uses neither the target nor the profile to generate an output; there
is no search, tuned mixing factor or clear-case routing. All 50 control arrays,
PNGs and recognizer vectors are independently checked. It retains 18.0237%
landmark-HF gain and reduces the brightness fraction to 1.6955%, but fails five
preservation checks: clear-group MSE, SSIM and ArcFace in one source;
compound-group SSIM in that source; and motion-group MSE in the other source.
It is insufficient and is not an app fix. Only two control examples were
visually inspected; no complete control visual acceptance is claimed.

The unchanged original objective has soft hinge penalties and no explicit
RGB-mean anchor. This is a demonstrated design limitation relative to the
strict acceptance gates; it does not identify the unique optimization cause.
No loss-weight sweep, earlier-checkpoint selection or unchanged V28 rerun is
justified. A distinct zero-update VM diagnostic is prepared to measure the
seven original objective gradients at the immutable final state and three
diagnostic mean/clear-preservation components. It has100 finite queries,
zero optimizer updates/epochs, a300-second worker limit and no new checkpoint.
Additional components are measurements only; no new training objective or
loss weights have been selected.

The initial local preparer's AST check incorrectly matched permitted NumPy/PNG
`save` calls as checkpoint creation. It failed before a packet or VM action.
The source and failure are preserved; the correction specifically prohibits
Torch checkpoint writes and optimizer calls. The published three-file packet
passes Python3.10 parsing, nine malformed-return/no-training regressions,
actual Windows rejection before neural/gradient work, all307 closed-V28
dependency bindings, manual-command readback and read-only Bash syntax checks.

[Next five manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md>)

V28 and its processing control remain rejected. The existing app model/design
and its22 evidence/source bindings are retained. No native development or
reserved-final pixels are opened. Useful native restoration, all seven covering
families, the full app flow and independent final review remain required.
The overall goal remains active and incomplete.
