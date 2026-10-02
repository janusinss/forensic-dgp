# Visible restoration comparison — 2 October 2026

The Phase 3 DGP checkpoint worsens known-visible error on this ten-source degraded
development cohort. Changing the stage order alone does not solve it. A 25% blend
reduces the amount of change but still raises error. Keep the checkpoint and all
previous training evidence; do not enable this route automatically in the new
covering-removal workflow based on these results.

| Location | Windows local | Linux VM after transfer |
| --- | --- | --- |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_RESTORATION_RESULTS.md` | `~/forensic-dgp/PRACTICAL_RESTORATION_RESULTS.md` |
| Protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_restoration_v1\frozen_protocol.json` | `~/forensic-dgp/outputs/practical_restoration_v1/frozen_protocol.json` |
| Outputs | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_restoration_outputs_v1\` | `~/forensic-dgp/outputs/practical_restoration_outputs_v1/` |
| Phase 3 baseline | `C:\xampp\htdocs\YEAR 4\Testing\checkpoints\dgp_zamboanga_final.pth` | `~/forensic-dgp/checkpoints/dgp_zamboanga_final.pth` |

No transfer, training, commit or push occurred. Actual training stays on the L4 VM.

## Frozen comparison

Ten original and ten previously prepared degraded inputs use fixed operator
removal proposals: V3 for the six targeted cases/controls, V2 for the other four.
All are previously inspected detector-training sources, not a pristine holdout.
Native completion outputs are reused exactly; degraded completion uses fresh
CodeFormer inference. No automatic detector is rerun or credited with these masks.

The four arms are completion with restoration off, DGP before completion (legacy),
DGP on the completed crop composited only onto visible pixels, and a fixed 25%
visible blend of that post-completion result. Post arms preserve completed pixels
exactly. Existing native256 crops are retained without another alignment stage.
Native forced-restoration arms are regression diagnostics, not automatic routing.

Eighty outputs, 40 raw DGP intermediates and ten reused native completion outputs
were saved. The run records 30 new completion requests (24 nonempty forwards and
six empty bypasses), 40 DGP forwards and zero detector/optimizer updates. CPU elapsed
124.0 seconds excludes initial model loading. Recorded before/after state hashes
match for both networks.

The independent saved-pixel audit verifies all output/intermediate/source hashes,
fixed blend/composition, known-visible MAE/MSE/PSNR, support counts and aggregates.
Completion-only outputs change zero pixels outside their masks. Post-restoration
arms change zero completed pixels. Call/state counts are checked as execution
records, not independently replayed. Hidden facial ground truth remains absent;
no hidden MAE or identity accuracy is reported.

| Arm | Native mean visible MAE | Degraded mean visible MAE |
| --- | ---: | ---: |
| Completion only | 0.000000 | 0.020387 |
| Legacy DGP before completion | 0.045367 | 0.046924 |
| DGP after completion, visible pixels only | 0.045429 | 0.046970 |
| Post-completion 25% visible blend | 0.011333 | 0.022522 |

Metrics use known original visible pixels outside the fixed removal proposal,
RGB normalized to [0,1], with an equal per-case average. These are developmental
regression measurements, not population error or hidden-face correctness.

## Visual inspection and decision

All native and degraded rows were inspected in four five-row preview crops;
full ten-row grids are saved as `native_preview.png` and `degraded_preview.png`.
The full DGP arms introduce visible color shifts and smoothing. Post composition
retains generated regions but makes their joins more conspicuous. The 25% arm is
closer to the input while not demonstrating useful restoration improvement. The
turned mask/clear-glasses example remains poor under every arm. Clear native
controls should bypass restoration rather than receive these forced changes.

This is assistant developmental review, not independent expert scoring. The result
does not claim that every DGP restoration task fails, or that a different checkpoint
or training recipe cannot work. It directly rejects these stage/strength choices
on the fixed practical inputs.

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `74c652e1d1c604c5229fc953117a45b3c994e17c01fb23158ee9b4bccc17bf00` |
| Results | `2f1b2ff00237302bb3fd42c9188f63541292f777ba0d2df8aa10cc1c21f6223d` |
| Independent verification | `569f1c2c5112e17104cb7cfce54f6f262fb8dd56b098bf8520ad47e7a3d982eb` |

## Next comparison

Benchmark the separate official CodeFormer **restoration** checkpoint with
codebook size 1024, four fusion scales and `adain=True`. It is distinct from the
codebook-512 inpainting checkpoint. The official inference exposes fidelity in
[0,1]; the new frozen comparison uses 1.0 and 0.5 with declared visible blending
after completion. Exact source/license stay pinned; no project environment
downgrade or training is needed. [Official inference](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/inference_codeformer.py)

Source-only RealOcc inspection is also needed for missing standalone hands,
obstructing hair, scarves and other objects. Main-app integration, reliable
visibility rejection, broad-family output review and end-to-end Playwright
verification remain incomplete. The Goal stays active.
