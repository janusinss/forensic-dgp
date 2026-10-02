# Selective face restoration — 2 October 2026

Use the separate CodeFormer restoration model at fidelity 1.0, blended 50% onto
visible pixels after completion, as the development route. On ten fixed degraded
inputs it lowers mean known-visible RGB MAE from 0.020387 to 0.015654 (23.2%),
improving all ten cases. This is a small, inspected development cohort, not proof
of hidden identity or population performance. Clear inputs should bypass this
stage; forced restoration changes their observed pixels.

| Evidence | Windows local | Linux VM after transfer |
| --- | --- | --- |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_FACE_RESTORATION_RESULTS.md` | `~/forensic-dgp/PRACTICAL_FACE_RESTORATION_RESULTS.md` |
| Weights/provenance | `C:\xampp\htdocs\YEAR 4\Testing\outputs\codeformer_restoration_pretrained_v1\` | `~/forensic-dgp/outputs/codeformer_restoration_pretrained_v1/` |
| Frozen protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_face_restoration_v1\frozen_protocol.json` | `~/forensic-dgp/outputs/practical_face_restoration_v1/frozen_protocol.json` |
| Saved outputs/audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_face_restoration_outputs_v1\` | `~/forensic-dgp/outputs/practical_face_restoration_outputs_v1/` |

The official restoration architecture uses a 1024-entry codebook, four fusion
scales and `adain=True`. It is distinct from the 512-entry inpainting model.
Source revision `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b` and S-Lab License
1.0 are retained; the existing pinned vendored implementation is unchanged.
[Official inference](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/inference_codeformer.py)

The official 376,637,898-byte checkpoint has observed SHA256
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.
This is an acquisition fingerprint, not a publisher-supplied checksum. Strict
restricted loading verifies 515 finite tensor states; parameters are frozen.
The acquisition record's `strict_load_pending` describes its earlier state;
this completed execution and independent audit supersede it without editing it.

## Fixed execution and independent verification

Twenty cached completed crops (ten native and their ten declared degraded copies)
use the previous frozen V3/V2 operator removal proposals. All sources were already
inspected; they are not an unseen holdout. Five arms compare completion alone,
fidelity 1.0 with full/half/quarter visible blending, and fidelity 0.5 with half
blending. No automatic detector masks are credited with these results.

There are 100 saved outputs, 40 raw restoration intermediates and 20 reused
completion crops. Forty restoration forwards took 230.3 CPU seconds excluding
model loading. No new completion, detector or optimizer forward/update occurred.
Recorded before/after model-state hashes match. Three adapter tests pass.

| Arm | Native mean visible MAE | Degraded mean visible MAE |
| --- | ---: | ---: |
| Restoration off | 0.000000 | 0.020387 |
| Fidelity 1.0 / full blend | 0.018064 | 0.019070 |
| Fidelity 1.0 / 50% blend | 0.009030 | **0.015654** |
| Fidelity 1.0 / 25% blend | 0.004405 | 0.017041 |
| Fidelity 0.5 / 50% blend | 0.012000 | 0.018292 |

The independent audit checks saved hashes, raw/composed pixels, masks, reused
crops and recomputed known-visible MAE/MSE/PSNR. It finds zero changed completed
pixels. Fidelity 1.0 / 50% blending improves all ten degraded cases (per-case
relative reduction 15.2–33.3%). Metrics compare only known original visible RGB
pixels outside each fixed removal proposal, with equal per-case averaging.
Call/state counts are verified execution records, not independently rerun models.

All 20 rows were inspected in four five-row previews. The half blend gives useful
visible detail with less appearance drift than full restoration. Clear controls
still drift when forced on; automatic routing must use input-only quality signals
and provide an off/on override. The difficult three-quarter mask/clear-glasses
case still has a poor generated nose join, beyond the initial frontal/mild-turn
scope. This is assistant developmental review, not independent expert scoring.

| Frozen evidence | SHA256 |
| --- | --- |
| Protocol | `c384f1b25e9ab80f44ba6ebef3e47c072a4dc57d406d525f9a637f3abf99b229` |
| Results | `de6a7bf7e177d7f2f8d3ed2c0536de05c64e6662e464046dda597a50bd9f7cb0` |
| Independent verification | `b34c4442ffbb43ffbb811cd02b631622603e6a2639f86e9741e5334be69c7635` |

Next: freeze and review the missing covering families, integrate this route with
mask correction and visibility rejection in the existing main application, and
verify the complete browser workflow. No training, transfer, commit or push took
place. All actual training remains on the L4 VM; the Goal stays active.
