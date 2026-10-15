# Activation remains valid; learning startup repaired — 13 October 2026

The subsequent R2 learning run stopped at a16×16 pixel fixture before any model
construction/inference, gradient or update. It did not test image quality.
[Audited startup failure and repair](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R2_RESULTS.md>).
[Current verified, unrun R3 manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R3_VM.md>).
The activation finding and exact previous report are retained below.

# Live conditioning R1 returned audit — 13 October 2026

**The initial gradient blockage is resolved on the fixed photographic TRAIN
diagnostic. Image quality and additional epochs remain unqualified.** All12
style/spatial conditioning tensors have finite nonzero aggregate gradients for
clear-preservation MSE and degraded-detail MSE in each sampled source. The100
initial rawRGB controls match the original DGP bit for bit. No optimizer update,
parameter change, new checkpoint or epoch was performed by this diagnostic.

The manual L4 measurement took **127.94s**. It completed40 candidate
forwards,40 live plus40 frozen feature-branch passes,20 recognizer forwards and320
gradient queries. Peak allocated VRAM was3.055GiB and peak
host RSS2.925GiB. The full candidate state/buffers and original
DGP/recognizer remained unchanged. The all12 requirement passed in all four
source/task combinations:

| Photographic source label | Task | Minimum tensor gradient norm | Maximum |
|---|---|---:|---:|
| dataset/asian_faces | clear/MSE | 7.5979745e-05 | 0.0020362009 |
| dataset/asian_faces | degraded/feature_MSE | 6.3568118e-06 | 0.00017984787 |
| dataset/thumbnails128x128 | clear/MSE | 3.6894972e-05 | 0.00098454649 |
| dataset/thumbnails128x128 | degraded/feature_MSE | 2.9196631e-05 | 0.00071898165 |

These source labels identify retained photographic archives, not ethnicity,
native capture origin or Zamboanga performance. Scope is20 TRAIN references,
100 paired cases and32 raw loss groups. Native CCTV and final identity pixels
were excluded. No new improved PNG output or covering completion was produced.
[Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_live_conditioning_v1_r1_analysis/independent_audit.json>).

The archive is765,752,227bytes, SHA256
`c6fe1990cef716479ae4514944d681eadb040aba14a918eb0a243424bce4dcc6`. Every one of92 returned file bindings and5,852
parent assets was rechecked. The independent audit recalculates saved norms,
direction dots, contribution scopes and controls; it does not rerun all320 neural
derivatives. Two clear-MSE decoder/readout sample derivatives were compared to
the prior measured gradients. This result establishes activation, not a finite
image-quality gain. Earlier B50 already changed all37 tensors and still failed;
the new finding does not explain away every historical failure.

## Selected finite learning test

The saved raw derivative-span review found a common local descent direction in
both the full37-tensor space and the12-tensor conditioning space while retaining
all32 losses. The latter has direction norm3.836663
and Gram condition number4163.94,
compared with210374.60 for all37.
These are arithmetic results, with prospective finite and PNG behavior pending.
[Independent subspace audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_live_conditioning_v1_r1_analysis/learning_subspace_audit.json>).

The next manual pilot trains only the12 now-active conditioning tensors. The
current decoder, readouts, same-input frozen reference, external generative bank,
recognizer and buffers stay fixed. It permits **at most2 accepted full-cohort
gradient-descent updates**, with at most3 actual backtracking trials per update.
It refreshes all32 raw and32 approximate PNG loss derivatives at the current
weights after an accepted update. Each accepted change must preserve all17 raw
and17 PNG metric groups, satisfy the0.2 brightness limit and improve degraded
detail globally and in both sources. The finite checkpoint still requires1%
structure gain globally and per source on raw and PNG. Full TRAIN's10% structure
requirement and useful native DEV qualification remain separate later gates.

Accepted research components and every rejected trial are retained with true
update counts, raw/PNG images, gradients and state receipts. The cumulative
parameter-distance limit remains0.1% of the initial37-tensor norm. No fixed old
direction is reused after weights change. No longer epoch run or app adoption
follows automatically. The100-case test is a finite training-method pilot; it
does not cover a full epoch, reserved final identities or seven completion families.
[Verified manual upload, tmux and download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R2_VM.md>).

The first two local drafts remain retained. Final preparation restored the
retained float32 scientific identity-dot scoring (maximum draft difference
7.7264e-8 on200 saved rows), freezes the original cached baseline scoring while
checking live embedding replay, checks the internal measurement cap at every
clock/final verification, and explicitly rejects nonfinite scientific metrics.
All original quality thresholds remain. R2 is the only published launch guide.

The design hypothesis uses a frozen same-observed-input feature reference and
active multiscale conditioning. GLEAN supplies primary evidence for an
encoder–bank–decoder using a frozen generative bank; our adaptation remains a
separate hypothesis, not a reproduced result or established identity recovery.
[GLEAN paper](https://arxiv.org/abs/2012.00739).
Separate loss constraints are motivated by documented conflicts in multi-objective
learning. Our full-span constrained minimum-norm direction is not claimed as
the paper's exact MGDA method or a finite preservation guarantee.
[Sener and Koltun](https://arxiv.org/abs/1810.04650).
The exact PNG forward uses an explicitly approximate identity backward; its
heuristic nature follows the general straight-through-estimator literature.
[Bengio, Léonard and Courville](https://arxiv.org/abs/1308.3432).

## Storage and remaining goal

The authorized fresh maintenance inventory at2026-10-13T14:37:56Z found
**99.93GiB free** and an idle GPU. No files were removed and no training was
launched by the agent. The named L4/g2-standard-4 instance and SSH host key were
verified. The pilot requires9GiB free after installation and rechecks its own
reserve; this snapshot is not a future free-space guarantee.
[Fresh inventory](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_storage/inventory/guest_inventory.json>).

The original app checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The DGP-led app, Auto/override design, native development/final separation,
paired/unpaired evidence boundary, original checkpoints/splits/failures and all
seven completion families remain in scope. All actual training is human-manual
on the existing L4. Direct agent VM access remains storage maintenance only.
Milestone3 advances from verified activation to a prepared finite learning pilot;
all five milestones and independent useful-output/final/app qualification remain
ACTIVE and incomplete.
