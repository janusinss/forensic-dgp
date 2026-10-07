# Original decoder gradient proof R2: failed all14 gate retained

The downloaded archive matches the reported SHA256
`dfd8e661126d144e87875f236a7eedb339f7803a65f57b92aa0fd128110d4e3e`
and 93,557,954 bytes. Safe import verifies all 172 regular files, the protocol,
original source assets and export manifest. The independent audit checks every
one of 46,909,863 saved gradient values across ten matrices and their sum,
all component/parameter statistics, 50 original-DGP CPU forwards, 50 fixed
recognizer forwards and the complete 50-case baseline cohort.

**Location:** original R2 worker line 263, the all14 initial-gradient assertion.
**Cause:** `head4.block0.weight` and `head4.block1.weight` have exactly zero
gradient in every component of every batch. This is not cancellation during
aggregation. The other 12 tensors have nonzero aggregate improvement gradients.
**Next design:** keep the demonstrated inactive branch frozen in a separate
candidate, and prospectively test the 12 active original decoder tensors.
The failed 14-tensor diagnostic remains failed; its source and gate are unchanged.

All 70 L4 gradient queries ran. The diagnostic worker stopped after 25.232
seconds; its supervisor took 26.416 seconds within the 660-second bound. No
optimizer was constructed, no updates or epochs ran, and no new checkpoint was
created. The four initial preservation terms and all their gradients are exactly
zero, as required when the candidate equals its baseline. Export completion means
failure evidence was packaged successfully; it does not mean training succeeded.

The original state stayed unchanged at the assertions preceding the stop.
No GPU intermediate activation trace is present. The independent CPU replay's
1e-5 raw and 5e-5 fixed-vector bounds check numerical compatibility; the quality
requirements are unchanged. The terminal all14 failure is never treated as a pass.

A separate forward-only trace identifies the inactive branch's numerical limit:

| Measurement | Result |
| --- | --- |
| Both original `head4` kernels | All 110,592 values are nonzero float32 subnormals; maximum absolute value about 6.31×10⁻⁴⁰ |
| Upstream fourth feature map | Nonzero in all 50 cases |
| Unchanged float32 `head4` output | Exactly zero in all 50 cases |
| Inference-only float64 branch output | Nonzero in all 50 cases, maxima 2.49×10⁻⁷⁶ to 4.29×10⁻⁷⁶ |
| Float32 smallest positive value | About 1.40×10⁻⁴⁵; every higher-precision branch output lies below it |

The original float32 path is retained throughout that trace. Separate higher
precision calculations are numerical evidence, not modified restoration outputs.
The independent NumPy readback verifies all 350 saved activation arrays and all
100 two-convolution calculations. No local derivative or optimizer is run. This
identifies an inactive branch at the retained checkpoint; it does not establish
when that checkpoint acquired these weights or explain every earlier small-head
failure. The fourth map still enters the FPN top-down path and contributes to
other heads; an inactive direct `head4` output does not remove all coarse features.

The actual app encoder is separately checked: CPU NumPy float32 division before
transfer and the older Torch division after transfer are exactly equivalent on
all byte values and all 50 inputs. No CPU normalization defect is found. R2 used
the declared older adapter on the L4; its “canonical” wording did not prove the
actual app encoder there. New V28 ships the exact app conversion function and
requires fresh same-batch initial output and gradient parity before optimization.

V28 is a distinct finite capacity pilot. A separate copy keeps the unchanged
original forward and freezes `head4`, the encoder/FPN and all stored buffers.
Only `head1`–`head3`, `smooth`, `smooth2` and `final` can train: 498,627 parameters
in 12 tensors. Its initial raw and PNG outputs exactly match the actual app
reference on all 50 CPU cases. It does not revive the inactive fourth branch or
reclassify R2. Its changed parameter scope and conservative fixed fine-tuning
rate are declared before any VM training.

The 800-update bound, 1% early structure gain at update 50, 10% final structure
gain, source nonregression, all 17-group MSE/SSIM/identity bounds and brightness
limit remain required. All facial features and visible appearance need review
together; a landmark metric alone cannot accept an outline. A new prospective
return audit covers gradients, frozen partitions, candidate checkpoints, every
raw/PNG/metric row and the original timing rules. The packet is prepared for the
human's manual Google Cloud SDK/SSH/tmux workflow; no assistant VM action occurs.

All inputs here are exposed photographic TRAIN cases. Native CCTV, reserved
final evaluation, seven-family completion, independent final review and useful
app outputs remain separate qualification work. No ethnicity, Zamboanga
performance or exact hidden identity is inferred. The goal remains incomplete.

[R2 independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json>) ·
[Branch trace](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_decoder_head4_review_v1/results.json>) ·
[Independent trace readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_decoder_head4_review_v1/independent_readback.json>) ·
[V28 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_VM.md>)
