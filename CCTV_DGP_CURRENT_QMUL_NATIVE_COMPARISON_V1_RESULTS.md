# Current DGP on frozen QMUL development crops — 9 October 2026

The current DGP still needs clearer structure. All 24 frozen QMUL development
crops now have a fresh, matched comparison of resizing, original Phase3, the
current identity-v2 DGP and the declared pretrained CodeFormer baseline. The
earlier QMUL pilot gallery used a different checkpoint. This comparison closes
that evidence gap without claiming that V42's failed learning has been fixed.

All four sheets and 96 delivered cells were visually reviewed at their saved
256-pixel size. On the six coarse frontal or mildly turned inputs selected by
the original input-only review, the current DGP gives **no convincing useful
clarity gain over resizing** in this assistant development review. Broad feature
placement often remains, but eye and mouth boundaries are smoother and less
distinct; several Phase3 outputs retain stronger central contrast. Softness
alone is acceptable under the goal. These findings concern absent useful gain
and weakened visible contrast, not a demand for invented sharp detail.

CodeFormer adds finer-looking faces, hair and texture, including on inputs whose
landmarks were judged insufficient before inference. Their truth cannot be
verified from these crops. This does not qualify CodeFormer as an identity-
preserving replacement or prove an identity error. The current DGP remains the
retained app checkpoint; no model is promoted from this comparison.

## Comparison contract

| Arm | Weights SHA256 | Declared behavior |
| --- | --- | --- |
| Resize | No weights | Native RGB128 center padding, PIL bilinear resize to 256 |
| Original Phase3 | `b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c` | Frozen original DGP, retained evaluation normalization |
| Current DGP | `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b` | Frozen selected identity-v2 DGP, same normalization policy |
| Pretrained CodeFormer | `1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7` | Fidelity1, AdaIN, internal512, bilinear return256 |

All three models receive the same exact RGB float32 tensor divided by255. The
prepared pixels also exactly match the previously frozen comparison inputs.
There is no extra alignment, denoising or display sharpening. The CodeFormer
comparison uses the common crop rather than its full detection/alignment
pipeline; this limit prevents treating the result as the publisher's complete
inference procedure. Its pinned source revision is
`b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`.
[Official CodeFormer inference source](https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py).

All72 untouched raw float32 arrays are retained separately. Delivered PNGs use
floor(raw*float32(255)) and restore the exact input padding outside observed
support. That conversion is declared delivery processing, not a measured model
improvement. Every raw array, delivered composition and gallery cell is checked.
Observed input-change MAE is a descriptive pixel-change diagnostic, not a quality
or identity metric. Both DGPs keep stored normalization statistics and disposable
InstanceNorm kernel copies. Model state hashes remain unchanged.

## Input roles and limits

The frozen subset SHA256 remains
`c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e`;
the pre-output input review remains
`41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2`.
All24 labeled development cases remain, with native short sides7–61 pixels.
No input annotation, source, split or favorable case replacement changes.
The six core cases are listed in `visual_review.json`. Seven predeclared
insufficient-information cases still require clearer crops. Weak-information,
full-profile and object-covered diagnostic controls are recorded separately
without being used to inflate the core finding.

All32 QMUL reserved cases remain unviewed in this work. Within-release labeled
roles are disjoint; cross-source and historical model/person overlap remain
unknown. QMUL is reported separately from the existing ChokePoint comparison.
No per-crop country or ethnicity is inferred. These are public native
surveillance observations released for research, with original-owner copyright
retained. Notices accompany derivatives; this is not unrestricted redistribution
permission. [QMUL-SurvFace publisher page](https://qmul-survface.github.io/).

There is no aligned clean reference. PSNR, SSIM, identity accuracy and a native
percentage structure gain are therefore not reported. The six-case count is
an assistant development judgement, not accuracy or independent final review.
No real Zamboanga sample or performance claim is added. The object-covered
profile remains a control; no seven-family completion qualification follows.

## Independent verification and timing

The bounded CPU comparison completes in167.20seconds internally and172.52seconds
under supervision, below its480/540second limits. It makes72 frozen model
forwards, with zero gradients or optimizer updates. The independent checker
completes in12.89seconds and verifies75 source bindings,200 artifacts,24 input
geometries,72 raw arrays,96 PNG compositions and96 unscaled gallery cells.
It freshly replays all24 current-DGP outputs with **maximum raw error0.0**.
It does not freshly replay Phase3 or CodeFormer; their saved arrays, hashes and
delivery compositions are audited. That scope is recorded explicitly.

Visual review follows the artifact audit and covers all24 cases/all four arms.
The creation-time `visual_review_pending` flags are retained unchanged in the
immutable run/audit receipts; `visual_review.json` supplies the later review.
This review does not fulfill the independent final reviewer requirement.

## Consequence for the 0.0069% training failure

V42's one-percent requirement remains unchanged. V42 trained the added17,952-
parameter decoder while freezing the original DGP; it did not add epochs to the
original model. Its50 updates also fail appearance preservation. Neither this
native comparison nor a longer failed trajectory demonstrates a remedy.

The already prepared original-feature diagnostic is the next finite manual VM
step. It tests feature-only, original-decoder-only and joint directions from
separate copies of the current checkpoint, with zero optimizer updates and
preservation checks. It must return and be independently audited before a
distinct epoch-training design can be justified. No returned probe is found in
the local outputs or Downloads during this review. That establishes local
availability only, not whether the user has run it on the VM.

Manual commands: [CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_VM.md>).
The full restoration/completion goal remains active and incomplete. The app,
current checkpoint, original checkpoints and historical failed stops are intact.

Evidence: `outputs/cctv_dgp_current_qmul_native_comparison_v1/` and
`outputs/cctv_dgp_current_qmul_native_comparison_v1_supervision/`.
