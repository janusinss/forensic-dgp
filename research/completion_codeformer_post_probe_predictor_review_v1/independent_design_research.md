# Independent post-probe predictor review

26 October 2026. Design recommendation only: no packet, model, optimizer, gradient or VM run created here. R1 stays selected. This review uses the completed same-codebook probe and retained source, not a claim that its oracle can be reproduced from a covered input.

## Recommendation

Review one finite **predictor-only CE adaptation from exact R1**, keeping the encoder, learned R1 fusion and original inpainting prior fixed. Train only the existing post-encoder token prediction path. Continue using ordinary covered inputs and ordinary predicted codes at evaluation and deployment; never pass clean-context oracle indices to an app.

The admitted metadata partitions all 497 state tensors as follows. The companion `independent_predictor_partition.json` lists every name, shape and R1 tensor hash, plus exact evidence pins. Before any VM optimization, assert that the runtime named parameters and optimizer allowlist match these intended sets exactly; the metadata count alone is not an optimizer verification.

| Path | Tensors | Scalar entries | Treatment |
|---|---:|---:|---|
| `position_emb`, `feat_emb`, nine `ft_layers`, `idx_pred_layer` | 114 | 19,450,880 | Only intended learning path |
| `encoder` | 164 | 29,416,000 | Freeze at R1 |
| `fuse_convs_dict` | 54 | 9,592,960 | Freeze at R1; retain w=1 |
| `quantize` and `generator` | 165 | 34,193,027 | Freeze original prior |

The predictor comprises position embedding (1 tensor), feature projection (2), transformer (108) and index head (3). All 383 other tensors must remain bitexact. Frozen encoder features keep the existing fusion conditioning unchanged; predictor learning can change the selected discrete codes without adapting the fusion network around a moving encoder.

## What the actual probe establishes

Independent audit `4f5042edc19496bd3dee515dc052928c03e4f6d9e12b246efdcca4262a54d02f` verifies all 192 outputs, 196 numeric archives, four fresh parity pairs, all 48 historical R1 outputs and 33,551,076 exact outside-mask RGB bytes.

At fixed w=1, unavailable clean-context SAME-codebook indices (C) reduce delivered PNG hole-L1 error relative to actual R1 predictions (A) by **16.4881%** on 32 clear monitor cases and **16.3641%** on 16 challenges, with all declared aggregate raw/PNG hole/boundary subgroups nonworse. However, three individual holes worsen (2 clear, 1 challenge), ten individual boundaries worsen (8 clear, 2 challenge), and gray chroma remains: the eight gray challenge cases average 4.6712 channel levels of PNG spread under A versus 6.2603 under C. The oracle is not an exact hidden identity or a mathematical optimum.

Removing fusion with actual codes (B versus A) raises hole error **24.1798% / 30.1557%**, and all 48 individual holes and boundaries worsen. Removing fusion is therefore not the recommended deployment or training direction. Oracle without fusion (D) has lower macro hole error than A but regresses boundaries and fails preservation requirements; its central gains do not justify discarding fusion.

Actual prediction agrees with clean-context labels on 63.2690% of the entire 256-position grid in the clear cohort and 32.4951% in challenges. These are whole-grid index agreements, not accuracy specifically inside the reviewed mask, image-quality probabilities or identity confidence. Lower challenge agreement supports investigating prediction under degradation, but the oracle improvement does **not** guarantee CE learning can achieve 16%, generalize to unseen identities, or resolve ambiguous hidden anatomy.

## Why this differs from the rejected paths

Retained V2 changed both encoder/predictor (278 tensors) and fusion (54), alternating CE plus MSE toward the original **unquantized** clean encoder latent with image/boundary fusion loss. Its joint quality result failed and its partial checkpoint remains unusable. Repeating that joint schedule or merely appending epochs would not isolate the demonstrated prediction opportunity.

V3/V4 froze the predictor and changed fusion. V4 aligned loss to delivered 256 pixels but did not materially improve the generated holes. The probe now shows a better same-book code choice can help at unchanged R1 fusion. That is evidence for a distinct prediction-path test, not proof that frozen code selection was the sole limitation.

The local architecture's `code_only=True` output provides logits and the encoder latent before hard top-1 lookup. CE on those logits has a differentiable path to the intended predictor. In contrast, with the encoder frozen, an MSE applied to its returned `lq_feat` is constant with respect to all 114 proposed trainable tensors. Retaining that term would change reported objective values without providing predictor gradients; omit it rather than pretending it trains the predictor. Do not use a straight-through hard-code image gradient or invent one as part of this first isolating test.

Official CodeFormer supervision can use fixed cached nearest-code labels and CE on logits; its optional feature term compares the low-quality encoder latent with **quantized codebook embeddings**, not the old unquantized target. That feature loss would also require an encoder learning path. This supports the declared CE interface, not our exact freeze policy or a guaranteed completion gain. [Official index-training implementation](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/models/codeformer_idx_model.py).

## Concrete finite review to freeze before a packet

- Exact R1 weights and frozen original prior; fixed w=1 and ordinary app preprocessing/delivery. One declared CE-only path: mean CE across all 256 predicted indices against the already audited original-inpainting SAME-codebook cached labels, divided once across four one-case microbatches. No new HQ512 teacher IDs, prior swap, GAN or fusion/encoder updates.
- Reuse the existing 32 FIT identities and pre-materialized 256 FIT cases, including retained RGB, strict gray and irregular mildly degraded conditions. Keep the eight monitor identities and all 48 probe cases out of optimizer updates. They are already adaptively exposed development evidence, not independent validation or FINAL. Do not train on the oracle monitor pairs to reproduce this diagnostic score.
- Review a maximum **32 predictor steps / 128 backwards**, four microbatches per step, with snapshots at 0, 8, 16 and 32, and no automatic continuation. Use the retained 1e-5 Adam rate, betas (0.9, 0.99), epsilon 1e-8, zero decay and finite L2 clipping at 1 for this scope change; avoid confounding it with an LR sweep. The initial CE, finite gradients and update deltas must be recorded, but lower CE or higher index agreement cannot qualify the model.
- Restore the full R1 parent and its RNG boundary. The predictor-only Adam is a new optimizer topology: map the exact parent per-name step/moment slots for these 114 tensors in declared order, preserving all original parent states and verifying the copied slots bitexact. Do not load 278-parameter optimizer IDs blindly into a 114-parameter group. No unreported optimizer reset. Keep frozen encoder/fusion optimizer state preserved as historical parent evidence.
- Propose 15-minute hard / 14-minute cooperative worker limits, 7 GiB start plus payload, 2 GiB free floor, 20 GiB CUDA allocation cap and bounded output/save/export counts. Exact planned whole/code-only/teacher/empty calls, serialization bounds and interruption-safe terminal delta+Adam/RNG/sampler evidence must be derived and independently admitted before transfer; this recommendation is not a launch packet.
- Audit actual hard-top-1 outputs under ordinary inference at every saved snapshot. Stop on nonfinite state/gradients, any changed frozen tensor, resource limits or the retained catastrophic regression rule. Keep all stopping evidence. No mandatory one-percent success at step8 merely to manufacture an early conclusion; the finite final endpoint must satisfy the unchanged clear and challenge >=1% PNG hole-error reduction and all raw/PNG hole/boundary subgroup nonworse screens. Visually inspect all planned outputs, strict-gray color, public unpaired covering families, exact visible bytes and empty/refusal controls before considering adoption.
- A valid full state or passing TRAIN screen remains insufficient for promotion. Broader development outputs, user acceptance, unseen final identities and blinded independent review are still required; preserve all R1 alternatives. If code agreement improves while delivered quality does not, reject the trial and review representation/labels or uncertainty rather than adding unchanged epochs.

The specific policy choices above are an inference from current evidence. Official stage-II restoration settings use a different 1024-code configuration and much larger training scale; their LR/iteration count must not be transplanted as a ready recipe for this 512-code inpainting adaptation. [Official stage-II settings](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/options/CodeFormer_stage2.yml).

Remaining limits: known pretraining overlap, synthetic masks and only eight adaptive monitor identities; clean-context codes can encode hidden features unavailable from a real mask; code CE optimizes a representation surrogate; no public CCTV clean truth, automatic-mask qualification or model-specific confidence calibration exists here. Request less-covered input when visible context is insufficient. No manuscript or app claim changes follow from this review.
