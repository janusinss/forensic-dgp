# Clean-code oracle diagnostic: modest gains, strong blur unresolved

19 October 2026. User testing found **“They look similar; neither fixes strong blur.”** R2 and fusion50 remain experimental comparisons, with their quality stops retained.

## Actual evidence

A new local CPU FP32 inference run completed in **255.63 seconds**. It made **43 top-level model calls**: one unwrapped R2 parity call, six HQ teacher reconstructions, and 18 natural plus 18 oracle R2 calls. **Zero optimizer updates, gradients, backwards or VM calls.** R2 and teacher states stayed exact; the main app default and all checkpoint files stayed unchanged. Peak sampled RSS was 2,043,396,096 bytes. Worker/external caps were 900/960 seconds; neither fired.

Six fixed, previously admitted TRAIN photographs supplied clear, existing Gaussian2.4, and fixed Gaussian3.2/LR64/JPEG50 inputs. Root reviewed them before outputs: 16 readable observations and two borderline strong cases. These are six photographs, not 18 independent identities. They were historically used in TRAIN. No DEV, native CCTV or FINAL pixels entered this test. The strong profile is a synthetic proxy, not calibrated CCTV or the unknown-origin wild inputs.

The intervention replaces only R2's selected codes. Genuine HQ encoder features from direct1024-to-LANCZOS512 targets are projected onto the exact frozen R2 codebook. AdaIN, observed features, fusion and weights stay fixed. Native HQ labels are retained separately; they agree with projected labels at all captured positions here. The HQ generator differs slightly from R2, so its reconstruction is a separate reference, not an exact R2-prior ceiling. **Clean-reference codes are unavailable at deployment.** This oracle is a diagnostic, not a restoration method users can run on unknown CCTV.

## Saved-image and visual checks

The independent no-neural audit verified 268 output files, 86 attempt/return records, actual unwrapped/natural parity, all 18 pairs of unchanged observed logits/features/natural selections, six projections, RAW/PNG conversion and every paired MAE. Root personally viewed all six exact256-cell sheets.

| Profile, six cases each | R2 PNG MAE | Oracle PNG MAE | Mean MAE reduction | Cases better |
|---|---:|---:|---:|---:|
| Clear | 0.036006 | 0.034468 | 4.27% | 4/6 |
| Moderate blur | 0.054767 | 0.053586 | 2.16% | 5/6 |
| Strong synthetic blur | 0.066333 | 0.064704 | 2.46% | 4/6 |

All six strong cases reduce mean PNG MAE by 2.46%. The four predeclared readable strong cases reduce 2.07% (3/4 better); the two borderline cases remain separate (one improves). This is **paired RGB error reduction**, not the old DGP 1% structure gate, SSIM improvement, identity accuracy or deployment quality. RAW shows the same small, mixed pattern.

Natural and oracle outputs remain visually close. Eye/mouth, tone and texture changes are modest; strong-blur softness and altered accessories remain. The separate HQ teacher also changes some hat/scarf details. High token disagreement even on clear inputs reflects different encoders and clean512 versus upscaled256 evidence; it is not calibrated face or identity confidence.

## Corrected history and next decision

The latest fusion run froze code prediction, but earlier V1/V2/FP32 pilots **already trained encoder/code prediction with HQ code CE and latent losses**. Repeating generic encoder unfreezing or CE would repeat earlier work. See [history comparison](outputs/codeformer_filter_fusion_next_design_20261019_v1/loss_review_peer/HISTORY_COMPARISON.md). The official [CodeFormer joint trainer](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/models/codeformer_joint_model.py#L135-L175) is the primary method reference, not proof that another such run will succeed.

This modest conditional oracle result does not justify a code-prediction-only pilot as the solution. Review **selective use of corrupted image features during reconstruction**, rather than the previously rejected global fidelity-off switch. That remains a hypothesis. A distinct finite training recipe, audited readable TRAIN inputs, unchanged preservation checks, resource/source verification and manual VM transfer are still required. No new training packet or trained checkpoint was created by this diagnostic.

The first local startup stopped on missing psutil before neural work. Its source/failure/empty log remain preserved. A host process inventory confirmed no matching live worker. Pinned psutil7.0.0 was installed, actual dependency preflight moved before spawning, and the same scientific plan ran once in a separate R1 directory. This environment repair changed no quality gate.

All original models, stopped states, splits, provenance, failures and migration backups remain. Native CCTV stays source-separated and unpaired; FINAL stays unused. No ethnicity, Zamboanga or exact-identity claim follows. Restoration acceptance, independent final review and main app Auto/override integration remain outstanding; completion development remains deferred. The full goal stays active.

Evidence: [actual result](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/result_v1.json), [independent saved-output audit](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/review_v1/audit_v1.json), [six-sheet root review](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/review_v1/root_visual_review_v1.json), [first comparison sheet](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/review_v1/sheets/tr_ffhq_03761.png). Existing model testing: [local guide](CCTV_CODEFORMER_FILTER_FUSION_LOCAL_TEST_GUIDE.md).
