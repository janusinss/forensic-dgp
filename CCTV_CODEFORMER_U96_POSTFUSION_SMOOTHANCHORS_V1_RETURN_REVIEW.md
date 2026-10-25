# Smoother-anchor U96 trial: measured learning progress, retained quality stop

The new loss helped the finite training process. Global structure gain rose from0.01313% to0.91927% in delivered PNGs; localized center gain rose from0.00436% to0.66466%. All preservation and dark checks passed. All four structure endpoints still missed the unchanged1% screen. The authoritative result remains QUALITY_STOP at64; U96 stays current.

## Verified return

Archive `outputs/cctv-codeformer-u96-postfusion-smoothanchors-v1-results.tar.gz` is1,166,119,912bytes, SHA256 `003674db37fb0bb317067591fcc324b32ea72bb34fa9735e8c3ae5653ac37caf`. Checksum and export match. ProtocolSHA `7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a`. All3,462 regular files/1,308,832,513payload bytes and243 returned nonweight static files plus protocol match; expanded gzip bytes1,315,143,680 and CRC/footer are verified. All244 local sealed static files remain exact.

Worker416.2247seconds, supervisor419.0522seconds/exit2/no timeout, export66.4009seconds.64 confirmed Adam commits/64 attempts, no unknown optimizer outcome. Failure phase `snapshot_u64_quality_gates`; no success result/final invariant receipt is invented. All committed checkpoints, per-commit frozen-state associations, partial/failure records and original weights remain retained. Resource limits passed; initial storage/workload on a future VM must still be checked afresh.

| TRAIN-monitor structure endpoint | Prior L1-anchor RAW | New RAW | Prior PNG | New PNG | Required |
|---|---:|---:|---:|---:|---:|
| Global blur, whole image |0.016153%|0.921550%|0.013132%|0.919274%|1% each|
| Local blur, central128 |0.010889%|0.674250%|0.004359%|0.664661%|1% each|

These are reductions in paired adjacent-residual error on reused TRAIN-monitor photographs, not percentages of recovered identity or native CCTV quality. Both recipes started from exactly the same U96, inputs, head seed, sampler, Adam/LR and64-update cap. The common anchor criterion changed. Scalar gate sums now explicitly match VM3.10 arithmetic; no threshold/tolerance changed. A larger numerical gain alone does not qualify the model.

## Independent audit coverage

[Pixel audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/saved_pixel_audit_v1.json>): all712 saved RAW/PNG pairs, floor conversion, teacher/parity associations and regional metrics verified. All12 entire gate dictionaries match exactly.256 objectives were independently reconstructed using both new SmoothL1 terms and retained L1 diagnostics; maximum GPU-vs-CPUfloat64 objective discrepancy5.05e-8, CPUfloat32 discrepancy5.96e-8. These reported reduction differences do not relax any quality gate. No Torch/model/neural/gradient/optimizer/VM call occurred in the pixel audit.

[State audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_state_audit/saved_state_audit_v1.json>): all65 head/resume checkpoints,64 saved gradient sets and4,644 arrays, all64 Adam commits/moments/steps, sampler exposures and RNG verified. All36 trainable tensors changed; final head movementL2=0.5714068613 versus0.3471820095 before. Norms0.02581–0.14920 never triggered clipping. Independent NumPy recurrence agrees within explicitly bounded FP32 arithmetic differences. Torch only decoded bounded CPU weights-only saved checkpoints; no model/optimizer construction, step, forward/backward or CUDA initialization occurred locally. Frozen539 payloads are omitted, so unchanged claims remain exact original-file and source/runtime-receipt associations, not independently decoded frozen return values.

[Metadata peer](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_peer/independent_return_review_v1.json>) passed29,128 semantic checks. [271-check crossread](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/peer_state_and_metadata_crossread_v1.json>) joins actual pixel/state/metadata evidence. Auditors were independently reviewed with20 saved-pixel and11 saved-state parser/numerical fixtures before reading this return. Their repaired preparation sources and findings remain retained; those parser repairs changed no VM source, model, result or quality requirement. Partial/incomplete returns have explicitly narrower audit coverage and must never manufacture a full64 success.

## All planned images reviewed

[Root visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/root_visual_review_v1.json>) records personal review of all30 sheets,952 total exact256 cells and all712 generated PNG cells exactly once. There are subtle tonal/edge changes and no obvious new catastrophic artifact. Existing softness and clean-target differences in eyes, teeth, skin and other facial details remain. No clear, consistent jump in useful facial readability over U96 is established. This is an unblinded assistant TRAIN review, not user acceptance or independent human FINAL review. Teacher comparisons do not establish true identity.

[Saved trajectory plot](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison/structure_trajectory_v1.png>) compares both fixed recipes on identical inputs. Lines connect the six measured snapshots; they do not forecast a later gain. New PNG global gains at16/32/64 are0.14027%,0.33208%,0.91927%; local gains0.07168%,0.22840%,0.66466%. The late upward course and consistent saved gradients support further investigation of this learning route, while preserving the stopped state.

## Next design decision, with safeguards

[Saved-course method review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison/NEXT_METHOD_REVIEW_V1.md>) finds actual paired objectives decreased across the four fixed data passes and same-slot gradient directions became more persistent than in the prior L1 trial. Clear L1 behavior deviation also increased. The cause is not proved to be a slow LR, and doubling LR cannot be assumed safe. The smallest historical clear mean preservation headroom across snapshots was only0.00239893; the smallest case headroom was0.00389103. These are observed scalar margins, not an authorized additional pixel or identity allowance.

Prepare one distinct fresh64-update comparison changing only maxAdamLR5e-5 to1e-4. Keep complete U96/head seed, SmoothL1 beta0.01, all data/order/coefficients/optimizer settings,64-update budget, snapshot/preservation/1% gates and resource caps fixed. Start fresh with Adam; do not resume B2, append epochs, alter thresholds or choose a lucky intermediate checkpoint retrospectively. The factor is justified as a limited learning-dynamics test, not because0.92 is close to1 or as a guaranteed quality improvement. A separately verified packet/manual guide is required before execution. No new training or VM call was launched here.

If a later finite TRAIN screen passes, broader fixed DEV, separate native/unpaired CCTV, user acceptance and independent FINAL review still decide usefulness. Stable faces or a passed1% screen cannot replace them. Preserve every original model, label, split, source term and failed proposal. These96 synthetic cases remain16 optimization/8 adaptive TRAIN-monitor photographs, not verified separate people or unseen validation. There are no new native/wild/FINAL results, no ethnicity or Zamboanga-performance claim.

Optional protected-surroundings preparation remains separate, with exact protected input pixels, original/derived previews, correction, confirmation and bypass. Prior paired studies did not justify always-on blur. Restoration stays first; completion/removal-mask development remains deferred, with its later full covering-family scope intact. The existing app/defaults are unchanged. The full active goal remains incomplete.
