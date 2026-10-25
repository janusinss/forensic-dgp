# U96 decoder-readout v1: audited quality stop at64

**The VM completed64 optimizer updates, but this candidate did not qualify as an
improvement over U96.** This was a model-quality stop, not a training crash or
download/export failure. Retain the returned checkpoint and failure evidence.
Keep U96 as the experimental incumbent. Do not rerun the old guide, append epochs
or relax the requirements for this failed recipe.

Returned archive: `outputs/cctv-codeformer-u96-decoder-readout-v1-results.tar.gz`,
963,822,573bytes, SHA256
`73c8f4bcca5814cb845058cbc7d553ad3ea1f29e97967449cad66ac848fed2c4`.
It matches both downloaded companions and the sealed protocol
`a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e`.

## What ran

Complete U96 was initialized from its retained R2 base and24 adapter tensors.
Only the original final RGB convolution's weight and bias trained:2tensors,
1,731scalars. The other537 tensors were frozen by the source/runtime checks.
Fresh Adam used maximum LR0.00005 with four-update warmup. The64updates represent
256role exposures/four passes through64correlated optimization cases, not64epochs.
All four roles used aligned photographic clean targets.

The saved ledger independently establishes712reserved and returned full forwards,
256backwards, one fresh optimizer,64attempted/returned updates and65committed
checkpoints. No optimizer outcome remains unknown. The failure occurs at
`snapshot_u64_quality_gates`, with worker exit2 and no timeout. Worker elapsed
time was1,476.1seconds (24.6minutes); supervisor elapsed1,480.6seconds. Export
completed in73.4seconds. All are below their sealed stops. The10–20minute planning
estimate underestimated this route's actual runtime.

Observed peak worker RSS was2,322,731,008bytes and peak CUDA reserved memory
1,549,795,328bytes. Minimum sampled free disk was156,663,394,304bytes. These are
saved VM observations, not a fresh current VM inventory.

## Why the final quality check failed

Each result below is relative to the same frozen complete U96 output on the eight
reused adaptive TRAIN-monitor photographs. RAW and floor-delivered PNG remain
separate; no epsilon was added to the quality thresholds.

| Structure endpoint | RAW gain | PNG gain | Required |
|---|---:|---:|---:|
| Local Gaussian1.2, central128 |0.124732%|0.131680%|At least1% in both views|
| Global Gaussian1.2, whole image |0.102673%|0.103591%|At least1% in both views|

All four endpoints failed. The best gain,0.131680%, is about13%of the required
1%improvement; it is not13%better image quality. This new image-adjacent metric
must not be compared numerically with older DGP feature-score percentages.

| Low-light clean-target mean MAE | Frozen U96 | Update64 | Change |
|---|---:|---:|---:|
| RAW |0.095769199378|0.097973030171|+0.002203830793|
| PNG |0.097589839593|0.099799997960|+0.002210158367|

Both dark MAE nonregression checks failed, even though dark adjacent error
improved slightly. There are therefore **six failed terminal checks**. All
24legacy guards, ordinary32-case monitor movement guards and update16 feasibility
trend passed. Passing those limited movement guards does not satisfy the separate
terminal structure and dark-accuracy requirements or repair inherited DEV failures.

## Evidence independently verified

| Audit | Actual evidence | Result |
|---|---|---|
| [Transport/events R1](outputs/codeformer_u96_decoder_readout_v1_return_transport_audit_r1/independent_transport_event_audit_v1.json) |3,482files/1,093,444,038payload bytes;259sealed nonweight files;3,445events;2,734resource samples |33,009checks; complete64-update reserved/returned prefix, quality stop retained|
| [Saved pixels](outputs/codeformer_u96_decoder_readout_v1_return_pixel_audit/saved_pixel_audit_v1.json) |All712RAW/PNG pairs,460unclipped arrays,8actual stock tuples,60initial RAW/PNG parity comparisons,256clean-target objectives and six56-case banks |68,019checks; exact metrics/gates; all712PNG floor deliveries match|
| [Saved states](outputs/codeformer_u96_decoder_readout_v1_return_state_audit/saved_state_audit_v1.json) |65two-tensor checkpoints,64postclip gradient sets, fresh Adam/RNG/sampler and failed-state parity with commit64 |708,383checks; exact R2 readout initializer; NumPy reconstructs every Adam step|
| [Root visual review](outputs/codeformer_u96_decoder_readout_v1_return_visuals/root_personal_visual_review_v1.json) |All22planned sheets covering712unique generated outputs,56reference repeats and1,120visible cells |No consistent conspicuous useful structure gain over U96|

The pixel audit found zero discrepancies in reconstructed float64 metrics/gate
values. Maximum FP32 objective readback difference was4.07e-8. Independent Adam
reconstruction's worst parameter discrepancy was7.45e-9, within its separately
declared arithmetic-only envelopes. Those numeric readback envelopes do not relax
any quality requirement. The optimizer actually changed the readout: total
parameter-change L2 was0.03300734. A disconnected or unused optimizer is not the
explanation for the weak output gains.

Separate peers reviewed the [pixel auditor](outputs/codeformer_u96_decoder_readout_v1_return_peer_review/pixel_source_review_v1.json),
[state auditor](outputs/codeformer_u96_decoder_readout_v1_return_peer_review/state_source_review_v1.json)
and [transport/visual bindings](outputs/codeformer_u96_decoder_readout_v1_return_peer_review/transport_visual_source_review_v1.json).
Every displayed cell was independently joined to its exact source pixels. Root
personally viewed all22sheets; the tool downscaled full pages for display, so no
pixel-exact *visual* claim is made. The saved cells remain exact256×256. This is
researcher/agent development review, not blinded independent human FINAL review.

Transport R0's tar-tail check missed bytes buffered inside the tar reader in a
synthetic mutation test. R0/source/tests remain retained. R1 checks the complete
decompressed EOF suffix, including reader-buffered bytes, and rejects the fixture;
the actual archive passed with9,216zero padding bytes. This checker repair changed
no model, archive, output or scientific gate result. Fixture counts are12transport,
13pixel,12state and6renderer tests; peers performed additional scoped checks.

## What this teaches us before further training

The [matched optimization-exposure readback](outputs/codeformer_u96_decoder_readout_v1_return_root_diagnosis/matched_training_exposure_readback_v1.json)
uses all64rolecases rather than selecting favorable examples. First and fourth
appearances are48updates apart. The fourth appearances are before steps49–64,
meaning parameter states after48–63commits; they are not one common update64model.

| Role,16photographs each | Mean combined objective decrease | Photographs with lower objective |
|---|---:|---:|
| Clear |4.240536%|15/16|
| Global blur |4.312127%|15/16|
| Local blur |4.288489%|15/16|
| Low light |−1.428238% (worse)|0/16|

This supports a concrete distinction: the final shared readout learned something,
but the normal/blur objective gains did not produce enough additional structure,
and the low-light tradeoff appeared even on optimization photographs. It does not
identify a unique causal layer failure, establish gradient conflict from separate
component vectors, or prove that more updates can solve the tradeoff. Only joint
gradients and component scalar losses were saved. Do not extrapolate the tiny
gain linearly into more epochs.

The next design review must address **spatial feature changes and low-light
selectivity together**, before any distinct finite pilot is justified. Repeating
the same final convolution with more updates or merely increasing LR is not the
next experiment. Preserve this result as a failed parameter-path-and-supervision
package; do not claim the final RGB layer is universally incapable.

These are small reused photographic TRAIN cohorts, not independent DEV/FINAL or
native CCTV ground truth. No native/wild/FINAL input was trained in this pilot.
Returned archives omit the complete post-training frozen537 tensors; their
exactness is a VM/source-bound assertion, not an independent rehash of those
values. No local model inference, optimizer construction/update, training or VM
connection occurred during this return review. Torch was used only for bounded
CPU `weights_only` saved-state decoding. Original checkpoints and the app remain
unchanged. Optional Protect/Apply and bypass remain separate; automatic blur
matching, independent FINAL and later completion scope remain unfinished.
The full goal remains ACTIVE/incomplete.
