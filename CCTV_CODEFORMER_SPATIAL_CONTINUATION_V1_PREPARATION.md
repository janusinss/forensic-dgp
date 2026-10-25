# Update-24 states retained; finite update-96 comparison prepared

21 October 2026. The next self-contained CodeFormer adaptation packet is staged, statically audited, sealed and fully read back locally. **The new continuation has not run.** It adds at most 72 updates to each audited U/S update-24 state. It changes the tested training horizon while retaining the paired objectives, matched sampling, own Adam/RNG states and original preservation rules. It does not retrain from scratch or overwrite the completed experiment.

[Five-step upload, tmux and download guide](CCTV_CODEFORMER_SPATIAL_CONTINUATION_V1_VM.md).

## Why this comparison

The [completed return review](CCTV_CODEFORMER_SPATIAL_OBJECTIVES_V1_RETURN_REVIEW.md) found real learning and six passing small TRAIN guard checks, with no material new visible structure improvement over R2. S centered-feature gain grew across updates 6/12/24 and exceeded U's targeted spatial gain, while U improved delivered image error more. This supports one controlled longer comparison of both branches. It does not prove that more training will be useful or that S is the better restorer.

Both branches continue from their own exact audited update-24 checkpoint, with the saved optimizer moments, counters and RNG. The absolute learning rate remains 0.0001 after the original warmup; warmup is not restarted. The schedule has 216 matched microbatch rows per branch, repeating the original 72-row sequence three times. There are at most 144 additional Adam updates/432 backwards total, not 144 epochs. The ten optimized observations still come from six repeated TRAIN photographs; no DEV, native CCTV, FINAL or unknown user image is optimized.

All 515 R2 base tensors remain frozen; 24 adapter tensors/51,328 scalar parameters are trainable. Both own-24 full image/upstream parity sets and all ten feature-array/metric parity sets must pass before the first additional backward. The worker builds ten fresh frozen-prefix caches. Guards run at absolute updates 30/36/48/72/96, with a checkpoint saved before every guard. Strong face outputs run only after guards pass at 48/72/96. All branch stops remain in the returned evidence; no automatic model choice or app promotion occurs.

## Verified transfer identity

| Artifact | Exact value |
| --- | --- |
| Execution archive | `outputs/cctv-codeformer-spatial-continuation-v1-execution.tar.gz` |
| Archive bytes | 668,645,843 |
| Archive SHA256 | `7a3c1a067ceef7bfe5a9ee6cf461bd9daaddeb8ea509a3eb7e429de04e26db5f` |
| Protocol SHA256 | `c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed` |
| Static archive members/payload | 230 regular files / 733,198,280 bytes |

The package carries its own R2 initializer, clean training features, exact U24/S24 resume states, parity references, parent audit/export provenance and required sources. It needs the existing VM Python environment, but no earlier experiment directory. The original checkpoints, original return and failure evidence remain intact. The ASCII/LF checksum has no CR character. Compression uses explicit `gzip.GzipFile` around an uncompressed tar stream, following the path that successfully recovered the prior VM export.

## Finite operation and storage bounds

The no-stop maxima are 272 full outputs, 3,136 adapter evaluations, 432 backwards and 144 optimizer steps, plus ten frozen-prefix passes. Mandatory early guards are included. Source-enforced output format caps and the exact closed namespace give a conservative 526,363,023-byte return bound (501.98 MiB) under 512 MiB. The physical file bound is 1,302 under 2,000. The 2,133 allowed final/partial names are alternatives; the single-writer/no-resume path prevents all partial alternatives from coexisting. The proof explicitly reserves retained failure/metadata partials. VM-only cache has its own 512 MiB cap; returned archive has a 544 MiB cap.

Require 6 GiB free after installation and keep 4 GiB free throughout. Keep 10 GiB host RSS and 18 GiB CUDA-reserved caps. Optimization and full parity/guard/probe time each have a shared 900-second ceiling. Worker, supervisor and whole-run limits are 2,700/2,760/3,120 seconds, with 40 seconds for bounded whole-deadline termination. Estimated worker time is 8–12 minutes from the preceding measured L4 run; this is not a measurement of the new continuation. Actual target runtime, memory and resume parity must pass during the user's manual launch.

## What was verified locally

The independent source review checked resume ordering, own checkpoint/Adam/RNG installation, absolute update/LR, early stop ordering, unchanged quality rules, operation reservations, Python 3.10 export format and namespace/resource bounds. Fourteen meaningful pure fixtures passed. Physical static readback checked all 229 pinned files, both checkpoint byte joins, eight own-24 image references, twenty own-24 feature-stat references, decoded RGB hashes, all 24 clean feature arrays/RMS and the exact sampler. The actual `--verify-transfer` CLI exited 0 without importing Torch.

The installed Bash launcher is at the experiment root. An actual staged-wrapper test using a fake Python executable verified its working directory, supervisor path and arguments without running the worker. Git Bash initially failed inside the Windows sandbox before executing the wrapper; the same inert test passed elevated. The original source and stage were unchanged. Archive readback streamed every member and checked exact bytes against the static stage. No new neural, gradient, optimizer, training, VM or browser call ran during preparation.

The independent future return auditor is prepared, with Linux RSS sources bounded independently and VM-ordered floating-point gate sums. It has not audited a future run. Safe checkpoint tensor/Adam/RNG content review and all planned visual sheets remain required after return. Static checks do not establish an unexecuted CUDA graph, successful training or accepted image quality.

Full development and native CCTV comparison, original preservation requirements, reserved FINAL review, user acceptance and the main app's Auto/override/download checks remain outstanding. The app default is unchanged; completion development remains deferred. The earlier storage offload remains closed and was not repeated. The overall goal remains active and incomplete.

Evidence: [source peer](outputs/codeformer_spatial_continuation_packet_review_v1/independent_source_peer_v1.json), [static/CLI readback](outputs/codeformer_spatial_continuation_packet_review_v1/independent_static_CLI_readback_v1.json), [stage and sealer review](outputs/codeformer_spatial_continuation_packet_review_v1/independent_stage_and_sealer_review_v1.json), [staged Bash check](outputs/codeformer_spatial_continuation_packet_review_v1/actual_staged_wrapper_mock_python_v1_r1.json), [manual guide review](outputs/codeformer_spatial_continuation_packet_review_v1/independent_manual_guide_check_v1.json), [archive seal/readback](outputs/codeformer_spatial_continuation_packet_review_v1/execution_archive_v1.json), [independent full archive readback](outputs/codeformer_spatial_continuation_packet_review_v1/independent_archive_readback_v1.json).
