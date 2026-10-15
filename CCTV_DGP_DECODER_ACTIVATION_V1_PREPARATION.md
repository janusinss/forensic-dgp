# Decoder activation V1: independently verified preparation

Status on15October2026: **VERIFIED / PREPARED / UNRUN**, not a trained candidate.
The preceding hinge-repair candidate remains rejected at0.113913%RAW and
0.128470%PNG structure-error reduction with30 preservation failures in each
path. No checkpoint has been selected or promoted.

This diagnostic checks a specific remaining assumption: whether the chosen
gradient of the corrected current A decoder explains finite objective changes
on the same five-image recognizer path. The earlier saved diagnostic used
15-image batches. Fresh five-image measurements keep that evidence separate.
The full objective retains its original weights, anchors and regression slack;
the per-image brightness guard retains coefficient10/fraction0.2. The two
derivative references are the existing first two scheduled TRAIN blocks,
`tr_asian_00176` and `tr_ffhq_53945`, chosen before new measurements.

The worker replays only initial,1/16,1/4 andA1 vectors already retained from
the rejected run. All400 fresh RAW/floor-PNG images must equal those controls.
It records80 exact batch-five objective decompositions. On the two references
only,70 full parameter gradients measure underlying RGB,SSIM,ArcFace,landmark
error,channel-mean and complete objective slopes along the retained A1 change.
Actual gradient-pass embeddings are retained to avoid assuming grad/no-grad
bit parity. No optimizer, backward call, new amplitude, new direction search,
parameter proposal, new epoch or automatic follow-up exists in this worker.

The one-sided calculation is an affine residual model using selected neural
branches and idealized real-valued clipping. Actual0/1 mean-shift clipping and
finite regression slack are explicit. It is not a certified derivative of the
literal float32 or PNG program, nor proof that every decoder direction fails.
Only two references have measured derivatives; the other18 have finite replay
records. Eight finite-versus-linear comparisons retain these scope limits.

**44 CPU tests pass**, including120 retained scalar rows, strict roles/fields,
actual Windows pre-neural guard, malformed/SIGKILL exports, count and gradient
corruption checks. A separate local-importer suite tests hashes and safe archive
names before extraction. Git Bash syntax and LF-only shell/sidecar checks pass.
Independent packet review verifies17 bound assets,18 archive files and1,577
ancestor bindings. Numerical helper bounds are frozen prospectively:objective
arithmetic1e-10, norm metadata2e-6relative/1e-12absolute, clipping-factor metadata
twofloat32ULPs. Actual clipped bytes remain exact, first-gradient replay remains
maxabs2e-7/cosine≥0.999999, and all scientific quality gates remain unchanged.

The finite limits are in the protocol and manual guide. No GPU time was consumed
by this preparation. Syntax/CPU success does not verify CUDA execution. Windows
fixture ACL failures, source revisions and all preceding stops are retained.

RouteA reuses the corrected DeblurGAN-v2-compatible conditional reconstruction
path. Frozen external GAN features are cached for source parity and are not
injected into this A decoder. This diagnostic is not a newly trained GAN or
qualified generative face prior. Native CCTV, reserved final identities and
covering pixels are not used. The permanent baseline stays
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

[Independent preparation receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_activation_v1_preparation/independent_preparation_audit.json>).
[CPU test evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_activation_v1_preparation/cpu_tests.json>).
[Five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_ACTIVATION_V1_VM.md>).
[Selected causal review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md>).
All five goal milestones, useful native DEV/final review and seven separately
reported automatic/assisted covering families remain required and unqualified.
