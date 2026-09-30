# Matched coverage comparison preparation — 30 September 2026

No model training or inference was performed. This document records preparation,
not a ready-to-run VM experiment or evidence of output improvement.

## Verified comparison inputs

Control: `dataset/detector_glare_review_v3/manifest.json` (68 train).
Treatment: `dataset/detector_training_extension_v2/manifest.json` (73 train).
Validation25/test7 records are unchanged. Use a common initial checkpoint for
both arms; do not reuse earlier results as an exact matched control.

The replay pool is the existing 352-source expanded-data manifest. One source,
`dataset/asian_faces/asian_face_06721.jpg`, is now a real clear control and was
excluded from both replay arms. Every pool source hash and original train
membership was checked. Select a shared 200 sources (100 Asian, 100 FFHQ) with
seed42 after union exclusion of all reviewed source/image hashes and the
pool's held-out exclusions. This protects the added examples from conflicting
synthetic targets; it does not certify identity separation or source cleanliness.

## Sampling confound fixed in a separate sampler

Legacy `MixedBatches` consumes one RNG sequentially across real and synthetic
groups. Changing real pool length can change later synthetic draws and batch
positions, even with the same seed. It remains unchanged for historical runs.

New `coverage_protocol.CoverageBatches` uses one independently seeded stream per
group plus a separate batch-permutation stream. Tests verify synthetic case IDs
and positions remain identical when real groups change from 43/25 to 47/26,
group balance holds, epochs are reproducible and invalid/overlapping groups fail.
Three tests passed, without any optimizer operation.

`scripts/prepare_coverage_protocol.py` froze ten epochs x21 updates, batch8,
two samples per real-covered/real-clear/synthetic-covered/synthetic-clear group.
Each arm has 840 real and 840 synthetic slots, sees every real training image,
and uses exactly the same synthetic case order and positions. Per-image real
frequency differs because the real pool size is the experimental variable.

Evidence: `outputs/coverage_protocol_v1/protocol.json`, SHA256
`46c438f6e72e9bd4709102dd8147dc926fff84de0e7e8a89f3bee1c22a9590fd`.
It stores source hashes, input manifest hashes, complete schedules and exposure.

## Next implementation

Build a VM-only runner that consumes these frozen schedules, archives actual
initial tensors and the common generated replay tensors, and checks hashes before
optimization. Choose one documented existing architecture/loss configuration;
change only the real dataset. Keep generator frozen and all original real/synthetic
selection gates unchanged. Save per-stratum masks and completion previews before
considering promotion. The five additions make this a bounded coverage diagnostic,
not an adequately powered generalization study or a guaranteed retention fix.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/`. Artifacts currently local only. No SSH run is due yet.
