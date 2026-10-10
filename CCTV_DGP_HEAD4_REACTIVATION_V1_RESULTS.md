# Head4 connected-gradient return — 10 October 2026

The manual L4 diagnostic completed without an optimizer update or a training
epoch. Its repaired deepest head and connected fusion slice receive finite,
nonzero improvement gradients on both exposed TRAIN cohorts. The corresponding
three original pieces have zero improvement gradients. This supports testing
learning through the repaired path; it does not establish clearer restoration.

The 123,543,967-byte returned archive has SHA256
`d9af857039777a58ab85a51ecca4b972e8783df9167950c8b966cb6ba12fb89a`.
Independent R2 audit verifies all327 archive members,160 saved gradient vectors,
480 component-part norms,100 initial raw/PNG/embedding records and40 loss
batches. Two fresh CPU inference replays match within the prospective
3e-6 raw/5e-5 embedding tolerances. All states remain unchanged. The VM worker
took28.95 seconds; local audit used no gradients or parameter updates.

Location: the original local return check's aggregate norm comparison. Cause:
different float64 reduction orders differ by at most3.89e-16. Fix: R2 permits
absolute1e-14 only when recomputing derived aggregate L2 norms; counts, names,
zero/nonzero route decisions and all image-quality thresholds remain exact.
The original checker, traceback, arithmetic diagnosis and prospective R2 plan
remain in `outputs/cctv_dgp_head4_reactivation_v1_return_audit_failure/`.
There was no VM rerun or re-extraction to conceal the original audit failure.

Evidence: `outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json`.
No new trained checkpoint, model adoption, native CCTV quality result or
completion-family qualification follows from this diagnostic. The app and
original checkpoints remain. The full five-milestone goal remains incomplete.
