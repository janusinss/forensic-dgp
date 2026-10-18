# Frozen CPU filter comparison

This is local inference, not training or another epoch. Compare three exact
retained models: untouched CodeFormer, the AdaIN50 parent retained in R2
snapshot0000, and the genuine R2 trained50 state. The worker constructs one
model at a time, freezes all parameters, uses CPU FP32/eval/no_grad, and never
constructs an optimizer, computes a loss, calls backward or changes the app.

The fixed96 synthetic pairs and input-only decisions stay hash-bound. Infer the
72 readable/borderline inputs with each model; retain all24 insufficient cases
as explicit skipped rows with no output. Borderline remains uncertain. DEV
stays excluded from optimization. The appearance flag on `va_ffhq_08025` stays
visible. Unknown user examples, native CCTV and reserved FINAL are not included.

The forward path matches the existing research comparison: NumPy float32/255,
bilinear512, [-1,1], CodeFormer w1/AdaIN, clamp the native RGB output, bilinear256,
and floor(float32*255) PNG. There is no alignment, enhancement, completion,
input-dependent routing or extra brightness processing. The worker does not
score against clean targets; independent CPU analysis audits saved pixels.

Estimate20–30 minutes from prior local timings. Enforce216 forward reservations,
3600seconds wall, processRSS≤3GiB, availableRAM≥768MiB and free disk≥1GiB. Check
resources every second and around each forward. The resource guard stops this
worker alone; it does not stop another app or server. A watchdog stop can leave
a reserved forward or files without a commit. Preserve that partial evidence;
there is no automatic resume, retry, quality acceptance or promotion.

Every forward receives a durable journal reservation before its call. RAW,
PNG and evidence files are written exclusively and their commit is recorded.
Canonical515 tensor fingerprints must remain exact before/after each model.
Prospective caps are660files and256MiB; all final hashes must close. Source,
model/data/review pins and pure CPU fixtures require root and peer review before
the216 actual forwards. A completed comparison is not an accepted model.
