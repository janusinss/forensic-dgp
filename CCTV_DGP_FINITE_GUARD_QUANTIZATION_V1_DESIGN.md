# Saved-output PNG conversion diagnostic

The finite-guard V1 R1 return rejected all three proposed parameter changes.
Raw preservation passes in both exposed TRAIN cohorts, while delivered PNG
ArcFace preservation rejects every proposal. No change or epoch was accepted.
All 400 outputs have received primary-assistant development visual review and
remain soft. The unchanged 1% structure requirement also fails independently.

Test one processing hypothesis before requesting another VM run: downward floor
quantization contributes to the discrepancy between raw and delivered results.
This local diagnostic compares the existing float32 floor conversion with one
fixed nearest-half-up conversion, `floor(float64(raw)*255 + 0.5)`. It uses exactly
the 100 saved cases and all four saved variants: baseline and all three rejected
proposals. Both the baseline and proposal receive the same conversion policy.
Camera pixels outside observed support remain exact. There is no search over
rounding policies, scales, cases, model parameters or scientific thresholds.

Fresh frozen CPU ArcFace embeddings are computed for floor, nearest and paired
target in the same 15-image batch context (five cases per batch). All 400 PNG
metric rows, source/profile groups, mean-shift controls and quality decisions
are recomputed. Historical raw/VM decisions are retained separately. CPU replay
tolerance applies only to embedding agreement, never to preservation gates.
The original 1% structure, 1e-12 MSE, 1e-6 SSIM/ArcFace, nonnegative source gain
and 0.2 brightness-share requirements are unchanged.

The prospective checker and complete source bindings are frozen before execution.
It independently checks every new PNG against the saved raw pixels, every metric
and group against saved embeddings, all historical floor failure keys, four
preselected fresh embedding cases, and unchanged sources and recognizer state.
It does not replay DGP or autograd. Worker limits are 600 internal/630 external
seconds; audit limits are 300 internal/330 external seconds. CPU threads: four.
Worker recognizer forwards: 80; audit: four. DGP/completion forwards, gradient
queries, parameter updates and epochs: zero. At most 512 MiB of derived output.

This is processing diagnosis on exposed paired photographic TRAIN data, not
training, useful native CCTV evidence or final evaluation. Neither policy nor
any rejected proposal is adopted in the app. A numeric conversion improvement
cannot supply missing facial detail or override the retained 1% failure. New
rounded images are not visually qualified by the existing floor-image review.
No historical pilot, failed gate or source is overwritten. No VM is started.

The DGP-led app, preservation of all visible features, independent final review,
all seven automatic/assisted covering families and inline Playwright verification
remain required. The overall goal remains active and incomplete.
