# V23 — full-resolution own-DGP detail capacity plan

6 October 2026. Preparation is complete; VM gradients, training, capacity and
quality are pending. This finite experiment follows the independently audited
V22 R1 failure and the user's requirement for clearer visible facial structure.
It does not change the current application or qualify a model for native CCTV.

V22 R1 stopped after 50 updates with only 0.0000322991% feature-error improvement
against its unchanged 1% early stop. All 50 outputs are reviewed: no visible
structural gain, with only 4,486 canvas pixels changing by at most one byte.
The exact-layer trace finds a very small spatial correction through the deep
pooled branch and detail projection. That supports testing a more direct path;
it does not isolate pooling as the sole cause or justify longer unchanged runs.
Original results, source, failure, stopped checkpoint and review are retained.

## Frozen change and evidence

V23 uses a new 4,613-parameter head. A zero-initialized 3×3 convolution maps
13 conditioning channels directly to RGB correction at 256×256. A shallow
16-channel nonlinear branch has a zero output layer. Inputs remain the camera
RGB, retained own-DGP RGB, their high-pass components and observed-area mask.
No target, provenance label, degradation label, CodeFormer features or generated
CodeFormer RGB enters the forward function. There is no pooling or normalization.

The same bounded correction, 13-tap Gaussian detail projection and observed RGB
mean subtraction precede clamping. Delivered pixels outside the observed area
come from the input. Clamping/quantization can affect the final mean; the original
brightness-only safeguard still evaluates the delivered result.

All 50 initial raw outputs and delivered PNGs match the retained cached DGP
exactly. Two prospectively fixed, no-target sensitivity probes compare a 0.001
camera-detail bypass weight perturbation with a 0.001 old latent-tail weight
perturbation. Interior response RMS ratios are 256.68 and 43.21. These are
different feature coordinates: the check shows available spatial signal, not
equivalent capacity, learned benefit, gradients or restored facial information.
Both disposable states are restored exactly; no parameter search occurs.

A separate fixed synthetic contract tests 39,936 observed and 25,600 padding
pixels with an explicit nonzero correction. Raw padding and delivered input
padding are exact, all outputs are finite/in range and pre-saturation observed
mean error is below 2e-8. Packet checks total 56 new-head and four closed-V22-head
CPU forwards, with no original-DGP/recognizer call, backward or optimizer.

Independent preparation verifies 209 assets, 211 safe archive members, modes,
checksum format, Python 3.10 grammar of 14 sources and Bash syntax. Windows
training rejects before neural work. The original objective/core metric functions,
case list, schedule, budgets and quality tolerances are unchanged. The first Bash
syntax attempt hit the sandbox signal-pipe restriction; the read-only syntax
check passes with that restriction lifted. No shell training command was run.

## Data, model contribution and finite limits

Use the same ten exposed photographic training references: five FFHQ and five
AsianCeleb release references, each with clear/blur/motion/low-light/compound
inputs, for 50 cases. Source names describe provenance, not inferred ethnicity.
There are 800 updates, 80 epochs, batch size five and 4,000 scheduled exposures.
Seed, AdamW learning rate 0.0003, weight decay 0.01 and clipping norm one remain
fixed; no AMP, EMA, retry, resume or checkpoint selection for app use.

The original own-DGP checkpoint remains frozen and supplies the baseline prior.
The project uses a trained feedforward FPN RGB-residual restorer, not the published
GAN-DGP optimization procedure. V23 learns a separate own detail head. A declared
pretrained ArcFace recognizer supplies a fixed preservation signal; pretrained
restoration models remain comparison baselines. Their contributions are separate.

This capacity cache retains the declared historical CUDA-scalar input convention.
Four fresh original-DGP cases must match on the L4. Canonical application
normalization/parity remains a later required gate. Do not infer app equivalence
from this cache or promote the head automatically.

| Stage | Frozen cap/stop |
| --- | --- |
| CUDA/source/data preflight | 300 seconds; exact initial cache; 3 GiB free disk |
| Fitting | 800 updates; 1,500 seconds; update-20 projection; update-50 early gate |
| Entire worker | 1,800 seconds; peak allocated VRAM at most 20 GiB |
| External supervisor | 2,100 seconds plus 30-second kill grace |
| Export | 120 seconds internally; 150 seconds externally plus 30-second kill grace |

The explicit `--preflight` performs fixed CUDA checks. Within `--run`, one batch
must produce finite nonzero new-head gradients, including the direct branch,
before the optimizer is constructed. The DGP and recognizer remain frozen.

The prospective quality conditions remain:

- At update 50, degraded landmark high-frequency error improves by at least 1%.
- At final update 800, degraded feature error improves by at least 10%.
- All 17 source/profile groups preserve paired MSE, SSIM and fixed ArcFace at the original tolerances.
- Neither source regresses in degraded feature error; brightness-only improvement contributes at most 20% of paired MSE gain.
- Final update 800 is the sole capacity result; partial checkpoints and failures remain evidence.

Step-time samples, full fitting/worker elapsed time, peak allocated VRAM,
per-component forward counts and pre-optimizer gradient receipts address the
V22 export's missing performance measurements. The parent shell records system
monotonic elapsed time without creating another CUDA context. Hard termination
can leave partial receipts; it is not assumed successful.

## Required continuation

The user runs the verified transfers and exact commands on the existing NVIDIA
L4/g2-standard-4 under `~/forensic-dgp`. No assistant VM/cloud execution occurred.
Independently audit every returned source, receipt, state, output and metric, then
review all 50 cases before declaring this necessary capacity condition passed.
Freeze any broader development/generalization stage separately after that review.

No native or reserved final pixels, covering inputs or COFW publisher-test pixels
enter V23. Paired synthetic training metrics remain separate from unpaired native
evidence; there are no real Zamboanga CCTV samples or local-performance claims.
Existing public-native source/split/overlap limits remain. Reserved native cases
stay unopened, and independent final reviewers/cohort are still unassigned.

The existing DGP-led Auto/On/Off workflow, input/removal-area review, original/
mask/result display and PNG/bundle downloads stay in place. Useful development
restoration, all seven automatic/assisted covering families, ordinary clear
glasses/hair preservation, independent final review and full app verification
remain required. A capacity archive or training completion does not complete the
Goal or establish exact hidden identity.

Commands: [V23 VM runbook](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
Failure: [V22 R1 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Evidence: `outputs/cctv_dgp_detail_skip_v23_preparation/` and
`outputs/cctv_dgp_detail_skip_vm_v23/`.
