# Audited A/B restoration comparison — 11 October 2026

Both trained routes failed the retained quality requirements. B reduced the
detail proxy error and the number of group/metric failures relative to A, but
neither is a qualified replacement. The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`. Training completion, export completion and numerical improvement
do not establish useful native CCTV restoration or hidden identity recovery.

| Route | Raw / delivered PNG detail gain | Raw / PNG overlapping failures | Decision |
|---|---:|---:|---|
| A | 0.471905% / 0.473663% | 14 / 15 | Rejected |
| B | 0.625591% / 0.627191% | 2 / 3 | Rejected |

The early requirement remains **1%** detail improvement, nonnegative detail
gains in each source, all retained MSE/SSIM/recognizer preservation limits and
at most **20%** brightness-only gain contribution. B's contribution was
**40.516852% raw / 38.623819% PNG**. Its raw low-light FFHQ-label MSE worsened;
delivered PNG also worsened the Asian-source-label low-light MSE. Counts above
overlap groups and metrics; they are not counts of failed faces.

**Metric correction:** the A/B detail gate is `landmark_high_frequency_MSE`,
grayscale high-frequency error in landmark supports. Earlier dated A status
text calling this frozen-recognizer feature MSE is incorrect. The recognizer
similarity is a separate preservation measure, not evidence of recovered identity.
Historical records retain their bytes; this definition supersedes that wording.

## What actually ran and was checked

Each route ran **50 updates**, **250 distinct TRAIN references**, **125 per
source**, **1,250 profile exposures**, **zero complete epochs** (250/781 =
0.3201 epoch). B worker time was **23.89 minutes**;
this includes preflight/baseline/snapshot work, and fit time was not separately
recorded in that historical run. Supervisor exit **2** was its intentional
quality stop; there was no model-loading exception. Failed states are retained
for audit/migration and cannot be resumed as a qualified recipe.

The full return archive `d49f30c01284fd4c27f45a161b1de4075ddf0a94b26eda4bd592a8a711bf4c6e`
contained **7,687 regular members**. Independent checks covered **7,910 paired
raw/PNG compositions**, **72 native unpaired compositions**, strict full state,
six CPU/CUDA prior fixtures and two limited fresh CPU replay batches. All
**781 baseline RGB/embedding packs** and **24 native baselines** match A/B.
This is not a full fresh neural replay of all 3,905 cases.

All **26 A/B sheets** were displayed and observations retained: 100 paired TRAIN
previews and 24 native DEV crops. Clear and higher-information faces retain
more input texture than the original baseline's smoothing. Blur/compound
cases remain soft; dark/downcast/turned native crops remain limited. B versus
A differences are mostly subtle. No convincing whole-face clarity improvement
supports adoption. The reviewer was the implementing assistant; independent
final visual review remains required. Review evidence:
[A/B audit and sheet closure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/visual_review.json>).

All **37 trainable tensors changed**, all received nonzero finite gradients
after update 1, and **755 frozen state tensors remained exact**. Changing-batch
objective reductions do not prove convergence. The 100-case bank-disabled
subset preserves clear detail less well but reduces some dark-image errors;
it is not a full-cohort causal bank ablation. The existing **391 high-quality
FFHQ counterpart targets** are hash bound. `native_size=128` describes the
observed source, not those prepared 256 targets. New target acquisition is not
justified by that field alone.

## Why global bank-feature centering was rejected

A separate **CPU inference-only** probe tested 100 paired TRAIN and 24 unpaired
native cases, using the frozen rejected B50 state. Removing every projected
bank-feature spatial mean reduced some low-light pixel errors, but worsened
clear-face detail error by about **26.7%**, lowered clear SSIM/recognizer
similarity and visibly softened clearer faces. All **five planned probe sheets**
were displayed. The independent counterfactual check covered 200 paired and
48 native compositions plus limited fresh model/recognizer spots. No local
gradient, optimizer update or failed-state training continuation occurred.
[Counterfactual audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis/independent_audit.json>)
and [viewed probe sheets](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis/visual_review.json>).

Constant feature planes influence spatial detail through the nonlinear decoder;
they are not a proven pure brightness channel or the sole cause of failure.
Global feature centering and global bank disabling are rejected for adoption.
The encoder–generative-bank–decoder design is informed by the primary
[GLEAN abstract](https://arxiv.org/abs/2012.00739); that paper does not validate
this custom guard, this implementation or CCTV performance.

## One finite next hypothesis, prepared but unrun

The original B training objective has no direct brightness-only-gain constraint,
although its return gate requires one. Add a **TRAIN-only observed-pixel hinge**
on the four degraded profiles, keeping bank spatial features and all original
losses. Let Gmean be positive MSE improvement from the candidate's mean-only
control, and Gtotal be positive candidate MSE improvement. The added term is
`10 × mean(max(Gmean − 0.2 × Gtotal, 0)) / retained_MSE_normalizer`.
This per-image constraint is stronger than the aggregate return criterion; it
avoids division by small/negative improvements. Coefficient 10 is a fixed finite
hypothesis at twice the existing per-metric regression-barrier scale, not a
claim of optimality. Clear profiles stay under the existing preservation losses.
The added term operates on raw float outputs; delivered PNG preservation and
the detail requirement must still pass their independent checks.

This changes **one training objective term**. Architecture, bank fusion,
initializer file, all frozen weights, optimizer, training data/exposures,
normalizers, schedule, scientific thresholds and resource caps remain exact.
Start from the **unchanged initial B checkpoint at the permanent original DGP**,
not rejected B50. No display correction, oracle inference or global DC deletion.
Local forward-only math checks covered 100 saved cases, exact baseline zero
penalty, pure brightness improvement and exclusion of unobserved pixels.
All 5,644 full-packet members and the 21-member reuse transfer passed independent
hash/source/namespace/syntax/CLI checks; eight unsafe protocol changes were
rejected. Actual CUDA gradients, runtime and useful quality remain unverified.

The experiment is **50 updates**, not extra epochs or a completed replacement.
Estimated **25–35 minutes** if runtime resembles B r4; new timing is unmeasured.
Unchanged caps: worker 90 minutes/external 90.5, fit 10, preflight 5,
snapshot 20, export 10/external 10.5, VRAM 20 GiB, output 6 GiB,
reserve 1 GiB and **16 GiB free after installation**. Record actual fit time
and the new objective contribution. Preserve any stop and full state.
After a third unsuccessful trained comparison, stop modification and ask one
diagnostic question before proposing another recipe. Extra epochs remain
conditional on independently audited preservation and useful DEV improvement.
[Exact manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_VM.md>).

## Maintenance and remaining goal

Authorized, backup-verified maintenance removed **17,347 exact
completed output/archive copies**. Observed free-space increase: **9.793 GiB**;
fresh free: **16.988 GiB**; conservative copied-install free: **16.051 GiB**.
The launch must check again. Checkpoints, full states, raw V42 arrays, training
caches, splits, logs, instructions, provenance and gate/failure records remain
on the VM. Complete local returned evidence remains available. Independent
closure compared 407,062 metadata entries and
31,103 retained research byte hashes. All scientific
cache bytes were not rehashed. No agent training or VM power change occurred.

Native evidence is **24 ChokePoint P1E_S1_C1 development crops**, unpaired and
without aligned clean targets. Synthetic paired TRAIN capacity and its metrics
remain separate. Source labels do not establish ethnicity or Zamboanga
performance; no real Zamboanga CCTV samples exist. Final identities remain
outside tuning. Resize/Phase3/pretrained comparison completion, useful DEV and
independent final review, primary DGP Auto/override integration, meaningful
regressions and bundled inline Playwright app verification remain required.
All seven completion families also remain required: masks, sunglasses, strong
glare, hands, obstructing hair, scarves and objects, with automatic mask preview,
optional correction, preservation of visible appearance/clear glasses/ordinary
hair, original/mask/one plausible estimate, PNG and optional bundle downloads,
and separate automatic/assisted review. This experiment tests none of those
completion families. Goal **ACTIVE**; no app adoption or completed milestone
claim. Preserve the existing app design and request clearer/less-covered crops
when usable information is insufficient.
