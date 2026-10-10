# Original DGP loss-direction balance diagnostic

Proposed after the independently audited original-feature return and all1000
visual observations,9 October2026. This is a separate finite manual L4 test,
with zero new derivatives, zero optimizer updates and zero epochs. It reuses
the audited saved derivatives and an isolated unchanged current DGP copy.
It is not a proven quality fix, an epoch continuation or an app promotion.

The prior pure decoder structure direction can exceed1% in the two small TRAIN
cohorts but fails appearance. Feature-only/joint structure directions also fail,
with large steps visibly washing out features. More epochs cannot be justified
by connectivity alone. V6/V9 original training, V33 decoder cone trials,
V41 added-decoder PCGrad and V42 remain failed; this proposal does not resume
them or weaken any requirement.

Use the unchanged two50-case photographic TRAIN cohorts, profiles, sources,
references, masks, model graph and evaluation normalization from the original
probe. Reuse the three audited ten-batch mean gradients and initial158-tensor
vector, retaining their original protocol/result/audit hashes. No new native,
DEV, final or completion cases enter this test. No source label is ethnicity.

Let d be negative decoder structure gradient normalized to unit L2 and f be
negative feature identity gradient normalized to unit L2. They occupy disjoint
partitions. At the initial checkpoint, the identity derivative along d is
+0.05330218 and along f is-1.80048474. Therefore r=0.02960434990273 neutralizes
the adverse mean identity derivative. Test three directions:

1. f alone, a feature identity control.
2. d+r*f, mean identity-neutral coordination.
3. d+2*r*f, extra mean identity protection.

For each direction, test three reset magnitudes. Set alpha to decoder weight
L2 times1e-5,1e-4 or1e-3, using the same alpha for all three directions for a
controlled comparison. Thus the feature-only control uses decoder-based scale,
not the earlier feature-weight-based pure structure scale. Coupled steps have
decoder displacement alpha and feature displacement r*alpha or2*r*alpha.
This is a sensitivity grid, not a recommended optimizer learning rate.
Round initial+alpha*direction once to float32; restore initial immediately
after every trial. Freeze all buffers and unselected tensors. Retain all1000
raw/PNG/mean-only images and embeddings plus every diagnostic parameter vector.

Canonical float64 direction vectors and the decoder norm are calculated only
from audited saved arrays and hashed into the packet. VM arithmetic independently
checks their derivation (direction/ratio1e-14 absolute, weight norm1e-10 absolute)
before using the canonical values. This avoids changing trial vectors because
of library reduction roundoff. These are preparation arithmetic checks, not
changes to any image, identity or scientific acceptance threshold. No model
trial is assigned or evaluated locally during preparation.

The saved first-order predictions are structure/RGB/identity derivatives:
f:-0.00152663/-0.00714378/-1.80048474;
d+r*f:-0.00075878/-0.00815586/~0;
d+2*r*f:-0.00080398/-0.00836735/-0.05330218.
These are infinitesimal mean predictions on the gradient cohort only. They do
not guarantee clear-image, per-source/profile, cross-cohort, delivered-PNG or
finite-step preservation. The unchanged1% structure,17-group MSE/SSIM/fixed
recognizer preservation, both-source nonregression and20% brightness maximum
apply separately to raw/PNG and both cohorts. All failing trials stay recorded.
Even a small-cohort pass needs full training-corpus, paired DEV and useful
native DEV review before any finite epoch recipe or app qualification.

This is distinct from PCGrad: it uses explicit disjoint original-model
partitions and audited absolute-loss directions without an optimizer,
projection loop, zero baseline hinge derivative or changing weights across
trials. Research motivates inspecting magnitude and directional interference,
not repeating failed optimizers:
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) and
[Gradient Surgery](https://arxiv.org/abs/2001.06782).

Require4GiB free after installation; retain512MiB reserve and20GiB allocated
VRAM cap. Cache120s, nine trials600s, worker1200s/external1230s with30s grace,
export300s/external330s with30s grace, return1.75GiB uncompressed. Before trials,
measure baseline timing/storage and enforce factor1.25 projections for the
remaining work and export. Prior measurements suggest5–10 minutes diagnostic
and1–5 minutes export, not a live VM estimate. Manual tmux only; no cleanup or
automatic execution is bundled.

Preparation must independently check every transferred member/hash, source
binding, Python3.10 AST, absence of derivative/optimizer APIs, local VM-guard
denial, unchanged initial CPU outputs and saved-vector formula. Pin a prospective
independent return checker before transfer. It must recompute every raw/PNG
metric, trial-vector arithmetic and stored-row gate, preserve arithmetic-only
brightness propagation bounds, and replay200 CPU outputs under the unchanged
prospective tolerances. Visually review every trial face before selection.
All prior originals, splits and gate failures remain immutable. The full goal,
five restoration milestones and seven automatic/assisted covering families
remain active/incomplete.
