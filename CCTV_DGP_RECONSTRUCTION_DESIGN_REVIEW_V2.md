# This selected comparison is now USED / FAILED — 14 October 2026

Both manual 50-update routes were independently audited and visually reviewed.
The spatial path learned but did not supply qualified structure or preservation.
[Actual result; original checkpoint unchanged](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_READOUT_V1_RESULTS.md>).
[Selected next diagnosis; no new certified packet yet](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md>).
The original decision below remains exact historical evidence, with its
scientific requirements intact. No unchanged rerun, epoch extension or app
promotion follows from its completed preparation. Exact prior review follows.

# Reconstruction design decision — 14 October 2026

The human answered the required diagnostic discussion with **“apply the best
 approach to improve”**. That delegates the design choice to the assistant;
training remains human-manual on the existing L4 inside tmux. The dated pending
question and three rejected recipes remain historical evidence. The full goal
is active and incomplete.

## Selected next comparison

Test **reconstruction capacity**, rather than another conditioning-gradient
solver or an epoch extension of a failed state. Route A trains an isolated copy
of the corrected current decoder. Route B trains that same decoder plus five
spatial **3×3** connections from the already declared, frozen StyleGAN2 feature
bank. The observation encoder, normalization, prior and input-to-prior
conditioning stay frozen. A and B receive identical prepared inputs, targets,
losses and update order.

This is a comparison of two reconstruction designs, not a causal isolation of
kernel size: B also has bank features and additional parameters. No pretrained
restoration checkpoint is substituted. The original app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

B's spatial connections reuse the previously audited nonzero pointwise readout
weights in the center of each 3×3 kernel. Neighbor coefficients start at zero.
A fixed same-input copy is subtracted from the live connection. This supplies
an unchanged initial output while letting neighboring feature coefficients
learn; it is not an RGB display correction or a clean-reference oracle. Both
the live connection and its reference consume unchanged frozen features.
The original corrected decoder's weights are reused. All new construction,
parameter assignment and derivatives occur only in the manual VM worker.

| Route | Trainable tensors | Elements | Initially delivered output |
|---|---:|---:|---|
| A: corrected current decoder | 15 | 609,219 | Must exactly equal fresh original |
| B: spatial bank reconstruction | 25 | 2,176,419 | Must exactly equal fresh original |

Require finite nonzero objective derivatives for every selected tensor before
an optimizer. A disconnected tensor, parity failure or resource projection
stop retains the failure and launches no optimizer. These checks demonstrate
execution/connectivity only, not useful learning.

## Why this choice follows the evidence

The latest nearby-gradient run could change only 12 conditioning tensors;
the decoder and ten pointwise readouts were frozen. Three rejected proposals
gave at most 0.687495% RAW / 0.706917% PNG structure gain and 32 preservation
failures per representation. More derivative sites did not solve that path.

Unfreezing the whole original feature extractor is not a new fix: the original
feature probe found all 158 reachable tensors connected, but stronger feature
and joint directions damaged detail and appearance. V6/V9 already trained
158 original tensors; V9's 20 epochs improved pixel scores while degrading
recognizer preservation in all eight degraded source/profile groups.
Their checkpoints and stopped recipes are retained.

Earlier bank/decoder joint training also failed. Its learnable bank readouts
were pointwise, its new conditioning initially disconnected behind zero
fusion, and it trained conditioning and reconstruction together. This design
instead holds conditioning fixed, uses nonzero anchored spatial readouts and
tests a learned reconstruction path. It does not claim that these changes
are already a solution or the unique cause of previous failures.

[GLEAN](https://arxiv.org/abs/2012.00739) uses an encoder, frozen generative bank
and learned decoder with multiresolution features. The inspected official
implementation uses spatial fusion and a trained image decoder.
[GFP-GAN](https://arxiv.org/html/2101.04061v2) also studies spatial connections
between restoration features and a frozen generative prior. These are reasons
to test a spatial reconstruction path, not evidence that our custom 256-to-256
adaptation will preserve CCTV appearance. No published restoration weights
or published performance claims transfer to this implementation.

## Finite learning and unchanged acceptance

Use the 20 already exposed TRAIN references / 100 five-profile views, ten
references per source, to test whether either path can learn a preserving
change. The sample is not all 781 TRAIN references, native CCTV, unexposed DEV
or reserved final identities. Current canonical HQ FFHQ target hashes are
authoritative; ancestral thumbnail hashes are retained as historical metadata.
Asian-source photographic replay is reported separately, without ethnicity
or capture-location inference. Acquisition is not justified merely by old
`native_size:128` fields; 391 audited HQ counterparts already exist.

Each route gets at most 50 manual optimizer updates, with the identical
deterministic reference order. That is 2.5 passes through this 20-reference
sample, **not 2.5 additional epochs of the full DGP corpus**. Neither route
continues a rejected state. Project Adam proposals into a common original-
weight distance bound; record the unconstrained proposal and actual projected
vector. The common radius is 0.001 times the smallest of the initial A, initial
B and historical conditioning-vector norms. Additional B parameters cannot
inflate its allowance. Projection, rounding and the actual final distance
must be independently audited. The changed bounded optimizer and small sample
are explicitly declared; historical A results are not rerun unchanged.
Adam proposals are computed on shadow values. Only the bounded result is
installed in the live decoder, so an unconstrained proposal is never used for
an image or a derivative. Intermediate fit states are research states, not
accepted preserving checkpoints; acceptance remains the declared endpoint
followed by independent output review.

The objective retains direct clear/blur RGB anchors, the existing normalized
RGB/SSIM/fixed-recognizer/detail terms, regression barriers and brightness
hinge. Original scientific tolerances stay: MSE regression ≤1e-12,
SSIM/recognizer regression ≤1e-6, brightness-only contribution ≤0.2, cumulative
distance ≤0.001. The endpoint requires ≥1% structure gain globally **and in
each source**, all 17 RAW and 17 PNG groups, and exact observed exterior pixels.
RAW and delivered PNG are separate. Training loss or finite derivatives cannot
replace the actual image checks.

The later full-TRAIN 10% requirement, useful native DEV review, independent
final review, app flow and all seven automatic/assisted completion families
remain required. A small numerical pass cannot select an incumbent or trigger
extra epochs automatically. Freeze a separate full-corpus / conditional 1, 2,
5-epoch study only after independent returned-image and preservation review.

## Preparation and evidence boundaries

Before transfer, independently check the packet namespace and all hashes,
canonical inputs/targets, Python 3.10 syntax, CPU array regressions, the Windows
pre-neural guard, Bash syntax, finite limits and a prospective return checker.
Keep full model/optimizer/RNG states, parameter trajectory, raw/PNG originals,
mask/target bindings, all stopped states and migration metadata. Resource
projections and export must be bounded; current VM space and new runtime remain
unmeasured until manual preflight. No cleanup, VM launch, model training or app
promotion is authorized to the agent by this design choice.

The separate detector diagnostic remains failed on missed and false covering
regions; reconstruction training does not fix its localization. The current
app design and clearer/less-covered input requests remain. Native unpaired and
synthetic paired evidence, automatic and assisted completion, and final roles
remain separate. This review selects a hypothesis; packet preparation and
returned results must report their actual status.

Evidence: `outputs/cctv_dgp_reconstruction_design_review_v2/user_decision.json`;
`CCTV_DGP_FROZEN_RECONSTRUCTION_PATH_REVIEW_V1.md`;
`CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md`;
`CCTV_DGP_MIXED_V9_RESULTS.md`; `CCTV_DGP_BANK_AB_COMPARISON_V1_RESULTS.md`;
`CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md`.
