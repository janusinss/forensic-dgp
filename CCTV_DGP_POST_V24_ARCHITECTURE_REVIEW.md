# Whole-face architecture discussion after V24

**User decision, 6 October 2026: Route A selected — revise our DGP's spatial/feature
path.** The direct reply is retained in
`outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json`.
The original review below explains the choice. A distinct design/finite VM packet
is the next work; no trained quality or app adoption follows from this decision.

The three stopped head pilots are closed after independent audits and all-case
review. The invalid assumption is that these small, heavily filtered corrections
on frozen DGP pixels would supply enough visible facial structure. A nonzero
gradient, changed weights and slightly lower loss have not delivered that result.
The user requires eyes, nose, mouth, face outline and overall visible appearance
together. Some softness is acceptable; feature preservation and useful structure
remain necessary.

| Audited pilot, stopped50 | Delivered degraded structure improvement | Typical correction in byte units |
| --- | ---: | ---: |
| V22 R1 | +0.0000322991% | 0.000210 |
| V23 | −0.0022330634% | 0.029127 |
| V24 | +0.0033227502% | 0.011075 |

Each required at least 1% improvement before continuation. “Typical” is median
observed correction RMS, rather than maximum per-pixel change. None reaches the
final800 test. This rejects the three finite recipes; it does not prove all
architectures or all possible heads incapable.

V24 has exactly the V23 forward-class AST. Its loss change removes the clear-
target reward correctly: all clear reward terms are zero, and the fixed saved-
output loss improvement now comes from degraded examples. Raw feature gain is
still only 0.00180347%, and the median stopped correction loses about 96.68% of
its spatial RMS through the final projection. That ratio describes the particular
saved fields, not an isolated proof of why optimization follows that path.
Quantization and the old clear-target reward cannot alone explain the remaining
failure. The eight learned tensor groups change; this is not an absent-gradient
or export-only problem.

## What needs to change in the research question

The head receives camera/DGP RGB, their fixed high frequencies and support. It
does not access the trained DGP's multiscale feature representations. The final
observed-mask-normalized Gaussian high-pass and channel-mean removal suppress
broad spatial corrections. They were added to avoid V18/V19's appearance and
brightness failures, but the tested heads have produced only sub-byte corrections.
The frozen base itself already has a feature pyramid; “add a pyramid” alone is
not a diagnosis or a distinct justified experiment.

The early metric uses five small landmark patches, covering eyes, nose and mouth
corners. It has no dedicated outline measure. Keep that original requirement and
all original MSE/SSIM/ArcFace guards, while prospectively adding whole-face visual
review and explicit outline/appearance checks to any new protocol. A patch-only
score cannot replace the user's complete visible-face requirement. Brightness-
only improvements remain separately measured, and raw outputs remain distinct
from display processing.

The measured inference/learning evidence supports reviewing a different spatial
reconstruction path. It does not establish whether representation, the final
filter, the optimizer or the loss-gradient balance is the dominant cause. There
are no new ablations, parameter sweeps, gradient probes or training fixes in this
review. More ten-photograph fitting or loss changes alone would not establish
generalization to native CCTV.

## Route A — revise our DGP spatial/feature path, recommended

Keep the original DGP checkpoint as the baseline and preserved starting lineage.
Design our own trainable reconstruction path using its multiscale features plus
the full-resolution observed crop. Give that path access to both fine feature
boundaries and broader face geometry, rather than requiring every correction to
survive the failed final high-pass projection. Initial output must exactly match
retained DGP before learning. This is a proposed custom DGP extension, not an
implemented or validated model.

Preserve visible appearance through explicit clear-input controls and measured
pixel/SSIM/appearance/brightness constraints. The V24 degraded-cohort objective
policy is useful evidence, not proof those terms will suffice for the new path.
Avoid the failed ten-face input selector and the failed V18 CodeFormer-feature
decoder recipe. Source/profile labels and targets remain training/evaluation
metadata; they never enter inference. No pretrained restoration output is
silently substituted for our trained restoration contribution.

This route keeps the thesis contribution and main restorer in our learned DGP.
Its major risk is renewed appearance drift when broad correction is allowed.
That risk must be tested before broader training or app adoption; increasing
capacity or changing the final filter is not a quality result.

## Route B — declared face prior with our learned conditioning

The user's earlier conditional permission permits a frozen pretrained face prior
with our explicitly trained conditioning/structure contribution if outputs improve.
This remains an alternative architecture discussion, not permission to call a
pretrained model our trained DGP. Separate the frozen generator, our learned
parameters and every input connection in provenance and result reports.

Historical V10/V13/V14/V15/V16/V18/V19 results already expose invented appearance,
memorization, rendering-statistic drift or preservation failures. Repeating hard
code fitting, direct cascades, the old statistic head or the old selector unchanged
is excluded. A distinct differentiable geometry/observation connection and an
initial-parity demonstration would be prerequisites for any new finite pilot.
Extra prior sharpness cannot be treated as correct hidden facial identity.

## Research grounding and the next boundary

Published DGP uses a pretrained GAN with progressive image-specific generator
adaptation. Our retained implementation is a feed-forward MobileNet/FPN RGB
residual model; preserve that architecture distinction in the thesis.
[Author DGP project, ECCV2020](https://xingangpan.github.io/projects/DGP.html).

Restormer studies efficient long-range image-restoration context. Its author
implementation combines four resolution levels, decoder feature skips and a
residual RGB output. This supports considering context plus fine observation;
it does not demonstrate CCTV face fidelity or require a Transformer here.
No Restormer code or weights are imported or trained.
[Author paper](https://arxiv.org/abs/2111.09881),
[author architecture](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py).

The user has chosen the architecture direction before another training fix. The applicable
[systematic-debugging rule](<C:/xampp/htdocs/YEAR 4/Testing/.codex/skills/systematic-debugging/SKILL.md>)
says “DON'T attempt Fix #4 without architectural discussion” and “Discuss with
your human partner before attempting more fixes.” The workspace AGENTS.md also
requires a circuit breaker after three failed attempts. No fourth model change
or executable training packet is prepared in this review.

After the discussion, a distinct prospective protocol must freeze its exact
model/initial parity, exposed capacity data, whole-face review criteria, original
quality guards, finite updates/epochs, timing projection and stop rules. Training
and gradient preflight stay manual on the existing L4/g2-standard-4 VM under
`~/forensic-dgp`; local work remains inference/audit only. Preserve failures,
checkpoints and splits. No local training or assistant cloud action occurs.

A necessary training-capacity result must precede separately frozen development
evaluation on the original photographic identities and native CCTV. Native
outputs remain unpaired, pretrained overlap limits stay declared, and final
reserved identities stay unviewed during repair. Useful development output must
then reach canonical local-app inference and independent final review. Automatic
and assisted mask/sunglasses/glare/hand/hair/scarf/other-object completion remain
separate obligations; this head experiment does not qualify them.

Evidence: [V24 result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_RESULTS.md>)
and `outputs/cctv_dgp_post_v24_architecture_review_v1/review.json`. All source
bindings, three pilot results, the exact V23/V24 head equality and saved loss/
correction arithmetic are recorded. The Goal stays active and incomplete.
