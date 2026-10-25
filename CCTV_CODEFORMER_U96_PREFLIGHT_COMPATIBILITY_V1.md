# U96 preflight: research tools and CUDA compatibility

## Material Passport

- Date: 21 October 2026.
- Scope: source-level compatibility review for the complete-U96 post-fusion diagnostic.
- Evidence: official PyTorch 2.9 documentation and the PyTorch v2.9.1 CUDA implementation, plus local source inspection. No CUDA execution is implied.
- Status: source finding; any repaired runtime remains unverified until the user's manual L4 run returns evidence.
- Limits: no model-quality, identity-recovery, or training-success claim.

## Plugin capability check

The user authorized useful plugin use and installation. A fresh catalog query for
Semantic Scholar, arXiv and academic research returned
`MCP tool codex_apps/plugin_management.search_plugins is not available to the model`.
This does not establish that the plugin directory has no suitable plugins. No new
plugin was installed, no external account was connected, and no private image or
research artifact was uploaded. The earlier capability receipt is retained.

The available local Academic Research Suite supplies experiment-review guidance.
It is a local skill, not evidence of a newly installed research plugin. Built-in
browsing can still check primary sources. Neither a plugin nor a successful code
audit replaces the saved-output and independent review requirements.

## Demonstrated source incompatibility

The draft worker enables strict deterministic algorithms. The candidate's
differentiable image delivery and coarse-feature path include bilinear resizing.
PyTorch documents that deterministic mode raises when an operation only has a
nondeterministic implementation. Its v2.9.1 CUDA bilinear backward implementation
explicitly calls `alertNotDeterministic`, citing atomic additions. Therefore the
draft policy would stop a backward query on that CUDA path; this conclusion is a
source-based inference, not an observed L4 traceback from this packet.

Sources: [PyTorch deterministic-algorithm policy](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html),
[versioned CUDA bilinear implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu).

## Bounded repair policy

Keep strict deterministic policy during initialization and forward comparisons.
Only the four explicit `autograd.grad` queries may enter a warning-only context;
restore the previous flags in `finally`. All operations inside that context are
subject to the warning-only policy. Capture all warnings, allow the documented
bilinear-backward nondeterminism warning, and retain/stop on unexpected
nondeterministic-operation warnings. Do not claim a per-kernel switch if the
runtime only changes a global flag for a bounded context.

Save all gradient tensors so the return auditor can recompute finite values,
norms and the final-projection checks. Exact RNG restoration and unchanged weights
do not establish bitwise reproducibility of CUDA gradients. The repair does not
change image preparation, losses, forward-parity tolerances, preservation gates,
the zero-optimizer budget, or the manual-only launch rule.

The remaining agreed source fixes retain the actual failing phase, distinguish
reserved hook-site slots from successful calls, record peak host RAM alongside
current usage, and require the exact NVIDIA L4 device name. These are preparation
fixes before any VM attempt, not a rerun of a failed training recipe.

## Decision

Finish one reviewed, byte-verified, finite manual preflight packet. Run no local
training, no automatic VM launch, and no historical pilot. A successful preflight
would establish initialization and graph feasibility on four TRAIN witnesses;
it would not create or qualify an improved model. U96 remains the incumbent.
