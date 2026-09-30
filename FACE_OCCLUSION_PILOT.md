# Direct occlusion transfer pilot — 30 September 2026

Hypothesis: a fully trainable face-specific encoder/decoder learns real covered
regions more effectively than the same randomly initialized architecture, while
meeting the original real/synthetic safeguards. This is unproven. The available
FaceExtraction checkpoint has visible-face supervision and is not itself a
covered-region detector. Its FFHQ pretraining overlap with our benchmark remains
unresolved; any results are development evidence, not independent final accuracy.

## Fixed comparison

| Item | Both arms |
| --- | --- |
| Architecture | SMP0.5.0 ResNet18 U-Net, new direct binary head,14,328,209 trainable parameters |
| Initialization | Identical new head and seed42; pretrained arm loads published encoder/decoder, random arm uses the same architecture's random initialization |
| Training | All encoder/decoder/head parameters and BatchNorm affine weights trainable; BatchNorm running statistics fixed at each arm's initialization |
| Data/order | Existing extended73 real labels and638 cached replay cases; original10x21x8 extended schedule, identical for both arms |
| Optimizer | Fresh AdamW, encoder LR1e-5, decoder/head LR1e-4, weight decay1e-4, gradient clip1;210 updates per arm |

Source checkpoint SHA256:
`01d3c3939c28e47a45acb9a5ea8f8ee460e5ecf046c0caa8404e05915e10901b`.
Initial states exported locally without optimization to
`outputs/face_occlusion_initial_v1/{pretrained,random}.pth`. Both exactly reload
their forward predictions; both new heads are identical. The stored frozen
visible-face head is retained for provenance and excluded from optimization.
Its outputs after backbone adaptation would not reproduce the original model;
use the untouched source checkpoint for original visible-face inference.

BatchNorm statistics differ between pretrained and random arms as part of their
initialization. The comparison tests full initialization, not encoder weights
alone or the effect of normalization alone. No third arm or learning-rate sweep.
Random initialization is a causal control, not an intended production fallback.
This differs from prior frozen SAM2 heads: the encoder and decoder are trainable.

## Supervision and safeguards

Masks explicitly use1 for covered regions and0 for visible/background regions.
No complement of a visible-face mask, ground-truth face support at inference,
new external data, modified labels, or augmentation is introduced in this pilot.

Loss remains mixed supervised BCE + soft Dice +0.25 hard-visible penalty, plus
weight1 Bernoulli-KL to the original frozen completion segmenter on synthetic
cases only. This is the existing loss recipe; any benefit is not attributed to
retuning it. The preserved original generator receives no optimization.

Use the pinned original completion parent as the **common** validation baseline.
Do not compare each arm against its own random initial predictions. Keep real
IoU improvement plus visible/empty/clear guards, and synthetic IoU/missed/visible/
empty/clear retention limits exactly as implemented before this experiment.
Evaluate and save candidate checkpoints/masks only at epochs1,5,10 (21,105,210
updates), fixed before execution. Baseline is measured once; selection cannot
choose a checkpoint from an unreported intermediate epoch. Report all candidate
checks, training-fit metrics and human/mannequin/glare validation separately.

Save original-parent baseline masks and candidate real/synthetic/training masks,
versioned detector states, batch indices, losses and artifact hashes. A selected
`best_detector.pth` only establishes metric eligibility. Promotion still requires
independent recount, checkpoint reproduction and reviewed10-row end-to-end
completion improvements. No application integration is part of launching this run.

## Execution bounds and provenance

Actual optimization requires the Linux CUDA VM. CPU/Windows runners refuse before
creating output, loading model or constructing an optimizer. Preflight performs
one mixed-batch forward under each arm with zero updates. The full comparison
runs420 total optimizer updates; it has no resume/automatic extra epochs, adaptive
rates, backtracking, validation-derived training data or test evaluation.
Three saved checkpoints per arm require approximately345MB before compression;
the result size/runtime on the L4 must be measured, not assumed. The return
records elapsed time and peak allocated/reserved CUDA memory.

Old coverage inventory/protocol/source/cache hashes are checked. New code,
requirements, fixed specification and exported initial states have a separate
bundle inventory. Installing pinned inference packages goes into a new target
directory and reuses the existing CUDA Torch runtime. Do not overwrite old scripts,
data, result folders, environments or model weights.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/coverage_vm_bundle/`.
Planned VM output: `outputs/face_occlusion_pilot_vm/` under that root.
Planned return: `face-occlusion-results.tar.gz`, downloaded locally to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-results.tar.gz`.

Next: run the packaged CUDA preflight/comparison, return the archive, independently
verify masks and states, then assess whether full direct-occlusion pretraining
deserves further work. The overall detector/restoration/completion goal remains open.
