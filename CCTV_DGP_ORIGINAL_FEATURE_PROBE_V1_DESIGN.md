# Original DGP feature-path diagnostic after V42

This is a proposed manual L4 diagnostic of a separate current-checkpoint copy,
not a new epoch-training recipe or an app promotion. The selected weights remain
SHA256 `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The current lineage has two selected fine-tuning epochs/226 updates after
original Phase3; lifetime epochs remain unknown. V42 stops after50 added-decoder
updates, with no complete epoch or original-DGP weight update.

AGENTS.md's three-attempt circuit breaker was invoked after V40–V42. The invalid
assumption was that changed added-decoder weights and direct correction loss
necessarily produce useful structure under a frozen original feature path.
The architecture preference question remains unanswered. Under the user's
existing repeated instruction to apply the best approach, this design defaults
to an original-feature diagnostic for investigation. This is an explicit design
assumption, not an inferred answer or approval. No quality fix is asserted and
no training or diagnostic is automatically launched.

## Existing failures constrain the hypothesis

V6 and V9 already changed158 original-model tensors. Neither higher-resolution
targets nor V9's20 epochs qualified preservation. A larger original trainable
path is therefore not new proof of capacity or an automatic solution.
The original-decoder R2 diagnostic has zero derivatives for both head4 kernels;
the retained trace has nonzero upstream maps but zero direct float32 head4 output.
Backbone blocks16–18 lie outside the retained forward and remain stored.

V28's12 active reconstruction tensors improve photographic TRAIN structure but
fail clear appearance and brightness. V29 fails development. V33's23-tensor
saved-gradient AdamW/cone trials fail original-state preservation; zero baseline
hinge gradients cannot supply a preservation direction. V35–V36's finite group
guards still fail both-cohort preservation. V37's PNG surrogate gradients do not
flag the observed failures. V38's quarter/eighth scales preserve the diagnostic
cohorts but have small structure gains and do not qualify useful development.
V40–V42's added17952-parameter path remains failed. All records remain binding.

The present hypothesis is narrower than “train more layers”: at the unchanged
current checkpoint, measure whether the original feature path has a connected
structure direction and how finite displacements affect appearance across
sources/profiles. This diagnostic uses158 uniquely owned reachable tensors,
separately comparing146 feature/FPN tensors against12 reconstruction tensors.
Joint original training has already been attempted, so only independently
reviewed new evidence can justify a distinct subsequent training design.

## Predeclared experiment

The first50 photographic TRAIN cases are the existing fixed preview cohort.
The second50 use five other TRAIN references per source, selected by an immutable
reference-ID hash before output inspection. They are cross-cohort diagnostic
cases, not previously unexposed identities, DEV or final holdouts. Both use all
five frozen clear/degradation profiles. Reuse current high-resolution targets,
source replay and their provenance/overlap limitations. No source label implies
ethnicity; native CCTV remains unpaired and outside this diagnostic.

The original graph, clamp and stored evaluation normalization are unchanged.
The candidate removes per-channel observed(candidate-original) mean before its
final clamp, as the retained original-model mean-centering policy does. Initial
outputs must exactly match the same-input original model on all100 cases.
The selected copy has no aliases to the original. There are158 uniquely owned
tensors/1996035 parameters:498627 in the reconstruction control,1497408 in the
feature/FPN partition. The unreachable21 tail tensors, two inactive head4 tensors
and all171 unique normalization buffers remain frozen and stored.

On the gradient cohort only, measure three separate derivatives: degraded
landmark high-frequency luma error, observed paired RGB MSE, and absolute fixed
recognizer error. The landmark filter is the retained13x13 Gaussian on eroded
support. The two other losses measure local interference; they are not summed
into a new optimizer objective. Thirty `autograd.grad` queries distinguish
unused graph inputs from connected numeric-zero derivatives, retain every batch
vector and the ten-batch means, and record encoder/coarse-head/final activations.
No Tensor.backward API call or optimizer is used. There is VM gradient computation.
PyTorch documents that this API returns derivatives without accumulating them
into parameter `.grad`, and exposes unused inputs separately.
[PyTorch2.9 autograd.grad](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad.html).
Unique parameter enumeration retains shared module aliases without duplicate
optimization ownership.
[PyTorch2.9 Module](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html).

Nine disposable trials subtract only the mean structure gradient, separately
restricted to decoder, feature and joint scopes. For each scope the displacement
L2 is0.00001,0.0001 or0.001 times the original scope weight L2. Every trial starts
at the same original vector, rounds once to float32 and restores it afterward.
No AdamW, learned projection, loss cone, pretrained-target substitute, optimizer
moment, schedule, epoch or committed trajectory is introduced. The scale grid is
a diagnostic sensitivity choice, not a proven learning rate. Inference sees
observed input/support/baseline only, never clean target, source or profile.

Retain all1000 raw float images, delivered floor-converted PNGs, mean-only PNGs,
raw/PNG/target embeddings, all30 gradient vectors, aggregate vectors, initial and
nine trial parameter vectors. Report the unchanged1% structure requirement,
all17 groups' MSE/SSIM/recognizer preservation, both-source nonregression and20%
brightness maximum separately for both cohorts and raw/PNG. These comparisons
are descriptive diagnostic gates, not the full3905-case TRAIN stop or a model
qualification. The later five-epoch study still needs a new reviewed recipe
and full-corpus scientific stops. No failed state is resumed.

## Execution and audit boundary

The self-contained packet uses the existing VM venv. Hardware/idle and manual
tmux guards, cache120s/gradient180s/trial600s/worker1200s/export300s limits,
external deadlines and30s grace are enforced. Require4GiB free after install,
20GiB allocated VRAM maximum,512MiB reserve and1.75GiB return contents.
Measured baseline timing/output size projects remaining trials and archive
space with factor1.25 before derivatives. Failures retain receipts and partial
outputs. No cleanup, VM connection, local gradient or automatic run occurs in
preparation. Trial parameters are diagnostic values, not a trained migration
checkpoint; existing full research-cache and checkpoint backups remain required.

The prospective checker is pinned before transfer. It safely imports regular
hash-bound return members, independently verifies all saved raw/PNG metrics,
30 gradient-vector aggregates and nine float32 displacement formulas, exact
stored-row groups and unchanged pass/failure decisions. Fresh CPU inference and
recognizer replay cover200 outputs. Gradients are not replayed locally.
CPU/GPU replay tolerances (3e-6 raw, one PNG byte,5e-5 embeddings) are arithmetic
checks, separate from scientific1e-12/1e-6 preservation thresholds; the embedding
bound accounts for the retained V37 replay discrepancy. No threshold is relaxed
after seeing a return. Independently review every trial face before selecting
a subsequent design. No subset average can override a source/profile failure.

This diagnostic cannot establish useful native CCTV restoration or hidden
identity. All five milestones, useful native development/final reviews and
all seven automatic/assisted covering families remain required. The app and
current checkpoint stay unchanged; the full goal stays active/incomplete.
