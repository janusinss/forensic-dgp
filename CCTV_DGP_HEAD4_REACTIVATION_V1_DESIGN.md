# Repairing the numerically collapsed original DGP branch

The current DGP's deepest original FPN head produces exact zero on all 100
previously exposed TRAIN inputs. Its two convolution kernels and the connected
first 64 fusion channels have maximum absolute values near 6.305e-40. The second
convolution underflows to zero. Both kernels and that fusion slice are bitwise
equal to the retained Phase 3 checkpoint. Heads 1–3 still respond on all 100
inputs; this is a demonstrated branch limitation, not proof the whole DGP is dead
or that it explains every prior failure.

The owned parameter inventory contains 181 tensors/3,312,707 elements. Its
forward reaches 160 tensors/2,106,627 elements. Twenty-one registered MobileNet
tail tensors are unreachable. The previous 158-tensor finite study omitted the
two reachable head4 kernels. Serialized alias entries must not be counted as
independent trainable parameters. Broadly enabling unused classifier-tail weights
cannot restore this path.

Float32's smallest normal magnitude is about 1.175e-38. Smaller subnormal values
exist with reduced precision; multiplying very small weights can underflow.
Our actual recorded activations, rather than the threshold alone, establish this
branch's zero response. [NumPy floating-point documentation](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html).

## Isolated repair and exact initial preservation

Copy the current DGP's trained adjacent head3 kernels into a separate live head4,
and its head3 fusion slice into the live head4 fusion slice. This fixed initializer
uses no external pretrained restorer, random seed search, target fitting or new
training data. Repairing only the kernels would retain the collapsed fusion path.

Six frozen buffers retain the two original kernels, original fusion slice, two
seed kernels and seed fusion slice. The fusion input becomes:

```
original deep contribution + current other-head contributions
    + (live repaired deep contribution - frozen seed deep contribution)
```

At initialization the parenthesized term is exactly zero. Both live and frozen
seed paths use the observed input; input derivatives are real on both paths.
There is no detached-feature surrogate, straight-through quantizer or target-fed
inference. The original normalization statistics, RGB residual, geometry and
other weights remain. This wrapper adds fixed buffers, not trainable parameters.

The local no-gradient worker verifies exact original/candidate float32 agreement
on 100 cases with 40 CNN calls. The repaired branch has positive activations on
all 100. Two independent fresh replays agree exactly; all 619 untouched full
state entries, the remaining 192 fusion channels and six anchors pass exact checks.
CPU/previous-VM maximum raw difference is 2.14577e-6, within the existing 3e-6
replay limit. Model/buffer hashes do not change during inference. Local gradient
entry points refuse execution. The saved initialization is explicitly not trained
and is not adopted into the app.

## First manual L4 experiment

Before committing another epoch recipe, measure the original versus repaired
connected-path gradients on the same 100 exposed TRAIN cases/20 reference batches.
The four unweighted diagnostic components are all-profile MSE, SSIM loss, fixed
ArcFace loss and degraded-profile landmark high-frequency MSE. Each of the two
models receives 80 queries: 160 total. Every individual connected-path vector is
saved in float64, including original zeros and failed measurements.

Only the two head4 kernels and fusion tensor need gradients for this experiment.
The retained vector selects just the fusion tensor's first 64 input channels:
147,456 connected elements in total. Other weights, normalization and the fixed
recognizer remain frozen. The two existing TRAIN cohorts retain their exposure
history; neither becomes independent evaluation. No native/DEV/final input guides
these gradients. Initial raw outputs and fixed embeddings must match exactly
on CUDA before the diagnostic can finish.

Require a finite, nonzero improvement-gradient norm in each of the three repaired
pieces in each existing cohort. Save the diagnostic result even if this routing
requirement fails. This is a gradient-path decision, not a structure improvement
decision. Zero optimizer steps, parameter-fitting changes or epochs are allowed.
There is no common-direction solver, line search, automatic follow-on or resume.

Enforced limits: 600-second worker, 630-second external deadline, 30-second kill
grace; 180-second export/210-second external export; 20 GiB allocated VRAM; 768 MiB
uncompressed return and 512 MiB reserve. Require 3 GiB free after installation.
Expected runtime is 4–10 minutes plus 1–3 minutes export, extrapolated from earlier
L4 diagnostics; this connected-path experiment has not run on the VM yet.

The archive/checker must verify every saved vector norm, model/buffer hashes,
count, file, bound and returned initial pixel output. Frozen local inference
replays two declared cases; autograd queries are not replayed locally. A routing
pass will justify designing the changed reconstruction training path. It will
not qualify a model or waive the original 1%-at50, 10%-at800, MSE/SSIM/ArcFace,
source-gain or brightness requirements. A finite 1/2/5 additional-epoch study is
prepared only after training capacity and development preservation justify it.

## Scope and records

Keep original checkpoints, splits, canonical hashes, research caches, failure
records and the immutable current-training plan. This evidence supersedes the
earlier unisolated hypothesis in the post-finite-guard design review. Full-corpus
coverage, the retained learning rates and frozen normalization are separately
audited; a single branch repair does not remove those other uncertainties.

Native CCTV remains unpaired, paired synthetic metrics stay separate, and no
ethnicity or Zamboanga performance claim follows. Restoration must preserve all
visible features; insufficient inputs need clearer crops. Existing app design,
Auto/override, independent final review and inline Playwright remain outstanding.
Masks, sunglasses, glare, hands, obstructing hair, scarves and objects remain
separate completion work with visible mask correction and original/mask/estimate
downloads. The full goal remains active/incomplete.

Evidence: [Original branch trace](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_head4_trace_v1/results.json>),
[weight audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_head4_weight_audit_v1/statistics.json>),
[exact initialization audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_head4_reactivation_parity_v1/independent_audit.json>),
[full training coverage](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md>).
