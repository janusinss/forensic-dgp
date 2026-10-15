# Loss-direction diagnostic failure and layout repair — 11 October 2026

**The manual diagnostic stopped before measuring any loss gradients.** It is a
processing failure, not evidence that more training worsened the model. The export
completed, but the loss-direction measurement did not. No new checkpoint exists.

Location: the first frozen-output replay at line 155 of the original diagnostic,
initial state, reference batch 180 (five profiles of v9_tr_ffhq_00084). Its assertion
was `Same L4 frozen raw control differs`. The VM ran 9.73 seconds:
**one model forward, zero recognizer forwards, gradients, backwards, updates or
epochs**. Python 3.10.12, PyTorch 2.9.1+cu129, NVIDIA L4 and manual tmux were recorded.

The archive SHA256 is `006e50ec62f27d24cfc227b80300b6747481dcb044109b0cb53ed21d4e631a1a`. All **nine returned regular
files**, member paths/sizes/hashes, prepared source and **5,852 retained parent
assets** passed independent checks. The log and exit code agree with the failure.
There are no measured loss rows, sampled gradients or timing-projection results.
The failed script recorded no raw error size; that omission is retained explicitly.
The old return, protocol, code, log and failure remain immutable.

## Demonstrated processing difference

The new diagnostic stacked HWC arrays and permuted them into NCHW dimensions,
leaving channels-last storage. The parent concatenated individual tensors into
contiguous NCHW storage. Pixel values and dimensions were identical, but strides
were **(196608, 1, 768, 3)** versus **(196608, 65536, 256, 1)** for the five-image
batch. This was an unintended input-layout change in our diagnostic preparation.

A finite frozen CPU reproduction used the same pinned initializer and first five
TRAIN cases, one original forward and two B forwards, **no gradients or updates**,
and unchanged model/buffer hashes. Restoring the parent's layout made initial B
**exactly equal to the original model on the same CPU batch**. Changing only layout
changed **616,175 values**, with maximum raw error
**1.07288360596e-06**. Independent saved-array checks reproduced
both results. The three saved raw arrays remain retained. This was a numerical
regression reproduction, not a new visual-quality comparison.

PyTorch describes channels-last storage as a different stride order with the same
dimensions and notes that floating-point calculations are not generally guaranteed
to be bitwise identical across implementations/devices. These references explain
why identical values alone are insufficient for an exact replay; the causal evidence
here is our controlled local reproduction. [Memory-format documentation](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
[PyTorch 2.9 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

The failed L4 worker did not retain its raw output, so its exact difference size
cannot be reconstructed. CPU reproduction demonstrates the layout defect;
**confirmation of the repaired exact L4 replay remains pending**. We do not treat
CPU/CUDA differences as model quality, or loosen the control check to force a pass.

## Corrected manual R1 diagnostic

The only numerical-path change is `.contiguous()` before CUDA transfer in the RGB
batch constructor. This matches the parent input and target values/strides for
**all 200 case tensors across 40 input/target batches**. Loss helpers, objective
weights, gradient queries, hypothetical Adam arithmetic, states, source data,
case lists, checkpoints, thresholds and resource limits remain byte/value matched
to the failed protocol. The exact L4 frozen replay and 1e-6 gradient/frozen forward
check remain. No optimizer is constructed and no training is resumed.

R1 also saves every frozen raw replay and error/stride receipt **before checking**
it. A future stop will therefore retain actual output/error evidence. All 40 planned
blocks add 157,286,400 raw payload bytes. The combined gradient/forward upper bound
is **712,406,496 bytes**, including the existing gradient metadata allowance,
below the unchanged 1 GiB output cap. R1 uses a new root/archive/protocol and never
overwrites the failed run.

The **256,341-byte packet** passed independent archive/source,
5,852 parent asset, LF checksum, readonly Bash syntax, input-layout regression,
three saved output-array and actual Windows launch-rejection checks. The Windows
worker rejects the run before neural or gradient imports. These are preparation
checks; actual repaired CUDA execution and loss-direction measurements are unrun.

Scope remains **20 TRAIN references, 100 distinct TRAIN cases, initial and joint50
states, 360 gradient queries, 80 B forwards, 40 recognizer forwards**, no native/final
input pixels and no checkpoint. Full gradient samples cover four reference/state
batches; the other 36 gradient summaries have narrower independent audit scope.
Hypothetical single-reference Adam directions cannot establish finite-step quality,
the actual accumulated five-reference update or restored identity.

Require **4 GiB free after installation** and an idle L4. Estimate **3–10 minutes
measurement plus 1–5 minutes export**, still unmeasured. Worker cap 900 seconds,
measurement stop 600 checked between reference batches, export cap 300; external
930/330 with 30-second kill grace. First two batch durations determine the retained
projection stop. VRAM 20 GiB, output 1 GiB, disk reserve 1 GiB. No current VM power,
free-space or credit claim is made. Actual runs remain human manual tmux.

[Five exact upload, tmux and download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_VM.md>).
[Independent failure audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_loss_alignment_v1_analysis/independent_failure_audit.json>);
[Frozen layout reproduction](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_loss_alignment_v1_analysis/layout_reproduction/run_results.json>);
[Independent R1 transfer and regression audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_loss_alignment_v1_r1_transfer_audit.json>).

## Goal and preservation

The app checkpoint remains the permanent fine-tuned DeblurGAN-v2-compatible
baseline `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Our candidate declares the external frozen StyleGAN2 bank and our learned
conditioning/reconstruction. No own-trained StyleGAN or exact published-DGP claim
is made. No candidate is qualified, extra epochs are not justified by this stop,
and no failed-state continuation or app promotion is authorized. The first
diagnostic failure is preserved and will not be rerun unchanged.

All five milestones, seven covering families, final identity separation and
independent final review remain required. Native CCTV remains unpaired; paired
TRAIN metrics remain separate. Clear glasses, ordinary hair, visible appearance,
automatic mask-area review/correction and original/mask/one plausible estimate
with PNG/bundle downloads remain in scope. Automatic/assisted completion is not
tested here. No local Zamboanga data, ethnicity inference or exact hidden identity
claim is introduced. Research caches, checkpoints, split/gate records, app bindings
and design remain retained. No agent VM connection, cleanup or power action occurred;
local work used frozen inference and audits only. Goal **ACTIVE and incomplete**.
