# Delivered PNG path review after V36

The user selected the best evidence-led design review after V33, V35 and V36
failed preservation. Nine of V36's 11 delivered PNG failures do not also fail
the same source/profile raw metric. Two fail both; a further raw ArcFace group
fails without a matching PNG failure. Raw feasibility therefore does not predict
all delivered-image decisions. This is a measured mismatch, not proof of a
single cause for every failure or every weak facial feature.

The application floors each observed RGB value after multiplying its float32
value by255. It copies camera bytes outside observed support and decodes PNG
bytes using NumPy float32 division. V34's preservation derivatives instead
measure raw float images. Its differentiable SSIM also uses a different
accumulation path from the independent delivered-image SSIM. Both limitations
must be made explicit before using those derivatives as PNG guards.

A local, frozen-output arithmetic check reproduces all500 actual PNG-valued
tensors exactly. A new separable seven-pixel SSIM forward calculation reproduces
the independently audited delivered measurements: maximum MSE difference
4.163336e-17; maximum SSIM difference2.220446e-16, against fixed1e-12/1e-7
forward tolerances. There are no loaded DGP/recognizer models, neural calls,
gradients, backwards, optimizer updates or VM calls. Runtime is24.298 seconds.
The independent7.170-second check verifies1,154 bound source files, all500 byte
conversions, all11 failure classifications and random, quantization-edge,
near-constant and image-boundary fixtures. This verifies forward arithmetic;
it does not test the proposed backward rule on the L4.

PyTorch2.9.1 defines the floor derivative as zero. Differentiating the exact
byte conversion therefore provides no useful ordinary derivative for selecting
small parameter changes. The VM experiment uses a declared identity surrogate
on observed support, while keeping its forward value equal to the actual PNG.
This is a coarse estimate, not the true derivative of the quantized metric.
[Official PyTorch2.9.1 derivative definitions](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/tools/autograd/derivatives.yaml)

Straight-through estimation is a heuristic for propagating learning signals
through discrete operations. It motivates measuring this candidate, not claiming
it will preserve a nonlinear face reconstructor's delivered output.
[Bengio, Leonard and Courville, 2013](https://arxiv.org/abs/1308.3432)

Research on coarse gradients proves results under a limited two-layer binarized
model and also shows instability from a poor estimator. Those assumptions do
not establish a guarantee for this DGP, its recognizer or JPEG/PNG face data.
Any later displacement must still be measured on actual delivered PNGs, with
all preservation and useful-structure requirements unchanged.
[Yin et al., 2019](https://arxiv.org/abs/1903.05662)

The independent SSIM reference uses RGB channels, a seven-pixel uniform window,
sample covariance and valid observed interior. The new forward helper matches
that definition and float32 intermediate filter outputs; the installed local
scikit-image0.26.0 implementation was inspected directly. The official0.25.2
source supplies a separately available primary definition, not a claim that
an unavailable0.26.0 GitHub path was read.
[Official structural-similarity source](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/metrics/_structural_similarity.py)

V37 is a distinct prospective diagnostic at the original unchanged DGP state:
100 cases, 300 non-hinged delivered-metric coarse gradient queries, 23 tensors,
978,243 parameter values per gradient, zero optimizer/parameter updates, zero
epochs and no checkpoint. It preserves five-case context and uses the actual
delivered PNG values for MSE, SSIM and fixed-grid ArcFace. Its custom backward
rule is used within `torch.autograd.grad`; zero `.backward()` API calls does
not mean zero gradient computation on the VM. No new training loss is adopted.

Forward mismatch, nonfinite values, changed original PNGs/embeddings, altered
dependencies, competing GPU work or a time/VRAM/disk limit stops the diagnostic
and exports evidence. Successful gradient collection alone cannot authorize an
optimizer, remove a preservation failure or qualify the app. The prospective
return checker verifies saved arrays and separately replays frozen inference
without local gradients. A changed finite direction would require its own
justification, manual VM execution and independent output audit afterward.

[Forward analysis](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v36_delivered_metric_path_v1/analysis.json>)
[Independent forward audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v36_delivered_metric_path_v1/independent_forward_audit.json>)
[Distinct V37 helper source](<C:/xampp/htdocs/YEAR 4/Testing/scripts/cctv_dgp_delivered_png_guard_v37.py>)

The original DGP remains primary. Native development restoration, all seven
covering families with separate automatic/assisted evidence, preservation of
visible appearance and independent final review still remain. Goal incomplete.
