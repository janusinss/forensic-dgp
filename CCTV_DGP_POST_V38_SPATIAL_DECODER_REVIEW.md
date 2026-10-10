# Spatial DGP decoder after the V38 development failure

The next design tests a different learned spatial path inside our own DGP.
It does not continue V38, select another step from development outcomes or
substitute a pretrained face restorer. The original trained DGP supplies its
input-specific features and remains the application checkpoint.

V38's quarter and eighth steps pass measured preservation on two exposed TRAIN
subsets. The quarter step was fixed before broader inference; it subsequently
fails the FFHQ-thumbnail compound-profile ArcFace check on520 paired development
cases. Its0.097128% high-frequency improvement is small. Actual review of all24
native crops finds no convincing added clarity. That evidence rejects adoption.
The invalid assumption is that a safe small displacement on these TRAIN subsets
would carry preservation and useful definition into the development workflow.

This is a reason to test a materially different spatial reconstruction path,
not proof that a unique architectural bottleneck has been established. Earlier
V22–V27 added limited heads; V28–V32 changed original reconstruction/fusion
weights; V33–V38 examined conflicting directions, finite preservation and PNG
discrepancy. Those failed gates remain. Their source, loss, gradient and output
reviews supply the basis for moving beyond another update-direction adjustment.

The user selected revision of our DGP's spatial/feature path and later asked
to apply the best approach, with research when needed. The V33/V35/V36 design
discussion was answered with the same direction. Those decisions permit this
separate design review. They do not waive preservation or authorize automatic
training. Every visible facial feature remains important.

The proposed V39 context contains102 channels at256x256: observed RGB, original
DGP RGB,64 lateral feature channels and32 channels after original reconstruction
smoothing. Both original feature maps are captured at128x128 and bilinearly
resampled. Their layers, normalization statistics and original weights stay
frozen. Inputs, references and labels do not condition inference through a
target, profile, source or person lookup.

The new decoder processes16 channels at256 and32 at128, using a spatial shortcut,
local depthwise convolutions, per-pixel channel normalization, multiplication
gates and channel-context scaling. It returns a spatial RGB residual at256.
The lower-resolution original features provide wider context while the direct
observed crop supplies visible local appearance. This is a trainable spatial
path, not display sharpening. Its17,952 parameters occupy57 tensors; the size
and learning capacity are hypotheses to test, not quality evidence.

For each context, a separately frozen copy computes the decoder's initial
response. The trainable response minus that initial response is passed through
half-tanh, centered over observed support per channel, added to the original
DGP output and clamped. Visible pixels outside support are copied exactly.
Both copies have identical seeded weights initially, so the initial residual
is exactly zero. No zero RGB-output initialization suppresses the entire new
path behind a final zero matrix; a VM gradient proof must still measure every
tensor. A constant final RGB bias is omitted because observed centering removes
that direction. Clamping can still change the final mean after learning.

The fixed initial copy remains a runtime dependency of any future trained
candidate. It must not be removed or reinitialized at inference. CPU-generated
initial weights are transferred explicitly so GPU seed/version behavior is not
used to reconstruct them. The seed asset is untrained; writing it locally is
initialization, with zero gradients, optimization or learning.

Primary restoration research motivates examining simple gated convolutional
blocks and a spatial encoder/decoder. NAFNet studies these components on
denoising/deblurring benchmarks. That supports a testable design choice; it
does not establish forensic face fidelity, native CCTV generalization or a
cause for our observed failures.
[Chen et al., Simple Baselines for Image Restoration, ECCV2022](https://arxiv.org/abs/2204.04676).

The author implementation uses channel normalization, depthwise convolution,
multiplication gates and channel scaling. Our different context, two-scale
decoder, fixed0.1 block shortcuts, bilinear fusion and initial-response
subtraction are explicit experimental choices. It is not the published NAFNet
architecture and uses none of its pretrained weights. No NAFNet benchmark
numbers are transferred to this thesis.
[Author implementation](https://raw.githubusercontent.com/megvii-research/NAFNet/main/basicsr/models/archs/NAFNet_arch.py).

Initial local proof is complete on all50 already-exposed V27 TRAIN cases, chosen
before new outputs. It takes46.123 seconds, with101 original DGP forwards and
51 forwards through each decoder copy, including a partial-support case. Every
fresh initial raw output and PNG equals the original exactly. Five invalid
inputs stop before model calls; actual Windows learning enablement is rejected.
No local derivative, optimizer or trained checkpoint occurs.

The independent9.282-second checker verifies90 sources,203 artifacts, all50 raw
pairs,100 PNG compositions,57 seed tensors and separate parameter storage. It
replays the first reference from each photographic source with all five profiles:
ten frozen inputs, with exact local raw parity. Historical GPU-cache maximum raw
difference is2.38419e-6 against the unchanged1e-5 compatibility limit. Original
and both seed states stay unchanged. No useful-output or gradient-connectivity
claim follows from matching the unchanged initial output.

The prospective next experiment is a **zero-update V39 L4 gradient proof**.
It preserves the same50 TRAIN inputs, ten five-profile reference batches, seven
loss terms, normalizers, fixed recognizer/affines and all original gates. Seventy
finite `autograd.grad` queries measure the seven terms over ten batches. All57
decoder tensors must have nonzero finite summed improvement gradients; all four
initial preservation values and gradients must be exactly zero. The entire
original DGP, fixed initial decoder and recognizer must remain unchanged.

There are zero optimizer/parameter updates, epochs or trained checkpoint writes.
Connectivity alone cannot qualify capacity or outputs. A successful independent
return audit can justify preparing a separate finite learning pilot; it cannot
launch one automatically. The1%-at50/10%-at800 requirements,17 preservation
groups, both-source nonregression and20% brightness-only limit remain intact.

V39's self-contained transfer includes verified frozen weights, code and the
existing50 TRAIN input/target/support/cache assets. It does not rerun historical
recipes or alter source/split/term records. Existing L4/g2-standard-4 and the
existing venv remain required; the packet never creates a replacement VM.
Worker600s/external630s plus30s grace, export300s/external330s plus30s grace,
allocated VRAM20GiB, free disk2GiB and uncompressed return192MiB are enforced.
Failures and partial evidence are exported. Estimated diagnostic2–6 minutes
and export1–2 minutes are prospective estimates, not measured V39 VM timing.

The original app design, primary trained DGP, Auto/On/Off and downloads remain.
Completion stays separate and unqualified across automatic/assisted coverings.
Native crops without aligned clean references remain unpaired. The photographic
TRAIN source labels do not infer ethnicity or Zamboanga performance. No real
Zamboanga samples or reserved final pixels are used. Request clearer or
less-covered crops only when usable information is insufficient; poor model
clarity must not reclassify an otherwise usable input. Goal active/incomplete.

[V38 development findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>)
[Initial proof](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1/results.json>)
[Independent initial audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1/independent_initial_audit.json>)
