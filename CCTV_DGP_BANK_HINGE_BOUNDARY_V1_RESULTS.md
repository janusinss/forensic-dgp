# Completed brightness-hinge diagnostic — 11 October 2026

**The inactive-brightness-gradient defect is corrected on the manual L4 run.**
Every initial guard gradient is exactly zero on all 20 sampled TRAIN references,
with exact scalar penalty values retained. This is a verified loss correction;
the run has zero optimizer updates and creates no improved checkpoint.

The run took **112.24 seconds**, with 80 B forwards,
40 fixed-recognizer forwards, 360 gradient queries, zero backwards calls and
zero optimizer updates/epochs. Peak allocated VRAM was 3.160 GiB;
export took 20.69 seconds. The 432,115,737-byte archive SHA256 is
`08d246d63dbfd31eb941a043a94d993bf57d125b9898bc7cd92bac2e0c47e929`. The file was downloaded as `outputs/.5`; an
exclusive-read, hash-verified copy recovered its expected name. Both originals
are retained; nothing was deleted. The independent importer checked 135 return
files and 5,852 required parent assets (1,282,410,271 bytes).

All 40 frozen control blocks/200 case outputs are bit-exact to retained initial
and B50 outputs. All 40 scalar rows were independently recomputed. Four full
saved gradient samples, 370 arrays each, passed NumPy Gram/clipping/Adam checks.
These checks verify saved derivative arithmetic; they do not independently
re-run all 40 neural derivatives. Old/new scalar components are exact. Pure
reconstruction/anchor derivatives are exact in the four saved cross-run samples.
Small ArcFace/active-barrier gradient variation is retained, rather than called
bit-exact. Maximum sampled ArcFace difference is 2.384185791015625e-07;
the active barrier sample differs by at most 3.5762786865234375e-07.
The identity path uses CUDA `grid_sample`, whose backward can be nondeterministic;
this is a source-supported explanation, not a freshly isolated replay of that
variation. [PyTorch 2.9 grid_sample](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.grid_sample.html),
[reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html).

| State | Nonzero guard gradients | Nominal Adam detail ascents | Clear-MSE ascents | Degraded-MSE ascents |
| --- | --- | --- | --- | --- |
| Pinned initial | 0/20 | 1/20 | 0/20 | 15/20 |
| Rejected B50 | 20/20 | 4/20 | 0/20 | 19/20 |

These are **20 photographic TRAIN references per state**, five synthetic profiles
each. Directions are hypothetical single-reference Adam directions, not the
historical accumulated five-reference update or measured finite image changes.
B50 is a rejected state used for diagnosis only; it is not resumed.

The boundary correction reduces initial degraded-MSE ascents from 20/20 to
15/20. It does not resolve the whole objective. Unconditional clear/blur absolute
anchor coefficients are 1484.09 and
142.62, compared with a per-profile
weighted reconstruction MSE coefficient of 1.8032.
That is **823.02×** and
**79.09×** per-profile weighting,
respectively; coefficient ratios alone are not gradient-norm ratios.

The combined anchor negative gradient opposes average degraded MSE on 16/20
initial references and 20/20 B50 references. The blur anchor alone opposes that
average on all 20 initial references. This exposes a tradeoff between one blur
profile and the four degraded profiles, not a claim that blur restoration is
unimportant. Removing the absolute anchors analytically reduces initial plain
negative-gradient degraded-MSE ascents to 3/20, but predicts clear-MSE ascent on
10/20. The two saved initial Adam examples also do not establish a safe recipe:
omitting all anchors worsens the clear-MSE direction on both, and worsens degraded
MSE on the first. All visible facial features and preservation gates stay required.

**Next decision: test an explicitly constrained parameter direction before
training more epochs.** Two saved initial reference batches from
`dataset/thumbnails128x128` and `dataset/asian_faces` supply derivatives. These
are source labels, not ethnicity or native Asian CCTV capture claims. Their
ordinary direction averages reconstruction, regression barrier and corrected
brightness losses without the conflicting absolute anchors. A convex projection
after Adam preconditioning imposes detail and preservation directions on both
sampled references. The ten first-order targets use a 1% predicted detail reduction
and 0.01% reductions in clear/degraded MSE, SSIM loss and ArcFace loss at nominal
LR 1e-4. These are proposed local directional targets, not changed scientific
output gates or observed improvements.

The saved two-reference constrained direction passes independent Gram, target,
primal/dual and complementarity arithmetic checks. Its norm is
1.000042201× the ordinary direction; no local model
parameters were displaced. It is not the old five-reference accumulated step.
Applicability to all profiles, native faces and finite outputs remains unproven.

The **20,632,115-byte R1 packet** compares ordinary/constrained
directions at two prospectively fixed rates, 1e-4 and 1e-5, from the exact initial
state. It plans four temporary model displacements, 100 B forwards, 160 fixed
recognizer forwards, zero new derivatives/backwards/optimizer updates and no
checkpoint. Each proposal is reset before the next; the initial state is restored
afterward. It saves all 400 finite raw case outputs and exact floor-policy PNGs,
with source/profile gates and 20 comparison sheets for independent visual review.
Successful arithmetic or export does not qualify any proposal or more epochs.

Prepublication checking caught the draft's rounding PNG policy. R1 uses the exact
retained `floor(RGB*255)` policy with input preservation outside observed pixels;
its actual export expression matches all 100 retained baseline PNGs. The draft
changed 9,344,541 channels across all 100 cases;
its original sealed packet is preserved, unrun and superseded. The future metric
checker uses float64 conversion from PNG bytes for the retained high-frequency
metric and passed all 200 saved raw/PNG control rows. Six synthetic boundary
fixtures independently match the retained 1%/20%/MSE/SSIM/ArcFace logic; they are
checker fixtures, not model results. A float32 sum-order audit retains 49
subnormal direction differences (maximum 1.4013178358747879e-36) from moving
division before summation; exact original arithmetic is used, without loosening
raw parity or scientific gates.

The R1 archive/source/checksum/Bash/Python-3.10 AST checks passed. Its Windows
run rejects before neural imports. Require **4 GiB free after installation**;
worker/measurement/export caps are 1200/900/300 seconds, external 1230/330 with
30-second kill grace. VRAM cap 20 GiB, output 768 MiB, reserve 1 GiB; conservative
payload forecast 449,099,657 bytes. The new runtime is
unmeasured; estimate 2–6 minutes measurement plus 1–3 minutes export. No agent
VM connection, compute, training, power or cleanup action occurred.

[Five exact R1 upload, tmux and download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_VM.md>).
[Independent returned hinge audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_hinge_boundary_v1_analysis/independent_audit.json>).
[Audited loss conflict review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_hinge_boundary_v1_analysis/loss_direction_review.json>).

Goal **ACTIVE**. The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
This fine-tuned DeblurGAN-v2-compatible restorer is the original comparator. B
uses our conditioning/reconstruction with a declared external frozen StyleGAN2
prior, not a newly own-trained StyleGAN2 or exact published-DGP reproduction.
The matched corrected-current-versus-replacement comparison remains unresolved.
No new incumbent, extra epoch qualification or app promotion is claimed.

Native CCTV remains unpaired; paired photographic TRAIN PSNR/SSIM and labeled
final identities remain separate. No real Zamboanga CCTV samples, ethnicity,
local performance or exact hidden identity are claimed. The input-only usable
criteria, all visible features, native/final independent reviews, design-preserving
Auto/override app flow and bundled inline Playwright verification remain required.
All seven covering families remain: masks, sunglasses, strong glare, hands,
obstructing hair, scarves and objects; automatic area preview/correction,
visible appearance/clear glasses/non-obstructing hair preservation, original/mask/
one plausible estimate, PNG and optional bundle downloads, and separate automatic
versus assisted evidence. All five milestones remain the completion criterion.
