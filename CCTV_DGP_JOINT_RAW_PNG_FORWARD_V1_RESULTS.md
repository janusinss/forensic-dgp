# Command correction — fresh R1 available

The audited joint-image findings below are unchanged. The selected live-path
diagnostic now has a fresh R1 that fixes an inherited export filename prefix.
Architecture and every quality/resource requirement are unchanged; it is
verified and UNRUN. [Exact corrected manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LIVE_CONDITIONING_V1_R1_VM.md>).
The original V1 packet, command audit stop and document bytes remain retained.

# Joint raw/PNG comparison and spatial-path decision — 13 October 2026

**The VM diagnostic completed correctly; every proposal remains unqualified.**
The largest step improves the pooled paired TRAIN detail proxy by **0.865441%
raw / 0.886155% PNG**, below the retained 1% requirement, with **37 raw and
37 PNG overlapping group/metric preservation failures**. Counts are not failed
people. Neither the permanent app checkpoint nor the output conversion changed.

| Positive step | Raw detail gain | PNG detail gain | Raw failures | PNG failures | Decision |
|---:|---:|---:|---:|---:|---|
| 0.0001 | 0.009967% | 0.018379% | 0 | 9 | Rejected |
| 0.0005 | 0.049125% | 0.061905% | 0 | 2 | Rejected |
| 0.005 | 0.425125% | 0.450136% | 24 | 25 | Rejected |
| 0.02 | 0.865441% | 0.886155% | 37 | 37 | Rejected |

Both sources also remain below 1% independently. At .02, source label
`dataset/asian_faces` has **0.559452% raw / 0.557613% PNG** gain;
`dataset/thumbnails128x128` has **0.942176% raw / 0.968584% PNG**.
The .0005 step has zero raw preservation failures but two PNG failures: clear
thumbnail-source RGB MSE and blur-source observed similarity. Its detail gain is
only 0.049125% raw / 0.061905% PNG. A small raw-safe step does not qualify a model.

The L4 took **57.52 seconds**:100 B and160 fixed recognizer
forwards, four temporary parameter proposals, and **zero optimizer updates,
gradients, epochs or checkpoints**. Each proposal reset to the initial B.
All 596 return members,5,852 parent bindings,100 raw
control replays,400 raw outputs and400 exact floor-converted PNGs pass the
independent saved-evidence audit. All four parameter/state hashes were rebuilt
from saved arrays. No full neural replay or fresh post-run RAM inspection is
claimed. Archive size is **296,822,428 bytes**; SHA256:
`a0b9f4ccf20d208b5fe27ef9d6c55fb7d592d760144c10af0ab4308d188da306`.
[Return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/independent_audit.json>).

## Why the favorable predictions did not qualify the result

All64 saved raw/PNG directional constraints predict initial descent, but the
finite model response and discontinuous byte conversion do not follow that
linear approximation sufficiently. For example, PNG clear thumbnail-source
MSE at .0005 was predicted to decrease by4.57310e-8 using the actual stored
parameter step, but increased by2.71190e-8. At .02 the same error increases
6.1321%. The separate saved-array review checks256 actual-step derivative dots;
it does not compute new neural derivatives or prove a unique nonlinear cause.
[Finite learning evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/finite_learning_review.json>).

The new review's initial exact baseline-loss guard stopped because differentiable
float64-moment SSIM and scientific skimage float32 SSIM differ by up to5.04591e-7;
recognizer replay reductions differ by up to1.52402e-8. That preparation stop and
its source are retained. The completed review records these offsets and uses
the existing derivative-alignment bound only for those representations. Exact
arrays, PNGs and every scientific acceptance check remain unchanged. These
derivatives remain approximate; an arithmetic check is not output qualification.

## All visible features still matter

All20 sheets were displayed at original2048x1460 resolution; all800 source
cells are bound to image hashes. Small steps have no convincing whole-face
clarity advantage. The largest step adds extra curved nose and lip outlines
in several degraded cases, while eyes, mouths, teeth and ordinary hair remain
soft. Clear glasses and non-obstructing hair remain present. The assistant's
development review is not independent final human review.
[Every visual observation](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/visual_review.json>).

An additional4,000-row diagnostic separates the five existing landmark patches.
At .02, pooled PNG image-left eye error decreases1.716538%, but nose error
increases0.012112% and image-right mouth error increases0.112131%. Nose error
increases in47 of80 degraded TRAIN cases. For the Asian source label, pooled
PNG nose error increases1.581780% and image-right eye error increases0.504818%.
These are overlapping photographic patches, not all visible features, independent
subjects, or new acceptance thresholds. They reinforce the visible result:
a pooled score near1% is not adequate whole-face restoration.
[Separate patch evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_joint_raw_png_forward_v1_analysis/spatial_patch_review.json>).

## Best next approach: verify a live spatial-conditioning path

The raw-only, signed small-step and joint raw/PNG comparisons did not meet the
combined requirements. AGENTS.md's three-attempt question was asked; the user
answered **apply best approach**. The invalid assumption was that a jointly
descending initial direction would yield enough preserved detail at a finite
step. Another unchanged fixed-direction or larger-step trial is not selected.

The current reset tests leave **all12 style/spatial tensors**, representing
**2,580,864 values**, unchanged. All64 saved derivatives are zero in those
partitions. Zero bank readouts initially block their output gradients. This is
a limit of these one-step reset experiments; the earlier audited B50 training
changed all37 tensors and still failed. It is not evidence that conditioning can
never learn or the sole cause of every earlier failure.

The selected separate prototype uses nonzero, declared deterministic channel
readouts and subtracts a **frozen same-input initial feature path**. Its intended
initial difference is zero, so the original output stays the control while
derivatives can reach live style/spatial conditioning immediately. The reference
uses the observed image, not a clean target or recovered identity. Original
observation/decoder weights and the declared external frozen StyleGAN2 bank are
reused. Its new readouts are untrained initializers; no new trained model exists.
This is a feature-path architecture hypothesis, not sharpening or display mixing.

GLEAN's experiments motivate using multiscale spatial encoder conditioning and
decoder fusion; they do not validate our anchored construction. It also reports
non-aligned examples, so copying a recognizer affine into the generative path
is not justified without a separate geometric test.
[Primary GLEAN paper, sections3 and5](https://arxiv.org/html/2012.00739).
FFHQ's official documentation describes aligned/cropped source images; that
does not prove misalignment is the unique cause here.
[Official FFHQ source](https://github.com/NVlabs/ffhq-dataset).

The new **verified, UNRUN manual diagnostic** uses the same100 paired TRAIN
cases,32 raw loss groups and320 gradient queries. It requires exact starting
RGB parity and finite nonzero gradients in all12 conditioning tensors for both
sources' detail and clear-MSE groups. Its frozen/live feature branches each run
40 times; no optimizer, displacement, committed update, epoch or checkpoint is
produced. Original failed recipes and full states remain. Gate fixtures reject
the actual old closed-branch gradients and zero/negative/nonfinite scalar cases.
CPU fixtures prove only matrix/state arithmetic; CUDA parity and live neural
gradients require the manual VM run. [Exact next commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LIVE_CONDITIONING_V1_VM.md>).

Only after that return is independently audited can a separately frozen finite
training design be prepared. It must refresh learning at changed weights and
evaluate actual raw/PNG outputs, source/profile preservation, brightness,1%
global/each-source structure and all visible features. Passing activation alone
does not authorize additional epochs or application promotion.

All data here remain exposed paired photographic TRAIN evidence. The existing
HQ FFHQ counterpart targets remain audited; a thumbnail source label alone does
not justify new acquisition. No native CCTV or reserved final pixels were used,
and no ethnicity, Zamboanga performance or hidden identity is claimed. The
original checkpoint remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
All five milestones and seven separate automatic/assisted completion families
remain required. The full goal stays active and incomplete.
