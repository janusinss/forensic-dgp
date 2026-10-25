# Finite filter-fusion pilot, 18 October 2026

Status: executable preparation, not a trained or accepted model. A separately
verified transfer archive and manual launch guide must exist before VM use.
No local training, automatic VM training, checkpoint promotion or app change
is authorized by this document. The earlier 157 and 154 update drafts remain
preserved; this 151 update design supersedes them for the new pilot only.

## Problem and changed route

The retained CodeFormer R2 adaptation brightens many dark faces usefully but
passes blur, mosaic and repeated-line artifacts through its fidelity branches.
The saved controlled path diagnostic found that globally disabling those
branches suppressed artifacts while changing visible appearance. This pilot
therefore trains the four existing fusion sites from R2, with all encoder,
token-predictor and generative-prior parameters frozen. It is our trained
adaptation of CodeFormer, with CodeFormer credited as the architecture and
pretrained initializer. It is not a new architecture or exact hidden identity
recovery. It does not develop covering detection or completion.

## Inputs and scope correction

Use the separately admitted 301 readable filtered TRAIN inputs, 55 matching
clear TRAIN inputs and 55 brightness-only retention inputs: 411 distinct
role inputs from 55 source photographs. Source photographs are not proven
distinct people. Fixed strata, licensing/provenance, identity/pretraining
overlap uncertainty and all earlier source memberships remain recorded.

Two additional whole-source holds, tr_ffhq_21374 and tr_ffhq_45780, follow the
same recognizable-background-face exclusion used earlier, including partial
faces. Earlier readable labels remain historical evidence. All old 512 cases,
the 370-candidate list, 157-update drafts and 57 generated exposure proxies
remain retained. New admission does not rewrite their optimization flags.

The new 55 dark anchors apply exposure 0.35 in linear sRGB to the same readable
clear inputs, with no noise, blur, resizing or JPEG. They retain frozen R2
behavior; they are not camera-calibrated CCTV degradations, old lowlight
equivalents or paired restoration-quality evidence. The older 57 lowlight
inputs remain stress evidence: root review found 26 readable, 30 borderline
and one insufficient. None is substituted into this pilot's training.

Unknown-origin user wild1–wild5 have no aligned clean originals or established
training rights. Keep them unpaired visual stress tests. They can expose
artifacts or insufficient information; generated sharpness cannot establish
that the depicted details match the actual person. No automatic sufficiency
threshold has been calibrated from these five examples.

## Exact finite schedule and objective

151 new Adam updates, four sequential microbatches per update: two filter,
one clear and one dark. Preserve the old seeded filter order as a subsequence
after removing the two held sources. Cover all 301 unique filter inputs and
retain the one previously declared padding duplicate,
tr_ffhq_03212__global_mosaic_8. No easier-source replacement or output-driven
selection is allowed. Matching clear/dark anchors cycle in the fixed hash
order: 41 photographs occur three times and 14 occur twice. There are 604
microbatches, not 151 epochs and not 151 full dataset passes.

Train exactly 72 fusion tensors / 10,659,584 parameters. Keep the other 443
tensors frozen. Candidate modules stay in evaluation mode, while ordinary
autograd propagates through the frozen decoder. Never use inference_mode or
detach that path. Fresh Adam: betas (0.9, 0.99), eps 1e-8, zero weight decay;
linear 10-update warmup to learning rate 1e-5, then constant through update151.
No new EMA, adversarial, identity-recognizer, LPIPS or token-loss optimizer is
introduced. Only the finite pilot can test whether this route is useful.

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch, with TF32 disabled
and deterministic algorithms required. This follows the documented CUDA
determinism prerequisite in [PyTorch's official API reference](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html).
It does not assert exact numerical reproducibility across different software,
hardware or CPU/CUDA environments; fresh same-runtime controls remain required.

Optimize unclipped RGB256 from the native512 network output. Each filter
microbatch uses RGB L1 plus 0.25 adjacent residual L1. A local corruption uses
equal inside/outside spatial weights on its exact documented rectangle.
Clear uses 2× L1 against untouched CodeFormer's frozen output on that input;
dark uses 2× L1 against frozen R2 on that input. Divide each microbatch scalar
by four, accumulate all four, then take one Adam step. Delivery remains the
original native clamp followed by bilinear256; no brightness postprocessing
is added. For historical observed-canvas inputs, restore known outside-support
input pixels exactly before saved RAW/PNG scoring.

The optimizing512-to256 reduction uses explicit horizontal half-pairs followed
by vertical half-pairs. For this exact2× geometry it is mathematically the
same nonoverlapping, half-pixel bilinear map. It avoids the CUDA bilinear
backward operation that the official PyTorch2.9 deterministic API documents
as an error under strict deterministic mode. The native clamp-to-delivered
bilinear path stays unchanged and is computed without a loss gradient.
Every existing forward compares the optimizing readout with a detached stock
bilinear reference from that same native output: maximum difference must be
at most8×float32 epsilon×max(1,max absolute native RGB). Save the actual
difference and bound. This is a prospective arithmetic witness, not an image
quality tolerance, new model call, extra gradient query or cross-runtime claim.

Before the first optimizer, verify the exact initialization and nine-forward,
seven-query gradient witness. All 72 named fusion tensors must be connected,
with finite nonzero improvement gradients at all four site aggregates. A zero
initial dark-retention gradient is expected because candidate equals R2 there.
Validate the additive loss-gradient witness with the prospective FP32 numerical
check; that tolerance is not a quality-gate relaxation. Restore context/RNG,
preserve state fingerprints and create a fresh empty Adam afterwards.

## Evaluation and stop rules

Current-runtime untouched original, parent and R2 controls each cover the same
478 inputs. These are 406 historical DEV slots (358 paired synthetic and 48
source-separated native unpaired) plus 72 historical filter probes (36 TRAIN,
36 DEV). They are not 478 independent DEV identities. Preserve the 63-core /
9-flagged breakdown and all24 previously refused controls, with no model calls
on refusals. Reserved FINAL remains unused.

Save candidate states and all478 RAW/PNG pairs at updates0,50,151. Require full
478 initialization parity with the fresh R2 controls. Keep the original-based
61-clear MAE and 53-normal-blur absolute luminance/coarse-RGB limits unchanged:
mean increase at most0.01 and worst increase at most0.03, separately RAW/PNG.
Keep parent-relative lowlight/compound PNG mean MAE nonincrease and SSIM
nondrop, with the additional R2 retention check. Native CCTV has no clean
PSNR/SSIM/MAE target. Report every source and split separately.

Initial R2 gate failures remain explicit. An extra no-new-regression stop uses
max(original limit, initial violation) per case and group, without claiming
qualification from an inherited failure. Twenty-four fixed TRAIN probes run
at0 and20: eight clear, eight matching dark, eight filters. They are repeated
training probes, not an independent holdout. Clear/dark delivered behavior
distances and filter MAE/adjacent errors have additional exact 0.01 mean /
0.03 case stops. Full original quality requirements remain authoritative at
the final state. A completed export or successful finite process is not model
acceptance; user-visible usefulness, all planned visual sheets, independent
FINAL review and app verification still follow.

All forwards are prospectively reserved: 3×478 controls, 3×478 candidate
snapshots, 110 cached teacher forwards, 48 TRAIN guard forwards, nine witness
forwards and 604 training forwards = 3,639. Additionally cap604 backwards,
seven gradient queries and151 optimizer attempts:4,401 reservation events.
No unenumerated retries, contingency queries or repeated teacher calls.

Internal work limit80minutes, training stage30minutes, each full output bank
10minutes, witness3minutes, initial/data preflight5minutes; the outer worker
deadline5300seconds includes internal preflight and20-second failure capture.
Export cap10minutes, with bounded terminal closure. Stop on finite time,
resource, trainability, numerical, parity or preservation failures. Retain
actual partial/failed state and gate evidence; never append epochs or resume
this immutable pilot unchanged.

## Storage and manual transfer

Use a fresh cctv_codeformer_filter_fusion_training_v1_vm directory below
~/forensic-dgp on the existing single NVIDIA L4. Require20GiB free after
installation, retaining a4GiB reserve. The self-contained upload carries three
exact input weights plus bounded source/data. Return caps:16,000 regular files,
7GiB uncompressed payload,7.25GiB archive. Checkpoint writes have physical caps
and interrupted temporary files remain evidence. Partial gradients are not
serialized; automatic resume is explicitly unsafe and unsupported.

All actual training is manually launched by the user inside tmux with sealed
hashes. Use one gcloud SCP remote source per Windows command. Installation
cannot overwrite an existing run. Download and independently audit returned
bytes, actual counters, physical states, gates and RAW/PNG images before any
app/default change. No cleanup, inference or training has been performed on
the VM by this preparation.
