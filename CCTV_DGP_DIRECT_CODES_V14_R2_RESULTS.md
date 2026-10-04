# V14 r2 direct face-code capacity result — 4 October 2026

**The trained conditioner can reconstruct coherent faces on its ten training
photographs. Generalization is not established.** All five original-cell grids
were reviewed. Code prediction improved substantially; copying degraded rendering
statistics still produces dark/noisy results, while predicted statistics introduce
color and brightness drift. Do not adopt the statistics head or promote this
checkpoint to the main application.

Windows workspace `C:\xampp\htdocs\YEAR 4\Testing\` ↔ VM `~/forensic-dgp/`.
This report's intended VM counterpart is
`~/forensic-dgp/CCTV_DGP_DIRECT_CODES_V14_R2_RESULTS.md` after document sync.
Local audited return `outputs\cctv_dgp_direct_codes_return_v14_r2\` ↔
VM execution `~/forensic-dgp/cctv_dgp_direct_codes_vm_v14_r2/`.
The local independent audit is not a VM training receipt.

## What actually ran

Our 2,619,808-parameter conditioner sees original RGB, frozen DGP RGB and frozen
original-input CodeFormer encoder features. It directly predicts residual code
logits and clean codebook mean/log-standard-deviation. Original DGP, declared
pretrained prior, unused earlier conditioner and recognizer remained frozen.
The contribution is our trained conditioning component; the renderer is a
declared pretrained component, not a newly trained DGP generator.

The exact same ten training references, five per source, were repeated across
five camera profiles: 50 cases, 40 balanced epochs, batch2, 1,000 updates and
2,000 exposures. Adam lr0.001; CE/mean-MSE/logstd-MSE weights1/1/1. Clean teacher
codes/statistics supervise training; fresh-image inference needs no clean target.
Cached original features/logits were used during this capacity fit.

The L4 trainer finished in **97.27 seconds**. VM training, independent arithmetic
audit and export took **163.34 seconds**, with peak allocated VRAM
**1,187,559,424 bytes**. There were 1,001 backward calls: one zero-update preflight
plus 1,000 training calls. All 18 head parameter tensors changed. Frozen model
states matched before/after. No validation, native development or reserved
images were accessed. No `best.pth`, checkpoint selection or production promotion.

## Rebuilt training-cohort measurements

PSNR below is computed from pooled mean MSE. ArcFace is an observed-alignment
embedding similarity, not proof of recovered identity. All measurements describe
the repeatedly seen training photographs.

| Forty degraded cases | PSNR dB | SSIM | MAE | Fixed ArcFace similarity | Code accuracy |
|---|---:|---:|---:|---:|---:|
| Starting prior, observed statistics | 13.706 | 0.5297 | 0.14489 | 0.18246 | 2.02% |
| Update1000, observed statistics | 14.464 | 0.6637 | 0.13244 | 0.47721 | 98.95% |
| Update1000, no statistic transfer | **24.272** | **0.7961** | **0.03553** | **0.56619** | 98.95% |
| Update1000, predicted statistics | 23.151 | 0.7796 | 0.04653 | 0.56190 | 98.95% |

The same rendering mode before/after also matters: no-statistic PSNR improved
13.522→24.272dB on the degraded training cases. Its clear-input PSNR was nearly
unchanged, 24.578→24.540dB, while the predicted-statistic arm fell
24.731→23.572dB. Predicted statistics therefore do not provide a consistent
improvement or satisfy clear preservation. Better token accuracy alone cannot
establish useful restoration.

## Visual review and remaining limits

All five ten-row, seven-column grids were reviewed at original256-cell detail:
clear, blur_lr24, lowlight_lr32, motion_lr48 and compound_lr24. Repeated/displaced
eyes and noses, invented glasses and large facial fragments largely resolve
after1,000 updates. The no-statistic render is broadly coherent but smooths skin
and changes eyes, lips, teeth or expressions in some rows. Observed statistics
retain gray/dark/noisy degradation; predicted statistics improve many such rows
but create uneven lighting, color drift or a plastic appearance.

These are ten-face fitting results and can reflect memorization. The Asian
targets remain lower-resolution photographic proxies; unknown FFHQ overlap with
the pretrained prior is still a limitation. No real Zamboanga CCTV performance,
unseen-person usefulness or exact hidden identity has been established. Existing
main-app inference remains Palette/CodeFormer; no DGP-led upgrade was integrated.

## Verification and preserved technical failure

The original V14 stopped with zero updates because its numeric oracle check
incorrectly expected near-identical decoded images after a float32 normalization
roundtrip. A separately frozen six-render diagnostic measured that roundoff.
R2 changes only this numeric control: exact codebook statistics and bounded latent
roundoff, with the decoded difference reported. Model/data/loss/LR/budget,
100 starting raw/PNG parity checks and quality safeguards are unchanged.
Original failed protocol, sources, partial head and failure audit remain intact.

The 587,564,966-byte successful archive was downloaded and independently audited
locally in28.10 seconds. That audit rebuilt450 raw/PNG renders and embedding
cosines,150 code/stat probes,2,560 labels,50 DGP cache arrays,100 starting-parity
renders,12 oracle arrays,350 grid cells and1,000 traces/2,000 exposures. It ran
zero neural forwards, backward calls or optimizer updates; it does not replay
CUDA gradients or certify output usefulness. All11 execution sources and116
parent assets were reverified on the VM after completion.

| Evidence | Windows path relative to workspace |
|---|---|
| Successful return | `outputs\cctv-dgp-direct-codes-v14-r2-results.tar.gz` |
| Independent audit | `outputs\cctv_dgp_direct_codes_return_v14_r2\local_independent_audit.json` |
| Visual review | `outputs\cctv_dgp_direct_codes_review_v14_r2.json` |
| VM source proof | `outputs\cctv_dgp_direct_codes_v14_r2_vm_sourceproof.json` |
| VM restoration of stopped state | `outputs\cctv_dgp_direct_codes_v14_r2_vm_stop.json` |

Protocol SHA256 `d7d5dcac64202c24a7c85658e6b660bdc806bf7d8c16b09542c53170f40a1da2`.
Execution archive SHA256 `810e5538bbd33defce500f9e4ef954579cf4bc7522cf4da6b3d6accdeb46ff11`.
Result archive SHA256 `a6da3aac0152a741b8acf876707597159009bf8c97ad3adbf046a9fd5e18d88f`.
Results JSON SHA256 `d26fb96ee86efeadbc9505f3c8ecf4e4ab4f6559cea7b33eac154204ee12337a`.

## Next bounded step

Freeze a separate **inference-only generalization probe** before more training.
Use fresh-image DGP/encoder/logit computation, first prove parity against the
cached ten-face outputs, then evaluate the existing update1000 head on the
unchanged104-reference/520-case development validation split. Freeze no statistic
transfer from the training-only evidence above; compare retained DGP and the
starting prior using identical inputs. Do not tune modes or pick validation
examples after seeing outputs. Report clear preservation and all degraded
source/profile groups separately. Keep native and reserved images untouched.

Only the resulting diagnosis can justify a broader training recipe. That later
training needs a separate protocol, finite VM budget and unchanged identity/
structure safeguards. The active Goal is incomplete until useful DGP-led local
inference and independent final review work.

The separately frozen V15 fresh-image comparison is now complete and negative;
see [V15 results](CCTV_DGP_GENERALIZATION_V15_RESULTS.md). Its50 parity cases
passed, but the fitted head failed new-face preservation. The broader-code
preparation above is now the next development step, with no checkpoint adoption.
