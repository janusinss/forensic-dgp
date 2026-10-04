# CCTV DGP V5: audited results and target-resolution check, 4 October 2026

No trained V5 epoch passed the frozen selection safeguards. Both `best.pth`
files retain the V2 identity epoch 2 starting tensors. The comparison improves
some averages but does not establish useful native CCTV restoration. The full
DGP-first Goal remains active; the application default was not changed.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM workspace: `~/forensic-dgp/cctv_dgp_vm_bundle/` on the existing NVIDIA L4.
This report's intended VM copy is `~/forensic-dgp/CCTV_DGP_CONFLICT_RESULTS.md`
after document transfer. Executed V1–V5 protocols, sources and returns stay intact.

## Audit once per new return

Each new returned experiment needs one successful independent local audit before
its metrics or checkpoint selection are accepted. The user completed the full
V5 audit, including `--verify-recognizer`. Do not repeat it on these unchanged
files. The assistant can handle subsequent audits after the user announces that
new result files have arrived; no long manual audit command is required each time.

The receipt applies to its pinned protocol and result files. Changed files, a
different experiment or an incomplete audit need verification again. The archive
extraction and receipt-writing options deliberately refuse an existing destination
or receipt. Keep the existing extraction and receipt; use a fresh version for a
different return rather than overwriting or deleting the old evidence.

## Verified return

Local archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-conflict-v5-results.tar.gz`
and `.sha256`, from VM
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-conflict-v5-results.tar.gz`.
Size: 347,721,073 bytes. SHA256:
`eec1a062ea3c3203e5bba4a7a12f0a8fdae6de1a1e37942c705ed318aa6cbcb2`.
The local checksum has the exact filename and LF ending and matches the archive.

Canonical local return:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_return_v5\outputs\cctv_dgp_conflict_v5\`,
corresponding to VM
`~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_conflict_v5/`.
Protocol SHA256:
`ba3a6775ca187ab8199fa2b278685f347004af84bb3a01049ec4e92feb06181f`.
Returned `results.json` SHA256:
`80d8a9b1d5cec0d6fbe325229664a5c5251ba2739791922ddc728d4466d8cd5b`.

The user's independent receipt at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_return_v5\local_independent_audit.json`
has SHA256
`93017a6afb593730afc999de6d8e18603a5f0e441a3faa1fc49517405c45f437`.
It records 2,750 PNG predictions, 50 raw previews, 3,300 reconstructed embedding
cosines, 452 update records, 454 policy summaries and 52 recognizer preview
forwards. Each of four trained checkpoints changed 293 parameter tensors.
Normalization buffers and teachers stayed frozen. Recorded VM autograd traversals:
912; CUDA gradients were not replayed by the local audit.

The L4 process completed 452 optimizer updates in 348.84 seconds (5 minutes
49 seconds), within its 1,800-second cap. Local auditing used zero restoration
forwards, backward calls and optimizer updates. Post-audit analysis reused that
receipt without repeating its model checks or overwriting it. All 550 V5 baseline
PNG files match V2 identity epoch 2 byte for byte.

## Paired results and unchanged safeguards

The held-out comparison contains 110 photo references (51 Asian-source and 59
FFHQ-source), yielding 440 synthetic degraded cases and 110 clear cases. These
are paired synthetic measurements, not native CCTV performance. ArcFace is
fixed-grid embedding similarity, not identification accuracy.

| Stage | Degraded PSNR | Degraded SSIM | Degraded ArcFace | Qualified |
| --- | ---: | ---: | ---: | --- |
| Retained V2 start | 16.2167 | 0.65590 | 0.32686 | Research baseline |
| Identity weight 0.4, epoch 1 | 16.1586 | 0.65665 | 0.34143 | No |
| Identity weight 0.4, epoch 2 | 16.3661 | 0.65567 | 0.34335 | No |
| Two-objective PCGrad, epoch 1 | 16.5047 | 0.66085 | 0.33278 | No |
| Two-objective PCGrad, epoch 2 | 16.9750 | 0.66173 | 0.33152 | No |

Selection requires at least 0.1 dB aggregate degraded PSNR gain while preserving
MSE, SSIM and fixed similarity in every frozen aggregate/source/profile group.
The tolerances remain MSE +1e-12 and SSIM/similarity -1e-6. These rules were not
relaxed after seeing the return.

Identity weight 0.4 epoch 1 loses aggregate PSNR and worsens low-light/compound
MSE and SSIM in both sources. Epoch 2 gains 0.1493 dB and 0.01649 similarity;
no similarity group regresses, but blur/motion MSE and SSIM worsen in both sources,
and aggregate degraded SSIM falls 0.00023446.

PCGrad epoch 1 gains 0.2880 dB but regresses Asian blur/motion MSE, FFHQ blur MSE
and Asian low-light similarity (-0.00116162). Epoch 2 gains 0.7583 dB, but blur
MSE rises 28.7% for the Asian source and 16.8% for FFHQ; both sources also worsen
motion MSE and blur/motion SSIM. Asian low-light similarity falls 0.00432121;
FFHQ compound similarity falls 0.00055395. Aggregate gains do not remove these
demonstrated weaknesses.

Each arm encountered conflicting recorded gradients in 131/226 updates. The
ordinary weight-0.4 sum had four negative identity-direction dot products and
34 negative reconstruction-direction dot products. PCGrad had none of either.
These are Euclidean parameter-gradient arithmetic before clipping and Adam;
they do not guarantee the Adam update direction or preserved output structure.
The diagnostic conflict treatment worked in that limited sense, but it did not
make every validation group pass.

Both selections are epoch 0. Serialized `best.pth` SHA256:
`2349624c2416870a3748debcd4cb1076a1f1c09af5fd90f7445821fd0ba48444`.
Their tensor fingerprint is the V2 start:
`d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3`.
Different serialization bytes from the original V2 file do not imply different
weights. Do not describe these fallback files as newly improved trained models.

## Preview review and target-resolution evidence

The assistant inspected all five fixed 10-row preview grids: baseline and the
four trained epochs. They cover two photo references with five synthetic profiles
each. Clear controls remain coherent with smoothing. Severe blur and compound
faces still have indistinct eyes and mouths. Brightening low-light outputs alone
does not establish useful recovered structure. This small development review
does not establish population-wide or native CCTV usefulness. No V5 candidate
was forwarded to native development data; the 32 reserved native crops remain
untouched. Independent final review is still pending.

A separate read-only target audit verified the native image headers and pinned
native/target hashes for all 1,012 references. Every target is 256×256, but every
native reference has at least one dimension below 256. The following count uses
the stricter condition that both dimensions are below 256:

| Source and role | References | Both native axes below 256 | Native width median | Native height median |
| --- | ---: | ---: | ---: | ---: |
| FFHQ training | 451 | 451 | 128 | 128 |
| Asian-source training | 451 | 445 | 175 | 200 |
| FFHQ validation | 59 | 59 | 128 | 128 |
| Asian-source validation | 51 | 48 | 176 | 200 |

Overall, 896/902 training targets enlarge originals smaller than 256 in both
dimensions. All FFHQ targets start at 128×128. This limits available true detail;
it does not prove that target resolution caused the failures or that higher
resolution alone will fix them. The inherited cohort has not been fully visually
reviewed, and exhaustive identity overlap across all 80,000 source images has
not been established.

Separate Windows ledgers, without changing the returned evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_v5_review\`;
optional VM copies: `~/forensic-dgp/outputs/cctv_dgp_conflict_v5_review/`.
`analysis.json` SHA256:
`5f012e6b49d17d6fd34f946b23e6f7ccad1363cfa92afd8b6b6f87167021c3af`.
`target_resolution_audit.json` SHA256:
`6bdb7193e90c2dbc8625c4ac3008d09908514363f3230aef24edf7943e643ae6`.
`visual_review.json` SHA256:
`eebde2fa3dcca359c871d6848d2be16c2ffad89848b1079f76ca39506284d345`.
This analysis and header audit used zero model forwards, backward calls or updates.

## Next action before another VM pilot

Audit genuine higher-quality source counterparts and clean target suitability
before another loss-only adjustment. The [official FFHQ repository](https://github.com/NVlabs/ffhq-dataset)
provides aligned 1024×1024 images and metadata linking each image to its 128×128
thumbnail, with dimensions, file/pixel checksums and attribution. A bounded
counterpart audit can preserve existing image IDs and train/validation roles.
It must verify the correspondence rather than assume filenames prove it.
Record per-image terms and provenance; the repository specifies noncommercial
conditions and excludes facial-recognition development. Our task is restoration;
fixed embedding similarity remains a limited evaluation proxy.

Do not download the entire collection or begin training just to replace the
thumbnails. First verify a bounded sample and review resolution, blur, coverings,
alignment and duplicates. Keep the existing Asian-source data separately reported;
an FFHQ counterpart is not an Asian-population label. Any changed target set needs
a new frozen protocol, a freshly evaluated starting baseline on those same
targets and unchanged historical results. Model-sharpened images must not be
presented as genuine clean references.

No new training recipe is ready or running. Next work is data preparation and
target-quality review. DGP-led app integration, useful native restoration,
covering-family output verification, Playwright workflow checks and independent
final assessment remain required for Goal completion.
